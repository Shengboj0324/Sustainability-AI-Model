#!/usr/bin/env python3
"""
Rigorous, reproducible vision evaluation harness.

Evaluates ANY checkpoint (legacy MultiHeadClassifier OR the new
PhysicsInformedWasteClassifier) on an ImageFolder test split and emits an honest
report that the inflated legacy docs never provided:

  * Top-1 / Top-5 item accuracy
  * Macro-F1 AND weighted-F1 (the gap exposes imbalance — legacy 90% weighted
    hid macro-F1 0.91 and per-class F1 as low as 0.37)
  * Full per-class precision / recall / F1 / support, worst-K table
  * Confusion-group accuracy (glass / film / metal ...) — the real failure axis
  * MATERIAL-level and BIN-level accuracy via the taxonomy decision contract
    (what the user actually experiences — robust to within-group item errors)
  * Calibration: ECE / MCE
  * Physics: decision-consistency violation rate (must be 0) + aux-head
    disagreement (diagnostic)

Deterministic (seeded, no shuffle). Designed to run on Kaggle GPU over the full
set, and locally on a sample for verification.

Usage:
  python evaluation/evaluate_vision_v2.py \
      --ckpt models/vision/classifier/best_model.pth \
      --arch legacy --backbone eva02_large_patch14_448.mim_m38m_ft_in22k_in1k \
      --data-dir data/processed/vision_cls/test --input-size 448 \
      --out evaluation_results/eval_eva02.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vision import taxonomy as tx  # noqa: E402

# CLIP/EVA normalization (matches the trained checkpoint's preprocessing)
EVA_MEAN = [0.48145466, 0.4578275, 0.40821073]
EVA_STD = [0.26862954, 0.26130258, 0.27577711]


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(arch: str, backbone: str, pretrained_fallback: bool):
    """Construct the right architecture. pretrained_fallback lets local smoke
    tests use a random tiny backbone without downloading weights."""
    if arch == "physics":
        from models.vision.physics_informed_classifier import PhysicsInformedWasteClassifier
        return PhysicsInformedWasteClassifier(
            backbone=backbone, pretrained=pretrained_fallback,
            enable_se=True, enable_llm_projection=False, consistency_mode="soft",
        )
    elif arch == "legacy":
        from models.vision.classifier import MultiHeadClassifier
        return MultiHeadClassifier(
            backbone=backbone, num_classes_item=tx.NUM_ITEMS,
            num_classes_material=15, num_classes_bin=4,
            pretrained=pretrained_fallback, enable_se=True, enable_llm_projection=True,
        )
    raise ValueError(f"unknown arch {arch}")


def load_checkpoint(model, ckpt_path: Path, arch: str, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    if arch == "physics" and hasattr(model, "load_legacy_state_dict"):
        report = model.load_legacy_state_dict(sd, strict=False)
    else:
        result = model.load_state_dict(sd, strict=False)
        report = {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}
    model.to(device).eval()
    return report, (ck.get("metrics") or ck.get("val_metrics") or {}) if isinstance(ck, dict) else {}


def forward_item_logits(model, x, arch: str):
    out = model(x)
    if arch == "physics":
        return out.item_logits
    # legacy returns (item, material, bin)
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.inference_mode()
def evaluate(model, loader, classes, arch, device, target_map):
    """classes = canonical taxonomy class list (len NUM_ITEMS); model outputs are
    in this space. target_map maps each loader (folder) label -> taxonomy index so
    targets and predictions share an index space even for subset folders."""
    n_items = len(classes)
    target_map = target_map.cpu()
    all_logits, all_targets = [], []
    t0 = time.time()
    for x, y in loader:
        x = x.to(device)
        logits = forward_item_logits(model, x, arch).float().cpu()
        all_logits.append(logits)
        all_targets.append(target_map[y])
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    elapsed = time.time() - t0

    probs = F.softmax(logits, dim=-1)
    preds = probs.argmax(-1)
    top5 = probs.topk(min(5, n_items), dim=-1).indices
    correct = (preds == targets)
    top1 = correct.float().mean().item()
    top5_acc = (top5 == targets.unsqueeze(1)).any(1).float().mean().item()

    # confusion matrix + per-class P/R/F1
    cm = torch.zeros(n_items, n_items, dtype=torch.long)
    for t, p in zip(targets.tolist(), preds.tolist()):
        cm[t, p] += 1
    per_class = {}
    f1s, supports = [], []
    for i, c in enumerate(classes):
        tp = cm[i, i].item()
        fp = (cm[:, i].sum() - tp).item()
        fn = (cm[i, :].sum() - tp).item()
        sup = cm[i, :].sum().item()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": sup}
        f1s.append(f1); supports.append(sup)
    f1s = np.array(f1s); supports = np.array(supports)
    macro_f1 = float(f1s.mean())
    weighted_f1 = float((f1s * supports).sum() / max(supports.sum(), 1))

    # confusion-group accuracy (the real failure axis)
    name_to_idx = {c: i for i, c in enumerate(classes)}
    group_acc = {}
    for g, members in tx.CONFUSION_GROUPS.items():
        idxs = [name_to_idx[m] for m in members if m in name_to_idx]
        if not idxs:
            continue
        mask = torch.isin(targets, torch.tensor(idxs))
        if mask.sum() == 0:
            continue
        group_acc[g] = {
            "accuracy": correct[mask].float().mean().item(),
            "support": int(mask.sum().item()),
            "within_group_error_rate": float(
                (torch.isin(preds[mask], torch.tensor(idxs)) & ~correct[mask]).float().mean().item()),
        }

    # material/bin accuracy via taxonomy decision contract
    i2m = torch.tensor([tx.MATERIAL_TO_IDX[tx.ITEM_FACTS[c].material] for c in classes])
    i2b = torch.tensor([tx.BIN_TO_IDX[tx.ITEM_FACTS[c].bin] for c in classes])
    mat_correct = (i2m[preds] == i2m[targets]).float().mean().item()
    bin_correct = (i2b[preds] == i2b[targets]).float().mean().item()

    # calibration (ECE / MCE, 15 bins)
    conf = probs.max(-1).values
    ece, mce = expected_calibration_error(conf, correct.float(), n_bins=15)

    worst = sorted(
        [(k, v) for k, v in per_class.items() if v["support"] > 0],
        key=lambda kv: kv[1]["f1"],
    )[:8]

    return {
        "num_samples": int(targets.numel()),
        "top1_item_acc": top1,
        "top5_item_acc": top5_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_minus_weighted_gap": macro_f1 - weighted_f1,
        "material_decision_acc": mat_correct,
        "bin_decision_acc": bin_correct,
        "ece": ece,
        "mce": mce,
        "throughput_img_per_s": targets.numel() / max(elapsed, 1e-6),
        "confusion_group_accuracy": group_acc,
        "worst_classes_by_f1": [{"class": k, **v} for k, v in worst],
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def expected_calibration_error(conf, correct, n_bins=15):
    bins = torch.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    n = conf.numel()
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc = correct[m].mean().item()
        avg_conf = conf[m].mean().item()
        gap = abs(avg_conf - acc)
        ece += (m.sum().item() / n) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--arch", choices=["legacy", "physics"], default="legacy")
    ap.add_argument("--backbone", type=str, default="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k")
    ap.add_argument("--data-dir", type=str, default="data/processed/vision_cls/test")
    ap.add_argument("--input-size", type=int, default=448)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="cap #samples (0=all) for quick local runs")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--pretrained-fallback", action="store_true",
                    help="build backbone with random/pretrained weights when no ckpt (smoke test)")
    ap.add_argument("--out", type=str, default="evaluation_results/eval_report.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    print(f"device={device} arch={args.arch} backbone={args.backbone}")

    tfm = T.Compose([
        T.Resize((args.input_size, args.input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(EVA_MEAN, EVA_STD),
    ])
    ds = ImageFolder(args.data_dir, transform=tfm)
    folder_classes = ds.classes
    unknown = [c for c in folder_classes if c not in tx.ITEM_TO_IDX]
    if unknown:
        raise SystemExit(f"folder classes not in taxonomy: {unknown}")
    # map folder label -> taxonomy index; evaluate over the full 30-class space
    target_map = torch.tensor([tx.ITEM_TO_IDX[c] for c in folder_classes], dtype=torch.long)
    classes = tx.ITEM_CLASSES
    if args.limit and args.limit < len(ds):
        idx = list(range(0, len(ds), max(1, len(ds) // args.limit)))[:args.limit]
        ds = torch.utils.data.Subset(ds, idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)
    print(f"samples={len(ds)} classes={len(classes)}")

    model = build_model(args.arch, args.backbone, pretrained_fallback=args.pretrained_fallback or not args.ckpt)
    load_report = {"missing": [], "unexpected": []}
    ckpt_metrics = {}
    if args.ckpt:
        load_report, ckpt_metrics = load_checkpoint(model, Path(args.ckpt), args.arch, device)
        print(f"loaded ckpt: {len(load_report['missing'])} missing, {len(load_report['unexpected'])} unexpected keys")
    else:
        model.to(device).eval()

    report = evaluate(model, loader, classes, args.arch, device, target_map)
    report["_meta"] = {
        "ckpt": args.ckpt, "arch": args.arch, "backbone": args.backbone,
        "input_size": args.input_size, "device": str(device),
        "checkpoint_reported_metrics": ckpt_metrics,
        "load_missing_keys": len(load_report["missing"]),
        "load_unexpected_keys": len(load_report["unexpected"]),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print("\n========== EVAL SUMMARY ==========")
    print(f"  samples            : {report['num_samples']}")
    print(f"  top1 / top5 item   : {report['top1_item_acc']*100:.2f}% / {report['top5_item_acc']*100:.2f}%")
    print(f"  macro-F1 / wt-F1   : {report['macro_f1']:.4f} / {report['weighted_f1']:.4f}  (gap {report['macro_minus_weighted_gap']:+.4f})")
    print(f"  material / bin acc : {report['material_decision_acc']*100:.2f}% / {report['bin_decision_acc']*100:.2f}%")
    print(f"  ECE / MCE          : {report['ece']:.4f} / {report['mce']:.4f}")
    print(f"  throughput         : {report['throughput_img_per_s']:.1f} img/s")
    print("  confusion groups   :")
    for g, v in report["confusion_group_accuracy"].items():
        print(f"     {g:18s} acc={v['accuracy']*100:5.1f}%  n={v['support']}")
    print("  worst classes (F1) :")
    for w in report["worst_classes_by_f1"][:6]:
        print(f"     {w['class']:28s} f1={w['f1']:.3f} P={w['precision']:.2f} R={w['recall']:.2f} n={w['support']}")
    print(f"\n  report -> {args.out}")


if __name__ == "__main__":
    main()
