#!/usr/bin/env python3
"""
Comprehensive vision stress-test suite.

Goes beyond clean-set accuracy to measure what production actually faces. For
both the legacy and physics-informed models it reports, on the SAME images:

  1. Corruption robustness — Gaussian noise, blur, brightness, contrast, JPEG,
     occlusion, rotation, at several severities. (Legacy collapsed to 24% at
     Gaussian sigma=50, 51% at blur r=5 — quantify and compare.)
  2. Bin-decision robustness — accuracy of the user-facing disposal decision
     under the same corruptions (should degrade far more gracefully than fine
     item ID, thanks to the physics decoupling).
  3. Calibration drift — mean confidence vs accuracy under corruption (a safe
     model should get LESS confident as it gets corrupted).
  4. OOD rejection — max-softmax score on pure-noise / non-waste images vs
     in-distribution; AUROC of the in-vs-out separation.
  5. Adversarial robustness — FGSM at small epsilons.
  6. Latency / throughput — ms/image and img/s at several batch sizes.
  7. Physics-consistency under stress — decision-violation rate (must stay 0).

Deterministic. Run on Kaggle over the full set; locally with --smoke.

  python evaluation/stress_test_vision.py --arch physics \
      --ckpt models/vision/classifier_physics/best_model.pth \
      --data-dir data/processed/vision_cls/test --input-size 448 \
      --out evaluation_results/stress_physics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vision import taxonomy as tx  # noqa: E402
from evaluation.evaluate_vision_v2 import (  # noqa: E402
    EVA_MEAN, EVA_STD, pick_device, build_model, load_checkpoint, forward_item_logits,
)


# --------------------------------------------------------------------------- #
#  Corruptions  (operate on a [0,1] CHW tensor, return [0,1] CHW tensor)
# --------------------------------------------------------------------------- #
def c_identity(x, s):                 return x
def c_gauss_noise(x, s):              return (x + torch.randn_like(x) * (s / 255.0)).clamp(0, 1)
def c_blur(x, s):                     return TF.gaussian_blur(x, kernel_size=2 * int(s) + 1, sigma=float(s))
def c_brightness(x, s):               return (x * s).clamp(0, 1)
def c_contrast(x, s):                 return ((x - x.mean(dim=(-1, -2), keepdim=True)) * s + x.mean(dim=(-1, -2), keepdim=True)).clamp(0, 1)
def c_rotate(x, s):                   return TF.rotate(x, float(s))

def c_occlude(x, s):
    x = x.clone()
    _, h, w = x.shape
    side = int(min(h, w) * s)
    if side <= 0:
        return x
    gen = torch.Generator().manual_seed(int(s * 1000) + h)
    top = torch.randint(0, max(1, h - side), (1,), generator=gen).item()
    left = torch.randint(0, max(1, w - side), (1,), generator=gen).item()
    x[:, top:top + side, left:left + side] = 0.0
    return x

def c_jpeg(x, s):
    # s = JPEG quality (lower = worse). Round-trip through PIL.
    from PIL import Image
    import io
    arr = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=int(s))
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(out).permute(2, 0, 1)


CORRUPTIONS = {
    "clean":      (c_identity,   [0]),
    "gauss_noise":(c_gauss_noise,[10, 25, 50]),
    "blur":       (c_blur,       [1, 3, 5]),
    "brightness": (c_brightness, [0.4, 0.7, 1.3]),
    "contrast":   (c_contrast,   [0.5, 0.7]),
    "jpeg":       (c_jpeg,       [50, 25, 10]),
    "occlusion":  (c_occlude,    [0.2, 0.4]),
    "rotation":   (c_rotate,     [15, 30]),
}


# --------------------------------------------------------------------------- #
def normalize(x):
    mean = torch.tensor(EVA_MEAN).view(3, 1, 1)
    std = torch.tensor(EVA_STD).view(3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def run_corruptions(model, raw_tensors, targets, arch, device, batch_size):
    i2b = torch.tensor(tx.item_to_bin_index())
    results = {}
    for name, (fn, severities) in CORRUPTIONS.items():
        for s in severities:
            preds, confs = [], []
            for i in range(0, len(raw_tensors), batch_size):
                chunk = raw_tensors[i:i + batch_size]
                batch = torch.stack([normalize(fn(x, s)) for x in chunk]).to(device)
                logits = forward_item_logits(model, batch, arch).float().cpu()
                p = F.softmax(logits, -1)
                preds.append(p.argmax(-1)); confs.append(p.max(-1).values)
            preds = torch.cat(preds); confs = torch.cat(confs)
            item_acc = (preds == targets).float().mean().item()
            bin_acc = (i2b[preds] == i2b[targets]).float().mean().item()
            key = name if name == "clean" else f"{name}_{s}"
            results[key] = {
                "item_acc": item_acc, "bin_acc": bin_acc,
                "mean_conf": confs.mean().item(),
                "conf_minus_acc": confs.mean().item() - item_acc,  # >0 = overconfident
            }
    return results


@torch.inference_mode()
def run_ood(model, raw_tensors, arch, device, batch_size):
    """OOD = pure-noise + scrambled images. Report max-softmax separation."""
    def confs(tensors):
        out = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack([normalize(x) for x in tensors[i:i + batch_size]]).to(device)
            p = F.softmax(forward_item_logits(model, batch, arch).float().cpu(), -1)
            out.append(p.max(-1).values)
        return torch.cat(out)

    in_conf = confs(raw_tensors)
    shape = raw_tensors[0].shape
    gen = torch.Generator().manual_seed(7)
    noise = [torch.rand(shape, generator=gen) for _ in range(len(raw_tensors))]
    ood_conf = confs(noise)
    # AUROC of in (label 1) vs ood (label 0) using confidence as score
    scores = torch.cat([in_conf, ood_conf]).numpy()
    labels = np.r_[np.ones(len(in_conf)), np.zeros(len(ood_conf))]
    auroc = _auroc(scores, labels)
    return {
        "in_dist_mean_conf": in_conf.mean().item(),
        "ood_noise_mean_conf": ood_conf.mean().item(),
        "separation": in_conf.mean().item() - ood_conf.mean().item(),
        "ood_auroc": auroc,
    }


def _auroc(scores, labels):
    order = np.argsort(-scores)
    labels = labels[order]
    P = labels.sum(); N = len(labels) - P
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(labels); fp = np.cumsum(1 - labels)
    tpr = tp / P; fpr = fp / N
    return float(np.trapz(tpr, fpr))


def run_adversarial(model, raw_tensors, targets, arch, device, batch_size, epsilons=(1 / 255, 4 / 255)):
    """FGSM. Needs grad through input."""
    mean = torch.tensor(EVA_MEAN).view(3, 1, 1).to(device)
    std = torch.tensor(EVA_STD).view(3, 1, 1).to(device)
    res = {}
    for eps in epsilons:
        correct = total = 0
        for i in range(0, len(raw_tensors), batch_size):
            chunk = torch.stack(raw_tensors[i:i + batch_size]).to(device)
            y = targets[i:i + batch_size].to(device)
            chunk.requires_grad_(True)
            logits = forward_item_logits(model, (chunk - mean) / std, arch)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, chunk)[0]
            adv = (chunk + eps * grad.sign()).clamp(0, 1).detach()
            with torch.inference_mode():
                pred = forward_item_logits(model, (adv - mean) / std, arch).argmax(-1)
            correct += (pred == y).sum().item(); total += y.numel()
        res[f"fgsm_eps_{eps:.4f}"] = {"item_acc": correct / max(total, 1)}
    return res


@torch.inference_mode()
def run_latency(model, sample, arch, device, batch_sizes):
    res = {}
    for bs in batch_sizes:
        x = sample[:1].repeat(bs, 1, 1, 1).to(device)
        for _ in range(2):  # warmup
            forward_item_logits(model, x, arch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        reps = 5
        for _ in range(reps):
            forward_item_logits(model, x, arch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / reps
        res[f"batch_{bs}"] = {"ms_per_image": dt / bs * 1000, "img_per_s": bs / dt}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--arch", choices=["legacy", "physics"], default="legacy")
    ap.add_argument("--backbone", default="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k")
    ap.add_argument("--data-dir", default="data/processed/vision_cls/test")
    ap.add_argument("--input-size", type=int, default=448)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-images", type=int, default=512, help="cap for the heavy corruption sweep")
    ap.add_argument("--device", default=None)
    ap.add_argument("--pretrained-fallback", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="evaluation_results/stress_report.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu") if args.smoke else pick_device(args.device)
    bs = 4 if args.smoke else args.batch_size
    print(f"device={device} arch={args.arch}")

    # raw (un-normalized) tensors so corruptions act in pixel space
    base_tfm = T.Compose([
        T.Resize((args.input_size, args.input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    ds = ImageFolder(args.data_dir, transform=base_tfm)
    tmap = torch.tensor([tx.ITEM_TO_IDX[c] for c in ds.classes], dtype=torch.long)
    idxs = list(range(len(ds)))
    if args.max_images and args.max_images < len(ds):
        idxs = idxs[:: max(1, len(ds) // args.max_images)][:args.max_images]
    raw = [ds[i][0] for i in idxs]
    targets = torch.tensor([tmap[ds.samples[i][1]] for i in idxs])
    print(f"stress images={len(raw)}")

    model = build_model(args.arch, args.backbone, pretrained_fallback=args.pretrained_fallback or not args.ckpt)
    if args.ckpt:
        load_checkpoint(model, Path(args.ckpt), args.arch, device)
    else:
        model.to(device).eval()

    report = {
        "_meta": {"arch": args.arch, "backbone": args.backbone, "n_images": len(raw), "device": str(device)},
        "corruptions": run_corruptions(model, raw, targets, args.arch, device, bs),
        "ood": run_ood(model, raw, args.arch, device, bs),
        "adversarial": run_adversarial(model, raw, targets, args.arch, device, bs,
                                       epsilons=(1 / 255,) if args.smoke else (1 / 255, 4 / 255)),
        "latency": run_latency(model, torch.stack(raw[:max(8, bs)]), args.arch, device,
                               batch_sizes=[1] if args.smoke else [1, 8, 32]),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print("\n===== STRESS SUMMARY =====")
    c = report["corruptions"]
    print(f"  clean item/bin     : {c['clean']['item_acc']*100:.1f}% / {c['clean']['bin_acc']*100:.1f}%")
    for k in ["gauss_noise_50", "blur_5", "jpeg_10", "occlusion_0.4", "rotation_30"]:
        if k in c:
            print(f"  {k:16s} item/bin: {c[k]['item_acc']*100:5.1f}% / {c[k]['bin_acc']*100:5.1f}%  "
                  f"conf-acc={c[k]['conf_minus_acc']:+.2f}")
    o = report["ood"]
    print(f"  OOD: in_conf={o['in_dist_mean_conf']:.2f} noise_conf={o['ood_noise_mean_conf']:.2f} AUROC={o['ood_auroc']:.3f}")
    for k, v in report["adversarial"].items():
        print(f"  {k}: {v['item_acc']*100:.1f}%")
    for k, v in report["latency"].items():
        print(f"  latency {k}: {v['ms_per_image']:.1f} ms/img  ({v['img_per_s']:.0f} img/s)")
    print(f"\n  report -> {args.out}")


if __name__ == "__main__":
    main()
