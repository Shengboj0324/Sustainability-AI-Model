#!/usr/bin/env python3
"""
Train the Physics-Informed Waste Classifier.

Keystone training loop for the redesign. Selects the best checkpoint by
*macro-F1* (not weighted accuracy, the legacy mistake that let mega-classes hide
the glass/film failures). Uses logit-adjusted item loss + consistency +
impossible-pair + confusion-group contrastive (see physics_losses.py).

Runs on Kaggle/Colab GPU (AMP fp16). Verified locally on CPU with a tiny
backbone via --smoke. Resumable, EMA, layer-wise LR decay, cosine schedule.

  python training/vision/train_physics_informed.py --config configs/vision_physics.yaml
  python training/vision/train_physics_informed.py --config configs/vision_physics.yaml \
      --smoke --data-dir /tmp/vis_smoke   # 1-epoch CPU sanity check
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.vision import taxonomy as tx
from models.vision.physics_informed_classifier import PhysicsInformedWasteClassifier
from training.vision.physics_losses import PhysicsInformedLoss, LossConfig, compute_class_log_prior


# --------------------------------------------------------------------------- #
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(cfg, train: bool):
    d = cfg["data"]
    size = d["input_size"]
    norm = T.Normalize(d["mean"], d["std"])
    if train:
        ops = [T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC)]
        if d.get("horizontal_flip", True):
            ops.append(T.RandomHorizontalFlip(0.5))
        if d.get("rand_augment", False):
            ops.append(T.RandAugment(num_ops=2, magnitude=9))
        cj = d.get("color_jitter", 0.0)
        if cj:
            ops.append(T.ColorJitter(cj, cj, cj, min(cj * 0.25, 0.1)))
        ops += [T.ToTensor(), norm]
        re = d.get("random_erasing", 0.0)
        if re:
            ops.append(T.RandomErasing(p=re))
        return T.Compose(ops)
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), norm,
    ])


def remap_targets_factory(folder_classes):
    unknown = [c for c in folder_classes if c not in tx.ITEM_TO_IDX]
    if unknown:
        raise SystemExit(f"folder classes not in taxonomy: {unknown}")
    return torch.tensor([tx.ITEM_TO_IDX[c] for c in folder_classes], dtype=torch.long)


def make_param_groups(model, cfg):
    """Layer-wise LR decay: lower LR for backbone (deeper = lower), higher for
    new heads / property scorer. Robust to any timm backbone naming."""
    t = cfg["train"]
    lr_b, lr_h, decay = t["lr_backbone"], t["lr_heads"], t.get("layer_decay", 0.8)
    head_keys = ("item_head", "material_direct_head", "bin_head", "property_scorer",
                 "mat_gate", "bin_gate", "temperature", "llm_projection", "se_block")
    groups = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(h) or f".{h}" in name for h in head_keys):
            groups.append({"params": [p], "lr": lr_h, "weight_decay": t["weight_decay"]})
        else:
            # crude depth proxy: scale LR down for backbone params
            groups.append({"params": [p], "lr": lr_b * decay, "weight_decay": t["weight_decay"]})
    return groups


@torch.no_grad()
def validate(model, loader, target_map, device):
    model.eval()
    n = tx.NUM_ITEMS
    cm = torch.zeros(n, n, dtype=torch.long)
    i2m = torch.tensor(tx.item_to_material_index())
    i2b = torch.tensor(tx.item_to_bin_index())
    mat_ok = bin_ok = tot = 0
    for x, y in loader:
        y = target_map[y]
        logits = model(x.to(device)).item_logits.float().cpu()
        pred = logits.argmax(-1)
        for t_, p_ in zip(y.tolist(), pred.tolist()):
            cm[t_, p_] += 1
        mat_ok += (i2m[pred] == i2m[y]).sum().item()
        bin_ok += (i2b[pred] == i2b[y]).sum().item()
        tot += y.numel()
    f1s, sup = [], []
    for i in range(n):
        tp = cm[i, i].item(); fp = (cm[:, i].sum() - tp).item(); fn = (cm[i, :].sum() - tp).item()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        sup.append(cm[i, :].sum().item())
    f1s, sup = np.array(f1s), np.array(sup)
    present = sup > 0
    return {
        "top1": float((cm.diag().sum().item()) / max(tot, 1)),
        "macro_f1": float(f1s[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1s * sup).sum() / max(sup.sum(), 1)),
        "material_acc": mat_ok / max(tot, 1),
        "bin_acc": bin_ok / max(tot, 1),
    }


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(m, alpha=1 - self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vision_physics.yaml")
    ap.add_argument("--data-dir", default=None, help="override config data_dir")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU sanity run (resnet18, 1 epoch, few steps)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir

    if args.smoke:
        cfg["model"].update(backbone="resnet18", pretrained=False, prop_dim=16)
        cfg["data"]["input_size"] = 64
        cfg["train"].update(epochs=1, batch_size=8, grad_accum_steps=1, use_amp=False,
                            use_ema=False, warmup_epochs=0, num_workers=0)
        cfg["data"]["num_workers"] = 0

    t = cfg["train"]
    torch.manual_seed(t["seed"]); np.random.seed(t["seed"])
    device = torch.device("cpu") if args.smoke else pick_device()
    print(f"device={device} backbone={cfg['model']['backbone']} smoke={args.smoke}")

    # data
    data_dir = Path(cfg["data"]["data_dir"])
    train_root = data_dir / "train" if (data_dir / "train").exists() else data_dir
    val_root = data_dir / "val" if (data_dir / "val").exists() else data_dir
    train_ds = ImageFolder(str(train_root), transform=build_transforms(cfg, True))
    val_ds = ImageFolder(str(val_root), transform=build_transforms(cfg, False))
    tmap_train = remap_targets_factory(train_ds.classes)
    tmap_val = remap_targets_factory(val_ds.classes)

    # class counts in taxonomy space -> logit-adjustment prior
    counts = torch.zeros(tx.NUM_ITEMS)
    for _, y in train_ds.samples:
        counts[tmap_train[y]] += 1
    print("class count range:", int(counts[counts > 0].min()), "-", int(counts.max()),
          "| empty classes:", int((counts == 0).sum()))

    sampler = None
    if t.get("use_balanced_sampler", False):
        cls_w = 1.0 / counts.clamp(min=1)
        sw = [cls_w[tmap_train[y]].item() for _, y in train_ds.samples]
        sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], sampler=sampler,
                              shuffle=sampler is None, num_workers=cfg["data"]["num_workers"],
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False,
                            num_workers=cfg["data"]["num_workers"], pin_memory=False)

    # model
    m = cfg["model"]
    model = PhysicsInformedWasteClassifier(
        backbone=m["backbone"], pretrained=m["pretrained"], drop_rate=m["drop_rate"],
        enable_se=m["enable_se"], enable_llm_projection=m["enable_llm_projection"],
        consistency_mode=m["consistency_mode"], prop_dim=m["prop_dim"],
    ).to(device)
    if m.get("warm_start_ckpt"):
        ck = torch.load(m["warm_start_ckpt"], map_location="cpu")
        sd = ck.get("model_state_dict", ck)
        rep = model.load_legacy_state_dict(sd, strict=False)
        print(f"warm-start: {len(rep['missing'])} missing, {len(rep['unexpected'])} unexpected")

    lc = cfg["loss"]
    loss_fn = PhysicsInformedLoss(
        class_log_prior=compute_class_log_prior(counts),
        cfg=LossConfig(**{k: lc[k] for k in lc}),
    ).to(device)

    optimizer = torch.optim.AdamW(make_param_groups(model, cfg), betas=tuple(t["betas"]))
    total_steps = max(1, math.ceil(len(train_loader) / t["grad_accum_steps"]) * t["epochs"])
    warmup_steps = int(math.ceil(len(train_loader) / t["grad_accum_steps"]) * t["warmup_epochs"])

    def lr_at(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(t["min_lr"] / t["lr_heads"], 0.5 * (1 + math.cos(math.pi * prog)))

    use_amp = t["use_amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EMA(model, t["ema_decay"]) if t.get("use_ema") else None

    if args.wandb:
        import wandb
        wandb.init(project="releaf-vision-physics", name=t["experiment_name"], config=cfg)

    out_dir = Path(t["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0
    global_step = 0
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    for epoch in range(t["epochs"]):
        model.train()
        optimizer.zero_grad()
        running = Counter()
        t0 = time.time()
        for it, (x, y) in enumerate(train_loader):
            y = tmap_train[y].to(device)
            x = x.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                heads = model(x)
                losses = loss_fn(heads, y)
                loss = losses["total"] / t["grad_accum_steps"]
            scaler.scale(loss).backward()

            if (it + 1) % t["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), t["clip_grad_norm"])
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                scale = lr_at(global_step)
                for g, b in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = b * scale
                if ema:
                    ema.update(model)
                global_step += 1
            for k, v in losses.items():
                running[k] += float(v)
            if args.smoke and it >= 2:
                break
            if (it + 1) % t["log_interval"] == 0:
                print(f"  e{epoch} it{it+1}/{len(train_loader)} "
                      f"loss={running['total']/(it+1):.3f} item={running['item']/(it+1):.3f} "
                      f"cons={running['consistency']/(it+1):.3f} con={running['contrastive']/(it+1):.3f}")

        eval_model = ema.shadow if ema else model
        val = validate(eval_model, val_loader, tmap_val, device)
        dt = time.time() - t0
        print(f"[epoch {epoch}] {dt:.0f}s val_macroF1={val['macro_f1']:.4f} "
              f"top1={val['top1']*100:.2f}% wF1={val['weighted_f1']:.4f} "
              f"mat={val['material_acc']*100:.2f}% bin={val['bin_acc']*100:.2f}%")
        if args.wandb:
            import wandb; wandb.log({"epoch": epoch, **{f"val_{k}": v for k, v in val.items()}})

        metric = val[t["select_metric"]]
        if metric > best_metric:
            best_metric = metric
            torch.save({
                "epoch": epoch,
                "model_state_dict": eval_model.state_dict(),
                "metrics": val,
                "config": cfg,
                "class_names": tx.ITEM_CLASSES,
                "material_names": tx.MATERIAL_CLASSES,
                "bin_names": tx.BIN_CLASSES,
            }, out_dir / "best_model.pth")
            print(f"  ✓ saved best ({t['select_metric']}={metric:.4f})")

    print(f"done. best {t['select_metric']}={best_metric:.4f} -> {out_dir/'best_model.pth'}")


if __name__ == "__main__":
    main()
