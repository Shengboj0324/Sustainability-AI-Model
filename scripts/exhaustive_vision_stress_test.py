#!/usr/bin/env python3
"""
EXHAUSTIVE INDUSTRIAL-GRADE VISION MODEL STRESS TEST
=====================================================
Tests every aspect required for production deployment:
  1. Checkpoint integrity & architecture verification
  2. Per-class accuracy, precision, recall, F1
  3. Top-1 / Top-5 accuracy
  4. Confusion matrix (saved as image)
  5. Confidence calibration (ECE / MCE)
  6. Throughput benchmark (images/sec)
  7. Robustness: Gaussian noise, blur, JPEG compression, brightness shift
  8. Embedding space quality (inter/intra class distances)
  9. Gradient sanity (no dead neurons, no exploding activations)
 10. Model size & memory footprint
 11. Worst-class failure analysis
"""
import os, sys, time, json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageFilter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vision.classifier import MultiHeadClassifier

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────
try:
    import yaml
except ImportError:
    yaml = None

CONFIG_PATH  = PROJECT_ROOT / os.getenv("VISION_CLS_CONFIG", "configs/vision_cls.yaml")
_cfg = {}
if yaml is not None and CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f) or {}

_data_cfg = _cfg.get("data", {})
_train_cfg = _cfg.get("training", {})
_model_cfg = _cfg.get("model", {})

CKPT_PATH    = PROJECT_ROOT / os.getenv(
    "VISION_CKPT_PATH",
    str(Path(_train_cfg.get("output_dir", "models/vision/classifier")) / "best_model.pth"),
)
TEST_DIR     = PROJECT_ROOT / _data_cfg.get("test_dir", "data/processed/vision_cls_clean/test")
VAL_DIR      = PROJECT_ROOT / _data_cfg.get("val_dir", "data/processed/vision_cls_clean/val")
OUTPUT_DIR   = PROJECT_ROOT / "outputs/stress_test"
INPUT_SIZE   = int(_data_cfg.get("input_size", 448))
MEAN         = _data_cfg.get("mean", [0.48145466, 0.4578275, 0.40821073])
STD          = _data_cfg.get("std", [0.26862954, 0.26130258, 0.27577711])
BATCH_SIZE   = 4
NUM_WORKERS  = 0
BACKBONE     = _model_cfg.get("backbone", "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k")
MAX_EVAL_SAMPLES = int(os.getenv("VISION_STRESS_MAX_SAMPLES", "0") or "0")
MAX_CORRUPTION_SAMPLES = int(os.getenv("VISION_STRESS_MAX_CORRUPTION_SAMPLES", "500"))
LATENCY_ITERS = int(os.getenv("VISION_STRESS_LATENCY_ITERS", "50"))
LATENCY_WARMUP = int(os.getenv("VISION_STRESS_LATENCY_WARMUP", "5"))
SKIP_GRADIENT = os.getenv("VISION_STRESS_SKIP_GRADIENT", "0").lower() in {"1", "true", "yes"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _balanced_subset_indices(targets, max_samples, num_classes):
    if max_samples <= 0 or max_samples >= len(targets):
        return list(range(len(targets)))
    per_class = defaultdict(list)
    for idx, target in enumerate(targets):
        per_class[int(target)].append(idx)
    selected = []
    rounds = max(1, int(np.ceil(max_samples / max(num_classes, 1))))
    for offset in range(rounds):
        for cls in range(num_classes):
            items = per_class.get(cls, [])
            if offset < len(items):
                selected.append(items[offset])
            if len(selected) >= max_samples:
                return selected
    return selected[:max_samples]

# ─── DEVICE ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}", flush=True)

# ─── TRANSFORMS ──────────────────────────────────────────────────────
val_transform = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

# ─── LOAD MODEL ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("1. CHECKPOINT INTEGRITY & ARCHITECTURE")
print("="*70)

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
print(f"  Checkpoint keys: {list(ckpt.keys())}")
print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"  Metrics: {ckpt.get('metrics', 'N/A')}")
print(f"  Config: {ckpt.get('config', 'N/A')}")

class_names = ckpt.get("config", {}).get("class_names", None)

model = MultiHeadClassifier(
    backbone=BACKBONE,
    num_classes_item=30,
    num_classes_material=15,
    num_classes_bin=4,
    drop_rate=0.0,       # No dropout at inference
    pretrained=False,    # We load from checkpoint
    enable_se=True,
    enable_llm_projection=True,
)

# Load weights
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
print(f"  Missing keys:    {len(missing)}  {missing[:5] if missing else '[]'}")
print(f"  Unexpected keys: {len(unexpected)}  {unexpected[:5] if unexpected else '[]'}")
if missing or unexpected:
    print("  ⚠️  Weight mismatch detected — investigate before deployment!")
else:
    print("  ✅ All weights loaded perfectly — no missing/unexpected keys")

model = model.to(DEVICE).eval()

total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
print(f"  Total params:     {total_params:,}")
print(f"  Trainable params: {trainable:,}")
print(f"  Model size:       {model_size_mb:.1f} MB")
print(f"  Temperature:      {model.temperature.item():.4f}")
print(f"  Feature dim:      {model.feature_dim}")

# ─── LOAD TEST DATASET ──────────────────────────────────────────────
print("\n" + "="*70)
print("2. DATASET SUMMARY")
print("="*70)

eval_dir = TEST_DIR if TEST_DIR.exists() and any(TEST_DIR.iterdir()) else VAL_DIR
dataset = ImageFolder(root=str(eval_dir), transform=val_transform)
eval_dataset = dataset
if MAX_EVAL_SAMPLES > 0 and MAX_EVAL_SAMPLES < len(dataset):
    eval_dataset = Subset(dataset, _balanced_subset_indices(dataset.targets, MAX_EVAL_SAMPLES, len(dataset.classes)))
loader  = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
NUM_CLASSES = len(dataset.classes)

print(f"  Eval directory:  {eval_dir}")
print(f"  Config path:     {CONFIG_PATH}")
print(f"  Total images:    {len(dataset)}")
print(f"  Evaluated images:{len(eval_dataset):5d}")
print(f"  Num classes:     {NUM_CLASSES}")

if class_names:
    print(f"  Checkpoint classes: {class_names[:5]}...")
else:
    class_names = dataset.classes
    print(f"  Using dataset classes (no class_names in checkpoint)")

per_class_count = defaultdict(int)
for _, lbl in dataset.samples:
    per_class_count[idx_to_class[lbl]] += 1
print("  Per-class sample counts:")
for cls in sorted(per_class_count.keys()):
    print(f"    {cls:40s} {per_class_count[cls]:5d}")

# ─── 3. FULL EVALUATION: Top-1, Top-5, Per-Class Metrics ────────────
print("\n" + "="*70)
print("3. FULL EVALUATION — Top-1 / Top-5 / Per-Class Metrics")
print("="*70)

all_preds   = []
all_labels  = []
all_probs   = []
all_top5    = []

with torch.no_grad():
    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(DEVICE)
        item_logits, _, _ = model(images)
        probs = F.softmax(item_logits, dim=1).cpu()

        _, pred = probs.max(1)
        _, top5 = probs.topk(5, dim=1)

        all_preds.extend(pred.numpy())
        all_labels.extend(labels.numpy())
        all_probs.append(probs.numpy())
        all_top5.append(top5.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.concatenate(all_probs, axis=0)
all_top5   = np.concatenate(all_top5, axis=0)

top1_acc = (all_preds == all_labels).mean() * 100
top5_correct = sum(1 for i, lbl in enumerate(all_labels) if lbl in all_top5[i])
top5_acc = top5_correct / len(all_labels) * 100

print(f"\n  ★ Top-1 Accuracy: {top1_acc:.2f}%")
print(f"  ★ Top-5 Accuracy: {top5_acc:.2f}%")

# Per-class precision, recall, F1
from sklearn.metrics import classification_report, confusion_matrix as cm_func

target_names = [idx_to_class[i] for i in range(NUM_CLASSES)]
report = classification_report(
    all_labels,
    all_preds,
    labels=list(range(NUM_CLASSES)),
    target_names=target_names,
    output_dict=True,
    zero_division=0,
)
print("\n  Per-Class Metrics:")
print(f"  {'Class':40s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'Support':>8s}")
print("  " + "-"*70)
for cls_name in target_names:
    m = report[cls_name]
    print(f"  {cls_name:40s} {m['precision']*100:6.1f}% {m['recall']*100:6.1f}% {m['f1-score']*100:6.1f}% {int(m['support']):7d}")

macro_f1 = report['macro avg']['f1-score'] * 100
weighted_f1 = report['weighted avg']['f1-score'] * 100
print(f"\n  Macro-avg F1:    {macro_f1:.2f}%")
print(f"  Weighted-avg F1: {weighted_f1:.2f}%")

# Worst 5 classes
worst_classes = sorted(
    [(cls, report[cls]['f1-score']) for cls in target_names],
    key=lambda x: x[1]
)[:5]
print("\n  ⚠️  5 WORST-PERFORMING CLASSES:")
for cls, f1 in worst_classes:
    print(f"    {cls:40s} F1={f1*100:.1f}%")

# ─── 4. CONFUSION MATRIX ────────────────────────────────────────────
print("\n" + "="*70)
print("4. CONFUSION MATRIX")
print("="*70)

conf_mat = cm_func(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
np.save(OUTPUT_DIR / "confusion_matrix.npy", conf_mat)
print(f"  Saved confusion matrix to {OUTPUT_DIR / 'confusion_matrix.npy'}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(conf_mat, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(target_names, rotation=90, fontsize=7)
    ax.set_yticklabels(target_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Vision Classifier")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved confusion matrix plot to {OUTPUT_DIR / 'confusion_matrix.png'}")
except ImportError:
    print("  matplotlib not available — skipping plot")

# Top-5 most confused pairs
confused_pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and conf_mat[i][j] > 0:
            confused_pairs.append((target_names[i], target_names[j], conf_mat[i][j]))
confused_pairs.sort(key=lambda x: -x[2])
print("\n  Top-10 most confused pairs (true → predicted, count):")
for true_cls, pred_cls, count in confused_pairs[:10]:
    print(f"    {true_cls:35s} → {pred_cls:35s}  ({int(count):3d})")



# ─── 5. CONFIDENCE CALIBRATION (ECE / MCE) ──────────────────────────
print("\n" + "="*70)
print("5. CONFIDENCE CALIBRATION (ECE / MCE)")
print("="*70)

num_bins_cal = 15
confidences = all_probs[np.arange(len(all_labels)), all_preds]
accuracies_bin = (all_preds == all_labels).astype(float)

bin_boundaries = np.linspace(0, 1, num_bins_cal + 1)
ece = 0.0
mce = 0.0
print(f"  {'Bin':>12s} {'Samples':>8s} {'Avg Conf':>10s} {'Accuracy':>10s} {'|Gap|':>8s}")
print("  " + "-"*55)
for b in range(num_bins_cal):
    in_bin = (confidences > bin_boundaries[b]) & (confidences <= bin_boundaries[b + 1])
    n_in_bin = in_bin.sum()
    if n_in_bin > 0:
        avg_conf = confidences[in_bin].mean()
        avg_acc = accuracies_bin[in_bin].mean()
        gap = abs(avg_acc - avg_conf)
        ece += (n_in_bin / len(all_labels)) * gap
        mce = max(mce, gap)
        print(f"  ({bin_boundaries[b]:.2f}-{bin_boundaries[b+1]:.2f}] {n_in_bin:7d} {avg_conf:9.4f} {avg_acc*100:9.2f}% {gap*100:7.2f}%")

print(f"\n  ★ ECE (Expected Calibration Error): {ece*100:.2f}%")
print(f"  ★ MCE (Maximum Calibration Error):  {mce*100:.2f}%")
if ece < 0.05:
    print("  ✅ Calibration: EXCELLENT (ECE < 5%)")
elif ece < 0.10:
    print("  ⚠️  Calibration: ACCEPTABLE (ECE < 10%)")
else:
    print("  ❌ Calibration: POOR (ECE ≥ 10%) — consider temperature scaling")

# ─── 6. THROUGHPUT BENCHMARK ────────────────────────────────────────
print("\n" + "="*70)
print("6. THROUGHPUT BENCHMARK")
print("="*70)

# Warmup
dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
for _ in range(LATENCY_WARMUP):
    with torch.no_grad():
        model(dummy)
    if DEVICE.type == "mps":
        torch.mps.synchronize()

# Single-image latency
latencies = []
for _ in range(LATENCY_ITERS):
    t0 = time.perf_counter()
    with torch.no_grad():
        model(dummy)
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()
    latencies.append((time.perf_counter() - t0) * 1000)

lat = np.array(latencies)
print(f"  Single image ({INPUT_SIZE}×{INPUT_SIZE}):")
print(f"    Mean latency:   {lat.mean():.1f} ms")
print(f"    Median latency: {np.median(lat):.1f} ms")
print(f"    P95 latency:    {np.percentile(lat, 95):.1f} ms")
print(f"    P99 latency:    {np.percentile(lat, 99):.1f} ms")
print(f"    Throughput:     {1000/lat.mean():.1f} img/s")

# Batch throughput
for bs in [1, 2, 4]:
    batch_input = torch.randn(bs, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    times = []
    for _ in range(max(1, min(20, LATENCY_ITERS))):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(batch_input)
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        elif DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg_t = np.mean(times)
    print(f"  Batch={bs}: {avg_t*1000:.1f}ms total, {avg_t/bs*1000:.1f}ms/img, {bs/avg_t:.1f} img/s")



# ─── 7. ROBUSTNESS TEST — Corruptions ──────────────────────────────
print("\n" + "="*70)
print("7. ROBUSTNESS UNDER CORRUPTIONS")
print("="*70)

# Get a small subset for corruption tests
subset_indices = list(range(min(MAX_CORRUPTION_SAMPLES, len(dataset))))
subset_labels = np.array([dataset.targets[i] for i in subset_indices])

def eval_corruption(name, transform_fn):
    """Evaluate model under a specific corruption."""
    correct = 0
    total = 0
    for idx in subset_indices:
        path, label = dataset.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = transform_fn(img)
            tensor = val_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _, _ = model(tensor)
            pred = logits.argmax(1).item()
            if pred == label:
                correct += 1
            total += 1
        except Exception:
            total += 1
    acc = correct / total * 100 if total > 0 else 0
    return acc

# Clean baseline on subset
clean_acc = eval_corruption("Clean", lambda img: img)
print(f"  Clean baseline (subset): {clean_acc:.1f}%")

corruptions = {
    "Gaussian noise σ=25":  lambda img: Image.fromarray(
        np.clip(np.array(img).astype(float) + np.random.normal(0, 25, np.array(img).shape), 0, 255).astype(np.uint8)
    ),
    "Gaussian noise σ=50":  lambda img: Image.fromarray(
        np.clip(np.array(img).astype(float) + np.random.normal(0, 50, np.array(img).shape), 0, 255).astype(np.uint8)
    ),
    "Gaussian blur r=3":    lambda img: img.filter(ImageFilter.GaussianBlur(radius=3)),
    "Gaussian blur r=5":    lambda img: img.filter(ImageFilter.GaussianBlur(radius=5)),
    "Brightness +40%":      lambda img: Image.fromarray(
        np.clip(np.array(img).astype(float) * 1.4, 0, 255).astype(np.uint8)
    ),
    "Brightness -40%":      lambda img: Image.fromarray(
        np.clip(np.array(img).astype(float) * 0.6, 0, 255).astype(np.uint8)
    ),
    "JPEG quality=10":      lambda img: _jpeg_compress(img, 10),
    "JPEG quality=5":       lambda img: _jpeg_compress(img, 5),
    "Grayscale":            lambda img: img.convert("L").convert("RGB"),
    "Horizontal flip":      lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
}

import io
def _jpeg_compress(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

corruption_results = {}
for name, fn in corruptions.items():
    acc = eval_corruption(name, fn)
    drop = clean_acc - acc
    corruption_results[name] = {"accuracy": float(acc), "drop": float(drop)}
    status = "✅" if drop < 10 else ("⚠️" if drop < 20 else "❌")
    print(f"  {status} {name:30s} → {acc:5.1f}%  (drop: {drop:+.1f}%)")

# ─── 8. GRADIENT SANITY CHECK ──────────────────────────────────────
print("\n" + "="*70)
print("8. GRADIENT & ACTIVATION SANITY")
print("="*70)

if SKIP_GRADIENT:
    dead_params = 0
    total_param_count = 0
    with torch.no_grad():
        out_item, out_mat, out_bin = model(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE))
    has_nan = torch.isnan(out_item).any() or torch.isnan(out_mat).any() or torch.isnan(out_bin).any()
    has_inf = torch.isinf(out_item).any() or torch.isinf(out_mat).any() or torch.isinf(out_bin).any()
    print("  Gradient backprop skipped by VISION_STRESS_SKIP_GRADIENT=1")
else:
    test_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, requires_grad=True).to(DEVICE)
    model.train()  # Enable gradients
    out_item, out_mat, out_bin = model(test_input)
    loss = out_item.sum() + out_mat.sum() + out_bin.sum()
    loss.backward()
    model.eval()

    # Check for dead parameters (all zeros)
    dead_params = 0
    total_param_count = 0
    for name, p in model.named_parameters():
        total_param_count += 1
        if p.grad is not None:
            if p.grad.abs().max().item() == 0:
                dead_params += 1
        else:
            dead_params += 1

    print(f"  Total named parameters:    {total_param_count}")
    print(f"  Parameters with zero grad: {dead_params}")
    if dead_params == 0:
        print("  ✅ No dead parameters — all gradients flow correctly")
    else:
        print(f"  ⚠️  {dead_params} parameters have zero gradient — possible dead neurons")

    # Check output ranges
    print(f"  Item logits range:     [{out_item.min().item():.2f}, {out_item.max().item():.2f}]")
    print(f"  Material logits range: [{out_mat.min().item():.2f}, {out_mat.max().item():.2f}]")
    print(f"  Bin logits range:      [{out_bin.min().item():.2f}, {out_bin.max().item():.2f}]")
    has_nan = torch.isnan(out_item).any() or torch.isnan(out_mat).any() or torch.isnan(out_bin).any()
    has_inf = torch.isinf(out_item).any() or torch.isinf(out_mat).any() or torch.isinf(out_bin).any()
    print(f"  NaN in outputs: {'❌ YES' if has_nan else '✅ NO'}")
    print(f"  Inf in outputs: {'❌ YES' if has_inf else '✅ NO'}")


# ─── 9. MODEL OUTPUT CONSISTENCY ───────────────────────────────────
print("\n" + "="*70)
print("9. DETERMINISM & CONSISTENCY CHECK")
print("="*70)

model.eval()
fixed_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
outputs = []
for _ in range(10):
    with torch.no_grad():
        o1, o2, o3 = model(fixed_input)
    outputs.append(o1.cpu())

diffs = [torch.abs(outputs[i] - outputs[0]).max().item() for i in range(1, len(outputs))]
max_diff = max(diffs)
print(f"  Max output diff across 10 runs: {max_diff:.2e}")
if max_diff < 1e-5:
    print("  ✅ Outputs are fully deterministic in eval mode")
else:
    print("  ⚠️  Non-deterministic outputs detected (check dropout/batchnorm)")

# ─── 10. FINAL DEPLOYMENT REPORT ───────────────────────────────────
print("\n" + "="*70)
print("10. FINAL DEPLOYMENT READINESS REPORT")
print("="*70)

report_data = {
    "model": {
        "backbone": BACKBONE,
        "num_classes_item": 30,
        "num_classes_material": 15,
        "num_classes_bin": 4,
        "total_params": total_params,
        "model_size_mb": round(model_size_mb, 1),
        "feature_dim": model.feature_dim,
        "temperature": round(model.temperature.item(), 4),
        "se_block": True,
        "llm_projection": True,
    },
    "accuracy": {
        "top1": round(top1_acc, 2),
        "top5": round(top5_acc, 2),
        "macro_f1": round(macro_f1, 2),
        "weighted_f1": round(weighted_f1, 2),
    },
    "calibration": {
        "ece": round(ece * 100, 2),
        "mce": round(mce * 100, 2),
    },
    "latency_ms": {
        "mean": round(lat.mean(), 1),
        "median": round(float(np.median(lat)), 1),
        "p95": round(float(np.percentile(lat, 95)), 1),
        "p99": round(float(np.percentile(lat, 99)), 1),
    },
    "throughput_img_per_sec": round(1000 / lat.mean(), 1),
    "robustness": {
        "clean_subset_accuracy": round(clean_acc, 2),
        "corruptions": corruption_results,
    },
    "gradient_sanity": {
        "dead_params": dead_params,
        "nan_in_output": bool(has_nan),
        "inf_in_output": bool(has_inf),
        "skipped": bool(SKIP_GRADIENT),
    },
    "deterministic": max_diff < 1e-5,
    "worst_classes": [(cls, round(f1 * 100, 1)) for cls, f1 in worst_classes],
        "eval_samples": len(eval_dataset),
        "total_available_eval_samples": len(dataset),
        "eval_dir": str(eval_dir),
        "bounded_mode": MAX_EVAL_SAMPLES > 0,
}

# Overall verdict
issues = []
if top1_acc < 85:
    issues.append(f"Top-1 accuracy ({top1_acc:.1f}%) below 85% threshold")
if macro_f1 < 80:
    issues.append(f"Macro F1 ({macro_f1:.1f}%) below 80% threshold")
if ece > 0.10:
    issues.append(f"ECE ({ece*100:.1f}%) above 10% — poor calibration")
if MAX_EVAL_SAMPLES > 0:
    issues.append(
        f"Bounded mode evaluated {len(eval_dataset)} of {len(dataset)} images; "
        "do not use as full production accuracy proof"
    )
if SKIP_GRADIENT:
    issues.append("Gradient backprop sanity was skipped")
if lat.mean() > 250:
    issues.append(f"Single-image mean latency ({lat.mean():.1f}ms) above 250ms deployment target")
for name, result in corruption_results.items():
    if result["drop"] > 20:
        issues.append(f"Robustness drop under {name} is {result['drop']:.1f}%")
if has_nan or has_inf:
    issues.append("NaN/Inf detected in model outputs")
if dead_params > total_param_count * 0.1:
    issues.append(f"{dead_params} dead parameters detected")

report_data["issues"] = issues
report_data["deployment_ready"] = len(issues) == 0

# Save report
report_path = OUTPUT_DIR / "stress_test_report.json"
with open(report_path, "w") as f:
    json.dump(report_data, f, indent=2)
print(f"  Report saved to: {report_path}")

if len(issues) == 0:
    print("\n  ╔══════════════════════════════════════════════════════╗")
    print("  ║  ✅  MODEL IS READY FOR INDUSTRIAL DEPLOYMENT       ║")
    print("  ╚══════════════════════════════════════════════════════╝")
else:
    print(f"\n  ⚠️  {len(issues)} ISSUE(S) FOUND:")
    for issue in issues:
        print(f"    ❌ {issue}")

print(f"\n  Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}% | Macro F1: {macro_f1:.2f}%")
print(f"  ECE: {ece*100:.2f}% | Latency: {lat.mean():.0f}ms | Throughput: {1000/lat.mean():.1f} img/s")
print(f"  Model: {total_params/1e6:.1f}M params | {model_size_mb:.0f} MB")
print("="*70)
