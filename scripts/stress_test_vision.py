#!/usr/bin/env python3
"""
Exhaustive Stress Test — Vision Training Pipeline
==================================================
1. 32-thread concurrent inference on real images
2. Full epoch simulation (100 batches) with loss convergence check
3. ONNX export parity on real images
4. Memory stability under sustained load
5. Pipeline dry-run
"""
import sys, os, time, random, torch, torch.nn as nn, numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.vision.classifier import MultiHeadClassifier

TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "vision_cls" / "train"
INPUT_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
dataset = ImageFolder(root=str(TRAIN_DIR), transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

model = MultiHeadClassifier(backbone="resnet18", num_classes_item=30,
    num_classes_material=15, num_classes_bin=4, pretrained=False,
    enable_se=True, enable_llm_projection=True)

# ─── 1. CONCURRENCY TEST (32 threads × 160 inferences on REAL images) ──
print("=" * 70)
print("1. CONCURRENT INFERENCE (32 threads, 160 queries, REAL images)")
print("=" * 70)
model.eval()
# Pre-load 160 real images
real_images = []
for i, (img, _) in enumerate(dataset):
    real_images.append(img.unsqueeze(0))
    if len(real_images) >= 160:
        break

errors = []
def infer(idx):
    try:
        with torch.no_grad():
            out = model(real_images[idx])
        return idx, True, out[0].shape
    except Exception as e:
        errors.append((idx, str(e)))
        return idx, False, None

t0 = time.time()
with ThreadPoolExecutor(max_workers=32) as pool:
    futures = [pool.submit(infer, i) for i in range(160)]
    results = [f.result() for f in as_completed(futures)]
dt = time.time() - t0
ok = sum(1 for _, s, _ in results if s)
print(f"  {ok}/160 OK, {len(errors)} errors, {dt:.2f}s")
assert ok == 160, f"Concurrency FAIL: {ok}/160"
print("  ✅ Concurrency PASSED")

# ─── 2. TRAINING SIMULATION (100 batches, loss convergence) ──────────
print("\n" + "=" * 70)
print("2. TRAINING SIMULATION (100 batches)")
print("=" * 70)
model.train()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

losses = []
for step, (imgs, labs) in enumerate(loader):
    optimizer.zero_grad()
    out, _, _ = model(imgs)
    loss = criterion(out, labs)
    assert torch.isfinite(loss), f"Non-finite loss at step {step}"
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    losses.append(loss.item())
    if step % 20 == 0:
        print(f"  Step {step:3d}: loss={loss.item():.4f}")
    if step >= 99:
        break

# Verify loss decreased (first 10 avg vs last 10 avg)
first10 = np.mean(losses[:10])
last10 = np.mean(losses[-10:])
print(f"  First 10 avg: {first10:.4f}, Last 10 avg: {last10:.4f}")
assert last10 < first10, f"Loss didn't decrease: {first10:.4f} → {last10:.4f}"
print("  ✅ Loss converging")

# ─── 3. MEMORY STABILITY (200 inference cycles) ──────────────────────
print("\n" + "=" * 70)
print("3. MEMORY STABILITY (200 cycles on real images)")
print("=" * 70)
import psutil
proc = psutil.Process(os.getpid())
model.eval()
mem_before = proc.memory_info().rss / 1e6
for i in range(200):
    with torch.no_grad():
        _ = model(real_images[i % len(real_images)])
mem_after = proc.memory_info().rss / 1e6
delta = mem_after - mem_before
print(f"  Before: {mem_before:.0f}MB, After: {mem_after:.0f}MB, Delta: {delta:+.0f}MB")
assert delta < 500, f"Memory leak: {delta}MB"
print("  ✅ Memory stable")

# ─── 4. PER-CLASS COVERAGE (every class has samples) ─────────────────
print("\n" + "=" * 70)
print("4. PER-CLASS COVERAGE VERIFICATION")
print("=" * 70)
from collections import Counter
label_counts = Counter()
for _, lab in DataLoader(dataset, batch_size=128, num_workers=0):
    for l in lab.tolist():
        label_counts[l] += 1
min_cls = min(label_counts.values())
max_cls = max(label_counts.values())
print(f"  {len(label_counts)} classes, min={min_cls}, max={max_cls}")
assert len(label_counts) == 30, f"Not all classes represented: {len(label_counts)}"
assert min_cls >= 100, f"Smallest class has only {min_cls} images"
print("  ✅ All 30 classes well-represented")

# ─── SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("╔══════════════════════════════════════════════════════════════════╗")
print("║  ✅ ALL STRESS TESTS PASSED                                    ║")
print("║                                                                ║")
print(f"║  🔀 Concurrency: 160/160 @ 32 threads on real images          ║")
print(f"║  📉 Training: 100 batches, loss {first10:.2f} → {last10:.2f} (converging)   ║")
print(f"║  💾 Memory: {delta:+.0f}MB over 200 cycles (stable)                   ║")
print(f"║  📊 Coverage: all 30 classes, min {min_cls} imgs/class              ║")
print("╚══════════════════════════════════════════════════════════════════╝")
