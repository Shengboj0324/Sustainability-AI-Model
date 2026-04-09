#!/usr/bin/env python3
"""
Industrial-Grade Comprehensive Evaluation of ResNet50d Waste Classification Model
==================================================================================
Four Pillars:
  1. Exhaustive Quantitative Performance Audit
  2. Robustness & Environmental Stress Testing
  3. Comparative Benchmark vs. Generalist Models
  4. Sustainability Feature Validation
"""

import os, sys, json, time, random, logging, warnings
from pathlib import Path
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import timm
from PIL import Image, ImageFilter, ImageEnhance

# Log to both console and file
_log_file = os.path.join(os.path.dirname(__file__), '..', 'evaluation_results', 'eval_log.txt')
os.makedirs(os.path.dirname(_log_file), exist_ok=True)
_fh = logging.FileHandler(_log_file, mode='w')
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[_fh, _sh], force=True)
logger = logging.getLogger(__name__)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
ROOT = Path("/Users/jiangshengbo/Desktop/Sustainability-AI-Model")
CKPT = ROOT / "checkpoints" / "best_model_epoch19_acc94.62.pth"
RESULTS_DIR = ROOT / "evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)

TARGET_CLASSES = [
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes',
    'cardboard_packaging', 'clothing', 'coffee_grounds', 'disposable_plastic_cutlery',
    'eggshells', 'food_waste', 'glass_beverage_bottles', 'glass_cosmetic_containers',
    'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'paper_cups',
    'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers',
    'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws',
    'plastic_trash_bags', 'plastic_water_bottles', 'shoes', 'steel_food_cans',
    'styrofoam_cups', 'styrofoam_food_containers', 'tea_bags'
]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}
NUM_CLASSES = 30
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# ── RECYCLABILITY MAP (Pillar 4) ──────────────────────────────────────────────
RECYCLABILITY = {
    'aerosol_cans':              {'bin': 'recycle', 'material': 'steel/aluminum', 'notes': 'Empty before recycling'},
    'aluminum_food_cans':        {'bin': 'recycle', 'material': 'aluminum',       'notes': 'Rinse, highly recyclable'},
    'aluminum_soda_cans':        {'bin': 'recycle', 'material': 'aluminum',       'notes': 'Most recycled item globally'},
    'cardboard_boxes':           {'bin': 'recycle', 'material': 'cardboard',      'notes': 'Flatten before recycling'},
    'cardboard_packaging':       {'bin': 'recycle', 'material': 'cardboard',      'notes': 'Remove tape/labels'},
    'clothing':                  {'bin': 'donate/textile', 'material': 'textile', 'notes': 'Donate or textile recycling'},
    'coffee_grounds':            {'bin': 'compost', 'material': 'organic',        'notes': 'Excellent compost material'},
    'disposable_plastic_cutlery':{'bin': 'landfill', 'material': 'PS (#6)',       'notes': 'Not recyclable in most areas'},
    'eggshells':                 {'bin': 'compost', 'material': 'organic/calcium','notes': 'Compostable, adds calcium'},
    'food_waste':                {'bin': 'compost', 'material': 'organic',        'notes': 'Compost or anaerobic digestion'},
    'glass_beverage_bottles':    {'bin': 'recycle', 'material': 'glass',          'notes': 'Infinitely recyclable'},
    'glass_cosmetic_containers': {'bin': 'recycle', 'material': 'glass',          'notes': 'Remove pumps/caps first'},
    'glass_food_jars':           {'bin': 'recycle', 'material': 'glass',          'notes': 'Rinse, remove lids'},
    'magazines':                 {'bin': 'recycle', 'material': 'coated paper',   'notes': 'Recyclable despite glossy coating'},
    'newspaper':                 {'bin': 'recycle', 'material': 'paper',          'notes': 'Highly recyclable'},
    'office_paper':              {'bin': 'recycle', 'material': 'paper',          'notes': 'Remove staples if possible'},
    'paper_cups':                {'bin': 'landfill', 'material': 'paper+PE lining','notes': 'PE lining prevents recycling'},
    'plastic_cup_lids':          {'bin': 'recycle', 'material': 'PP (#5)',        'notes': 'Check local recycling'},
    'plastic_detergent_bottles': {'bin': 'recycle', 'material': 'HDPE (#2)',      'notes': 'Rinse, highly recyclable'},
    'plastic_food_containers':   {'bin': 'recycle', 'material': 'PET (#1)/PP (#5)','notes': 'Rinse, check resin code'},
    'plastic_shopping_bags':     {'bin': 'special', 'material': 'LDPE (#4)',      'notes': 'Store drop-off only'},
    'plastic_soda_bottles':      {'bin': 'recycle', 'material': 'PET (#1)',       'notes': 'Most recyclable plastic'},
    'plastic_straws':            {'bin': 'landfill', 'material': 'PP (#5)',       'notes': 'Too small for recycling machinery'},
    'plastic_trash_bags':        {'bin': 'landfill', 'material': 'LDPE (#4)',     'notes': 'Not recyclable curbside'},
    'plastic_water_bottles':     {'bin': 'recycle', 'material': 'PET (#1)',       'notes': 'Most recyclable plastic'},
    'shoes':                     {'bin': 'donate/special', 'material': 'mixed',   'notes': 'Donate or specialty recycler'},
    'steel_food_cans':           {'bin': 'recycle', 'material': 'steel',          'notes': 'Magnetic, easy to sort'},
    'styrofoam_cups':            {'bin': 'landfill', 'material': 'EPS (#6)',      'notes': 'Not recyclable, harmful'},
    'styrofoam_food_containers': {'bin': 'landfill', 'material': 'EPS (#6)',      'notes': 'Not recyclable, banned in many areas'},
    'tea_bags':                  {'bin': 'compost', 'material': 'organic+paper',  'notes': 'Most are compostable'},
}

DATA_SOURCES = [
    {"name": "master_30",   "path": ROOT / "data/kaggle/recyclable-and-household-waste-classification/images/images", "type": "master"},
    {"name": "garbage_12",  "path": ROOT / "data/kaggle/garbage-classification-mostafa/garbage_classification",       "type": "mapped_12"},
    {"name": "waste_22k",   "path": ROOT / "data/kaggle/waste-classification-data/DATASET",                          "type": "mapped_2"},
    {"name": "garbage_v2_10","path": ROOT / "data/kaggle/garbage-classification-v2",                                 "type": "mapped_10"},
    {"name": "garbage_6",   "path": ROOT / "data/kaggle/garbage-classification",                                     "type": "mapped_6"},
    {"name": "garbage_balanced","path": ROOT / "data/kaggle/garbage-dataset-classification",                          "type": "mapped_6"},
    {"name": "warp_industrial","path": ROOT / "data/kaggle/warp-waste-recycling-plant-dataset/Warp-C/train_crops",   "type": "industrial"},
    {"name": "multiclass_garbage","path": ROOT / "data/kaggle/multi-class-garbage-classification-dataset",           "type": "multiclass"},
]


# ── HELPER: Load model ────────────────────────────────────────────────────────
def load_model():
    """Load the trained ResNet50d model from checkpoint."""
    model = timm.create_model('resnet50d', pretrained=False, num_classes=NUM_CLASSES,
                              drop_rate=0.1, drop_path_rate=0.1)
    ckpt = torch.load(str(CKPT), map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logger.info(f"✅ Model loaded from {CKPT.name} (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.2f}%)")
    return model


# ── HELPER: Transform ─────────────────────────────────────────────────────────
class EvalTransform:
    """Clean eval transform — PIL → Tensor, no NumPy."""
    def __init__(self, size=224, mean=MEAN, std=STD):
        self.size = size
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std  = torch.tensor(std).view(3,1,1)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert('RGB')
        img = img.resize((self.size, self.size), Image.BICUBIC)
        t = torch.frombuffer(img.tobytes(), dtype=torch.uint8).clone()
        t = t.view(self.size, self.size, 3).permute(2,0,1).float().div_(255.0)
        return t.sub_(self.mean).div_(self.std)


# ── DATASET: Unified loader (mirrors training notebook) ───────────────────────
class EvalWasteDataset(Dataset):
    """Minimal dataset that re-uses the same ingestion logic as training."""
    SKIP_DIRS = {'default', 'real_world', 'images', 'train', 'test', 'val',
                 'TRAIN', 'TEST', 'VAL', '.ipynb_checkpoints'}
    LABEL_MAP_12 = {
        'paper': 'office_paper', 'cardboard': 'cardboard_boxes', 'plastic': 'plastic_food_containers',
        'metal': 'aluminum_food_cans', 'trash': 'plastic_trash_bags', 'glass': 'glass_food_jars',
        'shoes': 'shoes', 'clothes': 'clothing',
        'biological': 'food_waste', 'white-glass': 'glass_beverage_bottles',
        'brown-glass': 'glass_food_jars', 'green-glass': 'glass_beverage_bottles',
        'battery': None,
    }
    LABEL_MAP_2 = {'O': 'plastic_food_containers', 'R': 'cardboard_boxes'}
    LABEL_MAP_10 = {
        'paper': 'office_paper', 'cardboard': 'cardboard_boxes', 'plastic': 'plastic_food_containers',
        'metal': 'aluminum_food_cans', 'trash': 'plastic_trash_bags', 'glass': 'glass_food_jars',
        'shoes': 'shoes', 'clothes': 'clothing',
        'organic waste': 'food_waste', 'household items': 'plastic_food_containers',
    }
    LABEL_MAP_6 = {
        'paper': 'office_paper', 'cardboard': 'cardboard_boxes', 'plastic': 'plastic_food_containers',
        'metal': 'aluminum_food_cans', 'trash': 'plastic_trash_bags', 'glass': 'glass_food_jars',
    }
    LABEL_MAP_INDUSTRIAL = {
        'paper': 'office_paper', 'cardboard': 'cardboard_boxes', 'plastic': 'plastic_food_containers',
        'metal': 'aluminum_food_cans', 'organic': 'food_waste', 'glass': 'glass_food_jars',
        'textile': 'clothing', 'wood': 'cardboard_boxes',
        'rubber': 'shoes', 'other': 'plastic_trash_bags',
    }
    LABEL_MAP_MULTI = {
        'paper': 'office_paper', 'cardboard': 'cardboard_boxes', 'plastic': 'plastic_food_containers',
        'metal': 'aluminum_food_cans', 'trash': 'plastic_trash_bags', 'glass': 'glass_food_jars',
        'shoes': 'shoes', 'clothes': 'clothing',
    }

    def __init__(self, sources, transform=None, real_world_only=False):
        self.transform = transform or EvalTransform()
        self.samples = []
        self.source_map = []
        for src in sources:
            self._ingest(src, real_world_only)
        logger.info(f"EvalWasteDataset: {len(self.samples)} samples loaded")

    def _ingest(self, source, real_world_only):
        src_path = Path(source['path'])
        src_type = source['type']
        src_name = source['name']
        if not src_path.exists():
            return
        lmap = {'mapped_12': self.LABEL_MAP_12, 'mapped_2': self.LABEL_MAP_2,
                'mapped_10': self.LABEL_MAP_10, 'mapped_6': self.LABEL_MAP_6,
                'industrial': self.LABEL_MAP_INDUSTRIAL, 'multiclass': self.LABEL_MAP_MULTI}.get(src_type)

        IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for cls_dir in sorted(src_path.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name.startswith('.'):
                continue
            raw = cls_dir.name.lower()
            if raw in self.SKIP_DIRS:
                continue
            if src_type == 'master':
                target = cls_dir.name
                if target not in CLASS_TO_IDX:
                    continue
                subdirs = [d for d in cls_dir.iterdir() if d.is_dir() and d.name in ('real_world','default')]
                if subdirs:
                    for sd in subdirs:
                        if real_world_only and sd.name != 'real_world':
                            continue
                        for f in sd.iterdir():
                            if f.suffix.lower() in IMG_EXT:
                                self.samples.append((str(f), CLASS_TO_IDX[target]))
                                self.source_map.append(src_name)
                elif not real_world_only:
                    for f in cls_dir.iterdir():
                        if f.suffix.lower() in IMG_EXT:
                            self.samples.append((str(f), CLASS_TO_IDX[cls_dir.name]))
                            self.source_map.append(src_name)
            elif not real_world_only and lmap:
                mapped = lmap.get(raw)
                if not mapped or mapped not in CLASS_TO_IDX:
                    continue
                def _add_files(directory):
                    for f in directory.iterdir():
                        if f.is_dir():
                            _add_files(f)
                        elif f.suffix.lower() in IMG_EXT:
                            self.samples.append((str(f), CLASS_TO_IDX[mapped]))
                            self.source_map.append(src_name)
                _add_files(cls_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        return self.transform(img), label



# ── NOISE INJECTION TRANSFORMS (Pillar 2) ─────────────────────────────────────
class NoisyTransform(EvalTransform):
    """Applies a perturbation BEFORE the standard eval transform."""
    def __init__(self, noise_type='gaussian', severity=1, **kwargs):
        super().__init__(**kwargs)
        self.noise_type = noise_type
        self.severity = severity

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert('RGB')
        # Apply perturbation on PIL image
        if self.noise_type == 'gaussian':
            arr = np.array(img).astype(np.float32)
            sigma = [10, 25, 50][min(self.severity, 2)]
            arr += np.random.normal(0, sigma, arr.shape).astype(np.float32)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        elif self.noise_type == 'blur':
            radius = [1, 3, 5][min(self.severity, 2)]
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        elif self.noise_type == 'brightness':
            factor = [0.7, 0.4, 0.2][min(self.severity, 2)]
            img = ImageEnhance.Brightness(img).enhance(factor)
        elif self.noise_type == 'contrast':
            factor = [0.7, 0.4, 0.2][min(self.severity, 2)]
            img = ImageEnhance.Contrast(img).enhance(factor)
        return super().__call__(img)


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 1: Exhaustive Quantitative Performance Audit
# ══════════════════════════════════════════════════════════════════════════════
def pillar1_quantitative_audit(model, dataset, batch_size=64):
    """Full classification report, confusion matrix, failure analysis, latency."""
    logger.info("\n" + "="*80)
    logger.info("PILLAR 1: Exhaustive Quantitative Performance Audit")
    logger.info("="*80)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_labels, all_probs = [], [], []

    # ── Inference + Latency benchmark ──
    times = []
    total_batches = len(loader)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            t0 = time.time()
            logits = model(images)
            t1 = time.time()
            times.append((t1 - t0, images.size(0)))
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
                done = sum(n for _, n in times)
                logger.info(f"  [P1] {batch_idx+1}/{total_batches} batches ({done}/{len(dataset)} images)")
                _fh.flush()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    total_time = sum(t for t, _ in times)
    total_imgs = sum(n for _, n in times)
    latency_ms = (total_time / total_imgs) * 1000
    throughput = total_imgs / total_time

    # ── Overall metrics ──
    accuracy = (all_preds == all_labels).mean() * 100
    logger.info(f"\n📊 Overall Accuracy: {accuracy:.2f}% ({(all_preds == all_labels).sum()}/{len(all_labels)})")
    logger.info(f"⏱  Latency: {latency_ms:.2f} ms/image")
    logger.info(f"🚀 Throughput: {throughput:.1f} images/sec (CPU, {os.cpu_count()} threads)")

    # ── Per-class metrics ──
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    report = classification_report(all_labels, all_preds, target_names=TARGET_CLASSES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    # ── Top 5 failure modes ──
    logger.info("\n🔍 TOP 5 FAILURE MODE ANALYSIS:")
    class_f1 = [(TARGET_CLASSES[i], report[TARGET_CLASSES[i]]['f1-score'],
                  report[TARGET_CLASSES[i]]['support']) for i in range(NUM_CLASSES)]
    class_f1.sort(key=lambda x: x[1])
    for rank, (cls, f1, sup) in enumerate(class_f1[:5], 1):
        idx = CLASS_TO_IDX[cls]
        row = cm[idx]
        total = row.sum()
        misclass = [(TARGET_CLASSES[j], row[j]) for j in range(NUM_CLASSES) if j != idx and row[j] > 0]
        misclass.sort(key=lambda x: -x[1])
        top_confusions = misclass[:3]
        logger.info(f"\n  #{rank} {cls} (F1={f1:.4f}, Support={sup})")
        logger.info(f"     Correct: {row[idx]}/{total} ({row[idx]/max(total,1)*100:.1f}%)")
        for conf_cls, conf_n in top_confusions:
            logger.info(f"     → Confused with '{conf_cls}': {conf_n} ({conf_n/max(total,1)*100:.1f}%)")

    results = {
        'accuracy': accuracy, 'latency_ms': latency_ms, 'throughput': throughput,
        'per_class_report': report, 'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.tolist(),
    }
    np.save(str(RESULTS_DIR / 'confusion_matrix_full.npy'), cm)
    logger.info(f"\n✅ Pillar 1 complete. CM saved to {RESULTS_DIR}/confusion_matrix_full.npy")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 2: Robustness & Environmental Stress Testing
# ══════════════════════════════════════════════════════════════════════════════
def pillar2_robustness(model, full_dataset, batch_size=64):
    """Noise injection + OOD real-world test."""
    logger.info("\n" + "="*80)
    logger.info("PILLAR 2: Robustness & Environmental Stress Testing")
    logger.info("="*80)

    # Use a 2000-sample subset for speed
    indices = random.sample(range(len(full_dataset)), min(2000, len(full_dataset)))
    results = {}

    # ── 2A: Noise injection suite ──
    logger.info("\n── 2A: Noise Injection Suite ──")
    conditions = [
        ('Clean (baseline)', None, None),
        ('Gaussian σ=10',  'gaussian',  0),
        ('Gaussian σ=25',  'gaussian',  1),
        ('Gaussian σ=50',  'gaussian',  2),
        ('Blur r=1',       'blur',      0),
        ('Blur r=3',       'blur',      1),
        ('Blur r=5',       'blur',      2),
        ('Dim (0.7x)',     'brightness',0),
        ('Dim (0.4x)',     'brightness',1),
        ('Low contrast',   'contrast',  1),
    ]
    noise_results = {}
    for name, ntype, severity in conditions:
        if ntype is None:
            tfm = EvalTransform()
        else:
            tfm = NoisyTransform(noise_type=ntype, severity=severity)
        # Rebuild subset with new transform
        subset_samples = [full_dataset.samples[i] for i in indices]
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(subset_samples), batch_size):
                batch = subset_samples[i:i+batch_size]
                imgs, labs = [], []
                for path, label in batch:
                    try:
                        img = Image.open(path).convert('RGB')
                    except Exception:
                        img = Image.new('RGB', (224,224), (128,128,128))
                    imgs.append(tfm(img))
                    labs.append(label)
                imgs = torch.stack(imgs)
                labs = torch.tensor(labs)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labs).sum().item()
                total += len(labs)
        acc = correct / total * 100
        noise_results[name] = acc
        delta = acc - noise_results.get('Clean (baseline)', acc)
        logger.info(f"  {name:20s}: {acc:6.2f}% (Δ={delta:+.2f}%)")

    results['noise_injection'] = noise_results

    # ── 2B: OOD Real-world test ──
    logger.info("\n── 2B: Out-of-Distribution (Real-World) Test ──")
    master_source = [s for s in DATA_SOURCES if s['name'] == 'master_30']
    rw_dataset = EvalWasteDataset(master_source, transform=EvalTransform(), real_world_only=True)
    logger.info(f"  Real-world images: {len(rw_dataset)}")

    if len(rw_dataset) > 0:
        rw_loader = DataLoader(rw_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labs in rw_loader:
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labs).sum().item()
                total += len(labs)
        rw_acc = correct / total * 100
        results['real_world_accuracy'] = rw_acc
        logger.info(f"  Real-world accuracy: {rw_acc:.2f}%")
        logger.info(f"  Full-dataset accuracy: ~94.62% (from training)")
        logger.info(f"  Domain gap: {94.62 - rw_acc:+.2f}%")
    else:
        logger.warning("  No real-world images found!")

    logger.info(f"\n✅ Pillar 2 complete.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 3: Comparative Benchmark vs. Generalist Models
# ══════════════════════════════════════════════════════════════════════════════
def pillar3_comparative(model, full_dataset, batch_size=64):
    """Compare fine-tuned model vs ImageNet-pretrained baseline + material analysis."""
    logger.info("\n" + "="*80)
    logger.info("PILLAR 3: Comparative Benchmark vs. Generalist Models")
    logger.info("="*80)

    results = {}

    # ── 3A: Fine-tuned vs ImageNet-pretrained baseline ──
    logger.info("\n── 3A: Specialized Model vs. ImageNet Baseline ──")
    baseline = timm.create_model('resnet50d', pretrained=True, num_classes=NUM_CLASSES)
    baseline.eval()

    # Use 2000-sample subset for speed
    indices = random.sample(range(len(full_dataset)), min(2000, len(full_dataset)))
    subset_samples = [full_dataset.samples[i] for i in indices]
    tfm = EvalTransform()

    for model_name, mdl in [('Fine-tuned ResNet50d', model), ('ImageNet Baseline', baseline)]:
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(subset_samples), batch_size):
                batch = subset_samples[i:i+batch_size]
                imgs, labs = [], []
                for path, label in batch:
                    try:
                        img = Image.open(path).convert('RGB')
                    except Exception:
                        img = Image.new('RGB',(224,224),(128,128,128))
                    imgs.append(tfm(img))
                    labs.append(label)
                imgs = torch.stack(imgs)
                labs = torch.tensor(labs)
                preds = mdl(imgs).argmax(dim=1)
                correct += (preds == labs).sum().item()
                total += len(labs)
        acc = correct / total * 100
        results[model_name] = acc
        logger.info(f"  {model_name:30s}: {acc:.2f}%")

    improvement = results.get('Fine-tuned ResNet50d',0) - results.get('ImageNet Baseline',0)
    results['improvement'] = improvement
    logger.info(f"  Improvement from specialized training: +{improvement:.2f}%")

    # ── 3B: Material distinction analysis ──
    logger.info("\n── 3B: Visually-Similar Material Distinction ──")
    confusion_groups = {
        'Glass types': ['glass_beverage_bottles', 'glass_cosmetic_containers', 'glass_food_jars'],
        'Cardboard types': ['cardboard_boxes', 'cardboard_packaging'],
        'Aluminum vs Steel': ['aluminum_food_cans', 'aluminum_soda_cans', 'steel_food_cans'],
        'PET plastics': ['plastic_soda_bottles', 'plastic_water_bottles', 'plastic_detergent_bottles'],
        'Plastic containers': ['plastic_food_containers', 'plastic_cup_lids', 'disposable_plastic_cutlery'],
        'Paper types': ['newspaper', 'office_paper', 'magazines'],
        'Styrofoam types': ['styrofoam_cups', 'styrofoam_food_containers'],
    }

    group_results = {}
    # Get the full-set predictions for material analysis
    logger.info("  Running full-dataset inference for material analysis...")
    loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bi, (imgs, labs) in enumerate(loader):
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
            if (bi+1) % 100 == 0:
                logger.info(f"  [P3] {bi+1}/{len(loader)} batches")
                _fh.flush()
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    for group_name, classes in confusion_groups.items():
        idxs = [CLASS_TO_IDX[c] for c in classes]
        mask = np.isin(all_labels, idxs)
        if mask.sum() == 0:
            continue
        group_preds = all_preds[mask]
        group_labels = all_labels[mask]
        # Within-group accuracy (did it pick the RIGHT class among confusable ones?)
        correct = (group_preds == group_labels).sum()
        total = len(group_labels)
        # How many went to ANOTHER class in the SAME group (confused) vs outside
        within_group_wrong = sum(1 for p, l in zip(group_preds, group_labels)
                                 if p != l and p in idxs)
        outside_wrong = sum(1 for p, l in zip(group_preds, group_labels)
                           if p != l and p not in idxs)
        acc = correct / total * 100
        group_results[group_name] = {
            'accuracy': acc, 'total': int(total), 'correct': int(correct),
            'within_group_errors': int(within_group_wrong),
            'outside_group_errors': int(outside_wrong),
        }
        logger.info(f"  {group_name:25s}: {acc:6.2f}% ({correct}/{total})")
        logger.info(f"    Intra-group confusion: {within_group_wrong}  |  Out-of-group errors: {outside_wrong}")

    results['material_groups'] = group_results
    logger.info(f"\n✅ Pillar 3 complete.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 4: Sustainability Feature Validation
# ══════════════════════════════════════════════════════════════════════════════
def pillar4_sustainability(model, full_dataset):
    """Recyclability mapping + class imbalance + multi-source analysis."""
    logger.info("\n" + "="*80)
    logger.info("PILLAR 4: Sustainability Feature Validation")
    logger.info("="*80)

    results = {}

    # ── 4A: Recyclability mapping ──
    logger.info("\n── 4A: Environmental Knowledge — Recyclability Mapping ──")
    bins = defaultdict(list)
    for cls, info in RECYCLABILITY.items():
        bins[info['bin']].append(cls)
    for b, classes in sorted(bins.items()):
        logger.info(f"  {b.upper():15s} ({len(classes)} classes): {', '.join(classes)}")

    logger.info(f"\n  📊 Disposal distribution:")
    logger.info(f"     Recyclable:  {len(bins['recycle'])} classes")
    logger.info(f"     Compostable: {len(bins['compost'])} classes")
    logger.info(f"     Landfill:    {len(bins['landfill'])} classes")
    logger.info(f"     Special:     {len(bins.get('special',[]))+len(bins.get('donate/special',[]))+len(bins.get('donate/textile',[]))} classes")

    # ── 4B: Bin-level accuracy (the REAL sustainability metric) ──
    logger.info("\n── 4B: Bin-Level Accuracy (Sustainability Impact) ──")
    loader = DataLoader(full_dataset, batch_size=64, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Map predictions and labels to bins
    idx_to_bin = {}
    for cls, info in RECYCLABILITY.items():
        idx_to_bin[CLASS_TO_IDX[cls]] = info['bin']
    pred_bins = np.array([idx_to_bin.get(p, 'unknown') for p in all_preds])
    true_bins = np.array([idx_to_bin.get(l, 'unknown') for l in all_labels])
    bin_acc = (pred_bins == true_bins).mean() * 100
    logger.info(f"  🎯 Bin-level accuracy: {bin_acc:.2f}%")
    logger.info(f"     (If the model puts waste in the CORRECT bin, even if exact class is wrong)")
    results['bin_level_accuracy'] = bin_acc

    # ── 4C: Class imbalance handling ──
    logger.info("\n── 4C: Class Imbalance Analysis ──")
    class_counts = Counter(all_labels)
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
    min_cls, min_n = sorted_counts[0]
    max_cls, max_n = sorted_counts[-1]
    logger.info(f"  Largest class:  {TARGET_CLASSES[max_cls]} ({max_n} samples)")
    logger.info(f"  Smallest class: {TARGET_CLASSES[min_cls]} ({min_n} samples)")
    logger.info(f"  Imbalance ratio: {max_n/max(min_n,1):.1f}:1")

    # Accuracy for top-5 and bottom-5 classes by sample count
    logger.info(f"\n  Accuracy by sample frequency:")
    for label, desc in [("Top-5 (most samples)", sorted_counts[-5:]),
                        ("Bottom-5 (fewest samples)", sorted_counts[:5])]:
        correct = sum((all_preds[all_labels == idx] == idx).sum() for idx, _ in desc)
        total = sum(n for _, n in desc)
        logger.info(f"    {label}: {correct/total*100:.2f}% ({correct}/{total})")

    # ── 4D: Multi-source contribution ──
    logger.info("\n── 4D: Multi-Source Data Lake Analysis ──")
    source_counts = Counter(full_dataset.source_map)
    logger.info(f"  Sources contributing to evaluation:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = count / len(full_dataset) * 100
        logger.info(f"    {src:25s}: {count:6d} images ({pct:5.1f}%)")
    results['source_distribution'] = dict(source_counts)

    logger.info(f"\n✅ Pillar 4 complete.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    logger.info("="*80)
    logger.info("COMPREHENSIVE MODEL EVALUATION — Industrial Grade")
    logger.info("="*80)

    # Setup
    n = os.cpu_count() or 4
    torch.set_num_threads(n)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load model
    model = load_model()

    # Load full dataset
    logger.info("\n📂 Loading full evaluation dataset...")
    dataset = EvalWasteDataset(DATA_SOURCES, transform=EvalTransform())

    # Run all four pillars
    all_results = {}

    all_results['pillar1'] = pillar1_quantitative_audit(model, dataset)
    all_results['pillar2'] = pillar2_robustness(model, dataset)
    all_results['pillar3'] = pillar3_comparative(model, dataset)
    all_results['pillar4'] = pillar4_sustainability(model, dataset)

    # ── Save final report ──
    logger.info("\n" + "="*80)
    logger.info("SAVING FINAL REPORT")
    logger.info("="*80)

    # Convert numpy types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        return obj

    report_path = RESULTS_DIR / "comprehensive_evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    logger.info(f"✅ Full report saved to {report_path}")

    # ── Summary ──
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    p1 = all_results['pillar1']
    p2 = all_results['pillar2']
    p3 = all_results['pillar3']
    p4 = all_results['pillar4']
    logger.info(f"  P1 Accuracy:         {p1['accuracy']:.2f}%")
    logger.info(f"  P1 Latency:          {p1['latency_ms']:.2f} ms/img")
    logger.info(f"  P1 Throughput:        {p1['throughput']:.1f} img/s")
    logger.info(f"  P2 Real-world OOD:   {p2.get('real_world_accuracy','N/A')}")
    logger.info(f"  P3 vs Baseline:      +{p3.get('improvement', 0):.2f}%")
    logger.info(f"  P4 Bin-level Acc:    {p4.get('bin_level_accuracy','N/A')}")
    logger.info(f"\n🏁 EVALUATION COMPLETE")


if __name__ == '__main__':
    main()
