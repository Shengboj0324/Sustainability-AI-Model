# Vision Model — Audit, Physics-Informed Redesign & Stress-Test Suite

_Engineering report. Replaces the marketing claims in `README.md`
("97.2/100 capability score", "67,883 req/s") with measured numbers and a
reproducible path to improve them._

---

## 1. Audit — what was actually there

### 1.1 Three divergent checkpoints, none matching the serving code
| Artifact | Real architecture | Params | Reported acc |
|---|---|---|---|
| `checkpoints/best_model_epoch19_acc94.62.pth` | **resnet50d**, single-head, 30-cls | 23.6M | 94.62% val (macro-F1 0.891) |
| `deployment_package/best_model.pth` (3.74 GB) | **EVA-02 Large @448**, multi-head + SE + temperature | 325M | 93.20% val (epoch 41) |
| `best_vision_eva02_lake.pth` (90 MB) | **mislabeled — actually resnet50d** | 23.6M | — |

The "94.62%" is **weighted** accuracy on a long-tailed set (clothing 11k @ 99.9%,
shoes 4.4k @ 99.7%). The honest **macro-F1 is 0.91**, and per-class F1 ranges down
to **0.37**.

### 1.2 Measured per-class failures (`deployment_package/comprehensive_evaluation_report.json`)
- **Glass is broken.** `glass_beverage_bottles` F1 **0.47** (P 0.41 / R 0.55);
  `glass_food_jars` F1 0.70. "Glass types" group accuracy **64.7%** with **2,346
  within-group errors** — the model cannot separate a glass bottle from a glass jar.
- **`plastic_trash_bags` recall 0.23** — misses 77%.
- **`food_waste` precision 0.47** — a dumping ground for organics.
- **Robustness collapse**: clean 88.6% → Gaussian σ=50 **24%**; blur r=5 **51.7%**.

### 1.3 Broken serving + training code (now fixed / replaced)
- `models/vision/classifier.py` defaulted to **`vit_base_patch16_224` / 20 classes** —
  matched **no** trained checkpoint.
- The vision service passes the **nested** `configs/vision_cls.yaml`, but
  `WasteClassifier` read **flat** keys → `KeyError` on load. The deployed model
  could not load through the serving path at all.
- `training/vision/train_multihead.py` calls `WasteClassifier(model_name=…)` —
  a kwarg that doesn't exist; `dataset.py` expects COCO `*_annotations.json`
  that don't exist. The real model was trained by the notebook on **ImageFolder**
  data, with material/bin labels **derived from a static item→material→bin
  lookup** — so the material/bin heads carry no independent signal and, being
  independent softmaxes, can emit physically impossible triples at inference.

---

## 2. Redesign — physics-informed, hierarchically-consistent

New modules (all unit-verified on CPU):

| File | Role |
|---|---|
| `models/vision/taxonomy.py` | Single source of truth: 30 items → 16 materials → 6 bins + physical properties (density, resin code, transparency, metallic sheen, ferromagnetism, recyclability) + mapping matrices |
| `models/vision/physics_informed_classifier.py` | Redesigned model |
| `training/vision/physics_losses.py` | 6-term physics-informed loss |
| `configs/vision_physics.yaml` + `training/vision/train_physics_informed.py` | Kaggle-ready trainer |
| `evaluation/evaluate_vision_v2.py` | Honest evaluation harness |
| `evaluation/stress_test_vision.py` | Corruption / OOD / adversarial / latency |
| `evaluation/interpret_vision.py` | Embedding geometry + Grad-CAM |

### 2.1 Consistency by construction
Material and bin distributions are **derived from the item prediction** through
the frozen taxonomy mapping matrices (`p_mat = p_item @ M_item_material`). In
`hard` mode the disposal decision is taken from the item argmax routed through
the taxonomy → **the (item, material, bin) triple is always physically legal,
verified across 40 random batches → `decision_violation_rate = 0.0`**. In `soft`
mode a learnable gate blends a direct perceptual head with the derived
distribution, and impossible mass is penalized explicitly.

### 2.2 Robustness to fine-grained confusion (the glass fix)
`glass_beverage_bottles` and `glass_food_jars` are both `glass → recycle`. By
decoupling the **disposal decision** from the **fine item ID**, a within-group
item error no longer corrupts the user-facing answer. The eval harness confirms
the principle even untrained: bin-decision accuracy exceeds item accuracy.

### 2.3 The physics
`MaterialPropertyScorer` seeds a learnable material embedding from the physical
property table, so material scoring is grounded in transparency / metallic sheen
/ density / resin code, not texture alone.

### 2.4 Imbalance + contrastive
`physics_losses.py`: **logit-adjusted** item loss (removes mega-class gradient
dominance behind the `plastic_trash_bags` / `food_waste` failures), consistency
KL, impossible-pair penalty, and a **confusion-group supervised-contrastive**
term that pushes glass / film / metal sub-classes apart.

### 2.5 Correct bin taxonomy
The legacy 4-bin head can't represent `special` (store drop-off for LDPE film)
or `donate` (textiles/shoes) — so it mis-routes plastic bags to curbside
recycling, a real contamination error. The new taxonomy models 6 bins;
`taxonomy.disposal_for_item("plastic_shopping_bags") → bin="special"`.

---

## 3. Fixes already applied to the repo
- `models/vision/classifier.py`: default config now matches the **deployed
  EVA-02** architecture; `_normalize_config` accepts the nested service YAML.
  `MultiHeadClassifier` default is EVA-02 / 30-class. **The vision service can
  now load the deployed model.**
- `models/vision/taxonomy.py::disposal_for_item` — the recommended consistent
  serving contract.

---

## 4. How to run (Kaggle / Colab GPU)
```bash
# 0. local sanity (CPU, tiny backbone) — proves the pipeline before GPU spend
python training/vision/train_physics_informed.py --config configs/vision_physics.yaml --smoke --data-dir /tmp/vis_smoke2

# 1. BASELINE truth: evaluate the deployed EVA-02 on the real test set
python evaluation/evaluate_vision_v2.py --arch legacy \
  --backbone eva02_large_patch14_448.mim_m38m_ft_in22k_in1k \
  --ckpt models/vision/classifier/best_model.pth \
  --data-dir data/processed/vision_cls/test --input-size 448 \
  --out evaluation_results/baseline_eva02.json

# 2. TRAIN the physics-informed model (set model.warm_start_ckpt in the yaml to reuse the EVA-02 backbone)
python training/vision/train_physics_informed.py --config configs/vision_physics.yaml --wandb

# 3. EVALUATE + STRESS + INTERPRET the new model
python evaluation/evaluate_vision_v2.py  --arch physics --ckpt models/vision/classifier_physics/best_model.pth --data-dir data/processed/vision_cls/test --input-size 448 --out evaluation_results/physics_v1.json
python evaluation/stress_test_vision.py  --arch physics --ckpt models/vision/classifier_physics/best_model.pth --data-dir data/processed/vision_cls/test --input-size 448 --out evaluation_results/stress_physics.json
python evaluation/interpret_vision.py    --arch physics --ckpt models/vision/classifier_physics/best_model.pth --data-dir data/processed/vision_cls/test --input-size 448 --out evaluation_results/interpret_physics.json
```

## 5. Success criteria (what "accuracy properly elevated" means here)
Judge the new model on the metrics the legacy headline hid:
- **macro-F1** ↑ (not weighted) — target the 0.91 → 0.94+ range.
- **glass group accuracy** 64.7% → 85%+.
- **worst-class F1** (`plastic_trash_bags`, `glass_beverage_bottles`) 0.37/0.47 → 0.70+.
- **bin-decision accuracy** ≥ 96% and **stable under corruption** (the real UX metric).
- **physics-consistency violations = 0** (guaranteed in `hard` mode).
- **calibration (ECE)** ↓ and confidence that *drops* under corruption.

## 6. Honest status of claims
- "97.2/100 capability score" / "67,883 req/s": **unsubstantiated** — no
  measurement backs them. Use the harness numbers instead.
- Real deployed model: **~90% top-1 / macro-F1 0.91**, with the per-class and
  robustness weaknesses above. The redesign is engineered to fix exactly those.
