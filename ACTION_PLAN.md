# Training System - Action Plan

## 🎯 Objective
Execute a complete, error-free training run of the Sustainability AI Model with comprehensive validation and monitoring.

---

## ⚡ Quick Start (3 Steps)

### Step 1: Fix Environment (2 minutes)
```bash
cd /Users/jiangshengbo/Desktop/Sustainability-AI-Model
./fix_environment.sh
```

**Expected Output**:
```
✅ NumPy fixed!
✅ PyTorch 2.2.0
✅ MPS (Apple Silicon) available
✅ EVA02 Base model available
✅ ALL CHECKS PASSED - READY TO TRAIN!
```

### Step 2: Restart Jupyter Kernel (30 seconds)
1. Open `Sustainability_AI_Model_Training.ipynb`
2. Click: **Kernel → Restart Kernel**
3. Confirm restart

### Step 3: Run Training (2-4 hours)
1. Run **Cell 4**: Import statements and function definitions
2. Run **Cell 15**: Start training

**Watch for these messages**:
```
✅ Transform validation passed
✅ Pre-training sanity check passed!
Epoch 1/20: Train Acc XX.XX%, Val Loss X.XXXX, Val Acc XX.XX%
```

---

## 📋 Detailed Execution Plan

### Phase 1: Environment Preparation (5 minutes)

#### 1.1 Fix NumPy Version
```bash
./fix_environment.sh
```

**Validation**:
- [ ] Script completes without errors
- [ ] NumPy version < 2.0
- [ ] All diagnostic checks pass

**If script fails**, manual fix:
```bash
pip install "numpy<2.0"
python3 training_diagnostics.py
```

#### 1.2 Verify Setup
```bash
python3 training_diagnostics.py
```

**Expected Results**:
- ✅ PyTorch Installation
- ✅ Model Availability
- ✅ Transform Pipeline
- ✅ Forward Pass

**If any check fails**: Review error message and fix before proceeding

---

### Phase 2: Jupyter Kernel Restart (1 minute)

#### 2.1 Open Notebook
- File: `Sustainability_AI_Model_Training.ipynb`
- Location: `/Users/jiangshengbo/Desktop/Sustainability-AI-Model/`

#### 2.2 Restart Kernel
- Menu: **Kernel → Restart Kernel**
- Confirm: Click "Restart"
- Wait: Kernel status shows "Idle"

**Why**: Loads the fixed `get_vision_transforms()` function into memory

---

### Phase 3: Training Execution (2-4 hours)

#### 3.1 Run Cell 4 (Imports and Functions)
**Cell Content**: Import statements and function definitions

**Expected Output**:
```
No output (silent success)
```

**Validation**:
- [ ] Cell executes without errors
- [ ] No import errors
- [ ] Cell number shows [4]

#### 3.2 Run Cell 15 (Training)
**Cell Content**: Main training execution

**Expected Startup Sequence** (first 2 minutes):
```
================================================================================
Phase 1: Multi-Source Data Lake Vision Training
================================================================================
✓ Random seed set to 42
🍎 Using Apple Silicon MPS (Metal Performance Shaders)
   Optimized for M1/M2/M3 chips
Creating model: eva02_base_patch14_224
Model parameters: 85.78M total, 85.78M trainable
⚠️  Gradient checkpointing disabled (incompatible with MPS - causes crashes)
🔍 Validating transform pipeline...
  Transform output shape: torch.Size([3, 224, 224])
  Expected: (3, 224, 224)
  ✅ Transform validation passed
📂 Ingesting master_30 from ./data/kaggle/...
...
📊 Dataset Summary:
  ✓ Total images loaded: XXXXX
🔍 Running pre-training sanity check...
  Batch shape: torch.Size([2, 3, 224, 224])
  Expected: [batch_size, 3, 224, 224]
  Model output shape: torch.Size([2, 30])
  Expected: [batch_size, 30]
  ✅ Pre-training sanity check passed!
  ✅ All images are 224x224
  ✅ Model accepts input and produces correct output shape
```

**Expected Training Progress** (per epoch):
```
Epoch 1/20: 100%|██████████| XXXX/XXXX [XX:XX<00:00, X.XXit/s, loss=X.XXXX, acc=XX.XX%]
Validation: 100%|██████████| XXX/XXX [XX:XX<00:00, X.XXit/s]
📊 Per-Class Performance:
  Macro F1: X.XXXX
  Worst 5 classes:
    class_name: F1=X.XXXX, Support=XXX
Epoch 1/20: Train Acc XX.XX%, Val Loss X.XXXX, Val Acc XX.XX%, Macro F1 X.XXXX
✓ Saved best model checkpoint: checkpoints/best_model_epochX_accXX.XX.pth
```

---

### Phase 4: Monitoring (During Training)

#### 4.1 Health Indicators (Every Epoch)

**✅ Healthy Training**:
- Loss decreases over epochs
- Accuracy increases over epochs
- No "Non-finite loss" warnings
- No "Non-finite gradient" warnings
- < 5% batches skipped
- Validation metrics computed successfully

**⚠️ Warning Signs**:
- Loss increases for 3+ consecutive epochs → Check learning rate
- Accuracy stuck at same value → May need more epochs
- 5-20% batches skipped → Some corrupt images (acceptable)
- Validation loss increases while training loss decreases → Overfitting (early stopping will handle)

**❌ Critical Issues**:
- "Non-finite loss detected" frequently → Stop and investigate
- "Non-finite gradient norm" frequently → Stop and investigate
- > 20% batches skipped → Dataset quality issue
- Python crashes → MPS instability (reduce batch size to 1)

#### 4.2 Expected Timeline

| Milestone | Time | What to Expect |
|-----------|------|----------------|
| Startup | 0-2 min | Model loading, dataset ingestion, validation |
| Epoch 1 | 2-15 min | First epoch (slowest due to compilation) |
| Epoch 2-5 | 15-60 min | Loss decreasing, accuracy improving |
| Epoch 6-10 | 60-120 min | Continued improvement, may plateau |
| Epoch 11-20 | 120-240 min | Fine-tuning, early stopping may trigger |

**Total Expected Time**: 2-4 hours (depends on dataset size and MPS performance)

---

### Phase 5: Validation (After Training)

#### 5.1 Check Training Completion

**Success Indicators**:
- [ ] Training completed without crashes
- [ ] Final message: "✓ Training completed successfully"
- [ ] Best checkpoint saved in `checkpoints/` directory
- [ ] Metrics history saved: `checkpoints/metrics_history.json`
- [ ] Confusion matrix saved: `checkpoints/confusion_matrix.npy`
- [ ] Classification report saved: `checkpoints/classification_report.json`

#### 5.2 Review Metrics

**Check `checkpoints/metrics_history.json`**:
```python
import json
with open('checkpoints/metrics_history.json', 'r') as f:
    metrics = json.load(f)

print(f"Best Val Accuracy: {max(metrics['val_acc']):.2f}%")
print(f"Final Train Loss: {metrics['train_loss'][-1]:.4f}")
print(f"Final Val Loss: {metrics['val_loss'][-1]:.4f}")
```

**Expected Results**:
- Best Val Accuracy > 50% (good)
- Best Val Accuracy > 70% (excellent)
- Training loss < 1.0
- Validation loss < 2.0

#### 5.3 Verify Reproducibility

**Re-run training with same seed**:
1. Restart kernel
2. Run Cell 4
3. Run Cell 15
4. Compare metrics with first run

**Expected**: Metrics should be within ±2% of first run

---

## 🚨 Troubleshooting

### Issue: "Transform validation failed"
**Cause**: Transform pipeline not producing correct size
**Fix**: Check config `input_size` is 224, restart kernel

### Issue: "Pre-training sanity check failed"
**Cause**: Batch shape mismatch
**Fix**: Check dataset loading, verify images exist

### Issue: "Non-finite loss detected" (frequent)
**Cause**: Training instability
**Fix**: Reduce learning rate to 1e-5, reduce batch size to 1

### Issue: Python crashes during training
**Cause**: MPS instability
**Fix**: 
1. Reduce batch_size to 1
2. Reduce input_size to 192
3. Switch to CPU: `device = torch.device("cpu")`

### Issue: "No images loaded"
**Cause**: Dataset paths incorrect
**Fix**: Verify datasets downloaded in `./data/kaggle/`

---

## ✅ Success Criteria Checklist

### Immediate Success (First 5 Minutes):
- [ ] Environment fix script completes successfully
- [ ] Diagnostic checks all pass
- [ ] Jupyter kernel restarts without errors
- [ ] Cell 4 runs without errors
- [ ] Transform validation passes
- [ ] Pre-training sanity check passes

### Training Success (Full Run):
- [ ] Training completes without crashes
- [ ] Loss decreases monotonically
- [ ] Validation accuracy > 50%
- [ ] No frequent NaN/Inf warnings
- [ ] Checkpoints saved successfully
- [ ] Metrics history saved

### Quality Success (Post-Training):
- [ ] Best validation accuracy > 70%
- [ ] Macro F1 score > 0.6
- [ ] Confusion matrix shows good class separation
- [ ] Reproducible results (±2% variance)

---

## 📞 Next Steps After Success

1. **Save the best model**:
   - Already saved in `checkpoints/best_model_epochX_accXX.XX.pth`

2. **Analyze results**:
   - Review confusion matrix
   - Identify worst-performing classes
   - Plan data augmentation improvements

3. **Test inference**:
   - Load best model
   - Test on new images
   - Validate predictions

4. **Optimize further**:
   - Experiment with learning rates
   - Try different augmentations
   - Consider ensemble methods

---

## 📚 Documentation Reference

- **TRAINING_FIXES_APPLIED.md**: Comprehensive list of all fixes
- **ITERATION_REPORT.md**: Detailed iteration report with validation
- **training_diagnostics.py**: Diagnostic script for environment validation
- **fix_environment.sh**: Automated environment fix script

---

**Status**: 🔧 READY FOR EXECUTION
**Confidence**: ⭐⭐⭐⭐⭐ (5/5)
**Estimated Time**: 2-4 hours total

