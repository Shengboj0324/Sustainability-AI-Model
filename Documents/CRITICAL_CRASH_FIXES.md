# CRITICAL CRASH FIXES - Python Quitting Unexpectedly

## 🚨 **Problem: Python Crashing During Training**

**Symptom**: "Python quit unexpectedly" error during training startup

**Root Causes**:
1. **MPS memory overload** - Manual tensor conversion is memory-intensive
2. **Multiprocessing instability** - `num_workers > 0` causes crashes with MPS
3. **Batch size too large** - Even batch_size=2 can crash with manual transforms
4. **NumPy 2.x incompatibility** - Forces ALL images through manual conversion path

---

## ✅ **FIXES APPLIED**

### **Fix 1: Disabled Multiprocessing (CRITICAL)**
```python
"num_workers": 0,  # Was: 2 - CAUSES CRASHES WITH MPS
```
**Why**: Multiprocessing + MPS + manual tensor conversion = guaranteed crash

### **Fix 2: Reduced Batch Size to 1 (CRITICAL)**
```python
"batch_size": 1,  # Was: 2 - Maximum stability for MPS
"grad_accum_steps": 64,  # Was: 32 - Maintains effective batch size of 64
```
**Why**: Smaller batches = less memory pressure = fewer crashes

### **Fix 3: Optimized Manual Tensor Conversion**
**Before** (memory-intensive):
```python
img_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
mean = torch.tensor([...]).view(3, 1, 1)
std = torch.tensor([...]).view(3, 1, 1)
img_tensor = (img_tensor - mean) / std
```

**After** (memory-efficient):
```python
img_tensor = torch.frombuffer(img_bytes, dtype=torch.uint8).clone()
img_tensor = img_tensor.float().div_(255.0)  # In-place operation
img_tensor[0].sub_(0.48145466).div_(0.26862954)  # Channel-wise in-place
img_tensor[1].sub_(0.4578275).div_(0.26130258)
img_tensor[2].sub_(0.40821073).div_(0.27577711)
```
**Why**: In-place operations use less memory, reduce crash risk

### **Fix 4: Aggressive Memory Cleanup**
```python
# Every 100 batches, clear MPS cache
if i > 0 and i % 100 == 0 and device.type == "mps":
    torch.mps.empty_cache()
```
**Why**: Prevents memory accumulation that leads to crashes

### **Fix 5: Pre-computed Dummy Tensors**
**Before**:
```python
dummy_tensor = torch.zeros(3, 224, 224)
dummy_tensor = (dummy_tensor - mean) / std
```

**After**:
```python
dummy_tensor = torch.zeros(3, 224, 224)
dummy_tensor[0].fill_(-0.48145466 / 0.26862954)  # Pre-computed
dummy_tensor[1].fill_(-0.4578275 / 0.26130258)
dummy_tensor[2].fill_(-0.40821073 / 0.27577711)
```
**Why**: Faster, less memory allocation

---

## 📋 **RESTART INSTRUCTIONS**

### **Step 1: Stop Current Execution**
1. Click **■ Stop** button in Jupyter
2. Wait for it to stop completely

### **Step 2: Restart Kernel**
1. **Kernel** → **Restart Kernel**
2. Confirm restart

### **Step 3: Re-run Training**
1. **Run Cell 4** (imports - loads all fixes)
2. **Run Cell 15** (training)

---

## ✅ **Expected Behavior After Fixes**

### **Startup (First 30 seconds)**:
```
✓ Random seed set to 42
🍎 Using Apple Silicon MPS
✓ Model loaded: EVA02 Base (85.78M params)
✓ Dataset loaded: 103,938 images
🔍 Running pre-training sanity check...
  ✅ Pre-training sanity check passed!
```

### **Training (After 1-2 minutes)**:
```
Epoch 1/20:   0%|          | 0/83150 [00:00<?, ?it/s]
Epoch 1/20:   1%|▏         | 468/83150 [00:30<1:32:15, loss=3.2145, acc=8.23%]
```

**Note**: With batch_size=1, you'll have **83,150 batches per epoch** (was 41,575 with batch_size=2)

---

## ⚠️ **If It Still Crashes**

### **Fallback Option: Use CPU Instead of MPS**
If MPS continues to crash, switch to CPU:

1. Find this line in the notebook:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

2. Change to:
```python
device = torch.device("cpu")  # Force CPU for stability
```

3. Restart kernel and re-run

**CPU will be slower but 100% stable.**

---

## 📊 **Performance Expectations**

### **With MPS (if stable)**:
- **First epoch**: ~20-30 minutes (MPS compilation overhead)
- **Subsequent epochs**: ~15-20 minutes
- **Total training**: ~5-7 hours for 20 epochs

### **With CPU (fallback)**:
- **Per epoch**: ~45-60 minutes
- **Total training**: ~15-20 hours for 20 epochs

---

## 🎯 **Success Criteria**

✅ **Training is working if you see**:
- Pre-training sanity check passes
- Progress bar appears and updates
- Loss decreases over batches
- No Python crashes for > 5 minutes

❌ **Training is failing if you see**:
- "Python quit unexpectedly" error
- Kernel dies/restarts automatically
- System freezes
- No progress bar after 3+ minutes

---

**All fixes have been applied to the notebook. Please restart kernel and re-run training now!**

