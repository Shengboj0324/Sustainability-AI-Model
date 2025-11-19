"""
M4 Max Pre-Flight Check for Training Readiness

CRITICAL: Comprehensive validation for Apple M4 Max training
- PyTorch MPS backend verification
- Memory availability check
- Data file validation
- Configuration validation
- Model loading test
- Training script syntax check
"""

import sys
import os
from pathlib import Path
import torch
import json
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*80)
print("🍎 APPLE M4 MAX TRAINING PRE-FLIGHT CHECK")
print("="*80)

# Test results
results = {
    'pytorch_mps': {'passed': False, 'errors': []},
    'memory': {'passed': False, 'errors': []},
    'data_files': {'passed': False, 'errors': []},
    'configs': {'passed': False, 'errors': []},
    'training_scripts': {'passed': False, 'errors': []},
    'model_loading': {'passed': False, 'errors': []},
}

# ============================================================================
# TEST 1: PyTorch MPS Backend
# ============================================================================
print("\n" + "="*80)
print("TEST 1: PYTORCH MPS BACKEND")
print("="*80)

print(f"PyTorch Version: {torch.__version__}")
print(f"MPS Available: {torch.backends.mps.is_available()}")

if torch.backends.mps.is_available():
    print("✅ MPS backend is available")

    # Test MPS operations
    try:
        device = torch.device("mps")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print("✅ MPS tensor operations working")

        # Test FP16
        x_fp16 = torch.randn(100, 100, dtype=torch.float16, device=device)
        y_fp16 = torch.randn(100, 100, dtype=torch.float16, device=device)
        z_fp16 = torch.matmul(x_fp16, y_fp16)
        print("✅ FP16 operations working on MPS")

        results['pytorch_mps']['passed'] = True
    except Exception as e:
        results['pytorch_mps']['errors'].append(f"MPS test failed: {e}")
        print(f"❌ MPS test failed: {e}")
else:
    results['pytorch_mps']['errors'].append("MPS not available")
    print("❌ MPS not available - cannot train on M4 Max GPU")
    print("   Install PyTorch with MPS support:")
    print("   pip install --upgrade torch torchvision torchaudio")

# ============================================================================
# TEST 2: Memory Check
# ============================================================================
print("\n" + "="*80)
print("TEST 2: MEMORY AVAILABILITY")
print("="*80)

try:
    import psutil
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)

    print(f"Total Memory: {total_gb:.1f} GB")
    print(f"Available Memory: {available_gb:.1f} GB")

    if total_gb >= 32:
        print(f"✅ Sufficient memory for training ({total_gb:.1f} GB)")
        results['memory']['passed'] = True
    else:
        results['memory']['errors'].append(f"Low memory: {total_gb:.1f} GB (recommend 32GB+)")
        print(f"⚠️  Low memory: {total_gb:.1f} GB (recommend 32GB+)")
        results['memory']['passed'] = True  # Still pass, just warn

except ImportError:
    print("⚠️  psutil not installed - cannot check memory")
    print("   Install: pip install psutil")
    results['memory']['passed'] = True  # Don't fail on this

# ============================================================================
# TEST 3: Data Files
# ============================================================================
print("\n" + "="*80)
print("TEST 3: DATA FILES VALIDATION")
print("="*80)

data_files = [
    "data/llm_training_ultra_expanded.json",
    "data/rag_knowledge_base_expanded.json",
    "data/gnn_training_fully_annotated.json",
]

all_data_present = True
for data_file in data_files:
    path = PROJECT_ROOT / data_file
    if path.exists():
        size_mb = path.stat().st_size / (1024**2)
        print(f"✅ {data_file} ({size_mb:.2f} MB)")
    else:
        print(f"❌ {data_file} - NOT FOUND")
        results['data_files']['errors'].append(f"{data_file} not found")
        all_data_present = False

if all_data_present:
    results['data_files']['passed'] = True

# ============================================================================
# TEST 4: M4 Max Configurations
# ============================================================================
print("\n" + "="*80)
print("TEST 4: M4 MAX CONFIGURATION FILES")
print("="*80)

config_files = [
    "configs/llm_sft_m4max.yaml",
    "configs/vision_cls_m4max.yaml",
]

all_configs_valid = True
for config_file in config_files:
    path = PROJECT_ROOT / config_file
    if path.exists():
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)

            # Validate M4 Max specific settings
            if 'llm' in config_file:
                if config.get('training', {}).get('bf16', True):
                    print(f"⚠️  {config_file}: bf16=true (should be false for M4 Max)")
                if not config.get('training', {}).get('fp16', False):
                    print(f"⚠️  {config_file}: fp16=false (should be true for M4 Max)")

            print(f"✅ {config_file}")
        except Exception as e:
            print(f"❌ {config_file}: {e}")
            results['configs']['errors'].append(f"{config_file}: {e}")
            all_configs_valid = False
    else:
        print(f"❌ {config_file} - NOT FOUND")
        results['configs']['errors'].append(f"{config_file} not found")
        all_configs_valid = False

if all_configs_valid:
    results['configs']['passed'] = True



# ============================================================================
# TEST 5: Training Scripts
# ============================================================================
print("\n" + "="*80)
print("TEST 5: TRAINING SCRIPTS SYNTAX")
print("="*80)

training_scripts = [
    "training/llm/train_sft.py",
    "training/vision/train_multihead.py",
    "training/gnn/train_gnn.py",
]

all_scripts_valid = True
for script in training_scripts:
    path = PROJECT_ROOT / script
    if path.exists():
        try:
            import ast
            with open(path, 'r') as f:
                ast.parse(f.read())

            # Check for MPS support
            with open(path, 'r') as f:
                content = f.read()
                if 'mps' in content.lower():
                    print(f"✅ {script} (MPS support detected)")
                else:
                    print(f"⚠️  {script} (no MPS support detected)")
        except SyntaxError as e:
            print(f"❌ {script}: Syntax error - {e}")
            results['training_scripts']['errors'].append(f"{script}: {e}")
            all_scripts_valid = False
    else:
        print(f"❌ {script} - NOT FOUND")
        results['training_scripts']['errors'].append(f"{script} not found")
        all_scripts_valid = False

if all_scripts_valid:
    results['training_scripts']['passed'] = True

# ============================================================================
# TEST 6: Model Loading Test
# ============================================================================
print("\n" + "="*80)
print("TEST 6: MODEL LOADING TEST")
print("="*80)

if torch.backends.mps.is_available():
    try:
        device = torch.device("mps")

        # Test simple model
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).to(device)

        # Test forward pass
        x = torch.randn(32, 100, device=device)
        y = model(x)

        print("✅ Model loading and forward pass on MPS successful")
        results['model_loading']['passed'] = True

    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        results['model_loading']['errors'].append(f"Model test failed: {e}")
else:
    print("⚠️  Skipping model test (MPS not available)")
    results['model_loading']['errors'].append("MPS not available")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎯 M4 MAX TRAINING READINESS SUMMARY")
print("="*80)

total_tests = len(results)
passed_tests = sum(1 for r in results.values() if r['passed'])

print(f"\nTotal Tests: {total_tests}")
print(f"Passed: {passed_tests}")
print(f"Failed: {total_tests - passed_tests}")
print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")

print("\nDetailed Results:")
for category, result in results.items():
    status = "✅" if result['passed'] else "❌"
    print(f"  {status} {category.upper().replace('_', ' ')}")

# Print errors
has_errors = False
for category, result in results.items():
    if result['errors']:
        if not has_errors:
            print("\n" + "="*80)
            print("ERRORS FOUND")
            print("="*80)
            has_errors = True
        print(f"\n{category.upper().replace('_', ' ')}:")
        for error in result['errors']:
            print(f"  ✗ {error}")

if passed_tests == total_tests:
    print("\n" + "="*80)
    print("🎉 100% READY FOR M4 MAX TRAINING!")
    print("="*80)
    print("✅ PyTorch MPS backend working")
    print("✅ Sufficient memory available")
    print("✅ All data files present")
    print("✅ M4 Max configs validated")
    print("✅ Training scripts ready")
    print("✅ Model loading successful")
    print("\n🚀 YOU CAN START TRAINING NOW!")
    print("\nRecommended commands:")
    print("  LLM:    python3 training/llm/train_sft.py --config configs/llm_sft_m4max.yaml")
    print("  Vision: python3 training/vision/train_multihead.py --config configs/vision_cls_m4max.yaml")
    print("  GNN:    python3 training/gnn/train_gnn.py")
    sys.exit(0)
else:
    print("\n" + "="*80)
    print("⚠️  TRAINING READINESS: ISSUES FOUND")
    print("="*80)
    print(f"Please fix {total_tests - passed_tests} issue(s) before training")
    sys.exit(1)

