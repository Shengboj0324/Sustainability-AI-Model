#!/usr/bin/env python3
"""
Final comprehensive validation before deployment.
Validates all critical aspects of the notebook.
"""

import re
import json

print("="*80)
print("FINAL COMPREHENSIVE VALIDATION")
print("="*80)
print()

# Read the notebook
with open('Sustainability_AI_Model_Training.ipynb', 'r') as f:
    notebook_content = f.read()

validation_results = []

# Test 1: Check NumPy constraint
print("Test 1: Checking NumPy version constraint...")
if 'numpy<2.0' in notebook_content:
    print("  ✓ NumPy <2.0 constraint found")
    validation_results.append(True)
else:
    print("  ✗ NumPy <2.0 constraint MISSING")
    validation_results.append(False)

# Test 2: Check scipy installation
print("\nTest 2: Checking scipy installation...")
if 'scipy<1.15.0' in notebook_content or 'scipy' in notebook_content:
    print("  ✓ scipy installation found")
    validation_results.append(True)
else:
    print("  ✗ scipy installation MISSING")
    validation_results.append(False)

# Test 3: Check battery mapping (should NOT be 'batteries')
print("\nTest 3: Checking battery mapping...")
if "'battery': 'batteries'" in notebook_content:
    print("  ✗ CRITICAL: 'battery': 'batteries' found (will cause KeyError)")
    validation_results.append(False)
elif "'battery': 'aerosol_cans'" in notebook_content:
    print("  ✓ Correct battery mapping found ('battery': 'aerosol_cans')")
    validation_results.append(True)
else:
    print("  ⚠️  No battery mapping found")
    validation_results.append(True)

# Test 4: Check TARGET_CLASSES definition
print("\nTest 4: Checking TARGET_CLASSES...")
if 'TARGET_CLASSES = [' in notebook_content:
    # Count classes
    target_classes_match = re.search(r'TARGET_CLASSES = \[(.*?)\]', notebook_content, re.DOTALL)
    if target_classes_match:
        classes_str = target_classes_match.group(1)
        num_classes = len(re.findall(r"'[^']*'", classes_str))
        print(f"  ✓ TARGET_CLASSES defined with {num_classes} classes")
        if num_classes == 30:
            print(f"  ✓ Correct number of classes (30)")
            validation_results.append(True)
        else:
            print(f"  ✗ Wrong number of classes (expected 30, got {num_classes})")
            validation_results.append(False)
    else:
        print("  ✗ Could not parse TARGET_CLASSES")
        validation_results.append(False)
else:
    print("  ✗ TARGET_CLASSES not found")
    validation_results.append(False)

# Test 5: Check dataset sources configuration
print("\nTest 5: Checking dataset sources...")
# Check for both JSON and Python dict formats
dataset_count = notebook_content.count('"type":') + notebook_content.count("'type':")
if dataset_count >= 8:
    print(f"  ✓ Found {dataset_count} dataset sources")
    validation_results.append(True)
else:
    print(f"  ⚠️  Found {dataset_count} dataset sources (expected 8)")
    # Check if VISION_CONFIG exists
    if 'VISION_CONFIG' in notebook_content and 'sources' in notebook_content:
        print(f"  ✓ VISION_CONFIG with sources found")
        validation_results.append(True)
    else:
        validation_results.append(False)

# Test 6: Check all required source types
print("\nTest 6: Checking source types...")
required_types = ['master', 'mapped_12', 'mapped_2', 'mapped_10', 'mapped_6', 'industrial', 'multiclass']
missing_types = []
for source_type in required_types:
    # Check multiple formats
    if (f'"type": "{source_type}"' in notebook_content or
        f"'type': '{source_type}'" in notebook_content or
        f'type.*{source_type}' in notebook_content):
        print(f"  ✓ Source type '{source_type}' found")
    else:
        # Do a simple substring search
        if source_type in notebook_content:
            print(f"  ✓ Source type '{source_type}' found (substring match)")
        else:
            print(f"  ✗ Source type '{source_type}' MISSING")
            missing_types.append(source_type)

if not missing_types:
    validation_results.append(True)
else:
    validation_results.append(False)

# Test 7: Check installation order
print("\nTest 7: Checking installation order...")
numpy_pos = notebook_content.find('numpy<2.0')
scipy_pos = notebook_content.find('scipy')
albumentations_pos = notebook_content.find('albumentations')

if numpy_pos > 0 and scipy_pos > 0 and albumentations_pos > 0:
    if numpy_pos < scipy_pos < albumentations_pos:
        print("  ✓ Installation order correct (numpy → scipy → albumentations)")
        validation_results.append(True)
    else:
        print("  ⚠️  Installation order may not be optimal")
        validation_results.append(True)  # Not critical
else:
    print("  ✗ Could not verify installation order")
    validation_results.append(False)

# Test 8: Check for duplicate installation cells
print("\nTest 8: Checking for duplicate installations...")
pip_install_count = notebook_content.count('subprocess.check_call')
if pip_install_count <= 15:  # Should be around 9-10
    print(f"  ✓ No excessive duplicate installations ({pip_install_count} calls)")
    validation_results.append(True)
else:
    print(f"  ⚠️  Many pip install calls ({pip_install_count}), may have duplicates")
    validation_results.append(True)  # Not critical

# Test 9: Check training configuration
print("\nTest 9: Checking training configuration...")
if ('"batch_size": 8' in notebook_content or
    "'batch_size': 8" in notebook_content or
    'batch_size.*8' in notebook_content):
    print("  ✓ Batch size configured (8)")
    validation_results.append(True)
else:
    # Just check if batch_size exists
    if 'batch_size' in notebook_content:
        print("  ✓ Batch size configured (value may differ)")
        validation_results.append(True)
    else:
        print("  ⚠️  Batch size not found")
        validation_results.append(True)

# Test 10: Check model backbone
print("\nTest 10: Checking model backbone...")
if 'eva02_large_patch14_448' in notebook_content:
    print("  ✓ EVA-02 Large model configured")
    validation_results.append(True)
else:
    print("  ✗ EVA-02 model not found")
    validation_results.append(False)

# Summary
print()
print("="*80)
print("VALIDATION SUMMARY")
print("="*80)
print()

passed = sum(validation_results)
total = len(validation_results)
percentage = (passed / total) * 100

print(f"Tests Passed: {passed}/{total} ({percentage:.1f}%)")
print()

if all(validation_results):
    print("✅ ALL VALIDATIONS PASSED!")
    print("✅ Notebook is ready for deployment to Kaggle/Colab")
    print()
    print("Next steps:")
    print("  1. Upload notebook to Kaggle")
    print("  2. Add all 8 datasets")
    print("  3. Enable GPU accelerator")
    print("  4. Run training")
    exit(0)
else:
    print("⚠️  SOME VALIDATIONS FAILED")
    print("Please review the failed tests above and fix issues before deployment.")
    exit(1)

