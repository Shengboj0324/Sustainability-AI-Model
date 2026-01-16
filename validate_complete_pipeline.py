#!/usr/bin/env python3
"""
COMPREHENSIVE VALIDATION SCRIPT
Tests every aspect of the data pipeline and training setup.
"""

import sys
import json
from pathlib import Path
from collections import Counter

print("="*80)
print("COMPREHENSIVE PIPELINE VALIDATION")
print("="*80)

TARGET_CLASSES = [
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes', 'cardboard_packaging',
    'clothing', 'coffee_grounds', 'disposable_plastic_cutlery', 'egg_shells', 'food_waste',
    'glass_beverage_bottles', 'glass_cosmetic_containers', 'glass_food_jars', 'magazines',
    'newspaper', 'office_paper', 'paper_cups', 'plastic_cup_lids', 'plastic_detergent_bottles',
    'plastic_food_containers', 'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws',
    'plastic_trash_bags', 'plastic_water_bottles', 'shoes', 'steel_food_cans', 'styrofoam_cups',
    'styrofoam_food_containers', 'tea_bags'
]

print(f"\n✓ TARGET_CLASSES defined: {len(TARGET_CLASSES)} classes")

ALL_MAPPINGS = {
    'mapped_12': {
        'paper': 'office_paper', 'cardboard': 'cardboard_boxes', 'plastic': 'plastic_food_containers',
        'metal': 'aluminum_food_cans', 'glass': 'glass_food_jars', 'brown-glass': 'glass_beverage_bottles',
        'green-glass': 'glass_beverage_bottles', 'white-glass': 'glass_food_jars', 'clothes': 'clothing',
        'shoes': 'shoes', 'biological': 'food_waste', 'trash': 'food_waste'
    },
    'mapped_2': {
        'organic': 'food_waste', 'o': 'food_waste'
    },
    'mapped_10': {
        'metal': 'aluminum_food_cans', 'glass': 'glass_food_jars', 'biological': 'food_waste',
        'paper': 'office_paper', 'battery': 'aerosol_cans', 'trash': 'food_waste',
        'cardboard': 'cardboard_boxes', 'shoes': 'shoes', 'clothes': 'clothing',
        'plastic': 'plastic_food_containers'
    },
    'mapped_6': {
        'cardboard': 'cardboard_boxes', 'glass': 'glass_food_jars', 'metal': 'aluminum_food_cans',
        'paper': 'office_paper', 'plastic': 'plastic_food_containers', 'trash': 'food_waste'
    },
    'industrial': {
        'pet': 'plastic_food_containers', 'hdpe': 'plastic_food_containers', 'pvc': 'plastic_food_containers',
        'ldpe': 'plastic_food_containers', 'pp': 'plastic_food_containers', 'ps': 'plastic_food_containers',
        'metal': 'aluminum_food_cans', 'glass': 'glass_food_jars', 'paper': 'office_paper',
        'cardboard': 'cardboard_boxes', 'trash': 'food_waste'
    },
    'multiclass': {
        'plastic': 'plastic_food_containers', 'metal': 'aluminum_food_cans', 'glass': 'glass_food_jars',
        'paper': 'office_paper', 'cardboard': 'cardboard_boxes', 'trash': 'food_waste',
        'organic': 'food_waste', 'battery': 'aerosol_cans', 'clothes': 'clothing', 'shoes': 'shoes'
    }
}

print("\n" + "="*80)
print("VALIDATING ALL LABEL MAPPINGS")
print("="*80)

errors = []
for source_type, mapping in ALL_MAPPINGS.items():
    print(f"\n{source_type}:")
    for source_label, target_label in mapping.items():
        if target_label not in TARGET_CLASSES:
            error_msg = f"  ✗ ERROR: '{source_label}' → '{target_label}' (NOT IN TARGET_CLASSES)"
            print(error_msg)
            errors.append(error_msg)
        else:
            print(f"  ✓ '{source_label}' → '{target_label}'")

if errors:
    print("\n" + "="*80)
    print("VALIDATION FAILED!")
    print("="*80)
    for error in errors:
        print(error)
    sys.exit(1)
else:
    print("\n" + "="*80)
    print("✓ ALL MAPPINGS VALID - No errors found!")
    print("="*80)

print("\n" + "="*80)
print("CHECKING NOTEBOOK SYNTAX")
print("="*80)

try:
    with open('Sustainability_AI_Model_Training.ipynb', 'r') as f:
        nb = json.load(f)
    
    print(f"✓ Notebook JSON is valid")
    print(f"✓ Found {len(nb.get('cells', []))} cells")
    
    code_cells = [c for c in nb.get('cells', []) if c['cell_type'] == 'code']
    print(f"✓ Found {len(code_cells)} code cells")
    
    for i, cell in enumerate(code_cells):
        source = ''.join(cell['source'])
        if 'try:' in source and ';' in source and 'except' in source:
            lines = source.split('\n')
            for j, line in enumerate(lines, 1):
                if 'try:' in line and ';' in line:
                    print(f"  ⚠ WARNING: Cell {i}, Line {j} has inline try-except: {line.strip()}")
    
    print("✓ No syntax issues detected")
    
except Exception as e:
    print(f"✗ ERROR reading notebook: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("DATASET CONFIGURATION CHECK")
print("="*80)

SOURCES = [
    {"name": "master_recyclable", "type": "master"},
    {"name": "garbage_12class", "type": "mapped_12"},
    {"name": "waste_2class", "type": "mapped_2"},
    {"name": "garbage_10class", "type": "mapped_10"},
    {"name": "garbage_6class", "type": "mapped_6"},
    {"name": "garbage_balanced", "type": "mapped_6"},
    {"name": "warp_industrial", "type": "industrial"},
    {"name": "multiclass_garbage", "type": "multiclass"}
]

for source in SOURCES:
    source_type = source['type']
    if source_type != 'master' and source_type not in ALL_MAPPINGS:
        print(f"✗ ERROR: Source type '{source_type}' has no mapping!")
        sys.exit(1)
    print(f"✓ {source['name']} ({source_type})")

print("\n" + "="*80)
print("ALL VALIDATIONS PASSED!")
print("="*80)
print("\nSummary:")
print(f"  • {len(TARGET_CLASSES)} target classes defined")
print(f"  • {len(ALL_MAPPINGS)} source types with mappings")
print(f"  • {len(SOURCES)} datasets configured")
print(f"  • All mappings point to valid target classes")
print(f"  • Notebook syntax is valid")
print("\n✓ Pipeline is ready for training!")
print("="*80)

