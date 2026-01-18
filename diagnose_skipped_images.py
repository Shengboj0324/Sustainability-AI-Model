#!/usr/bin/env python3
"""
Diagnose exactly which labels are being skipped and why.
This will tell us what mappings need to be added.
"""

TARGET_CLASSES = [
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes', 'cardboard_packaging',
    'clothing', 'coffee_grounds', 'disposable_plastic_cutlery', 'egg_shells', 'food_waste',
    'glass_beverage_bottles', 'glass_cosmetic_containers', 'glass_food_jars', 'magazines',
    'newspaper', 'office_paper', 'paper_cups', 'plastic_cup_lids', 'plastic_detergent_bottles',
    'plastic_food_containers', 'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws',
    'plastic_trash_bags', 'plastic_water_bottles', 'shoes', 'steel_food_cans', 'styrofoam_cups',
    'styrofoam_food_containers', 'tea_bags'
]

def _map_label(raw_label, source_type, target_classes):
    raw = raw_label.lower().strip()
    
    if source_type == 'master':
        return raw if raw in target_classes else None
    
    if source_type == 'mapped_12':
        mapping = {
            'paper': 'office_paper', 'cardboard': 'cardboard_boxes', 'plastic': 'plastic_food_containers',
            'metal': 'aluminum_food_cans', 'glass': 'glass_food_jars', 'brown-glass': 'glass_beverage_bottles',
            'green-glass': 'glass_beverage_bottles', 'white-glass': 'glass_food_jars', 'clothes': 'clothing',
            'shoes': 'shoes', 'biological': 'food_waste', 'trash': 'food_waste'
        }
        return mapping.get(raw)
    
    if source_type == 'mapped_2':
        if raw == 'organic' or raw == 'o':
            return 'food_waste'
        return None  # ← THIS IS THE PROBLEM!
    
    if source_type == 'mapped_10':
        mapping = {
            'metal': 'aluminum_food_cans', 'glass': 'glass_food_jars', 'biological': 'food_waste',
            'paper': 'office_paper', 'battery': 'aerosol_cans', 'trash': 'food_waste',
            'cardboard': 'cardboard_boxes', 'shoes': 'shoes', 'clothes': 'clothing',
            'plastic': 'plastic_food_containers'
        }
        return mapping.get(raw)
    
    if source_type == 'mapped_6':
        mapping = {
            'cardboard': 'cardboard_boxes', 'glass': 'glass_food_jars', 'metal': 'aluminum_food_cans',
            'paper': 'office_paper', 'plastic': 'plastic_food_containers', 'trash': 'food_waste'
        }
        return mapping.get(raw)
    
    if source_type == 'industrial':
        mapping = {
            'pet': 'plastic_food_containers', 'hdpe': 'plastic_food_containers', 'pvc': 'plastic_food_containers',
            'ldpe': 'plastic_food_containers', 'pp': 'plastic_food_containers', 'ps': 'plastic_food_containers',
            'metal': 'aluminum_food_cans', 'glass': 'glass_food_jars', 'paper': 'office_paper',
            'cardboard': 'cardboard_boxes', 'trash': 'food_waste'
        }
        return mapping.get(raw)
    
    if source_type == 'multiclass':
        mapping = {
            'plastic': 'plastic_food_containers', 'metal': 'aluminum_food_cans', 'glass': 'glass_food_jars',
            'paper': 'office_paper', 'cardboard': 'cardboard_boxes', 'trash': 'food_waste',
            'organic': 'food_waste', 'battery': 'aerosol_cans', 'clothes': 'clothing', 'shoes': 'shoes'
        }
        return mapping.get(raw)
    
    return None

# Simulate what labels would be skipped
print("="*80)
print("DIAGNOSING SKIPPED IMAGES")
print("="*80)
print()

# Test mapped_2 (waste_22k dataset)
print("Testing mapped_2 (waste_22k - 22,000 images):")
print("  This dataset has 2 classes: 'O' (organic) and 'R' (recyclable)")
print()

test_labels_2 = ['O', 'o', 'organic', 'Organic', 'R', 'r', 'recyclable', 'Recyclable']
for label in test_labels_2:
    result = _map_label(label, 'mapped_2', TARGET_CLASSES)
    if result:
        print(f"  ✓ '{label}' → '{result}'")
    else:
        print(f"  ✗ '{label}' → SKIPPED (no mapping)")

print()
print("🚨 PROBLEM IDENTIFIED:")
print("  The 'R' (recyclable) class is being SKIPPED!")
print("  This is ~11,000 images being wasted!")
print()

# Calculate impact
print("="*80)
print("ESTIMATED IMPACT")
print("="*80)
print()
print("waste_22k dataset breakdown:")
print("  - Total images: ~22,000")
print("  - 'O' (organic): ~11,000 → MAPPED to 'food_waste' ✓")
print("  - 'R' (recyclable): ~11,000 → SKIPPED ✗")
print()
print("Other potential issues:")
print()

# Check if there are other unmapped categories
all_datasets = {
    'master': ['Should match TARGET_CLASSES exactly'],
    'mapped_12': ['paper', 'cardboard', 'plastic', 'metal', 'glass', 'brown-glass', 'green-glass', 'white-glass', 'clothes', 'shoes', 'biological', 'trash'],
    'mapped_2': ['o', 'organic', 'r', 'recyclable'],
    'mapped_10': ['metal', 'glass', 'biological', 'paper', 'battery', 'trash', 'cardboard', 'shoes', 'clothes', 'plastic'],
    'mapped_6': ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
    'industrial': ['pet', 'hdpe', 'pvc', 'ldpe', 'pp', 'ps', 'metal', 'glass', 'paper', 'cardboard', 'trash'],
    'multiclass': ['plastic', 'metal', 'glass', 'paper', 'cardboard', 'trash', 'organic', 'battery', 'clothes', 'shoes']
}

print("Checking all dataset mappings:")
for source_type, labels in all_datasets.items():
    if source_type == 'master':
        continue
    unmapped = []
    for label in labels:
        result = _map_label(label, source_type, TARGET_CLASSES)
        if not result:
            unmapped.append(label)
    
    if unmapped:
        print(f"  ✗ {source_type}: Missing mappings for {unmapped}")
    else:
        print(f"  ✓ {source_type}: All labels mapped")

print()
print("="*80)
print("SOLUTION")
print("="*80)
print()
print("Add mapping for 'recyclable' / 'r' in mapped_2:")
print()
print("  if source_type == 'mapped_2':")
print("      if raw in ['organic', 'o']:")
print("          return 'food_waste'")
print("      if raw in ['recyclable', 'r']:")
print("          return 'plastic_food_containers'  # or appropriate recyclable class")
print("      return None")
print()
print("This will recover ~11,000 images!")

