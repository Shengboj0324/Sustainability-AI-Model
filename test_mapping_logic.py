#!/usr/bin/env python3
"""
Test the exact mapping logic from the notebook to ensure it works correctly.
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
    """Exact copy of the mapping logic from the notebook"""
    raw = raw_label.lower().strip()
    
    if source_type == 'master':
        if raw in target_classes:
            return raw
        return None
    
    if source_type == 'mapped_12':
        mapping = {
            'paper': 'office_paper',
            'cardboard': 'cardboard_boxes',
            'plastic': 'plastic_food_containers',
            'metal': 'aluminum_food_cans',
            'glass': 'glass_food_jars',
            'brown-glass': 'glass_beverage_bottles',
            'green-glass': 'glass_beverage_bottles',
            'white-glass': 'glass_food_jars',
            'clothes': 'clothing',
            'shoes': 'shoes',
            'biological': 'food_waste',
            'trash': 'food_waste'
        }
        return mapping.get(raw)
    
    if source_type == 'mapped_2':
        if raw == 'organic' or raw == 'o':
            return 'food_waste'
        return None
    
    if source_type == 'mapped_10':
        mapping = {
            'metal': 'aluminum_food_cans',
            'glass': 'glass_food_jars',
            'biological': 'food_waste',
            'paper': 'office_paper',
            'battery': 'aerosol_cans',
            'trash': 'food_waste',
            'cardboard': 'cardboard_boxes',
            'shoes': 'shoes',
            'clothes': 'clothing',
            'plastic': 'plastic_food_containers'
        }
        return mapping.get(raw)
    
    if source_type == 'mapped_6':
        mapping = {
            'cardboard': 'cardboard_boxes',
            'glass': 'glass_food_jars',
            'metal': 'aluminum_food_cans',
            'paper': 'office_paper',
            'plastic': 'plastic_food_containers',
            'trash': 'food_waste'
        }
        return mapping.get(raw)
    
    if source_type == 'industrial':
        mapping = {
            'pet': 'plastic_food_containers',
            'hdpe': 'plastic_food_containers',
            'pvc': 'plastic_food_containers',
            'ldpe': 'plastic_food_containers',
            'pp': 'plastic_food_containers',
            'ps': 'plastic_food_containers',
            'metal': 'aluminum_food_cans',
            'glass': 'glass_food_jars',
            'paper': 'office_paper',
            'cardboard': 'cardboard_boxes',
            'trash': 'food_waste'
        }
        return mapping.get(raw)
    
    if source_type == 'multiclass':
        mapping = {
            'plastic': 'plastic_food_containers',
            'metal': 'aluminum_food_cans',
            'glass': 'glass_food_jars',
            'paper': 'office_paper',
            'cardboard': 'cardboard_boxes',
            'trash': 'food_waste',
            'organic': 'food_waste',
            'battery': 'aerosol_cans',
            'clothes': 'clothing',
            'shoes': 'shoes'
        }
        return mapping.get(raw)
    
    return None

# Test cases
test_cases = [
    ('paper', 'mapped_12', 'office_paper'),
    ('battery', 'mapped_10', 'aerosol_cans'),
    ('battery', 'multiclass', 'aerosol_cans'),
    ('pet', 'industrial', 'plastic_food_containers'),
    ('organic', 'mapped_2', 'food_waste'),
    ('o', 'mapped_2', 'food_waste'),
    ('r', 'mapped_2', None),
    ('cardboard', 'mapped_6', 'cardboard_boxes'),
    ('food_waste', 'master', 'food_waste'),
    ('invalid', 'master', None),
]

print("="*80)
print("TESTING MAPPING LOGIC")
print("="*80)

all_passed = True
for input_label, source_type, expected in test_cases:
    result = _map_label(input_label, source_type, TARGET_CLASSES)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"{status} _map_label('{input_label}', '{source_type}') = '{result}' (expected: '{expected}')")

print("\n" + "="*80)
if all_passed:
    print("✓ ALL TESTS PASSED!")
else:
    print("✗ SOME TESTS FAILED!")
print("="*80)

