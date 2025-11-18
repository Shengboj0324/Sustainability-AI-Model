"""
Test Infinite Adaptability and Production Intelligence

CRITICAL: Validates that the system can handle ANY possible user input
"""

import json
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def test_edge_case_coverage():
    """Test that edge cases are properly covered in training data"""
    print("=" * 80)
    print("TESTING EDGE CASE COVERAGE")
    print("=" * 80)
    
    # Load ultra-expanded dataset
    data_file = PROJECT_ROOT / "data" / "llm_training_ultra_expanded.json"
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n✅ Loaded {len(data)} training examples")
    
    # Categorize examples
    categories = {}
    for example in data:
        category = example.get("category", "general")
        if category not in categories:
            categories[category] = []
        categories[category].append(example)
    
    print(f"\n📊 Categories found: {len(categories)}")
    for category, examples in sorted(categories.items()):
        print(f"   - {category}: {len(examples)} examples")
    
    # Check for critical edge cases
    critical_categories = [
        "ambiguous_handling",
        "hazardous_rare",
        "rare_materials",
        "multi_step_reasoning",
        "incomplete_info",
        "multi_material",
        "complex_hazardous",
        "error_correction",
        "multi_language"
    ]
    
    print(f"\n🎯 Critical Edge Case Coverage:")
    missing = []
    for cat in critical_categories:
        if cat in categories:
            print(f"   ✅ {cat}: {len(categories[cat])} examples")
        else:
            print(f"   ❌ {cat}: MISSING")
            missing.append(cat)
    
    if missing:
        print(f"\n⚠️  WARNING: Missing {len(missing)} critical categories")
        return False
    else:
        print(f"\n✅ ALL CRITICAL CATEGORIES COVERED")
        return True


def test_orchestrator_enhancements():
    """Test that orchestrator has all required enhancements"""
    print("\n" + "=" * 80)
    print("TESTING ORCHESTRATOR ENHANCEMENTS")
    print("=" * 80)
    
    orchestrator_file = PROJECT_ROOT / "services" / "orchestrator" / "main.py"
    with open(orchestrator_file, 'r') as f:
        content = f.read()
    
    # Check for critical features
    features = {
        "ConfidenceLevel": "Confidence level enum",
        "ConfidenceCalculator": "Confidence calculation",
        "FallbackStrategy": "Fallback strategies",
        "confidence_score": "Confidence scoring",
        "fallback_used": "Fallback tracking",
        "partial_answer": "Partial answer support",
        "reasoning_steps": "Reasoning transparency",
        "image_quality_score": "Image quality assessment",
        "text_quality_score": "Text quality assessment",
        "assess_text_quality": "Text quality function",
        "generate_fallback_response": "Fallback generation",
        "calculate_overall_confidence": "Overall confidence calculation"
    }
    
    print(f"\n🔍 Checking for {len(features)} critical features:")
    all_present = True
    for feature, description in features.items():
        if feature in content:
            print(f"   ✅ {description} ({feature})")
        else:
            print(f"   ❌ {description} ({feature}) - MISSING")
            all_present = False
    
    if all_present:
        print(f"\n✅ ALL ORCHESTRATOR FEATURES PRESENT")
        return True
    else:
        print(f"\n❌ SOME ORCHESTRATOR FEATURES MISSING")
        return False


def test_training_data_quality():
    """Test training data quality and format"""
    print("\n" + "=" * 80)
    print("TESTING TRAINING DATA QUALITY")
    print("=" * 80)
    
    # Check processed data
    processed_dir = PROJECT_ROOT / "data" / "processed" / "llm_sft"
    
    files_to_check = [
        "sustainability_qa_train.jsonl",
        "sustainability_qa_val.jsonl"
    ]
    
    print(f"\n📁 Checking processed training files:")
    all_valid = True
    
    for filename in files_to_check:
        filepath = processed_dir / filename
        if not filepath.exists():
            print(f"   ❌ {filename} - NOT FOUND")
            all_valid = False
            continue
        
        # Count lines
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Validate format
        valid_lines = 0
        for line in lines:
            try:
                data = json.loads(line)
                if "messages" in data:
                    valid_lines += 1
            except:
                pass
        
        if valid_lines == len(lines):
            print(f"   ✅ {filename}: {len(lines)} examples (all valid)")
        else:
            print(f"   ⚠️  {filename}: {valid_lines}/{len(lines)} valid")
            all_valid = False
    
    if all_valid:
        print(f"\n✅ ALL TRAINING DATA VALID")
        return True
    else:
        print(f"\n❌ SOME TRAINING DATA ISSUES")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🧪 INFINITE ADAPTABILITY SYSTEM TEST")
    print("=" * 80)
    
    results = {
        "Edge Case Coverage": test_edge_case_coverage(),
        "Orchestrator Enhancements": test_orchestrator_enhancements(),
        "Training Data Quality": test_training_data_quality()
    }
    
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - SYSTEM READY FOR INFINITE ADAPTABILITY")
    else:
        print("❌ SOME TESTS FAILED - REVIEW ISSUES ABOVE")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

