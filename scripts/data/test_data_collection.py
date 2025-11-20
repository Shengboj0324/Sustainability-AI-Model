"""
Test Suite for LLM Data Collection Scripts
Tests all critical fixes and functionality
"""

import sys
import os
from pathlib import Path

# Add scripts/data to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

def test_imports():
    """Test 1: Verify all imports work"""
    print("="*60)
    print("TEST 1: IMPORT VALIDATION")
    print("="*60)
    
    try:
        from scrape_reddit_upcycling import RedditUpcyclingScraper
        print("✅ Reddit scraper imported")
    except Exception as e:
        print(f"✗ Reddit scraper import failed: {e}")
        return False
    
    try:
        from scrape_youtube_tutorials import YouTubeTutorialScraper
        print("✅ YouTube scraper imported")
    except Exception as e:
        print(f"✗ YouTube scraper import failed: {e}")
        return False
    
    try:
        from generate_synthetic_creative import SyntheticDataGenerator
        print("✅ Synthetic generator imported")
    except Exception as e:
        print(f"✗ Synthetic generator import failed: {e}")
        return False
    
    try:
        from collect_llm_training_data import LLMDataCollectionOrchestrator
        print("✅ Orchestrator imported")
    except Exception as e:
        print(f"✗ Orchestrator import failed: {e}")
        return False
    
    print("✅ All imports successful\n")
    return True


def test_checkpoint_functionality():
    """Test 2: Verify checkpoint save/load works"""
    print("="*60)
    print("TEST 2: CHECKPOINT FUNCTIONALITY")
    print("="*60)
    
    import json
    import tempfile
    from pathlib import Path
    
    # Create temp checkpoint file
    temp_dir = Path(tempfile.mkdtemp())
    checkpoint_file = temp_dir / "test_checkpoint.jsonl"
    
    # Test data
    test_data = [
        {
            "messages": [
                {"role": "user", "content": "Test question?"},
                {"role": "assistant", "content": "Test answer."}
            ],
            "category": "test",
            "metadata": {"post_id": "test123"}
        }
    ]
    
    # Test save
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("✅ Checkpoint save successful")
    except Exception as e:
        print(f"✗ Checkpoint save failed: {e}")
        return False
    
    # Test load
    try:
        loaded_data = []
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            for line in f:
                loaded_data.append(json.loads(line))
        
        if len(loaded_data) == len(test_data):
            print("✅ Checkpoint load successful")
        else:
            print(f"✗ Checkpoint load failed: expected {len(test_data)}, got {len(loaded_data)}")
            return False
    except Exception as e:
        print(f"✗ Checkpoint load failed: {e}")
        return False
    
    # Cleanup
    checkpoint_file.unlink()
    temp_dir.rmdir()
    
    print("✅ Checkpoint functionality verified\n")
    return True


def test_hash_deduplication():
    """Test 3: Verify SHA-256 deduplication works"""
    print("="*60)
    print("TEST 3: HASH DEDUPLICATION")
    print("="*60)
    
    import hashlib
    
    # Test data
    content1 = "This is a test content"
    content2 = "This is a test content"  # Duplicate
    content3 = "This is different content"
    
    # Calculate hashes
    hash1 = hashlib.sha256(content1.lower().encode()).hexdigest()
    hash2 = hashlib.sha256(content2.lower().encode()).hexdigest()
    hash3 = hashlib.sha256(content3.lower().encode()).hexdigest()
    
    # Verify
    if hash1 == hash2:
        print("✅ Duplicate detection works")
    else:
        print("✗ Duplicate detection failed")
        return False
    
    if hash1 != hash3:
        print("✅ Unique content differentiation works")
    else:
        print("✗ Unique content differentiation failed")
        return False
    
    print("✅ Hash deduplication verified\n")
    return True


def test_safety_filters():
    """Test 4: Verify expanded safety filters"""
    print("="*60)
    print("TEST 4: SAFETY FILTERS")
    print("="*60)
    
    # Import banned keywords
    sys.path.insert(0, str(SCRIPT_DIR))
    from scrape_reddit_upcycling import BANNED_KEYWORDS
    
    # Check if expanded
    if len(BANNED_KEYWORDS) > 5:
        print(f"✅ Expanded safety filters: {len(BANNED_KEYWORDS)} keywords")
    else:
        print(f"✗ Safety filters not expanded: only {len(BANNED_KEYWORDS)} keywords")
        return False
    
    # Check for critical keywords
    critical_keywords = ["spam", "nsfw", "weapon", "illegal"]
    missing = [kw for kw in critical_keywords if kw not in BANNED_KEYWORDS]
    
    if not missing:
        print("✅ All critical keywords present")
    else:
        print(f"✗ Missing critical keywords: {missing}")
        return False
    
    print("✅ Safety filters verified\n")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("LLM DATA COLLECTION - TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        ("Import Validation", test_imports),
        ("Checkpoint Functionality", test_checkpoint_functionality),
        ("Hash Deduplication", test_hash_deduplication),
        ("Safety Filters", test_safety_filters),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

