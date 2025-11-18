#!/usr/bin/env python3
"""
Comprehensive System Validation - ReleAF AI

CRITICAL: Final validation before production deployment
- Validates all modules can be imported
- Tests all critical functions
- Validates data integrity
- Checks model compatibility
- Performance benchmarks
"""

import sys
import time
import json
from pathlib import Path
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class ComprehensiveValidator:
    """Comprehensive system validation"""

    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.warnings = 0

    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}{text}{RESET}")
        print(f"{BLUE}{'='*80}{RESET}\n")

    def test_nlp_modules(self) -> bool:
        """Test NLP modules"""
        print("Testing NLP modules...")

        try:
            from services.llm_service.intent_classifier import IntentClassifier
            from services.llm_service.entity_extractor import EntityExtractor
            from services.llm_service.language_handler import LanguageHandler

            # Test intent classification
            classifier = IntentClassifier()
            intent, confidence = classifier.classify("How do I recycle plastic bottles?")
            print(f"{GREEN}✅{RESET} Intent Classification: {intent.value} ({confidence:.2f})")

            # Test entity extraction
            extractor = EntityExtractor()
            entities = extractor.extract("I have plastic bottles and aluminum cans")
            print(f"{GREEN}✅{RESET} Entity Extraction: Found {len(entities)} entities")

            # Test language detection
            handler = LanguageHandler()
            lang, conf = handler.detect_language("¿Cómo reciclo botellas de plástico?")
            print(f"{GREEN}✅{RESET} Language Detection: {lang.value} ({conf:.2f})")

            return True

        except Exception as e:
            print(f"{RED}❌ NLP modules failed: {e}{RESET}")
            return False

    def test_vision_modules(self) -> bool:
        """Test vision modules"""
        print("\nTesting vision modules...")

        try:
            from models.vision.image_quality import AdvancedImageQualityPipeline
            import numpy as np
            from PIL import Image

            # Create test image
            test_img = Image.new('RGB', (224, 224), color='red')

            # Test image quality enhancer
            pipeline = AdvancedImageQualityPipeline()
            enhanced, report = pipeline.process(test_img)

            print(f"{GREEN}✅{RESET} Image Quality Enhancement: {enhanced.size}, Quality: {report.quality_score:.2f}")

            return True

        except Exception as e:
            print(f"{RED}❌ Vision modules failed: {e}{RESET}")
            return False

    def test_data_integrity(self) -> bool:
        """Test data file integrity"""
        print("\nTesting data integrity...")

        data_files = [
            "data/llm_training_expanded.json",
            "data/rag_knowledge_base_expanded.json",
            "data/gnn_training_expanded.json",
            "data/organizations_database.json",
            "data/sustainability_knowledge_base.json"
        ]

        all_valid = True
        for data_file in data_files:
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    count = len(data)
                elif isinstance(data, dict):
                    count = len(data.keys())
                else:
                    count = 1

                print(f"{GREEN}✅{RESET} {data_file}: {count} items")

            except FileNotFoundError:
                print(f"{YELLOW}⚠{RESET} {data_file}: Not found")
                all_valid = False
            except json.JSONDecodeError as e:
                print(f"{RED}❌{RESET} {data_file}: Invalid JSON - {e}")
                all_valid = False
            except Exception as e:
                print(f"{RED}❌{RESET} {data_file}: Error - {e}")
                all_valid = False

        return all_valid

    def test_model_imports(self) -> bool:
        """Test all model imports"""
        print("\nTesting model imports...")

        models_to_test = [
            ("LLM Service", "services.llm_service.server_v2", "LLMServiceV2"),
            ("Vision Classifier", "models.vision.classifier", "WasteClassifier"),
            ("Vision Detector", "models.vision.detector", "WasteDetector"),
            ("GNN Inference", "models.gnn.inference", "GNNInference"),
        ]

        all_imported = True
        for name, module_path, class_name in models_to_test:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                print(f"{GREEN}✅{RESET} {name}: {class_name} imported")
            except ImportError as e:
                print(f"{YELLOW}⚠{RESET} {name}: Import failed - {e}")
                all_imported = False
            except Exception as e:
                print(f"{YELLOW}⚠{RESET} {name}: Error - {e}")
                all_imported = False

        return all_imported

    def performance_benchmark(self) -> bool:
        """Run performance benchmarks"""
        print("\nRunning performance benchmarks...")

        try:
            from services.llm_service.intent_classifier import IntentClassifier
            from services.llm_service.entity_extractor import EntityExtractor
            from services.llm_service.language_handler import LanguageHandler

            # Benchmark intent classification
            classifier = IntentClassifier()
            test_queries = [
                "How do I recycle plastic?",
                "Where can I donate old clothes?",
                "What can I make from bottles?",
                "Is this biodegradable?",
                "Find recycling centers near me"
            ]

            start = time.time()
            for query in test_queries * 10:  # 50 queries
                classifier.classify(query)
            duration = time.time() - start
            avg_time = (duration / 50) * 1000  # ms

            print(f"{GREEN}✅{RESET} Intent Classification: {avg_time:.2f}ms avg (50 queries in {duration:.2f}s)")

            # Benchmark entity extraction
            extractor = EntityExtractor()
            start = time.time()
            for query in test_queries * 10:
                extractor.extract(query)
            duration = time.time() - start
            avg_time = (duration / 50) * 1000

            print(f"{GREEN}✅{RESET} Entity Extraction: {avg_time:.2f}ms avg (50 queries in {duration:.2f}s)")

            # Benchmark language detection
            handler = LanguageHandler()
            multilang_queries = [
                "How do I recycle?",
                "¿Cómo reciclo?",
                "Comment recycler?",
                "Wie recycelt man?",
                "リサイクルの方法は？"
            ]

            start = time.time()
            for query in multilang_queries * 10:
                handler.detect_language(query)
            duration = time.time() - start
            avg_time = (duration / 50) * 1000

            print(f"{GREEN}✅{RESET} Language Detection: {avg_time:.2f}ms avg (50 queries in {duration:.2f}s)")

            return True

        except Exception as e:
            print(f"{RED}❌ Performance benchmark failed: {e}{RESET}")
            return False

    def run_validation(self) -> bool:
        """Run all validation tests"""
        self.print_header("COMPREHENSIVE SYSTEM VALIDATION")

        tests = [
            ("NLP Modules", self.test_nlp_modules),
            ("Vision Modules", self.test_vision_modules),
            ("Data Integrity", self.test_data_integrity),
            ("Model Imports", self.test_model_imports),
            ("Performance Benchmarks", self.performance_benchmark),
        ]

        for test_name, test_func in tests:
            try:
                if test_func():
                    self.tests_passed += 1
                else:
                    self.tests_failed += 1
                    self.warnings += 1
            except Exception as e:
                print(f"{RED}❌ {test_name} crashed: {e}{RESET}")
                self.tests_failed += 1

        # Final summary
        self.print_header("VALIDATION SUMMARY")
        print(f"Tests passed: {GREEN}{self.tests_passed}{RESET}")
        print(f"Tests failed: {RED if self.tests_failed > 0 else GREEN}{self.tests_failed}{RESET}")
        print(f"Warnings: {YELLOW if self.warnings > 0 else GREEN}{self.warnings}{RESET}")
        print()

        if self.tests_failed == 0:
            print(f"{GREEN}{'='*80}{RESET}")
            print(f"{GREEN}✅ ALL VALIDATION TESTS PASSED!{RESET}")
            print(f"{GREEN}{'='*80}{RESET}")
            print()
            print("System is ready for production deployment!")
            print()
            print("Next steps:")
            print("1. Review production configuration: configs/production.json")
            print("2. Start services: ./scripts/start_services.sh")
            print("3. Run integration tests")
            print("4. Deploy to Digital Ocean")
            print()
            return True
        else:
            print(f"{YELLOW}{'='*80}{RESET}")
            print(f"{YELLOW}⚠ VALIDATION COMPLETED WITH WARNINGS{RESET}")
            print(f"{YELLOW}{'='*80}{RESET}")
            print()
            print("Some tests failed. Review errors above.")
            print("System may still be functional for development.")
            print()
            return False


if __name__ == "__main__":
    validator = ComprehensiveValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)

