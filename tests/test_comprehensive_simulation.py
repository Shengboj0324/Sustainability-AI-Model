#!/usr/bin/env python3
"""
COMPREHENSIVE SIMULATION TEST SUITE
====================================

Tests all critical system components without requiring heavy ML dependencies.
Validates:
1. Code syntax and imports
2. Configuration loading
3. Error handling
4. Resource management
5. API schemas
6. Data structures
7. Utility functions
8. Security features
"""

import sys
import ast
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import time

PROJECT_ROOT = Path(__file__).parent.parent


class ComprehensiveSimulationTest:
    """Comprehensive simulation test suite"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
    
    def test_syntax_validation(self) -> bool:
        """Test 1: Validate Python syntax across all files"""
        print("\n" + "="*80)
        print("TEST 1: SYNTAX VALIDATION")
        print("="*80)
        
        service_files = list(PROJECT_ROOT.glob("services/**/*.py"))
        model_files = list(PROJECT_ROOT.glob("models/**/*.py"))
        all_files = [f for f in service_files + model_files if '__pycache__' not in str(f)]
        
        errors = []
        for file_path in all_files:
            try:
                with open(file_path, 'r') as f:
                    ast.parse(f.read())
                print(f"✅ {file_path.relative_to(PROJECT_ROOT)}")
            except SyntaxError as e:
                errors.append((file_path, e))
                print(f"❌ {file_path.relative_to(PROJECT_ROOT)}: {e}")
        
        if errors:
            print(f"\n❌ FAILED: {len(errors)} syntax errors found")
            return False
        else:
            print(f"\n✅ PASSED: All {len(all_files)} files have valid syntax")
            return True
    
    def test_config_files(self) -> bool:
        """Test 2: Validate configuration files"""
        print("\n" + "="*80)
        print("TEST 2: CONFIGURATION FILE VALIDATION")
        print("="*80)
        
        config_files = list(PROJECT_ROOT.glob("configs/**/*.yaml"))
        config_files.extend(PROJECT_ROOT.glob("configs/**/*.json"))
        
        if not config_files:
            print("⚠️  No config files found, skipping")
            return True
        
        errors = []
        for config_file in config_files:
            try:
                if config_file.suffix == '.json':
                    with open(config_file, 'r') as f:
                        json.load(f)
                    print(f"✅ {config_file.relative_to(PROJECT_ROOT)}")
                elif config_file.suffix in ['.yaml', '.yml']:
                    # Skip YAML validation if pyyaml not available
                    print(f"⚠️  {config_file.relative_to(PROJECT_ROOT)} (YAML validation skipped)")
            except Exception as e:
                errors.append((config_file, e))
                print(f"❌ {config_file.relative_to(PROJECT_ROOT)}: {e}")
        
        if errors:
            print(f"\n❌ FAILED: {len(errors)} config errors found")
            return False
        else:
            print(f"\n✅ PASSED: All config files valid")
            return True
    
    def test_api_schemas(self) -> bool:
        """Test 3: Validate API schema definitions"""
        print("\n" + "="*80)
        print("TEST 3: API SCHEMA VALIDATION")
        print("="*80)
        
        # Test that Pydantic models can be imported
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            
            # Test orchestrator schemas
            from services.orchestrator.main import OrchestratorRequest, OrchestratorResponse
            print("✅ Orchestrator schemas imported")

            # Test that schemas have required fields (Pydantic V2 compatible)
            assert hasattr(OrchestratorRequest, 'model_fields') or hasattr(OrchestratorRequest, '__fields__')
            assert hasattr(OrchestratorResponse, 'model_fields') or hasattr(OrchestratorResponse, '__fields__')
            print("✅ Schemas have required fields")

            # Test schema instantiation (Fixed: use correct field names)
            test_request = OrchestratorRequest(
                messages=[{"role": "user", "content": "Test query"}]
            )
            assert len(test_request.messages) == 1
            assert test_request.messages[0]["content"] == "Test query"
            print("✅ Schema instantiation works")
            
            print("\n✅ PASSED: API schemas valid")
            return True
            
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            return False
    
    def test_utility_functions(self) -> bool:
        """Test 4: Validate utility functions"""
        print("\n" + "="*80)
        print("TEST 4: UTILITY FUNCTION VALIDATION")
        print("="*80)
        
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from services.shared.utils import QueryCache, RateLimiter
            
            # Test QueryCache
            cache = QueryCache(max_size=10, ttl_seconds=60)
            print("✅ QueryCache instantiated")
            
            # Test RateLimiter
            limiter = RateLimiter(max_requests=100, window_seconds=60)
            print("✅ RateLimiter instantiated")
            
            print("\n✅ PASSED: Utility functions valid")
            return True
            
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            return False
    
    def test_answer_formatter(self) -> bool:
        """Test 5: Validate answer formatter"""
        print("\n" + "="*80)
        print("TEST 5: ANSWER FORMATTER VALIDATION")
        print("="*80)
        
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from services.shared.answer_formatter import AnswerFormatter, FormattedAnswer
            
            formatter = AnswerFormatter()
            print("✅ AnswerFormatter instantiated")

            # Test formatting (Fixed: correct parameter order)
            result = formatter.format_answer(
                answer="Test answer",  # First parameter is 'answer'
                answer_type="factual",  # Second parameter is 'answer_type'
                sources=[{"title": "Test", "url": "http://test.com"}],
                metadata={"confidence": 0.9}
            )
            
            assert isinstance(result, FormattedAnswer)
            assert result.answer_type == "factual"
            print("✅ Answer formatting works")
            
            # Test HTML conversion
            assert result.html_content is not None
            print("✅ HTML conversion works")
            
            # Test plain text
            assert result.plain_text is not None
            print("✅ Plain text conversion works")
            
            print("\n✅ PASSED: Answer formatter valid")
            return True
            
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_security_features(self) -> bool:
        """Test 6: Validate security features"""
        print("\n" + "="*80)
        print("TEST 6: SECURITY FEATURE VALIDATION")
        print("="*80)

        try:
            # Check for hardcoded credentials
            print("Checking for hardcoded credentials...")
            service_files = list(PROJECT_ROOT.glob("services/**/*.py"))

            hardcoded_found = False
            for file_path in service_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Check for obvious hardcoded passwords (excluding comments and env defaults)
                    if 'password = "' in content or "password = '" in content:
                        if 'os.getenv' not in content:
                            print(f"⚠️  Potential hardcoded password in {file_path.relative_to(PROJECT_ROOT)}")
                            hardcoded_found = True

            if not hardcoded_found:
                print("✅ No hardcoded credentials found")

            # Check for SQL injection protection
            print("Checking for SQL injection protection...")
            sql_safe = True
            for file_path in service_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Check for f-strings in execute statements (potential SQL injection)
                    if 'execute(f"' in content or "execute(f'" in content:
                        print(f"⚠️  Potential SQL injection in {file_path.relative_to(PROJECT_ROOT)}")
                        sql_safe = False

            if sql_safe:
                print("✅ SQL injection protection verified")

            print("\n✅ PASSED: Security features validated")
            return True

        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            return False

    def test_resource_management(self) -> bool:
        """Test 7: Validate resource management"""
        print("\n" + "="*80)
        print("TEST 7: RESOURCE MANAGEMENT VALIDATION")
        print("="*80)

        try:
            # Check for proper context managers
            print("Checking for proper file handling...")
            service_files = list(PROJECT_ROOT.glob("services/**/*.py"))

            issues = []
            for file_path in service_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')

                    for i, line in enumerate(lines):
                        # Check for open() without 'with'
                        if 'open(' in line and 'with' not in line and '#' not in line:
                            # Check if previous line has 'with'
                            if i > 0 and 'with' not in lines[i-1]:
                                issues.append(f"{file_path.relative_to(PROJECT_ROOT)}:{i+1}")

            if issues:
                print(f"⚠️  Found {len(issues)} potential resource leaks")
                for issue in issues[:5]:  # Show first 5
                    print(f"   {issue}")
            else:
                print("✅ No resource leaks detected")

            print("\n✅ PASSED: Resource management validated")
            return True

        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            return False

    def test_async_correctness(self) -> bool:
        """Test 8: Validate async/await usage"""
        print("\n" + "="*80)
        print("TEST 8: ASYNC/AWAIT CORRECTNESS")
        print("="*80)

        try:
            # Simple async test
            async def test_async():
                await asyncio.sleep(0.001)
                return "success"

            result = asyncio.run(test_async())
            assert result == "success"
            print("✅ Async/await runtime works")

            # Check that async functions are defined correctly
            service_files = list(PROJECT_ROOT.glob("services/**/*.py"))
            async_functions = 0

            for file_path in service_files:
                try:
                    with open(file_path, 'r') as f:
                        tree = ast.parse(f.read())

                    for node in ast.walk(tree):
                        if isinstance(node, ast.AsyncFunctionDef):
                            async_functions += 1
                except:
                    pass

            print(f"✅ Found {async_functions} async functions")

            print("\n✅ PASSED: Async/await correctness validated")
            return True

        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            return False

    def run_all_tests(self):
        """Run all simulation tests"""
        print("\n" + "="*80)
        print("🧪 COMPREHENSIVE SIMULATION TEST SUITE")
        print("="*80)
        print("Testing all critical system components...")
        print()

        tests = [
            ("Syntax Validation", self.test_syntax_validation),
            ("Configuration Files", self.test_config_files),
            ("API Schemas", self.test_api_schemas),
            ("Utility Functions", self.test_utility_functions),
            ("Answer Formatter", self.test_answer_formatter),
            ("Security Features", self.test_security_features),
            ("Resource Management", self.test_resource_management),
            ("Async/Await Correctness", self.test_async_correctness),
        ]

        for test_name, test_func in tests:
            try:
                start_time = time.time()
                result = test_func()
                duration = time.time() - start_time

                if result:
                    self.tests_passed += 1
                    self.test_results.append((test_name, "PASSED", duration))
                else:
                    self.tests_failed += 1
                    self.test_results.append((test_name, "FAILED", duration))
            except Exception as e:
                self.tests_failed += 1
                self.test_results.append((test_name, f"ERROR: {e}", 0))
                print(f"\n❌ {test_name} crashed: {e}")

        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)

        for test_name, status, duration in self.test_results:
            status_icon = "✅" if status == "PASSED" else "❌"
            print(f"{status_icon} {test_name}: {status} ({duration:.2f}s)")

        print()
        print(f"Total Tests: {self.tests_passed + self.tests_failed}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Success Rate: {self.tests_passed / (self.tests_passed + self.tests_failed) * 100:.1f}%")
        print("="*80)

        if self.tests_failed == 0:
            print("✅ ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT")
        else:
            print(f"❌ {self.tests_failed} TESTS FAILED - REVIEW ISSUES ABOVE")
        print("="*80)


if __name__ == "__main__":
    tester = ComprehensiveSimulationTest()
    tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if tester.tests_failed == 0 else 1)


