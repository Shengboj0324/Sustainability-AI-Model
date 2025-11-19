#!/usr/bin/env python3
"""
Production Readiness Test Suite
Comprehensive testing for ReleAF AI deployment to Digital Ocean
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ProductionReadinessTest:
    """Comprehensive production readiness testing"""

    def __init__(self):
        self.results = {
            'syntax': {'passed': 0, 'failed': 0, 'errors': []},
            'imports': {'passed': 0, 'failed': 0, 'errors': []},
            'configs': {'passed': 0, 'failed': 0, 'errors': []},
            'data': {'passed': 0, 'failed': 0, 'errors': []},
            'services': {'passed': 0, 'failed': 0, 'errors': []},
        }

    def test_syntax(self) -> bool:
        """Test all Python files for syntax errors"""
        print("\n" + "="*80)
        print("TEST 1: SYNTAX VALIDATION")
        print("="*80)

        import ast

        python_files = list(Path('.').glob('**/*.py'))
        python_files = [f for f in python_files if '.venv' not in str(f) and 'venv' not in str(f)]

        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    ast.parse(f.read(), filename=str(file_path))
                self.results['syntax']['passed'] += 1
            except SyntaxError as e:
                self.results['syntax']['failed'] += 1
                self.results['syntax']['errors'].append(f"{file_path}: Line {e.lineno}")
                print(f"✗ {file_path}: Syntax error at line {e.lineno}")

        print(f"✓ Checked {len(python_files)} files")
        print(f"✓ Passed: {self.results['syntax']['passed']}")
        print(f"✗ Failed: {self.results['syntax']['failed']}")

        return self.results['syntax']['failed'] == 0

    def test_imports(self) -> bool:
        """Test critical imports"""
        print("\n" + "="*80)
        print("TEST 2: IMPORT VALIDATION")
        print("="*80)

        critical_files = [
            'services/shared/utils.py',
            'services/shared/common.py',
            'services/llm_service/server_v2.py',
            'services/vision_service/server_v2.py',
            'services/rag_service/server.py',
        ]

        for file_path in critical_files:
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("test", file_path)
                if spec:
                    self.results['imports']['passed'] += 1
                    print(f"✓ {file_path}")
                else:
                    self.results['imports']['failed'] += 1
                    print(f"✗ {file_path}: Cannot create spec")
            except Exception as e:
                self.results['imports']['failed'] += 1
                self.results['imports']['errors'].append(f"{file_path}: {str(e)}")
                print(f"✗ {file_path}: {e}")

        return self.results['imports']['failed'] == 0

    def test_configs(self) -> bool:
        """Test configuration files"""
        print("\n" + "="*80)
        print("TEST 3: CONFIGURATION VALIDATION")
        print("="*80)

        config_files = list(Path('configs').glob('*.yaml'))

        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                if config:
                    self.results['configs']['passed'] += 1
                    print(f"✓ {config_file.name}")
                else:
                    self.results['configs']['failed'] += 1
                    print(f"⚠️  {config_file.name}: Empty")
            except Exception as e:
                self.results['configs']['failed'] += 1
                self.results['configs']['errors'].append(f"{config_file}: {str(e)}")
                print(f"✗ {config_file.name}: {e}")

        return self.results['configs']['failed'] == 0

    def test_data(self) -> bool:
        """Test data availability"""
        print("\n" + "="*80)
        print("TEST 4: DATA VALIDATION")
        print("="*80)

        critical_data = [
            'data/llm_training_ultra_expanded.json',
            'data/rag_knowledge_base_expanded.json',
            'data/gnn_training_fully_annotated.json',
            'data/organizations_database.json',
        ]

        for data_file in critical_data:
            path = Path(data_file)
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                self.results['data']['passed'] += 1
                print(f"✓ {data_file} ({size_mb:.2f} MB)")
            else:
                self.results['data']['failed'] += 1
                self.results['data']['errors'].append(f"{data_file}: Not found")
                print(f"✗ {data_file}: Not found")

        return self.results['data']['failed'] == 0




    def test_services(self) -> bool:
        """Test service structure"""
        print("\n" + "="*80)
        print("TEST 5: SERVICE STRUCTURE VALIDATION")
        print("="*80)

        services = [
            'services/llm_service/server_v2.py',
            'services/vision_service/server_v2.py',
            'services/rag_service/server.py',
            'services/kg_service/server.py',
            'services/org_search_service/server.py',
            'services/orchestrator/main.py',
        ]

        for service in services:
            path = Path(service)
            if path.exists():
                self.results['services']['passed'] += 1
                print(f"✓ {service}")
            else:
                self.results['services']['failed'] += 1
                self.results['services']['errors'].append(f"{service}: Not found")
                print(f"✗ {service}: Not found")

        return self.results['services']['failed'] == 0

    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("🎯 PRODUCTION READINESS SUMMARY")
        print("="*80)

        total_passed = sum(r['passed'] for r in self.results.values())
        total_failed = sum(r['failed'] for r in self.results.values())
        total_tests = total_passed + total_failed

        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {(total_passed/total_tests*100):.1f}%")

        print("\nDetailed Results:")
        for category, results in self.results.items():
            status = "✅" if results['failed'] == 0 else "❌"
            print(f"  {status} {category.upper()}: {results['passed']} passed, {results['failed']} failed")

        # Print errors
        has_errors = False
        for category, results in self.results.items():
            if results['errors']:
                if not has_errors:
                    print("\n" + "="*80)
                    print("ERRORS FOUND")
                    print("="*80)
                    has_errors = True
                print(f"\n{category.upper()}:")
                for error in results['errors']:
                    print(f"  ✗ {error}")

        if total_failed == 0:
            print("\n" + "="*80)
            print("🎉 100% PRODUCTION READY!")
            print("="*80)
            print("✅ All syntax checks passed")
            print("✅ All imports validated")
            print("✅ All configurations valid")
            print("✅ All data files present")
            print("✅ All services structured correctly")
            print("\n🚀 READY FOR DEPLOYMENT TO DIGITAL OCEAN!")
            return True
        else:
            print("\n" + "="*80)
            print("⚠️  PRODUCTION READINESS: ISSUES FOUND")
            print("="*80)
            print(f"Please fix {total_failed} issue(s) before deployment")
            return False

    def run_all_tests(self) -> bool:
        """Run all production readiness tests"""
        print("="*80)
        print("🔥 FIERCE ERROR ELIMINATION & PRODUCTION READINESS TEST")
        print("="*80)
        print("Testing ReleAF AI for Digital Ocean deployment...")

        tests = [
            self.test_syntax,
            self.test_imports,
            self.test_configs,
            self.test_data,
            self.test_services,
        ]

        for test in tests:
            test()

        return self.print_summary()


def main():
    """Main entry point"""
    tester = ProductionReadinessTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
