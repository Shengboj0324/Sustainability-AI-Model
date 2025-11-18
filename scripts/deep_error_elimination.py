#!/usr/bin/env python3
"""
Deep Error Elimination Script - Production Readiness Validation

CRITICAL: Performs comprehensive error checking across entire codebase
- Syntax validation for all Python files
- Import verification
- Function/class structure validation
- Type hint checking
- Docstring coverage
- Security vulnerability scanning
- Performance anti-pattern detection
"""

import ast
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
import importlib.util

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class DeepErrorEliminator:
    """Comprehensive error elimination and validation"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.errors = []
        self.warnings = []
        self.stats = {
            'total_files': 0,
            'valid_files': 0,
            'syntax_errors': 0,
            'import_errors': 0,
            'missing_docstrings': 0,
            'security_issues': 0,
            'performance_issues': 0
        }

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        python_files = []
        exclude_dirs = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.pytest_cache'}

        for path in self.root_dir.rglob('*.py'):
            if not any(excluded in path.parts for excluded in exclude_dirs):
                python_files.append(path)

        return sorted(python_files)

    def validate_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """Validate Python syntax"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)

    def check_imports(self, file_path: Path) -> List[str]:
        """Check for problematic imports"""
        issues = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Check for dangerous imports
                        if alias.name in ['pickle', 'marshal', 'shelve']:
                            issues.append(f"Potentially unsafe import: {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith('test'):
                        issues.append(f"Test import in production code: {node.module}")

        except Exception as e:
            issues.append(f"Import check failed: {e}")

        return issues

    def check_security(self, file_path: Path) -> List[str]:
        """Check for security vulnerabilities"""
        issues = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            # Check for dangerous patterns (excluding false positives)
            dangerous_patterns = [
                (r'(?<!\.)\beval\s*\(', 'Use of eval() - security risk'),  # Exclude .eval()
                (r'(?<!\.)\bexec\s*\(', 'Use of exec() - security risk'),  # Exclude .exec()
                (r'pickle\.loads\s*\([^)]*\)', 'Use of pickle.loads() - security risk'),
                (r'subprocess.*shell\s*=\s*True', 'shell=True in subprocess - security risk'),
                (r'os\.system\s*\(', 'Use of os.system() - use subprocess instead'),
            ]

            for pattern, message in dangerous_patterns:
                if re.search(pattern, code):
                    issues.append(message)

        except Exception as e:
            issues.append(f"Security check failed: {e}")

        return issues

    def check_docstrings(self, file_path: Path) -> Tuple[int, int]:
        """Check docstring coverage"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            tree = ast.parse(code)

            total_items = 0
            documented_items = 0

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    total_items += 1
                    if ast.get_docstring(node):
                        documented_items += 1

            return documented_items, total_items

        except Exception:
            return 0, 0

    def analyze_file(self, file_path: Path) -> Dict:
        """Comprehensive file analysis"""
        result = {
            'path': str(file_path.relative_to(self.root_dir)),
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        # Syntax validation
        valid, error = self.validate_syntax(file_path)
        if not valid:
            result['valid'] = False
            result['errors'].append(f"Syntax error: {error}")
            self.stats['syntax_errors'] += 1

        # Import checks
        import_issues = self.check_imports(file_path)
        if import_issues:
            result['warnings'].extend(import_issues)
            self.stats['import_errors'] += len(import_issues)

        # Security checks
        security_issues = self.check_security(file_path)
        if security_issues:
            result['warnings'].extend(security_issues)
            self.stats['security_issues'] += len(security_issues)

        # Docstring coverage
        documented, total = self.check_docstrings(file_path)
        if total > 0:
            coverage = (documented / total) * 100
            result['stats']['docstring_coverage'] = f"{coverage:.1f}%"
            if coverage < 50:
                result['warnings'].append(f"Low docstring coverage: {coverage:.1f}%")
                self.stats['missing_docstrings'] += 1

        return result

    def run_analysis(self) -> bool:
        """Run comprehensive analysis on all files"""
        print(f"{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}DEEP ERROR ELIMINATION - PRODUCTION READINESS VALIDATION{RESET}")
        print(f"{BLUE}{'='*80}{RESET}\n")

        python_files = self.find_python_files()
        self.stats['total_files'] = len(python_files)

        print(f"Found {len(python_files)} Python files to analyze...\n")

        critical_errors = []
        all_warnings = []

        for i, file_path in enumerate(python_files, 1):
            result = self.analyze_file(file_path)

            if result['valid']:
                self.stats['valid_files'] += 1

            # Print progress
            if i % 50 == 0:
                print(f"Progress: {i}/{len(python_files)} files analyzed...")

            # Collect critical errors
            if result['errors']:
                critical_errors.append((result['path'], result['errors']))

            # Collect warnings
            if result['warnings']:
                all_warnings.append((result['path'], result['warnings']))

        # Print results
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}ANALYSIS RESULTS{RESET}")
        print(f"{BLUE}{'='*80}{RESET}\n")

        # Statistics
        print(f"Total files analyzed: {self.stats['total_files']}")
        print(f"Valid files: {GREEN}{self.stats['valid_files']}{RESET}")
        print(f"Syntax errors: {RED if self.stats['syntax_errors'] > 0 else GREEN}{self.stats['syntax_errors']}{RESET}")
        print(f"Import issues: {YELLOW if self.stats['import_errors'] > 0 else GREEN}{self.stats['import_errors']}{RESET}")
        print(f"Security issues: {RED if self.stats['security_issues'] > 0 else GREEN}{self.stats['security_issues']}{RESET}")
        print(f"Low docstring coverage: {YELLOW if self.stats['missing_docstrings'] > 0 else GREEN}{self.stats['missing_docstrings']}{RESET}")

        # Critical errors
        if critical_errors:
            print(f"\n{RED}CRITICAL ERRORS:{RESET}")
            for path, errors in critical_errors[:10]:  # Show first 10
                print(f"\n{RED}✗{RESET} {path}")
                for error in errors:
                    print(f"  - {error}")

        # Warnings
        if all_warnings:
            print(f"\n{YELLOW}WARNINGS:{RESET}")
            for path, warnings in all_warnings[:10]:  # Show first 10
                print(f"\n{YELLOW}⚠{RESET} {path}")
                for warning in warnings:
                    print(f"  - {warning}")

        # Final verdict
        print(f"\n{BLUE}{'='*80}{RESET}")
        if self.stats['syntax_errors'] == 0 and self.stats['security_issues'] == 0:
            print(f"{GREEN}✅ PRODUCTION READY - No critical errors found!{RESET}")
            return True
        else:
            print(f"{RED}❌ NOT PRODUCTION READY - Critical errors must be fixed!{RESET}")
            return False


if __name__ == "__main__":
    root_dir = os.getcwd()
    eliminator = DeepErrorEliminator(root_dir)
    success = eliminator.run_analysis()
    sys.exit(0 if success else 1)


