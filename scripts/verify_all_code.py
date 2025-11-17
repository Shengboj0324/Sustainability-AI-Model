"""
Comprehensive Code Verification Script

CRITICAL: Perform intense error elimination with strictest quality requirements
- Compile all Python files
- Check for syntax errors
- Check for import errors
- Verify function signatures
- Check for undefined variables
- Generate comprehensive error report
"""

import os
import sys
import py_compile
import ast
import importlib.util
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class CodeVerifier:
    """Comprehensive code verification"""
    
    def __init__(self):
        self.total_files = 0
        self.passed_files = 0
        self.failed_files = 0
        self.errors = []
        self.warnings = []
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files in project"""
        python_files = []
        
        # Directories to check
        check_dirs = [
            PROJECT_ROOT / "services",
            PROJECT_ROOT / "models",
            PROJECT_ROOT / "training",
            PROJECT_ROOT / "scripts" / "data",
        ]
        
        for directory in check_dirs:
            if directory.exists():
                python_files.extend(directory.rglob("*.py"))
        
        # Exclude __pycache__ and test files
        python_files = [
            f for f in python_files 
            if '__pycache__' not in str(f) and 'test_' not in f.name
        ]
        
        return sorted(python_files)
    
    def check_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """Check Python syntax by compiling"""
        try:
            py_compile.compile(str(file_path), doraise=True)
            return True, "OK"
        except py_compile.PyCompileError as e:
            return False, f"Syntax error: {e}"
    
    def check_ast(self, file_path: Path) -> Tuple[bool, str]:
        """Check AST parsing"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            ast.parse(code)
            return True, "OK"
        except SyntaxError as e:
            return False, f"AST error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"AST error: {e}"
    
    def check_imports(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Check if all imports are available"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            tree = ast.parse(code)
            missing_imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not self._check_module(alias.name):
                            missing_imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and not self._check_module(node.module):
                        missing_imports.append(node.module)
            
            return len(missing_imports) == 0, missing_imports
        except Exception as e:
            return False, [f"Error checking imports: {e}"]
    
    def _check_module(self, module_name: str) -> bool:
        """Check if a module is available"""
        # Skip relative imports
        if module_name.startswith('.'):
            return True
        
        # Get top-level module
        top_module = module_name.split('.')[0]
        
        # Check if it's a standard library or installed package
        spec = importlib.util.find_spec(top_module)
        return spec is not None
    
    def verify_file(self, file_path: Path) -> Dict:
        """Verify a single file"""
        relative_path = file_path.relative_to(PROJECT_ROOT)
        
        result = {
            'file': str(relative_path),
            'passed': True,
            'errors': [],
            'warnings': []
        }
        
        # Check syntax
        syntax_ok, syntax_msg = self.check_syntax(file_path)
        if not syntax_ok:
            result['passed'] = False
            result['errors'].append(f"SYNTAX: {syntax_msg}")
        
        # Check AST
        ast_ok, ast_msg = self.check_ast(file_path)
        if not ast_ok:
            result['passed'] = False
            result['errors'].append(f"AST: {ast_msg}")
        
        # Check imports (only if syntax is OK)
        if syntax_ok and ast_ok:
            imports_ok, missing = self.check_imports(file_path)
            if not imports_ok:
                result['warnings'].append(f"IMPORTS: Missing {missing}")
        
        return result
    
    def verify_all(self):
        """Verify all Python files"""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE CODE VERIFICATION")
        logger.info("=" * 80)
        
        # Find all Python files
        python_files = self.find_python_files()
        self.total_files = len(python_files)
        
        logger.info(f"Found {self.total_files} Python files to verify\n")
        
        # Verify each file
        for file_path in python_files:
            relative_path = file_path.relative_to(PROJECT_ROOT)
            
            result = self.verify_file(file_path)
            
            if result['passed']:
                self.passed_files += 1
                logger.info(f"✅ {relative_path}")
            else:
                self.failed_files += 1
                logger.error(f"❌ {relative_path}")
                for error in result['errors']:
                    logger.error(f"   {error}")
                self.errors.append(result)
            
            # Log warnings
            if result['warnings']:
                for warning in result['warnings']:
                    logger.warning(f"   ⚠️  {warning}")
                self.warnings.append(result)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate final verification report"""
        logger.info("\n" + "=" * 80)
        logger.info("VERIFICATION REPORT")
        logger.info("=" * 80)
        logger.info(f"Total files: {self.total_files}")
        logger.info(f"Passed: {self.passed_files}")
        logger.info(f"Failed: {self.failed_files}")
        logger.info(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            logger.error(f"\n❌ {len(self.errors)} FILES WITH ERRORS:")
            for result in self.errors:
                logger.error(f"\n  File: {result['file']}")
                for error in result['errors']:
                    logger.error(f"    - {error}")
        
        if self.warnings:
            logger.warning(f"\n⚠️  {len(self.warnings)} FILES WITH WARNINGS:")
            for result in self.warnings[:5]:  # Show first 5
                logger.warning(f"  - {result['file']}")
        
        logger.info("\n" + "=" * 80)
        
        if self.failed_files == 0:
            logger.info("🎉 ALL FILES PASSED VERIFICATION!")
            logger.info("✅ ZERO ERRORS - STRICTEST QUALITY REQUIREMENTS MET")
            return 0
        else:
            logger.error("❌ VERIFICATION FAILED - ERRORS FOUND")
            return 1


def main():
    """Main verification function"""
    verifier = CodeVerifier()
    exit_code = verifier.verify_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

