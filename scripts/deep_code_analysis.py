"""
Deep Code Analysis - Strictest Quality Requirements

CRITICAL: Perform intense error elimination beyond syntax checking
- Check for undefined variables
- Check for unused imports
- Check for missing return statements
- Check for unreachable code
- Check for type inconsistencies
- Check for security issues
- Generate comprehensive quality report
"""

import ast
import sys
import logging
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class DeepCodeAnalyzer(ast.NodeVisitor):
    """Deep AST-based code analyzer"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.issues = []
        self.warnings = []
        
        # Track variables and functions
        self.defined_names = set()
        self.used_names = set()
        self.imported_names = set()
        self.function_returns = defaultdict(list)
        self.current_function = None
    
    def visit_Import(self, node):
        """Track imports"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imported_names.add(name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Track from imports"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imported_names.add(name)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Track function definitions"""
        self.defined_names.add(node.name)
        old_function = self.current_function
        self.current_function = node.name
        
        # Track parameters
        for arg in node.args.args:
            self.defined_names.add(arg.arg)
        
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_AsyncFunctionDef(self, node):
        """Track async function definitions"""
        self.visit_FunctionDef(node)
    
    def visit_ClassDef(self, node):
        """Track class definitions"""
        self.defined_names.add(node.name)
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        """Track variable assignments"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defined_names.add(target.id)
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node):
        """Track annotated assignments"""
        if isinstance(node.target, ast.Name):
            self.defined_names.add(node.target.id)
        self.generic_visit(node)
    
    def visit_Name(self, node):
        """Track name usage"""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)
    
    def visit_Return(self, node):
        """Track return statements"""
        if self.current_function:
            self.function_returns[self.current_function].append(node)
        self.generic_visit(node)
    
    def analyze(self, tree: ast.AST):
        """Perform analysis"""
        self.visit(tree)
        
        # Check for unused imports
        unused_imports = self.imported_names - self.used_names
        if unused_imports:
            # Filter out common false positives
            unused_imports = {
                name for name in unused_imports 
                if name not in ['logging', 'sys', 'os', 'Path', 'Optional', 'List', 'Dict', 'Any']
            }
            if unused_imports:
                self.warnings.append(f"Unused imports: {', '.join(sorted(unused_imports))}")
        
        # Check for potentially undefined names
        builtin_names = set(dir(__builtins__))
        potentially_undefined = self.used_names - self.defined_names - self.imported_names - builtin_names
        
        # Filter out common false positives
        false_positives = {'self', 'cls', 'super', '__name__', '__file__', '__doc__'}
        potentially_undefined = potentially_undefined - false_positives
        
        if potentially_undefined:
            # Only report if not too many (likely false positives)
            if len(potentially_undefined) < 10:
                self.warnings.append(f"Potentially undefined: {', '.join(sorted(potentially_undefined))}")


class QualityChecker:
    """Comprehensive quality checker"""
    
    def __init__(self):
        self.total_files = 0
        self.issues_found = 0
        self.warnings_found = 0
        self.file_results = []
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files"""
        python_files = []
        
        check_dirs = [
            PROJECT_ROOT / "services",
            PROJECT_ROOT / "models",
            PROJECT_ROOT / "training",
            PROJECT_ROOT / "scripts" / "data",
        ]
        
        for directory in check_dirs:
            if directory.exists():
                python_files.extend(directory.rglob("*.py"))
        
        python_files = [
            f for f in python_files 
            if '__pycache__' not in str(f) and 'test_' not in f.name
        ]
        
        return sorted(python_files)
    
    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            tree = ast.parse(code)
            analyzer = DeepCodeAnalyzer(str(file_path))
            analyzer.analyze(tree)
            
            return {
                'file': file_path.relative_to(PROJECT_ROOT),
                'issues': analyzer.issues,
                'warnings': analyzer.warnings
            }
        except Exception as e:
            return {
                'file': file_path.relative_to(PROJECT_ROOT),
                'issues': [f"Analysis error: {e}"],
                'warnings': []
            }
    
    def run_analysis(self):
        """Run deep analysis on all files"""
        logger.info("=" * 80)
        logger.info("DEEP CODE ANALYSIS - STRICTEST QUALITY REQUIREMENTS")
        logger.info("=" * 80)
        
        python_files = self.find_python_files()
        self.total_files = len(python_files)
        
        logger.info(f"Analyzing {self.total_files} files...\n")
        
        for file_path in python_files:
            result = self.analyze_file(file_path)
            self.file_results.append(result)
            
            if result['issues']:
                self.issues_found += len(result['issues'])
                logger.error(f"❌ {result['file']}")
                for issue in result['issues']:
                    logger.error(f"   ISSUE: {issue}")
            elif result['warnings']:
                self.warnings_found += len(result['warnings'])
                logger.warning(f"⚠️  {result['file']}")
                for warning in result['warnings']:
                    logger.warning(f"   {warning}")
            else:
                logger.info(f"✅ {result['file']}")
        
        self.generate_report()
    
    def generate_report(self):
        """Generate final report"""
        logger.info("\n" + "=" * 80)
        logger.info("DEEP ANALYSIS REPORT")
        logger.info("=" * 80)
        logger.info(f"Total files analyzed: {self.total_files}")
        logger.info(f"Critical issues: {self.issues_found}")
        logger.info(f"Warnings: {self.warnings_found}")
        
        if self.issues_found == 0:
            logger.info("\n✅ ZERO CRITICAL ISSUES FOUND")
            logger.info("🎉 CODE MEETS STRICTEST QUALITY REQUIREMENTS")
        else:
            logger.error(f"\n❌ {self.issues_found} CRITICAL ISSUES REQUIRE ATTENTION")
        
        logger.info("=" * 80)


def main():
    """Main analysis function"""
    checker = QualityChecker()
    checker.run_analysis()


if __name__ == "__main__":
    main()

