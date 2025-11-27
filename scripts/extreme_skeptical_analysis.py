#!/usr/bin/env python3
"""
EXTREME SKEPTICAL CODE ANALYSIS
Zero-tolerance for ANY potential issues

Checks for:
1. asyncio.get_event_loop() deprecation (Python 3.10+)
2. Potential race conditions in shared state
3. Missing connection cleanup
4. Timeout edge cases
5. Cache invalidation issues
6. Error handling gaps
7. Resource exhaustion scenarios
8. Thread safety violations
9. Async/await anti-patterns
10. Production deployment risks
"""

import os
import sys
import ast
import re
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict

class ExtremeSkepticalAnalyzer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.critical_issues = []
        self.warnings = []
        self.suggestions = []
        
    def analyze_file(self, filepath: Path) -> Dict:
        """Perform extreme skeptical analysis on a file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(filepath))
            
            # Run all checks
            self._check_event_loop_deprecation(filepath, content, tree)
            self._check_shared_state_race_conditions(filepath, content, tree)
            self._check_connection_cleanup(filepath, content, tree)
            self._check_timeout_edge_cases(filepath, content, tree)
            self._check_cache_invalidation(filepath, content, tree)
            self._check_error_handling_gaps(filepath, content, tree)
            self._check_resource_exhaustion(filepath, content, tree)
            self._check_thread_safety(filepath, content, tree)
            self._check_async_antipatterns(filepath, content, tree)
            self._check_production_risks(filepath, content, tree)
            
            return {'analyzed': True}
            
        except Exception as e:
            self.critical_issues.append(f"❌ ANALYSIS FAILED: {filepath} - {e}")
            return {}
    
    def _check_event_loop_deprecation(self, filepath: Path, content: str, tree: ast.AST):
        """Check for deprecated asyncio.get_event_loop() usage"""
        if 'asyncio.get_event_loop()' in content or 'get_event_loop()' in content:
            for i, line in enumerate(content.split('\n'), 1):
                # Skip comments and fixed lines
                if '#' in line and 'get_event_loop()' in line:
                    continue
                if 'get_event_loop()' in line and 'asyncio.get_running_loop()' not in line:
                    self.critical_issues.append(
                        f"🔴 CRITICAL: {filepath}:{i} - Using deprecated asyncio.get_event_loop(). "
                        f"Use asyncio.get_running_loop() or asyncio.new_event_loop() instead."
                    )
    
    def _check_shared_state_race_conditions(self, filepath: Path, content: str, tree: ast.AST):
        """Check for potential race conditions in shared state"""
        # Check for class-level mutable defaults
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                # Check if default value is mutable
                                if isinstance(item.value, (ast.List, ast.Dict, ast.Set)):
                                    self.warnings.append(
                                        f"⚠️  WARNING: {filepath}:{item.lineno} - "
                                        f"Mutable class attribute '{target.id}' may cause race conditions"
                                    )
    
    def _check_connection_cleanup(self, filepath: Path, content: str, tree: ast.AST):
        """Check for missing connection cleanup"""
        # Check for database/client connections without proper cleanup
        connection_patterns = [
            'AsyncQdrantClient', 'AsyncGraphDatabase', 'asyncpg.create_pool',
            'httpx.AsyncClient', 'aiohttp.ClientSession'
        ]
        
        for pattern in connection_patterns:
            if pattern in content:
                # Check if there's a corresponding close() or cleanup
                if 'close()' not in content and 'cleanup()' not in content:
                    self.warnings.append(
                        f"⚠️  WARNING: {filepath} - {pattern} used but no explicit close() found"
                    )
    
    def _check_timeout_edge_cases(self, filepath: Path, content: str, tree: ast.AST):
        """Check for timeout edge cases"""
        # Check for asyncio.wait_for without proper timeout handling
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'wait_for':
                        # Check if timeout is hardcoded or configurable
                        if node.keywords:
                            for kw in node.keywords:
                                if kw.arg == 'timeout':
                                    if isinstance(kw.value, ast.Constant) and kw.value.value > 60:
                                        self.warnings.append(
                                            f"⚠️  WARNING: {filepath}:{node.lineno} - "
                                            f"Long timeout ({kw.value.value}s) may cause request queueing"
                                        )
    
    def _check_cache_invalidation(self, filepath: Path, content: str, tree: ast.AST):
        """Check for cache invalidation issues"""
        if 'cache' in content.lower():
            # Check if cache has TTL
            if 'ttl' not in content.lower() and 'expire' not in content.lower():
                self.suggestions.append(
                    f"💡 SUGGESTION: {filepath} - Cache without TTL may grow unbounded"
                )
    
    def _check_error_handling_gaps(self, filepath: Path, content: str, tree: ast.AST):
        """Check for error handling gaps"""
        # Check for async functions without try-except
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                # Check if function has any try-except
                has_try = any(isinstance(child, ast.Try) for child in ast.walk(node))
                
                # Skip simple functions (health checks, etc.)
                if not has_try and len(list(ast.walk(node))) > 10:
                    # Check if it's an endpoint or important function
                    if not node.name.startswith('_'):
                        self.warnings.append(
                            f"⚠️  WARNING: {filepath}:{node.lineno} - "
                            f"Public async function '{node.name}' has no error handling"
                        )
    
    def _check_resource_exhaustion(self, filepath: Path, content: str, tree: ast.AST):
        """Check for resource exhaustion scenarios"""
        # Check for unbounded loops
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                # Check if while True without break
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    has_break = any(isinstance(child, ast.Break) for child in ast.walk(node))
                    if not has_break:
                        self.critical_issues.append(
                            f"🔴 CRITICAL: {filepath}:{node.lineno} - "
                            f"Infinite loop without break condition"
                        )
    
    def _check_thread_safety(self, filepath: Path, content: str, tree: ast.AST):
        """Check for thread safety violations"""
        # Check for global state modifications
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                self.warnings.append(
                    f"⚠️  WARNING: {filepath}:{node.lineno} - "
                    f"Global variable modification may not be thread-safe"
                )
    
    def _check_async_antipatterns(self, filepath: Path, content: str, tree: ast.AST):
        """Check for async/await anti-patterns"""
        # Check for time.sleep in async functions
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            if child.func.attr == 'sleep' and isinstance(child.func.value, ast.Name):
                                if child.func.value.id == 'time':
                                    self.critical_issues.append(
                                        f"🔴 CRITICAL: {filepath}:{child.lineno} - "
                                        f"Using time.sleep() in async function. Use asyncio.sleep() instead."
                                    )
    
    def _check_production_risks(self, filepath: Path, content: str, tree: ast.AST):
        """Check for production deployment risks"""
        # Check for debug mode
        if 'debug=True' in content or 'DEBUG = True' in content:
            self.critical_issues.append(
                f"🔴 CRITICAL: {filepath} - Debug mode enabled in production code"
            )
        
        # Check for print statements (should use logging)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'print':
                    self.suggestions.append(
                        f"💡 SUGGESTION: {filepath}:{node.lineno} - "
                        f"Use logging instead of print() for production"
                    )

    def analyze_directory(self, directory: str) -> Dict:
        """Analyze all Python files in directory"""
        dir_path = self.root_dir / directory

        if not dir_path.exists():
            print(f"❌ Directory not found: {dir_path}")
            return {}

        python_files = list(dir_path.rglob("*.py"))
        print(f"\n🔬 EXTREME SKEPTICAL ANALYSIS: {len(python_files)} files in {directory}/\n")

        analyzed = 0
        for filepath in python_files:
            if '.venv' in str(filepath) or '__pycache__' in str(filepath):
                continue
            result = self.analyze_file(filepath)
            if result:
                analyzed += 1

        return {'files_analyzed': analyzed}

    def print_report(self):
        """Print comprehensive report"""
        print("\n" + "="*80)
        print("🔬 EXTREME SKEPTICAL ANALYSIS REPORT")
        print("="*80)

        # Critical issues
        if self.critical_issues:
            print(f"\n🔴 CRITICAL ISSUES: {len(self.critical_issues)}")
            print("-" * 80)
            for issue in self.critical_issues:
                print(f"  {issue}")
        else:
            print("\n✅ NO CRITICAL ISSUES FOUND")

        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            print("-" * 80)
            for warning in self.warnings[:20]:  # Show first 20
                print(f"  {warning}")
            if len(self.warnings) > 20:
                print(f"  ... and {len(self.warnings) - 20} more warnings")
        else:
            print("\n✅ NO WARNINGS")

        # Suggestions
        if self.suggestions:
            print(f"\n💡 SUGGESTIONS: {len(self.suggestions)}")
            print("-" * 80)
            for suggestion in self.suggestions[:10]:  # Show first 10
                print(f"  {suggestion}")
            if len(self.suggestions) > 10:
                print(f"  ... and {len(self.suggestions) - 10} more suggestions")

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80 + "\n")

def main():
    """Main entry point"""
    root_dir = "/Users/jiangshengbo/Desktop/Sustainability-AI-Model"

    print("🚀 STARTING EXTREME SKEPTICAL CODE ANALYSIS")
    print("="*80)
    print("Zero-tolerance for ANY potential issues")
    print("="*80)

    analyzer = ExtremeSkepticalAnalyzer(root_dir)

    # Analyze all services
    services_result = analyzer.analyze_directory("services")

    # Analyze models
    models_result = analyzer.analyze_directory("models")

    # Analyze training
    training_result = analyzer.analyze_directory("training")

    # Print report
    analyzer.print_report()

    # Summary
    total_files = (services_result.get('files_analyzed', 0) +
                   models_result.get('files_analyzed', 0) +
                   training_result.get('files_analyzed', 0))

    print(f"📊 Summary:")
    print(f"   Files Analyzed: {total_files}")
    print(f"   Critical Issues: {len(analyzer.critical_issues)}")
    print(f"   Warnings: {len(analyzer.warnings)}")
    print(f"   Suggestions: {len(analyzer.suggestions)}")

    if len(analyzer.critical_issues) == 0:
        print("\n✅ ZERO CRITICAL ISSUES - EXCELLENT CODE QUALITY")
        return 0
    else:
        print(f"\n🔴 {len(analyzer.critical_issues)} CRITICAL ISSUES MUST BE FIXED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

