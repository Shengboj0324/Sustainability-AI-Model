#!/usr/bin/env python3
"""
CODE QUALITY UNCERTAINTY ASSESSMENT
Static analysis, complexity metrics, security vulnerabilities, performance bottlenecks
"""

import os
import ast
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class QualityIssue:
    file_path: str
    line_number: int
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    description: str
    recommendation: str

class CodeQualityAssessor:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.issues: List[QualityIssue] = []
        self.metrics: Dict[str, Any] = defaultdict(int)
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            # Skip virtual environments and cache
            dirs[:] = [d for d in dirs if d not in ['.venv', 'venv', '__pycache__', '.git', 'node_modules']]
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    line_number=e.lineno or 0,
                    severity="CRITICAL",
                    category="SYNTAX_ERROR",
                    description=f"Syntax error: {e.msg}",
                    recommendation="Fix syntax error immediately"
                ))
                self.metrics["syntax_errors"] += 1
                return
            
            # Analyze complexity
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self.calculate_complexity(node)
                    self.metrics["total_functions"] += 1
                    
                    if complexity > 15:
                        self.issues.append(QualityIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            severity="HIGH",
                            category="HIGH_COMPLEXITY",
                            description=f"Function '{node.name}' has complexity {complexity} (threshold: 15)",
                            recommendation="Refactor into smaller functions"
                        ))
                        self.metrics["high_complexity_functions"] += 1
                    elif complexity > 10:
                        self.metrics["medium_complexity_functions"] += 1
                
                elif isinstance(node, ast.ClassDef):
                    self.metrics["total_classes"] += 1
                    # Check class size
                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    if len(methods) > 20:
                        self.issues.append(QualityIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            severity="MEDIUM",
                            category="LARGE_CLASS",
                            description=f"Class '{node.name}' has {len(methods)} methods (threshold: 20)",
                            recommendation="Consider splitting into smaller classes"
                        ))
                        self.metrics["large_classes"] += 1
            
            # Security checks
            self.check_security_issues(file_path, content)
            
            # Performance checks
            self.check_performance_issues(file_path, content, tree)
            
            # Code smell checks
            self.check_code_smells(file_path, content, tree)
            
            self.metrics["files_analyzed"] += 1
            
        except Exception as e:
            self.issues.append(QualityIssue(
                file_path=str(file_path),
                line_number=0,
                severity="MEDIUM",
                category="ANALYSIS_ERROR",
                description=f"Failed to analyze: {str(e)}",
                recommendation="Manual review required"
            ))
            self.metrics["analysis_errors"] += 1
    
    def check_security_issues(self, file_path: Path, content: str) -> None:
        """Check for security vulnerabilities"""
        lines = content.split('\n')
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Exclude test files and examples
                    if 'test' not in str(file_path).lower() and 'example' not in line.lower():
                        self.issues.append(QualityIssue(
                            file_path=str(file_path),
                            line_number=i,
                            severity="CRITICAL",
                            category="SECURITY",
                            description=description,
                            recommendation="Use environment variables or secure vault"
                        ))
                        self.metrics["security_issues"] += 1
        
        # Check for SQL injection risks
        if re.search(r'execute\s*\([^)]*%s[^)]*\)', content):
            self.issues.append(QualityIssue(
                file_path=str(file_path),
                line_number=0,
                severity="HIGH",
                category="SECURITY",
                description="Potential SQL injection vulnerability",
                recommendation="Use parameterized queries"
            ))
            self.metrics["security_issues"] += 1
        
        # Check for eval/exec usage
        if re.search(r'\beval\s*\(', content) or re.search(r'\bexec\s*\(', content):
            self.issues.append(QualityIssue(
                file_path=str(file_path),
                line_number=0,
                severity="CRITICAL",
                category="SECURITY",
                description="Use of eval() or exec() detected",
                recommendation="Avoid eval/exec - use safer alternatives"
            ))
            self.metrics["security_issues"] += 1

    def check_performance_issues(self, file_path: Path, content: str, tree: ast.AST) -> None:
        """Check for performance bottlenecks"""
        # Check for nested loops
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if child != node and isinstance(child, (ast.For, ast.While)):
                        self.issues.append(QualityIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            severity="MEDIUM",
                            category="PERFORMANCE",
                            description="Nested loops detected - potential O(n²) complexity",
                            recommendation="Consider optimization or vectorization"
                        ))
                        self.metrics["performance_issues"] += 1
                        break

        # Check for string concatenation in loops
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if re.search(r'for\s+\w+\s+in', line):
                # Check next few lines for string concatenation
                for j in range(i, min(i + 10, len(lines))):
                    if '+=' in lines[j] and 'str' in lines[j].lower():
                        self.issues.append(QualityIssue(
                            file_path=str(file_path),
                            line_number=i,
                            severity="LOW",
                            category="PERFORMANCE",
                            description="String concatenation in loop",
                            recommendation="Use list.append() and ''.join() instead"
                        ))
                        self.metrics["performance_issues"] += 1
                        break

        # Check for global variables
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    line_number=node.lineno,
                    severity="LOW",
                    category="PERFORMANCE",
                    description="Global variable usage",
                    recommendation="Avoid globals - use function parameters or class attributes"
                ))
                self.metrics["performance_issues"] += 1

    def check_code_smells(self, file_path: Path, content: str, tree: ast.AST) -> None:
        """Check for code smells"""
        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if func_lines > 100:
                    self.issues.append(QualityIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        severity="MEDIUM",
                        category="CODE_SMELL",
                        description=f"Function '{node.name}' is {func_lines} lines (threshold: 100)",
                        recommendation="Break into smaller functions"
                    ))
                    self.metrics["code_smells"] += 1

                # Check for too many parameters
                if len(node.args.args) > 7:
                    self.issues.append(QualityIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        severity="LOW",
                        category="CODE_SMELL",
                        description=f"Function '{node.name}' has {len(node.args.args)} parameters (threshold: 7)",
                        recommendation="Use dataclass or config object"
                    ))
                    self.metrics["code_smells"] += 1

        # Check for commented code
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#') and len(stripped) > 50:
                # Heuristic: long comments might be commented code
                if any(keyword in stripped for keyword in ['def ', 'class ', 'import ', 'return ', 'if ', 'for ']):
                    self.issues.append(QualityIssue(
                        file_path=str(file_path),
                        line_number=i,
                        severity="LOW",
                        category="CODE_SMELL",
                        description="Possible commented-out code",
                        recommendation="Remove dead code or use version control"
                    ))
                    self.metrics["code_smells"] += 1

        # Check for TODO/FIXME
        for i, line in enumerate(lines, 1):
            if re.search(r'#\s*(TODO|FIXME|XXX|HACK)', line, re.IGNORECASE):
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    line_number=i,
                    severity="LOW",
                    category="CODE_SMELL",
                    description="TODO/FIXME comment found",
                    recommendation="Address or create issue tracker item"
                ))
                self.metrics["todos"] += 1

    def generate_report(self) -> None:
        """Generate comprehensive quality report"""
        print("\n" + "="*80)
        print("📊 CODE QUALITY UNCERTAINTY ASSESSMENT REPORT")
        print("="*80)

        # Overall metrics
        print(f"\nOVERALL METRICS:")
        print(f"  Files Analyzed: {self.metrics['files_analyzed']}")
        print(f"  Total Functions: {self.metrics['total_functions']}")
        print(f"  Total Classes: {self.metrics['total_classes']}")
        print(f"  Total Issues: {len(self.issues)}")

        # Issue breakdown by severity
        severity_counts = defaultdict(int)
        for issue in self.issues:
            severity_counts[issue.severity] += 1

        print(f"\nISSUES BY SEVERITY:")
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = severity_counts[severity]
            status = "❌" if severity == "CRITICAL" and count > 0 else "⚠️" if count > 0 else "✅"
            print(f"  {status} {severity:10s}: {count:4d}")

        # Issue breakdown by category
        category_counts = defaultdict(int)
        for issue in self.issues:
            category_counts[issue.category] += 1

        print(f"\nISSUES BY CATEGORY:")
        for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  {category:20s}: {count:4d}")

        # Complexity metrics
        print(f"\nCOMPLEXITY METRICS:")
        print(f"  High Complexity Functions: {self.metrics['high_complexity_functions']}")
        print(f"  Medium Complexity Functions: {self.metrics['medium_complexity_functions']}")
        print(f"  Large Classes: {self.metrics['large_classes']}")

        # Security metrics
        print(f"\nSECURITY METRICS:")
        print(f"  Security Issues: {self.metrics['security_issues']}")

        # Performance metrics
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Performance Issues: {self.metrics['performance_issues']}")

        # Code smell metrics
        print(f"\nCODE SMELL METRICS:")
        print(f"  Code Smells: {self.metrics['code_smells']}")
        print(f"  TODOs/FIXMEs: {self.metrics['todos']}")

        # Critical issues details
        critical_issues = [i for i in self.issues if i.severity == "CRITICAL"]
        if critical_issues:
            print(f"\n❌ CRITICAL ISSUES ({len(critical_issues)}):")
            for issue in critical_issues[:10]:  # Show first 10
                print(f"  - {issue.file_path}:{issue.line_number}")
                print(f"    {issue.category}: {issue.description}")
                print(f"    → {issue.recommendation}")

        # Quality score
        total_issues = len(self.issues)
        critical_count = severity_counts["CRITICAL"]
        high_count = severity_counts["HIGH"]

        # Scoring: deduct points for issues
        score = 100
        score -= critical_count * 10
        score -= high_count * 5
        score -= severity_counts["MEDIUM"] * 2
        score -= severity_counts["LOW"] * 0.5
        score = max(0, score)

        print(f"\n" + "="*80)
        print(f"QUALITY SCORE: {score:.1f}/100")

        if score >= 95 and critical_count == 0:
            print("✅ VERDICT: INDUSTRIAL-GRADE CODE QUALITY")
            grade = "A+"
        elif score >= 90 and critical_count == 0:
            print("✅ VERDICT: EXCELLENT CODE QUALITY")
            grade = "A"
        elif score >= 80 and critical_count == 0:
            print("⚠️  VERDICT: GOOD CODE QUALITY - MINOR IMPROVEMENTS NEEDED")
            grade = "B+"
        elif score >= 70:
            print("⚠️  VERDICT: ACCEPTABLE - SIGNIFICANT IMPROVEMENTS RECOMMENDED")
            grade = "B"
        else:
            print("❌ VERDICT: POOR CODE QUALITY - MAJOR REFACTORING REQUIRED")
            grade = "C"

        print(f"GRADE: {grade}")
        print("="*80)

def main():
    print("="*80)
    print("🔍 CODE QUALITY UNCERTAINTY ASSESSMENT")
    print("="*80)
    print("Analyzing: Static analysis, Complexity, Security, Performance")
    print("="*80)

    # Get project root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    assessor = CodeQualityAssessor(root_dir)

    # Find and analyze all Python files
    print("\n📝 Finding Python files...")
    python_files = assessor.find_python_files()
    print(f"✅ Found {len(python_files)} Python files")

    print("\n🔍 Analyzing files...")
    for i, file_path in enumerate(python_files, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(python_files)}")
        assessor.analyze_file(file_path)

    # Generate report
    assessor.generate_report()

    # Exit code
    critical_issues = sum(1 for i in assessor.issues if i.severity == "CRITICAL")
    if critical_issues == 0 and len(assessor.issues) < 50:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()


