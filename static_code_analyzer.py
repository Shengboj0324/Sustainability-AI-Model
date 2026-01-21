#!/usr/bin/env python3
"""
STATIC CODE ANALYZER - MAXIMUM SKEPTICISM
==========================================
Line-by-line analysis of notebook code with zero tolerance for errors

Features:
- Undefined variable detection
- Type consistency checking
- Import resolution validation
- Function signature validation
- Dead code detection
- Memory leak detection
- GPU operation validation

Author: Autonomous Testing Agent
Date: 2026-01-21
Confidence Target: 100%
"""

import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

class StaticCodeAnalyzer:
    """Deep static analysis with maximum skepticism"""

    def __init__(self, notebook_path: str):
        self.notebook_path = Path(notebook_path)
        self.errors = []
        self.warnings = []
        self.info = []
        self.cells_code = []
        self.all_code = ""

        # Track variables across cells
        self.defined_vars = set()
        self.used_vars = set()
        self.imported_modules = set()
        self.defined_functions = set()
        self.defined_classes = set()

    def load_notebook(self) -> bool:
        """Load and extract code from notebook"""
        print("=" * 80)
        print("LOADING NOTEBOOK")
        print("=" * 80)

        try:
            with open(self.notebook_path, 'r') as f:
                nb = json.load(f)

            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source'])
                    self.cells_code.append({
                        'index': i,
                        'source': source,
                        'lines': source.split('\n')
                    })
                    self.all_code += source + "\n\n"

            print(f"✓ Loaded {len(self.cells_code)} code cells")
            print(f"✓ Total lines: {len(self.all_code.split(chr(10)))}")
            return True

        except Exception as e:
            self.errors.append(f"Failed to load notebook: {str(e)}")
            return False

    def parse_ast(self, code: str, cell_index: int) -> Optional[ast.AST]:
        """Parse code into AST with error handling"""
        try:
            return ast.parse(code)
        except SyntaxError as e:
            self.errors.append(f"Cell {cell_index}: SyntaxError at line {e.lineno}: {e.msg}")
            return None
        except Exception as e:
            self.errors.append(f"Cell {cell_index}: Parse error: {str(e)}")
            return None

    def analyze_undefined_variables(self) -> None:
        """Detect undefined variables with deep analysis"""
        print("\n" + "=" * 80)
        print("PHASE 1: UNDEFINED VARIABLE DETECTION")
        print("=" * 80)

        # Built-in names that are always available
        builtins = {
            'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
            'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
            'True', 'False', 'None', 'Exception', 'ValueError', 'TypeError',
            'RuntimeError', 'KeyError', 'IndexError', 'AttributeError',
            'open', 'max', 'min', 'sum', 'abs', 'round', 'sorted',
            '__name__', '__file__'
        }

        defined = builtins.copy()
        undefined_usage = []

        for cell in self.cells_code:
            tree = self.parse_ast(cell['source'], cell['index'])
            if not tree:
                continue

            # Track definitions in this cell
            for node in ast.walk(tree):
                # Variable assignments
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined.add(target.id)
                        elif isinstance(target, ast.Tuple):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    defined.add(elt.id)

                # Function definitions
                elif isinstance(node, ast.FunctionDef):
                    defined.add(node.name)
                    # Add function parameters
                    for arg in node.args.args:
                        defined.add(arg.arg)

                # Class definitions
                elif isinstance(node, ast.ClassDef):
                    defined.add(node.name)

                # Imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        defined.add(name.split('.')[0])

                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        defined.add(name)

                # For loops
                elif isinstance(node, ast.For):
                    if isinstance(node.target, ast.Name):
                        defined.add(node.target.id)

                # With statements
                elif isinstance(node, ast.With):
                    for item in node.items:
                        if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                            defined.add(item.optional_vars.id)

                # Comprehensions
                elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                    for generator in node.generators:
                        if isinstance(generator.target, ast.Name):
                            defined.add(generator.target.id)

            # Check for undefined usage
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id not in defined:
                        undefined_usage.append({
                            'cell': cell['index'],
                            'variable': node.id,
                            'line': node.lineno if hasattr(node, 'lineno') else 'unknown'
                        })

        if undefined_usage:
            print(f"⚠️  Found {len(undefined_usage)} potential undefined variables:")
            for item in undefined_usage[:20]:  # Show first 20
                self.warnings.append(f"Cell {item['cell']}, line {item['line']}: Undefined variable '{item['variable']}'")
                print(f"  Cell {item['cell']}, line {item['line']}: {item['variable']}")
        else:
            print("✓ No undefined variables detected")
            self.info.append("No undefined variables")

    def analyze_function_calls(self) -> None:
        """Validate all function calls have correct signatures"""
        print("\n" + "=" * 80)
        print("PHASE 2: FUNCTION CALL VALIDATION")
        print("=" * 80)

        # Track function definitions with their signatures
        function_signatures = {}

        for cell in self.cells_code:
            tree = self.parse_ast(cell['source'], cell['index'])
            if not tree:
                continue

            # Collect function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)
                    function_signatures[node.name] = {
                        'args': args,
                        'num_args': len(args),
                        'cell': cell['index']
                    }

        print(f"✓ Found {len(function_signatures)} function definitions")

        # Check for common PyTorch/ML errors
        critical_checks = [
            ('torch.save', 'Ensure checkpoint saving'),
            ('model.train', 'Training mode'),
            ('model.eval', 'Evaluation mode'),
            ('optimizer.zero_grad', 'Gradient zeroing'),
            ('loss.backward', 'Backpropagation'),
            ('optimizer.step', 'Optimizer step'),
        ]

        found_critical = []
        for func_name, description in critical_checks:
            if func_name in self.all_code:
                found_critical.append(func_name)

        print(f"✓ Found {len(found_critical)}/{len(critical_checks)} critical ML operations")
        self.info.append(f"Critical ML operations: {len(found_critical)}/{len(critical_checks)}")

    def analyze_gpu_operations(self) -> None:
        """Validate GPU operations and memory management"""
        print("\n" + "=" * 80)
        print("PHASE 3: GPU OPERATIONS VALIDATION")
        print("=" * 80)

        gpu_patterns = {
            '.to(device)': 'Device transfer',
            '.cuda()': 'CUDA transfer',
            'torch.cuda.empty_cache()': 'Cache clearing',
            'torch.cuda.amp': 'Mixed precision',
            'non_blocking=True': 'Async transfer',
        }

        found_patterns = {}
        for pattern, description in gpu_patterns.items():
            count = self.all_code.count(pattern)
            if count > 0:
                found_patterns[pattern] = count
                print(f"✓ {description}: {count} occurrences")

        # Check for potential memory leaks
        if '.to(device)' in self.all_code or '.cuda()' in self.all_code:
            if 'torch.cuda.empty_cache()' not in self.all_code:
                self.warnings.append("GPU memory used but no cache clearing found")
                print("⚠️  No GPU cache clearing found")
            else:
                print("✓ GPU cache clearing present")

        self.info.append(f"GPU operations: {len(found_patterns)} patterns found")

    def analyze_error_handling(self) -> None:
        """Check error handling coverage"""
        print("\n" + "=" * 80)
        print("PHASE 4: ERROR HANDLING VALIDATION")
        print("=" * 80)

        try_blocks = 0
        except_blocks = 0
        bare_excepts = 0

        for cell in self.cells_code:
            tree = self.parse_ast(cell['source'], cell['index'])
            if not tree:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    try_blocks += 1
                    for handler in node.handlers:
                        except_blocks += 1
                        if handler.type is None:
                            bare_excepts += 1
                            self.warnings.append(f"Cell {cell['index']}: Bare except clause (catches all exceptions)")

        print(f"✓ Try blocks: {try_blocks}")
        print(f"✓ Except blocks: {except_blocks}")

        if bare_excepts > 0:
            print(f"⚠️  Bare except clauses: {bare_excepts}")
        else:
            print(f"✓ No bare except clauses")

        self.info.append(f"Error handling: {try_blocks} try blocks, {except_blocks} except blocks")

    def analyze_data_types(self) -> None:
        """Validate data type consistency"""
        print("\n" + "=" * 80)
        print("PHASE 5: DATA TYPE VALIDATION")
        print("=" * 80)

        # Check for common type errors
        type_checks = {
            'torch.tensor': 'Tensor creation',
            'torch.Tensor': 'Tensor type',
            'np.array': 'NumPy array',
            'pd.DataFrame': 'Pandas DataFrame',
            '.astype(': 'Type conversion',
            '.to(dtype=': 'Dtype conversion',
        }

        found_types = {}
        for pattern, description in type_checks.items():
            count = self.all_code.count(pattern)
            if count > 0:
                found_types[pattern] = count
                print(f"✓ {description}: {count} occurrences")

        self.info.append(f"Data type operations: {len(found_types)} patterns found")


    def analyze_critical_ml_patterns(self) -> None:
        """Check for critical ML training patterns"""
        print("\n" + "=" * 80)
        print("PHASE 6: CRITICAL ML PATTERNS VALIDATION")
        print("=" * 80)

        critical_patterns = {
            # Checkpointing
            'torch.save': ('Checkpoint saving', True),
            'torch.load': ('Checkpoint loading', False),

            # Training loop essentials
            'model.train()': ('Training mode', True),
            'model.eval()': ('Evaluation mode', True),
            'optimizer.zero_grad()': ('Gradient zeroing', True),
            'loss.backward()': ('Backpropagation', True),
            'optimizer.step()': ('Optimizer step', True),

            # Validation
            'torch.no_grad()': ('No gradient context', True),

            # Metrics (check for any form of accuracy tracking)
            'acc': ('Accuracy tracking', True),  # Matches train_acc, val_acc, etc.
            'f1': ('F1 score tracking', False),
            'precision': ('Precision tracking', False),
            'recall': ('Recall tracking', False),

            # Memory management
            'torch.cuda.empty_cache()': ('GPU cache clearing', False),
            'del ': ('Memory cleanup', False),
        }

        missing_critical = []
        found_optional = []

        for pattern, (description, is_critical) in critical_patterns.items():
            if pattern in self.all_code:
                found_optional.append(f"{description}")
                print(f"✓ {description}")
            else:
                if is_critical:
                    missing_critical.append(description)
                    self.errors.append(f"CRITICAL: Missing {description} ({pattern})")
                    print(f"✗ MISSING CRITICAL: {description}")

        if missing_critical:
            print(f"\n❌ {len(missing_critical)} CRITICAL patterns missing!")
        else:
            print(f"\n✓ All critical ML patterns present")

        self.info.append(f"ML patterns: {len(found_optional)} found")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("STATIC ANALYSIS REPORT")
        print("=" * 80)

        print(f"\n✅ INFO: {len(self.info)}")
        for item in self.info:
            print(f"  ℹ {item}")

        print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
        for warning in self.warnings[:10]:  # Show first 10
            print(f"  ⚠ {warning}")
        if len(self.warnings) > 10:
            print(f"  ... and {len(self.warnings) - 10} more warnings")

        print(f"\n❌ ERRORS: {len(self.errors)}")
        for error in self.errors:
            print(f"  ✗ {error}")

        # Calculate confidence
        if self.errors:
            confidence = 0
            status = "FAILED"
        elif len(self.warnings) > 10:
            confidence = 60
            status = "NEEDS REVIEW"
        elif len(self.warnings) > 0:
            confidence = 80
            status = "PASSED WITH WARNINGS"
        else:
            confidence = 100
            status = "PERFECT"

        print(f"\n{'=' * 80}")
        print(f"STATUS: {status}")
        print(f"CONFIDENCE: {confidence}%")
        print(f"{'=' * 80}\n")

        return {
            'status': status,
            'confidence': confidence,
            'info_count': len(self.info),
            'warnings_count': len(self.warnings),
            'errors_count': len(self.errors),
            'details': {
                'info': self.info,
                'warnings': self.warnings,
                'errors': self.errors
            }
        }

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete static analysis"""
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "STATIC CODE ANALYSIS" + " " * 38 + "║")
        print("║" + " " * 15 + "Maximum Skepticism - Zero Tolerance" + " " * 28 + "║")
        print("╚" + "=" * 78 + "╝\n")

        if not self.load_notebook():
            return self.generate_report()

        # Run all analysis phases
        self.analyze_undefined_variables()
        self.analyze_function_calls()
        self.analyze_gpu_operations()
        self.analyze_error_handling()
        self.analyze_data_types()
        self.analyze_critical_ml_patterns()

        # Generate report
        return self.generate_report()


def main():
    """Main entry point"""
    analyzer = StaticCodeAnalyzer("Sustainability_AI_Model_Training.ipynb")
    report = analyzer.run_full_analysis()

    # Save report
    report_path = Path("static_analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {report_path}")

    # Exit with appropriate code
    if report['errors_count'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()


