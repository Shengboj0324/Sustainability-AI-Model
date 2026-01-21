#!/usr/bin/env python3
"""
Validate Python syntax in Jupyter notebook cells.
"""
import json
import ast
import sys

def validate_notebook(notebook_path):
    """Validate all code cells in a Jupyter notebook for syntax errors."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    errors = []
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if not source.strip():
                continue
            
            try:
                ast.parse(source)
                print(f"✓ Cell {i+1}: OK")
            except SyntaxError as e:
                error_msg = f"✗ Cell {i+1}: SyntaxError at line {e.lineno}: {e.msg}"
                print(error_msg)
                errors.append((i+1, e))
    
    if errors:
        print(f"\n❌ Found {len(errors)} syntax error(s)")
        return False
    else:
        print(f"\n✅ All cells validated successfully!")
        return True

if __name__ == '__main__':
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'Sustainability_AI_Model_Training.ipynb'
    success = validate_notebook(notebook_path)
    sys.exit(0 if success else 1)

