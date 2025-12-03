#!/usr/bin/env python3
"""
Full-Scale Upgrade Analysis Script
Analyzes all upgrades and validates compatibility
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def run_command(cmd: str) -> Tuple[int, str, str]:
    """Run shell command and return exit code, stdout, stderr"""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr

def check_python_version() -> Dict[str, any]:
    """Check Python version compatibility"""
    print("\n" + "="*80)
    print("🐍 PYTHON VERSION CHECK")
    print("="*80)
    
    code, stdout, stderr = run_command("python3 --version")
    version = stdout.strip()
    print(f"Current Python: {version}")
    
    # Check if Python 3.11+
    major, minor = sys.version_info[:2]
    compatible = major == 3 and minor >= 11
    
    status = "✅ COMPATIBLE" if compatible else "⚠️  UPGRADE NEEDED"
    print(f"Status: {status}")
    print(f"Required: Python 3.11+")
    print(f"Detected: Python {major}.{minor}")
    
    return {
        "compatible": compatible,
        "version": f"{major}.{minor}",
        "required": "3.11+"
    }

def check_package_versions() -> Dict[str, any]:
    """Check installed package versions"""
    print("\n" + "="*80)
    print("📦 PACKAGE VERSION CHECK")
    print("="*80)
    
    packages = [
        "torch",
        "transformers",
        "fastapi",
        "pydantic",
        "qdrant-client",
        "neo4j",
        "uvicorn",
    ]
    
    results = {}
    for package in packages:
        code, stdout, stderr = run_command(f"pip show {package} 2>/dev/null | grep Version")
        if code == 0:
            version = stdout.split(":")[1].strip() if ":" in stdout else "Unknown"
            results[package] = version
            print(f"  {package}: {version}")
        else:
            results[package] = "Not installed"
            print(f"  {package}: ❌ Not installed")
    
    return results

def check_docker_images() -> Dict[str, any]:
    """Check Docker image versions"""
    print("\n" + "="*80)
    print("🐳 DOCKER IMAGE CHECK")
    print("="*80)
    
    images = {
        "python:3.11-slim": "Base Python image",
        "postgis/postgis:16-3.4": "PostgreSQL with PostGIS",
        "neo4j:5.16": "Neo4j Graph Database",
        "qdrant/qdrant:v1.8.0": "Qdrant Vector DB",
    }
    
    results = {}
    for image, description in images.items():
        code, stdout, stderr = run_command(f"docker pull {image} 2>&1 | tail -1")
        if "Downloaded" in stdout or "up to date" in stdout or "Image is up to date" in stdout:
            results[image] = "✅ Available"
            print(f"  {image}: ✅ Available")
        else:
            results[image] = "⚠️  Check needed"
            print(f"  {image}: ⚠️  Check needed")
    
    return results

def validate_syntax() -> Dict[str, any]:
    """Validate Python syntax for all service files"""
    print("\n" + "="*80)
    print("✅ SYNTAX VALIDATION")
    print("="*80)
    
    service_files = list(Path("services").rglob("*.py"))
    model_files = list(Path("models").rglob("*.py"))
    all_files = service_files + model_files
    
    errors = []
    for file in all_files:
        code, stdout, stderr = run_command(f"python3 -m py_compile {file}")
        if code != 0:
            errors.append((str(file), stderr))
            print(f"  ❌ {file}: SYNTAX ERROR")
        else:
            print(f"  ✅ {file}: OK")
    
    print(f"\nTotal files checked: {len(all_files)}")
    print(f"Errors found: {len(errors)}")
    
    return {
        "total_files": len(all_files),
        "errors": len(errors),
        "error_details": errors
    }

def generate_upgrade_report(results: Dict) -> None:
    """Generate comprehensive upgrade report"""
    print("\n" + "="*80)
    print("📊 FULL-SCALE UPGRADE REPORT")
    print("="*80)
    
    print("\n### Python Compatibility")
    print(f"  Current: Python {results['python']['version']}")
    print(f"  Required: Python {results['python']['required']}")
    print(f"  Status: {'✅ READY' if results['python']['compatible'] else '⚠️  UPGRADE NEEDED'}")
    
    print("\n### Package Versions")
    for package, version in results['packages'].items():
        print(f"  {package}: {version}")
    
    print("\n### Docker Images")
    for image, status in results['docker'].items():
        print(f"  {image}: {status}")
    
    print("\n### Syntax Validation")
    print(f"  Total files: {results['syntax']['total_files']}")
    print(f"  Errors: {results['syntax']['errors']}")
    
    # Overall status
    print("\n" + "="*80)
    all_good = (
        results['python']['compatible'] and
        results['syntax']['errors'] == 0
    )
    
    if all_good:
        print("🎉 UPGRADE STATUS: ✅ READY FOR DEPLOYMENT")
    else:
        print("⚠️  UPGRADE STATUS: ISSUES NEED ATTENTION")
    print("="*80)

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("🚀 FULL-SCALE UPGRADE ANALYSIS")
    print("="*80)
    
    results = {
        "python": check_python_version(),
        "packages": check_package_versions(),
        "docker": check_docker_images(),
        "syntax": validate_syntax(),
    }
    
    generate_upgrade_report(results)
    
    return 0 if results['syntax']['errors'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

