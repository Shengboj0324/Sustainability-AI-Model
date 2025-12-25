"""
Environment Detection and Validation

Detects architecture mismatches and provides actionable guidance
"""

import platform
import sys
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """Environment information"""
    python_version: str
    architecture: str
    platform_name: str
    is_arm: bool
    is_x86: bool
    is_macos: bool
    is_linux: bool
    has_mps: bool
    has_cuda: bool
    issues: List[str]
    warnings: List[str]


def detect_environment() -> EnvironmentInfo:
    """
    Detect current environment and identify issues
    
    Returns comprehensive environment information with issues/warnings
    """
    arch = platform.machine().lower()
    platform_name = platform.platform()
    python_version = platform.python_version()
    
    is_arm = arch in ('arm64', 'aarch64')
    is_x86 = arch in ('x86_64', 'amd64', 'i386', 'i686')
    is_macos = sys.platform == 'darwin'
    is_linux = sys.platform.startswith('linux')
    
    issues = []
    warnings = []
    
    # Check for MPS (Apple Silicon GPU)
    has_mps = False
    try:
        import torch
        has_mps = torch.backends.mps.is_available()
    except ImportError:
        warnings.append("PyTorch not installed - cannot detect MPS")
    
    # Check for CUDA
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass
    
    # CRITICAL: Detect x86 Python on ARM Mac (Rosetta)
    if is_macos and is_x86:
        # Check if running on ARM hardware via Rosetta
        import subprocess
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=2
            )
            cpu_brand = result.stdout.strip().lower()
            if 'apple' in cpu_brand or 'm1' in cpu_brand or 'm2' in cpu_brand or 'm3' in cpu_brand or 'm4' in cpu_brand:
                issues.append(
                    "🚨 CRITICAL: Running x86 Python on ARM Mac (Rosetta emulation)\n"
                    "   This causes JAX/TensorFlow/Transformers import failures.\n"
                    "   FIX: Install ARM Python:\n"
                    "   $ brew install python@3.11\n"
                    "   $ python3.11 -m venv venv-arm\n"
                    "   $ source venv-arm/bin/activate\n"
                    "   $ pip install -r requirements-arm.txt"
                )
        except Exception:
            pass
    
    # Check Python version
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        issues.append(f"Python {python_version} is too old. Requires Python 3.9+")
    
    if major == 3 and minor == 9:
        warnings.append(
            "Python 3.9 is minimum version. Python 3.11+ recommended for better performance"
        )
    
    return EnvironmentInfo(
        python_version=python_version,
        architecture=arch,
        platform_name=platform_name,
        is_arm=is_arm,
        is_x86=is_x86,
        is_macos=is_macos,
        is_linux=is_linux,
        has_mps=has_mps,
        has_cuda=has_cuda,
        issues=issues,
        warnings=warnings
    )


def check_environment(raise_on_issues: bool = False) -> EnvironmentInfo:
    """
    Check environment and log issues/warnings
    
    Args:
        raise_on_issues: If True, raise RuntimeError on critical issues
        
    Returns:
        EnvironmentInfo with detected issues
    """
    env = detect_environment()
    
    # Log environment info
    logger.info(f"Environment: {env.platform_name}")
    logger.info(f"Python: {env.python_version} ({env.architecture})")
    
    if env.has_mps:
        logger.info("🍎 Apple Silicon GPU (MPS) available")
    if env.has_cuda:
        logger.info("🔥 CUDA GPU available")
    
    # Log warnings
    for warning in env.warnings:
        logger.warning(warning)
    
    # Log issues
    for issue in env.issues:
        logger.error(issue)
    
    if env.issues and raise_on_issues:
        raise RuntimeError(
            f"Environment has {len(env.issues)} critical issue(s). "
            "See logs for details and fixes."
        )
    
    return env


def get_optimal_device() -> str:
    """
    Get optimal device for PyTorch
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    
    return 'cpu'

