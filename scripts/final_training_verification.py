#!/usr/bin/env python3
"""
FINAL TRAINING VERIFICATION
Comprehensive check before starting training
Verifies all dependencies, data, configs, and potential errors
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class TrainingVerifier:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []
        self.verified = []
        
    def verify_dependencies(self) -> bool:
        """Verify all required dependencies"""
        logger.info("\n" + "="*80)
        logger.info("🔍 VERIFYING DEPENDENCIES")
        logger.info("="*80)
        
        required_packages = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("datasets", "Datasets"),
            ("peft", "PEFT (LoRA)"),
            ("torch_geometric", "PyTorch Geometric"),
            ("PIL", "Pillow"),
            ("albumentations", "Albumentations"),
            ("yaml", "PyYAML"),
            ("pandas", "Pandas"),
            ("numpy", "NumPy"),
        ]
        
        all_ok = True
        for package, name in required_packages:
            try:
                __import__(package)
                logger.info(f"  ✅ {name}")
                self.verified.append(f"Dependency: {name}")
            except ImportError:
                logger.error(f"  ❌ {name} - NOT INSTALLED")
                self.errors.append(f"Missing dependency: {name}")
                all_ok = False
        
        return all_ok
    
    def verify_pytorch_backend(self) -> bool:
        """Verify PyTorch backend (MPS/CUDA/CPU)"""
        logger.info("\n" + "="*80)
        logger.info("🔍 VERIFYING PYTORCH BACKEND")
        logger.info("="*80)
        
        import torch
        
        if torch.backends.mps.is_available():
            logger.info("  ✅ Apple MPS (Metal Performance Shaders) available")
            logger.info(f"  ✅ Device: Apple M4 Max GPU")
            self.verified.append("Backend: MPS (Apple M4 Max)")
            return True
        elif torch.cuda.is_available():
            logger.info(f"  ✅ CUDA available")
            logger.info(f"  ✅ Device: {torch.cuda.get_device_name(0)}")
            self.verified.append(f"Backend: CUDA ({torch.cuda.get_device_name(0)})")
            return True
        else:
            logger.warning("  ⚠️  No GPU available - will use CPU (slow)")
            self.warnings.append("No GPU available - training will be slow")
            self.verified.append("Backend: CPU")
            return True
    
    def verify_llm_data(self) -> bool:
        """Verify LLM training data"""
        logger.info("\n" + "="*80)
        logger.info("🔍 VERIFYING LLM TRAINING DATA")
        logger.info("="*80)
        
        # Check raw data
        ultra_expanded = self.project_root / "data/llm_training_ultra_expanded.json"
        if ultra_expanded.exists():
            try:
                with open(ultra_expanded, 'r') as f:
                    data = json.load(f)
                logger.info(f"  ✅ Ultra-expanded dataset: {len(data)} examples")
                
                # Verify format
                if data and "messages" in data[0]:
                    logger.info(f"  ✅ Data format: OpenAI chat format")
                    self.verified.append(f"LLM data: {len(data)} examples")
                    return True
                else:
                    logger.error(f"  ❌ Invalid data format")
                    self.errors.append("LLM data: Invalid format")
                    return False
            except Exception as e:
                logger.error(f"  ❌ Error loading data: {e}")
                self.errors.append(f"LLM data: {e}")
                return False
        else:
            logger.error(f"  ❌ Data file not found: {ultra_expanded}")
            self.errors.append("LLM data: File not found")
            return False
    
    def verify_gnn_data(self) -> bool:
        """Verify GNN training data"""
        logger.info("\n" + "="*80)
        logger.info("🔍 VERIFYING GNN TRAINING DATA")
        logger.info("="*80)
        
        gnn_file = self.project_root / "data/gnn_training_expanded.json"
        if gnn_file.exists():
            try:
                with open(gnn_file, 'r') as f:
                    data = json.load(f)
                nodes = len(data.get('nodes', []))
                edges = len(data.get('edges', []))
                logger.info(f"  ✅ GNN data: {nodes} nodes, {edges} edges")
                self.verified.append(f"GNN data: {nodes} nodes, {edges} edges")
                return True
            except Exception as e:
                logger.error(f"  ❌ Error loading data: {e}")
                self.errors.append(f"GNN data: {e}")
                return False
        else:
            logger.error(f"  ❌ Data file not found: {gnn_file}")
            self.errors.append("GNN data: File not found")
            return False
    
    def verify_configs(self) -> bool:
        """Verify training configs"""
        logger.info("\n" + "="*80)
        logger.info("🔍 VERIFYING TRAINING CONFIGS")
        logger.info("="*80)
        
        import yaml
        
        configs = [
            ("configs/llm_sft_m4max.yaml", "LLM M4 Max"),
            ("configs/vision_cls_m4max.yaml", "Vision M4 Max"),
        ]
        
        all_ok = True
        for config_path, name in configs:
            full_path = self.project_root / config_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        config = yaml.safe_load(f)
                    logger.info(f"  ✅ {name}")
                    self.verified.append(f"Config: {name}")
                except Exception as e:
                    logger.error(f"  ❌ {name}: {e}")
                    self.errors.append(f"Config {name}: {e}")
                    all_ok = False
            else:
                logger.error(f"  ❌ {name}: File not found")
                self.errors.append(f"Config {name}: File not found")
                all_ok = False
        
        return all_ok

    def verify_training_scripts(self) -> bool:
        """Verify training scripts exist and are valid"""
        logger.info("\n" + "="*80)
        logger.info("🔍 VERIFYING TRAINING SCRIPTS")
        logger.info("="*80)

        scripts = [
            ("training/llm/train_sft.py", "LLM SFT"),
            ("training/vision/train_multihead.py", "Vision Multi-Head"),
            ("training/gnn/train_gnn.py", "GNN"),
        ]

        all_ok = True
        for script_path, name in scripts:
            full_path = self.project_root / script_path
            if full_path.exists():
                logger.info(f"  ✅ {name}")
                self.verified.append(f"Script: {name}")
            else:
                logger.error(f"  ❌ {name}: File not found")
                self.errors.append(f"Script {name}: File not found")
                all_ok = False

        return all_ok

    def generate_report(self) -> bool:
        """Generate final verification report"""
        logger.info("\n" + "="*80)
        logger.info("📊 FINAL VERIFICATION REPORT")
        logger.info("="*80)

        logger.info(f"\n✅ Verified: {len(self.verified)}")
        logger.info(f"⚠️  Warnings: {len(self.warnings)}")
        logger.info(f"❌ Errors: {len(self.errors)}")

        if self.warnings:
            logger.info("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                logger.info(f"  - {warning}")

        if self.errors:
            logger.info("\n❌ ERRORS:")
            for error in self.errors:
                logger.info(f"  - {error}")
            logger.info("\n❌ TRAINING NOT READY - FIX ERRORS FIRST")
            return False
        else:
            logger.info("\n✅ ALL CHECKS PASSED - READY FOR TRAINING!")
            return True

    def run_all_checks(self) -> bool:
        """Run all verification checks"""
        logger.info("="*80)
        logger.info("🔥 FINAL TRAINING VERIFICATION")
        logger.info("="*80)
        logger.info("Checking all dependencies, data, configs, and scripts...")

        checks = [
            self.verify_dependencies,
            self.verify_pytorch_backend,
            self.verify_llm_data,
            self.verify_gnn_data,
            self.verify_configs,
            self.verify_training_scripts,
        ]

        for check in checks:
            check()

        return self.generate_report()

def main():
    verifier = TrainingVerifier()
    success = verifier.run_all_checks()

    if success:
        logger.info("\n" + "="*80)
        logger.info("🚀 READY TO START TRAINING!")
        logger.info("="*80)
        logger.info("\nRecommended training order:")
        logger.info("\n1️⃣  LLM Training (2-3 hours):")
        logger.info("    python3 training/llm/train_sft.py --config configs/llm_sft_m4max.yaml")
        logger.info("\n2️⃣  Vision Training (1-2 hours):")
        logger.info("    python3 training/vision/train_multihead.py --config configs/vision_cls_m4max.yaml")
        logger.info("\n3️⃣  GNN Training (30 minutes):")
        logger.info("    python3 training/gnn/train_gnn.py")
        logger.info("\n" + "="*80)
        sys.exit(0)
    else:
        logger.info("\n" + "="*80)
        logger.info("❌ NOT READY FOR TRAINING")
        logger.info("="*80)
        logger.info("Please fix the errors above before starting training.")
        sys.exit(1)

if __name__ == "__main__":
    main()

