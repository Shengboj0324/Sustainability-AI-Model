"""
Master Data Preparation Script

CRITICAL: Orchestrate complete data preparation pipeline
- Download all datasets
- Clean and validate
- Augment
- Create train/val/test splits
- Generate final reports
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import List
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "data"


class DataPreparationPipeline:
    """Master data preparation pipeline"""
    
    def __init__(self):
        self.start_time = time.time()
        self.completed_steps = []
        self.failed_steps = []
    
    def run_script(self, script_name: str, description: str) -> bool:
        """Run a data preparation script"""
        logger.info("=" * 80)
        logger.info(f"STEP: {description}")
        logger.info("=" * 80)
        
        script_path = SCRIPTS_DIR / script_name
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            self.failed_steps.append((description, "Script not found"))
            return False
        
        try:
            # Run script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Log output
            if result.stdout:
                logger.info(result.stdout)
            
            if result.returncode == 0:
                logger.info(f"✅ {description} - COMPLETE")
                self.completed_steps.append(description)
                return True
            else:
                logger.error(f"❌ {description} - FAILED")
                if result.stderr:
                    logger.error(result.stderr)
                self.failed_steps.append((description, result.stderr))
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description} - TIMEOUT")
            self.failed_steps.append((description, "Timeout after 1 hour"))
            return False
        except Exception as e:
            logger.error(f"❌ {description} - ERROR: {e}")
            self.failed_steps.append((description, str(e)))
            return False
    
    def run_phase_1_download(self) -> bool:
        """Phase 1: Download all datasets"""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: DATA DOWNLOAD")
        logger.info("=" * 80 + "\n")
        
        steps = [
            ("download_taco.py", "Download TACO dataset"),
            ("download_kaggle.py", "Download Kaggle datasets"),
            ("scrape_epa.py", "Scrape EPA knowledge base"),
        ]
        
        success = True
        for script, description in steps:
            if not self.run_script(script, description):
                success = False
                # Continue with other downloads even if one fails
        
        return success
    
    def run_phase_2_clean(self) -> bool:
        """Phase 2: Clean and validate datasets"""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: DATA CLEANING")
        logger.info("=" * 80 + "\n")
        
        steps = [
            ("clean_images.py", "Clean vision datasets"),
        ]
        
        success = True
        for script, description in steps:
            if not self.run_script(script, description):
                success = False
        
        return success
    
    def run_phase_3_augment(self) -> bool:
        """Phase 3: Data augmentation"""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: DATA AUGMENTATION")
        logger.info("=" * 80 + "\n")
        
        steps = [
            ("augment_images.py", "Augment vision datasets"),
        ]
        
        success = True
        for script, description in steps:
            if not self.run_script(script, description):
                success = False
        
        return success
    
    def run_phase_4_validate(self) -> bool:
        """Phase 4: Final validation"""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4: DATA VALIDATION")
        logger.info("=" * 80 + "\n")
        
        steps = [
            ("validate_datasets.py", "Validate all datasets"),
        ]
        
        success = True
        for script, description in steps:
            if not self.run_script(script, description):
                success = False
        
        return success
    
    def generate_final_report(self):
        """Generate final preparation report"""
        elapsed_time = time.time() - self.start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        
        logger.info("\n" + "=" * 80)
        logger.info("FINAL REPORT")
        logger.info("=" * 80)
        logger.info(f"Total time: {hours}h {minutes}m")
        logger.info(f"Completed steps: {len(self.completed_steps)}")
        logger.info(f"Failed steps: {len(self.failed_steps)}")
        logger.info("")
        
        if self.completed_steps:
            logger.info("✅ COMPLETED:")
            for step in self.completed_steps:
                logger.info(f"  - {step}")
            logger.info("")
        
        if self.failed_steps:
            logger.info("❌ FAILED:")
            for step, error in self.failed_steps:
                logger.info(f"  - {step}: {error}")
            logger.info("")
        
        if not self.failed_steps:
            logger.info("🎉 ALL PHASES COMPLETE - DATASET READY FOR TRAINING!")
        else:
            logger.info("⚠️  SOME STEPS FAILED - REVIEW ERRORS ABOVE")
        
        logger.info("=" * 80)
    
    def run_all(self):
        """Run complete pipeline"""
        logger.info("=" * 80)
        logger.info("RELEAF AI - MASTER DATA PREPARATION PIPELINE")
        logger.info("=" * 80)
        logger.info("")
        
        # Run all phases
        self.run_phase_1_download()
        self.run_phase_2_clean()
        self.run_phase_3_augment()
        self.run_phase_4_validate()
        
        # Generate report
        self.generate_final_report()


def main():
    """Main function"""
    pipeline = DataPreparationPipeline()
    pipeline.run_all()


if __name__ == "__main__":
    main()

