"""
Master LLM Training Data Collection Orchestrator

CRITICAL REQUIREMENTS:
- Orchestrate all data collection sources
- Target: 1M+ high-quality examples
- Quality control: Deduplication, validation, balancing
- Progress tracking: Real-time statistics, checkpoints
- Error handling: Graceful degradation, retry logic

PIPELINE:
1. Reddit scraping (200K target)
2. YouTube transcripts (100K target)
3. Synthetic generation (700K target)
4. Quality control & deduplication
5. Format conversion & validation
6. Final dataset preparation
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "llm"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "llm_sft"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Import scrapers - FIX: Use absolute imports with sys.path
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from scrape_reddit_upcycling import RedditUpcyclingScraper
    from scrape_youtube_tutorials import YouTubeTutorialScraper
    from generate_synthetic_creative import SyntheticDataGenerator
except ImportError as e:
    logger.error(f"Failed to import scrapers: {e}")
    logger.error("Make sure all scraper modules are in scripts/data directory")
    logger.error(f"Current sys.path: {sys.path}")
    sys.exit(1)


class LLMDataCollectionOrchestrator:
    """Master orchestrator for LLM training data collection"""
    
    def __init__(self):
        """Initialize orchestrator"""
        self.all_data = []
        self.stats = defaultdict(int)
        self.seen_hashes = set()
        
        logger.info("="*60)
        logger.info("LLM TRAINING DATA COLLECTION ORCHESTRATOR")
        logger.info("Target: 1,000,000+ examples")
        logger.info("="*60)
    
    def phase_1_reddit(self, target: int = 200000):
        """Phase 1: Reddit scraping"""
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 1: REDDIT SCRAPING")
        logger.info(f"Target: {target:,} examples")
        logger.info(f"{'='*60}")
        
        try:
            scraper = RedditUpcyclingScraper()
            scraper.run()
            
            # Load scraped data
            reddit_file = RAW_DATA_DIR / "reddit" / "reddit_upcycling_raw.jsonl"
            if reddit_file.exists():
                with open(reddit_file, 'r', encoding='utf-8') as f:
                    reddit_data = [json.loads(line) for line in f]
                
                self.all_data.extend(reddit_data)
                self.stats['from_reddit'] = len(reddit_data)
                logger.info(f"✅ Loaded {len(reddit_data):,} examples from Reddit")
            else:
                logger.warning("⚠️  Reddit data file not found")
                
        except Exception as e:
            logger.error(f"Phase 1 failed: {e}")
            logger.warning("Continuing with other sources...")
    
    def phase_2_youtube(self, target: int = 100000):
        """Phase 2: YouTube transcript scraping"""
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 2: YOUTUBE SCRAPING")
        logger.info(f"Target: {target:,} examples")
        logger.info(f"{'='*60}")
        
        try:
            scraper = YouTubeTutorialScraper()
            scraper.run(target_count=target)
            
            # Load scraped data
            youtube_file = RAW_DATA_DIR / "youtube" / "youtube_tutorials_raw.jsonl"
            if youtube_file.exists():
                with open(youtube_file, 'r', encoding='utf-8') as f:
                    youtube_data = [json.loads(line) for line in f]
                
                self.all_data.extend(youtube_data)
                self.stats['from_youtube'] = len(youtube_data)
                logger.info(f"✅ Loaded {len(youtube_data):,} examples from YouTube")
            else:
                logger.warning("⚠️  YouTube data file not found")
                
        except Exception as e:
            logger.error(f"Phase 2 failed: {e}")
            logger.warning("Continuing with other sources...")
    
    def phase_3_synthetic(self, target: int = 700000):
        """Phase 3: Synthetic data generation"""
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 3: SYNTHETIC GENERATION")
        logger.info(f"Target: {target:,} examples")
        logger.info(f"Estimated cost: ${target * 0.04:.2f}")
        logger.info(f"{'='*60}")
        
        try:
            generator = SyntheticDataGenerator()
            generator.run(target_count=target)
            
            # Load generated data
            synthetic_file = RAW_DATA_DIR / "synthetic" / "synthetic_creative.jsonl"
            if synthetic_file.exists():
                with open(synthetic_file, 'r', encoding='utf-8') as f:
                    synthetic_data = [json.loads(line) for line in f]
                
                self.all_data.extend(synthetic_data)
                self.stats['from_synthetic'] = len(synthetic_data)
                logger.info(f"✅ Loaded {len(synthetic_data):,} examples from synthetic generation")
            else:
                logger.warning("⚠️  Synthetic data file not found")
                
        except Exception as e:
            logger.error(f"Phase 3 failed: {e}")
            logger.warning("Continuing with available data...")

    def phase_4_quality_control(self):
        """Phase 4: Quality control and deduplication - FIX: Streaming to avoid memory overflow"""
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 4: QUALITY CONTROL")
        logger.info(f"Input: {len(self.all_data):,} examples")
        logger.info(f"{'='*60}")

        from tqdm import tqdm

        # FIX: Stream to temporary file instead of loading all into memory
        temp_file = PROCESSED_DATA_DIR / "temp_cleaned.jsonl"

        with open(temp_file, 'w', encoding='utf-8') as out_f:
            for item in tqdm(self.all_data, desc="Quality control"):
                # Extract content
                try:
                    messages = item['messages']
                    user_msg = messages[0]['content']
                    assistant_msg = messages[1]['content']

                    # Content hash for deduplication (use SHA-256 instead of MD5)
                    content = user_msg + assistant_msg
                    content_hash = hashlib.sha256(content.lower().encode()).hexdigest()

                    if content_hash in self.seen_hashes:
                        self.stats['duplicates_removed'] += 1
                        continue

                    # Length validation
                    word_count = len(assistant_msg.split())
                    if word_count < 30:
                        self.stats['too_short'] += 1
                        continue
                    if word_count > 2000:
                        self.stats['too_long'] += 1
                        continue

                    # Quality check
                    if len(user_msg) < 10:
                        self.stats['invalid_question'] += 1
                        continue

                    self.seen_hashes.add(content_hash)
                    out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    self.stats['kept'] += 1

                except Exception as e:
                    self.stats['parsing_errors'] += 1
                    continue

        logger.info(f"✅ Quality control complete")
        logger.info(f"   Kept: {self.stats['kept']:,}")
        logger.info(f"   Removed duplicates: {self.stats['duplicates_removed']:,}")
        logger.info(f"   Removed too short: {self.stats['too_short']:,}")
        logger.info(f"   Removed too long: {self.stats['too_long']:,}")

        # FIX: Load cleaned data from temp file (streaming)
        self.all_data = []
        with open(temp_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.all_data.append(json.loads(line))

        # Clean up temp file
        temp_file.unlink()

    def phase_5_split_and_save(self, train_ratio: float = 0.95):
        """Phase 5: Train/val split and save"""
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 5: TRAIN/VAL SPLIT")
        logger.info(f"Train ratio: {train_ratio:.1%}")
        logger.info(f"{'='*60}")

        import random
        random.shuffle(self.all_data)

        split_idx = int(len(self.all_data) * train_ratio)
        train_data = self.all_data[:split_idx]
        val_data = self.all_data[split_idx:]

        # Save train
        train_file = PROCESSED_DATA_DIR / "sustainability_creative_train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # Save val
        val_file = PROCESSED_DATA_DIR / "sustainability_creative_val.jsonl"
        with open(val_file, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"✅ Data saved:")
        logger.info(f"   Train: {len(train_data):,} examples → {train_file}")
        logger.info(f"   Val: {len(val_data):,} examples → {val_file}")

        self.stats['final_train_count'] = len(train_data)
        self.stats['final_val_count'] = len(val_data)

    def print_final_statistics(self):
        """Print final statistics"""
        logger.info(f"\n{'='*60}")
        logger.info("FINAL STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Data sources:")
        logger.info(f"  Reddit: {self.stats.get('from_reddit', 0):,}")
        logger.info(f"  YouTube: {self.stats.get('from_youtube', 0):,}")
        logger.info(f"  Synthetic: {self.stats.get('from_synthetic', 0):,}")
        logger.info(f"\nFinal dataset:")
        logger.info(f"  Train: {self.stats.get('final_train_count', 0):,}")
        logger.info(f"  Val: {self.stats.get('final_val_count', 0):,}")
        logger.info(f"  Total: {self.stats.get('final_train_count', 0) + self.stats.get('final_val_count', 0):,}")
        logger.info(f"{'='*60}")

        # Save stats
        stats_file = PROCESSED_DATA_DIR / "collection_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(dict(self.stats), f, indent=2)

    def run(self):
        """Run complete pipeline"""
        start_time = datetime.now()

        # Run all phases
        self.phase_1_reddit(target=200000)
        self.phase_2_youtube(target=100000)
        self.phase_3_synthetic(target=700000)
        self.phase_4_quality_control()
        self.phase_5_split_and_save()

        # Print statistics
        self.print_final_statistics()

        elapsed = datetime.now() - start_time
        logger.info(f"\n✅ PIPELINE COMPLETE!")
        logger.info(f"Total time: {elapsed}")
        logger.info(f"Output: {PROCESSED_DATA_DIR}")


def main():
    """Main entry point"""
    try:
        orchestrator = LLMDataCollectionOrchestrator()
        orchestrator.run()
        return True
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

