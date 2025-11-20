"""
Reddit Upcycling Data Scraper - PRODUCTION GRADE

CRITICAL REQUIREMENTS:
- Target: 200K+ high-quality upcycling examples
- Sources: r/upcycling, r/ZeroWaste, r/DIY, r/crafts, r/somethingimade
- Quality: Filter low-effort posts, require detailed descriptions
- Rate limiting: Respect Reddit API limits (60 req/min)
- Error handling: Retry logic, graceful degradation
- Validation: Content length, creativity score, safety checks

OUTPUT FORMAT:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "category": "...", "metadata": {...}}
"""

import os
import sys
import time
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

import praw
from prawcore.exceptions import ResponseException, RequestException
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "llm" / "reddit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reddit API Configuration
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = "ReleAF-AI-Data-Collector/1.0"

# Target subreddits with priorities
SUBREDDITS = {
    "upcycling": {"priority": "CRITICAL", "target": 50000, "min_score": 10},
    "ZeroWaste": {"priority": "CRITICAL", "target": 40000, "min_score": 15},
    "DIY": {"priority": "HIGH", "target": 30000, "min_score": 20},
    "crafts": {"priority": "HIGH", "target": 25000, "min_score": 15},
    "somethingimade": {"priority": "MEDIUM", "target": 20000, "min_score": 25},
    "Repurpose": {"priority": "MEDIUM", "target": 15000, "min_score": 10},
    "ThriftStoreHauls": {"priority": "LOW", "target": 10000, "min_score": 15},
    "Frugal": {"priority": "LOW", "target": 10000, "min_score": 20},
}

# Quality thresholds
MIN_TITLE_LENGTH = 10
MIN_BODY_LENGTH = 50
MIN_COMMENT_LENGTH = 30
MAX_BODY_LENGTH = 5000
MIN_CREATIVITY_SCORE = 0.3
BANNED_KEYWORDS = ["buy", "sell", "spam", "advertisement", "promo"]

# Rate limiting
REQUESTS_PER_MINUTE = 55  # Conservative (Reddit allows 60)
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


class RedditUpcyclingScraper:
    """Production-grade Reddit scraper for upcycling data"""
    
    def __init__(self):
        """Initialize Reddit API client"""
        if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
            raise ValueError(
                "Reddit API credentials not found. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.\n"
                "Get credentials at: https://www.reddit.com/prefs/apps"
            )
        
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        
        self.scraped_data = []
        self.stats = defaultdict(int)
        self.seen_ids = set()
        
        logger.info("✅ Reddit API client initialized")
    
    def validate_post(self, post) -> tuple[bool, str]:
        """Validate post quality with strict criteria"""
        try:
            # Check if already seen
            if post.id in self.seen_ids:
                return False, "duplicate"
            
            # Check title length
            if len(post.title) < MIN_TITLE_LENGTH:
                return False, "title_too_short"
            
            # Check body length
            body_text = post.selftext or ""
            if len(body_text) < MIN_BODY_LENGTH:
                return False, "body_too_short"
            
            if len(body_text) > MAX_BODY_LENGTH:
                return False, "body_too_long"
            
            # Check for banned keywords (spam detection)
            combined_text = (post.title + " " + body_text).lower()
            if any(keyword in combined_text for keyword in BANNED_KEYWORDS):
                return False, "spam_detected"
            
            # Check if removed/deleted
            if post.removed_by_category or post.author is None:
                return False, "removed_or_deleted"
            
            # Check NSFW (safety)
            if post.over_18:
                return False, "nsfw"
            
            return True, "valid"
            
        except Exception as e:
            logger.warning(f"Validation error for post {post.id}: {e}")
            return False, "validation_error"

    def calculate_creativity_score(self, post) -> float:
        """Calculate creativity score based on content analysis"""
        score = 0.0
        text = (post.title + " " + (post.selftext or "")).lower()

        # Positive indicators (creative upcycling)
        creative_keywords = [
            "transform", "repurpose", "upcycle", "diy", "handmade", "creative",
            "recycle", "reuse", "made from", "turned into", "converted",
            "art", "craft", "project", "build", "create", "design"
        ]
        score += sum(0.05 for kw in creative_keywords if kw in text)

        # Material mentions (shows specificity)
        materials = [
            "plastic", "glass", "wood", "metal", "fabric", "cardboard",
            "bottle", "jar", "pallet", "tire", "can", "paper"
        ]
        score += sum(0.03 for mat in materials if mat in text)

        # Detailed instructions (high value)
        if "step" in text or "how to" in text or "tutorial" in text:
            score += 0.15

        # Images (visual proof)
        if post.url and any(ext in post.url for ext in ['.jpg', '.png', '.gif']):
            score += 0.1

        # Engagement (community validation)
        score += min(post.score / 1000.0, 0.2)  # Cap at 0.2
        score += min(post.num_comments / 100.0, 0.1)  # Cap at 0.1

        return min(score, 1.0)  # Cap at 1.0

    def extract_qa_pairs(self, post) -> List[Dict]:
        """Extract Q&A pairs from post and comments"""
        qa_pairs = []

        # Main post as Q&A
        if post.selftext and len(post.selftext) >= MIN_BODY_LENGTH:
            # Question: "How can I upcycle [item from title]?"
            question = self.generate_question_from_title(post.title)
            answer = post.selftext[:1000]  # Limit answer length

            qa_pairs.append({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "category": "upcycling_ideas",
                "metadata": {
                    "source": "reddit",
                    "subreddit": str(post.subreddit),
                    "post_id": post.id,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "creativity_score": self.calculate_creativity_score(post),
                    "scraped_at": datetime.now().isoformat()
                }
            })

        # Extract from top comments (high-quality responses)
        try:
            post.comments.replace_more(limit=0)  # Don't expand "more comments"
            for comment in post.comments[:10]:  # Top 10 comments
                if len(comment.body) >= MIN_COMMENT_LENGTH and comment.score >= 5:
                    # Use post title as question, comment as answer
                    question = post.title
                    if not question.endswith("?"):
                        question = f"How can I {question.lower()}?"

                    qa_pairs.append({
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": comment.body[:800]}
                        ],
                        "category": "upcycling_ideas",
                        "metadata": {
                            "source": "reddit_comment",
                            "subreddit": str(post.subreddit),
                            "post_id": post.id,
                            "comment_id": comment.id,
                            "score": comment.score,
                            "scraped_at": datetime.now().isoformat()
                        }
                    })
        except Exception as e:
            logger.debug(f"Failed to extract comments from {post.id}: {e}")

        return qa_pairs

    def generate_question_from_title(self, title: str) -> str:
        """Generate natural question from post title"""
        title = title.strip()

        # If already a question, return as-is
        if "?" in title:
            return title

        # Common patterns
        if title.lower().startswith(("made", "turned", "transformed", "upcycled")):
            return f"How did you make this? {title}"

        # Default: ask for ideas
        return f"What are some creative upcycling ideas for {title.lower()}?"

    def scrape_subreddit(self, subreddit_name: str, config: Dict) -> int:
        """Scrape a single subreddit"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping r/{subreddit_name} (Priority: {config['priority']})")
        logger.info(f"Target: {config['target']} posts, Min score: {config['min_score']}")
        logger.info(f"{'='*60}")

        subreddit = self.reddit.subreddit(subreddit_name)
        collected = 0
        target = config['target']

        # Scrape from multiple time periods for diversity
        time_filters = ['all', 'year', 'month']

        for time_filter in time_filters:
            if collected >= target:
                break

            try:
                posts = subreddit.top(time_filter=time_filter, limit=None)

                for post in tqdm(posts, desc=f"r/{subreddit_name} ({time_filter})", total=target):
                    if collected >= target:
                        break

                    # Validate post
                    is_valid, reason = self.validate_post(post)
                    if not is_valid:
                        self.stats[f"rejected_{reason}"] += 1
                        continue

                    # Check minimum score
                    if post.score < config['min_score']:
                        self.stats["rejected_low_score"] += 1
                        continue

                    # Check creativity score
                    creativity = self.calculate_creativity_score(post)
                    if creativity < MIN_CREATIVITY_SCORE:
                        self.stats["rejected_low_creativity"] += 1
                        continue

                    # Extract Q&A pairs
                    qa_pairs = self.extract_qa_pairs(post)
                    self.scraped_data.extend(qa_pairs)
                    self.seen_ids.add(post.id)

                    collected += len(qa_pairs)
                    self.stats["total_collected"] += len(qa_pairs)
                    self.stats[f"from_{subreddit_name}"] += len(qa_pairs)

                    # Rate limiting
                    time.sleep(REQUEST_DELAY)

            except Exception as e:
                logger.error(f"Error scraping r/{subreddit_name} ({time_filter}): {e}")
                continue

        logger.info(f"✅ Collected {collected} examples from r/{subreddit_name}")
        return collected

    def save_data(self, output_file: str = "reddit_upcycling_raw.jsonl"):
        """Save scraped data to JSONL file"""
        output_path = OUTPUT_DIR / output_file

        logger.info(f"\nSaving {len(self.scraped_data)} examples to {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.scraped_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"✅ Data saved successfully")

        # Save statistics
        stats_path = OUTPUT_DIR / "reddit_scraping_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(dict(self.stats), f, indent=2)

        logger.info(f"✅ Statistics saved to {stats_path}")

    def print_statistics(self):
        """Print scraping statistics"""
        logger.info(f"\n{'='*60}")
        logger.info("SCRAPING STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total collected: {self.stats['total_collected']}")
        logger.info(f"\nBy subreddit:")
        for subreddit in SUBREDDITS.keys():
            count = self.stats.get(f"from_{subreddit}", 0)
            if count > 0:
                logger.info(f"  r/{subreddit}: {count}")

        logger.info(f"\nRejection reasons:")
        rejection_keys = [k for k in self.stats.keys() if k.startswith("rejected_")]
        for key in sorted(rejection_keys):
            logger.info(f"  {key.replace('rejected_', '')}: {self.stats[key]}")
        logger.info(f"{'='*60}")

    def run(self):
        """Run complete scraping pipeline"""
        logger.info("="*60)
        logger.info("REDDIT UPCYCLING DATA SCRAPER - PRODUCTION MODE")
        logger.info("="*60)

        # Scrape each subreddit
        for subreddit_name, config in SUBREDDITS.items():
            try:
                self.scrape_subreddit(subreddit_name, config)
            except Exception as e:
                logger.error(f"Failed to scrape r/{subreddit_name}: {e}")
                continue

        # Print statistics
        self.print_statistics()

        # Save data
        self.save_data()

        logger.info("\n✅ REDDIT SCRAPING COMPLETE!")
        logger.info(f"Total examples collected: {len(self.scraped_data)}")
        logger.info(f"Output: {OUTPUT_DIR}")


def main():
    """Main entry point"""
    try:
        scraper = RedditUpcyclingScraper()
        scraper.run()
        return True
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

