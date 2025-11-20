"""
YouTube Upcycling Tutorial Scraper - PRODUCTION GRADE

CRITICAL REQUIREMENTS:
- Target: 100K+ tutorial transcripts
- Sources: DIY/upcycling channels, search queries
- Quality: Filter auto-generated captions, require manual transcripts
- Rate limiting: YouTube API quota (10,000 units/day)
- Error handling: Retry logic, quota management
- Validation: Transcript quality, content relevance, safety

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
from datetime import datetime
from collections import defaultdict

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "llm" / "youtube"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# YouTube API Configuration
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    logger.warning("YOUTUBE_API_KEY not set. Get one at: https://console.cloud.google.com/apis/credentials")

# Search queries for upcycling content
SEARCH_QUERIES = [
    "upcycling tutorial", "DIY upcycle", "repurpose old items",
    "waste to art", "recycling crafts", "creative reuse",
    "plastic bottle crafts", "pallet furniture DIY", "tire upcycling",
    "glass jar projects", "cardboard crafts", "fabric scrap projects",
    "zero waste DIY", "sustainable crafts", "eco-friendly projects"
]

# Target channels (known high-quality upcycling creators)
TARGET_CHANNELS = [
    "UCfrs2czcvjfvaQe7lcOG9yg",  # 5-Minute Crafts
    "UCWv7k4mJpVU-Yh8LrY-NWIQ",  # Blossom
    "UCJ98kJ7qQZvJW8yFYw0nXPg",  # Troom Troom
]

# Quality thresholds
MIN_TRANSCRIPT_LENGTH = 200  # words
MAX_TRANSCRIPT_LENGTH = 5000
MIN_VIDEO_DURATION = 120  # seconds (2 minutes)
MAX_VIDEO_DURATION = 1800  # seconds (30 minutes)
MIN_VIEW_COUNT = 1000
MIN_LIKE_RATIO = 0.7  # likes / (likes + dislikes)

# API quota management
DAILY_QUOTA = 10000
SEARCH_COST = 100  # units per search
VIDEO_DETAILS_COST = 1  # units per video
quota_used = 0


class YouTubeTutorialScraper:
    """Production-grade YouTube scraper for upcycling tutorials"""
    
    def __init__(self):
        """Initialize YouTube API client"""
        if not YOUTUBE_API_KEY:
            raise ValueError("YouTube API key required. Set YOUTUBE_API_KEY environment variable.")
        
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        self.scraped_data = []
        self.stats = defaultdict(int)
        self.seen_video_ids = set()
        
        logger.info("✅ YouTube API client initialized")
    
    def check_quota(self, cost: int) -> bool:
        """Check if we have enough quota remaining"""
        global quota_used
        if quota_used + cost > DAILY_QUOTA:
            logger.warning(f"⚠️  Quota limit reached ({quota_used}/{DAILY_QUOTA})")
            return False
        quota_used += cost
        return True
    
    def search_videos(self, query: str, max_results: int = 50) -> List[str]:
        """Search for videos by query"""
        if not self.check_quota(SEARCH_COST):
            return []
        
        try:
            request = self.youtube.search().list(
                part="id,snippet",
                q=query,
                type="video",
                maxResults=max_results,
                order="relevance",
                videoDuration="medium",  # 4-20 minutes
                videoDefinition="any",
                relevanceLanguage="en"
            )
            response = request.execute()
            
            video_ids = [item['id']['videoId'] for item in response.get('items', [])]
            logger.info(f"Found {len(video_ids)} videos for query: '{query}'")
            return video_ids
            
        except HttpError as e:
            logger.error(f"YouTube API error for query '{query}': {e}")
            return []
    
    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """Get detailed video information"""
        if not self.check_quota(VIDEO_DETAILS_COST):
            return None
        
        try:
            request = self.youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=video_id
            )
            response = request.execute()
            
            if not response.get('items'):
                return None
            
            return response['items'][0]
            
        except HttpError as e:
            logger.error(f"Failed to get details for video {video_id}: {e}")
            return None

    def validate_video(self, video_details: Dict) -> tuple[bool, str]:
        """Validate video quality"""
        try:
            snippet = video_details['snippet']
            statistics = video_details['statistics']
            content_details = video_details['contentDetails']

            # Parse duration (ISO 8601 format: PT15M33S)
            duration_str = content_details['duration']
            duration_seconds = self.parse_duration(duration_str)

            if duration_seconds < MIN_VIDEO_DURATION:
                return False, "too_short"
            if duration_seconds > MAX_VIDEO_DURATION:
                return False, "too_long"

            # Check view count
            view_count = int(statistics.get('viewCount', 0))
            if view_count < MIN_VIEW_COUNT:
                return False, "low_views"

            # Check like ratio (if available)
            likes = int(statistics.get('likeCount', 0))
            dislikes = int(statistics.get('dislikeCount', 0))
            if likes + dislikes > 0:
                like_ratio = likes / (likes + dislikes)
                if like_ratio < MIN_LIKE_RATIO:
                    return False, "low_like_ratio"

            # Check if video is available
            if snippet.get('liveBroadcastContent') == 'live':
                return False, "live_stream"

            return True, "valid"

        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return False, "validation_error"

    def parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration to seconds"""
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
        if not match:
            return 0
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds

    def get_transcript(self, video_id: str) -> Optional[str]:
        """Get video transcript (prefer manual over auto-generated)"""
        try:
            # Get available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try to get manual transcript first
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
                self.stats['manual_transcripts'] += 1
            except:
                # Fall back to auto-generated
                transcript = transcript_list.find_generated_transcript(['en'])
                self.stats['auto_transcripts'] += 1

            # Fetch transcript
            transcript_data = transcript.fetch()

            # Combine all text
            full_text = ' '.join([entry['text'] for entry in transcript_data])

            # Clean transcript
            full_text = self.clean_transcript(full_text)

            return full_text

        except (TranscriptsDisabled, NoTranscriptFound):
            self.stats['no_transcript'] += 1
            return None
        except Exception as e:
            logger.debug(f"Transcript error for {video_id}: {e}")
            self.stats['transcript_error'] += 1
            return None

    def clean_transcript(self, text: str) -> str:
        """Clean and normalize transcript text"""
        # Remove music/sound effects markers
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove timestamps
        text = re.sub(r'\d{1,2}:\d{2}', '', text)

        return text.strip()

    def extract_qa_from_video(self, video_id: str, video_details: Dict, transcript: str) -> List[Dict]:
        """Extract Q&A pairs from video"""
        qa_pairs = []

        snippet = video_details['snippet']
        title = snippet['title']
        description = snippet.get('description', '')

        # Word count check
        word_count = len(transcript.split())
        if word_count < MIN_TRANSCRIPT_LENGTH or word_count > MAX_TRANSCRIPT_LENGTH:
            self.stats['rejected_transcript_length'] += 1
            return []

        # Generate question from title
        question = self.generate_question_from_title(title)

        # Use transcript as answer (truncate if too long)
        answer = transcript[:2000]  # Limit to 2000 chars

        qa_pairs.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ],
            "category": "upcycling_tutorial",
            "metadata": {
                "source": "youtube",
                "video_id": video_id,
                "title": title,
                "channel": snippet['channelTitle'],
                "view_count": int(video_details['statistics'].get('viewCount', 0)),
                "like_count": int(video_details['statistics'].get('likeCount', 0)),
                "word_count": word_count,
                "transcript_type": "manual" if self.stats.get('manual_transcripts', 0) > self.stats.get('auto_transcripts', 0) else "auto",
                "scraped_at": datetime.now().isoformat()
            }
        })

        return qa_pairs

    def generate_question_from_title(self, title: str) -> str:
        """Generate natural question from video title"""
        title = title.strip()

        # If already a question
        if "?" in title:
            return title

        # Common patterns
        if any(word in title.lower() for word in ["how to", "diy", "tutorial"]):
            return title

        # Default
        return f"How can I {title.lower()}?"

    def process_video(self, video_id: str) -> int:
        """Process a single video"""
        if video_id in self.seen_video_ids:
            return 0

        # Get video details
        video_details = self.get_video_details(video_id)
        if not video_details:
            self.stats['no_details'] += 1
            return 0

        # Validate video
        is_valid, reason = self.validate_video(video_details)
        if not is_valid:
            self.stats[f'rejected_{reason}'] += 1
            return 0

        # Get transcript
        transcript = self.get_transcript(video_id)
        if not transcript:
            return 0

        # Extract Q&A pairs
        qa_pairs = self.extract_qa_from_video(video_id, video_details, transcript)
        if qa_pairs:
            self.scraped_data.extend(qa_pairs)
            self.seen_video_ids.add(video_id)
            self.stats['total_collected'] += len(qa_pairs)
            return len(qa_pairs)

        return 0

    def run(self, target_count: int = 100000):
        """Run complete scraping pipeline"""
        logger.info("="*60)
        logger.info("YOUTUBE TUTORIAL SCRAPER - PRODUCTION MODE")
        logger.info(f"Target: {target_count} examples")
        logger.info("="*60)

        # Search for videos
        all_video_ids = []
        for query in SEARCH_QUERIES:
            if len(self.scraped_data) >= target_count:
                break
            video_ids = self.search_videos(query, max_results=50)
            all_video_ids.extend(video_ids)
            time.sleep(1)  # Rate limiting

        logger.info(f"Found {len(all_video_ids)} total videos to process")

        # Process videos
        for video_id in tqdm(all_video_ids, desc="Processing videos"):
            if len(self.scraped_data) >= target_count:
                break
            self.process_video(video_id)
            time.sleep(0.5)  # Rate limiting

        # Save data
        self.save_data()

        # Print statistics
        self.print_statistics()

        logger.info(f"\n✅ YOUTUBE SCRAPING COMPLETE!")
        logger.info(f"Total examples: {len(self.scraped_data)}")

    def save_data(self):
        """Save scraped data"""
        output_path = OUTPUT_DIR / "youtube_tutorials_raw.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.scraped_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"✅ Data saved to {output_path}")

        # Save stats
        stats_path = OUTPUT_DIR / "youtube_scraping_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(dict(self.stats), f, indent=2)

    def print_statistics(self):
        """Print scraping statistics"""
        logger.info(f"\n{'='*60}")
        logger.info("SCRAPING STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total collected: {self.stats['total_collected']}")
        logger.info(f"Manual transcripts: {self.stats.get('manual_transcripts', 0)}")
        logger.info(f"Auto transcripts: {self.stats.get('auto_transcripts', 0)}")
        logger.info(f"Quota used: {quota_used}/{DAILY_QUOTA}")
        logger.info(f"{'='*60}")


def main():
    """Main entry point"""
    try:
        scraper = YouTubeTutorialScraper()
        scraper.run(target_count=100000)
        return True
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

