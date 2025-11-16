"""
Scrape EPA Sustainability Knowledge Base

CRITICAL: Collect authoritative sustainability knowledge for LLM fine-tuning
- EPA recycling guidelines
- Waste management best practices
- Material safety data
- Environmental regulations
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Set
from urllib.parse import urljoin, urlparse
import hashlib

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "text" / "epa"

# EPA URLs to scrape
EPA_BASE_URL = "https://www.epa.gov"
EPA_PAGES = [
    "/recycle",
    "/facts-and-figures-about-materials-waste-and-recycling",
    "/sustainable-management-food",
    "/composting-home",
    "/reducing-wasted-food-home",
    "/how-do-i-recycle-common-recyclables",
]

# Scraping parameters
MAX_PAGES = 1000
REQUEST_DELAY = 1.0  # seconds between requests
TIMEOUT = 10  # seconds


class EPAScraper:
    """EPA website scraper"""
    
    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.scraped_pages: List[Dict] = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for scraping"""
        parsed = urlparse(url)
        
        # Must be EPA domain
        if not parsed.netloc.endswith('epa.gov'):
            return False
        
        # Skip certain file types
        skip_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip'}
        if any(parsed.path.endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip already visited
        if url in self.visited_urls:
            return False
        
        return True
    
    def extract_text(self, soup: BeautifulSoup) -> str:
        """Extract main text content from page"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text from main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='main-content')
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
        
        # Clean up text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from page"""
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            
            # Remove fragments
            full_url = full_url.split('#')[0]
            
            if self.is_valid_url(full_url):
                links.append(full_url)
        
        return links
    
    def scrape_page(self, url: str) -> Dict:
        """Scrape a single page"""
        try:
            logger.info(f"Scraping: {url}")
            
            # Make request
            response = self.session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content
            title = soup.find('title').get_text() if soup.find('title') else ""
            text = self.extract_text(soup)
            links = self.extract_links(soup, url)
            
            # Create page data
            page_data = {
                "url": url,
                "title": title,
                "text": text,
                "links": links,
                "scraped_at": time.time(),
                "word_count": len(text.split())
            }
            
            # Mark as visited
            self.visited_urls.add(url)
            
            return page_data
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return None
    
    def scrape(self, start_urls: List[str], max_pages: int = MAX_PAGES):
        """Scrape EPA website starting from given URLs"""
        logger.info(f"Starting EPA scrape with {len(start_urls)} seed URLs")
        
        # Queue of URLs to scrape
        url_queue = list(start_urls)
        
        # Progress bar
        pbar = tqdm(total=max_pages, desc="Scraping pages")
        
        while url_queue and len(self.scraped_pages) < max_pages:
            # Get next URL
            url = url_queue.pop(0)
            
            # Skip if already visited
            if url in self.visited_urls:
                continue
            
            # Scrape page
            page_data = self.scrape_page(url)
            
            if page_data:
                self.scraped_pages.append(page_data)
                pbar.update(1)
                
                # Add new links to queue
                for link in page_data['links']:
                    if link not in self.visited_urls and link not in url_queue:
                        url_queue.append(link)
            
            # Rate limiting
            time.sleep(REQUEST_DELAY)
        
        pbar.close()
        logger.info(f"Scraped {len(self.scraped_pages)} pages")
    
    def save(self):
        """Save scraped data"""
        # Create output directory
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save all pages as JSON
        output_file = DATA_DIR / "epa_pages.json"
        with open(output_file, 'w') as f:
            json.dump(self.scraped_pages, f, indent=2)
        
        logger.info(f"Saved {len(self.scraped_pages)} pages to {output_file}")
        
        # Save individual text files
        text_dir = DATA_DIR / "texts"
        text_dir.mkdir(exist_ok=True)
        
        for i, page_data in enumerate(self.scraped_pages):
            # Create filename from URL
            url_hash = hashlib.md5(page_data['url'].encode()).hexdigest()[:8]
            text_file = text_dir / f"epa_{i:04d}_{url_hash}.txt"
            
            with open(text_file, 'w') as f:
                f.write(f"Title: {page_data['title']}\n")
                f.write(f"URL: {page_data['url']}\n")
                f.write(f"Word Count: {page_data['word_count']}\n")
                f.write("\n" + "=" * 80 + "\n\n")
                f.write(page_data['text'])
        
        logger.info(f"Saved individual text files to {text_dir}")
        
        # Save statistics
        stats = {
            "total_pages": len(self.scraped_pages),
            "total_words": sum(p['word_count'] for p in self.scraped_pages),
            "avg_words_per_page": sum(p['word_count'] for p in self.scraped_pages) / len(self.scraped_pages) if self.scraped_pages else 0,
            "unique_urls": len(self.visited_urls)
        }
        
        stats_file = DATA_DIR / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics: {stats}")


def main():
    """Main scraping function"""
    logger.info("=" * 60)
    logger.info("EPA Knowledge Base Scraper")
    logger.info("=" * 60)
    
    # Create full URLs
    start_urls = [urljoin(EPA_BASE_URL, path) for path in EPA_PAGES]
    
    # Create scraper
    scraper = EPAScraper()
    
    # Scrape
    scraper.scrape(start_urls, max_pages=MAX_PAGES)
    
    # Save
    scraper.save()
    
    logger.info("=" * 60)
    logger.info("✅ EPA scraping complete!")
    logger.info(f"Data saved to: {DATA_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

