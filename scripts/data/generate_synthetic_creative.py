"""
Synthetic Creative Upcycling Data Generator - PRODUCTION GRADE

CRITICAL REQUIREMENTS:
- Target: 500K+ high-quality synthetic examples
- Method: GPT-4 API with carefully crafted prompts
- Quality: Diversity metrics, creativity scoring, deduplication
- Cost management: Batch processing, caching, token optimization
- Validation: Content safety, factual accuracy, creativity assessment

OUTPUT FORMAT:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "category": "...", "metadata": {...}}
"""

import os
import sys
import time
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import random

from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "llm" / "synthetic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. Get one at: https://platform.openai.com/api-keys")

# Generation parameters
MODEL = "gpt-4-turbo-preview"  # or "gpt-4" for higher quality
TEMPERATURE = 0.9  # High creativity
MAX_TOKENS = 800
BATCH_SIZE = 10
TARGET_COUNT = 500000

# Cost estimation (as of 2024)
COST_PER_1K_INPUT_TOKENS = 0.01  # $0.01/1K tokens
COST_PER_1K_OUTPUT_TOKENS = 0.03  # $0.03/1K tokens

# Waste items and materials for diversity
WASTE_ITEMS = [
    "plastic bottles", "glass jars", "cardboard boxes", "old tires", "wine corks",
    "tin cans", "egg cartons", "newspaper", "magazines", "fabric scraps",
    "wooden pallets", "broken furniture", "old clothes", "plastic bags", "bottle caps",
    "coffee grounds", "tea bags", "broken ceramics", "old books", "CDs/DVDs",
    "light bulbs", "mason jars", "wine bottles", "toilet paper rolls", "shoe boxes",
    "plastic containers", "aluminum cans", "paper bags", "bubble wrap", "styrofoam",
    "old electronics", "broken toys", "leather scraps", "denim jeans", "t-shirts",
    "sweaters", "curtains", "bed sheets", "towels", "socks", "buttons", "zippers",
    "guitar strings", "bicycle parts", "car tires", "license plates", "keys",
    "spoons", "forks", "knives", "plates", "cups", "mugs", "teapots", "vases"
]

MATERIALS = [
    "plastic (PET)", "plastic (HDPE)", "plastic (PVC)", "plastic (LDPE)", "plastic (PP)",
    "glass", "cardboard", "paper", "aluminum", "steel", "wood", "fabric (cotton)",
    "fabric (polyester)", "fabric (wool)", "fabric (silk)", "leather", "rubber",
    "ceramic", "porcelain", "bamboo", "cork", "foam", "styrofoam", "wax"
]

ART_FORMS = [
    "sculpture", "wall art", "mosaic", "collage", "installation art", "mixed media",
    "assemblage", "found object art", "kinetic sculpture", "light art", "textile art",
    "fiber art", "paper art", "origami", "quilling", "decoupage", "macrame",
    "weaving", "embroidery", "patchwork", "applique", "stained glass", "pottery"
]

FUNCTIONAL_ITEMS = [
    "planter", "lamp", "organizer", "storage container", "shelf", "table", "chair",
    "bench", "coat rack", "jewelry holder", "phone stand", "bookend", "picture frame",
    "mirror frame", "clock", "candle holder", "vase", "bird feeder", "bird house",
    "pet bed", "toy", "game", "puzzle", "instrument", "speaker", "headphone stand",
    "desk organizer", "pen holder", "notebook cover", "bookmark", "coaster", "trivet"
]

# Prompt templates for diversity
PROMPT_TEMPLATES = [
    # Creative transformation
    """Generate a detailed, creative upcycling idea for {item}.

Requirements:
- Explain what the item can be transformed into
- Describe the transformation process step-by-step
- Mention required tools and materials
- Estimate difficulty level and time required
- Highlight the environmental impact
- Be specific, creative, and inspiring

Format as a natural conversation where someone asks "What can I do with {item}?" and you provide a comprehensive, enthusiastic answer.""",
    
    # Art-focused
    """Explain how to turn {item} into {art_form}.

Requirements:
- Describe the artistic vision and final appearance
- Detail the creative process with specific techniques
- Mention color schemes, textures, and design elements
- Provide tips for achieving professional results
- Discuss how this transforms waste into beauty

Format as answering: "How can I create {art_form} from {item}?"
""",

    # Functional upcycling
    """Provide detailed instructions for upcycling {item} into a functional {functional_item}.

Requirements:
- Explain the practical use and benefits
- Provide step-by-step construction instructions
- List all materials and tools needed
- Include measurements and assembly tips
- Mention customization options
- Emphasize sustainability and cost savings

Format as answering: "How do I make a {functional_item} from {item}?"
""",

    # Multi-item combination
    """Create an innovative upcycling project combining {item1} and {item2}.

Requirements:
- Describe what unique item can be created
- Explain how the two materials complement each other
- Provide detailed assembly instructions
- Discuss the creative and practical value
- Mention variations and alternatives

Format as answering: "What can I make by combining {item1} and {item2}?"
""",

    # Advanced techniques
    """Explain advanced upcycling techniques for {item} using {material} properties.

Requirements:
- Discuss material science (durability, flexibility, etc.)
- Describe specialized techniques (cutting, molding, joining)
- Provide safety precautions
- Explain finishing and sealing methods
- Suggest professional-grade results

Format as answering: "What are advanced techniques for upcycling {item}?"
"""
]


class SyntheticDataGenerator:
    """Production-grade synthetic data generator using GPT-4"""

    def __init__(self):
        """Initialize OpenAI client"""
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.generated_data = []
        self.stats = defaultdict(int)
        self.seen_hashes = set()  # For deduplication
        self.total_cost = 0.0

        logger.info("✅ OpenAI client initialized")

    def generate_prompt(self) -> tuple[str, str, Dict]:
        """Generate a random prompt with diversity"""
        template = random.choice(PROMPT_TEMPLATES)

        # Select random items/materials
        item = random.choice(WASTE_ITEMS)
        item1 = random.choice(WASTE_ITEMS)
        item2 = random.choice([i for i in WASTE_ITEMS if i != item1])
        art_form = random.choice(ART_FORMS)
        functional_item = random.choice(FUNCTIONAL_ITEMS)
        material = random.choice(MATERIALS)

        # Fill template
        prompt = template.format(
            item=item,
            item1=item1,
            item2=item2,
            art_form=art_form,
            functional_item=functional_item,
            material=material
        )

        # Extract question for metadata
        if "How can I create" in prompt:
            question = f"How can I create {art_form} from {item}?"
            category = "creative_art"
        elif "How do I make" in prompt:
            question = f"How do I make a {functional_item} from {item}?"
            category = "functional_upcycling"
        elif "What can I make by combining" in prompt:
            question = f"What can I make by combining {item1} and {item2}?"
            category = "multi_item_upcycling"
        elif "What are advanced techniques" in prompt:
            question = f"What are advanced techniques for upcycling {item}?"
            category = "advanced_techniques"
        else:
            question = f"What can I do with {item}?"
            category = "upcycling_ideas"

        metadata = {
            "item": item,
            "art_form": art_form if "art_form" in template else None,
            "functional_item": functional_item if "functional_item" in template else None,
            "category": category
        }

        return prompt, question, metadata

    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response using GPT-4"""
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a creative sustainability expert specializing in innovative upcycling and waste transformation. Provide detailed, inspiring, and practical advice."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )

            # Track costs
            usage = response.usage
            input_cost = (usage.prompt_tokens / 1000) * COST_PER_1K_INPUT_TOKENS
            output_cost = (usage.completion_tokens / 1000) * COST_PER_1K_OUTPUT_TOKENS
            self.total_cost += input_cost + output_cost

            self.stats['total_input_tokens'] += usage.prompt_tokens
            self.stats['total_output_tokens'] += usage.completion_tokens

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Generation error: {e}")
            self.stats['generation_errors'] += 1
            return None

    def calculate_content_hash(self, text: str) -> str:
        """Calculate hash for deduplication"""
        return hashlib.md5(text.lower().encode()).hexdigest()

    def validate_response(self, response: str) -> tuple[bool, str]:
        """Validate generated response quality"""
        if not response:
            return False, "empty_response"

        # Length check
        word_count = len(response.split())
        if word_count < 50:
            return False, "too_short"
        if word_count > 1000:
            return False, "too_long"

        # Deduplication
        content_hash = self.calculate_content_hash(response)
        if content_hash in self.seen_hashes:
            return False, "duplicate"

        # Quality indicators
        quality_keywords = ["step", "material", "tool", "create", "make", "transform"]
        if sum(1 for kw in quality_keywords if kw in response.lower()) < 2:
            return False, "low_quality"

        # Safety check (no harmful content)
        harmful_keywords = ["dangerous", "toxic", "poison", "explosive", "illegal"]
        if any(kw in response.lower() for kw in harmful_keywords):
            return False, "safety_concern"

        self.seen_hashes.add(content_hash)
        return True, "valid"

    def generate_batch(self, batch_size: int) -> List[Dict]:
        """Generate a batch of examples"""
        batch = []

        for _ in range(batch_size):
            # Generate prompt
            prompt, question, metadata = self.generate_prompt()

            # Generate response
            response = self.generate_response(prompt)
            if not response:
                continue

            # Validate
            is_valid, reason = self.validate_response(response)
            if not is_valid:
                self.stats[f'rejected_{reason}'] += 1
                continue

            # Create Q&A pair
            qa_pair = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ],
                "category": metadata['category'],
                "metadata": {
                    "source": "synthetic_gpt4",
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "item": metadata['item'],
                    "art_form": metadata.get('art_form'),
                    "functional_item": metadata.get('functional_item'),
                    "word_count": len(response.split()),
                    "generated_at": datetime.now().isoformat()
                }
            }

            batch.append(qa_pair)
            self.stats['total_generated'] += 1

        return batch

    def run(self, target_count: int = TARGET_COUNT):
        """Run complete generation pipeline"""
        logger.info("="*60)
        logger.info("SYNTHETIC DATA GENERATOR - PRODUCTION MODE")
        logger.info(f"Model: {MODEL}")
        logger.info(f"Target: {target_count:,} examples")
        logger.info(f"Estimated cost: ${(target_count * 0.04):.2f}")  # ~$0.04 per example
        logger.info("="*60)

        # Generate in batches
        num_batches = target_count // BATCH_SIZE

        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            batch = self.generate_batch(BATCH_SIZE)
            self.generated_data.extend(batch)

            # Save periodically (every 100 batches)
            if (batch_idx + 1) % 100 == 0:
                self.save_data(checkpoint=True)
                logger.info(f"Checkpoint: {len(self.generated_data)} examples, Cost: ${self.total_cost:.2f}")

            # Rate limiting (avoid API throttling)
            time.sleep(0.5)

        # Final save
        self.save_data()
        self.print_statistics()

        logger.info(f"\n✅ GENERATION COMPLETE!")
        logger.info(f"Total examples: {len(self.generated_data):,}")
        logger.info(f"Total cost: ${self.total_cost:.2f}")

    def save_data(self, checkpoint: bool = False):
        """Save generated data"""
        suffix = "_checkpoint" if checkpoint else ""
        output_path = OUTPUT_DIR / f"synthetic_creative{suffix}.jsonl"

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.generated_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        if not checkpoint:
            logger.info(f"✅ Data saved to {output_path}")

            # Save stats
            stats_path = OUTPUT_DIR / "generation_stats.json"
            stats = dict(self.stats)
            stats['total_cost'] = self.total_cost
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

    def print_statistics(self):
        """Print generation statistics"""
        logger.info(f"\n{'='*60}")
        logger.info("GENERATION STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total generated: {self.stats['total_generated']:,}")
        logger.info(f"Input tokens: {self.stats['total_input_tokens']:,}")
        logger.info(f"Output tokens: {self.stats['total_output_tokens']:,}")
        logger.info(f"Total cost: ${self.total_cost:.2f}")
        logger.info(f"Cost per example: ${self.total_cost / max(self.stats['total_generated'], 1):.4f}")
        logger.info(f"{'='*60}")


def main():
    """Main entry point"""
    try:
        generator = SyntheticDataGenerator()
        generator.run(target_count=TARGET_COUNT)
        return True
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
