"""
Entity Extraction System for LLM Service

CRITICAL: Extracts key entities from user queries for context-aware responses

Entity Types:
1. MATERIAL - plastic, metal, glass, paper, cardboard, etc.
2. ITEM - bottle, can, box, bag, container, etc.
3. LOCATION - city, state, zip code, address, "near me"
4. ACTION - recycle, dispose, donate, upcycle, reuse, etc.
5. ORGANIZATION - charity, recycling center, donation center, etc.
6. QUANTITY - numbers, amounts, sizes
7. TIME - today, tomorrow, this week, etc.
"""

import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Extracted entity"""
    text: str
    type: str
    start: int
    end: int
    confidence: float = 1.0


class EntityExtractor:
    """
    Rule-based entity extractor for waste management domain

    CRITICAL: Fast, accurate entity extraction without ML model
    """

    def __init__(self):
        # Define entity dictionaries
        self.materials = {
            'plastic', 'metal', 'glass', 'paper', 'cardboard', 'aluminum', 'steel',
            'wood', 'fabric', 'textile', 'leather', 'rubber', 'foam', 'styrofoam',
            'ceramic', 'porcelain', 'electronics', 'e-waste', 'battery', 'batteries',
            'organic', 'food waste', 'compost', 'biodegradable',
            'hdpe', 'ldpe', 'pet', 'pp', 'ps', 'pvc',  # Plastic types
            'type 1', 'type 2', 'type 3', 'type 4', 'type 5', 'type 6', 'type 7',
        }

        self.items = {
            'bottle', 'can', 'jar', 'container', 'box', 'bag', 'wrapper', 'packaging',
            'cup', 'plate', 'bowl', 'utensil', 'fork', 'knife', 'spoon',
            'straw', 'lid', 'cap', 'label', 'tape', 'string', 'ribbon',
            'newspaper', 'magazine', 'book', 'envelope', 'receipt',
            'phone', 'computer', 'laptop', 'tablet', 'charger', 'cable',
            'battery', 'lightbulb', 'bulb', 'appliance', 'furniture',
            'clothing', 'shoes', 'shirt', 'pants', 'jacket', 'dress',
            'toy', 'game', 'puzzle', 'doll', 'action figure',
            'tire', 'mattress', 'pillow', 'blanket', 'towel',
        }

        self.actions = {
            'recycle', 'recycling', 'dispose', 'disposal', 'throw away', 'discard',
            'donate', 'donation', 'give away', 'drop off', 'drop-off',
            'upcycle', 'upcycling', 'repurpose', 'reuse', 'reusing',
            'compost', 'composting', 'decompose',
            'trash', 'garbage', 'waste',
            'sort', 'sorting', 'separate', 'separating',
            'clean', 'cleaning', 'wash', 'washing', 'rinse', 'rinsing',
        }

        self.organizations = {
            'charity', 'charities', 'non-profit', 'nonprofit', 'ngo',
            'goodwill', 'salvation army', 'habitat for humanity',
            'recycling center', 'recycling facility', 'transfer station',
            'donation center', 'drop-off center', 'collection point',
            'thrift store', 'thrift shop', 'second-hand store',
        }

        # Location patterns
        self.location_patterns = [
            r'\b\d{5}(?:-\d{4})?\b',  # ZIP code
            r'\bnear me\b',
            r'\bin my area\b',
            r'\bmy location\b',
            r'\bmy city\b',
            r'\bmy town\b',
            r'\bmy neighborhood\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b',  # City, ST
        ]

        # Quantity patterns
        self.quantity_patterns = [
            r'\b\d+\s*(?:kg|g|lb|oz|lbs|pounds|ounces|kilograms|grams)\b',
            r'\b\d+\s*(?:liters?|gallons?|ml|l)\b',
            r'\b\d+\s*(?:pieces?|items?|units?)\b',
            r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:pieces?|items?|units?)\b',
        ]

        # Time patterns
        self.time_patterns = [
            r'\btoday\b',
            r'\btomorrow\b',
            r'\byesterday\b',
            r'\bthis week\b',
            r'\bnext week\b',
            r'\bthis month\b',
            r'\bnext month\b',
            r'\bmonday|tuesday|wednesday|thursday|friday|saturday|sunday\b',
        ]

        # Compile patterns
        self.compiled_location_patterns = [re.compile(p, re.IGNORECASE) for p in self.location_patterns]
        self.compiled_quantity_patterns = [re.compile(p, re.IGNORECASE) for p in self.quantity_patterns]
        self.compiled_time_patterns = [re.compile(p, re.IGNORECASE) for p in self.time_patterns]

        # Cache for extraction results
        self._cache = {}
        self._cache_max_size = 500

        logger.info("Entity extractor initialized with 7 entity types")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def extract(self, text: str) -> List[Entity]:
        """
        Extract entities from text

        Args:
            text: Input text

        Returns:
            List of extracted entities
        """
        try:
            # Input validation
            if not text or not isinstance(text, str):
                logger.warning(f"Invalid input type: {type(text)}")
                return []

            if not text.strip():
                return []

            # Check cache first
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for entity extraction")
                return self._cache[cache_key]

            # Truncate very long text
            max_length = 2000
            if len(text) > max_length:
                logger.warning(f"Text truncated from {len(text)} to {max_length} chars")
                text = text[:max_length]

            entities = []
            text_lower = text.lower()

            # Extract materials
            for material in self.materials:
                pattern = r'\b' + re.escape(material) + r'\b'
                for match in re.finditer(pattern, text_lower):
                    entities.append(Entity(
                        text=match.group(),
                        type='MATERIAL',
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0
                    ))

            # Extract items
            for item in self.items:
                pattern = r'\b' + re.escape(item) + r'\b'
                for match in re.finditer(pattern, text_lower):
                    entities.append(Entity(
                        text=match.group(),
                        type='ITEM',
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0
                    ))

            # Extract actions
            for action in self.actions:
                pattern = r'\b' + re.escape(action) + r'\b'
                for match in re.finditer(pattern, text_lower):
                    entities.append(Entity(
                        text=match.group(),
                        type='ACTION',
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0
                    ))

            # Extract organizations
            for org in self.organizations:
                pattern = r'\b' + re.escape(org) + r'\b'
                for match in re.finditer(pattern, text_lower):
                    entities.append(Entity(
                        text=match.group(),
                        type='ORGANIZATION',
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0
                    ))

            # Extract locations
            for pattern in self.compiled_location_patterns:
                for match in pattern.finditer(text):
                    entities.append(Entity(
                        text=match.group(),
                        type='LOCATION',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    ))

            # Extract quantities
            for pattern in self.compiled_quantity_patterns:
                for match in pattern.finditer(text):
                    entities.append(Entity(
                        text=match.group(),
                        type='QUANTITY',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95
                    ))

            # Extract time expressions
            for pattern in self.compiled_time_patterns:
                for match in pattern.finditer(text):
                    entities.append(Entity(
                        text=match.group(),
                        type='TIME',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95
                    ))

            # Remove duplicates (keep highest confidence)
            entities = self._remove_duplicates(entities)

            # Sort by position
            entities.sort(key=lambda e: e.start)

            logger.info(f"Extracted {len(entities)} entities from text")

            # Cache result (with FIFO eviction)
            if len(self._cache) >= self._cache_max_size:
                # Remove oldest entry
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = entities

            return entities

        except Exception as e:
            logger.error(f"Error in entity extraction: {e}", exc_info=True)
            return []

    def _remove_duplicates(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping highest confidence"""
        if not entities:
            return []

        # Sort by start position
        sorted_entities = sorted(entities, key=lambda e: (e.start, -e.confidence))

        result = []
        last_end = -1

        for entity in sorted_entities:
            # Skip if overlaps with previous entity
            if entity.start < last_end:
                continue

            result.append(entity)
            last_end = entity.end

        return result

    def get_entity_summary(self, entities: List[Entity]) -> Dict[str, List[str]]:
        """
        Get summary of entities by type

        Returns:
            Dictionary mapping entity type to list of entity texts
        """
        summary = {}
        for entity in entities:
            if entity.type not in summary:
                summary[entity.type] = []
            summary[entity.type].append(entity.text)

        return summary
