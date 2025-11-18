"""
Intent Classification System for LLM Service

CRITICAL: Classifies user intent to provide context-aware responses

Intent Categories:
1. WASTE_IDENTIFICATION - "What is this item?" "Can I recycle this?"
2. DISPOSAL_GUIDANCE - "How do I dispose of X?" "Which bin for Y?"
3. UPCYCLING_IDEAS - "How can I reuse X?" "Upcycling ideas for Y?"
4. ORGANIZATION_SEARCH - "Where can I donate X?" "Recycling centers near me?"
5. SUSTAINABILITY_INFO - "Why is recycling important?" "Environmental impact of X?"
6. GENERAL_QUESTION - "How does recycling work?" "What is composting?"
7. CHITCHAT - "Hello" "Thank you" "How are you?"
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


class IntentCategory(str, Enum):
    """Intent categories"""
    WASTE_IDENTIFICATION = "waste_identification"
    DISPOSAL_GUIDANCE = "disposal_guidance"
    UPCYCLING_IDEAS = "upcycling_ideas"
    ORGANIZATION_SEARCH = "organization_search"
    SUSTAINABILITY_INFO = "sustainability_info"
    GENERAL_QUESTION = "general_question"
    CHITCHAT = "chitchat"


class IntentClassifier:
    """
    Rule-based + pattern-matching intent classifier

    CRITICAL: Fast, accurate intent classification without ML model
    """

    def __init__(self):
        # Define patterns for each intent
        self.patterns = {
            IntentCategory.WASTE_IDENTIFICATION: [
                r'\b(what is|identify|recognize|detect|classify)\b.*\b(this|item|object|material|waste)\b',
                r'\b(can i|is this|is it)\b.*\b(recycle|recyclable|compost|compostable)\b',
                r'\b(type of|kind of|category)\b.*\b(waste|material|item)\b',
                r'\b(plastic|metal|glass|paper|cardboard)\b.*\b(type|number|grade)\b',
                r'\b(what type|what kind|what material)\b',
                r'\b(made of)\b',
                r'\b(is this|is it)\b.*\b(metal|plastic|glass|paper)\b',
            ],

            IntentCategory.DISPOSAL_GUIDANCE: [
                r'\b(how|where|which bin)\b.*\b(dispose|throw|discard|get rid)\b',
                r'\b(which bin|what bin|correct bin|which container)\b',
                r'\b(trash|garbage|waste|recycling)\b.*\b(bin|container|disposal)\b',
                r'\b(goes in|put in|belongs in)\b.*\b(bin|trash|recycling)\b',
                r'\b(dispose of|disposal|throw away)\b',
                r'\b(how do i|how to|how should i)\b.*\b(dispose|throw|discard)\b',
            ],

            IntentCategory.UPCYCLING_IDEAS: [
                r'\b(upcycle|upcycling|repurpose|reuse|diy)\b',
                r'\b(creative|ideas|projects)\b.*\b(reuse|repurpose|upcycle|old)\b',
                r'\b(what can i|how can i)\b.*\b(reuse|repurpose|make)\b',
                r'\b(turn into|transform|convert)\b.*\b(something|useful)\b',
                r'\b(second life|new use|alternative use)\b',
                r'\b(creative ideas|ideas for)\b',
                r'\b(turn|transform)\b.*\b(old|clothes|into)\b',
            ],

            IntentCategory.ORGANIZATION_SEARCH: [
                r'\b(where|find|locate|search)\b.*\b(donate|donation|charity|organization)\b',
                r'\b(recycling center|drop.?off|collection point|recycling facilities)\b',
                r'\b(near me|nearby|local|in my area)\b.*\b(recycle|donate|disposal)\b',
                r'\b(accept|take|collect)\b.*\b(donations|recyclables|waste)\b',
                r'\b(charity|non.?profit|organization|charities)\b.*\b(accept|take)\b',
                r'\b(find|locate)\b.*\b(recycling center|recycling facilities)\b',
                r'\b(recycling centers|recycling facilities)\b.*\b(near|in my area|local)\b',
                r'\b(collection points|thrift stores)\b.*\b(for|nearby)\b',
            ],

            IntentCategory.SUSTAINABILITY_INFO: [
                r'\b(why|importance|benefit|impact)\b.*\b(recycle|recycling|sustainability|environment)\b',
                r'\b(environmental|ecological|carbon)\b.*\b(impact|footprint|effect)\b',
                r'\b(climate|global warming|pollution)\b',
                r'\b(sustainable|sustainability|eco.?friendly|green)\b',
                r'\b(statistics|facts|data)\b.*\b(recycling|waste|environment)\b',
                r'\b(benefits of|how does)\b.*\b(recycling|composting)\b',
            ],

            IntentCategory.GENERAL_QUESTION: [
                r'\b(how does|how do|what is|what are|explain)\b.*\b(recycling|composting|waste management)\b',
                r'\b(difference between|types of|categories of)\b.*\b(waste|recycling|materials)\b',
                r'\b(process|system|method)\b.*\b(recycling|waste|disposal)\b',
                r'\b(learn|understand|know more)\b.*\b(recycling|waste|sustainability)\b',
            ],

            IntentCategory.CHITCHAT: [
                r'^\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b',
                r'^\b(thank you|thanks|appreciate|grateful)\b',
                r'^\b(bye|goodbye|see you|farewell)\b',
                r'^\b(how are you|how\'s it going|what\'s up)\b',
                r'^\b(yes|no|ok|okay|sure|alright)\b$',
            ],
        }

        # Compile patterns for efficiency
        self.compiled_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.patterns.items()
        }

        # Cache for classification results
        self._cache = {}
        self._cache_max_size = 1000

        logger.info("Intent classifier initialized with 7 categories")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def classify(self, text: str) -> Tuple[IntentCategory, float]:
        """
        Classify user intent

        Args:
            text: User input text

        Returns:
            (intent_category, confidence_score)
        """
        try:
            # Input validation
            if not text or not isinstance(text, str):
                logger.warning(f"Invalid input type: {type(text)}")
                return IntentCategory.GENERAL_QUESTION, 0.5

            if not text.strip():
                logger.warning("Empty text input")
                return IntentCategory.GENERAL_QUESTION, 0.5

            # Check cache first
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for intent classification")
                return self._cache[cache_key]

            # Truncate very long text to prevent performance issues
            max_length = 1000
            if len(text) > max_length:
                logger.warning(f"Text truncated from {len(text)} to {max_length} chars")
                text = text[:max_length]

            text = text.strip().lower()

            # Score each intent with early exit optimization
            scores = {}
            max_possible_score = 0
            for intent, patterns in self.compiled_patterns.items():
                score = 0
                max_possible_score = max(max_possible_score, len(patterns))
                for pattern in patterns:
                    try:
                        if pattern.search(text):
                            score += 1
                            # Early exit if we have high confidence
                            if score >= 3:
                                break
                    except Exception as e:
                        logger.error(f"Pattern search error: {e}")
                        continue
                scores[intent] = score

            # Get best match
            if not scores or max(scores.values()) == 0:
                # No pattern matched - default to general question
                logger.info("No pattern matched, defaulting to GENERAL_QUESTION")
                result = (IntentCategory.GENERAL_QUESTION, 0.3)
                self._cache[cache_key] = result
                return result

            best_intent = max(scores, key=scores.get)
            max_score = scores[best_intent]

            # Calculate confidence (normalize by number of patterns)
            num_patterns = len(self.compiled_patterns[best_intent])
            confidence = min(1.0, max_score / num_patterns) if num_patterns > 0 else 0.5

            logger.info(f"Intent classified: {best_intent.value} (confidence: {confidence:.2f})")

            # Cache result (with LRU eviction)
            if len(self._cache) >= self._cache_max_size:
                # Remove oldest entry (simple FIFO for now)
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = (best_intent, confidence)

            return best_intent, confidence

        except Exception as e:
            logger.error(f"Error in intent classification: {e}", exc_info=True)
            return IntentCategory.GENERAL_QUESTION, 0.5

    def get_context_hints(self, intent: IntentCategory) -> Dict[str, any]:
        """
        Get context hints for each intent to guide LLM response

        Returns:
            Dictionary with response guidelines
        """
        hints = {
            IntentCategory.WASTE_IDENTIFICATION: {
                "use_vision": True,
                "use_rag": True,
                "use_kg": False,
                "response_style": "informative",
                "max_length": 200,
            },
            IntentCategory.DISPOSAL_GUIDANCE: {
                "use_vision": True,
                "use_rag": True,
                "use_kg": False,
                "response_style": "instructional",
                "max_length": 150,
            },
            IntentCategory.UPCYCLING_IDEAS: {
                "use_vision": True,
                "use_rag": True,
                "use_kg": True,  # Use GNN for recommendations
                "response_style": "creative",
                "max_length": 300,
            },
            IntentCategory.ORGANIZATION_SEARCH: {
                "use_vision": False,
                "use_rag": True,
                "use_kg": False,
                "use_org_search": True,  # Use organization search service
                "response_style": "helpful",
                "max_length": 250,
            },
            IntentCategory.SUSTAINABILITY_INFO: {
                "use_vision": False,
                "use_rag": True,
                "use_kg": False,
                "response_style": "educational",
                "max_length": 400,
            },
            IntentCategory.GENERAL_QUESTION: {
                "use_vision": False,
                "use_rag": True,
                "use_kg": False,
                "response_style": "informative",
                "max_length": 300,
            },
            IntentCategory.CHITCHAT: {
                "use_vision": False,
                "use_rag": False,
                "use_kg": False,
                "response_style": "friendly",
                "max_length": 50,
            },
        }

        return hints.get(intent, hints[IntentCategory.GENERAL_QUESTION])

