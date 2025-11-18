"""
Multi-Language Support for LLM Service

CRITICAL: Handles 8 languages for global reach

Supported Languages:
1. English (en) - Primary
2. Spanish (es) - European
3. French (fr) - European
4. German (de) - European
5. Italian (it) - European
6. Portuguese (pt) - European
7. Dutch (nl) - European
8. Japanese (ja) - Asian

Features:
- Language detection
- Translation to/from English
- Language-specific response formatting
"""

import re
from typing import Dict, Tuple, Optional
from enum import Enum
import logging
import hashlib

logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Supported languages"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    JAPANESE = "ja"


class LanguageHandler:
    """
    Multi-language handler with detection and translation

    CRITICAL: Lightweight implementation without external translation APIs
    Uses pattern matching for detection and basic phrase translation
    """

    def __init__(self):
        # Language detection patterns (common words/phrases)
        self.detection_patterns = {
            Language.ENGLISH: [
                r'\b(the|this|that|these|those|can|could|would|should)\b',
                r'\b(recycle|waste|trash|garbage|bin|donate|where|how)\b',
                r'\b(what|when|why|which|who)\b',
            ],
            Language.SPANISH: [
                r'\b(hola|gracias|por favor|cómo|qué|dónde|cuándo|reciclar|basura|residuos)\b',
                r'\b(el|la|los|las|un|una|de|del|para|con)\b',
            ],
            Language.FRENCH: [
                r'\b(bonjour|merci|s\'il vous plaît|comment|quoi|où|quand|recycler|déchets)\b',
                r'\b(le|la|les|un|une|de|du|pour|avec|dans|quelle|poubelle|centres)\b',
                r'\b(puis-je|va-t-il|trouver|près)\b',
            ],
            Language.GERMAN: [
                r'\b(hallo|danke|bitte|wie|was|wo|wann|recyceln|müll|abfall)\b',
                r'\b(der|die|das|ein|eine|von|für|mit|kann|ich|in|welche|tonne)\b',
                r'\b(lebensmittelabfälle|kompostieren|finde|meiner|nähe)\b',
            ],
            Language.ITALIAN: [
                r'\b(ciao|grazie|per favore|come|cosa|dove|quando|riciclare|rifiuti)\b',
                r'\b(il|lo|la|i|gli|le|un|una|di|del|per|con|in|quale|bidone|va|questo)\b',
                r'\b(posso|trova|vicino|centri)\b',
            ],
            Language.PORTUGUESE: [
                r'\b(olá|obrigado|por favor|como|o que|onde|quando|reciclar|lixo|resíduos)\b',
                r'\b(o|a|os|as|um|uma|de|do|para|com|em|qual|lixeira|isso|vai)\b',
                r'\b(posso|doar|roupas|velhas|encontre|centros|perto)\b',
            ],
            Language.DUTCH: [
                r'\b(hallo|dank je|alstublieft|hoe|wat|waar|wanneer|recyclen|afval)\b',
                r'\b(de|het|een|van|voor|met|in|welke|bak|gaat|dit|deze)\b',
                r'\b(ik|kan|plastic|fles|bij|mij|buurt)\b',
            ],
            Language.JAPANESE: [
                r'[\u3040-\u309F]',  # Hiragana
                r'[\u30A0-\u30FF]',  # Katakana
                r'[\u4E00-\u9FAF]',  # Kanji
            ],
        }

        # Compile patterns
        self.compiled_patterns = {
            lang: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]
            for lang, patterns in self.detection_patterns.items()
        }

        # Common waste management phrases for translation
        self.common_phrases = {
            Language.SPANISH: {
                "recycle": "reciclar",
                "waste": "residuos",
                "trash": "basura",
                "bin": "contenedor",
                "plastic": "plástico",
                "metal": "metal",
                "glass": "vidrio",
                "paper": "papel",
                "how to dispose": "cómo desechar",
                "where to recycle": "dónde reciclar",
            },
            Language.FRENCH: {
                "recycle": "recycler",
                "waste": "déchets",
                "trash": "ordures",
                "bin": "poubelle",
                "plastic": "plastique",
                "metal": "métal",
                "glass": "verre",
                "paper": "papier",
                "how to dispose": "comment jeter",
                "where to recycle": "où recycler",
            },
            Language.GERMAN: {
                "recycle": "recyceln",
                "waste": "abfall",
                "trash": "müll",
                "bin": "tonne",
                "plastic": "kunststoff",
                "metal": "metall",
                "glass": "glas",
                "paper": "papier",
                "how to dispose": "wie entsorgen",
                "where to recycle": "wo recyceln",
            },
            Language.ITALIAN: {
                "recycle": "riciclare",
                "waste": "rifiuti",
                "trash": "spazzatura",
                "bin": "bidone",
                "plastic": "plastica",
                "metal": "metallo",
                "glass": "vetro",
                "paper": "carta",
                "how to dispose": "come smaltire",
                "where to recycle": "dove riciclare",
            },
            Language.PORTUGUESE: {
                "recycle": "reciclar",
                "waste": "resíduos",
                "trash": "lixo",
                "bin": "lixeira",
                "plastic": "plástico",
                "metal": "metal",
                "glass": "vidro",
                "paper": "papel",
                "how to dispose": "como descartar",
                "where to recycle": "onde reciclar",
            },
            Language.JAPANESE: {
                "recycle": "リサイクル",
                "waste": "廃棄物",
                "trash": "ゴミ",
                "bin": "ゴミ箱",
                "plastic": "プラスチック",
                "metal": "金属",
                "glass": "ガラス",
                "paper": "紙",
                "how to dispose": "処分方法",
                "where to recycle": "リサイクル場所",
            },
        }

        # Cache for language detection
        self._cache = {}
        self._cache_max_size = 500

        logger.info("Language handler initialized with 8 languages")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def detect_language(self, text: str) -> Tuple[Language, float]:
        """
        Detect language from text

        Args:
            text: Input text

        Returns:
            (detected_language, confidence_score)
        """
        try:
            # Input validation
            if not text or not isinstance(text, str):
                logger.warning(f"Invalid input type: {type(text)}")
                return Language.ENGLISH, 0.5

            if not text.strip():
                return Language.ENGLISH, 0.5

            # Check cache first
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for language detection")
                return self._cache[cache_key]

            # Truncate very long text
            max_length = 1000
            if len(text) > max_length:
                logger.warning(f"Text truncated from {len(text)} to {max_length} chars")
                text = text[:max_length]

            text = text.strip()

            # Check for Japanese first (unique character sets)
            try:
                if self.compiled_patterns[Language.JAPANESE][0].search(text):
                    logger.info("Japanese detected via character set")
                    result = (Language.JAPANESE, 0.95)
                    # Cache result
                    if len(self._cache) >= self._cache_max_size:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[cache_key] = result
                    return result
            except Exception as e:
                logger.error(f"Error checking Japanese patterns: {e}")

            # Score each language
            scores = {}

            for lang, patterns in self.compiled_patterns.items():
                if lang == Language.JAPANESE:
                    continue  # Already checked

                score = 0
                for pattern in patterns:
                    try:
                        matches = pattern.findall(text)
                        score += len(matches)
                    except Exception as e:
                        logger.error(f"Pattern matching error for {lang.value}: {e}")
                        continue
                scores[lang] = score

            # Get best match
            if not scores or max(scores.values()) == 0:
                # No pattern matched - default to English
                logger.info("No language pattern matched, defaulting to English")
                return Language.ENGLISH, 0.5

            best_lang = max(scores, key=scores.get)
            max_score = scores[best_lang]

            # Calculate confidence
            total_score = sum(scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.5

            # Boost confidence if score is significantly higher
            if max_score >= 3:
                confidence = min(0.95, confidence + 0.2)

            logger.info(f"Language detected: {best_lang.value} (confidence: {confidence:.2f})")

            # Cache result (with FIFO eviction)
            result = (best_lang, confidence)
            if len(self._cache) >= self._cache_max_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Error in language detection: {e}", exc_info=True)
            return Language.ENGLISH, 0.5

    def translate_to_english(self, text: str, source_lang: Language) -> str:
        """
        Translate text to English (basic phrase translation)

        Note: This is a simple phrase-based translation for common waste management terms.
        For production, consider using Google Translate API or similar service.

        Args:
            text: Input text in source language
            source_lang: Source language

        Returns:
            Translated text in English
        """
        if source_lang == Language.ENGLISH:
            return text

        if source_lang not in self.common_phrases:
            logger.warning(f"Translation not available for {source_lang.value}")
            return text

        # Simple phrase replacement
        translated = text
        phrases = self.common_phrases[source_lang]

        for english, foreign in phrases.items():
            # Replace foreign phrase with English
            pattern = re.compile(re.escape(foreign), re.IGNORECASE | re.UNICODE)
            translated = pattern.sub(english, translated)

        logger.info(f"Translated from {source_lang.value} to English")

        return translated

    def translate_from_english(self, text: str, target_lang: Language) -> str:
        """
        Translate text from English to target language (basic phrase translation)

        Args:
            text: Input text in English
            target_lang: Target language

        Returns:
            Translated text in target language
        """
        if target_lang == Language.ENGLISH:
            return text

        if target_lang not in self.common_phrases:
            logger.warning(f"Translation not available for {target_lang.value}")
            return text

        # Simple phrase replacement
        translated = text
        phrases = self.common_phrases[target_lang]

        for english, foreign in phrases.items():
            # Replace English phrase with foreign
            pattern = re.compile(r'\b' + re.escape(english) + r'\b', re.IGNORECASE)
            translated = pattern.sub(foreign, translated)

        logger.info(f"Translated from English to {target_lang.value}")

        return translated

    def get_language_name(self, lang: Language) -> str:
        """Get human-readable language name"""
        names = {
            Language.ENGLISH: "English",
            Language.SPANISH: "Español",
            Language.FRENCH: "Français",
            Language.GERMAN: "Deutsch",
            Language.ITALIAN: "Italiano",
            Language.PORTUGUESE: "Português",
            Language.DUTCH: "Nederlands",
            Language.JAPANESE: "日本語",
        }
        return names.get(lang, "Unknown")

    def format_response(self, text: str, lang: Language) -> str:
        """
        Format response for specific language

        Adds language-specific formatting and politeness markers
        """
        if lang == Language.JAPANESE:
            # Add polite ending for Japanese
            if not text.endswith(('。', '！', '？')):
                text += '。'

        return text
