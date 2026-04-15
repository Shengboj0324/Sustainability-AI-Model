"""
Entity Extraction System for LLM Service — v2 (Semantic)

ARCHITECTURE:
- Primary: LLM-based structured entity extraction via prompt templates
- Fallback: Massively expanded dictionary + regex extraction
- API: Fully backward-compatible (same Entity dataclass, extract(), get_entity_summary())

Entity Types (expanded from 7 to 10):
1. MATERIAL    - Specific materials, polymers, alloys, compounds
2. ITEM        - Physical objects, products, packaging
3. CHEMICAL    - Chemical compounds, toxins, additives
4. PROCESS     - Actions: recycle, compost, incinerate, upcycle
5. LOCATION    - Places, regions, "near me"
6. ORGANIZATION- Named orgs, facility types
7. REGULATION  - Laws, standards, certifications
8. METRIC      - Quantities, measurements
9. ENVIRONMENTAL_CONCEPT - Abstract sustainability concepts
10. TIME       - Temporal expressions
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Callable, Awaitable
from dataclasses import dataclass, asdict
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Extracted entity — backward compatible with v1"""
    text: str
    type: str
    start: int
    end: int
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class EntityExtractor:
    """
    Semantic entity extractor with LLM primary and expanded dictionary fallback.

    Usage:
        extractor = EntityExtractor()
        entities = extractor.extract("Can I recycle HDPE #2 plastic bottles?")
        summary = extractor.get_entity_summary(entities)

        # LLM-based (async):
        entities = await extractor.extract_with_llm(query, llm_call_fn)
    """

    def __init__(self):
        self._cache: Dict[str, List[Entity]] = {}
        self._cache_max_size = 500

        # Build the expanded dictionaries
        self._dicts = self._build_entity_dicts()
        self._regex_entities = self._build_regex_patterns()

        total = sum(len(v) for v in self._dicts.values())
        logger.info(f"Entity extractor initialized: {total} dictionary terms, "
                     f"{len(self._regex_entities)} regex patterns")

    # ------------------------------------------------------------------
    # PUBLIC API — synchronous (backward compatible)
    # ------------------------------------------------------------------

    def extract(self, text: str) -> List[Entity]:
        """Extract entities using expanded dictionary + regex."""
        if not text or not isinstance(text, str) or not text.strip():
            return []

        cache_key = hashlib.sha256(text.lower().strip().encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        text = text[:2000]
        text_lower = text.lower()
        entities: List[Entity] = []

        # 1. Dictionary-based extraction (multi-word first for longest match)
        for entity_type, terms in self._dicts.items():
            sorted_terms = sorted(terms, key=len, reverse=True)
            for term in sorted_terms:
                pattern = r'\b' + re.escape(term) + r'(?:s|es|ing|ed)?\b'
                for match in re.finditer(pattern, text_lower):
                    entities.append(Entity(
                        text=match.group(),
                        type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                    ))

        # 2. Regex-based extraction
        for entity_type, compiled_patterns in self._regex_entities.items():
            for pat in compiled_patterns:
                for match in pat.finditer(text if entity_type == "LOCATION" else text_lower):
                    entities.append(Entity(
                        text=match.group(),
                        type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85,
                    ))

        entities = self._remove_overlaps(entities)
        entities.sort(key=lambda e: e.start)

        if len(self._cache) >= self._cache_max_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = entities
        return entities

    # ------------------------------------------------------------------
    # PUBLIC API — async LLM-based extraction
    # ------------------------------------------------------------------

    async def extract_with_llm(
        self,
        text: str,
        llm_call: Callable[[str], Awaitable[str]],
    ) -> List[Entity]:
        """Extract entities using LLM for semantic understanding."""
        if not text or not text.strip():
            return []

        try:
            from .prompt_templates.sustainability_prompts import extract_entities_prompt
            prompt = extract_entities_prompt(text)
            raw_output = await llm_call(prompt)
            return self.parse_llm_entity_response(raw_output, text)
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}, falling back to dictionary")
            return self.extract(text)

    def parse_llm_entity_response(self, raw_output: str, original_text: str = "") -> List[Entity]:
        """Parse LLM entity extraction response into Entity list."""
        try:
            raw_output = raw_output.strip()
            if raw_output.startswith("```"):
                raw_output = re.sub(r"^```(?:json)?\s*", "", raw_output)
                raw_output = re.sub(r"\s*```$", "", raw_output)

            parsed = json.loads(raw_output)
            entities = []
            text_lower = original_text.lower()

            for e in parsed.get("entities", []):
                e_text = e.get("text", "")
                e_type = e.get("type", "MATERIAL").upper()
                e_conf = float(e.get("confidence", 0.8))

                # Try to find position in original text
                pos = text_lower.find(e_text.lower())
                start = pos if pos >= 0 else 0
                end = start + len(e_text) if pos >= 0 else len(e_text)

                entities.append(Entity(
                    text=e_text,
                    type=e_type,
                    start=start,
                    end=end,
                    confidence=min(1.0, max(0.0, e_conf)),
                ))

            return self._remove_overlaps(entities)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # PUBLIC API — summary (backward compatible)
    # ------------------------------------------------------------------

    def get_entity_summary(self, entities: List[Entity]) -> Dict[str, List[str]]:
        """Get summary of entities grouped by type."""
        summary: Dict[str, List[str]] = {}
        for entity in entities:
            summary.setdefault(entity.type, []).append(entity.text)
        return summary

    # ------------------------------------------------------------------
    # PRIVATE — overlap removal
    # ------------------------------------------------------------------

    def _remove_overlaps(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, preferring longer matches and higher confidence."""
        if not entities:
            return []

        # Sort by: longer span first, then higher confidence, then earlier position
        sorted_ents = sorted(
            entities,
            key=lambda e: (-(e.end - e.start), -e.confidence, e.start),
        )

        result = []
        occupied = set()

        for entity in sorted_ents:
            span = set(range(entity.start, entity.end))
            if not span & occupied:
                result.append(entity)
                occupied |= span

        return result

    # ------------------------------------------------------------------
    # PRIVATE — dictionary builders
    # ------------------------------------------------------------------

    def _build_entity_dicts(self) -> Dict[str, set]:
        """Build comprehensive entity dictionaries for domain extraction."""
        return {
            "MATERIAL": {
                # Plastics by resin code
                "pet", "hdpe", "pvc", "ldpe", "pp", "ps", "abs",
                "polypropylene", "polyethylene", "polystyrene", "polyester",
                "polycarbonate", "polyurethane", "nylon", "acrylic",
                "pla", "polylactic acid", "bioplastic",
                "#1 plastic", "#2 plastic", "#3 plastic", "#4 plastic",
                "#5 plastic", "#6 plastic", "#7 plastic",
                # Metals
                "aluminum", "steel", "tin", "copper", "brass", "iron",
                "stainless steel", "galvanized steel",
                # Glass
                "glass", "borosilicate", "tempered glass", "pyrex",
                # Paper & wood
                "paper", "cardboard", "corrugated cardboard", "paperboard",
                "kraft paper", "wax paper", "parchment paper",
                "wood", "plywood", "mdf", "particle board", "bamboo", "cork",
                # Textiles
                "cotton", "polyester", "nylon", "silk", "wool", "linen",
                "denim", "leather", "faux leather", "suede",
                "fabric", "textile",
                # Other
                "rubber", "silicone", "foam", "styrofoam", "expanded polystyrene",
                "ceramic", "porcelain", "clay", "concrete",
                "fiberglass", "carbon fiber", "kevlar",
                "compostable", "biodegradable", "organic",
                "e-waste", "electronic waste",
                "plastic", "metal",
            },
            "ITEM": {
                # Containers
                "bottle", "can", "jar", "container", "jug", "carton", "tub",
                "box", "bag", "wrapper", "packaging", "blister pack", "clamshell",
                # Food/drink
                "cup", "plate", "bowl", "utensil", "fork", "knife", "spoon",
                "straw", "lid", "cap", "six-pack ring",
                "pizza box", "takeout container", "coffee cup", "water bottle",
                # Paper
                "newspaper", "magazine", "book", "envelope", "receipt",
                "junk mail", "catalog", "phone book", "tissue", "paper towel",
                # Electronics
                "phone", "cell phone", "smartphone", "computer", "laptop",
                "tablet", "monitor", "tv", "television", "printer",
                "charger", "cable", "headphones", "earbuds",
                "battery", "batteries", "lightbulb", "led bulb", "cfl bulb",
                "fluorescent tube", "appliance",
                # Household
                "furniture", "mattress", "pillow", "blanket", "towel",
                "carpet", "rug", "curtain", "mirror",
                # Clothing
                "clothing", "clothes", "shoes", "shirt", "pants", "jeans",
                "jacket", "coat", "dress", "sweater", "socks", "underwear",
                # Other
                "toy", "tire", "paint can", "motor oil", "propane tank",
                "fire extinguisher", "aerosol can", "spray can",
                "prescription bottle", "medicine bottle", "syringe", "needle",
                "diaper", "sanitary product",
            },
            "CHEMICAL": {
                "bpa", "bisphenol a", "phthalate", "phthalates",
                "lead", "mercury", "cadmium", "arsenic", "chromium",
                "asbestos", "formaldehyde", "benzene", "toluene",
                "dioxin", "dioxins", "furan", "furans",
                "pcb", "pcbs", "pfas", "pfoa", "pfos",
                "flame retardant", "flame retardants",
                "pesticide", "herbicide", "insecticide",
                "ammonia", "bleach", "chlorine",
                "antifreeze", "motor oil", "brake fluid",
                "lithium", "acid", "alkali", "solvent",
                "microplastic", "microplastics", "nanoplastic",
            },
            "PROCESS": {
                "recycle", "recycling", "dispose", "disposal",
                "compost", "composting", "vermicompost",
                "incinerate", "incineration", "landfill",
                "upcycle", "upcycling", "repurpose", "reuse",
                "donate", "donation",
                "sort", "sorting", "separate", "separation",
                "shred", "shredding", "crush", "crushing",
                "decontaminate", "clean", "rinse", "wash",
                "disassemble", "dismantle",
                "chemical recycling", "mechanical recycling",
                "pyrolysis", "gasification",
                "anaerobic digestion", "biodegradation",
            },
            "ORGANIZATION": {
                "goodwill", "salvation army", "habitat for humanity",
                "epa", "environmental protection agency",
                "sierra club", "greenpeace", "world wildlife fund", "wwf",
                "recycling center", "recycling facility",
                "transfer station", "materials recovery facility", "mrf",
                "donation center", "drop-off center", "collection point",
                "thrift store", "thrift shop", "second-hand store",
                "best buy", "staples", "home depot", "lowes",
            },
            "REGULATION": {
                "rcra", "cercla", "superfund",
                "clean air act", "clean water act",
                "tsca", "toxic substances control act",
                "basel convention", "stockholm convention",
                "eu waste framework directive", "reach",
                "iso 14040", "iso 14044", "iso 14001",
                "epr", "extended producer responsibility",
                "rohs", "weee directive",
                "fda", "osha",
            },
            "ENVIRONMENTAL_CONCEPT": {
                "carbon footprint", "carbon neutral", "net zero",
                "circular economy", "linear economy",
                "cradle to cradle", "cradle to grave",
                "lifecycle assessment", "life cycle assessment", "lca",
                "embodied energy", "embodied carbon",
                "greenhouse gas", "global warming potential",
                "zero waste", "waste hierarchy",
                "reduce reuse recycle", "extended producer responsibility",
                "single-stream recycling", "dual-stream recycling",
                "waste-to-energy", "landfill gas",
                "ocean plastic", "great pacific garbage patch",
                "microplastic pollution", "plastic pollution",
                "deforestation", "biodiversity loss",
                "water footprint", "ecological footprint",
                "sustainability", "sustainable development",
                "greenwashing", "downcycling",
            },
        }

    def _build_regex_patterns(self) -> Dict[str, list]:
        """Build regex patterns for entities not well served by dictionaries."""
        def _c(patterns):
            return [re.compile(p, re.IGNORECASE) for p in patterns]

        return {
            "LOCATION": _c([
                r'\b\d{5}(?:-\d{4})?\b',
                r'\bnear me\b', r'\bin my area\b', r'\bmy location\b',
                r'\bmy city\b', r'\bmy town\b', r'\bnearby\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b',
            ]),
            "METRIC": _c([
                r'\b\d+\s*(?:kg|g|lb|oz|lbs|pounds|ounces|kilograms|grams)\b',
                r'\b\d+\s*(?:liters?|gallons?|ml|l)\b',
                r'\b\d+\s*(?:pieces?|items?|units?|tons?|tonnes?)\b',
                r'\b\d+\s*(?:percent|%)\b',
                r'\b\d+\s*(?:years?|months?|weeks?|days?|hours?)\b',
                r'\b\d+\s*(?:ppm|ppb|mg/l|ug/l)\b',
                r'\b\d+\s*(?:mj|kwh|kw|watts?|joules?)\b',
                r'\b\d+\s*(?:co2e?|co2-?eq)\b',
            ]),
            "TIME": _c([
                r'\btoday\b', r'\btomorrow\b', r'\byesterday\b',
                r'\bthis week\b', r'\bnext week\b',
                r'\bthis month\b', r'\bnext month\b',
                r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            ]),
        }
