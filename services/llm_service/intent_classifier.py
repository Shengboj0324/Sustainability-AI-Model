"""
Intent Classification System for LLM Service — v2 (Semantic)

ARCHITECTURE:
- Primary: LLM-based semantic classification via prompt templates
- Fallback: Enhanced heuristic scoring (NOT simple regex — uses weighted semantic signals)
- API: Fully backward-compatible with v1 (same classify() and get_context_hints() signatures)

The LLM-based classifier is invoked by the orchestrator/LLM service when an LLM
backend is available. The heuristic fallback runs synchronously and handles cases
where the LLM is unavailable or for latency-sensitive paths.
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Callable, Awaitable
from enum import Enum
import logging
import hashlib

logger = logging.getLogger(__name__)


class IntentCategory(str, Enum):
    """Intent categories — expanded for deep sustainability reasoning"""
    WASTE_IDENTIFICATION = "waste_identification"
    DISPOSAL_GUIDANCE = "disposal_guidance"
    UPCYCLING_IDEAS = "upcycling_ideas"
    ORGANIZATION_SEARCH = "organization_search"
    SUSTAINABILITY_INFO = "sustainability_info"
    MATERIAL_SCIENCE = "material_science"
    SAFETY_CHECK = "safety_check"
    LIFECYCLE_ANALYSIS = "lifecycle_analysis"
    POLICY_REGULATION = "policy_regulation"
    GENERAL_QUESTION = "general_question"
    CHITCHAT = "chitchat"


# Maps intent names (from LLM output) to IntentCategory enum values
_INTENT_ALIAS_MAP: Dict[str, IntentCategory] = {
    # Direct matches
    "waste_identification": IntentCategory.WASTE_IDENTIFICATION,
    "disposal_guidance": IntentCategory.DISPOSAL_GUIDANCE,
    "upcycling_ideas": IntentCategory.UPCYCLING_IDEAS,
    "organization_search": IntentCategory.ORGANIZATION_SEARCH,
    "sustainability_info": IntentCategory.SUSTAINABILITY_INFO,
    "material_science": IntentCategory.MATERIAL_SCIENCE,
    "safety_check": IntentCategory.SAFETY_CHECK,
    "lifecycle_analysis": IntentCategory.LIFECYCLE_ANALYSIS,
    "policy_regulation": IntentCategory.POLICY_REGULATION,
    "general_question": IntentCategory.GENERAL_QUESTION,
    "chitchat": IntentCategory.CHITCHAT,
    # Aliases from task type classification
    "bin_decision": IntentCategory.DISPOSAL_GUIDANCE,
    "upcycling_idea": IntentCategory.UPCYCLING_IDEAS,
    "environmental_qa": IntentCategory.SUSTAINABILITY_INFO,
    "org_search": IntentCategory.ORGANIZATION_SEARCH,
    "general": IntentCategory.GENERAL_QUESTION,
}


class IntentClassifier:
    """
    Semantic intent classifier with LLM-based primary and heuristic fallback.

    Usage:
        classifier = IntentClassifier()
        intent, confidence = classifier.classify("How do I recycle motor oil?")

        # For LLM-based classification (async):
        intent, confidence = await classifier.classify_with_llm(
            "How do I recycle motor oil?", llm_call_fn
        )
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[IntentCategory, float]] = {}
        self._cache_max_size = 1000

        # Weighted semantic signals for heuristic fallback
        # Each entry: (weight, compiled_patterns)
        self._signals = self._build_semantic_signals()

        logger.info(
            f"Intent classifier initialized: {len(IntentCategory)} categories, "
            f"heuristic fallback with {sum(len(v[1]) for v in self._signals.values())} signals"
        )

    # ------------------------------------------------------------------
    # PUBLIC API — synchronous (backward compatible)
    # ------------------------------------------------------------------

    def classify(self, text: str) -> Tuple[IntentCategory, float]:
        """
        Classify user intent using enhanced heuristic scoring.

        This is the synchronous fallback. For LLM-based classification,
        use classify_with_llm() (async).

        Args:
            text: User input text

        Returns:
            (intent_category, confidence_score)
        """
        if not text or not isinstance(text, str) or not text.strip():
            return IntentCategory.GENERAL_QUESTION, 0.3

        cache_key = hashlib.sha256(text.lower().strip().encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        text_clean = text.strip().lower()[:1000]
        result = self._heuristic_classify(text_clean)

        if len(self._cache) >= self._cache_max_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # PUBLIC API — async LLM-based classification
    # ------------------------------------------------------------------

    async def classify_with_llm(
        self,
        text: str,
        llm_call: Callable[[str], Awaitable[str]],
    ) -> Tuple[IntentCategory, float]:
        """
        Classify intent using the LLM for semantic understanding.

        Args:
            text: User input text
            llm_call: Async function that takes a prompt string and returns LLM output string.

        Returns:
            (intent_category, confidence_score)
        """
        if not text or not text.strip():
            return IntentCategory.GENERAL_QUESTION, 0.3

        try:
            from .prompt_templates.sustainability_prompts import classify_intent_prompt

            prompt = classify_intent_prompt(text)
            raw_output = await llm_call(prompt)

            # Parse JSON response
            raw_output = raw_output.strip()
            # Handle cases where LLM wraps in markdown code blocks
            if raw_output.startswith("```"):
                raw_output = re.sub(r"^```(?:json)?\s*", "", raw_output)
                raw_output = re.sub(r"\s*```$", "", raw_output)

            parsed = json.loads(raw_output)
            intent_str = parsed.get("intent", "general_question").lower().strip()
            confidence = float(parsed.get("confidence", 0.7))

            intent = _INTENT_ALIAS_MAP.get(intent_str, IntentCategory.GENERAL_QUESTION)
            confidence = max(0.0, min(1.0, confidence))

            logger.info(f"LLM intent: {intent.value} ({confidence:.2f})")
            return intent, confidence

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"LLM intent parse failed: {e}, falling back to heuristic")
            return self.classify(text)
        except Exception as e:
            logger.error(f"LLM intent classification error: {e}", exc_info=True)
            return self.classify(text)

    def parse_llm_intent_response(self, raw_output: str) -> Tuple[IntentCategory, float]:
        """
        Parse a raw LLM response into intent + confidence.
        Useful when the LLM call is made externally (e.g., by the orchestrator).
        """
        try:
            raw_output = raw_output.strip()
            if raw_output.startswith("```"):
                raw_output = re.sub(r"^```(?:json)?\s*", "", raw_output)
                raw_output = re.sub(r"\s*```$", "", raw_output)
            parsed = json.loads(raw_output)
            intent_str = parsed.get("intent", "general_question").lower().strip()
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.7))))
            intent = _INTENT_ALIAS_MAP.get(intent_str, IntentCategory.GENERAL_QUESTION)
            return intent, confidence
        except Exception:
            return IntentCategory.GENERAL_QUESTION, 0.3

    # ------------------------------------------------------------------
    # CONTEXT HINTS — maps intent to service routing config
    # ------------------------------------------------------------------

    def get_context_hints(self, intent: IntentCategory) -> Dict[str, any]:
        """Get context hints for each intent to guide LLM response and service routing."""
        hints = {
            IntentCategory.WASTE_IDENTIFICATION: {
                "use_vision": True, "use_rag": True, "use_kg": False,
                "task_type": "bin_decision",
                "response_style": "informative", "max_length": 500,
            },
            IntentCategory.DISPOSAL_GUIDANCE: {
                "use_vision": True, "use_rag": True, "use_kg": False,
                "task_type": "bin_decision",
                "response_style": "instructional", "max_length": 500,
            },
            IntentCategory.UPCYCLING_IDEAS: {
                "use_vision": True, "use_rag": True, "use_kg": True,
                "task_type": "upcycling_idea",
                "response_style": "creative", "max_length": 800,
            },
            IntentCategory.ORGANIZATION_SEARCH: {
                "use_vision": False, "use_rag": True, "use_kg": False,
                "use_org_search": True,
                "task_type": "org_search",
                "response_style": "helpful", "max_length": 600,
            },
            IntentCategory.SUSTAINABILITY_INFO: {
                "use_vision": False, "use_rag": True, "use_kg": False,
                "task_type": "environmental_qa",
                "response_style": "educational", "max_length": 1000,
            },
            IntentCategory.MATERIAL_SCIENCE: {
                "use_vision": False, "use_rag": True, "use_kg": True,
                "task_type": "material_science",
                "response_style": "scientific", "max_length": 1000,
            },
            IntentCategory.SAFETY_CHECK: {
                "use_vision": False, "use_rag": True, "use_kg": False,
                "task_type": "safety_check",
                "response_style": "cautionary", "max_length": 800,
            },
            IntentCategory.LIFECYCLE_ANALYSIS: {
                "use_vision": False, "use_rag": True, "use_kg": True,
                "task_type": "lifecycle_analysis",
                "response_style": "analytical", "max_length": 1200,
            },
            IntentCategory.POLICY_REGULATION: {
                "use_vision": False, "use_rag": True, "use_kg": False,
                "task_type": "policy_regulation",
                "response_style": "authoritative", "max_length": 1000,
            },
            IntentCategory.GENERAL_QUESTION: {
                "use_vision": False, "use_rag": True, "use_kg": False,
                "task_type": "general",
                "response_style": "informative", "max_length": 800,
            },
            IntentCategory.CHITCHAT: {
                "use_vision": False, "use_rag": False, "use_kg": False,
                "task_type": "general",
                "response_style": "friendly", "max_length": 150,
            },
        }
        return hints.get(intent, hints[IntentCategory.GENERAL_QUESTION])

    # ------------------------------------------------------------------
    # PRIVATE — enhanced heuristic classification
    # ------------------------------------------------------------------

    def _heuristic_classify(self, text: str) -> Tuple[IntentCategory, float]:
        """
        Enhanced heuristic classification using weighted semantic signals.
        Much more sophisticated than simple regex pattern counting.
        """
        scores: Dict[IntentCategory, float] = {cat: 0.0 for cat in IntentCategory}

        for intent, (weight, patterns) in self._signals.items():
            for pattern in patterns:
                if pattern.search(text):
                    scores[intent] += weight

        best = max(scores, key=scores.get)
        raw_score = scores[best]

        if raw_score <= 0:
            return IntentCategory.GENERAL_QUESTION, 0.3

        # Normalize confidence: map raw score to 0.4-0.95 range
        confidence = min(0.95, 0.4 + raw_score * 0.15)
        return best, round(confidence, 2)

    def _build_semantic_signals(self) -> Dict[IntentCategory, Tuple[float, list]]:
        """Build weighted semantic signal patterns for heuristic classification."""
        def _compile(patterns: List[str]) -> list:
            return [re.compile(p, re.IGNORECASE) for p in patterns]

        return {
            IntentCategory.CHITCHAT: (2.0, _compile([
                r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))[\s!.?,]*$",
                r"^(thank(s| you)[\w\s]*)[\s!.?,]*$",
                r"^(bye|goodbye|see you|farewell)[\s!.?,]*$",
                r"^(how are you|what'?s up|how'?s it going)[\s!.?,]*$",
                r"^(yes|no|ok|okay|sure|alright|yep|nope)[\s!.?,]*$",
            ])),
            IntentCategory.SAFETY_CHECK: (1.5, _compile([
                r"\b(safe|danger|toxic|hazard|harmful|poison|asbestos|mercury|lead)\b",
                r"\b(chemical|exposure|inhal|carcino|radioactive|flammable)\b",
                r"\b(ppe|protective equipment|gloves|mask|goggles)\b",
                r"\b(osha|epa|hazmat|msds|sds)\b",
                r"\b(burn|incinerat|ignite|set\s+fire)\b.*(styrofoam|plastic|waste|trash|foam)",
                r"\b(swallow|ingest|eat|ate|drank|drink|consume)\b",
                r"\bmy\s+(kid|child|dog|cat|baby|toddler)\b.*(swallow|eat|ate|drank|bit|chew)",
                r"\b(mix|combin)\b.*(bleach|ammonia|chemical|acid|chlorine)\b",
                r"\b(bleach)\b.*\b(ammonia)\b",
                r"\b(ammonia)\b.*\b(bleach)\b",
                r"\b(is it|is this|is that)\b.*(safe|okay|ok)\b.*(burn|throw|crush|break|mix)",
                r"\b(motor\s+oil|used\s+oil|antifreeze)\b.*(trash|throw|dump|pour|drain)\b",
                r"\b(throw|toss|put|dump)\b.*(motor\s+oil|used\s+oil|antifreeze|paint|solvent)\b.*(trash|garbage|bin|drain)\b",
            ])),
            IntentCategory.DISPOSAL_GUIDANCE: (1.0, _compile([
                r"\b(which|what|correct)\s+(bin|container|dumpster)\b",
                r"\b(dispose|throw\s*(away|out)|discard|get\s*rid)\b",
                r"\b(goes?\s+in|put\s+in|belongs?\s+in)\b.*(bin|trash|recycl|compost)",
                r"\b(recycle|recyclable|recycling)\b",
                r"\b(can\s+i|is\s+this|is\s+it|should\s+i)\b.*\b(recycle|recyclable|compost|trash|bin)\b",
                r"\b(put|place|toss|throw)\b.*\b(recycl|bin|trash|compost|curbside)\b",
                r"\b(landfill|waste\s*bin|garbage\s*can|blue\s*bin|green\s*bin)\b",
                r"\b(ok|okay|safe|allowed)\b.*\b(throw|put|dispose|toss)\b.*\b(trash|bin|garbage)\b",
                r"\b(lithium|alkaline|lead[\s-]?acid)\b.*\b(batter)\b",
                r"\b(motor\s+oil|used\s+oil|cooking\s+oil|antifreeze|brake\s+fluid)\b.*\b(trash|throw|dispose|drain|pour)\b",
            ])),
            IntentCategory.WASTE_IDENTIFICATION: (1.0, _compile([
                r"\b(what\s+is|identify|recognize|classify|detect)\b.*\b(this|item|object|material)",
                r"\b(what\s+(type|kind|material|sort))\b",
                r"\b(made\s+of|composed\s+of|consists?\s+of)\b",
                r"\b(is\s+(this|it))\b.*\b(plastic|metal|glass|paper|organic)\b",
            ])),
            IntentCategory.UPCYCLING_IDEAS: (1.0, _compile([
                r"\b(upcycl|repurpos|diy|craft)",
                r"\b(creative|idea|project).*\b(reus|repurpos|old|make)",
                r"\b(turn|transform|convert)\b.*\b(into|something|useful)\b",
                r"\b(second\s+life|new\s+use|alternative\s+use)\b",
                r"\b(what\s+can\s+i|how\s+can\s+i)\b.*(reus|mak|do\s+with|repurpos)",
                r"\b(ideas?\s+for)\b.*(reus|repurpos|recycl|old)",
            ])),
            IntentCategory.ORGANIZATION_SEARCH: (1.0, _compile([
                r"\b(where|find|locate|search|look)\b.*(donat|charit|recycl|facilit|center)",
                r"\b(near\s+me|nearby|local|in\s+my\s+area|closest)\b",
                r"\b(recycling\s+(center|facilit|depot|station))",
                r"\b(drop[\s-]?off|collection\s+point|take[\s-]?back)\b",
                r"\b(goodwill|salvation\s+army|habitat)\b",
                r"\b(thrift\s+store|donation\s+center)\b",
            ])),
            IntentCategory.MATERIAL_SCIENCE: (1.0, _compile([
                r"\b(polymer|resin|alloy|composite|molecular|chemical\s+composition)\b",
                r"\b(hdpe|ldpe|pet|pp|ps|pvc|abs|nylon|polycarbonate|polyethylene)\b",
                r"\b(biodegrad|decompos|photo[\s-]?degrad).*(time|rate|how\s+long)",
                r"\b(melting\s+point|tensile|density|thermal)\b",
                r"\b(plastic\s+(type|number)|resin\s+code|recycling\s+(number|symbol)|#[1-7])\b",
                r"\b(what\s+is|properties\s+of|chemistry\s+of|composition\s+of)\b.*(material|plastic|glass|metal)",
                r"\b(what|which)\b.*\b(plastic|material|number|type|kind)\b.*\b(is|made|bottle|cup|container)\b",
                r"\b(how\s+(many|long))\b.*\b(times?|years?)\b.*\b(recycl|decompos|break\s+down|degrad)\b",
                r"\b(how\s+many\s+times)\b.*\b(recycl)\b",
                r"\b(how\s+long)\b.*(decompos|break\s+down|degrad|last|take\s+to)\b",
                r"\b(what\s+(is|are))\b.*(pla|pha|pbs|pbat|bioplastic|polylactic|pfas|bpa|styrene)\b",
                r"\b(difference|between)\b.*\b(#[1-7]|plastic|pet|hdpe|pvc|ldpe|pp|ps)\b",
            ])),
            IntentCategory.SUSTAINABILITY_INFO: (0.8, _compile([
                r"\b(why|importance|benefit|impact|effect)\b.*(recycl|sustainab|environment)",
                r"\b(carbon|ecological|environmental)\b.*(footprint|impact|effect|crisis)",
                r"\b(climate\s+change|global\s+warming|greenhouse|pollution|deforestation)\b",
                r"\b(sustainab|eco[\s-]?friendly|green\s+living|zero\s+waste)\b",
                r"\b(circular\s+economy|waste\s+reduction|conservation)\b",
                r"\b(ocean|microplastic|single[\s-]?use|disposable).*(impact|problem|issue|crisis)",
                r"\b(recycling\s+rate|recovery\s+rate|diversion\s+rate)\b",
                r"\b(what\s+happen|where\s+does)\b.*\b(recycl|trash|waste|garbage)\b",
                r"\b(what\s+(is|does|are))\b.*\b(chasing\s+arrows|recycling\s+symbol|single[\s-]?stream)\b",
                r"\b(compostable|biodegradable)\b.*\b(same|difference|vs|mean|actually|really)\b",
                r"\b(actually|really)\b.*\b(compostable|biodegradable|recyclable)\b",
                r"\b(what\s+is)\b.*\b(single[\s-]?stream|curbside|mrf)\b",
            ])),
            IntentCategory.LIFECYCLE_ANALYSIS: (1.0, _compile([
                r"\b(lifecycle|life[\s-]?cycle|lca|cradle[\s-]?to[\s-]?(grave|cradle))\b",
                r"\b(carbon\s+footprint|embodied\s+energy|water\s+footprint)\b.*(of|for|compar)",
                r"\b(environment).*(impact|cost|compar).*(vs|versus|or|compared)\b",
                r"\b(better|worse|greener|more\s+sustainable)\b.*(than|vs|versus|compared)\b",
            ])),
            IntentCategory.POLICY_REGULATION: (1.0, _compile([
                r"\b(law|regulation|policy|legislation|mandate|ban|ordinance)",
                r"\b(rcra|cercla|clean\s+air|clean\s+water|basel\s+convention)\b",
                r"\b(epr|extended\s+producer|producer\s+responsibility)\b",
                r"\b(complian|legal|illegal|penalty|fine|enforcement).*(waste|recycl|dispos)",
                r"\b(regulation|rule|law).*(govern|require|mandate|control)",
            ])),
            IntentCategory.GENERAL_QUESTION: (0.5, _compile([
                r"\b(how\s+does|how\s+do|what\s+is|what\s+are|explain)\b.*(recycl|compost|waste\s+manage)",
                r"\b(difference|types?\s+of|categories?\s+of)\b.*(waste|recycl|material)",
                r"\b(process|system|method|work)\b.*(recycl|waste|compost)",
            ])),
        }

