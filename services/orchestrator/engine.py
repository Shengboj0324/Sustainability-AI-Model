"""Import-safe multimodal orchestration engine.

This module contains deterministic development/test adapters plus the routing
logic that the networked FastAPI orchestrator can delegate to. It deliberately
does not import model frameworks, create HTTP clients, or open external
connections at module import time.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from PIL import Image, ImageOps, UnidentifiedImageError

from services.shared.schemas import (
    Citation,
    ConfidenceScore,
    DetectedObject,
    FinalAnswer,
    KGResult,
    MultimodalRequest,
    OrganizationResult,
    RequestType,
    RetrievedDocument,
    ServiceError,
    ServiceMetadata,
    ServiceMode,
    TaskType,
    Timer,
    VisionResult,
    WarningMessage,
)


_MATERIAL_RULES: Dict[str, Dict[str, object]] = {
    "plastic": {
        "item": "plastic bottle",
        "material": "PET plastic",
        "bin": "recycle",
        "confidence": 0.72,
        "upcycling": ["self-watering planter", "seed starter", "small parts organizer"],
    },
    "bottle": {
        "item": "plastic bottle",
        "material": "PET plastic",
        "bin": "recycle",
        "confidence": 0.74,
        "upcycling": ["self-watering planter", "seed starter", "small parts organizer"],
    },
    "cardboard": {
        "item": "cardboard box",
        "material": "paper fiber",
        "bin": "recycle",
        "confidence": 0.78,
        "upcycling": ["drawer divider", "shipping organizer", "compost carbon layer"],
    },
    "pizza box": {
        "item": "pizza box",
        "material": "paper fiber with possible food residue",
        "bin": "conditional",
        "confidence": 0.7,
        "upcycling": ["compost clean torn sections", "craft template"],
    },
    "battery": {
        "item": "battery",
        "material": "battery chemistry",
        "bin": "special_dropoff",
        "confidence": 0.85,
        "hazard": "Batteries can spark or leak and should not go in curbside trash or recycling.",
        "upcycling": [],
    },
    "lithium": {
        "item": "lithium battery",
        "material": "lithium-ion battery",
        "bin": "special_dropoff",
        "confidence": 0.88,
        "hazard": "Lithium batteries are a fire risk if punctured, crushed, or placed in curbside bins.",
        "upcycling": [],
    },
    "motor oil": {
        "item": "used motor oil",
        "material": "petroleum oil",
        "bin": "hazardous_waste",
        "confidence": 0.86,
        "hazard": "Used motor oil must be taken to a used-oil collection or household hazardous waste site.",
        "upcycling": [],
    },
    "paint": {
        "item": "paint",
        "material": "coating/solvent mixture",
        "bin": "hazardous_waste",
        "confidence": 0.76,
        "hazard": "Oil-based paint and many solvents require hazardous waste handling.",
        "upcycling": [],
    },
    "glass": {
        "item": "glass bottle",
        "material": "glass",
        "bin": "recycle",
        "confidence": 0.73,
        "upcycling": ["vase", "propagation jar", "pantry storage"],
    },
}


_KNOWLEDGE_DOCS: List[RetrievedDocument] = [
    RetrievedDocument(
        doc_id="releaf-locality-limits",
        title="Local Recycling Rules Vary",
        source="ReleAF sustainability knowledge base",
        snippet=(
            "Curbside recycling rules vary by municipality; location-specific disposal "
            "decisions should be verified against the local waste authority."
        ),
        score=0.82,
        retrieval_mode="rule_based",
        doc_type="recycling_guideline",
        provenance={
            "corpus": "data/sustainability_knowledge_base.json",
            "ingestion_mode": "deterministic_test_fixture",
        },
        trust={"quality": "curated_fixture", "production": False},
    ),
    RetrievedDocument(
        doc_id="releaf-clean-dry-empty",
        title="Clean, Dry, and Empty Recycling Rule",
        source="ReleAF sustainability knowledge base",
        snippet=(
            "Containers are more likely to be accepted when they are empty, clean, "
            "and dry; food residue can contaminate paper and container streams."
        ),
        score=0.86,
        retrieval_mode="rule_based",
        doc_type="recycling_guideline",
        provenance={
            "corpus": "data/sustainability_knowledge_base.json",
            "ingestion_mode": "deterministic_test_fixture",
        },
        trust={"quality": "curated_fixture", "production": False},
    ),
    RetrievedDocument(
        doc_id="releaf-battery-safety",
        title="Battery and Hazardous Material Safety",
        source="ReleAF sustainability knowledge base",
        snippet=(
            "Batteries, motor oil, solvents, and suspicious chemical containers should "
            "be kept out of curbside recycling and routed to approved collection sites."
        ),
        score=0.9,
        retrieval_mode="rule_based",
        doc_type="safety_info",
        provenance={
            "corpus": "data/sustainability_knowledge_base.json",
            "ingestion_mode": "deterministic_test_fixture",
        },
        trust={"quality": "curated_fixture", "production": False},
    ),
    RetrievedDocument(
        doc_id="releaf-upcycling-fit",
        title="Upcycling Fit Heuristics",
        source="ReleAF sustainability knowledge base",
        snippet=(
            "Good upcycling candidates are clean, dry, structurally stable, and do not "
            "contain hazardous residues or sharp contamination."
        ),
        score=0.8,
        retrieval_mode="rule_based",
        doc_type="upcycling_project",
        provenance={
            "corpus": "data/sustainability_knowledge_base.json",
            "ingestion_mode": "deterministic_test_fixture",
        },
        trust={"quality": "curated_fixture", "production": False},
    ),
]

_RETRIEVAL_SIGNAL_TERMS = {
    "battery",
    "bin",
    "bottle",
    "cardboard",
    "chemical",
    "compost",
    "contamination",
    "dispose",
    "disposal",
    "donate",
    "drop",
    "glass",
    "hazard",
    "hazardous",
    "local",
    "material",
    "oil",
    "paint",
    "plastic",
    "recycle",
    "recycling",
    "reuse",
    "safe",
    "safety",
    "trash",
    "upcycle",
    "waste",
}


def _service_metadata(service: str, reason: str) -> ServiceMetadata:
    return ServiceMetadata(
        service=service,
        version="deterministic-fixture-v1",
        mode=ServiceMode.DETERMINISTIC_TEST,
        model_name="rule-based-development-adapter",
        model_version="fixture-v1",
        degraded_reason=reason,
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _informative_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9+_-]{1,}", _normalize_text(text))


def _assess_text_quality(text: str) -> float:
    normalized = _normalize_text(text)
    if not normalized:
        return 0.0

    tokens = _informative_tokens(normalized)
    if not tokens:
        return 0.02

    alnum_chars = sum(1 for char in normalized if char.isalnum())
    signal_ratio = alnum_chars / max(len(normalized), 1)
    unique_ratio = len(set(normalized)) / max(len(normalized), 1)
    token_score = min(1.0, len(tokens) / 6.0)
    length_score = min(1.0, len(normalized) / 48.0)
    quality = 0.55 * token_score + 0.45 * length_score

    if signal_ratio < 0.5:
        quality *= 0.35
    if unique_ratio < 0.15:
        quality *= 0.4
    return round(max(0.0, min(1.0, quality)), 3)


def _has_retrieval_signal(query: str) -> bool:
    normalized = _normalize_text(query)
    if not _informative_tokens(normalized):
        return False
    if _match_material(normalized)["material"] != "unknown material":
        return True
    return any(term in normalized for term in _RETRIEVAL_SIGNAL_TERMS)


def _match_material(text: str) -> Dict[str, object]:
    normalized = _normalize_text(text)
    for key, rule in _MATERIAL_RULES.items():
        if key in normalized:
            return rule
    return {
        "item": "unknown household item",
        "material": "unknown material",
        "bin": "unknown",
        "confidence": 0.34,
        "upcycling": [],
    }


def classify_request_type(request: MultimodalRequest) -> RequestType:
    if request.batch:
        return RequestType.BATCH
    has_text = bool(request.text)
    has_image = bool(request.image or request.image_url)
    if has_text and has_image:
        return RequestType.MULTIMODAL
    if has_image:
        return RequestType.IMAGE_ONLY
    if has_text:
        return RequestType.TEXT_ONLY
    raise ValueError("Request must include text, image, image_url, or batch input")


def classify_task_type(request: MultimodalRequest) -> tuple[TaskType, float]:
    text = _normalize_text(request.text)
    if request.context.get("task_type"):
        try:
            return TaskType(str(request.context["task_type"])), 0.95
        except ValueError:
            pass
    if any(token in text for token in ["near me", "where can i", "facility", "donate", "drop off", "drop-off"]):
        return TaskType.ORG_SEARCH, 0.82
    if any(token in text for token in ["upcycle", "upcycling", "reuse", "diy", "turn into", "make with"]):
        return TaskType.UPCYCLING_IDEA, 0.84
    if any(token in text for token in ["safe", "hazard", "toxic", "danger", "battery", "motor oil", "mercury", "asbestos"]):
        return TaskType.SAFETY_CHECK, 0.86
    if any(token in text for token in ["material", "made of", "property", "properties"]):
        return TaskType.MATERIAL_INFO, 0.72
    if request.image or any(token in text for token in ["recycle", "trash", "compost", "bin", "dispose"]):
        return TaskType.BIN_DECISION, 0.78
    return TaskType.THEORY_QA, 0.62


@dataclass
class DeterministicServiceClients:
    """Development/test adapters that never masquerade as production services."""

    image_min_quality: float = 0.45

    async def analyze_vision(self, request: MultimodalRequest) -> VisionResult:
        warnings: List[WarningMessage] = []
        image_quality = 0.5
        image_size: Optional[tuple[int, int]] = None

        if request.image:
            try:
                raw = base64.b64decode(request.image, validate=True)
                with Image.open(io.BytesIO(raw)) as img:
                    img = ImageOps.exif_transpose(img)
                    image_size = img.size
                    width, height = img.size
                    image_quality = min(1.0, max(0.2, (min(width, height) / 512.0)))
                    if min(width, height) < 64:
                        warnings.append(
                            WarningMessage(
                                code="LOW_IMAGE_RESOLUTION",
                                message="Image is very small; visual classification confidence is limited.",
                                service="vision_service",
                            )
                        )
            except (binascii.Error, UnidentifiedImageError, OSError, ValueError) as exc:
                warnings.append(
                    WarningMessage(
                        code="IMAGE_DECODE_FAILED",
                        message=f"Image could not be decoded in deterministic mode: {exc}",
                        severity="critical",
                        service="vision_service",
                    )
                )
                image_quality = 0.0
        elif request.image_url:
            warnings.append(
                WarningMessage(
                    code="IMAGE_URL_NOT_FETCHED",
                    message="Deterministic test mode does not fetch remote image URLs.",
                    service="vision_service",
                )
            )

        hint = request.text or str(request.context.get("vision_hint", "plastic bottle"))
        rule = _match_material(hint)
        confidence = float(rule["confidence"]) * (0.65 + 0.35 * image_quality)
        if confidence < 0.55:
            warnings.append(
                WarningMessage(
                    code="LOW_VISION_CONFIDENCE",
                    message="Vision result is low confidence and should be confirmed with user details.",
                    service="vision_service",
                )
            )

        obj = DetectedObject(
            label=str(rule["item"]),
            item_type=str(rule["item"]),
            material_type=str(rule["material"]),
            bin_type=str(rule["bin"]),
            confidence=round(confidence, 3),
            bbox=[0.1, 0.1, 0.8, 0.8] if image_size else None,
        )
        return VisionResult(
            objects=[obj],
            item_type=obj.item_type,
            material_type=obj.material_type,
            recommended_bin=obj.bin_type,
            confidence=ConfidenceScore.from_score(confidence, "Deterministic vision adapter confidence."),
            image_quality_score=round(image_quality, 3),
            warnings=warnings,
            metadata=_service_metadata(
                "vision_service",
                "No production checkpoint was invoked; using deterministic development fixture.",
            ),
        )

    async def retrieve(self, query: str, task_type: TaskType) -> List[RetrievedDocument]:
        normalized = _normalize_text(query)
        if not _has_retrieval_signal(normalized):
            return []
        docs = []
        for doc in _KNOWLEDGE_DOCS:
            score = doc.score
            if task_type == TaskType.SAFETY_CHECK and doc.doc_type == "safety_info":
                score += 0.05
            if "upcycl" in normalized and doc.doc_type == "upcycling_project":
                score += 0.05
            if "local" in normalized and doc.doc_id == "releaf-locality-limits":
                score += 0.05
            docs.append(doc.model_copy(update={"score": min(1.0, round(score, 3))}))
        docs.sort(key=lambda d: d.score, reverse=True)
        return docs[:3] if normalized else []

    async def query_kg(self, material: str, task_type: TaskType) -> KGResult:
        rule = _match_material(material)
        results: List[Dict[str, object]] = []
        hazard = rule.get("hazard")
        if hazard:
            results.append(
                {
                    "relationship": "material_has_hazard",
                    "material": rule["material"],
                    "hazard": hazard,
                    "recommended_bin": rule["bin"],
                    "confidence": rule["confidence"],
                }
            )
        if task_type == TaskType.UPCYCLING_IDEA:
            for idea in rule.get("upcycling", []):
                results.append(
                    {
                        "relationship": "material_can_become",
                        "source_material": rule["material"],
                        "target_product": idea,
                        "difficulty": "easy",
                        "confidence": 0.64,
                    }
                )
        if not results:
            results.append(
                {
                    "relationship": "item_to_material",
                    "item": rule["item"],
                    "material": rule["material"],
                    "recommended_bin": rule["bin"],
                    "confidence": rule["confidence"],
                }
            )
        return KGResult(
            query_type=task_type.value.lower(),
            results=results,
            confidence=ConfidenceScore.from_score(float(rule["confidence"]), "Rule-based KG fixture match."),
            explanation="Rule-based relationship reasoning from deterministic development fixture.",
            metadata=_service_metadata(
                "kg_service",
                "Neo4j/GNN was not invoked; using transparent rule-based fallback.",
            ),
        )

    async def search_orgs(self, request: MultimodalRequest, material: str) -> tuple[List[OrganizationResult], List[WarningMessage]]:
        warnings: List[WarningMessage] = []
        if not request.location:
            warnings.append(
                WarningMessage(
                    code="LOCATION_REQUIRED_FOR_LOCAL_RESULTS",
                    message="No location was supplied; organization results are generic examples, not local referrals.",
                    service="org_search_service",
                )
            )
        rule = _match_material(material)
        org_type = "household hazardous waste" if "hazard" in rule or rule["bin"] in {"hazardous_waste", "special_dropoff"} else "recycling facility"
        org = OrganizationResult(
            name="Example Municipal Recycling and HHW Center",
            org_type=org_type,
            services=["drop-off", "sorting guidance"],
            accepted_materials=[str(rule["material"])],
            distance_km=3.2 if request.location else None,
            notes="Deterministic fixture result; verify hours, fees, and accepted materials before visiting.",
        )
        return [org], warnings


class MultimodalOrchestratorEngine:
    def __init__(self, clients: Optional[DeterministicServiceClients] = None) -> None:
        self.clients = clients or DeterministicServiceClients()

    async def handle(self, request: MultimodalRequest) -> FinalAnswer:
        timer = Timer()
        warnings: List[WarningMessage] = [
            WarningMessage(
                code="DETERMINISTIC_TEST_MODE",
                message="Response used deterministic non-production adapters; do not treat as a model-backed production result.",
                severity="info",
                service="orchestrator",
            )
        ]
        errors: List[ServiceError] = []

        try:
            request_type = classify_request_type(request)
        except ValueError as exc:
            return FinalAnswer(
                answer_text="I need text, an image, an image URL, or a batch payload before I can help.",
                confidence=ConfidenceScore.from_score(0.0, "No usable input."),
                warnings=warnings,
                errors=[
                    ServiceError(
                        service="orchestrator",
                        code="INVALID_INPUT",
                        message=str(exc),
                        retryable=True,
                    )
                ],
                suggestions=["Add a question, an item description, or an image."],
                metadata={"mode": ServiceMode.DETERMINISTIC_TEST.value},
                processing_time_ms=timer.elapsed_ms,
            )

        task_type, task_confidence = classify_task_type(request)
        query = request.text
        text_quality = _assess_text_quality(query)

        if request_type == RequestType.TEXT_ONLY and text_quality < 0.2:
            warnings.append(
                WarningMessage(
                    code="LOW_TEXT_QUALITY",
                    message="The text input does not contain enough usable sustainability detail for grounded retrieval.",
                    service="orchestrator",
                )
            )
            return FinalAnswer(
                answer_text=(
                    "I do not have enough usable detail to give a grounded sustainability answer. "
                    "Please add the item, material, condition, and your local context if disposal rules matter."
                ),
                confidence=ConfidenceScore.from_score(0.18, "Text-only request lacked retrievable evidence."),
                warnings=warnings,
                suggestions=[
                    "Describe the item and material, for example: 'clean plastic bottle' or 'swollen lithium battery'.",
                    "Add your city or ZIP code for location-specific disposal rules.",
                ],
                metadata={
                    "mode": ServiceMode.DETERMINISTIC_TEST.value,
                    "request_type": request_type.value,
                    "task_type": task_type.value,
                    "task_confidence": round(task_confidence, 3),
                    "text_quality": text_quality,
                    "services_used": ["orchestrator"],
                    "evidence_counts": {
                        "vision_objects": 0,
                        "retrieved_documents": 0,
                        "kg_results": 0,
                        "organizations": 0,
                    },
                },
                processing_time_ms=timer.elapsed_ms,
            )

        vision: Optional[VisionResult] = None
        if request_type in {RequestType.IMAGE_ONLY, RequestType.MULTIMODAL}:
            try:
                vision = await self.clients.analyze_vision(request)
                warnings.extend(vision.warnings)
            except Exception as exc:
                errors.append(
                    ServiceError(
                        service="vision_service",
                        code="VISION_ADAPTER_FAILED",
                        message=str(exc),
                        retryable=False,
                    )
                )

        material_hint = query
        if vision and vision.material_type:
            material_hint = f"{query} {vision.item_type} {vision.material_type}"

        sources: List[RetrievedDocument] = []
        if task_type in {
            TaskType.BIN_DECISION,
            TaskType.THEORY_QA,
            TaskType.MATERIAL_INFO,
            TaskType.SAFETY_CHECK,
            TaskType.UPCYCLING_IDEA,
        }:
            sources = await self.clients.retrieve(material_hint, task_type)
            if not sources:
                warnings.append(
                    WarningMessage(
                        code="NO_RETRIEVAL_EVIDENCE",
                        message="No retrieval evidence was found; answer confidence is intentionally low.",
                        service="rag_service",
                    )
                )

        kg: Optional[KGResult] = None
        if task_type in {TaskType.BIN_DECISION, TaskType.UPCYCLING_IDEA, TaskType.MATERIAL_INFO, TaskType.SAFETY_CHECK}:
            kg = await self.clients.query_kg(material_hint, task_type)

        orgs: List[OrganizationResult] = []
        if task_type == TaskType.ORG_SEARCH:
            orgs, org_warnings = await self.clients.search_orgs(request, material_hint)
            warnings.extend(org_warnings)

        answer_text = self._synthesize_answer(task_type, request_type, query, vision, sources, kg, orgs)
        citations = [doc.to_citation() for doc in sources]
        confidence = self._calculate_confidence(task_confidence, vision, sources, kg, errors)
        if confidence.score < 0.55:
            warnings.append(
                WarningMessage(
                    code="LOW_FINAL_CONFIDENCE",
                    message="Evidence is incomplete or low-confidence; answer avoids definitive disposal claims.",
                    service="orchestrator",
                )
            )

        return FinalAnswer(
            answer_text=answer_text,
            confidence=confidence,
            citations=citations,
            sources=sources,
            warnings=warnings,
            errors=errors,
            suggestions=self._suggestions(task_type, confidence),
            metadata={
                "mode": ServiceMode.DETERMINISTIC_TEST.value,
                "request_type": request_type.value,
                "task_type": task_type.value,
                "task_confidence": round(task_confidence, 3),
                "text_quality": text_quality,
                "services_used": self._services_used(request_type, task_type),
                "evidence_counts": {
                    "vision_objects": len(vision.objects) if vision else 0,
                    "retrieved_documents": len(sources),
                    "kg_results": len(kg.results) if kg else 0,
                    "organizations": len(orgs),
                },
            },
            processing_time_ms=timer.elapsed_ms,
        )

    def _synthesize_answer(
        self,
        task_type: TaskType,
        request_type: RequestType,
        query: str,
        vision: Optional[VisionResult],
        sources: List[RetrievedDocument],
        kg: Optional[KGResult],
        orgs: List[OrganizationResult],
    ) -> str:
        item = vision.item_type if vision else _match_material(query)["item"]
        material = vision.material_type if vision else _match_material(query)["material"]
        bin_type = vision.recommended_bin if vision else _match_material(query)["bin"]
        citation_refs = " ".join(f"[{doc.doc_id}]" for doc in sources[:2])

        if task_type == TaskType.SAFETY_CHECK:
            hazard = None
            if kg:
                for result in kg.results:
                    hazard = result.get("hazard") or hazard
            if hazard:
                return (
                    f"Treat this as a safety-sensitive item: {hazard} Keep it intact, avoid heat or puncture, "
                    f"and use an approved drop-off or hazardous-waste channel. {citation_refs}".strip()
                )
            return (
                f"I do not see strong hazard evidence for {material}, but verify labels and local rules before disposal. "
                f"{citation_refs}".strip()
            )

        if task_type == TaskType.UPCYCLING_IDEA:
            ideas = []
            if kg:
                ideas = [str(r["target_product"]) for r in kg.results if r.get("relationship") == "material_can_become"]
            if not ideas:
                return (
                    f"I do not have enough safe upcycling evidence for {material}. Avoid upcycling items with chemical, "
                    f"battery, or unknown residue risk. {citation_refs}".strip()
                )
            return (
                f"For a clean {item}, good upcycling options are: {', '.join(ideas[:3])}. "
                f"Only reuse it if it is clean, dry, and free of hazardous residue. {citation_refs}".strip()
            )

        if task_type == TaskType.ORG_SEARCH:
            if not orgs:
                return "I could not find an organization result in deterministic mode. Add a location and material type."
            org = orgs[0]
            return (
                f"Start with {org.name}, a {org.org_type} fixture that accepts {', '.join(org.accepted_materials)}. "
                "Verify local availability, hours, fees, and acceptance rules before traveling."
            )

        if task_type == TaskType.MATERIAL_INFO:
            return (
                f"The best-supported material identification is {material} from {item}. "
                f"The fixture knowledge base recommends verifying local handling rules for final disposal. {citation_refs}".strip()
            )

        if task_type == TaskType.THEORY_QA:
            if not sources:
                return "I do not have retrieved evidence for this question, so I cannot give a grounded answer."
            return (
                f"Based on retrieved ReleAF knowledge, the key point is: {sources[0].snippet} {citation_refs}".strip()
            )

        if request_type == RequestType.IMAGE_ONLY:
            return (
                f"The image is classified as {item} ({material}) with a recommended bin of {bin_type}. "
                f"Because this is deterministic test mode, confirm the item and local rules before acting. {citation_refs}".strip()
            )

        return (
            f"This looks like {item} ({material}). Preliminary disposal route: {bin_type}. "
            f"Empty, clean, and dry recyclable containers; use special drop-off for hazardous or battery items. "
            f"{citation_refs}".strip()
        )

    def _calculate_confidence(
        self,
        task_confidence: float,
        vision: Optional[VisionResult],
        sources: List[RetrievedDocument],
        kg: Optional[KGResult],
        errors: Iterable[ServiceError],
    ) -> ConfidenceScore:
        scores = [task_confidence]
        if vision:
            scores.append(vision.confidence.score)
            scores.append(vision.image_quality_score)
        if sources:
            scores.append(max(doc.score for doc in sources))
        if kg:
            scores.append(kg.confidence.score)
        if list(errors):
            scores.append(0.2)
        if not sources and not vision and not kg:
            scores.append(0.2)
        score = sum(scores) / len(scores)
        return ConfidenceScore.from_score(score, "Weighted deterministic evidence confidence.")

    def _suggestions(self, task_type: TaskType, confidence: ConfidenceScore) -> List[str]:
        suggestions = []
        if confidence.score < 0.65:
            suggestions.append("Add a clearer photo or more item details.")
        if task_type in {TaskType.BIN_DECISION, TaskType.ORG_SEARCH, TaskType.SAFETY_CHECK}:
            suggestions.append("Add your city or ZIP code for location-specific rules.")
        if task_type == TaskType.UPCYCLING_IDEA:
            suggestions.append("Confirm the item is clean and free of chemical residue before reuse.")
        return suggestions

    def _services_used(self, request_type: RequestType, task_type: TaskType) -> List[str]:
        services = ["orchestrator"]
        if request_type in {RequestType.IMAGE_ONLY, RequestType.MULTIMODAL}:
            services.append("vision_service")
        if task_type != TaskType.ORG_SEARCH:
            services.append("rag_service")
        if task_type in {TaskType.BIN_DECISION, TaskType.UPCYCLING_IDEA, TaskType.MATERIAL_INFO, TaskType.SAFETY_CHECK}:
            services.append("kg_service")
        if task_type == TaskType.ORG_SEARCH:
            services.append("org_search_service")
        services.append("llm_synthesis_adapter")
        return services


def response_id(answer: FinalAnswer) -> str:
    payload = f"{answer.answer_text}|{answer.processing_time_ms}|{answer.metadata}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
