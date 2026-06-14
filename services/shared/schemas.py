"""Shared machine-readable contracts for ReleAF AI services.

These schemas are intentionally lightweight and import-safe. They are suitable
for API gateway clients, orchestrator tests, and deterministic development
fallbacks without importing model frameworks or opening network connections.
"""

from __future__ import annotations

from enum import Enum
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class RequestType(str, Enum):
    TEXT_ONLY = "TEXT_ONLY"
    IMAGE_ONLY = "IMAGE_ONLY"
    MULTIMODAL = "MULTIMODAL"
    BATCH = "BATCH"


class TaskType(str, Enum):
    BIN_DECISION = "BIN_DECISION"
    UPCYCLING_IDEA = "UPCYCLING_IDEA"
    ORG_SEARCH = "ORG_SEARCH"
    THEORY_QA = "THEORY_QA"
    MATERIAL_INFO = "MATERIAL_INFO"
    SAFETY_CHECK = "SAFETY_CHECK"


class ServiceMode(str, Enum):
    PRODUCTION = "production"
    DEGRADED = "degraded"
    DETERMINISTIC_TEST = "deterministic_test"


class Location(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None


class WarningMessage(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "critical"] = "warning"
    service: Optional[str] = None


class ServiceError(BaseModel):
    service: str
    code: str
    message: str
    retryable: bool = False


class ServiceMetadata(BaseModel):
    service: str
    version: str = "unknown"
    mode: ServiceMode = ServiceMode.PRODUCTION
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    degraded_reason: Optional[str] = None


class ConfidenceScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    level: Literal["very_low", "low", "medium", "high"]
    rationale: Optional[str] = None

    @classmethod
    def from_score(cls, score: float, rationale: Optional[str] = None) -> "ConfidenceScore":
        bounded = max(0.0, min(1.0, score))
        if bounded >= 0.8:
            level = "high"
        elif bounded >= 0.55:
            level = "medium"
        elif bounded >= 0.3:
            level = "low"
        else:
            level = "very_low"
        return cls(score=round(bounded, 3), level=level, rationale=rationale)


class Citation(BaseModel):
    id: str
    title: str
    source: str
    snippet: str
    url: Optional[str] = None
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    doc_id: str
    title: str
    source: str
    snippet: str
    score: float = Field(..., ge=0.0, le=1.0)
    retrieval_mode: Literal["dense", "sparse", "hybrid", "rule_based"]
    doc_type: str = "general_knowledge"
    provenance: Dict[str, Any] = Field(default_factory=dict)
    trust: Dict[str, Any] = Field(default_factory=dict)

    def to_citation(self) -> Citation:
        return Citation(
            id=self.doc_id,
            title=self.title,
            source=self.source,
            snippet=self.snippet,
            score=self.score,
            provenance=self.provenance,
        )


class DetectedObject(BaseModel):
    label: str
    item_type: str
    material_type: str
    bin_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Optional[List[float]] = None


class VisionResult(BaseModel):
    objects: List[DetectedObject] = Field(default_factory=list)
    item_type: Optional[str] = None
    material_type: Optional[str] = None
    recommended_bin: Optional[str] = None
    confidence: ConfidenceScore
    image_quality_score: float = Field(..., ge=0.0, le=1.0)
    warnings: List[WarningMessage] = Field(default_factory=list)
    metadata: ServiceMetadata


class KGResult(BaseModel):
    query_type: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: ConfidenceScore
    explanation: str
    metadata: ServiceMetadata


class LLMResult(BaseModel):
    answer: str
    confidence: ConfidenceScore
    citations: List[Citation] = Field(default_factory=list)
    warnings: List[WarningMessage] = Field(default_factory=list)
    metadata: ServiceMetadata


class OrganizationResult(BaseModel):
    name: str
    org_type: str
    services: List[str]
    accepted_materials: List[str]
    distance_km: Optional[float] = None
    url: Optional[str] = None
    notes: Optional[str] = None


class MultimodalRequest(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    image: Optional[str] = None
    image_url: Optional[str] = None
    location: Optional[Location] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    batch: Optional[List[Dict[str, Any]]] = None
    enable_fallback: bool = True
    require_high_confidence: bool = False

    @field_validator("image_url")
    @classmethod
    def validate_one_image_source(cls, value: Optional[str], info):
        if value and info.data.get("image"):
            raise ValueError("Provide either image or image_url, not both")
        return value

    @property
    def text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user" and message.content:
                return message.content
        return ""


class FinalAnswer(BaseModel):
    answer_text: str
    confidence: ConfidenceScore
    citations: List[Citation] = Field(default_factory=list)
    sources: List[RetrievedDocument] = Field(default_factory=list)
    warnings: List[WarningMessage] = Field(default_factory=list)
    errors: List[ServiceError] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float


class Timer:
    def __init__(self) -> None:
        self._start = perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return round((perf_counter() - self._start) * 1000, 3)
