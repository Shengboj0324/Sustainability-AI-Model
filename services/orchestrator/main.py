"""
Orchestrator Service - Advanced Multi-Modal Intelligence & Request Routing

CRITICAL ENHANCEMENTS FOR PRODUCTION:
- Handles ANY user input (text, image, or both)
- Intelligent fallback strategies for low-quality images
- Confidence scoring and uncertainty handling
- Partial answer generation when data is incomplete
- Multi-stage reasoning with quality validation
- Graceful degradation with helpful suggestions
- Advanced error recovery
- Rich answer formatting with markdown and citations
- User feedback integration for continuous improvement
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal, Tuple
import httpx
import asyncio
import logging
from datetime import datetime
from enum import Enum
import yaml
import hashlib
import json
import sys
from pathlib import Path

# Import answer formatter
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.answer_formatter import AnswerFormatter, AnswerType, FormattedAnswer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ReleAF AI Orchestrator - Advanced Multi-Modal Intelligence",
    description="Production-grade request routing with intelligent fallback strategies",
    version="2.0.0"
)

# Load configuration
with open("configs/orchestrator.yaml", "r") as f:
    config = yaml.safe_load(f)


class ConfidenceLevel(str, Enum):
    """Confidence levels for responses"""
    HIGH = "high"  # 0.8+
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"  # 0.3-0.5
    VERY_LOW = "very_low"  # <0.3


class OrchestratorRequest(BaseModel):
    """Advanced orchestrator request with validation"""
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Chat messages")
    image: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="Image URL")
    location: Optional[Dict[str, float]] = Field(None, description="User location (lat, lon)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    enable_fallback: bool = Field(True, description="Enable fallback strategies")
    require_high_confidence: bool = Field(False, description="Only return high-confidence answers")


class OrchestratorResponse(BaseModel):
    """Advanced orchestrator response with confidence and quality metrics"""
    response: str = Field(..., description="Final answer (plain text)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence (0-1)")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence category")
    sources: Optional[List[Dict[str, str]]] = Field(None, description="Information sources")
    suggestions: Optional[List[str]] = Field(None, description="Follow-up suggestions")
    warnings: Optional[List[str]] = Field(None, description="Quality warnings")
    fallback_used: bool = Field(False, description="Whether fallback strategies were used")
    partial_answer: bool = Field(False, description="Whether this is a partial answer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    processing_time_ms: float = Field(..., description="Total processing time")

    # Quality metrics
    image_quality_score: Optional[float] = Field(None, description="Image quality (0-1)")
    text_quality_score: Optional[float] = Field(None, description="Text quality (0-1)")
    reasoning_steps: Optional[List[str]] = Field(None, description="Reasoning chain")

    # Rich formatting (NEW)
    formatted_answer: Optional[Dict[str, Any]] = Field(None, description="Rich formatted answer with markdown/HTML")
    answer_type: Optional[str] = Field(None, description="Answer type (how_to, factual, creative, etc.)")
    citations: Optional[List[Dict[str, Any]]] = Field(None, description="Structured citations")

    # Feedback integration (NEW)
    response_id: Optional[str] = Field(None, description="Unique response ID for feedback tracking")


class ConfidenceCalculator:
    """
    Calculate confidence scores for responses

    CRITICAL: Provides transparency about answer quality
    """

    @staticmethod
    def calculate_overall_confidence(
        vision_confidence: Optional[float] = None,
        llm_confidence: Optional[float] = None,
        rag_confidence: Optional[float] = None,
        image_quality: Optional[float] = None,
        text_quality: Optional[float] = None
    ) -> Tuple[float, ConfidenceLevel]:
        """
        Calculate overall confidence from multiple sources

        Returns: (confidence_score, confidence_level)
        """
        scores = []
        weights = []

        if vision_confidence is not None:
            scores.append(vision_confidence)
            weights.append(0.3)

        if llm_confidence is not None:
            scores.append(llm_confidence)
            weights.append(0.4)

        if rag_confidence is not None:
            scores.append(rag_confidence)
            weights.append(0.2)

        if image_quality is not None:
            scores.append(image_quality)
            weights.append(0.05)

        if text_quality is not None:
            scores.append(text_quality)
            weights.append(0.05)

        if not scores:
            return 0.5, ConfidenceLevel.MEDIUM

        # Weighted average
        total_weight = sum(weights[:len(scores)])
        confidence = sum(s * w for s, w in zip(scores, weights[:len(scores)])) / total_weight

        # Determine level
        if confidence >= 0.8:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            level = ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW

        return confidence, level


class RequestClassifier:
    """
    Advanced request classifier with quality assessment

    CRITICAL: Handles ANY input combination with quality scoring
    """

    @staticmethod
    def classify_request_type(request: OrchestratorRequest) -> str:
        """Determine request type"""
        has_image = bool(request.image or request.image_url)
        has_text = bool(request.messages)

        if has_image and has_text:
            return "MULTIMODAL"
        elif has_image:
            return "IMAGE_ONLY"
        elif has_text:
            return "TEXT_ONLY"
        else:
            raise ValueError("Invalid request: no image or text provided")

    @staticmethod
    def assess_text_quality(messages: List[Dict[str, Any]]) -> float:
        """
        Assess text input quality

        Returns: quality_score (0-1)
        """
        if not messages:
            return 0.0

        # Get last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return 0.0

        quality = 1.0

        # Check length
        if len(user_message) < 5:
            quality *= 0.5
        elif len(user_message) < 10:
            quality *= 0.7

        # Check if it's just punctuation or numbers
        if user_message.replace(" ", "").replace("?", "").replace("!", "").replace(".", "") == "":
            quality *= 0.3

        # Check for meaningful words
        words = user_message.split()
        if len(words) < 2:
            quality *= 0.6

        return quality
    
    @staticmethod
    async def classify_task_type(request: OrchestratorRequest) -> Tuple[str, float]:
        """
        Determine task type from user intent with confidence

        Returns: (task_type, confidence)
        """
        # Simple keyword-based classification (can be replaced with LLM)
        if not request.messages:
            # Image-only request
            if request.image or request.image_url:
                return "BIN_DECISION", 0.7
            return "THEORY_QA", 0.3

        last_message = request.messages[-1].get("content", "").lower()

        # Keyword matching with confidence scoring
        task_scores = {
            "BIN_DECISION": 0.0,
            "UPCYCLING_IDEA": 0.0,
            "ORG_SEARCH": 0.0,
            "SAFETY_CHECK": 0.0,
            "MATERIAL_INFO": 0.0,
            "THEORY_QA": 0.1  # Base score
        }

        # Bin decision keywords
        bin_keywords = ["bin", "recycle", "dispose", "throw", "trash", "garbage", "waste"]
        task_scores["BIN_DECISION"] += sum(0.2 for word in bin_keywords if word in last_message)

        # Upcycling keywords
        upcycle_keywords = ["upcycle", "reuse", "make", "create", "diy", "craft", "repurpose"]
        task_scores["UPCYCLING_IDEA"] += sum(0.2 for word in upcycle_keywords if word in last_message)

        # Organization search keywords
        org_keywords = ["where", "find", "location", "near", "facility", "center", "place"]
        task_scores["ORG_SEARCH"] += sum(0.15 for word in org_keywords if word in last_message)

        # Safety keywords
        safety_keywords = ["safe", "danger", "toxic", "hazard", "harmful", "poison"]
        task_scores["SAFETY_CHECK"] += sum(0.2 for word in safety_keywords if word in last_message)

        # Material info keywords
        material_keywords = ["material", "property", "chemistry", "composition", "made of", "type"]
        task_scores["MATERIAL_INFO"] += sum(0.15 for word in material_keywords if word in last_message)

        # Get best match
        best_task = max(task_scores, key=task_scores.get)
        confidence = min(1.0, task_scores[best_task])

        # If confidence is too low, default to THEORY_QA
        if confidence < 0.3:
            return "THEORY_QA", 0.5

        return best_task, confidence


class FallbackStrategy:
    """
    Intelligent fallback strategies for handling failures

    CRITICAL: Ensures users ALWAYS get a helpful response
    """

    @staticmethod
    def generate_fallback_response(
        error_type: str,
        request: OrchestratorRequest,
        partial_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate helpful fallback response when primary workflow fails

        Returns: response dict with fallback content
        """
        response_parts = []
        suggestions = []
        warnings = []

        if error_type == "IMAGE_QUALITY_TOO_LOW":
            response_parts.append(
                "I'm having difficulty analyzing this image due to quality issues. "
                "The image appears to be blurry, too dark, or low resolution."
            )
            suggestions.extend([
                "Try taking a clearer photo with better lighting",
                "Move closer to the object",
                "Clean the camera lens",
                "Describe the item in text and I can help identify it"
            ])
            warnings.append("Image quality too low for reliable analysis")

        elif error_type == "IMAGE_PROCESSING_FAILED":
            response_parts.append(
                "I encountered an issue processing the image. "
            )
            suggestions.extend([
                "Try uploading a different image format (JPG, PNG)",
                "Ensure the image file is not corrupted",
                "Describe what you see and I can provide guidance"
            ])
            warnings.append("Image processing failed")

        elif error_type == "NO_OBJECTS_DETECTED":
            response_parts.append(
                "I couldn't detect any clear objects in the image. "
            )
            suggestions.extend([
                "Try taking a photo with the object more centered",
                "Ensure good lighting and contrast",
                "Move closer to the object",
                "Describe the item and I can help identify it"
            ])
            warnings.append("No objects detected in image")

        elif error_type == "AMBIGUOUS_REQUEST":
            response_parts.append(
                "I'm not quite sure what you're asking about. "
            )
            suggestions.extend([
                "Could you provide more details?",
                "Try uploading an image of the item",
                "Specify what you want to know (recycling, upcycling, disposal, etc.)"
            ])
            warnings.append("Request too ambiguous")

        elif error_type == "SERVICE_UNAVAILABLE":
            response_parts.append(
                "Some of our services are temporarily unavailable. "
            )
            suggestions.extend([
                "Please try again in a moment",
                "Try a simpler question",
                "Contact support if the issue persists"
            ])
            warnings.append("Service temporarily unavailable")

        # Add partial results if available
        if partial_results:
            if "vision" in partial_results:
                response_parts.append(f"\n\nPartial analysis: {partial_results['vision']}")
            if "rag" in partial_results:
                response_parts.append(f"\n\nRelated information: {partial_results['rag']}")

        # Add general sustainability tip
        response_parts.append(
            "\n\nGeneral tip: When in doubt, check local recycling guidelines or "
            "contact your local waste management facility for specific guidance."
        )

        return {
            "response": " ".join(response_parts),
            "suggestions": suggestions,
            "warnings": warnings,
            "fallback_used": True,
            "partial_answer": bool(partial_results)
        }


class WorkflowExecutor:
    """
    Advanced workflow executor with intelligent error handling

    CRITICAL: Handles failures gracefully, tracks confidence, enables fallbacks
    """

    def __init__(self):
        self.services = config["services"]
        self.workflows = config["workflows"]
        self.client = httpx.AsyncClient(timeout=60.0)
        self.fallback_strategy = FallbackStrategy()
        self.confidence_calculator = ConfidenceCalculator()
    
    async def execute_workflow(
        self,
        workflow_name: str,
        request: OrchestratorRequest,
        enable_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a workflow with advanced error handling

        Returns: context dict with results and metadata
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        workflow = self.workflows[workflow_name]
        context = {
            "workflow_name": workflow_name,
            "steps_completed": [],
            "steps_failed": [],
            "warnings": [],
            "confidence_scores": {},
            "reasoning_steps": []
        }

        for step in workflow["steps"]:
            service = step["service"]
            action = step["action"]
            required = step["required"]
            step_name = f"{service}_{action}"

            try:
                logger.info(f"Executing step: {step_name}")
                context["reasoning_steps"].append(f"Calling {service} for {action}")

                result = await self._execute_step(service, action, request, context)
                context[step_name] = result
                context["steps_completed"].append(step_name)

                # Extract confidence if available
                if isinstance(result, dict):
                    if "confidence" in result:
                        context["confidence_scores"][step_name] = result["confidence"]
                    if "quality_score" in result:
                        context["confidence_scores"][f"{step_name}_quality"] = result["quality_score"]
                    if "warnings" in result:
                        context["warnings"].extend(result["warnings"])

                logger.info(f"Step completed: {step_name}")

            except Exception as e:
                logger.error(f"Step failed: {step_name} - {e}", exc_info=True)
                context["steps_failed"].append(step_name)
                context["warnings"].append(f"{step_name} failed: {str(e)}")

                if required and not enable_fallback:
                    raise
                elif required and enable_fallback:
                    # Try to continue with partial results
                    logger.warning(f"Required step {step_name} failed, attempting fallback")
                    context["reasoning_steps"].append(f"Step {step_name} failed, using fallback")

        return context
    
    async def _execute_step(
        self,
        service: str,
        action: str,
        request: OrchestratorRequest,
        context: Dict[str, Any]
    ) -> Any:
        """Execute a single workflow step"""
        service_url = self.services[service]["url"]
        
        # Route to appropriate service endpoint
        if service == "vision_service":
            return await self._call_vision_service(service_url, action, request)
        elif service == "llm_service":
            return await self._call_llm_service(service_url, action, request, context)
        elif service == "rag_service":
            return await self._call_rag_service(service_url, action, request)
        elif service == "kg_service":
            return await self._call_kg_service(service_url, action, request, context)
        elif service == "org_search_service":
            return await self._call_org_search_service(service_url, action, request)
        else:
            raise ValueError(f"Unknown service: {service}")
    
    async def _call_vision_service(self, url: str, action: str, request: OrchestratorRequest):
        """Call vision service"""
        endpoint = f"{url}/{action}"
        payload = {
            "image": request.image,
            "image_url": request.image_url
        }
        response = await self.client.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()
    
    async def _call_llm_service(
        self,
        url: str,
        action: str,
        request: OrchestratorRequest,
        context: Dict[str, Any]
    ):
        """Call LLM service"""
        endpoint = f"{url}/{action}"
        payload = {
            "messages": request.messages,
            "context": context
        }
        response = await self.client.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()
    
    async def _call_rag_service(self, url: str, action: str, request: OrchestratorRequest):
        """Call RAG service"""
        endpoint = f"{url}/{action}"
        query = request.messages[-1].get("content", "") if request.messages else ""
        payload = {
            "query": query,
            "location": request.location
        }
        response = await self.client.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()
    
    async def _call_kg_service(
        self,
        url: str,
        action: str,
        request: OrchestratorRequest,
        context: Dict[str, Any]
    ):
        """Call knowledge graph service"""
        endpoint = f"{url}/{action}"
        # Extract relevant info from context
        payload = {"context": context}
        response = await self.client.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()
    
    async def _call_org_search_service(self, url: str, action: str, request: OrchestratorRequest):
        """Call organization search service"""
        endpoint = f"{url}/{action}"
        payload = {
            "query": request.messages[-1].get("content", "") if request.messages else "",
            "location": request.location
        }
        response = await self.client.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()


# Initialize workflow executor
executor = WorkflowExecutor()

# Initialize answer formatter
answer_formatter = AnswerFormatter()


def generate_response_id(request: OrchestratorRequest) -> str:
    """Generate unique response ID for feedback tracking"""
    import hashlib
    from datetime import datetime

    content = f"{datetime.now().isoformat()}_{str(request.messages)}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def format_response_with_rich_content(
    response_text: str,
    task_type: str,
    sources: Optional[List[Dict]] = None,
    confidence_score: float = 0.0,
    result: Optional[Dict] = None
) -> Tuple[str, Dict[str, Any], str, List[Dict]]:
    """
    Format response with rich content based on task type

    Returns: (response_text, formatted_answer_dict, answer_type, citations)
    """
    # Determine answer type from task type
    answer_type_map = {
        "UPCYCLING_IDEA": AnswerType.CREATIVE,
        "ORG_SEARCH": AnswerType.ORG_SEARCH,
        "BIN_DECISION": AnswerType.FACTUAL,
        "THEORY_QA": AnswerType.GENERAL,
        "MATERIAL_INFO": AnswerType.FACTUAL,
        "SAFETY_CHECK": AnswerType.FACTUAL
    }

    answer_type = answer_type_map.get(task_type, AnswerType.GENERAL)

    # Extract type-specific data from result
    kwargs = {}

    if answer_type == AnswerType.CREATIVE and result:
        # Extract creative ideas
        ideas = result.get("ideas", [])
        if ideas:
            kwargs["ideas"] = ideas

    elif answer_type == AnswerType.ORG_SEARCH and result:
        # Extract organizations
        orgs = result.get("organizations", [])
        if orgs:
            kwargs["organizations"] = orgs

    elif answer_type == AnswerType.FACTUAL:
        # Add confidence indicator
        kwargs["confidence"] = confidence_score

    # Format answer
    formatted = answer_formatter.format_answer(
        answer=response_text,
        answer_type=answer_type,
        sources=sources,
        metadata={"confidence": confidence_score},
        **kwargs
    )

    return (
        response_text,
        formatted.to_dict(),
        answer_type.value,
        formatted.citations
    )


@app.post("/orchestrate", response_model=OrchestratorResponse)
async def orchestrate(request: OrchestratorRequest):
    """
    Advanced orchestration endpoint with multi-modal intelligence

    CRITICAL: Handles ANY input with confidence scoring and fallback strategies
    """
    start_time = datetime.now()
    warnings = []
    fallback_used = False
    partial_answer = False

    try:
        # Classify request
        request_type = RequestClassifier.classify_request_type(request)
        task_type, task_confidence = await RequestClassifier.classify_task_type(request)

        # Assess input quality
        text_quality = RequestClassifier.assess_text_quality(request.messages)

        logger.info(
            f"Request classified: type={request_type}, task={task_type} "
            f"(confidence={task_confidence:.2f}), text_quality={text_quality:.2f}"
        )

        # Check if request quality is too low
        if text_quality < 0.2 and not (request.image or request.image_url):
            # Very low quality text-only request
            if request.require_high_confidence:
                raise HTTPException(
                    status_code=400,
                    detail="Request quality too low. Please provide more details or an image."
                )
            warnings.append("Input quality is low - response may be less accurate")

        # Map task type to workflow
        workflow_map = {
            "BIN_DECISION": "bin_decision",
            "UPCYCLING_IDEA": "upcycling_idea",
            "ORG_SEARCH": "org_search",
            "THEORY_QA": "theory_qa",
            "MATERIAL_INFO": "theory_qa",
            "SAFETY_CHECK": "theory_qa"
        }

        workflow_name = workflow_map.get(task_type, "theory_qa")

        # Execute workflow with fallback enabled
        try:
            result = await executor.execute_workflow(
                workflow_name,
                request,
                enable_fallback=request.enable_fallback
            )

            # Check if any required steps failed
            if result.get("steps_failed"):
                fallback_used = True
                partial_answer = True
                warnings.extend(result.get("warnings", []))

        except Exception as workflow_error:
            logger.error(f"Workflow execution failed: {workflow_error}", exc_info=True)

            if not request.enable_fallback:
                raise

            # Generate fallback response
            fallback_data = executor.fallback_strategy.generate_fallback_response(
                error_type="SERVICE_UNAVAILABLE",
                request=request,
                partial_results=None
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return OrchestratorResponse(
                response=fallback_data["response"],
                confidence_score=0.3,
                confidence_level=ConfidenceLevel.LOW,
                sources=None,
                suggestions=fallback_data["suggestions"],
                warnings=fallback_data["warnings"],
                fallback_used=True,
                partial_answer=False,
                metadata={
                    "request_type": request_type,
                    "task_type": task_type,
                    "workflow": workflow_name,
                    "error": str(workflow_error)
                },
                processing_time_ms=processing_time,
                image_quality_score=None,
                text_quality_score=text_quality,
                reasoning_steps=["Workflow failed", "Generated fallback response"]
            )

        # Extract final response
        response_text = (
            result.get("llm_service_synthesize_decision", {}).get("response") or
            result.get("llm_service_generate_ideas", {}).get("response") or
            result.get("llm_service_answer_question", {}).get("response") or
            result.get("llm_service_rank_and_explain", {}).get("response", "")
        )

        # If no response generated, use fallback
        if not response_text:
            logger.warning("No response generated from workflow, using fallback")
            fallback_data = executor.fallback_strategy.generate_fallback_response(
                error_type="AMBIGUOUS_REQUEST",
                request=request,
                partial_results=result
            )
            response_text = fallback_data["response"]
            fallback_used = True
            warnings.extend(fallback_data["warnings"])

        # Calculate overall confidence
        vision_conf = result.get("confidence_scores", {}).get("vision_service_classify")
        llm_conf = result.get("confidence_scores", {}).get("llm_service_answer_question")
        rag_conf = result.get("confidence_scores", {}).get("rag_service_retrieve_knowledge")
        image_quality = result.get("confidence_scores", {}).get("vision_service_classify_quality")

        overall_confidence, confidence_level = executor.confidence_calculator.calculate_overall_confidence(
            vision_confidence=vision_conf,
            llm_confidence=llm_conf,
            rag_confidence=rag_conf,
            image_quality=image_quality,
            text_quality=text_quality
        )

        # Adjust confidence if fallback was used
        if fallback_used:
            overall_confidence *= 0.7
            confidence_level = ConfidenceLevel.LOW if overall_confidence < 0.5 else ConfidenceLevel.MEDIUM

        # Check if confidence meets requirements
        if request.require_high_confidence and overall_confidence < 0.8:
            raise HTTPException(
                status_code=422,
                detail=f"Confidence too low ({overall_confidence:.2f}). Unable to provide high-confidence answer."
            )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Extract suggestions
        suggestions = result.get("suggestions", [])
        if not suggestions and confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]:
            suggestions = [
                "Try providing a clearer image",
                "Add more details about the item",
                "Specify what you want to know"
            ]

        # Generate response ID for feedback tracking
        response_id = generate_response_id(request)

        # Format response with rich content
        _, formatted_answer, answer_type, citations = format_response_with_rich_content(
            response_text=response_text,
            task_type=task_type,
            sources=result.get("sources"),
            confidence_score=overall_confidence,
            result=result
        )

        return OrchestratorResponse(
            response=response_text,
            confidence_score=overall_confidence,
            confidence_level=confidence_level,
            sources=result.get("sources"),
            suggestions=suggestions,
            warnings=warnings + result.get("warnings", []),
            fallback_used=fallback_used,
            partial_answer=partial_answer,
            metadata={
                "request_type": request_type,
                "task_type": task_type,
                "task_confidence": task_confidence,
                "workflow": workflow_name,
                "steps_completed": result.get("steps_completed", []),
                "steps_failed": result.get("steps_failed", [])
            },
            processing_time_ms=processing_time,
            image_quality_score=image_quality,
            text_quality_score=text_quality,
            reasoning_steps=result.get("reasoning_steps", []),
            # Rich formatting
            formatted_answer=formatted_answer,
            answer_type=answer_type,
            citations=citations,
            response_id=response_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)

        # Final fallback
        if request.enable_fallback:
            fallback_data = executor.fallback_strategy.generate_fallback_response(
                error_type="SERVICE_UNAVAILABLE",
                request=request,
                partial_results=None
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return OrchestratorResponse(
                response=fallback_data["response"],
                confidence_score=0.2,
                confidence_level=ConfidenceLevel.VERY_LOW,
                sources=None,
                suggestions=fallback_data["suggestions"],
                warnings=fallback_data["warnings"] + [f"Critical error: {str(e)}"],
                fallback_used=True,
                partial_answer=False,
                metadata={"error": str(e)},
                processing_time_ms=processing_time,
                image_quality_score=None,
                text_quality_score=0.0,
                reasoning_steps=["Critical error occurred", "Generated emergency fallback"]
            )
        else:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "orchestrator"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

