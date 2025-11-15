"""
Orchestrator Service - Request routing and workflow coordination
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal
import httpx
import asyncio
import logging
from datetime import datetime
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ReleAF AI Orchestrator",
    description="Request routing and workflow coordination service",
    version="0.1.0"
)

# Load configuration
with open("configs/orchestrator.yaml", "r") as f:
    config = yaml.safe_load(f)


class OrchestratorRequest(BaseModel):
    """Orchestrator request"""
    messages: List[Dict[str, Any]]
    image: Optional[str] = None
    image_url: Optional[str] = None
    location: Optional[Dict[str, float]] = None
    context: Optional[Dict[str, Any]] = None


class OrchestratorResponse(BaseModel):
    """Orchestrator response"""
    response: str
    sources: Optional[List[Dict[str, str]]] = None
    suggestions: Optional[List[str]] = None
    metadata: Dict[str, Any]
    processing_time_ms: float


class RequestClassifier:
    """Classify incoming requests"""
    
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
    async def classify_task_type(request: OrchestratorRequest) -> str:
        """Determine task type from user intent"""
        # Simple keyword-based classification (can be replaced with LLM)
        if not request.messages:
            return "BIN_DECISION"
        
        last_message = request.messages[-1].get("content", "").lower()
        
        # Keyword matching
        if any(word in last_message for word in ["bin", "recycle", "dispose", "throw"]):
            return "BIN_DECISION"
        elif any(word in last_message for word in ["upcycle", "reuse", "make", "create", "diy"]):
            return "UPCYCLING_IDEA"
        elif any(word in last_message for word in ["where", "find", "location", "near", "facility"]):
            return "ORG_SEARCH"
        elif any(word in last_message for word in ["safe", "danger", "toxic", "hazard"]):
            return "SAFETY_CHECK"
        elif any(word in last_message for word in ["material", "property", "chemistry", "composition"]):
            return "MATERIAL_INFO"
        else:
            return "THEORY_QA"


class WorkflowExecutor:
    """Execute predefined workflows"""
    
    def __init__(self):
        self.services = config["services"]
        self.workflows = config["workflows"]
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def execute_workflow(
        self,
        workflow_name: str,
        request: OrchestratorRequest
    ) -> Dict[str, Any]:
        """Execute a workflow"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        workflow = self.workflows[workflow_name]
        context = {}
        
        for step in workflow["steps"]:
            service = step["service"]
            action = step["action"]
            required = step["required"]
            
            try:
                result = await self._execute_step(service, action, request, context)
                context[f"{service}_{action}"] = result
            except Exception as e:
                logger.error(f"Step failed: {service}.{action} - {e}")
                if required:
                    raise
        
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


@app.post("/orchestrate", response_model=OrchestratorResponse)
async def orchestrate(request: OrchestratorRequest):
    """Main orchestration endpoint"""
    start_time = datetime.now()
    
    try:
        # Classify request
        request_type = RequestClassifier.classify_request_type(request)
        task_type = await RequestClassifier.classify_task_type(request)
        
        logger.info(f"Request type: {request_type}, Task type: {task_type}")
        
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
        
        # Execute workflow
        result = await executor.execute_workflow(workflow_name, request)
        
        # Extract final response
        response_text = result.get("llm_service_synthesize_decision") or \
                       result.get("llm_service_generate_ideas") or \
                       result.get("llm_service_answer_question") or \
                       result.get("llm_service_rank_and_explain", {}).get("response", "")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return OrchestratorResponse(
            response=response_text,
            sources=result.get("sources"),
            suggestions=result.get("suggestions"),
            metadata={
                "request_type": request_type,
                "task_type": task_type,
                "workflow": workflow_name
            },
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "orchestrator"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

