"""
Chat Router - Handles chat/conversation endpoints

Routes requests through orchestrator for intelligent workflow execution
"""

import logging
from typing import List, Dict, Any, Optional
import httpx
import os

from fastapi import APIRouter, HTTPException, Request

# CRITICAL: Import schemas from central location - eliminates duplication
from services.api_gateway.schemas import ChatRequest, ChatResponse, ChatMessage

logger = logging.getLogger(__name__)

router = APIRouter()

# Service URLs
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")

# REMOVED: Duplicate schema definitions (Message, ChatRequest, ChatResponse)
# Now using centralized schemas from services/api_gateway/schemas.py


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """
    Main chat endpoint
    
    Routes through orchestrator for intelligent workflow execution:
    - Determines if vision analysis is needed
    - Retrieves relevant knowledge from RAG
    - Queries knowledge graph for relationships
    - Searches for organizations if needed
    - Generates final response with LLM
    """
    try:
        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Build orchestrator request
        orchestrator_request = {
            "messages": messages,
            "location": request.location,
            "image": request.image,
            "image_url": request.image_url,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        # Call orchestrator
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ORCHESTRATOR_URL}/orchestrate",
                json=orchestrator_request
            )
            response.raise_for_status()
            result = response.json()
        
        return ChatResponse(
            response=result.get("response", ""),
            context=result.get("context"),
            metadata=result.get("metadata")
        )
        
    except httpx.TimeoutException:
        logger.error("Orchestrator timeout")
        raise HTTPException(status_code=504, detail="Request timeout")
    except httpx.HTTPStatusError as e:
        logger.error(f"Orchestrator error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simple", response_model=ChatResponse)
async def simple_chat(request: ChatRequest):
    """
    Simple chat endpoint (direct to LLM, no orchestration)
    
    For basic questions that don't need vision/RAG/KG
    """
    try:
        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Build LLM request
        llm_request = {
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": 0.9
        }
        
        # Call LLM service directly
        llm_url = os.getenv("LLM_SERVICE_URL", "http://localhost:8002")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{llm_url}/generate",
                json=llm_request
            )
            response.raise_for_status()
            result = response.json()
        
        return ChatResponse(
            response=result.get("response", ""),
            metadata={"usage": result.get("usage"), "model": result.get("model")}
        )
        
    except httpx.TimeoutException:
        logger.error("LLM timeout")
        raise HTTPException(status_code=504, detail="Request timeout")
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Simple chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "router": "chat"}

