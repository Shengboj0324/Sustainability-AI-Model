"""
Organizations Router - Handles organization search endpoints

Routes requests to org search service for finding charities, recycling centers, etc.
"""

import logging
from typing import List, Dict, Any, Optional
import httpx
import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Service URLs
ORG_SEARCH_SERVICE_URL = os.getenv("ORG_SEARCH_SERVICE_URL", "http://localhost:8005")


class Location(BaseModel):
    """Geographic location"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")


class OrganizationSearchRequest(BaseModel):
    """Organization search request"""
    location: Location = Field(..., description="Search center location")
    radius_km: float = Field(10.0, ge=0.1, le=100, description="Search radius in kilometers")
    org_type: Optional[str] = Field(None, description="Organization type filter")
    accepted_materials: Optional[List[str]] = Field(None, description="Filter by accepted materials")
    limit: int = Field(20, ge=1, le=100, description="Max results")


class Organization(BaseModel):
    """Organization result"""
    id: int
    name: str
    org_type: str
    address: str
    city: str
    state: str
    zip_code: str
    latitude: float
    longitude: float
    distance_km: float
    phone: Optional[str]
    website: Optional[str]
    email: Optional[str]
    accepted_materials: List[str]
    operating_hours: Optional[Dict[str, str]]
    description: Optional[str]


class OrganizationSearchResponse(BaseModel):
    """Organization search response"""
    organizations: List[Organization]
    num_results: int
    search_location: Location
    search_radius_km: float
    query_time_ms: float


@router.post("/search", response_model=OrganizationSearchResponse)
async def search_organizations(request: OrganizationSearchRequest, http_request: Request):
    """
    Search for organizations near a location
    
    Finds:
    - Charities accepting donations
    - Recycling centers
    - Sustainability organizations
    - Waste management facilities
    """
    try:
        # Build search request
        search_request = {
            "latitude": request.location.latitude,
            "longitude": request.location.longitude,
            "radius_km": request.radius_km,
            "org_type": request.org_type,
            "accepted_materials": request.accepted_materials,
            "limit": request.limit
        }
        
        # Call org search service
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ORG_SEARCH_SERVICE_URL}/search",
                json=search_request
            )
            response.raise_for_status()
            result = response.json()
        
        return OrganizationSearchResponse(**result)
        
    except httpx.TimeoutException:
        logger.error("Org search service timeout")
        raise HTTPException(status_code=504, detail="Organization search timeout")
    except httpx.HTTPStatusError as e:
        logger.error(f"Org search service error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Organization search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def get_organization_types():
    """Get available organization types"""
    return {
        "types": [
            "charity",
            "recycling_center",
            "donation_center",
            "sustainability_org",
            "waste_management",
            "repair_cafe",
            "upcycling_workshop",
            "community_garden"
        ]
    }


@router.get("/materials")
async def get_accepted_materials():
    """Get list of materials that organizations might accept"""
    return {
        "materials": [
            "plastic",
            "glass",
            "metal",
            "paper",
            "cardboard",
            "electronics",
            "batteries",
            "textiles",
            "furniture",
            "appliances",
            "hazardous_waste",
            "organic_waste",
            "construction_debris"
        ]
    }


@router.get("/health")
async def health():
    """Health check"""
    # Check org search service health
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ORG_SEARCH_SERVICE_URL}/health")
            org_search_health = response.json()
        
        return {
            "status": "healthy" if org_search_health.get("status") == "healthy" else "unhealthy",
            "router": "organizations",
            "org_search_service": org_search_health
        }
    except Exception as e:
        logger.error(f"Org search service health check failed: {e}")
        return {"status": "unhealthy", "router": "organizations", "error": str(e)}

