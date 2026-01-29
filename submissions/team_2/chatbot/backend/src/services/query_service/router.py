"""
MetaKGP Query Service Router
FastAPI routes for semantic search
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from src.services.query_service.service import QueryService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/query",
    tags=["query"],
    responses={404: {"description": "Not found"}},
)

# Global service instance (will be set by main app)
_query_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    """Dependency to get the query service instance"""
    if _query_service is None:
        raise HTTPException(
            status_code=503,
            detail="Query service not initialized"
        )
    return _query_service


def set_query_service(service: QueryService):
    """Set the global query service instance"""
    global _query_service
    _query_service = service


# Pydantic models
class SearchFilters(BaseModel):
    """Optional filters for search"""
    source_page: Optional[str] = Field(None, description="Filter by specific page name")
    category: Optional[str] = Field(None, description="Filter by category")
    min_entity_count: Optional[int] = Field(None, description="Minimum number of entities")


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    filters: Optional[SearchFilters] = Field(None, description="Optional metadata filters")


class SearchResultMetadata(BaseModel):
    """Metadata for a search result"""
    source_page: str
    title: str
    section: str = ""
    chunk_index: int
    total_chunks: int = 0  # Default to 0 if not present
    categories: List[str] = []
    entities: List[str] = []
    entity_count: int = 0
    relationship_count: int = 0  # Default to 0 if not present


class SearchResult(BaseModel):
    """Single search result"""
    chunk_id: str
    text: str
    score: float = Field(..., ge=0.0, description="Relevance score (similarity)")
    distance: Optional[float] = Field(None, description="Vector distance")
    rank: int = Field(..., description="Result rank (1-indexed)")
    metadata: SearchResultMetadata


class SearchResponse(BaseModel):
    """Search response with results and metadata"""
    results: List[SearchResult]
    query_time_ms: float
    total_results: int
    search_mode: str = Field(default="semantic", description="Search mode: semantic")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    num_documents: int
    embedding_service: str


# API Endpoints

@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    service: QueryService = Depends(get_query_service)
):
    """
    Semantic search over wiki chunks using vector similarity
    
    Uses 768-dimensional embeddings from all-mpnet-base-v2 model
    to find semantically similar content.
    
    Example request:
    ```json
    {
        "query": "Who is the vice president of TSG?",
        "top_k": 5
    }
    ```
    
    With filters:
    ```json
    {
        "query": "What are the selection criteria?",
        "top_k": 5,
        "filters": {
            "source_page": "180 Degrees Consulting"
        }
    }
    ```
    """
    try:
        # Prepare filters
        filters_dict = None
        category_filter = None
        if request.filters:
            filters_dict = request.filters.model_dump(exclude_none=True)
            # Extract category for post-processing (ChromaDB doesn't support substring matching)
            category_filter = filters_dict.pop("category", None)
        
        # Perform search
        result = service.search(
            query=request.query,
            top_k=request.top_k,
            filters=filters_dict,
            category_filter=category_filter
        )
        
        # Convert dict results to Pydantic models
        search_results = [
            SearchResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                distance=r.get("distance"),
                rank=r["rank"],
                metadata=SearchResultMetadata(**r["metadata"])
            )
            for r in result["results"]
        ]
        
        return SearchResponse(
            results=search_results,
            query_time_ms=result["query_time_ms"],
            total_results=result["total_results"],
            search_mode=result.get("search_mode", "semantic")
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health(service: QueryService = Depends(get_query_service)):
    """
    Health check endpoint
    
    Returns service status and document count
    """
    try:
        num_docs = service.get_document_count()
        
        return HealthResponse(
            status="ok",
            num_documents=num_docs,
            embedding_service="modal"
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )
