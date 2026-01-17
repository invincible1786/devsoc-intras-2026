"""
FastAPI Router for Graph of Thoughts + MoE System
"""
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services.chat_agent.got_moe_engine import GoTMoEEngine
from src.utils.chroma_client import MetaKGPChromaClient
from src.utils.embedding_client import ModalEmbeddingClient
from src.utils.groq_client import GroqClient

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/got",
    tags=["Graph of Thoughts"]
)

_engine: Optional[GoTMoEEngine] = None


class QueryRequest(BaseModel):
    query: str
    use_cache: Optional[bool] = True
    max_depth: Optional[int] = 3
    max_branches: Optional[int] = 3


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    sources: list[str] = []
    reasoning_path: list[Dict[str, Any]] = []
    graph_stats: Optional[Dict[str, Any]] = None
    visualization_path: Optional[str] = None
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    engine_initialized: bool
    cache_enabled: bool


class StatsResponse(BaseModel):
    total_thoughts: int
    verified_thoughts: int
    cached_graphs: int
    cache_size_mb: float


def get_engine() -> GoTMoEEngine:
    global _engine
    if _engine is None:
        logger.info("Initializing GoTMoEEngine...")
        try:
            chroma_client = MetaKGPChromaClient()
            embedding_client = ModalEmbeddingClient()
            groq_client = GroqClient()
            _engine = GoTMoEEngine(
                chroma_client=chroma_client,
                embedding_client=embedding_client,
                groq_client=groq_client
            )
            logger.info("GoTMoEEngine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GoTMoEEngine: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize reasoning engine: {str(e)}")
    return _engine


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> QueryResponse:
    try:
        engine = get_engine()
        logger.info(f"Processing query: {request.query}")
        result = await engine.process_query(query=request.query)
        response = QueryResponse(
            query=request.query,
            answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.0),
            sources=result.get("sources", []),
            reasoning_path=result.get("reasoning_path", []),
            graph_stats=result.get("graph_stats"),
            visualization_path=None,
            cached=False
        )
        logger.info(f"Query processed: {request.query[:50]}...")
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    try:
        engine = get_engine()
        return HealthResponse(status="healthy", engine_initialized=True, cache_enabled=True)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="unhealthy", engine_initialized=False, cache_enabled=False)


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    try:
        engine = get_engine()
        graph_stats = engine.get_graph_stats()
        return StatsResponse(
            total_thoughts=graph_stats["total_nodes"],
            verified_thoughts=graph_stats["total_nodes"],  # All stored nodes are verified
            cached_graphs=0,
            cache_size_mb=0.0
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")
