"""
MetaKGP Query Service - Business Logic
Handles semantic search operations over wiki chunks
"""

import logging
import time
from typing import List, Optional, Dict

from src.utils.embedding_client import ModalEmbeddingClient
from src.utils.chroma_client import MetaKGPChromaClient

logger = logging.getLogger(__name__)


class QueryService:
    """
    Service class for semantic search operations
    
    Features:
    - Semantic search via 768-dim embeddings (all-mpnet-base-v2)
    - Vector similarity using ChromaDB
    - Optional metadata filtering
    """
    
    def __init__(
        self,
        modal_url: str,
        chroma_dir: str = "./chroma_data",
        collection_name: str = "metakgp_wiki"
    ):
        """
        Initialize the query service
        
        Args:
            modal_url: URL of the Modal embedding service
            chroma_dir: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
        """
        # Initialize embedding client (Modal API)
        self.embedding_client = ModalEmbeddingClient(modal_url)
        
        # Initialize ChromaDB client
        self.chroma_client = MetaKGPChromaClient(
            persist_dir=chroma_dir,
            collection_name=collection_name
        )
        
        logger.info("✓ QueryService initialized with SEMANTIC search")
        
        doc_count = self.chroma_client.get_count()
        logger.info(f"📚 Loaded {doc_count} documents from ChromaDB")
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the collection"""
        return self.chroma_client.get_count()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        category_filter: Optional[str] = None
    ) -> Dict:
        """
        Perform semantic search over wiki chunks
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional metadata filters for ChromaDB
            category_filter: Optional category filter (applied post-search)
        
        Returns:
            Dict with 'results', 'query_time_ms', 'total_results', and 'search_mode'
        """
        start_time = time.time()
        
        logger.info(f"🔍 Query: {query[:100]}...")
        
        # Perform semantic search
        logger.info("🧠 Using SEMANTIC search (embeddings)")
        search_results = self._semantic_search(
            query=query,
            top_k=top_k * 2 if category_filter else top_k,
            filters=filters
        )
        
        # Post-process: filter by category if requested
        if category_filter:
            search_results = [
                result for result in search_results
                if category_filter.lower() in [cat.lower() for cat in result["metadata"]["categories"]]
            ]
            # Trim to requested top_k
            search_results = search_results[:top_k]
        
        # Calculate query time
        query_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"✓ Found {len(search_results)} results in {query_time_ms:.1f}ms"
        )
        
        return {
            "results": search_results,
            "query_time_ms": query_time_ms,
            "total_results": len(search_results),
            "search_mode": "semantic"
        }
    
    def _semantic_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict]
    ) -> List[Dict]:
        """Perform semantic search using embeddings and ChromaDB"""
        # Generate query embedding
        query_embedding = self.embedding_client.encode(query)
        
        if query_embedding is None:
            raise ValueError("Failed to generate query embedding")
        
        # Convert to list if numpy array
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        
        logger.info(f"📊 Generated embedding (dimension: {len(query_embedding)})")
        
        # Search ChromaDB
        results = self.chroma_client.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        logger.info(f"📦 ChromaDB returned {len(results['ids'])} results")
        
        # Format results
        search_results = []
        
        for i, chunk_id in enumerate(results["ids"]):
            # Get distance and convert to similarity score
            # ChromaDB returns cosine distance in [0, 2] range
            # Convert to similarity score in [0, 1] where 1 is most similar
            distance = results["distances"][i]
            score = 1.0 - (distance / 2.0)  # Normalize to [0, 1] range
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
            # Parse metadata
            raw_metadata = results["metadatas"][i]
            
            # Deserialize comma-separated fields
            categories = raw_metadata.get("categories", "").split(",") if raw_metadata.get("categories") else []
            categories = [c.strip() for c in categories if c.strip()]
            
            entities = raw_metadata.get("entities", "").split(",") if raw_metadata.get("entities") else []
            entities = [e.strip() for e in entities if e.strip()]
            
            # Build result dictionary
            result = {
                "chunk_id": chunk_id,
                "text": results["documents"][i],
                "score": score,
                "distance": distance,
                "rank": i + 1,
                "metadata": {
                    "source_page": raw_metadata.get("source_page", ""),
                    "title": raw_metadata.get("title", ""),
                    "section": raw_metadata.get("section", ""),
                    "parent_section": raw_metadata.get("parent_section", ""),
                    "chunk_index": raw_metadata.get("chunk_index", 0),
                    "total_chunks": raw_metadata.get("total_chunks", 0),
                    "categories": categories,
                    "entities": entities,
                    "entity_count": raw_metadata.get("entity_count", 0),
                    "relationship_count": raw_metadata.get("relationship_count", 0)
                }
            }
            
            search_results.append(result)
        
        return search_results
