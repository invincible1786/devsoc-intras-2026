"""
ChromaDB Client for MetaKGP Wiki
Vector storage with rich metadata and graph relationships
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging
import json

logger = logging.getLogger(__name__)


class MetaKGPChromaClient:
    """ChromaDB client for MetaKGP wiki chunks with entity/relationship support"""
    
    def __init__(
        self,
        persist_dir: str = "./chroma_data",
        collection_name: str = "metakgp_wiki"
    ):
        """
        Initialize ChromaDB client
        
        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name of the collection
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "MetaKGP wiki chunks with entities and relationships",
                "hnsw:space": "cosine"
            }
        )
        
        logger.info(f" ChromaDB collection '{collection_name}' initialized at {persist_dir}")
        logger.info(f" Current document count: {self.collection.count()}")
    
    def _flatten_metadata(self, chunk_obj: Dict) -> Dict:
        """
        Flatten nested chunk object for ChromaDB storage
        
        ChromaDB doesn't support nested dictionaries or arrays in metadata,
        so we serialize complex types to strings.
        
        Args:
            chunk_obj: Chunk object with nested metadata
            
        Returns:
            Flattened metadata dictionary
        """
        meta = chunk_obj["metadata"]
        
        return {
            # Basic metadata
            "source_page": meta["source_page"],
            "title": meta["title"],
            "chunk_index": meta["chunk_index"],
            "total_chunks": meta.get("total_chunks", 0),  # Default to 0 if not present
            
            # Serialize arrays as comma-separated strings
            "categories": ",".join(meta["categories"]) if meta["categories"] else "",
            "entities": ",".join(chunk_obj.get("entities", [])),
            
            # Store counts for filtering
            "entity_count": len(chunk_obj.get("entities", [])),
            "relationship_count": len(chunk_obj.get("relationships", [])),
            
            # Serialize relationships as JSON string
            "relationships_json": json.dumps(chunk_obj.get("relationships", []))
        }
    
    def add_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        chunk_objects: List[Dict]
    ):
        """
        Add chunks to ChromaDB with embeddings and metadata
        
        Args:
            chunk_ids: List of unique chunk IDs
            texts: List of chunk texts
            embeddings: List of embedding vectors
            chunk_objects: List of full chunk objects (with metadata, entities, relationships)
        """
        if not chunk_ids:
            logger.warning("️ No chunks to add")
            return
        
        try:
            # Flatten all metadata
            metadatas = [self._flatten_metadata(obj) for obj in chunk_objects]
            
            # Add to ChromaDB
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f" Added {len(chunk_ids)} chunks to ChromaDB")
        
        except Exception as e:
            logger.error(f" Failed to add chunks to ChromaDB: {e}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Search ChromaDB with optional metadata filters
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional filters like:
                {
                    "source_page": "IIT_Kharagpur",
                    "category": "Academics",
                    "min_entity_count": 3
                }
        
        Returns:
            Search results with ids, documents, metadatas, distances
        """
        try:
            # Build ChromaDB where clause
            where = None
            if filters:
                where = {}
                
                # Exact match filters
                if "source_page" in filters:
                    where["source_page"] = filters["source_page"]
                
                # Note: Category filtering is disabled because ChromaDB doesn't support
                # substring/contains operators on comma-separated strings.
                # To filter by category, you would need to do post-processing on results.
                
                # Numeric filters
                if "min_entity_count" in filters:
                    where["entity_count"] = {"$gte": filters["min_entity_count"]}
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where if where else None
            )
            
            # Return flattened results
            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else []
            }
        
        except Exception as e:
            logger.error(f" ChromaDB search failed: {e}")
            raise
    
    def get_count(self) -> int:
        """Get total number of chunks in collection"""
        return self.collection.count()
    
    def get_by_ids(self, chunk_ids: List[str]) -> Dict:
        """
        Retrieve chunks by IDs
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            Dictionary with ids, documents, metadatas
        """
        try:
            results = self.collection.get(ids=chunk_ids)
            return results
        except Exception as e:
            logger.error(f" Failed to retrieve chunks: {e}")
            return {"ids": [], "documents": [], "metadatas": []}
    
    def reset(self):
        """Delete all data (for testing/reindexing)"""
        logger.warning("️ Resetting ChromaDB collection")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
