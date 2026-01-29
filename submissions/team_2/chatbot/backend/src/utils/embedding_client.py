"""
Embedding Client for Modal Embedding Service
Features:
- Connection pooling with requests.Session
- Retry logic (1 retry, 0.2s backoff)
- Timeouts (5s connect, 30s read)
- Keep-alive headers
- Async support with httpx
- Batch processing support
"""

import os
import logging
from typing import List, Optional, Dict, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import numpy as np

logger = logging.getLogger(__name__)


class ModalEmbeddingClient:
    """
    Embedding client for Modal API service
    
    Supports the Modal embedding service with connection pooling and retry logic.
    """
    
    def __init__(self, modal_url: Optional[str] = None):
        """
        Initialize embedding client
        
        Args:
            modal_url: Modal embedding service URL (e.g., https://...modal.run)
                      If not provided, reads from MODAL_URL env variable
        """
        self.modal_url = modal_url or os.getenv("MODAL_URL")
        
        if not self.modal_url:
            raise ValueError("Modal URL not provided and MODAL_URL env variable not set")
        
        # Ensure URL ends without trailing slash
        self.modal_url = self.modal_url.rstrip('/')
        
        # Create session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=1,  # 1 retry
            backoff_factor=0.2,  # 0.2s backoff
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set keep-alive headers
        self.session.headers.update({
            "Connection": "keep-alive",
            "Accept": "application/json",
            "Content-Type": "application/json"
        })
        
        # Timeouts - increased for batch processing
        self.connect_timeout = 10
        self.read_timeout = 120  # 2 minutes for large batches
        
        logger.info(f"✓ ModalEmbeddingClient initialized with URL: {self.modal_url}")
    
    def encode(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[float], None]:
        """
        Generate embedding(s) for text(s)
        
        Args:
            text: Single text string or list of texts
        
        Returns:
            - Single text: list of floats
            - Multiple texts: list of lists
            - None if failed
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        results = []
        for t in texts:
            emb = self._embed_single(t)
            if emb is None:
                return None
            results.append(emb)
        
        return results[0] if is_single else results
    
    def _embed_single(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Modal API"""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            payload = {
                "doc_id": f"doc_{hash(text)}",
                "content": text,
                "metadata": {}
            }
            
            response = self.session.post(
                f"{self.modal_url}/embedding/embed",
                json=payload,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            result = response.json()
            embeddings = result.get("embeddings", [])
            
            return embeddings[0] if embeddings else None
        
        except Exception as e:
            logger.error(f"❌ Modal API error: {e}")
            return None
    
    def __call__(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text (backward compatibility)
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector or None
        """
        return self.encode(text)
    
    def get_dimension(self) -> int:
        """Get embedding dimension (default for Modal service)"""
        return 768  # Modal uses all-mpnet-base-v2
    
    def health_check(self) -> bool:
        """
        Check if embedding service is healthy
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.modal_url}/embedding/health",
                timeout=(self.connect_timeout, 10)
            )
            
            response.raise_for_status()
            result = response.json()
            
            is_healthy = result.get("status") == "ok"
            
            if is_healthy:
                logger.info(f"✓ Embedding service healthy (dimension: {result.get('embedding_dimension')})")
            else:
                logger.warning("⚠️ Embedding service returned unhealthy status")
            
            return is_healthy
        
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
    
    def close(self):
        """Close session and cleanup"""
        self.session.close()
        logger.info("✓ Embedding client session closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

