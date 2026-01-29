"""
Modal Embedding Service for MetaKGP RAG System
Deploys sentence-transformers/all-mpnet-base-v2 (768 dimensions) on Modal
"""

import modal
from typing import List, Dict

app = modal.App("metakgp-embeddings")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "sentence-transformers==3.0.1",
    "torch==2.4.1",
    "transformers==4.45.0",
    "fastapi==0.109.0",
    "pydantic==2.5.3",
    "huggingface-hub>=0.20.0",
    "numpy>=1.24.0"
)

@app.function(
    image=image,
    gpu="A100",
    min_containers=1,
    timeout=300,
    scaledown_window=120
)
def embed_batch(texts: List[str]) -> List[List[float]]:
    """Batch embed texts using sentence-transformers"""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(texts, batch_size=100, show_progress_bar=False)
    return embeddings.tolist()

@app.function(image=image)
def get_embedding_dimension() -> int:
    """Return embedding dimension"""
    return 768

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    web_app = FastAPI()
    
    class EmbedRequest(BaseModel):
        doc_id: str
        content: str
        metadata: Dict = {}
    
    class HealthResponse(BaseModel):
        status: str
        embedding_dimension: int
    
    @web_app.post("/embedding/embed")
    async def embed(request: EmbedRequest):
        """
        Expected request format:
        {
            "doc_id": "unique_id",
            "content": "text to embed",
            "metadata": {"source": "..."}
        }
        
        Returns:
        {
            "embeddings": [[float, float, ...]]
        }
        """
        # Call the embed_batch function
        embeddings = embed_batch.remote([request.content])
        return {"embeddings": embeddings}
    
    @web_app.get("/embedding/health")
    async def health():
        """
        Returns:
        {
            "status": "ok",
            "embedding_dimension": 768
        }
        """
        return HealthResponse(
            status="ok",
            embedding_dimension=768
        )
    
    return web_app
