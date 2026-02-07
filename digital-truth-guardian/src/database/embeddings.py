"""
Embedding service for generating dense and sparse vectors.

Uses Google gemini-embedding-001 for dense vectors and
FastEmbed SPLADE for sparse vectors.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

from google import genai
from google.genai import types
from fastembed import SparseTextEmbedding

from ..core.config import settings
from ..utils.logger import get_logger
from ..utils.helpers import async_retry, chunk_text


logger = get_logger()


class EmbeddingService:
    """
    Service for generating both dense and sparse embeddings.
    
    Implements a hybrid embedding strategy:
    - Dense: Google gemini-embedding-001 (768d) for semantic similarity
    - Sparse: FastEmbed SPLADE for keyword matching
    """
    
    def __init__(
        self,
        dense_model: str = None,
        sparse_model: str = None,
        api_key: str = None
    ):
        """
        Initialize the embedding service.
        
        Args:
            dense_model: Google embedding model name
            sparse_model: FastEmbed sparse model name
            api_key: Google API key
        """
        self.dense_model = dense_model or settings.dense_embedding_model
        self.sparse_model = sparse_model or settings.sparse_embedding_model
        self.api_key = api_key or settings.gemini_api_key
        
        # Initialize Google AI client
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
        
        # Lazy-load sparse model
        self._sparse_encoder: Optional[SparseTextEmbedding] = None
        
        logger.with_agent("Embedding").info(
            f"Initialized embedding service: dense={self.dense_model}, sparse={self.sparse_model}"
        )
    
    @property
    def sparse_encoder(self) -> SparseTextEmbedding:
        """Lazy-load the sparse embedding model."""
        if self._sparse_encoder is None:
            self._sparse_encoder = SparseTextEmbedding(
                model_name=self.sparse_model
            )
        return self._sparse_encoder
    
    @async_retry(max_retries=3, delay=1.0)
    async def embed_dense(self, text: str) -> List[float]:
        """
        Generate dense embedding using Google gemini-embedding-001.
        
        Args:
            text: Text to embed
            
        Returns:
            3072-dimensional dense vector
        """
        if not self.client:
            raise ValueError("Google API client not initialized")
        
        response = await self.client.aio.models.embed_content(
            model=self.dense_model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        return list(response.embeddings[0].values)
    
    @async_retry(max_retries=3, delay=1.0)
    async def embed_dense_query(self, text: str) -> List[float]:
        """
        Generate dense embedding for a query (retrieval_query task).
        
        Args:
            text: Query text to embed
            
        Returns:
            768-dimensional dense vector
        """
        if not self.client:
            raise ValueError("Google API client not initialized")
        
        response = await self.client.aio.models.embed_content(
            model=self.dense_model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return list(response.embeddings[0].values)
    
    async def embed_sparse(self, text: str) -> Dict[str, Any]:
        """
        Generate sparse embedding using FastEmbed BM25.
        
        Args:
            text: Text to embed
            
        Returns:
            Sparse vector dict with indices and values
        """
        loop = asyncio.get_event_loop()
        
        def _encode():
            embeddings = list(self.sparse_encoder.embed([text]))
            if embeddings:
                sparse = embeddings[0]
                return {
                    "indices": sparse.indices.tolist(),
                    "values": sparse.values.tolist()
                }
            return {"indices": [], "values": []}
        
        return await loop.run_in_executor(None, _encode)
    
    async def embed_hybrid(self, text: str) -> Tuple[List[float], Dict[str, Any]]:
        """
        Generate both dense and sparse embeddings.
        
        Args:
            text: Text to embed
            
        Returns:
            Tuple of (dense_vector, sparse_vector)
        """
        # Run both in parallel
        dense_task = self.embed_dense(text)
        sparse_task = self.embed_sparse(text)
        
        dense_vector, sparse_vector = await asyncio.gather(
            dense_task, sparse_task
        )
        
        return dense_vector, sparse_vector
    
    async def embed_batch_dense(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Batch embed multiple texts with dense vectors.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            
        Returns:
            List of dense vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Google's batch embedding
            if not self.client:
                raise ValueError("Google API client not initialized")
            
            response = await self.client.aio.models.embed_content(
                model=self.dense_model,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            
            all_embeddings.extend([list(e.values) for e in response.embeddings])
        
        return all_embeddings
    
    async def embed_batch_sparse(
        self,
        texts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Batch embed multiple texts with sparse vectors.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of sparse vectors
        """
        loop = asyncio.get_event_loop()
        
        def _batch_encode():
            embeddings = list(self.sparse_encoder.embed(texts))
            return [
                {
                    "indices": emb.indices.tolist(),
                    "values": emb.values.tolist()
                }
                for emb in embeddings
            ]
        
        return await loop.run_in_executor(None, _batch_encode)
    
    async def embed_batch_hybrid(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[Tuple[List[float], Dict[str, Any]]]:
        """
        Batch embed texts with both dense and sparse vectors.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for dense embedding
            
        Returns:
            List of (dense_vector, sparse_vector) tuples
        """
        dense_vectors = await self.embed_batch_dense(texts, batch_size)
        sparse_vectors = await self.embed_batch_sparse(texts)
        
        return list(zip(dense_vectors, sparse_vectors))
    
    async def embed_long_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Embed long text by chunking and averaging.
        
        For texts longer than the model's context window, this method
        chunks the text and averages the embeddings.
        
        Args:
            text: Long text to embed
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            Averaged (dense_vector, sparse_vector)
        """
        chunks = chunk_text(text, chunk_size, overlap)
        
        if len(chunks) == 1:
            return await self.embed_hybrid(chunks[0])
        
        # Embed all chunks
        chunk_embeddings = await self.embed_batch_hybrid(chunks)
        
        # Average dense vectors
        dense_vectors = [d for d, _ in chunk_embeddings]
        avg_dense = [
            sum(v[i] for v in dense_vectors) / len(dense_vectors)
            for i in range(len(dense_vectors[0]))
        ]
        
        # Merge sparse vectors (union of indices, max values)
        sparse_indices = {}
        for _, sparse in chunk_embeddings:
            for idx, val in zip(sparse["indices"], sparse["values"]):
                if idx not in sparse_indices or val > sparse_indices[idx]:
                    sparse_indices[idx] = val
        
        merged_sparse = {
            "indices": list(sparse_indices.keys()),
            "values": list(sparse_indices.values())
        }
        
        return avg_dense, merged_sparse


# ==================== Cached Instance ====================

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Get cached embedding service instance."""
    return EmbeddingService()
