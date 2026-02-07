"""
Qdrant client manager for database operations.

Handles connection, collection management, and CRUD operations
for the knowledge base with hybrid search support.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    PointIdsList,
    Filter,
    FieldCondition,
    MatchValue,
    IsNullCondition,
    Range,
    SearchParams,
    QuantizationSearchParams,
    ScoredPoint,
    UpdateStatus,
    PayloadSchemaType,
    SparseVector,
)

from ..core.config import settings
from ..core.state import Verdict, FactType, RetrievedDocument
from ..utils.logger import get_logger
from ..utils.helpers import generate_content_hash, async_retry
from .schema import (
    KnowledgeRecord,
    CollectionSchema,
    KNOWLEDGE_BASE_SCHEMA,
    PAYLOAD_INDEXES
)
from .embeddings import EmbeddingService, get_embedding_service


logger = get_logger()


class QdrantManager:
    """
    Manager for Qdrant database operations.
    
    Provides high-level operations for:
    - Collection management
    - Hybrid search (dense + sparse)
    - Knowledge base CRUD
    - Temporal versioning
    """
    
    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        collection_name: str = None,
        embedding_service: EmbeddingService = None
    ):
        """
        Initialize Qdrant manager.
        
        Args:
            url: Qdrant server URL
            api_key: Qdrant API key (for cloud)
            collection_name: Default collection name
            embedding_service: Embedding service instance
        """
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.embedding_service = embedding_service or get_embedding_service()
        
        # Initialize clients
        client_kwargs = {"url": self.url, "timeout": settings.qdrant_timeout}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        
        self.sync_client = QdrantClient(**client_kwargs)
        self.async_client = AsyncQdrantClient(**client_kwargs)
        
        logger.with_agent("Database").info(
            f"Initialized Qdrant manager: url={self.url}, collection={self.collection_name}"
        )
    
    # ==================== Collection Management ====================
    
    async def ensure_collection_exists(
        self,
        schema: CollectionSchema = None
    ) -> bool:
        """
        Ensure the collection exists, creating if necessary.
        
        Args:
            schema: Collection schema (defaults to knowledge_base schema)
            
        Returns:
            True if collection exists or was created
        """
        schema = schema or KNOWLEDGE_BASE_SCHEMA
        
        try:
            collections = await self.async_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if schema.name in collection_names:
                logger.with_agent("Database").info(
                    f"Collection '{schema.name}' already exists"
                )
                return True
            
            # Create collection
            config = schema.to_qdrant_config()
            await self.async_client.create_collection(
                collection_name=schema.name,
                **config
            )
            
            # Apply quantization
            quantization = schema.get_quantization_config()
            if quantization:
                await self.async_client.update_collection(
                    collection_name=schema.name,
                    quantization_config=quantization
                )
            
            # Create payload indexes
            for index in PAYLOAD_INDEXES:
                await self.async_client.create_payload_index(
                    collection_name=schema.name,
                    field_name=index["field_name"],
                    field_schema=index["field_schema"]
                )
            
            logger.with_agent("Database").info(
                f"Created collection '{schema.name}' with hybrid vectors"
            )
            return True
            
        except Exception as e:
            logger.with_agent("Database").error(
                f"Error ensuring collection: {str(e)}"
            )
            raise
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics and info."""
        info = await self.async_client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
            "config": {
                "dense_size": info.config.params.vectors.get("dense").size if hasattr(info.config.params.vectors, "get") else None,
            }
        }
    
    # ==================== Hybrid Search ====================
    
    @async_retry(max_retries=2, delay=0.5)
    async def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        alpha: float = None,
        filters: Optional[Filter] = None,
        score_threshold: float = None
    ) -> List[RetrievedDocument]:
        """
        Perform hybrid search combining dense and sparse vectors.
        
        Uses RRF (Reciprocal Rank Fusion) to combine results from
        both dense (semantic) and sparse (keyword) searches.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for dense vs sparse (0=sparse, 1=dense)
            filters: Qdrant filters to apply
            score_threshold: Minimum score threshold
            
        Returns:
            List of RetrievedDocument objects
        """
        top_k = top_k or settings.retriever_top_k
        alpha = alpha if alpha is not None else settings.hybrid_search_alpha
        score_threshold = score_threshold or settings.confidence_threshold
        
        # Generate embeddings
        dense_vector, sparse_vector = await asyncio.gather(
            self.embedding_service.embed_dense_query(query),
            self.embedding_service.embed_sparse(query)
        )
        
        # Build base filter for currently valid records
        base_filter = Filter(
            must=[
                FieldCondition(
                    key="valid_to",
                    is_null=IsNullCondition(is_null=True)
                )
            ]
        )
        
        # Merge with user filters
        if filters:
            if filters.must:
                base_filter.must.extend(filters.must)
            if filters.should:
                base_filter.should = filters.should
            if filters.must_not:
                base_filter.must_not = filters.must_not
        
        # Perform hybrid search using query API
        results = await self.async_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # Dense search
                {
                    "query": dense_vector,
                    "using": "dense",
                    "limit": top_k * 2,
                    "filter": base_filter
                },
                # Sparse search
                {
                    "query": SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"]
                    ),
                    "using": "sparse",
                    "limit": top_k * 2,
                    "filter": base_filter
                }
            ],
            query={"fusion": "rrf"},  # Reciprocal Rank Fusion
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold * 0.5  # RRF scores are lower
        )
        
        # Convert to RetrievedDocument objects
        documents = []
        for point in results.points:
            payload = point.payload or {}
            documents.append(RetrievedDocument(
                id=str(point.id),
                text=payload.get("text", ""),
                score=point.score,
                source_domain=payload.get("source_domain"),
                verdict=Verdict(payload["verdict"]) if payload.get("verdict") else None,
                fact_type=FactType(payload["fact_type"]) if payload.get("fact_type") else None,
                valid_from=datetime.fromisoformat(payload["valid_from"]) if payload.get("valid_from") else None,
                valid_to=datetime.fromisoformat(payload["valid_to"]) if payload.get("valid_to") else None,
                metadata={
                    "confidence": payload.get("confidence", 0),
                    "explanation": payload.get("explanation", ""),
                    "content_hash": payload.get("content_hash", "")
                }
            ))
        
        logger.with_agent("Retriever").info(
            f"Hybrid search returned {len(documents)} results for: {query[:50]}..."
        )
        
        return documents
    
    async def dense_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Filter] = None
    ) -> List[RetrievedDocument]:
        """Perform dense-only semantic search."""
        dense_vector = await self.embedding_service.embed_dense_query(query)
        
        results = await self.async_client.search(
            collection_name=self.collection_name,
            query_vector=("dense", dense_vector),
            limit=top_k,
            query_filter=filters,
            with_payload=True
        )
        
        return self._convert_search_results(results)
    
    async def sparse_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Filter] = None
    ) -> List[RetrievedDocument]:
        """Perform sparse-only keyword search."""
        sparse_vector = await self.embedding_service.embed_sparse(query)
        
        results = await self.async_client.search(
            collection_name=self.collection_name,
            query_vector=(
                "sparse",
                SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"]
                )
            ),
            limit=top_k,
            query_filter=filters,
            with_payload=True
        )
        
        return self._convert_search_results(results)
    
    def _convert_search_results(
        self,
        results: List[ScoredPoint]
    ) -> List[RetrievedDocument]:
        """Convert Qdrant search results to RetrievedDocument objects."""
        documents = []
        for point in results:
            payload = point.payload or {}
            documents.append(RetrievedDocument(
                id=str(point.id),
                text=payload.get("text", ""),
                score=point.score,
                source_domain=payload.get("source_domain"),
                verdict=Verdict(payload["verdict"]) if payload.get("verdict") else None,
                fact_type=FactType(payload["fact_type"]) if payload.get("fact_type") else None,
                metadata=payload
            ))
        return documents
    
    # ==================== CRUD Operations ====================
    
    async def upsert_record(
        self,
        record: KnowledgeRecord
    ) -> bool:
        """
        Insert or update a knowledge record.
        
        Args:
            record: KnowledgeRecord to upsert
            
        Returns:
            True if successful
        """
        # Generate embeddings if not present
        if record.vector_dense is None or record.vector_sparse is None:
            dense, sparse = await self.embedding_service.embed_hybrid(record.text)
            record.vector_dense = dense
            record.vector_sparse = sparse
        
        # Generate content hash for deduplication
        if not record.content_hash:
            record.content_hash = generate_content_hash(record.text)
        
        # Create point
        point = PointStruct(
            id=record.id,
            vector={
                "dense": record.vector_dense,
                "sparse": SparseVector(
                    indices=record.vector_sparse["indices"],
                    values=record.vector_sparse["values"]
                )
            },
            payload=record.to_payload()
        )
        
        # Upsert
        result = await self.async_client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        success = result.status == UpdateStatus.COMPLETED
        
        if success:
            logger.with_agent("Archivist").info(
                f"Upserted record: {record.id} (verdict={record.verdict.value})"
            )
        
        return success
    
    async def batch_upsert(
        self,
        records: List[KnowledgeRecord],
        batch_size: int = 100
    ) -> int:
        """
        Batch upsert multiple records.
        
        Args:
            records: List of records to upsert
            batch_size: Number of records per batch
            
        Returns:
            Number of records successfully upserted
        """
        # Generate embeddings in batches
        texts = [r.text for r in records]
        embeddings = await self.embedding_service.embed_batch_hybrid(texts, batch_size)
        
        # Assign embeddings to records
        for record, (dense, sparse) in zip(records, embeddings):
            record.vector_dense = dense
            record.vector_sparse = sparse
            if not record.content_hash:
                record.content_hash = generate_content_hash(record.text)
        
        # Create points
        points = []
        for record in records:
            points.append(PointStruct(
                id=record.id,
                vector={
                    "dense": record.vector_dense,
                    "sparse": SparseVector(
                        indices=record.vector_sparse["indices"],
                        values=record.vector_sparse["values"]
                    )
                },
                payload=record.to_payload()
            ))
        
        # Batch upsert
        success_count = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            result = await self.async_client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            if result.status == UpdateStatus.COMPLETED:
                success_count += len(batch)
        
        logger.with_agent("Archivist").info(
            f"Batch upserted {success_count}/{len(records)} records"
        )
        
        return success_count
    
    async def check_duplicate(
        self,
        text: str,
        threshold: float = 0.95
    ) -> Optional[KnowledgeRecord]:
        """
        Check if a similar record already exists.
        
        Args:
            text: Text to check for duplicates
            threshold: Similarity threshold
            
        Returns:
            Existing record if duplicate found, None otherwise
        """
        content_hash = generate_content_hash(text)
        
        # First check by hash (exact match)
        results = await self.async_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="content_hash",
                        match=MatchValue(value=content_hash)
                    )
                ]
            ),
            limit=1,
            with_payload=True
        )
        
        if results[0]:
            point = results[0][0]
            return KnowledgeRecord.from_payload(str(point.id), point.payload)
        
        # Then check by semantic similarity
        docs = await self.dense_search(text, top_k=1)
        if docs and docs[0].score >= threshold:
            # Fetch full record
            point = await self.async_client.retrieve(
                collection_name=self.collection_name,
                ids=[docs[0].id],
                with_payload=True
            )
            if point:
                return KnowledgeRecord.from_payload(
                    str(point[0].id),
                    point[0].payload
                )
        
        return None
    
    async def expire_record(
        self,
        record_id: str
    ) -> bool:
        """
        Expire a record (set valid_to timestamp).
        
        Used for temporal versioning of transient facts.
        
        Args:
            record_id: ID of record to expire
            
        Returns:
            True if successful
        """
        result = await self.async_client.set_payload(
            collection_name=self.collection_name,
            payload={
                "valid_to": datetime.utcnow().isoformat()
            },
            points=[record_id]
        )
        
        success = result.status == UpdateStatus.COMPLETED
        
        if success:
            logger.with_agent("Archivist").info(
                f"Expired record: {record_id}"
            )
        
        return success
    
    async def update_transient_fact(
        self,
        old_record_id: str,
        new_record: KnowledgeRecord
    ) -> bool:
        """
        Update a transient fact with temporal versioning.
        
        Expires the old record and inserts the new one.
        
        Args:
            old_record_id: ID of the existing record to expire
            new_record: New version of the record
            
        Returns:
            True if both operations succeed
        """
        # Expire old record
        expired = await self.expire_record(old_record_id)
        if not expired:
            return False
        
        # Insert new record
        return await self.upsert_record(new_record)
    
    async def delete_record(self, record_id: str) -> bool:
        """Delete a record by ID."""
        result = await self.async_client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[record_id])
        )
        return result.status == UpdateStatus.COMPLETED
    
    async def get_record_by_id(
        self,
        record_id: str
    ) -> Optional[KnowledgeRecord]:
        """Retrieve a record by its ID."""
        points = await self.async_client.retrieve(
            collection_name=self.collection_name,
            ids=[record_id],
            with_payload=True
        )
        
        if points:
            return KnowledgeRecord.from_payload(
                str(points[0].id),
                points[0].payload
            )
        return None
    
    # ==================== Analytics ====================
    
    async def get_verdict_stats(self) -> Dict[str, int]:
        """Get count of records by verdict."""
        stats = {}
        
        for verdict in Verdict:
            results = await self.async_client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="verdict",
                            match=MatchValue(value=verdict.value)
                        ),
                        FieldCondition(
                            key="valid_to",
                            is_null=IsNullCondition(is_null=True)
                        )
                    ]
                )
            )
            stats[verdict.value] = results.count
        
        return stats
    
    async def close(self):
        """Close client connections."""
        await self.async_client.close()
        self.sync_client.close()


# ==================== Cached Instance ====================

@lru_cache()
def get_qdrant_manager() -> QdrantManager:
    """Get cached Qdrant manager instance."""
    return QdrantManager()
