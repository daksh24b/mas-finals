"""
Qdrant Database Client

Manages connections and operations with Qdrant vector database.
Implements hybrid search, temporal versioning, and collection management.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    SparseVector,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    SearchRequest,
    ScoredPoint,
    UpdateStatus,
)

from ..core.config import settings
from ..core.state import Verdict, FactType
from ..utils.logger import get_logger
from .schema import (
    KnowledgeRecord,
    SearchResult,
    VerdictType,
    FactType as SchemaFactType,
    SourceTier,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    DENSE_VECTOR_SIZE,
)
from .embeddings import get_embedding_service


logger = get_logger("Database")


class QdrantManager:
    """
    Manages Qdrant database operations for the Digital Truth Guardian.
    
    Implements:
    - Hybrid search (dense + sparse vectors)
    - Temporal versioning for transient facts
    - Binary quantization for efficiency
    - Metadata filtering
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.collection_name = collection_name or settings.qdrant_collection_name
        
        # Initialize clients
        client_kwargs = {"url": self.url, "timeout": 30}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        
        self.client = QdrantClient(**client_kwargs)
        self.async_client = AsyncQdrantClient(**client_kwargs)
        
        # Embedding service
        self._embedding_service = None
        
        logger.info(f"Initialized Qdrant manager: url={self.url}, collection={self.collection_name}")
    
    @property
    def embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service
    
    # ==================== Collection Management ====================
    
    async def ensure_collection_exists(self) -> bool:
        """Create collection if it doesn't exist."""
        try:
            collections = await self.async_client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)
            
            if not exists:
                await self._create_collection()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    async def _create_collection(self):
        """Create the knowledge base collection with hybrid vectors."""
        await self.async_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE,
                    on_disk=True,
                    # Binary quantization for efficiency
                    quantization_config=models.BinaryQuantization(
                        binary=models.BinaryQuantizationConfig(
                            always_ram=True
                        )
                    )
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False
                    )
                )
            },
            # Optimized for hybrid search
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000,
            ),
            # Enable payload indexing for filters
            on_disk_payload=True,
        )
        
        # Create payload indexes for common filters
        await self._create_payload_indexes()
        
        logger.info(f"Created collection '{self.collection_name}' with hybrid vectors")
    
    async def _create_payload_indexes(self):
        """Create indexes on commonly filtered fields."""
        index_fields = [
            ("verdict", models.PayloadSchemaType.KEYWORD),
            ("fact_type", models.PayloadSchemaType.KEYWORD),
            ("source_tier", models.PayloadSchemaType.KEYWORD),
            ("source_domain", models.PayloadSchemaType.KEYWORD),
            ("valid_to", models.PayloadSchemaType.DATETIME),
        ]
        
        for field_name, field_type in index_fields:
            try:
                await self.async_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as e:
                # Index might already exist
                logger.debug(f"Index creation for {field_name}: {e}")
    
    async def delete_collection(self):
        """Delete the collection."""
        await self.async_client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection '{self.collection_name}'")
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            info = await self.async_client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "status": str(info.status),
                "points_count": info.points_count or 0,
                "vectors_count": info.points_count or 0,  # Use points_count as proxy
                "indexed_vectors_count": info.indexed_vectors_count or 0,
                "segments_count": info.segments_count or 0,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "name": self.collection_name,
                "status": "error",
                "points_count": 0,
                "vectors_count": 0,
                "error": str(e)
            }
    
    # ==================== Write Operations ====================
    
    async def upsert_record(self, record: KnowledgeRecord) -> str:
        """
        Insert or update a knowledge record.
        
        For TRANSIENT facts, implements temporal versioning.
        """
        # Generate embeddings
        dense_vector, sparse_vector = await self.embedding_service.embed_hybrid(record.text)
        
        # Handle temporal versioning for transient facts
        if record.fact_type == FactType.TRANSIENT:
            await self._expire_old_versions(record.text, record.content_hash)
        
        # Prepare point
        point = PointStruct(
            id=record.id,
            vector={
                DENSE_VECTOR_NAME: dense_vector,
                SPARSE_VECTOR_NAME: sparse_vector,
            },
            payload=record.to_payload(),
        )
        
        # Upsert to Qdrant
        result = await self.async_client.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=True,
        )
        
        if result.status == UpdateStatus.COMPLETED:
            logger.info(f"Upserted record: {record.id} ({record.verdict})")
            return record.id
        else:
            raise Exception(f"Upsert failed with status: {result.status}")
    
    async def _expire_old_versions(self, text: str, claim_hash: Optional[str] = None):
        """Mark old versions of a transient fact as expired."""
        if not claim_hash:
            return
        
        # Find existing records with same claim hash that are still valid
        results = await self.async_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="claim_hash",
                        match=MatchValue(value=claim_hash)
                    ),
                    FieldCondition(
                        key="valid_to",
                        is_null=True
                    )
                ]
            ),
            limit=100,
        )
        
        # Update valid_to for all found records
        now = datetime.utcnow().isoformat()
        for point in results[0]:
            await self.async_client.set_payload(
                collection_name=self.collection_name,
                payload={"valid_to": now},
                points=[point.id],
            )
            logger.debug(f"Expired old version: {point.id}")
    
    async def batch_upsert(self, records: List[KnowledgeRecord]) -> List[str]:
        """Batch insert multiple records."""
        if not records:
            return []
        
        points = []
        for record in records:
            dense_vector, sparse_vector = await self.embedding_service.embed_hybrid(record.text)
            
            points.append(PointStruct(
                id=record.id,
                vector={
                    DENSE_VECTOR_NAME: dense_vector,
                    SPARSE_VECTOR_NAME: sparse_vector,
                },
                payload=record.to_payload(),
            ))
        
        result = await self.async_client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        
        if result.status == UpdateStatus.COMPLETED:
            logger.info(f"Batch upserted {len(records)} records")
            return [r.id for r in records]
        else:
            raise Exception(f"Batch upsert failed: {result.status}")
    
    # ==================== Search Operations ====================
    
    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5,
        fact_type: Optional[FactType] = None,
        verdict: Optional[VerdictType] = None,
        only_valid: bool = True,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining dense and sparse vectors.
        
        Uses Reciprocal Rank Fusion (RRF) to combine results.
        """
        # Generate query embeddings (using RETRIEVAL_QUERY task type)
        dense_vector, sparse_vector = await self.embedding_service.embed_hybrid_query(query)
        
        # Build filter
        filter_conditions = []
        
        if only_valid:
            # Only get currently valid records
            filter_conditions.append(
                FieldCondition(
                    key="valid_to",
                    is_null=True
                )
            )
        
        if fact_type:
            filter_conditions.append(
                FieldCondition(
                    key="fact_type",
                    match=MatchValue(value=fact_type.value)
                )
            )
        
        if verdict:
            filter_conditions.append(
                FieldCondition(
                    key="verdict",
                    match=MatchValue(value=verdict.value)
                )
            )
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Perform dense search using query_points (new API)
        dense_results = await self.async_client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=search_filter,
            limit=limit * 2,  # Over-fetch for fusion
            with_payload=True,
            score_threshold=score_threshold * 0.8,  # Slightly lower threshold
        )
        
        # Perform sparse search using query_points (new API)
        sparse_results = await self.async_client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"],
            ),
            using=SPARSE_VECTOR_NAME,
            query_filter=search_filter,
            limit=limit * 2,
            with_payload=True,
        )
        
        # Fuse results using RRF
        fused_results = self._reciprocal_rank_fusion(
            dense_results.points,
            sparse_results.points,
            k=60  # RRF constant
        )
        
        # Convert to SearchResult objects
        # Note: RRF scores are naturally small (0.01-0.02 range), so we don't
        # apply score_threshold here. The pre-search thresholds already filtered.
        search_results = []
        for point, rrf_score in fused_results[:limit]:
            # Use the original dense score for the result (more interpretable)
            original_score = point.score if point.score else rrf_score
            search_results.append(SearchResult(
                id=str(point.id),
                text=point.payload.get("text", ""),
                score=original_score,
                verdict=VerdictType(point.payload.get("verdict", "UNCERTAIN")),
                fact_type=FactType(point.payload.get("fact_type", "STATIC")),
                source_domain=point.payload.get("source_domain"),
                source_tier=SourceTier(point.payload.get("source_tier", "TIER_5")),
                explanation=point.payload.get("explanation"),
                valid_from=point.payload.get("valid_from"),
                valid_to=point.payload.get("valid_to"),
            ))
        
        logger.debug(f"Hybrid search returned {len(search_results)} results for: {query[:50]}...")
        return search_results
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[ScoredPoint],
        sparse_results: List[ScoredPoint],
        k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> List[Tuple[ScoredPoint, float]]:
        """
        Combine dense and sparse search results using Reciprocal Rank Fusion.
        
        RRF Score = Σ (weight / (k + rank))
        """
        scores: Dict[str, float] = {}
        points: Dict[str, ScoredPoint] = {}
        
        # Process dense results
        for rank, point in enumerate(dense_results, 1):
            point_id = str(point.id)
            scores[point_id] = scores.get(point_id, 0) + dense_weight / (k + rank)
            points[point_id] = point
        
        # Process sparse results
        for rank, point in enumerate(sparse_results, 1):
            point_id = str(point.id)
            scores[point_id] = scores.get(point_id, 0) + sparse_weight / (k + rank)
            if point_id not in points:
                points[point_id] = point
        
        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [(points[pid], scores[pid]) for pid in sorted_ids]
    
    async def search_similar(
        self,
        text: str,
        limit: int = 5,
        threshold: float = 0.6,
    ) -> List[SearchResult]:
        """Find highly similar records (for deduplication)."""
        return await self.hybrid_search(
            query=text,
            limit=limit,
            score_threshold=threshold,
        )
    
    async def check_duplicate(
        self,
        claim: str,
        threshold: float = 0.9,
    ) -> Optional[KnowledgeRecord]:
        """
        Check if a similar claim already exists in the knowledge base.
        
        Args:
            claim: The claim text to check
            threshold: Similarity threshold for duplicate detection
            
        Returns:
            Existing KnowledgeRecord if duplicate found, None otherwise
        """
        try:
            similar = await self.search_similar(
                text=claim,
                limit=1,
                threshold=threshold,
            )
            
            if similar and len(similar) > 0:
                # Found a potential duplicate - convert RetrievedDocument to KnowledgeRecord
                doc = similar[0]
                return KnowledgeRecord(
                    id=doc.id,
                    text=doc.text,
                    verdict=doc.verdict if doc.verdict else Verdict.PENDING,
                    fact_type=doc.fact_type if doc.fact_type else FactType.STATIC,
                    source_domain=doc.source_domain or "unknown",
                    confidence=doc.score,
                )
            return None
        except Exception as e:
            logger.debug(f"Duplicate check failed: {e}")
            return None
    
    async def get_by_id(self, record_id: str) -> Optional[KnowledgeRecord]:
        """Retrieve a specific record by ID."""
        try:
            results = await self.async_client.retrieve(
                collection_name=self.collection_name,
                ids=[record_id],
                with_payload=True,
            )
            
            if results:
                return KnowledgeRecord.from_payload(
                    results[0].id,
                    results[0].payload
                )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get record {record_id}: {e}")
            return None
    
    # ==================== Statistics ====================
    
    async def get_verdict_stats(self) -> Dict[str, int]:
        """Get count of records by verdict type."""
        stats = {}
        
        for verdict in VerdictType:
            try:
                count = await self.async_client.count(
                    collection_name=self.collection_name,
                    count_filter=Filter(
                        must=[
                            FieldCondition(
                                key="verdict",
                                match=MatchValue(value=verdict.value)
                            ),
                            FieldCondition(
                                key="valid_to",
                                is_null=True
                            )
                        ]
                    )
                )
                stats[verdict.value] = count.count
            except Exception:
                stats[verdict.value] = 0
        
        return stats
    
    async def close(self):
        """Close client connections."""
        await self.async_client.close()


# ==================== Singleton Instance ====================

_qdrant_manager: Optional[QdrantManager] = None


def get_qdrant_manager() -> QdrantManager:
    """Get singleton Qdrant manager instance."""
    global _qdrant_manager
    if _qdrant_manager is None:
        _qdrant_manager = QdrantManager()
    return _qdrant_manager


async def init_qdrant() -> QdrantManager:
    """Initialize Qdrant and ensure collection exists."""
    manager = get_qdrant_manager()
    await manager.ensure_collection_exists()
    return manager