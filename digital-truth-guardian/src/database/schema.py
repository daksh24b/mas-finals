"""
Schema definitions for Qdrant collections.

Defines the data models and collection schemas for the knowledge base.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..core.state import Verdict, FactType


@dataclass
class KnowledgeRecord:
    """
    A record in the knowledge base collection.
    
    Represents a verified or debunked claim with full provenance.
    """
    text: str
    verdict: Verdict
    fact_type: FactType
    source_domain: str
    
    # Auto-generated fields
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Temporal versioning
    valid_from: datetime = field(default_factory=datetime.utcnow)
    valid_to: Optional[datetime] = None  # None means currently valid
    
    # Embeddings (set during ingestion)
    vector_dense: Optional[List[float]] = None
    vector_sparse: Optional[Dict[str, Any]] = None
    
    # Additional metadata
    confidence: float = 0.0
    explanation: str = ""
    related_claims: List[str] = field(default_factory=list)
    content_hash: str = ""
    query_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format."""
        return {
            "text": self.text,
            "verdict": self.verdict.value,
            "fact_type": self.fact_type.value,
            "source_domain": self.source_domain,
            "created_at": self.created_at.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "related_claims": self.related_claims,
            "content_hash": self.content_hash,
            "query_count": self.query_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }
    
    @classmethod
    def from_payload(cls, id: str, payload: Dict[str, Any]) -> "KnowledgeRecord":
        """Create from Qdrant payload."""
        return cls(
            id=id,
            text=payload.get("text", ""),
            verdict=Verdict(payload.get("verdict", "UNCERTAIN")),
            fact_type=FactType(payload.get("fact_type", "STATIC")),
            source_domain=payload.get("source_domain", ""),
            created_at=datetime.fromisoformat(payload["created_at"]) if payload.get("created_at") else datetime.utcnow(),
            valid_from=datetime.fromisoformat(payload["valid_from"]) if payload.get("valid_from") else datetime.utcnow(),
            valid_to=datetime.fromisoformat(payload["valid_to"]) if payload.get("valid_to") else None,
            confidence=payload.get("confidence", 0.0),
            explanation=payload.get("explanation", ""),
            related_claims=payload.get("related_claims", []),
            content_hash=payload.get("content_hash", ""),
            query_count=payload.get("query_count", 0),
            last_accessed=datetime.fromisoformat(payload["last_accessed"]) if payload.get("last_accessed") else None
        )
    
    def is_currently_valid(self) -> bool:
        """Check if this record is currently valid (not expired)."""
        if self.valid_to is None:
            return True
        return datetime.utcnow() < self.valid_to
    
    def expire(self) -> "KnowledgeRecord":
        """Mark this record as expired (for temporal versioning)."""
        self.valid_to = datetime.utcnow()
        return self


@dataclass
class CollectionSchema:
    """
    Schema definition for a Qdrant collection.
    
    Configures vectors, indexes, and optimization settings.
    """
    name: str
    dense_vector_size: int = 768
    dense_distance: str = "Cosine"
    enable_sparse: bool = True
    
    # Optimization settings
    on_disk_payload: bool = True
    enable_quantization: bool = True
    
    # HNSW index settings
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    
    # Sharding for horizontal scaling
    shard_number: int = 2
    replication_factor: int = 1
    
    def to_qdrant_config(self) -> Dict[str, Any]:
        """Generate Qdrant collection configuration."""
        from qdrant_client.models import (
            VectorParams,
            SparseVectorParams,
            Distance,
            HnswConfigDiff,
            OptimizersConfigDiff,
            BinaryQuantization,
            BinaryQuantizationConfig,
        )
        
        config = {
            "vectors_config": {
                "dense": VectorParams(
                    size=self.dense_vector_size,
                    distance=Distance.COSINE if self.dense_distance == "Cosine" else Distance.DOT,
                    on_disk=self.on_disk_payload,
                    hnsw_config=HnswConfigDiff(
                        m=self.hnsw_m,
                        ef_construct=self.hnsw_ef_construct
                    )
                )
            },
            "shard_number": self.shard_number,
            "replication_factor": self.replication_factor,
            "on_disk_payload": self.on_disk_payload,
        }
        
        if self.enable_sparse:
            config["sparse_vectors_config"] = {
                "sparse": SparseVectorParams()
            }
        
        return config
    
    def get_quantization_config(self) -> Optional[Any]:
        """Get binary quantization configuration."""
        if not self.enable_quantization:
            return None
        
        from qdrant_client.models import BinaryQuantization, BinaryQuantizationConfig
        
        return BinaryQuantization(
            binary=BinaryQuantizationConfig(
                always_ram=True
            )
        )


# ==================== Default Schemas ====================

KNOWLEDGE_BASE_SCHEMA = CollectionSchema(
    name="knowledge_base",
    dense_vector_size=768,  # Google text-embedding-004 dimension
    dense_distance="Cosine",
    enable_sparse=True,
    on_disk_payload=True,
    enable_quantization=True,
    hnsw_m=16,
    hnsw_ef_construct=100,
    shard_number=2,
    replication_factor=1
)


# ==================== Index Definitions ====================

PAYLOAD_INDEXES = [
    {"field_name": "verdict", "field_schema": "keyword"},
    {"field_name": "fact_type", "field_schema": "keyword"},
    {"field_name": "source_domain", "field_schema": "keyword"},
    {"field_name": "valid_to", "field_schema": "datetime"},
    {"field_name": "content_hash", "field_schema": "keyword"},
]
