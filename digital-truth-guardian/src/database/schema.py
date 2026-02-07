"""
Schema definitions for Qdrant collections.

Defines the data models and collection schemas for the knowledge base.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..core.state import Verdict, FactType


# ==================== Type Aliases for backward compatibility ====================

VerdictType = Verdict  # Alias for use in qdrant_client


# ==================== Enums ====================

class SourceTier(str, Enum):
    """Source credibility tiers."""
    TIER_1 = "TIER_1"  # Government, Academic, Scientific journals
    TIER_2 = "TIER_2"  # Major news, Fact-checkers
    TIER_3 = "TIER_3"  # Reputable tech/science publications
    TIER_4 = "TIER_4"  # General news, Wikipedia
    TIER_5 = "TIER_5"  # Unverified sources


# ==================== Constants ====================

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 768  # Google text-embedding-004


# ==================== Data Classes ====================

@dataclass
class SearchResult:
    """Result from a hybrid search operation."""
    id: str
    text: str
    score: float
    verdict: Verdict
    fact_type: FactType
    source_domain: Optional[str] = None
    source_tier: SourceTier = SourceTier.TIER_5
    explanation: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "verdict": self.verdict.value,
            "fact_type": self.fact_type.value,
            "source_domain": self.source_domain,
            "source_tier": self.source_tier.value,
            "explanation": self.explanation,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }


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


# ==================== Episodic Memory Record ====================

@dataclass
class EpisodicRecord:
    """
    A record in the episodic memory collection.
    
    Stores past agent interactions, decisions, and outcomes
    for learning from experience. This enables agents to recall
    what strategies worked for similar queries in the past.
    """
    session_id: str
    agent_name: str
    action_type: str  # "retrieval", "search", "critique", "archive", "route"
    query: str
    outcome: str  # "success", "failure", "uncertain", "cache_hit"
    
    # Auto-generated fields
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Context about the decision
    decision_reasoning: str = ""
    input_summary: str = ""
    output_summary: str = ""
    confidence: float = 0.0
    
    # Metrics for learning
    retrieval_score: Optional[float] = None
    loop_count: int = 0
    tools_used: List[str] = field(default_factory=list)
    
    # Feedback signals
    feedback_received: Optional[str] = None
    was_helpful: Optional[bool] = None
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format."""
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "action_type": self.action_type,
            "query": self.query,
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
            "decision_reasoning": self.decision_reasoning,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "confidence": self.confidence,
            "retrieval_score": self.retrieval_score,
            "loop_count": self.loop_count,
            "tools_used": self.tools_used,
            "feedback_received": self.feedback_received,
            "was_helpful": self.was_helpful,
        }
    
    @classmethod
    def from_payload(cls, id: str, payload: Dict[str, Any]) -> "EpisodicRecord":
        """Create from Qdrant payload."""
        return cls(
            id=id,
            session_id=payload.get("session_id", ""),
            agent_name=payload.get("agent_name", ""),
            action_type=payload.get("action_type", ""),
            query=payload.get("query", ""),
            outcome=payload.get("outcome", ""),
            timestamp=datetime.fromisoformat(payload["timestamp"]) if payload.get("timestamp") else datetime.utcnow(),
            decision_reasoning=payload.get("decision_reasoning", ""),
            input_summary=payload.get("input_summary", ""),
            output_summary=payload.get("output_summary", ""),
            confidence=payload.get("confidence", 0.0),
            retrieval_score=payload.get("retrieval_score"),
            loop_count=payload.get("loop_count", 0),
            tools_used=payload.get("tools_used", []),
            feedback_received=payload.get("feedback_received"),
            was_helpful=payload.get("was_helpful"),
        )
    
    def get_embedding_text(self) -> str:
        """Get text for embedding generation."""
        return f"{self.query} | {self.action_type} | {self.outcome} | {self.decision_reasoning}"


# ==================== Shared Context Record ====================

@dataclass
class SharedContextRecord:
    """
    A record in the shared context collection.
    
    Enables multiple agents to read from and write to
    a shared memory space for coordination and collective intelligence.
    """
    context_type: str  # "task_context", "insight", "warning", "strategy", "resource"
    agent_source: str  # Which agent wrote this
    content: str
    
    # Auto-generated fields
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None  # For temporary context (None = permanent)
    
    # Targeting specific agents
    target_agents: List[str] = field(default_factory=list)  # Empty = all agents can read
    priority: int = 1  # 1-5, higher = more important
    
    # Context metadata
    related_query: Optional[str] = None
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Usage tracking
    read_count: int = 0
    last_read_by: Optional[str] = None
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format."""
        return {
            "context_type": self.context_type,
            "agent_source": self.agent_source,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "target_agents": self.target_agents,
            "priority": self.priority,
            "related_query": self.related_query,
            "session_id": self.session_id,
            "tags": self.tags,
            "read_count": self.read_count,
            "last_read_by": self.last_read_by,
        }
    
    @classmethod
    def from_payload(cls, id: str, payload: Dict[str, Any]) -> "SharedContextRecord":
        """Create from Qdrant payload."""
        return cls(
            id=id,
            context_type=payload.get("context_type", ""),
            agent_source=payload.get("agent_source", ""),
            content=payload.get("content", ""),
            created_at=datetime.fromisoformat(payload["created_at"]) if payload.get("created_at") else datetime.utcnow(),
            expires_at=datetime.fromisoformat(payload["expires_at"]) if payload.get("expires_at") else None,
            target_agents=payload.get("target_agents", []),
            priority=payload.get("priority", 1),
            related_query=payload.get("related_query"),
            session_id=payload.get("session_id"),
            tags=payload.get("tags", []),
            read_count=payload.get("read_count", 0),
            last_read_by=payload.get("last_read_by"),
        )
    
    def is_expired(self) -> bool:
        """Check if this context has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_for_agent(self, agent_name: str) -> bool:
        """Check if this context is targeted at a specific agent."""
        if not self.target_agents:  # Empty = all agents
            return True
        return agent_name in self.target_agents
    
    def get_embedding_text(self) -> str:
        """Get text for embedding generation."""
        return f"{self.context_type}: {self.content}"


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

# Episodic Memory: Stores past agent interactions and decisions
EPISODIC_MEMORY_SCHEMA = CollectionSchema(
    name="episodic_memory",
    dense_vector_size=768,
    dense_distance="Cosine",
    enable_sparse=False,  # Semantic search only for episodes
    on_disk_payload=True,
    enable_quantization=True,
    hnsw_m=16,
    hnsw_ef_construct=100,
    shard_number=1,
    replication_factor=1
)

# Shared Agent Context: For inter-agent communication and coordination
SHARED_CONTEXT_SCHEMA = CollectionSchema(
    name="shared_context",
    dense_vector_size=768,
    dense_distance="Cosine",
    enable_sparse=False,
    on_disk_payload=True,
    enable_quantization=False,  # Small collection, no need for quantization
    hnsw_m=8,
    hnsw_ef_construct=64,
    shard_number=1,
    replication_factor=1
)


# ==================== Collection Names ====================

COLLECTION_KNOWLEDGE_BASE = "knowledge_base"
COLLECTION_EPISODIC_MEMORY = "episodic_memory"
COLLECTION_SHARED_CONTEXT = "shared_context"


# ==================== Index Definitions ====================

PAYLOAD_INDEXES = [
    {"field_name": "verdict", "field_schema": "keyword"},
    {"field_name": "fact_type", "field_schema": "keyword"},
    {"field_name": "source_domain", "field_schema": "keyword"},
    {"field_name": "valid_to", "field_schema": "datetime"},
    {"field_name": "content_hash", "field_schema": "keyword"},
]

EPISODIC_INDEXES = [
    {"field_name": "session_id", "field_schema": "keyword"},
    {"field_name": "agent_name", "field_schema": "keyword"},
    {"field_name": "action_type", "field_schema": "keyword"},
    {"field_name": "outcome", "field_schema": "keyword"},
    {"field_name": "timestamp", "field_schema": "datetime"},
]

SHARED_CONTEXT_INDEXES = [
    {"field_name": "context_type", "field_schema": "keyword"},
    {"field_name": "agent_source", "field_schema": "keyword"},
    {"field_name": "session_id", "field_schema": "keyword"},
    {"field_name": "priority", "field_schema": "integer"},
    {"field_name": "expires_at", "field_schema": "datetime"},
]
