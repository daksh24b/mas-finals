"""
Database module for Qdrant operations and embedding generation.

Collections:
- knowledge_base: Verified facts and claims
- episodic_memory: Agent decision history
- shared_context: Inter-agent communication
"""

from .qdrant_client import QdrantManager, get_qdrant_manager, init_qdrant
from .embeddings import EmbeddingService, get_embedding_service
from .memory_manager import MemoryManager, get_memory_manager, init_memory_collections
from .schema import (
    KnowledgeRecord,
    EpisodicRecord,
    SharedContextRecord,
    CollectionSchema,
    SearchResult,
    SourceTier,
    COLLECTION_KNOWLEDGE_BASE,
    COLLECTION_EPISODIC_MEMORY,
    COLLECTION_SHARED_CONTEXT,
)

__all__ = [
    # Managers
    "QdrantManager",
    "get_qdrant_manager",
    "init_qdrant",
    "EmbeddingService",
    "get_embedding_service",
    "MemoryManager",
    "get_memory_manager",
    "init_memory_collections",
    # Records
    "KnowledgeRecord",
    "EpisodicRecord",
    "SharedContextRecord",
    "CollectionSchema",
    "SearchResult",
    "SourceTier",
    # Collection names
    "COLLECTION_KNOWLEDGE_BASE",
    "COLLECTION_EPISODIC_MEMORY",
    "COLLECTION_SHARED_CONTEXT",
]
