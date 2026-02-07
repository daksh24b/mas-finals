"""
Memory Manager for Digital Truth Guardian.

Handles episodic memory and shared context collections.
Provides a unified interface for agents to:
- Record past decisions (episodic memory)
- Share context with other agents (shared context)
- Learn from past experiences
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    UpdateStatus,
)

from ..core.config import settings
from ..utils.logger import get_logger
from .schema import (
    EpisodicRecord,
    SharedContextRecord,
    EPISODIC_MEMORY_SCHEMA,
    SHARED_CONTEXT_SCHEMA,
    COLLECTION_EPISODIC_MEMORY,
    COLLECTION_SHARED_CONTEXT,
    EPISODIC_INDEXES,
    SHARED_CONTEXT_INDEXES,
    DENSE_VECTOR_NAME,
    DENSE_VECTOR_SIZE,
)
from .embeddings import get_embedding_service


logger = get_logger("Memory")


class MemoryManager:
    """
    Unified memory manager for episodic and shared context.
    
    This class implements:
    - Episodic Memory: Recording agent decisions and outcomes
    - Shared Context: Inter-agent communication via vector store
    - Experience Retrieval: Learning from similar past situations
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        
        # Initialize async client
        client_kwargs = {"url": self.url, "timeout": 30}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        
        self.client = AsyncQdrantClient(**client_kwargs)
        self._embedding_service = None
        
        logger.info("Memory Manager initialized")
    
    @property
    def embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service
    
    # ==================== Collection Management ====================
    
    async def ensure_collections_exist(self) -> Dict[str, bool]:
        """Create memory collections if they don't exist."""
        results = {}
        
        # Check existing collections
        collections = await self.client.get_collections()
        existing = {c.name for c in collections.collections}
        
        # Create episodic memory collection
        if COLLECTION_EPISODIC_MEMORY not in existing:
            await self._create_collection(
                COLLECTION_EPISODIC_MEMORY,
                EPISODIC_MEMORY_SCHEMA,
                EPISODIC_INDEXES
            )
            results[COLLECTION_EPISODIC_MEMORY] = True
            logger.info(f"Created collection: {COLLECTION_EPISODIC_MEMORY}")
        else:
            results[COLLECTION_EPISODIC_MEMORY] = False
        
        # Create shared context collection
        if COLLECTION_SHARED_CONTEXT not in existing:
            await self._create_collection(
                COLLECTION_SHARED_CONTEXT,
                SHARED_CONTEXT_SCHEMA,
                SHARED_CONTEXT_INDEXES
            )
            results[COLLECTION_SHARED_CONTEXT] = True
            logger.info(f"Created collection: {COLLECTION_SHARED_CONTEXT}")
        else:
            results[COLLECTION_SHARED_CONTEXT] = False
        
        return results
    
    async def _create_collection(
        self,
        name: str,
        schema,
        indexes: List[Dict]
    ):
        """Create a collection with the given schema."""
        # Create collection with dense vectors only (no sparse for memory)
        await self.client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE,
                    on_disk=schema.on_disk_payload,
                )
            },
            on_disk_payload=schema.on_disk_payload,
        )
        
        # Create payload indexes
        for idx in indexes:
            try:
                schema_type = getattr(
                    models.PayloadSchemaType,
                    idx["field_schema"].upper()
                )
                await self.client.create_payload_index(
                    collection_name=name,
                    field_name=idx["field_name"],
                    field_schema=schema_type,
                )
            except Exception as e:
                logger.debug(f"Index creation for {idx['field_name']}: {e}")
    
    # ==================== Episodic Memory Operations ====================
    
    async def record_episode(
        self,
        session_id: str,
        agent_name: str,
        action_type: str,
        query: str,
        outcome: str,
        decision_reasoning: str = "",
        confidence: float = 0.0,
        retrieval_score: Optional[float] = None,
        loop_count: int = 0,
        tools_used: Optional[List[str]] = None,
    ) -> str:
        """
        Record an agent's decision/action to episodic memory.
        
        This allows agents to learn from past decisions and outcomes.
        """
        record = EpisodicRecord(
            session_id=session_id,
            agent_name=agent_name,
            action_type=action_type,
            query=query,
            outcome=outcome,
            decision_reasoning=decision_reasoning,
            confidence=confidence,
            retrieval_score=retrieval_score,
            loop_count=loop_count,
            tools_used=tools_used or [],
        )
        
        # Generate embedding for the episode
        embedding_text = record.get_embedding_text()
        dense_vector = await self.embedding_service.embed_dense(embedding_text)
        
        # Create point
        point = PointStruct(
            id=record.id,
            vector={DENSE_VECTOR_NAME: dense_vector},
            payload=record.to_payload(),
        )
        
        # Upsert to episodic memory
        result = await self.client.upsert(
            collection_name=COLLECTION_EPISODIC_MEMORY,
            points=[point],
            wait=True,
        )
        
        if result.status == UpdateStatus.COMPLETED:
            logger.debug(f"Recorded episode: {agent_name}/{action_type} -> {outcome}")
            return record.id
        else:
            raise Exception(f"Failed to record episode: {result.status}")
    
    async def recall_similar_episodes(
        self,
        query: str,
        agent_name: Optional[str] = None,
        action_type: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 5,
        min_score: float = 0.6,
    ) -> List[EpisodicRecord]:
        """
        Retrieve similar past episodes to inform current decisions.
        
        This is the 'recall' mechanism for episodic memory.
        """
        # Generate query embedding
        dense_vector = await self.embedding_service.embed_dense(query)
        
        # Build filter conditions
        filter_conditions = []
        
        if agent_name:
            filter_conditions.append(
                FieldCondition(
                    key="agent_name",
                    match=MatchValue(value=agent_name)
                )
            )
        
        if action_type:
            filter_conditions.append(
                FieldCondition(
                    key="action_type",
                    match=MatchValue(value=action_type)
                )
            )
        
        if outcome:
            filter_conditions.append(
                FieldCondition(
                    key="outcome",
                    match=MatchValue(value=outcome)
                )
            )
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Search episodic memory
        results = await self.client.search(
            collection_name=COLLECTION_EPISODIC_MEMORY,
            query_vector=(DENSE_VECTOR_NAME, dense_vector),
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
            score_threshold=min_score,
        )
        
        # Convert to EpisodicRecord objects
        episodes = []
        for point in results:
            episodes.append(EpisodicRecord.from_payload(
                str(point.id),
                point.payload
            ))
        
        logger.debug(f"Recalled {len(episodes)} similar episodes for: {query[:50]}...")
        return episodes
    
    async def get_agent_success_rate(
        self,
        agent_name: str,
        action_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get success rate statistics for an agent."""
        filter_conditions = [
            FieldCondition(
                key="agent_name",
                match=MatchValue(value=agent_name)
            )
        ]
        
        if action_type:
            filter_conditions.append(
                FieldCondition(
                    key="action_type",
                    match=MatchValue(value=action_type)
                )
            )
        
        if since:
            filter_conditions.append(
                FieldCondition(
                    key="timestamp",
                    range=Range(gte=since.isoformat())
                )
            )
        
        # Count by outcome
        stats = {"total": 0, "success": 0, "failure": 0, "uncertain": 0}
        
        for outcome in ["success", "failure", "uncertain"]:
            conditions = filter_conditions + [
                FieldCondition(
                    key="outcome",
                    match=MatchValue(value=outcome)
                )
            ]
            
            try:
                count = await self.client.count(
                    collection_name=COLLECTION_EPISODIC_MEMORY,
                    count_filter=Filter(must=conditions)
                )
                stats[outcome] = count.count
                stats["total"] += count.count
            except Exception:
                pass
        
        # Calculate success rate
        if stats["total"] > 0:
            stats["success_rate"] = stats["success"] / stats["total"]
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    # ==================== Shared Context Operations ====================
    
    async def write_context(
        self,
        agent_source: str,
        context_type: str,
        content: str,
        session_id: Optional[str] = None,
        target_agents: Optional[List[str]] = None,
        priority: int = 1,
        ttl_minutes: Optional[int] = None,
        tags: Optional[List[str]] = None,
        related_query: Optional[str] = None,
    ) -> str:
        """
        Write shared context for other agents to read.
        
        Context types:
        - task_context: Information about current task
        - insight: Discovered patterns or useful information
        - warning: Potential issues or errors
        - strategy: Suggested approach for handling query
        - resource: Useful resource or reference
        """
        expires_at = None
        if ttl_minutes:
            expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
        
        record = SharedContextRecord(
            context_type=context_type,
            agent_source=agent_source,
            content=content,
            session_id=session_id,
            target_agents=target_agents or [],
            priority=priority,
            expires_at=expires_at,
            tags=tags or [],
            related_query=related_query,
        )
        
        # Generate embedding
        embedding_text = record.get_embedding_text()
        dense_vector = await self.embedding_service.embed_dense(embedding_text)
        
        # Create point
        point = PointStruct(
            id=record.id,
            vector={DENSE_VECTOR_NAME: dense_vector},
            payload=record.to_payload(),
        )
        
        # Upsert to shared context
        result = await self.client.upsert(
            collection_name=COLLECTION_SHARED_CONTEXT,
            points=[point],
            wait=True,
        )
        
        if result.status == UpdateStatus.COMPLETED:
            logger.debug(f"Wrote context: {agent_source}/{context_type}")
            return record.id
        else:
            raise Exception(f"Failed to write context: {result.status}")
    
    async def read_context(
        self,
        agent_name: str,
        session_id: Optional[str] = None,
        context_type: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 10,
        min_priority: int = 1,
    ) -> List[SharedContextRecord]:
        """
        Read shared context relevant to an agent.
        
        Filters out expired context and context not targeted at this agent.
        """
        filter_conditions = [
            # Priority filter
            FieldCondition(
                key="priority",
                range=Range(gte=min_priority)
            )
        ]
        
        if session_id:
            filter_conditions.append(
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id)
                )
            )
        
        if context_type:
            filter_conditions.append(
                FieldCondition(
                    key="context_type",
                    match=MatchValue(value=context_type)
                )
            )
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # If query provided, do semantic search; otherwise scroll
        if query:
            dense_vector = await self.embedding_service.embed_dense(query)
            results = await self.client.search(
                collection_name=COLLECTION_SHARED_CONTEXT,
                query_vector=(DENSE_VECTOR_NAME, dense_vector),
                query_filter=search_filter,
                limit=limit * 2,  # Over-fetch to filter
                with_payload=True,
            )
            points = results
        else:
            results = await self.client.scroll(
                collection_name=COLLECTION_SHARED_CONTEXT,
                scroll_filter=search_filter,
                limit=limit * 2,
                with_payload=True,
            )
            points = results[0]
        
        # Filter and convert to records
        contexts = []
        for point in points:
            payload = point.payload if hasattr(point, 'payload') else point.payload
            record = SharedContextRecord.from_payload(str(point.id), payload)
            
            # Skip expired context
            if record.is_expired():
                continue
            
            # Skip if not targeted at this agent
            if not record.is_for_agent(agent_name):
                continue
            
            contexts.append(record)
            
            if len(contexts) >= limit:
                break
        
        # Sort by priority (descending)
        contexts.sort(key=lambda x: x.priority, reverse=True)
        
        logger.debug(f"Read {len(contexts)} context items for agent: {agent_name}")
        return contexts
    
    async def clear_expired_context(self) -> int:
        """Remove expired shared context records."""
        now = datetime.utcnow().isoformat()
        
        # Find expired records
        results = await self.client.scroll(
            collection_name=COLLECTION_SHARED_CONTEXT,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="expires_at",
                        range=Range(lt=now)
                    )
                ]
            ),
            limit=1000,
        )
        
        expired_ids = [str(p.id) for p in results[0]]
        
        if expired_ids:
            await self.client.delete(
                collection_name=COLLECTION_SHARED_CONTEXT,
                points_selector=models.PointIdsList(points=expired_ids)
            )
            logger.info(f"Cleared {len(expired_ids)} expired context records")
        
        return len(expired_ids)
    
    # ==================== Collection Info ====================
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics for memory collections."""
        stats = {}
        
        for collection_name in [COLLECTION_EPISODIC_MEMORY, COLLECTION_SHARED_CONTEXT]:
            try:
                info = await self.client.get_collection(collection_name)
                stats[collection_name] = {
                    "status": str(info.status),
                    "points_count": info.points_count or 0,
                }
            except Exception as e:
                stats[collection_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return stats
    
    async def close(self):
        """Close the client connection."""
        await self.client.close()


# ==================== Singleton Instance ====================

_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get singleton memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


async def init_memory_collections() -> Dict[str, bool]:
    """Initialize memory collections."""
    manager = get_memory_manager()
    return await manager.ensure_collections_exist()
