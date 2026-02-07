"""
Retriever Agent (The Memory Specialist)

Responsible for:
- Interfacing with Qdrant for knowledge retrieval
- Executing hybrid search (dense + sparse)
- Applying metadata filters
- Determining cache hits/misses
"""

from typing import List, Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue, IsNullCondition

from .base import BaseAgent
from ..core.config import settings
from ..core.state import (
    AgentState,
    RetrievedDocument,
    add_agent_to_trace
)
from ..database.qdrant_client import QdrantManager, get_qdrant_manager
from ..utils.logger import get_logger


logger = get_logger()


class RetrieverAgent(BaseAgent):
    """
    The Retriever Agent (The Memory Specialist).
    
    Interfaces with Qdrant to perform contextual retrieval.
    Implements hybrid search combining:
    - Dense vectors (Google text-embedding-004) for semantic matching
    - Sparse vectors (FastEmbed/BM25) for keyword precision
    """
    
    name = "Retriever"
    
    def __init__(self, qdrant_manager: QdrantManager = None):
        """
        Initialize the Retriever agent.
        
        Args:
            qdrant_manager: Qdrant manager instance
        """
        self.qdrant = qdrant_manager or get_qdrant_manager()
        self.top_k = settings.retriever_top_k
        self.confidence_threshold = settings.confidence_threshold
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process state by retrieving relevant documents.
        
        Performs hybrid search and evaluates if results are sufficient.
        """
        state = add_agent_to_trace(state, self.name)
        
        # Get current task query
        query = self._get_current_query(state)
        
        logger.with_agent(self.name).info(
            f"Retrieving for: {query[:50]}..."
        )
        
        # Build filters based on context
        filters = self._build_filters(state)
        
        # Perform hybrid search
        documents = await self.qdrant.hybrid_search(
            query=query,
            top_k=self.top_k,
            filters=filters,
            score_threshold=self.confidence_threshold * 0.5  # RRF scores are lower
        )
        
        # Evaluate retrieval quality
        cache_hit, avg_score = self._evaluate_retrieval(documents)
        
        # Update state
        state["retrieved_documents"] = [doc.to_dict() for doc in documents]
        state["retrieval_scores"] = [doc.score for doc in documents]
        state["cache_hit"] = cache_hit
        
        logger.with_agent(self.name).info(
            f"Retrieved {len(documents)} documents, "
            f"cache_hit={cache_hit}, avg_score={avg_score:.3f}"
        )
        
        return state
    
    def _get_current_query(self, state: AgentState) -> str:
        """Get the current task's query."""
        sub_tasks = state.get("sub_tasks", [])
        current_idx = state.get("current_task_index", 0)
        
        if sub_tasks and current_idx < len(sub_tasks):
            return sub_tasks[current_idx].get("query", state["original_query"])
        
        return state.get("original_query", "")
    
    def _build_filters(self, state: AgentState) -> Optional[Filter]:
        """
        Build Qdrant filters based on context.
        
        Currently filters for:
        - Currently valid records only (valid_to is null)
        """
        # Base filter: only currently valid records
        conditions = [
            FieldCondition(
                key="valid_to",
                is_null=IsNullCondition(is_null=True)
            )
        ]
        
        # Could add more filters based on state context:
        # - Filter by fact_type if specified
        # - Filter by source domain for trusted sources
        # - Filter by date range for temporal queries
        
        return Filter(must=conditions) if conditions else None
    
    def _evaluate_retrieval(
        self,
        documents: List[RetrievedDocument]
    ) -> tuple[bool, float]:
        """
        Evaluate if retrieval results are sufficient.
        
        Returns:
            Tuple of (cache_hit, average_score)
        """
        if not documents:
            return False, 0.0
        
        avg_score = sum(d.score for d in documents) / len(documents)
        
        # Cache hit if:
        # 1. At least one document with high confidence
        # 2. Average score above threshold
        high_confidence_docs = [
            d for d in documents
            if d.score >= self.confidence_threshold
        ]
        
        cache_hit = (
            len(high_confidence_docs) >= 1 and
            avg_score >= self.confidence_threshold * 0.7
        )
        
        return cache_hit, avg_score
    
    async def search_by_verdict(
        self,
        query: str,
        verdict: str,
        top_k: int = 5
    ) -> List[RetrievedDocument]:
        """
        Search for documents with a specific verdict.
        
        Useful for finding related verified/debunked claims.
        """
        filters = Filter(
            must=[
                FieldCondition(
                    key="verdict",
                    match=MatchValue(value=verdict)
                ),
                FieldCondition(
                    key="valid_to",
                    is_null=IsNullCondition(is_null=True)
                )
            ]
        )
        
        return await self.qdrant.hybrid_search(
            query=query,
            top_k=top_k,
            filters=filters
        )
    
    async def search_similar_claims(
        self,
        claim: str,
        top_k: int = 3
    ) -> List[RetrievedDocument]:
        """
        Find claims similar to the given one.
        
        Used for deduplication and context building.
        """
        return await self.qdrant.dense_search(
            query=claim,
            top_k=top_k
        )
    
    async def search_by_source(
        self,
        query: str,
        source_domain: str,
        top_k: int = 5
    ) -> List[RetrievedDocument]:
        """
        Search for documents from a specific source.
        """
        filters = Filter(
            must=[
                FieldCondition(
                    key="source_domain",
                    match=MatchValue(value=source_domain)
                )
            ]
        )
        
        return await self.qdrant.hybrid_search(
            query=query,
            top_k=top_k,
            filters=filters
        )
    
    def is_cache_sufficient(self, state: AgentState) -> bool:
        """Check if cached retrieval is sufficient."""
        return state.get("cache_hit", False)
    
    def get_best_match(self, state: AgentState) -> Optional[RetrievedDocument]:
        """Get the highest-scoring retrieved document."""
        docs = state.get("retrieved_documents", [])
        if not docs:
            return None
        
        # Find highest score
        best_doc = max(docs, key=lambda d: d.get("score", 0))
        return RetrievedDocument(**best_doc)
