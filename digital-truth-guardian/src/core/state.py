"""
State definitions for the LangGraph orchestration.

Defines the shared state that flows between agents and tracks
the verification workflow progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from uuid import UUID, uuid4

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


# ==================== Enums ====================

class Intent(str, Enum):
    """User query intent classification."""
    CONVERSATIONAL = "conversational"  # Casual chat, no retrieval needed
    INFORMATIONAL = "informational"    # Fact-checking query
    CLARIFICATION = "clarification"    # Follow-up question
    OUT_OF_SCOPE = "out_of_scope"      # Cannot handle


class Verdict(str, Enum):
    """Fact-checking verdict from the Critic."""
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNCERTAIN = "UNCERTAIN"
    PENDING = "PENDING"


class FactType(str, Enum):
    """Type of fact for memory categorization."""
    STATIC = "STATIC"      # Immutable historical facts
    TRANSIENT = "TRANSIENT"  # Facts that can change over time


class TaskStatus(str, Enum):
    """Status of a decomposed task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentAction(str, Enum):
    """Actions agents can request."""
    RETRIEVE = "retrieve"
    SEARCH = "search"
    CRITIQUE = "critique"
    ARCHIVE = "archive"
    RESPOND = "respond"
    LOOP_BACK = "loop_back"
    END = "end"


# ==================== Data Classes ====================

@dataclass
class SubTask:
    """A decomposed sub-task from a complex query."""
    id: str
    query: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[TaskResult] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class RetrievedDocument:
    """A document retrieved from Qdrant."""
    id: str
    text: str
    score: float
    source_domain: Optional[str] = None
    verdict: Optional[Verdict] = None
    fact_type: Optional[FactType] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "source_domain": self.source_domain,
            "verdict": self.verdict.value if self.verdict else None,
            "fact_type": self.fact_type.value if self.fact_type else None,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "metadata": self.metadata
        }


@dataclass
class SearchResult:
    """A result from external web search."""
    title: str
    url: str
    content: str
    domain: str
    trust_tier: int  # 1 = highest trust, 5 = lowest
    published_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "domain": self.domain,
            "trust_tier": self.trust_tier,
            "published_date": self.published_date
        }


@dataclass
class Evidence:
    """Compiled evidence from retrieval and search."""
    retrieved_docs: List[RetrievedDocument] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    confidence_score: float = 0.0
    source_quality: str = "unknown"  # high, medium, low, unknown
    
    @property
    def is_sufficient(self) -> bool:
        """Check if evidence is sufficient for verdict."""
        return (
            self.confidence_score >= 0.7 and
            (len(self.retrieved_docs) > 0 or len(self.search_results) > 0)
        )
    
    @property
    def best_sources(self) -> List[str]:
        """Get the best source domains."""
        sources = []
        for doc in self.retrieved_docs:
            if doc.source_domain:
                sources.append(doc.source_domain)
        for result in self.search_results:
            sources.append(result.domain)
        return list(set(sources))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieved_docs": [d.to_dict() for d in self.retrieved_docs],
            "search_results": [r.to_dict() for r in self.search_results],
            "confidence_score": self.confidence_score,
            "source_quality": self.source_quality,
            "is_sufficient": self.is_sufficient
        }


@dataclass
class TaskResult:
    """Result of processing a task or sub-task."""
    task_id: str
    query: str
    verdict: Verdict
    explanation: str
    evidence: Evidence
    confidence: float
    should_memorize: bool = False
    fact_type: Optional[FactType] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "verdict": self.verdict.value,
            "explanation": self.explanation,
            "evidence": self.evidence.to_dict(),
            "confidence": self.confidence,
            "should_memorize": self.should_memorize,
            "fact_type": self.fact_type.value if self.fact_type else None
        }


@dataclass
class FeedbackSignal:
    """Feedback from Critic to Planner for loop control."""
    is_sufficient: bool
    reason: str
    suggested_action: AgentAction
    missing_info: Optional[str] = None
    loop_count: int = 0


# ==================== LangGraph State ====================

class AgentState(TypedDict, total=False):
    """
    Shared state for the LangGraph state machine.
    
    This state flows between all agents and maintains the
    complete context of the verification workflow.
    """
    # ===== Conversation Context =====
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    
    # ===== Query Analysis =====
    original_query: str
    intent: str  # Intent enum value
    is_multi_part: bool
    sub_tasks: List[Dict[str, Any]]  # Serialized SubTask objects
    current_task_index: int
    
    # ===== Retrieval State =====
    retrieved_documents: List[Dict[str, Any]]  # Serialized RetrievedDocument
    retrieval_scores: List[float]
    cache_hit: bool
    
    # ===== Search State =====
    search_triggered: bool
    search_results: List[Dict[str, Any]]  # Serialized SearchResult
    trusted_results_count: int
    
    # ===== Evidence & Verdict =====
    evidence: Dict[str, Any]  # Serialized Evidence
    verdict: str  # Verdict enum value
    explanation: str
    confidence: float
    
    # ===== Feedback Loop =====
    feedback: Dict[str, Any]  # Serialized FeedbackSignal
    loop_count: int
    max_loops: int
    
    # ===== Memory Management =====
    should_memorize: bool
    fact_type: str  # FactType enum value
    memory_written: bool
    memory_write_reason: str
    
    # ===== Response =====
    final_response: str
    sources_cited: List[str]
    
    # ===== Metadata =====
    processing_started: str
    processing_completed: str
    agent_trace: List[str]  # Track which agents processed
    error: Optional[str]


def create_initial_state(query: str, session_id: Optional[str] = None) -> AgentState:
    """
    Create initial state for a new verification request.
    
    Args:
        query: User's input query
        session_id: Optional session identifier
        
    Returns:
        AgentState: Initialized state dictionary
    """
    return AgentState(
        messages=[],
        session_id=session_id or str(uuid4()),
        original_query=query,
        intent="",
        is_multi_part=False,
        sub_tasks=[],
        current_task_index=0,
        retrieved_documents=[],
        retrieval_scores=[],
        cache_hit=False,
        search_triggered=False,
        search_results=[],
        trusted_results_count=0,
        evidence={},
        verdict=Verdict.PENDING.value,
        explanation="",
        confidence=0.0,
        feedback={},
        loop_count=0,
        max_loops=3,
        should_memorize=False,
        fact_type="",
        memory_written=False,
        memory_write_reason="",
        final_response="",
        sources_cited=[],
        processing_started=datetime.utcnow().isoformat(),
        processing_completed="",
        agent_trace=[],
        error=None
    )


# ==================== State Helper Functions ====================

def add_agent_to_trace(state: AgentState, agent_name: str) -> AgentState:
    """Add an agent to the processing trace."""
    trace = list(state.get("agent_trace", []))
    trace.append(f"{agent_name}:{datetime.utcnow().isoformat()}")
    return {**state, "agent_trace": trace}


def increment_loop_count(state: AgentState) -> AgentState:
    """Increment the feedback loop counter."""
    return {**state, "loop_count": state.get("loop_count", 0) + 1}


def should_continue_loop(state: AgentState) -> bool:
    """Check if we should continue the feedback loop."""
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 3)
    return loop_count < max_loops


def compile_evidence(state: AgentState) -> Evidence:
    """Compile evidence from state into Evidence object."""
    retrieved_docs = [
        RetrievedDocument(**doc) 
        for doc in state.get("retrieved_documents", [])
    ]
    search_results = [
        SearchResult(**result)
        for result in state.get("search_results", [])
    ]
    
    # Calculate confidence based on available evidence
    confidence = 0.0
    if retrieved_docs:
        avg_score = sum(d.score for d in retrieved_docs) / len(retrieved_docs)
        confidence = max(confidence, avg_score)
    if search_results:
        trusted_count = sum(1 for r in search_results if r.trust_tier <= 2)
        if trusted_count > 0:
            confidence = max(confidence, 0.8)
    
    # Determine source quality
    source_quality = "unknown"
    if search_results:
        min_tier = min(r.trust_tier for r in search_results) if search_results else 5
        if min_tier == 1:
            source_quality = "high"
        elif min_tier <= 2:
            source_quality = "medium"
        else:
            source_quality = "low"
    
    return Evidence(
        retrieved_docs=retrieved_docs,
        search_results=search_results,
        confidence_score=confidence,
        source_quality=source_quality
    )
