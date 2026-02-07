"""
Tests for the state management module.
"""

import pytest
from datetime import datetime

from src.core.state import (
    Intent,
    Verdict,
    FactType,
    TaskStatus,
    AgentAction,
    SubTask,
    RetrievedDocument,
    SearchResult,
    Evidence,
    TaskResult,
    FeedbackSignal,
    AgentState,
    create_initial_state,
    add_agent_to_trace,
    increment_loop_count,
    should_continue_loop,
    compile_evidence
)


class TestEnums:
    """Tests for enum types."""
    
    def test_intent_values(self):
        """Test Intent enum values."""
        assert Intent.CONVERSATIONAL.value == "conversational"
        assert Intent.INFORMATIONAL.value == "informational"
        assert Intent.CLARIFICATION.value == "clarification"
        assert Intent.OUT_OF_SCOPE.value == "out_of_scope"
    
    def test_verdict_values(self):
        """Test Verdict enum values."""
        assert Verdict.TRUE.value == "TRUE"
        assert Verdict.FALSE.value == "FALSE"
        assert Verdict.UNCERTAIN.value == "UNCERTAIN"
        assert Verdict.PENDING.value == "PENDING"
    
    def test_fact_type_values(self):
        """Test FactType enum values."""
        assert FactType.STATIC.value == "STATIC"
        assert FactType.TRANSIENT.value == "TRANSIENT"


class TestSubTask:
    """Tests for SubTask dataclass."""
    
    def test_creation(self):
        """Test SubTask creation."""
        task = SubTask(
            id="task_1",
            query="Is the Earth round?"
        )
        
        assert task.id == "task_1"
        assert task.query == "Is the Earth round?"
        assert task.status == TaskStatus.PENDING
        assert task.result is None
    
    def test_to_dict(self):
        """Test SubTask serialization."""
        task = SubTask(
            id="task_1",
            query="Test query"
        )
        
        d = task.to_dict()
        
        assert d["id"] == "task_1"
        assert d["query"] == "Test query"
        assert d["status"] == "pending"
        assert "created_at" in d


class TestRetrievedDocument:
    """Tests for RetrievedDocument dataclass."""
    
    def test_creation(self):
        """Test RetrievedDocument creation."""
        doc = RetrievedDocument(
            id="doc_1",
            text="The Earth is approximately 4.5 billion years old.",
            score=0.95,
            source_domain="nasa.gov",
            verdict=Verdict.TRUE,
            fact_type=FactType.STATIC
        )
        
        assert doc.id == "doc_1"
        assert doc.score == 0.95
        assert doc.verdict == Verdict.TRUE
    
    def test_to_dict(self):
        """Test RetrievedDocument serialization."""
        doc = RetrievedDocument(
            id="doc_1",
            text="Test text",
            score=0.8,
            verdict=Verdict.FALSE
        )
        
        d = doc.to_dict()
        
        assert d["id"] == "doc_1"
        assert d["verdict"] == "FALSE"


class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_creation(self):
        """Test SearchResult creation."""
        result = SearchResult(
            title="NASA Confirms Earth's Age",
            url="https://nasa.gov/article",
            content="Scientists have confirmed...",
            domain="nasa.gov",
            trust_tier=1
        )
        
        assert result.title == "NASA Confirms Earth's Age"
        assert result.trust_tier == 1


class TestEvidence:
    """Tests for Evidence dataclass."""
    
    def test_is_sufficient_with_docs(self):
        """Test is_sufficient with retrieved documents."""
        doc = RetrievedDocument(
            id="1",
            text="Test",
            score=0.9
        )
        
        evidence = Evidence(
            retrieved_docs=[doc],
            confidence_score=0.85
        )
        
        assert evidence.is_sufficient == True
    
    def test_is_sufficient_without_docs(self):
        """Test is_sufficient without documents."""
        evidence = Evidence(
            retrieved_docs=[],
            confidence_score=0.3
        )
        
        assert evidence.is_sufficient == False
    
    def test_best_sources(self):
        """Test best_sources property."""
        doc = RetrievedDocument(
            id="1",
            text="Test",
            score=0.9,
            source_domain="nasa.gov"
        )
        
        result = SearchResult(
            title="Test",
            url="https://bbc.com/article",
            content="Test content",
            domain="bbc.com",
            trust_tier=2
        )
        
        evidence = Evidence(
            retrieved_docs=[doc],
            search_results=[result]
        )
        
        sources = evidence.best_sources
        assert "nasa.gov" in sources
        assert "bbc.com" in sources


class TestAgentState:
    """Tests for AgentState and helper functions."""
    
    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state(
            query="Is the Earth flat?",
            session_id="test_session"
        )
        
        assert state["original_query"] == "Is the Earth flat?"
        assert state["session_id"] == "test_session"
        assert state["intent"] == ""
        assert state["verdict"] == Verdict.PENDING.value
        assert state["loop_count"] == 0
    
    def test_add_agent_to_trace(self):
        """Test adding agent to trace."""
        state = create_initial_state("Test query")
        
        updated = add_agent_to_trace(state, "Planner")
        
        assert len(updated["agent_trace"]) == 1
        assert "Planner:" in updated["agent_trace"][0]
    
    def test_increment_loop_count(self):
        """Test loop count incrementing."""
        state = create_initial_state("Test query")
        assert state["loop_count"] == 0
        
        updated = increment_loop_count(state)
        assert updated["loop_count"] == 1
        
        updated = increment_loop_count(updated)
        assert updated["loop_count"] == 2
    
    def test_should_continue_loop(self):
        """Test loop continuation check."""
        state = create_initial_state("Test query")
        state["max_loops"] = 3
        
        # Should continue at loop 0, 1, 2
        for i in range(3):
            state["loop_count"] = i
            assert should_continue_loop(state) == True
        
        # Should not continue at loop 3
        state["loop_count"] = 3
        assert should_continue_loop(state) == False
    
    def test_compile_evidence(self):
        """Test evidence compilation from state."""
        state = create_initial_state("Test query")
        state["retrieved_documents"] = [
            {
                "id": "doc_1",
                "text": "Test document",
                "score": 0.9,
                "source_domain": "nasa.gov"
            }
        ]
        state["search_results"] = [
            {
                "title": "Test Article",
                "url": "https://bbc.com/article",
                "content": "Test content",
                "domain": "bbc.com",
                "trust_tier": 2
            }
        ]
        
        evidence = compile_evidence(state)
        
        assert len(evidence.retrieved_docs) == 1
        assert len(evidence.search_results) == 1
        assert evidence.source_quality != "unknown"
