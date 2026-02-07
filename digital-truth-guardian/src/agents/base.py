"""
Base agent class for all Digital Truth Guardian agents.

Provides common functionality and interface for agent implementations.
Includes episodic memory recording and shared context access.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.state import AgentState
from ..core.config import settings
from ..utils.logger import get_logger


logger = get_logger()


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    Defines the common interface and shared functionality.
    Provides hooks for:
    - Episodic memory recording (learning from past decisions)
    - Shared context reading/writing (inter-agent communication)
    """
    
    name: str = "BaseAgent"
    
    # Memory manager (lazy loaded)
    _memory_manager = None
    
    @property
    def memory_manager(self):
        """Lazy load memory manager."""
        if self._memory_manager is None:
            from ..database.memory_manager import get_memory_manager
            self._memory_manager = get_memory_manager()
        return self._memory_manager
    
    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the current state and return updated state.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        pass
    
    def should_run(self, state: AgentState) -> bool:
        """
        Determine if this agent should run given the current state.
        
        Override in subclasses for conditional execution.
        
        Args:
            state: Current agent state
            
        Returns:
            True if agent should process state
        """
        return True
    
    def get_agent_trace_entry(self) -> str:
        """Get trace entry for this agent."""
        return f"{self.name}:{datetime.utcnow().isoformat()}"
    
    # ==================== Episodic Memory ====================
    
    async def record_episode(
        self,
        state: AgentState,
        action_type: str,
        outcome: str,
        decision_reasoning: str = "",
        confidence: float = 0.0,
        retrieval_score: Optional[float] = None,
        tools_used: Optional[List[str]] = None,
    ):
        """
        Record this agent's action to episodic memory.
        
        This enables learning from past decisions.
        """
        try:
            session_id = state.get("session_id", "unknown")
            query = state.get("original_query", "")
            loop_count = state.get("loop_count", 0)
            
            await self.memory_manager.record_episode(
                session_id=session_id,
                agent_name=self.name,
                action_type=action_type,
                query=query,
                outcome=outcome,
                decision_reasoning=decision_reasoning,
                confidence=confidence,
                retrieval_score=retrieval_score,
                loop_count=loop_count,
                tools_used=tools_used,
            )
        except Exception as e:
            # Don't fail the main flow if memory recording fails
            logger.with_agent(self.name).debug(f"Failed to record episode: {e}")
    
    async def recall_similar_experiences(
        self,
        query: str,
        outcome_filter: Optional[str] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Recall similar past experiences to inform current decision.
        
        Returns:
            List of past episodes with similar queries
        """
        try:
            episodes = await self.memory_manager.recall_similar_episodes(
                query=query,
                agent_name=self.name,
                outcome=outcome_filter,
                limit=limit,
            )
            return [ep.to_payload() for ep in episodes]
        except Exception as e:
            logger.with_agent(self.name).debug(f"Failed to recall episodes: {e}")
            return []
    
    # ==================== Shared Context ====================
    
    async def write_shared_context(
        self,
        content: str,
        context_type: str = "insight",
        session_id: Optional[str] = None,
        target_agents: Optional[List[str]] = None,
        priority: int = 1,
        ttl_minutes: Optional[int] = 30,
        tags: Optional[List[str]] = None,
    ):
        """
        Write context to shared memory for other agents.
        
        Context types:
        - task_context: Information about current task
        - insight: Discovered patterns or useful information
        - warning: Potential issues or errors to watch for
        - strategy: Suggested approach for handling query
        """
        try:
            await self.memory_manager.write_context(
                agent_source=self.name,
                context_type=context_type,
                content=content,
                session_id=session_id,
                target_agents=target_agents,
                priority=priority,
                ttl_minutes=ttl_minutes,
                tags=tags,
            )
        except Exception as e:
            logger.with_agent(self.name).debug(f"Failed to write context: {e}")
    
    async def read_shared_context(
        self,
        session_id: Optional[str] = None,
        context_type: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Read shared context from other agents.
        
        Returns:
            List of context items sorted by priority
        """
        try:
            contexts = await self.memory_manager.read_context(
                agent_name=self.name,
                session_id=session_id,
                context_type=context_type,
                query=query,
                limit=limit,
            )
            return [ctx.to_payload() for ctx in contexts]
        except Exception as e:
            logger.with_agent(self.name).debug(f"Failed to read context: {e}")
            return []
    
    # ==================== Helper Methods ====================
    
    def _get_current_query(self, state: AgentState) -> str:
        """Get the current task's query from state."""
        sub_tasks = state.get("sub_tasks", [])
        current_idx = state.get("current_task_index", 0)
        
        if sub_tasks and current_idx < len(sub_tasks):
            return sub_tasks[current_idx].get("query", state.get("original_query", ""))
        
        return state.get("original_query", "")
    
    def _summarize_state(self, state: AgentState) -> str:
        """Create a brief summary of the current state for logging."""
        return (
            f"query='{state.get('original_query', '')[:50]}...', "
            f"loop={state.get('loop_count', 0)}, "
            f"intent={state.get('intent', 'unknown')}"
        )
