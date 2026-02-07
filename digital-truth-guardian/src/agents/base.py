"""
Base agent class for all Digital Truth Guardian agents.

Provides common functionality and interface for agent implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..core.state import AgentState


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    Defines the common interface and shared functionality.
    """
    
    name: str = "BaseAgent"
    
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
        from datetime import datetime
        return f"{self.name}:{datetime.utcnow().isoformat()}"
