"""
LangGraph Orchestration for Digital Truth Guardian.

Implements the state graph with feedback loops for the
5-Agent Orchestration Squad (Planner-Led architecture).
"""

from datetime import datetime
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from ..agents.planner import PlannerAgent
from ..agents.retriever import RetrieverAgent
from ..agents.executor import ExecutorAgent
from ..agents.critic import CriticAgent
from ..agents.archivist import ArchivistAgent
from ..core.state import (
    AgentState,
    Intent,
    Verdict,
    AgentAction,
    create_initial_state,
    should_continue_loop
)
from ..core.config import settings
from ..utils.logger import get_logger


logger = get_logger()


class TruthGuardianGraph:
    """
    Main orchestration graph for Digital Truth Guardian.
    
    Implements a "Hub-and-Spoke" pattern where the Planner (hub)
    coordinates specialized worker agents (spokes).
    
    Graph Flow:
    1. Planner classifies intent and decomposes tasks
    2. Router decides: Retrieve → Search → Critique loop
    3. Critic evaluates evidence and provides feedback
    4. Archivist handles memory management
    5. Response generation
    """
    
    def __init__(self):
        """Initialize the orchestration graph."""
        # Initialize agents
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.executor = ExecutorAgent()
        self.critic = CriticAgent()
        self.archivist = ArchivistAgent()
        
        # Build graph
        self.graph = self._build_graph()
        
        # Compile with memory saver for persistence
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)
        
        logger.with_agent("System").info(
            "Truth Guardian Graph initialized"
        )
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        # Create state graph
        graph = StateGraph(AgentState)
        
        # ==================== Add Nodes ====================
        
        # Planner node (intent classification and routing)
        graph.add_node("planner", self._planner_node)
        
        # Retriever node (Qdrant search)
        graph.add_node("retriever", self._retriever_node)
        
        # Executor node (external search)
        graph.add_node("executor", self._executor_node)
        
        # Critic node (verdict determination)
        graph.add_node("critic", self._critic_node)
        
        # Archivist node (memory management)
        graph.add_node("archivist", self._archivist_node)
        
        # Response generation node
        graph.add_node("respond", self._respond_node)
        
        # ==================== Add Edges ====================
        
        # Entry point
        graph.set_entry_point("planner")
        
        # Planner routes based on intent
        graph.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {
                "retrieve": "retriever",
                "respond": "respond",
                "end": END
            }
        )
        
        # Retriever always goes to executor (web search)
        graph.add_edge("retriever", "executor")
        
        # Executor always goes to critic for evaluation
        graph.add_edge("executor", "critic")
        
        # Critic routes based on verdict/feedback (only point for loops)
        graph.add_conditional_edges(
            "critic",
            self._route_from_critic,
            {
                "respond": "respond",  # Go to respond first (user sees result faster)
                "planner": "planner",  # Feedback loop (max 3)
            }
        )
        
        # Response routes to archivist for memorization or ends
        graph.add_conditional_edges(
            "respond",
            self._route_from_respond,
            {
                "archivist": "archivist",
                "end": END
            }
        )
        
        # Archivist ends the graph
        graph.add_edge("archivist", END)
        
        return graph
    
    # ==================== Node Functions ====================
    
    async def _planner_node(self, state: AgentState) -> AgentState:
        """Planner node: classify intent and make routing decisions."""
        logger.with_agent("Graph").info("Executing Planner node")
        return await self.planner.process(state)
    
    async def _retriever_node(self, state: AgentState) -> AgentState:
        """Retriever node: search Qdrant knowledge base."""
        logger.with_agent("Graph").info("Executing Retriever node")
        return await self.retriever.process(state)
    
    async def _executor_node(self, state: AgentState) -> AgentState:
        """Executor node: external web search."""
        logger.with_agent("Graph").info("Executing Executor node")
        return await self.executor.process(state)
    
    async def _critic_node(self, state: AgentState) -> AgentState:
        """Critic node: analyze evidence and determine verdict."""
        logger.with_agent("Graph").info("Executing Critic node")
        return await self.critic.process(state)
    
    async def _archivist_node(self, state: AgentState) -> AgentState:
        """Archivist node: manage memory and persistence."""
        logger.with_agent("Graph").info("Executing Archivist node")
        return await self.archivist.process(state)
    
    async def _respond_node(self, state: AgentState) -> AgentState:
        """Response node: generate final response for user."""
        logger.with_agent("Graph").info("Generating response")
        
        intent = state.get("intent", "")
        verdict = state.get("verdict", "PENDING")
        
        # Handle conversational intent
        if intent == Intent.CONVERSATIONAL.value:
            response = await self.planner.generate_conversational_response(state)
            state["final_response"] = response
            state["processing_completed"] = datetime.utcnow().isoformat()
            return state
        
        # Handle out of scope
        if intent == Intent.OUT_OF_SCOPE.value:
            state["final_response"] = (
                "I'm sorry, but I cannot help with that request. "
                "I'm designed to verify factual claims and combat misinformation. "
                "Please ask me about a claim you'd like me to verify."
            )
            state["processing_completed"] = datetime.utcnow().isoformat()
            return state
        
        # Generate fact-check response
        response = self._generate_verdict_response(state)
        state["final_response"] = response
        
        # Add AI message to history
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=response)
        ]
        
        state["processing_completed"] = datetime.utcnow().isoformat()
        
        return state
    
    # ==================== Routing Functions ====================
    
    def _route_from_planner(
        self,
        state: AgentState
    ) -> Literal["retrieve", "respond", "end"]:
        """Route from Planner based on intent.
        
        Sequential flow: Planner -> Retriever -> Executor -> Critic
        Loops only happen from Critic back to Planner.
        """
        intent = state.get("intent", "")
        
        # Handle non-informational intents
        if intent == Intent.CONVERSATIONAL.value:
            logger.with_agent("Graph").info("Routing: conversational -> respond")
            return "respond"
        
        if intent == Intent.OUT_OF_SCOPE.value:
            logger.with_agent("Graph").info("Routing: out_of_scope -> respond")
            return "respond"
        
        # For informational queries, always start with retrieval
        logger.with_agent("Graph").info("Routing: informational -> retrieve")
        return "retrieve"
    
    def _route_from_critic(
        self,
        state: AgentState
    ) -> Literal["archivist", "planner", "respond"]:
        """Route from Critic based on verdict and feedback.
        
        This is the ONLY point where loops can occur.
        Max loops is enforced here.
        """
        feedback = state.get("feedback", {})
        verdict = state.get("verdict", "PENDING")
        loop_count = state.get("loop_count", 0)
        max_loops = state.get("max_loops", 3)
        
        logger.with_agent("Graph").info(
            f"Critic routing: verdict={verdict}, loop={loop_count}/{max_loops}, "
            f"sufficient={feedback.get('is_sufficient', True)}"
        )
        
        # If evidence insufficient, trigger feedback loop (max 3)
        if not feedback.get("is_sufficient", True):
            if loop_count < max_loops:
                logger.with_agent("Graph").info(
                    f"Critic feedback loop {loop_count + 1}/{max_loops}: evidence insufficient"
                )
                return "planner"
            else:
                # Max loops reached, proceed anyway
                logger.with_agent("Graph").warning(
                    f"Max loops ({max_loops}) reached, proceeding to response"
                )
                return "respond"
        
        # Always go to respond first (user sees result faster)
        # Archivist runs after response
        logger.with_agent("Graph").info(f"Verdict {verdict} -> respond")
        return "respond"
    
    def _route_from_respond(
        self,
        state: AgentState
    ) -> Literal["archivist", "end"]:
        """Route from Respond to optionally archive the result.
        
        Only archive if verdict is conclusive (TRUE or FALSE).
        """
        verdict = state.get("verdict", "PENDING")
        intent = state.get("intent", "")
        
        # Skip archiving for non-informational intents
        if intent in [Intent.CONVERSATIONAL.value, Intent.OUT_OF_SCOPE.value]:
            logger.with_agent("Graph").info("Non-informational intent -> end")
            return "end"
        
        # Archive conclusive verdicts
        if verdict in [Verdict.TRUE.value, Verdict.FALSE.value]:
            logger.with_agent("Graph").info(f"Verdict {verdict} -> archivist (background)")
            return "archivist"
        
        # Uncertain/pending verdicts, skip archiving
        logger.with_agent("Graph").info("Uncertain verdict -> end (skip archiving)")
        return "end"
    
    # ==================== Response Generation ====================
    
    def _generate_verdict_response(self, state: AgentState) -> str:
        """Generate the final verdict response."""
        verdict = state.get("verdict", "PENDING")
        confidence = state.get("confidence", 0.0)
        explanation = state.get("explanation", "")
        query = state.get("original_query", "")
        
        # Get sources
        sources = self._compile_sources(state)
        
        # Build response based on verdict
        if verdict == Verdict.TRUE.value:
            verdict_emoji = "✅"
            verdict_text = "VERIFIED AS TRUE"
        elif verdict == Verdict.FALSE.value:
            verdict_emoji = "❌"
            verdict_text = "VERIFIED AS FALSE"
        else:
            verdict_emoji = "⚠️"
            verdict_text = "UNCERTAIN"
        
        # Confidence level
        if confidence >= 0.9:
            confidence_text = "Very High"
        elif confidence >= 0.75:
            confidence_text = "High"
        elif confidence >= 0.5:
            confidence_text = "Medium"
        else:
            confidence_text = "Low"
        
        # Build response
        response_parts = [
            f"## {verdict_emoji} Verdict: {verdict_text}",
            f"**Confidence:** {confidence_text} ({int(confidence * 100)}%)",
            "",
            f"**Claim:** {query}",
            "",
            "### Analysis",
            explanation,
        ]
        
        # Add sources
        if sources:
            response_parts.extend([
                "",
                "### Sources",
            ])
            for i, source in enumerate(sources[:5], 1):
                response_parts.append(f"{i}. {source}")
        
        # Add memory status
        if state.get("memory_written"):
            response_parts.extend([
                "",
                "*This verification has been saved to our knowledge base.*"
            ])
        
        # Add disclaimer for uncertain verdicts
        if verdict == Verdict.UNCERTAIN.value:
            response_parts.extend([
                "",
                "---",
                "⚠️ **Note:** I could not conclusively verify this claim. "
                "My search did not return sufficient evidence from trusted sources. "
                "This may be an unverified rumor or a claim requiring further investigation."
            ])
        
        return "\n".join(response_parts)
    
    def _compile_sources(self, state: AgentState) -> list:
        """Compile source citations from evidence."""
        sources = []
        
        # From retrieved documents
        for doc in state.get("retrieved_documents", []):
            domain = doc.get("source_domain")
            if domain:
                sources.append(f"Knowledge Base: {domain}")
        
        # From search results
        for result in state.get("search_results", []):
            domain = result.get("domain", "")
            title = result.get("title", "")
            url = result.get("url", "")
            if domain:
                sources.append(f"{domain}: {title[:50]}..." if title else domain)
        
        return list(set(sources))  # Deduplicate
    
    # ==================== Public Interface ====================
    
    async def verify(
        self,
        query: str,
        session_id: str = None
    ) -> dict:
        """
        Verify a claim through the full agent pipeline.
        
        Args:
            query: The claim or question to verify
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Dictionary with verification results
        """
        # Create initial state
        initial_state = create_initial_state(query, session_id)
        
        # Configure thread for memory persistence
        config = {"configurable": {"thread_id": session_id or "default"}}
        
        logger.with_agent("System").info(
            f"Starting verification: {query[:50]}..."
        )
        
        # Run the graph
        final_state = await self.app.ainvoke(initial_state, config)
        
        logger.with_agent("System").info(
            f"Verification complete: {final_state.get('verdict', 'N/A')}"
        )
        
        # Return structured result
        return {
            "query": query,
            "verdict": final_state.get("verdict"),
            "confidence": final_state.get("confidence"),
            "explanation": final_state.get("explanation"),
            "response": final_state.get("final_response"),
            "sources": self._compile_sources(final_state),
            "memory_written": final_state.get("memory_written", False),
            "processing_time": self._calculate_processing_time(final_state),
            "agent_trace": final_state.get("agent_trace", [])
        }
    
    def _calculate_processing_time(self, state: AgentState) -> float:
        """Calculate total processing time in seconds."""
        started = state.get("processing_started")
        completed = state.get("processing_completed")
        
        if not started or not completed:
            return 0.0
        
        try:
            start_dt = datetime.fromisoformat(started)
            end_dt = datetime.fromisoformat(completed)
            return (end_dt - start_dt).total_seconds()
        except:
            return 0.0
    
    async def get_graph_visualization(self) -> str:
        """Get Mermaid diagram of the graph."""
        return self.app.get_graph().draw_mermaid()


# ==================== Factory Function ====================

_graph_instance = None

def get_truth_guardian_graph() -> TruthGuardianGraph:
    """Get or create the singleton graph instance."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = TruthGuardianGraph()
    return _graph_instance
