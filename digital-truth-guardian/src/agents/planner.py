"""
Planner-Router Agent (The Brain)

Responsible for:
- Intent classification (conversational vs informational)
- Task decomposition for complex queries
- Routing decisions based on feedback
- Feedback loop handling
"""

import json
from typing import List, Optional

from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage, AIMessage

from .base import BaseAgent
from ..core.config import settings, ModelConfig
from ..core.state import (
    AgentState,
    Intent,
    SubTask,
    TaskStatus,
    AgentAction,
    FeedbackSignal,
    add_agent_to_trace,
    increment_loop_count,
    should_continue_loop
)
from ..utils.logger import get_logger
from ..utils.helpers import parse_multi_query


logger = get_logger()


# ==================== Prompts ====================

INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a fact-checking system. Analyze the user's query and classify it.

User Query: {query}

Classify the intent as ONE of:
- CONVERSATIONAL: Casual greetings, small talk, or non-factual queries (e.g., "Hi", "How are you?", "Thanks")
- INFORMATIONAL: Questions about facts, claims, events, or statements that can be verified (e.g., "Did X happen?", "Is Y true?", "Who is Z?")
- CLARIFICATION: Follow-up questions asking for more details about a previous response
- OUT_OF_SCOPE: Requests that cannot be handled (e.g., generating harmful content, unrelated tasks)

Also determine if this is a multi-part query that should be decomposed.

Respond in JSON format:
{{
    "intent": "<CONVERSATIONAL|INFORMATIONAL|CLARIFICATION|OUT_OF_SCOPE>",
    "is_multi_part": <true|false>,
    "sub_queries": ["query1", "query2"] or null if not multi-part,
    "reasoning": "<brief explanation>"
}}
"""

TASK_DECOMPOSITION_PROMPT = """You are a task decomposition agent. Break down this complex query into independent, verifiable sub-tasks.

Original Query: {query}

Rules:
1. Each sub-task should be independently verifiable
2. Maintain the original intent of each part
3. Use clear, searchable language
4. Maximum 5 sub-tasks

Respond in JSON format:
{{
    "sub_tasks": [
        {{"id": "task_1", "query": "reformulated query 1"}},
        {{"id": "task_2", "query": "reformulated query 2"}}
    ],
    "decomposition_reasoning": "<explanation>"
}}
"""

ROUTING_DECISION_PROMPT = """You are a routing agent for a fact-checking system. Based on the current state, decide the next action.

Current State:
- Query: {query}
- Retrieval Results: {retrieval_count} documents found
- Search Triggered: {search_triggered}
- Loop Count: {loop_count}/{max_loops}
- Feedback: {feedback}

Available Actions:
- RETRIEVE: Search internal knowledge base
- SEARCH: Trigger external web search (use if retrieval insufficient)
- CRITIQUE: Send evidence to Critic for verdict
- RESPOND: Generate final response (only if we have a verdict)
- LOOP_BACK: Try different retrieval strategy

Rules:
1. If no retrieval done yet, action should be RETRIEVE
2. If retrieval gave low scores and search not done, action should be SEARCH
3. If evidence available, action should be CRITIQUE
4. If max loops reached, must proceed to CRITIQUE anyway

Respond in JSON format:
{{
    "action": "<RETRIEVE|SEARCH|CRITIQUE|RESPOND|LOOP_BACK>",
    "reasoning": "<brief explanation>"
}}
"""

TOOL_SELECTION_PROMPT = """You are a tool selection agent. Analyze the query and determine the best tool strategy.

Query: {query}
Query Type Analysis: {query_analysis}

Available Tools:
1. QDRANT_SEARCH - Internal knowledge base with verified facts
   - Best for: Historical facts, previously verified claims, scientific facts
   - Fast, high confidence when matched

2. TAVILY_WEB_SEARCH - External web search with trusted source filtering
   - Best for: Current events, recent news, real-time information
   - Slower but access to fresh data

3. BOTH_SEQUENTIAL - Qdrant first, then Tavily if insufficient
   - Best for: General fact-checking where cached results may exist

4. BOTH_PARALLEL - Query both simultaneously
   - Best for: Time-sensitive queries needing comprehensive evidence

Past Experiences with Similar Queries:
{past_experiences}

Based on the query type and past experiences, select the optimal tool strategy.

Respond in JSON format:
{{
    "strategy": "<QDRANT_SEARCH|TAVILY_WEB_SEARCH|BOTH_SEQUENTIAL|BOTH_PARALLEL>",
    "reasoning": "<explanation based on query type and past experiences>",
    "confidence": <0.0-1.0>
}}
"""


class PlannerAgent(BaseAgent):
    """
    The Planner-Router Agent (The Brain).
    
    Uses Gemini Flash for low-latency routing decisions.
    Implements:
    - Intent classification
    - Task decomposition
    - Feedback-driven routing
    - Dynamic tool selection based on query type and past experiences
    - Episodic memory for learning from past decisions
    """
    
    name = "Planner"
    
    def __init__(self):
        """Initialize the Planner agent."""
        self.model_name = ModelConfig.PLANNER["model"]
        self.temperature = ModelConfig.PLANNER["temperature"]
        self.max_tokens = ModelConfig.PLANNER["max_tokens"]
        
        # Configure Gemini client
        if settings.gemini_api_key:
            self.client = genai.Client(api_key=settings.gemini_api_key)
        else:
            self.client = None
            logger.with_agent(self.name).warning(
                "Gemini API key not configured"
            )
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process state and make routing decisions.
        
        Flow:
        1. First call: Classify intent, select tools
        2. Subsequent calls (feedback loops): Increment loop count and reprocess
        """
        state = add_agent_to_trace(state, self.name)
        
        # Read any shared context from other agents
        shared_context = await self.read_shared_context(
            session_id=state.get("session_id"),
            query=state.get("original_query"),
            limit=3
        )
        if shared_context:
            state["_shared_context"] = shared_context
        
        # Check if this is first processing or a feedback loop
        if not state.get("intent"):
            # First time: classify intent and select tools
            state = await self._classify_and_decompose(state)
            state = await self._select_tools(state)
            logger.with_agent(self.name).info("Initial classification complete")
            return state
        else:
            # Feedback loop from Critic - increment loop count
            state = increment_loop_count(state)
            loop_count = state.get("loop_count", 0)
            max_loops = state.get("max_loops", 3)
            logger.with_agent(self.name).info(
                f"Feedback loop iteration {loop_count}/{max_loops}"
            )
            
            # Reset flags for new search attempt
            state["_retrieval_done"] = False
            state["_search_done"] = False
            
            return state
    
    async def _classify_and_decompose(self, state: AgentState) -> AgentState:
        """Classify intent and decompose multi-part queries."""
        query = state["original_query"]
        
        logger.with_agent(self.name).info(
            f"Classifying intent for: {query[:50]}..."
        )
        
        # Classify intent
        intent_result = await self._call_llm(
            INTENT_CLASSIFICATION_PROMPT.format(query=query)
        )
        
        try:
            parsed = json.loads(intent_result)
            # Convert to lowercase to match enum values
            intent_str = parsed["intent"].lower()
            intent = Intent(intent_str)
            is_multi_part = parsed.get("is_multi_part", False)
            sub_queries = parsed.get("sub_queries", [])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.with_agent(self.name).warning(
                f"Failed to parse intent response: {e}"
            )
            intent = Intent.INFORMATIONAL
            is_multi_part = False
            sub_queries = []
        
        # Build sub-tasks
        sub_tasks = []
        if is_multi_part and sub_queries:
            for i, sq in enumerate(sub_queries):
                sub_tasks.append(SubTask(
                    id=f"task_{i+1}",
                    query=sq,
                    status=TaskStatus.PENDING
                ).to_dict())
        elif intent == Intent.INFORMATIONAL:
            # Single task
            sub_tasks.append(SubTask(
                id="task_1",
                query=query,
                status=TaskStatus.PENDING
            ).to_dict())
        
        # Update state
        state["intent"] = intent.value
        state["is_multi_part"] = is_multi_part
        state["sub_tasks"] = sub_tasks
        state["current_task_index"] = 0
        
        # Add message to history
        state["messages"] = state.get("messages", []) + [
            HumanMessage(content=query)
        ]
        
        logger.with_agent(self.name).info(
            f"Intent: {intent.value}, Multi-part: {is_multi_part}, "
            f"Tasks: {len(sub_tasks)}"
        )
        
        return state
    
    async def _route(self, state: AgentState) -> AgentState:
        """Make routing decision based on current state."""
        query = state.get("original_query", "")
        retrieval_count = len(state.get("retrieved_documents", []))
        search_triggered = state.get("search_triggered", False)
        loop_count = state.get("loop_count", 0)
        max_loops = state.get("max_loops", 3)
        feedback = state.get("feedback", {})
        
        logger.with_agent(self.name).info(
            f"Routing (loop {loop_count}/{max_loops}): "
            f"docs={retrieval_count}, searched={search_triggered}"
        )
        
        # Get LLM routing decision
        routing_result = await self._call_llm(
            ROUTING_DECISION_PROMPT.format(
                query=query,
                retrieval_count=retrieval_count,
                search_triggered=search_triggered,
                loop_count=loop_count,
                max_loops=max_loops,
                feedback=json.dumps(feedback) if feedback else "None"
            )
        )
        
        try:
            parsed = json.loads(routing_result)
            action = AgentAction(parsed["action"].lower())
            reasoning = parsed.get("reasoning", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            # Default routing logic when LLM fails
            reasoning = "fallback logic"
            if not search_triggered:
                # Haven't searched yet, trigger external search
                action = AgentAction.SEARCH
            elif retrieval_count == 0:
                action = AgentAction.SEARCH
            elif retrieval_count < 2:
                action = AgentAction.SEARCH
            else:
                action = AgentAction.CRITIQUE
        
        # Store routing decision in state
        state["_next_action"] = action.value
        
        # Record this routing decision to episodic memory
        await self.record_episode(
            state=state,
            action_type="route",
            outcome="pending",  # Will be updated based on result
            decision_reasoning=reasoning,
            confidence=0.8,
        )
        
        # Share routing context with other agents
        await self.write_shared_context(
            content=f"Routing to {action.value}: {reasoning}",
            context_type="task_context",
            session_id=state.get("session_id"),
            priority=2,
            ttl_minutes=10,
        )
        
        logger.with_agent(self.name).info(f"Routing decision: {action.value}")
        
        return state
    
    async def _select_tools(self, state: AgentState) -> AgentState:
        """
        Dynamically select tools based on query analysis and past experiences.
        
        This is a key differentiator - tools are chosen based on:
        1. Query characteristics (current events vs historical facts)
        2. Past experiences with similar queries
        """
        query = state.get("original_query", "")
        intent = state.get("intent", "")
        
        # Skip tool selection for non-informational queries
        if intent not in [Intent.INFORMATIONAL.value]:
            state["tool_strategy"] = "NONE"
            return state
        
        # Recall past experiences with similar queries
        past_episodes = await self.recall_similar_experiences(
            query=query,
            outcome_filter="success",  # Learn from successful outcomes
            limit=3
        )
        
        # Analyze query type
        query_analysis = await self._analyze_query_type(query)
        
        # Format past experiences for prompt
        past_exp_text = "No relevant past experiences."
        if past_episodes:
            past_exp_text = "\n".join([
                f"- Query: '{ep.get('query', '')[:50]}...' | "
                f"Action: {ep.get('action_type')} | "
                f"Outcome: {ep.get('outcome')} | "
                f"Tools: {ep.get('tools_used', [])}"
                for ep in past_episodes
            ])
        
        # Get tool selection from LLM
        tool_result = await self._call_llm(
            TOOL_SELECTION_PROMPT.format(
                query=query,
                query_analysis=query_analysis,
                past_experiences=past_exp_text
            )
        )
        
        try:
            parsed = json.loads(tool_result)
            strategy = parsed.get("strategy", "BOTH_SEQUENTIAL")
            reasoning = parsed.get("reasoning", "")
        except (json.JSONDecodeError, KeyError):
            strategy = "BOTH_SEQUENTIAL"
            reasoning = "default strategy"
        
        state["tool_strategy"] = strategy
        state["tool_reasoning"] = reasoning
        
        logger.with_agent(self.name).info(
            f"Tool strategy: {strategy} ({reasoning[:50]}...)"
        )
        
        return state
    
    async def _analyze_query_type(self, query: str) -> str:
        """Analyze query to determine its type for tool selection."""
        query_lower = query.lower()
        
        # Simple heuristics for query type
        indicators = []
        
        # Current events indicators
        if any(word in query_lower for word in ["today", "yesterday", "recent", "latest", "current", "now", "2026", "2025"]):
            indicators.append("CURRENT_EVENTS")
        
        # Historical facts indicators
        if any(word in query_lower for word in ["was", "were", "did", "history", "historical", "in 1", "in 2", "century"]):
            indicators.append("HISTORICAL")
        
        # Scientific facts
        if any(word in query_lower for word in ["scientific", "study", "research", "data", "statistics"]):
            indicators.append("SCIENTIFIC")
        
        # Political/news
        if any(word in query_lower for word in ["president", "election", "government", "policy", "politician"]):
            indicators.append("POLITICAL")
        
        if not indicators:
            indicators.append("GENERAL")
        
        return ", ".join(indicators)
    
    async def _call_llm(self, prompt: str) -> str:
        """Call Gemini model with prompt."""
        if not self.client:
            return "{}"
        
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            text = response.text
            # Clean up markdown code blocks if present
            if text.startswith("```"):
                lines = text.split("\n")
                # Remove first line (```json) and last line (```)
                if lines[-1].strip() == "```":
                    text = "\n".join(lines[1:-1])
                else:
                    text = "\n".join(lines[1:])
            return text.strip()
        except Exception as e:
            logger.with_agent(self.name).error(f"LLM call failed: {e}")
            return "{}"
    
    def get_next_action(self, state: AgentState) -> AgentAction:
        """Get the next action from state."""
        action_str = state.get("_next_action", "retrieve")
        try:
            return AgentAction(action_str)
        except ValueError:
            return AgentAction.RETRIEVE
    
    def handle_insufficient_feedback(self, state: AgentState) -> AgentState:
        """Handle INSUFFICIENT feedback from Critic."""
        if not should_continue_loop(state):
            logger.with_agent(self.name).warning(
                "Max loops reached, forcing critique"
            )
            state["_next_action"] = AgentAction.CRITIQUE.value
            return state
        
        state = increment_loop_count(state)
        
        # If not searched yet, trigger search
        if not state.get("search_triggered"):
            logger.with_agent(self.name).info(
                "Escalating to external search"
            )
            state["_next_action"] = AgentAction.SEARCH.value
        else:
            # Already searched, proceed to critique anyway
            state["_next_action"] = AgentAction.CRITIQUE.value
        
        return state
    
    def should_skip_retrieval(self, state: AgentState) -> bool:
        """Check if retrieval should be skipped (conversational intent)."""
        intent = state.get("intent")
        return intent in [Intent.CONVERSATIONAL.value, Intent.OUT_OF_SCOPE.value]
    
    async def generate_conversational_response(
        self,
        state: AgentState
    ) -> str:
        """Generate response for conversational queries."""
        query = state.get("original_query", "")
        
        prompt = f"""You are a helpful fact-checking assistant. The user sent a conversational message.
        
User: {query}

Respond naturally and briefly. If they seem to want fact-checking, guide them to ask a specific question.
"""
        
        response = await self._call_llm(prompt)
        return response.strip()
