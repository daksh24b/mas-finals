"""
Executor Agent (The Investigator)

Responsible for:
- External web search when internal memory is insufficient
- Using Tavily AI for clean parsed content
- Applying trusted source filtering (Allow-List)
- Managing search results
"""

from typing import List, Optional

from .base import BaseAgent
from ..core.config import settings
from ..core.state import (
    AgentState,
    SearchResult,
    add_agent_to_trace
)
from ..tools.tavily_search import TavilySearchTool, get_tavily_tool
from ..tools.source_filter import SourceFilter, TrustTier, get_source_filter
from ..utils.logger import get_logger


logger = get_logger()


class ExecutorAgent(BaseAgent):
    """
    The Executor Agent (The Investigator).
    
    Activated only when Qdrant internal memory lacks confidence (Cache Miss).
    Uses Tavily AI for clean, parsed web content.
    
    Implements Safety Filter 1 (Pre-Process):
    - Allow-List filtering of trusted sources
    - Discards results not from trusted_sources.json
    """
    
    name = "Executor"
    
    def __init__(
        self,
        tavily_tool: TavilySearchTool = None,
        source_filter: SourceFilter = None
    ):
        """
        Initialize the Executor agent.
        
        Args:
            tavily_tool: Tavily search tool instance
            source_filter: Source filter instance
        """
        self.tavily = tavily_tool or get_tavily_tool()
        self.source_filter = source_filter or get_source_filter()
        self.max_results = settings.tavily_max_results
        self.max_trust_tier = TrustTier(settings.min_source_tier_for_memory)
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process state by executing external search.
        
        Flow:
        1. Execute Tavily search
        2. Apply trusted source filter (Safety Filter 1)
        3. Update state with trusted results
        """
        state = add_agent_to_trace(state, self.name)
        
        # Get current query
        query = self._get_current_query(state)
        
        logger.with_agent(self.name).info(
            f"Executing external search for: {query[:50]}..."
        )
        
        # Perform fact-check search (includes debunking queries)
        search_results = await self.tavily.fact_check_search(
            claim=query,
            include_debunk=True
        )
        
        # Apply Safety Filter 1: Trusted source filtering
        trusted_results = self._apply_trust_filter(search_results)
        
        # Update state
        state["search_triggered"] = True
        state["search_results"] = [r.to_dict() for r in trusted_results]
        state["trusted_results_count"] = len(trusted_results)
        
        # Log filtering results
        filtered_count = len(search_results) - len(trusted_results)
        logger.with_agent(self.name).info(
            f"Search complete: {len(trusted_results)} trusted results "
            f"(filtered {filtered_count} untrusted)"
        )
        
        # Log detailed search results for debugging
        for i, result in enumerate(trusted_results, 1):
            logger.with_agent(self.name).info(
                f"Result {i}: [{result.domain}] {result.title}"
            )
            logger.with_agent(self.name).debug(
                f"Content preview: {result.content[:300]}..."
            )
        
        # Generate source summary
        source_urls = [r.url for r in trusted_results]
        source_summary = self.source_filter.get_source_summary(source_urls)
        
        logger.with_agent(self.name).info(
            f"Source quality: {source_summary['overall_quality']}"
        )
        
        return state
    
    def _get_current_query(self, state: AgentState) -> str:
        """Get the current task's query."""
        sub_tasks = state.get("sub_tasks", [])
        current_idx = state.get("current_task_index", 0)
        
        if sub_tasks and current_idx < len(sub_tasks):
            return sub_tasks[current_idx].get("query", state["original_query"])
        
        return state.get("original_query", "")
    
    def _apply_trust_filter(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Apply Safety Filter 1: Trusted source filtering.
        
        Implements the "Allow-List" protocol.
        """
        trusted = []
        
        for result in results:
            tier = TrustTier(result.trust_tier)
            
            # Check against allow-list (max tier threshold)
            if tier <= self.max_trust_tier:
                trusted.append(result)
                logger.with_agent(self.name).debug(
                    f"Accepted: {result.domain} (tier {tier.value})"
                )
            else:
                logger.with_agent(self.name).info(
                    f"Filtered: {result.domain} (tier {tier.value} > max {self.max_trust_tier.value})"
                )
        
        # Sort by trust tier (most trusted first)
        trusted.sort(key=lambda r: r.trust_tier)
        
        return trusted
    
    async def search_specific_claim(
        self,
        claim: str,
        context: str = None
    ) -> List[SearchResult]:
        """
        Search for a specific claim with optional context.
        
        Args:
            claim: The claim to verify
            context: Additional context for the search
            
        Returns:
            List of trusted search results
        """
        result = await self.tavily.search_with_context(
            query=claim,
            context=context
        )
        
        return result["results"]
    
    async def search_multiple_claims(
        self,
        claims: List[str]
    ) -> dict:
        """
        Search for multiple claims in parallel.
        
        Args:
            claims: List of claims to search
            
        Returns:
            Dictionary mapping claims to results
        """
        results_dict = await self.tavily.search_multiple(
            queries=claims,
            max_results_per_query=5,
            filter_trusted=True
        )
        
        return results_dict
    
    def should_run(self, state: AgentState) -> bool:
        """
        Determine if Executor should run.
        
        Only runs if:
        1. Cache miss (retrieval insufficient)
        2. Search not already triggered
        """
        cache_hit = state.get("cache_hit", False)
        search_triggered = state.get("search_triggered", False)
        
        return not cache_hit and not search_triggered
    
    def get_evidence_quality(self, state: AgentState) -> str:
        """
        Assess the quality of search evidence.
        
        Returns:
            Quality string: "high", "medium", "low", "none"
        """
        results = state.get("search_results", [])
        
        if not results:
            return "none"
        
        # Get best trust tier
        best_tier = min(r.get("trust_tier", 5) for r in results)
        
        if best_tier == 1:
            return "high"
        elif best_tier == 2:
            return "medium"
        elif best_tier == 3:
            return "moderate"
        else:
            return "low"
    
    def compile_evidence_text(self, state: AgentState) -> str:
        """
        Compile all search results into a single evidence text.
        
        Used by the Critic for analysis.
        """
        results = state.get("search_results", [])
        
        if not results:
            return "No external search results available."
        
        evidence_parts = []
        for i, result in enumerate(results, 1):
            evidence_parts.append(
                f"[Source {i}: {result.get('domain', 'Unknown')} "
                f"(Trust Tier {result.get('trust_tier', '?')})]\n"
                f"Title: {result.get('title', 'No title')}\n"
                f"Content: {result.get('content', 'No content')}\n"
            )
        
        return "\n---\n".join(evidence_parts)
