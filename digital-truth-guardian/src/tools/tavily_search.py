"""
Tavily AI search tool integration.

Provides clean, parsed web search results for the Executor agent.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import lru_cache

from tavily import TavilyClient, AsyncTavilyClient

from ..core.config import settings
from ..core.state import SearchResult
from ..utils.logger import get_logger
from ..utils.helpers import async_retry, extract_domain, sanitize_text
from .source_filter import SourceFilter, TrustTier, get_source_filter


logger = get_logger()


@dataclass
class TavilySearchConfig:
    """Configuration for Tavily search."""
    max_results: int = 10
    search_depth: str = "advanced"  # "basic" or "advanced"
    include_raw_content: bool = False
    include_images: bool = False
    include_answer: bool = True


class TavilySearchTool:
    """
    Tavily AI search tool for the Executor agent.
    
    Features:
    - Clean, parsed text output (no raw HTML)
    - Automatic source trust filtering
    - Async support for parallel searches
    """
    
    def __init__(
        self,
        api_key: str = None,
        source_filter: SourceFilter = None,
        config: TavilySearchConfig = None
    ):
        """
        Initialize Tavily search tool.
        
        Args:
            api_key: Tavily API key
            source_filter: Source filter instance
            config: Search configuration
        """
        self.api_key = api_key or settings.tavily_api_key
        self.source_filter = source_filter or get_source_filter()
        self.config = config or TavilySearchConfig(
            max_results=settings.tavily_max_results
        )
        
        # Initialize clients
        if self.api_key:
            self.sync_client = TavilyClient(api_key=self.api_key)
            self.async_client = AsyncTavilyClient(api_key=self.api_key)
        else:
            self.sync_client = None
            self.async_client = None
            logger.with_agent("Executor").warning(
                "Tavily API key not configured - search will be disabled"
            )
    
    @async_retry(max_retries=3, delay=1.0)
    async def search(
        self,
        query: str,
        max_results: int = None,
        filter_trusted: bool = True,
        max_trust_tier: TrustTier = TrustTier.TIER_3_REPUTABLE
    ) -> List[SearchResult]:
        """
        Perform a web search with automatic source filtering.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            filter_trusted: Whether to filter by trusted sources
            max_trust_tier: Maximum acceptable trust tier
            
        Returns:
            List of SearchResult objects
        """
        if not self.async_client:
            logger.with_agent("Executor").error("Tavily client not initialized")
            return []
        
        max_results = max_results or self.config.max_results
        
        logger.with_agent("Executor").info(
            f"Searching: {query[:50]}... (max_results={max_results})"
        )
        
        # Execute search
        response = await self.async_client.search(
            query=query,
            max_results=max_results,
            search_depth=self.config.search_depth,
            include_raw_content=self.config.include_raw_content,
            include_images=self.config.include_images,
            include_answer=self.config.include_answer
        )
        
        # Process results
        results = []
        raw_results = response.get("results", [])
        
        for item in raw_results:
            url = item.get("url", "")
            domain = extract_domain(url)
            trust_tier = self.source_filter.get_trust_tier(url)
            
            # Apply trust filter
            if filter_trusted and trust_tier > max_trust_tier:
                logger.with_agent("Executor").debug(
                    f"Filtered out {domain} (tier {trust_tier.value})"
                )
                continue
            
            result = SearchResult(
                title=sanitize_text(item.get("title", "")),
                url=url,
                content=sanitize_text(item.get("content", "")),
                domain=domain,
                trust_tier=trust_tier.value,
                published_date=item.get("published_date")
            )
            results.append(result)
        
        # Sort by trust tier (most trusted first)
        results.sort(key=lambda r: r.trust_tier)
        
        logger.with_agent("Executor").info(
            f"Search complete: {len(results)} trusted results "
            f"(filtered {len(raw_results) - len(results)} untrusted)"
        )
        
        return results
    
    async def search_multiple(
        self,
        queries: List[str],
        max_results_per_query: int = 5,
        filter_trusted: bool = True
    ) -> Dict[str, List[SearchResult]]:
        """
        Search multiple queries in parallel.
        
        Args:
            queries: List of search queries
            max_results_per_query: Max results per query
            filter_trusted: Whether to filter by trusted sources
            
        Returns:
            Dictionary mapping queries to results
        """
        tasks = [
            self.search(q, max_results_per_query, filter_trusted)
            for q in queries
        ]
        
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        results_dict = {}
        for query, results in zip(queries, all_results):
            if isinstance(results, Exception):
                logger.with_agent("Executor").error(
                    f"Search failed for '{query}': {results}"
                )
                results_dict[query] = []
            else:
                results_dict[query] = results
        
        return results_dict
    
    async def fact_check_search(
        self,
        claim: str,
        include_debunk: bool = True
    ) -> List[SearchResult]:
        """
        Specialized search for fact-checking a claim.
        
        Searches both for supporting evidence and fact-check articles.
        
        Args:
            claim: Claim to fact-check
            include_debunk: Include searches for debunking
            
        Returns:
            Combined list of results
        """
        queries = [claim]
        
        if include_debunk:
            queries.extend([
                f"{claim} fact check",
                f"is it true that {claim}",
                f"{claim} debunked OR verified"
            ])
        
        all_results = await self.search_multiple(
            queries,
            max_results_per_query=3,
            filter_trusted=True
        )
        
        # Merge and deduplicate
        seen_urls = set()
        merged = []
        
        for results in all_results.values():
            for result in results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    merged.append(result)
        
        # Sort by trust tier
        merged.sort(key=lambda r: r.trust_tier)
        
        return merged
    
    def get_answer_summary(self, response: Dict[str, Any]) -> Optional[str]:
        """
        Extract the AI-generated answer summary from Tavily response.
        
        Args:
            response: Raw Tavily response
            
        Returns:
            Answer string if available
        """
        return response.get("answer")
    
    async def search_with_context(
        self,
        query: str,
        context: str = None
    ) -> Dict[str, Any]:
        """
        Search with additional context for better results.
        
        Args:
            query: Main search query
            context: Additional context to refine search
            
        Returns:
            Dict with results and metadata
        """
        # Build enhanced query
        enhanced_query = query
        if context:
            enhanced_query = f"{query}. Context: {context}"
        
        results = await self.search(enhanced_query)
        
        # Generate source analysis
        source_urls = [r.url for r in results]
        source_summary = self.source_filter.get_source_summary(source_urls)
        
        return {
            "query": query,
            "results": results,
            "result_count": len(results),
            "source_analysis": source_summary,
            "searched_at": datetime.utcnow().isoformat()
        }


# ==================== Cached Instance ====================

@lru_cache()
def get_tavily_tool() -> TavilySearchTool:
    """Get cached Tavily search tool instance."""
    return TavilySearchTool()
