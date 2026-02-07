"""
Archivist Agent (The Guardian)

Responsible for:
- Maintaining Qdrant integrity
- Context-aware memory evolution (Static vs Transient facts)
- Temporal versioning for changing facts
- Safety Filter 3: Source-tier based memory write control
- Deduplication and cleanup
"""

import json
from datetime import datetime
from typing import Optional, Tuple

import google.generativeai as genai

from .base import BaseAgent
from ..core.config import settings, ModelConfig
from ..core.state import (
    AgentState,
    Verdict,
    FactType,
    Evidence,
    add_agent_to_trace,
    compile_evidence
)
from ..database.qdrant_client import QdrantManager, get_qdrant_manager
from ..database.schema import KnowledgeRecord
from ..tools.source_filter import SourceFilter, TrustTier, get_source_filter
from ..utils.logger import get_logger
from ..utils.helpers import generate_content_hash


logger = get_logger()


# ==================== Prompts ====================

FACT_CLASSIFICATION_PROMPT = """You are classifying a verified claim for memory storage.

## Verified Claim
{claim}

## Verdict
{verdict}

## Context
{context}

## Classification Task
Determine if this fact is:

1. **STATIC**: Immutable historical facts that never change
   - Historical events (e.g., "World War II ended in 1945")
   - Scientific constants (e.g., "Speed of light is 299,792,458 m/s")
   - Biographical facts (e.g., "Einstein was born in 1879")
   - Debunked misinformation (e.g., "The Earth is NOT flat")

2. **TRANSIENT**: Facts that may change over time
   - Current positions (e.g., "X is the CEO of Y")
   - Records and statistics (e.g., "X has won Y awards")
   - Current events (e.g., "The latest version is X")
   - Policy/regulation status

## Response Format (JSON)
{{
    "fact_type": "STATIC" | "TRANSIENT",
    "reasoning": "<brief explanation>",
    "should_memorize": true | false,
    "memorization_reason": "<why this should/shouldn't be memorized>"
}}
"""

DEDUPLICATION_PROMPT = """You are checking if a new fact duplicates or updates an existing one.

## New Fact
{new_fact}
Verdict: {new_verdict}

## Existing Fact
{existing_fact}
Verdict: {existing_verdict}
Recorded: {existing_date}

## Question
Are these facts:
1. DUPLICATE: Same fact, same verdict (skip insert)
2. UPDATE: Same topic but new information (need versioning)
3. RELATED: Similar but distinct facts (can coexist)
4. DIFFERENT: Unrelated facts

## Response Format (JSON)
{{
    "relationship": "DUPLICATE" | "UPDATE" | "RELATED" | "DIFFERENT",
    "reasoning": "<explanation>",
    "action": "SKIP" | "VERSION" | "INSERT"
}}
"""


class ArchivistAgent(BaseAgent):
    """
    The Archivist Agent (The Guardian).
    
    Asynchronous worker that maintains Qdrant integrity.
    Uses Gemini Flash for background classification.
    
    Implements:
    - Context-Aware Memory Evolution (Static vs Transient)
    - Temporal Versioning for transient facts
    - Safety Filter 3: Source-tier based write control
    - Deduplication
    """
    
    name = "Archivist"
    
    def __init__(
        self,
        qdrant_manager: QdrantManager = None,
        source_filter: SourceFilter = None
    ):
        """
        Initialize the Archivist agent.
        
        Args:
            qdrant_manager: Qdrant manager instance
            source_filter: Source filter instance
        """
        self.qdrant = qdrant_manager or get_qdrant_manager()
        self.source_filter = source_filter or get_source_filter()
        self.model_name = ModelConfig.ARCHIVIST["model"]
        self.min_tier_for_memory = TrustTier(settings.min_source_tier_for_memory)
        
        # Configure Gemini
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": ModelConfig.ARCHIVIST["temperature"],
                    "max_output_tokens": ModelConfig.ARCHIVIST["max_tokens"]
                }
            )
        else:
            self.model = None
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process state and handle memory management.
        
        Flow:
        1. Check if verdict should be memorized
        2. Apply Safety Filter 3 (source tier check)
        3. Classify fact type (Static vs Transient)
        4. Check for duplicates
        5. Write to Qdrant with appropriate versioning
        """
        state = add_agent_to_trace(state, self.name)
        
        # Skip if memory write is disabled
        if not settings.enable_memory_write:
            state["memory_written"] = False
            state["memory_write_reason"] = "Memory write disabled"
            return state
        
        # Skip if verdict is UNCERTAIN or PENDING
        verdict_str = state.get("verdict", "PENDING")
        if verdict_str in [Verdict.UNCERTAIN.value, Verdict.PENDING.value]:
            state["memory_written"] = False
            state["memory_write_reason"] = f"Verdict is {verdict_str}, not memorizing"
            logger.with_agent(self.name).info(
                f"Skipping memorization: verdict is {verdict_str}"
            )
            return state
        
        # Get current query and compile evidence
        query = self._get_current_query(state)
        evidence = compile_evidence(state)
        
        # Safety Filter 3: Check source tier
        should_memorize, rejection_reason = await self._apply_safety_filter(
            evidence=evidence
        )
        
        if not should_memorize:
            state["should_memorize"] = False
            state["memory_written"] = False
            state["memory_write_reason"] = rejection_reason
            logger.with_agent(self.name).info(
                f"Safety Filter 3 rejected: {rejection_reason}"
            )
            return state
        
        # Classify fact type
        verdict = Verdict(verdict_str)
        fact_type = await self._classify_fact_type(
            claim=query,
            verdict=verdict,
            context=state.get("explanation", "")
        )
        
        # Check for duplicates
        duplicate_check = await self._check_duplicates(
            claim=query,
            verdict=verdict
        )
        
        if duplicate_check["action"] == "SKIP":
            state["should_memorize"] = False
            state["memory_written"] = False
            state["memory_write_reason"] = "Duplicate fact already exists"
            logger.with_agent(self.name).info(
                f"Skipping duplicate: {duplicate_check['reasoning']}"
            )
            return state
        
        # Get best source domain
        source_domain = self._get_best_source(evidence)
        
        # Create knowledge record
        record = KnowledgeRecord(
            text=query,
            verdict=verdict,
            fact_type=fact_type,
            source_domain=source_domain,
            confidence=state.get("confidence", 0.0),
            explanation=state.get("explanation", "")
        )
        
        # Handle versioning for updates
        if duplicate_check["action"] == "VERSION":
            existing_id = duplicate_check.get("existing_id")
            if existing_id:
                success = await self.qdrant.update_transient_fact(
                    old_record_id=existing_id,
                    new_record=record
                )
                action = "versioned"
            else:
                success = await self.qdrant.upsert_record(record)
                action = "inserted"
        else:
            # Regular insert
            success = await self.qdrant.upsert_record(record)
            action = "inserted"
        
        # Update state
        state["should_memorize"] = True
        state["fact_type"] = fact_type.value
        state["memory_written"] = success
        state["memory_write_reason"] = f"Successfully {action}" if success else "Write failed"
        
        logger.with_agent(self.name).info(
            f"Memory {action}: {verdict.value} fact ({fact_type.value})"
        )
        
        return state
    
    def _get_current_query(self, state: AgentState) -> str:
        """Get the current task's query."""
        sub_tasks = state.get("sub_tasks", [])
        current_idx = state.get("current_task_index", 0)
        
        if sub_tasks and current_idx < len(sub_tasks):
            return sub_tasks[current_idx].get("query", state["original_query"])
        
        return state.get("original_query", "")
    
    async def _apply_safety_filter(
        self,
        evidence: Evidence
    ) -> Tuple[bool, str]:
        """
        Apply Safety Filter 3: Source-tier based write control.
        
        Returns:
            Tuple of (should_memorize, reason)
        """
        # Get all source URLs
        source_urls = evidence.best_sources
        
        if not source_urls:
            return False, "No sources available for verification"
        
        # Check if any source meets the minimum tier
        should_memorize = self.source_filter.should_memorize(
            sources=source_urls,
            min_tier=self.min_tier_for_memory
        )
        
        if not should_memorize:
            source_summary = self.source_filter.get_source_summary(source_urls)
            return False, (
                f"Sources do not meet trust threshold. "
                f"Quality: {source_summary['overall_quality']}, "
                f"Best tier: {source_summary['best_tier']}"
            )
        
        return True, "Sources meet trust threshold"
    
    async def _classify_fact_type(
        self,
        claim: str,
        verdict: Verdict,
        context: str
    ) -> FactType:
        """Classify the fact as STATIC or TRANSIENT."""
        if not self.model:
            # Default to STATIC if no LLM
            return FactType.STATIC
        
        prompt = FACT_CLASSIFICATION_PROMPT.format(
            claim=claim,
            verdict=verdict.value,
            context=context[:500] if context else "No additional context"
        )
        
        try:
            response = await self.model.generate_content_async(prompt)
            text = response.text
            
            # Extract JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            parsed = json.loads(text.strip())
            fact_type_str = parsed.get("fact_type", "STATIC").upper()
            
            return FactType(fact_type_str)
            
        except Exception as e:
            logger.with_agent(self.name).warning(
                f"Failed to classify fact type: {e}"
            )
            return FactType.STATIC
    
    async def _check_duplicates(
        self,
        claim: str,
        verdict: Verdict
    ) -> dict:
        """
        Check for duplicate or related existing facts.
        
        Returns:
            Dict with action (SKIP, VERSION, INSERT) and reasoning
        """
        # Check for exact or near-exact duplicates
        existing = await self.qdrant.check_duplicate(claim)
        
        if not existing:
            return {
                "action": "INSERT",
                "reasoning": "No existing similar facts found"
            }
        
        # Use LLM to determine relationship
        if self.model:
            prompt = DEDUPLICATION_PROMPT.format(
                new_fact=claim,
                new_verdict=verdict.value,
                existing_fact=existing.text,
                existing_verdict=existing.verdict.value,
                existing_date=existing.created_at.isoformat()
            )
            
            try:
                response = await self.model.generate_content_async(prompt)
                text = response.text
                
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                
                parsed = json.loads(text.strip())
                
                result = {
                    "action": parsed.get("action", "INSERT"),
                    "reasoning": parsed.get("reasoning", ""),
                    "relationship": parsed.get("relationship", "DIFFERENT")
                }
                
                if result["action"] == "VERSION":
                    result["existing_id"] = existing.id
                
                return result
                
            except Exception as e:
                logger.with_agent(self.name).warning(
                    f"Deduplication check failed: {e}"
                )
        
        # Fallback: simple comparison
        if existing.verdict == verdict:
            return {
                "action": "SKIP",
                "reasoning": "Similar fact with same verdict exists"
            }
        else:
            return {
                "action": "VERSION",
                "reasoning": "Similar fact exists with different verdict",
                "existing_id": existing.id
            }
    
    def _get_best_source(self, evidence: Evidence) -> str:
        """Get the most trusted source domain."""
        all_sources = []
        
        # From retrieved docs
        for doc in evidence.retrieved_docs:
            if doc.source_domain:
                tier = self.source_filter.get_trust_tier(doc.source_domain)
                all_sources.append((doc.source_domain, tier.value))
        
        # From search results
        for result in evidence.search_results:
            all_sources.append((result.domain, result.trust_tier))
        
        if not all_sources:
            return "unknown"
        
        # Return source with best (lowest) tier
        return min(all_sources, key=lambda x: x[1])[0]
    
    async def cleanup_expired_records(self, days_old: int = 30) -> int:
        """
        Clean up old expired records.
        
        Args:
            days_old: Delete records expired more than this many days ago
            
        Returns:
            Number of records deleted
        """
        # Implementation would scroll through expired records and delete old ones
        logger.with_agent(self.name).info(
            f"Cleanup requested for records expired > {days_old} days"
        )
        return 0  # Placeholder
    
    async def get_memory_stats(self) -> dict:
        """Get statistics about the knowledge base."""
        stats = await self.qdrant.get_verdict_stats()
        info = await self.qdrant.get_collection_info()
        
        return {
            "total_records": info.get("points_count", 0),
            "verdicts": stats,
            "collection_status": info.get("status", "unknown")
        }
