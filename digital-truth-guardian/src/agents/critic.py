"""
Critic Agent (The Judge)

Responsible for:
- Deep reasoning and entailment checking
- Comparing user claims against retrieved evidence
- Determining verdict (TRUE, FALSE, UNCERTAIN)
- Generating feedback signals for the Planner
"""

import json
from typing import Optional, Tuple

from google import genai
from google.genai import types

from .base import BaseAgent
from ..core.config import settings, ModelConfig
from ..core.state import (
    AgentState,
    Verdict,
    Evidence,
    FeedbackSignal,
    AgentAction,
    add_agent_to_trace,
    compile_evidence
)
from ..utils.logger import get_logger
from ..utils.helpers import format_confidence


logger = get_logger()


# ==================== Prompts ====================

ENTAILMENT_PROMPT = """You are a fact-checking judge with expertise in logical reasoning and evidence evaluation.

## Task
Analyze whether the provided evidence PROVES, DISPROVES, or DOES NOT CONCLUSIVELY ADDRESS the user's claim.

## User Claim
{claim}

## Evidence from Knowledge Base
{kb_evidence}

## Evidence from Web Search
{web_evidence}

## Instructions
1. Carefully read all evidence provided
2. Determine if the evidence directly addresses the claim
3. Check for logical consistency between claim and evidence
4. Consider source reliability (government/academic sources are most reliable)
5. If evidence is contradictory, weigh by source quality

## Verdict Guidelines
- **TRUE**: Evidence clearly and conclusively SUPPORTS the claim
- **FALSE**: Evidence clearly and conclusively CONTRADICTS/DEBUNKS the claim
- **UNCERTAIN**: Evidence is insufficient, conflicting, or doesn't directly address the claim

## Response Format (JSON)
{{
    "verdict": "TRUE" | "FALSE" | "UNCERTAIN",
    "confidence": <0.0-1.0>,
    "reasoning": "<detailed step-by-step analysis>",
    "key_evidence": ["<most important evidence point 1>", "<point 2>"],
    "evidence_quality": "high" | "medium" | "low" | "insufficient",
    "caveats": ["<any limitations or caveats>"] or null
}}
"""

INSUFFICIENT_CHECK_PROMPT = """You are evaluating whether the available evidence is sufficient to make a verdict.

## User Claim
{claim}

## Available Evidence
- Knowledge Base Documents: {kb_count}
- Web Search Results: {web_count}
- Best Source Quality: {source_quality}

## Evidence Preview
{evidence_preview}

## Question
Is this evidence SUFFICIENT to make a reliable verdict on the claim?

Consider:
1. Does the evidence directly address the specific claim?
2. Are the sources reliable enough?
3. Is there enough corroborating information?

## Response Format (JSON)
{{
    "is_sufficient": true | false,
    "reason": "<explanation>",
    "suggested_action": "CRITIQUE" | "SEARCH" | "LOOP_BACK",
    "missing_info": "<what information would help>" or null
}}
"""


class CriticAgent(BaseAgent):
    """
    The Critic Agent (The Judge).
    
    Uses Gemini Pro for high reasoning capabilities.
    Implements:
    - Entailment checking (evidence vs claim)
    - Verdict determination with confidence scores
    - Feedback generation for the Planner loop
    """
    
    name = "Critic"
    
    def __init__(self):
        """Initialize the Critic agent."""
        self.model_name = ModelConfig.CRITIC["model"]
        self.temperature = ModelConfig.CRITIC["temperature"]
        self.max_tokens = ModelConfig.CRITIC["max_tokens"]
        
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
        Process state by analyzing evidence and determining verdict.
        
        Flow:
        1. Check if evidence is sufficient
        2. If insufficient, generate feedback for Planner
        3. If sufficient, perform entailment check and determine verdict
        4. Record episode and share insights with other agents
        """
        state = add_agent_to_trace(state, self.name)
        
        # Get current query
        query = self._get_current_query(state)
        
        logger.with_agent(self.name).info(
            f"Analyzing evidence for: {query[:50]}..."
        )
        
        # Compile evidence
        evidence = compile_evidence(state)
        
        # Check if evidence is sufficient
        is_sufficient, feedback = await self._check_evidence_sufficiency(
            claim=query,
            evidence=evidence
        )
        
        if not is_sufficient:
            # Generate feedback for Planner
            state["feedback"] = {
                "is_sufficient": False,
                "reason": feedback.reason,
                "suggested_action": feedback.suggested_action.value,
                "missing_info": feedback.missing_info
            }
            state["verdict"] = Verdict.PENDING.value
            
            # Record insufficient evidence episode
            await self.record_episode(
                state=state,
                action_type="critique",
                outcome="insufficient",
                decision_reasoning=feedback.reason,
                confidence=0.3,
            )
            
            # Share warning with other agents
            await self.write_shared_context(
                content=f"Insufficient evidence for '{query[:50]}...': {feedback.reason}",
                context_type="warning",
                session_id=state.get("session_id"),
                priority=3,
                ttl_minutes=15,
            )
            
            logger.with_agent(self.name).info(
                f"Evidence insufficient: {feedback.reason}"
            )
            
            return state
        
        # Perform entailment check
        verdict, confidence, explanation = await self._entailment_check(
            claim=query,
            evidence=evidence
        )
        
        # Update state with verdict
        state["evidence"] = evidence.to_dict()
        state["verdict"] = verdict.value
        state["confidence"] = confidence
        state["explanation"] = explanation
        state["feedback"] = {
            "is_sufficient": True,
            "reason": "Evidence analyzed successfully"
        }
        
        # Record successful critique episode
        await self.record_episode(
            state=state,
            action_type="critique",
            outcome="success",
            decision_reasoning=f"Verdict: {verdict.value}",
            confidence=confidence,
        )
        
        # Share insight about this verdict
        await self.write_shared_context(
            content=f"Verdict for '{query[:50]}...': {verdict.value} (confidence: {confidence:.2f})",
            context_type="insight",
            session_id=state.get("session_id"),
            priority=2,
            ttl_minutes=60,
            tags=[verdict.value, "verdict"],
        )
        
        logger.with_agent(self.name).info(
            f"Verdict: {verdict.value} ({format_confidence(confidence)})"
        )
        
        return state
    
    def _get_current_query(self, state: AgentState) -> str:
        """Get the current task's query."""
        sub_tasks = state.get("sub_tasks", [])
        current_idx = state.get("current_task_index", 0)
        
        if sub_tasks and current_idx < len(sub_tasks):
            return sub_tasks[current_idx].get("query", state["original_query"])
        
        return state.get("original_query", "")
    
    async def _check_evidence_sufficiency(
        self,
        claim: str,
        evidence: Evidence
    ) -> Tuple[bool, Optional[FeedbackSignal]]:
        """
        Check if evidence is sufficient for a verdict.
        
        Returns:
            Tuple of (is_sufficient, feedback_signal)
        """
        # Quick check: if no evidence at all
        if not evidence.retrieved_docs and not evidence.search_results:
            return False, FeedbackSignal(
                is_sufficient=False,
                reason="No evidence available",
                suggested_action=AgentAction.SEARCH,
                missing_info="External search needed"
            )
        
        # If we have trusted search results (tier 1-3), consider evidence sufficient
        # and proceed to entailment check - let the LLM judge the content
        trusted_results = [r for r in evidence.search_results if r.trust_tier <= 3]
        if trusted_results:
            logger.with_agent(self.name).info(
                f"Found {len(trusted_results)} trusted sources - proceeding to verdict"
            )
            return True, None
        
        # If we have high-scoring KB docs, also sufficient
        high_score_docs = [d for d in evidence.retrieved_docs if d.score >= 0.7]
        if high_score_docs:
            logger.with_agent(self.name).info(
                f"Found {len(high_score_docs)} high-scoring KB docs - proceeding to verdict"
            )
            return True, None
        
        # Build evidence preview for LLM check (increased from 200 to 500 chars)
        preview_parts = []
        for doc in evidence.retrieved_docs[:2]:
            preview_parts.append(f"[KB] {doc.text[:500]}...")
        for result in evidence.search_results[:3]:
            preview_parts.append(f"[Web: {result.domain}] {result.content[:500]}...")
        
        evidence_preview = "\n".join(preview_parts) if preview_parts else "No evidence"
        
        # LLM check for complex cases
        prompt = INSUFFICIENT_CHECK_PROMPT.format(
            claim=claim,
            kb_count=len(evidence.retrieved_docs),
            web_count=len(evidence.search_results),
            source_quality=evidence.source_quality,
            evidence_preview=evidence_preview
        )
        
        result = await self._call_llm(prompt)
        
        try:
            parsed = json.loads(result)
            is_sufficient = parsed.get("is_sufficient", False)
            
            if not is_sufficient:
                action_str = parsed.get("suggested_action", "SEARCH")
                try:
                    action = AgentAction(action_str.lower())
                except ValueError:
                    action = AgentAction.SEARCH
                
                return False, FeedbackSignal(
                    is_sufficient=False,
                    reason=parsed.get("reason", "Evidence insufficient"),
                    suggested_action=action,
                    missing_info=parsed.get("missing_info")
                )
            
            return True, None
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.with_agent(self.name).warning(
                f"Failed to parse sufficiency check: {e}"
            )
            # Default to sufficient if we have any evidence from trusted sources
            if evidence.search_results or evidence.retrieved_docs:
                return True, None
            return False, FeedbackSignal(
                is_sufficient=False,
                reason="Unable to evaluate evidence",
                suggested_action=AgentAction.SEARCH
            )
    
    async def _entailment_check(
        self,
        claim: str,
        evidence: Evidence
    ) -> Tuple[Verdict, float, str]:
        """
        Perform the entailment check to determine verdict.
        
        Returns:
            Tuple of (verdict, confidence, explanation)
        """
        # Format evidence
        kb_evidence = self._format_kb_evidence(evidence)
        web_evidence = self._format_web_evidence(evidence)
        
        prompt = ENTAILMENT_PROMPT.format(
            claim=claim,
            kb_evidence=kb_evidence,
            web_evidence=web_evidence
        )
        
        result = await self._call_llm(prompt)
        
        try:
            parsed = json.loads(result)
            
            verdict_str = parsed.get("verdict", "UNCERTAIN").upper()
            verdict = Verdict(verdict_str)
            
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            reasoning = parsed.get("reasoning", "Analysis complete.")
            key_evidence = parsed.get("key_evidence", [])
            caveats = parsed.get("caveats", [])
            
            # Build explanation
            explanation = self._build_explanation(
                reasoning=reasoning,
                key_evidence=key_evidence,
                caveats=caveats,
                evidence_quality=parsed.get("evidence_quality", "unknown")
            )
            
            return verdict, confidence, explanation
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.with_agent(self.name).error(
                f"Failed to parse entailment result: {e}"
            )
            return Verdict.UNCERTAIN, 0.3, "Unable to determine verdict from available evidence."
    
    def _format_kb_evidence(self, evidence: Evidence) -> str:
        """Format knowledge base evidence for the prompt."""
        if not evidence.retrieved_docs:
            return "No relevant documents found in knowledge base."
        
        parts = []
        for i, doc in enumerate(evidence.retrieved_docs, 1):
            verdict_note = f" [Previously verified: {doc.verdict.value}]" if doc.verdict else ""
            parts.append(
                f"Document {i} (Score: {doc.score:.3f}){verdict_note}:\n{doc.text}"
            )
        
        return "\n\n".join(parts)
    
    def _format_web_evidence(self, evidence: Evidence) -> str:
        """Format web search evidence for the prompt."""
        if not evidence.search_results:
            return "No web search results available."
        
        parts = []
        for i, result in enumerate(evidence.search_results, 1):
            parts.append(
                f"Source {i}: {result.domain} (Trust Tier: {result.trust_tier})\n"
                f"Title: {result.title}\n"
                f"Content: {result.content}"
            )
        
        return "\n\n---\n\n".join(parts)
    
    def _build_explanation(
        self,
        reasoning: str,
        key_evidence: list,
        caveats: list,
        evidence_quality: str
    ) -> str:
        """Build a comprehensive explanation string."""
        parts = [reasoning]
        
        if key_evidence:
            parts.append("\n\n**Key Evidence:**")
            for i, point in enumerate(key_evidence, 1):
                parts.append(f"{i}. {point}")
        
        if caveats:
            parts.append("\n\n**Caveats:**")
            for caveat in caveats:
                parts.append(f"- {caveat}")
        
        parts.append(f"\n\n*Evidence Quality: {evidence_quality}*")
        
        return "\n".join(parts)
    
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
            # Extract JSON from response
            text = response.text
            
            # Try to find JSON in the response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return text.strip()
        except Exception as e:
            logger.with_agent(self.name).error(f"LLM call failed: {e}")
            return "{}"
    
    def is_verdict_conclusive(self, state: AgentState) -> bool:
        """Check if the verdict is conclusive (not UNCERTAIN or PENDING)."""
        verdict = state.get("verdict", "PENDING")
        return verdict in [Verdict.TRUE.value, Verdict.FALSE.value]
    
    def should_trigger_feedback_loop(self, state: AgentState) -> bool:
        """Check if a feedback loop should be triggered."""
        feedback = state.get("feedback", {})
        return not feedback.get("is_sufficient", True)
