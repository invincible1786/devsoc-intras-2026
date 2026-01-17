"""
Mixture of Experts (MoE) for Graph of Thought Verification
Three specialized experts: Hallucination Hunter, Source Matcher, Logic Expert
Implements parallel execution with weighted voting
"""

import logging
import asyncio
import json
import re
from typing import Dict, List, Optional
from enum import Enum

from src.utils.groq_client import GroqClient

logger = logging.getLogger(__name__)


class ExpertType(Enum):
    """Types of verification experts"""
    HALLUCINATION_HUNTER = "hallucination_hunter"
    SOURCE_MATCHER = "source_matcher"
    LOGIC_EXPERT = "logic_expert"


class NodeAction(Enum):
    """Actions that can be taken on a node"""
    ACCEPT = "accept"
    REJECT = "reject"
    MERGE = "merge"
    RETHINK = "rethink"


def strip_markdown_json(text: str) -> str:
    """
    Remove markdown code fences and extract JSON from LLM responses.
    
    Args:
        text: Raw LLM response
        
    Returns:
        Cleaned JSON string
    """
    if not text:
        return text
    
    text = text.strip()
    
    # Try to find JSON within code fences
    code_fence_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_fence_pattern, text, re.DOTALL)
    if matches:
        text = matches[-1].strip()
    
    # Extract the JSON object/array
    for i, char in enumerate(text):
        if char in '{[':
            try:
                decoder = json.JSONDecoder()
                obj, end_idx = decoder.raw_decode(text[i:])
                return text[i:i+end_idx]
            except json.JSONDecodeError:
                continue
    
    return text.strip()


class HallucinationHunter:
    """
    Expert 1: Detects hallucinations by comparing thought against context.
    Ensures no fabricated information (names, dates, facts) not present in source.
    """
    
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client
        self.weight = 3.0  # High weight - hallucinations are critical
    
    async def verify(self, thought: str, context: str, metadata: Dict) -> Dict:
        """
        Check if the thought contains information not present in context.
        
        Args:
            thought: The thought/claim to verify
            context: Retrieved context from MetaKGP
            metadata: Additional metadata (sources, etc.)
            
        Returns:
            Verification result with score, confidence, and findings
        """
        prompt = f"""You are a Hallucination Detector for MetaKGP wiki verification.

CONTEXT FROM METAKGP:
{context}

CLAIM TO VERIFY:
{thought}

TASK: Check if the claim contains information NOT present in the context.

WHAT COUNTS AS HALLUCINATION:
- Names, dates, numbers not mentioned in context
- Events or facts completely fabricated
- Relationships or roles invented without basis

WHAT IS ACCEPTABLE:
- Direct extraction of information from context
- Listing names found under specific headings (e.g., "governors", "directors")
- Stating years or sessions exactly as mentioned in context

RULES:
- If context says "The governors for 2024-25 are: X, Y, Z" and claim says "The governors are X, Y, Z", that's VALID
- Only flag clear fabrications or incorrect information

Return valid JSON:
{{
    "hallucinations": [
        {{"claim": "specific fabricated claim", "reason": "why it's wrong"}}
    ],
    "confidence": 0.9,
    "verdict": "PASS",
    "reasoning": "explanation"
}}"""
        
        try:
            response = await self.groq_client.generate_expert(prompt, max_tokens=512)
            cleaned = strip_markdown_json(response)
            result = json.loads(cleaned)
            
            hallucinations = result.get("hallucinations", [])
            confidence = float(result.get("confidence", 0.5))
            verdict = result.get("verdict", "FAIL")
            
            # Calculate score: high if no hallucinations
            score = confidence if verdict == "PASS" and len(hallucinations) == 0 else 0.0
            
            return {
                "expert": ExpertType.HALLUCINATION_HUNTER.value,
                "score": score,
                "confidence": confidence,
                "passed": verdict == "PASS" and len(hallucinations) == 0,
                "hallucinations": hallucinations,
                "reasoning": result.get("reasoning", ""),
                "weight": self.weight
            }
            
        except Exception as e:
            logger.error(f"HallucinationHunter error: {e}")
            return {
                "expert": ExpertType.HALLUCINATION_HUNTER.value,
                "score": 0.0,
                "confidence": 0.0,
                "passed": False,
                "hallucinations": [],
                "reasoning": f"Error: {str(e)}",
                "weight": self.weight
            }


class SourceMatcher:
    """
    Expert 2: Verifies thought semantically matches the retrieved context.
    Ensures the meaning is contained in the chunks.
    """
    
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client
        self.weight = 2.5  # High weight - source matching is important
    
    async def verify(self, thought: str, context: str, metadata: Dict) -> Dict:
        """
        Check if the thought's meaning is supported by the context.
        
        Args:
            thought: The thought/claim to verify
            context: Retrieved context from MetaKGP
            metadata: Additional metadata
            
        Returns:
            Verification result with confidence score
        """
        prompt = f"""You are a Source Matcher. Verify that information is supported by the context.

CONTEXT FROM METAKGP:
{context}

CLAIM TO VERIFY:
{thought}

TASK: Does the context contain information that supports this claim?

EVALUATION:
1. Check if key facts (names, roles, dates) are present in context
2. Verify the claim accurately represents context information
3. Rate confidence from 0.0 to 1.0

SCORING:
- 0.8-1.0: All information clearly present in context
- 0.6-0.7: Most information present, minor details unclear
- 0.4-0.5: Partial match
- 0.0-0.3: Information not in context

Return valid JSON only:
{{
    "confidence": 0.9,
    "matching_snippets": ["relevant context excerpt"],
    "verdict": "PASS",
    "reasoning": "brief explanation"
}}"""
        
        try:
            response = await self.groq_client.generate_expert(prompt, max_tokens=512)
            cleaned = strip_markdown_json(response)
            result = json.loads(cleaned)
            
            confidence = float(result.get("confidence", 0.5))
            verdict = result.get("verdict", "FAIL")
            
            # Require at least 0.6 confidence to pass
            passed = verdict == "PASS" and confidence >= 0.6
            
            return {
                "expert": ExpertType.SOURCE_MATCHER.value,
                "score": confidence,
                "confidence": confidence,
                "passed": passed,
                "matching_snippets": result.get("matching_snippets", []),
                "reasoning": result.get("reasoning", ""),
                "weight": self.weight
            }
            
        except Exception as e:
            logger.error(f"SourceMatcher error: {e}")
            return {
                "expert": ExpertType.SOURCE_MATCHER.value,
                "score": 0.0,
                "confidence": 0.0,
                "passed": False,
                "matching_snippets": [],
                "reasoning": f"Error: {str(e)}",
                "weight": self.weight
            }


class LogicExpert:
    """
    Expert 3: Ensures reasoning chain makes sense and detects redundancy.
    Can suggest merging or discarding nodes.
    """
    
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client
        self.weight = 1.5  # Lower weight - soft gate
    
    async def verify(self, thought: str, context: str, metadata: Dict) -> Dict:
        """
        Check if the thought fits logically in the reasoning chain.
        
        Args:
            thought: The current thought
            context: Retrieved context
            metadata: Must include 'parent_thoughts' for reasoning chain
            
        Returns:
            Verification result with action suggestion
        """
        parent_thoughts = metadata.get("parent_thoughts", [])
        query = metadata.get("query", "")
        
        # Format parent thoughts
        history = "\n".join([f"{i+1}. {p}" for i, p in enumerate(parent_thoughts[-3:])])
        if not history:
            history = "[This is the first thought]"
        
        prompt = f"""You are a Logic Expert evaluating reasoning coherence.

ORIGINAL QUERY: {query}

PREVIOUS THOUGHTS:
{history}

NEW THOUGHT:
{thought}

TASK: Evaluate if this thought fits logically in the reasoning chain.

EVALUATION CRITERIA:
1. Does it logically follow from previous thoughts?
2. Is it redundant (says same thing as previous thoughts)?
3. Does it help answer the original query?

ACTIONS:
- "accept": Good thought, adds value
- "merge": Too similar to parent, can combine
- "reject": Completely redundant or illogical
- "rethink": Needs refinement

Return valid JSON only:
{{
    "coherence_score": 0.9,
    "is_redundant": false,
    "action": "accept",
    "reasoning": "brief explanation"
}}"""
        
        try:
            response = await self.groq_client.generate_expert(prompt, max_tokens=512)
            cleaned = strip_markdown_json(response)
            result = json.loads(cleaned)
            
            coherence = float(result.get("coherence_score", 0.5))
            is_redundant = result.get("is_redundant", False)
            action = result.get("action", "reject")
            
            # Pass if coherence >= 0.5 and action is accept/merge
            passed = coherence >= 0.5 and action in ["accept", "merge"]
            
            return {
                "expert": ExpertType.LOGIC_EXPERT.value,
                "score": coherence,
                "confidence": coherence,
                "passed": passed,
                "is_redundant": is_redundant,
                "action": action,
                "reasoning": result.get("reasoning", ""),
                "weight": self.weight
            }
            
        except Exception as e:
            logger.error(f"LogicExpert error: {e}")
            return {
                "expert": ExpertType.LOGIC_EXPERT.value,
                "score": 0.5,
                "confidence": 0.5,
                "passed": True,  # Default to pass on error (soft gate)
                "is_redundant": False,
                "action": "accept",
                "reasoning": f"Error: {str(e)}",
                "weight": self.weight
            }


class MoERouter:
    """
    Router that orchestrates the three experts with parallel execution and weighted voting.
    """
    
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client
        
        # Initialize experts
        self.hallucination_hunter = HallucinationHunter(groq_client)
        self.source_matcher = SourceMatcher(groq_client)
        self.logic_expert = LogicExpert(groq_client)
        
        logger.info("MoERouter initialized with 3 experts")
    
    async def verify_thought(
        self,
        thought: str,
        context: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Run all experts in parallel and aggregate results with weighted voting.
        
        Args:
            thought: The thought to verify
            context: Retrieved context
            metadata: Additional metadata (parent_thoughts, sources, etc.)
            
        Returns:
            Aggregated verification result with action
        """
        if metadata is None:
            metadata = {}
        
        logger.info(f"MoE verifying thought: {thought[:100]}...")
        
        # Check if this is a simple query - skip LogicExpert to save tokens
        is_simple = metadata.get("is_simple_query", False)
        
        if is_simple:
            # For simple queries, only run critical experts
            logger.info("Running lightweight verification for simple query")
            results = await asyncio.gather(
                self.hallucination_hunter.verify(thought, context, metadata),
                self.source_matcher.verify(thought, context, metadata),
                return_exceptions=True
            )
        else:
            # Run all experts in parallel
            results = await asyncio.gather(
                self.hallucination_hunter.verify(thought, context, metadata),
                self.source_matcher.verify(thought, context, metadata),
                self.logic_expert.verify(thought, context, metadata),
                return_exceptions=True
            )
        
        # Filter out any exceptions
        valid_results = [r for r in results if isinstance(r, dict)]
        
        if not valid_results:
            logger.error("All experts failed")
            return {
                "action": NodeAction.REJECT.value,
                "passed": False,
                "score": 0.0,
                "expert_results": [],
                "reasoning": "All experts failed to verify"
            }
        
        # Weighted voting
        total_weight = sum(r["weight"] for r in valid_results)
        weighted_score = sum(r["score"] * r["weight"] for r in valid_results) / total_weight
        
        # Decision logic
        hallucination_result = next((r for r in valid_results if r["expert"] == ExpertType.HALLUCINATION_HUNTER.value), None)
        source_result = next((r for r in valid_results if r["expert"] == ExpertType.SOURCE_MATCHER.value), None)
        logic_result = next((r for r in valid_results if r["expert"] == ExpertType.LOGIC_EXPERT.value), None)
        
        # Critical rules
        if source_result and not source_result["passed"]:
            # Source Matcher fail = immediate reject
            action = NodeAction.REJECT
            reasoning = f"Source match failed: {source_result['reasoning']}"
        elif hallucination_result and not hallucination_result["passed"]:
            # Hallucination detected = rethink
            action = NodeAction.RETHINK
            reasoning = f"Hallucinations detected: {hallucination_result['reasoning']}"
        elif logic_result and logic_result["action"] == "merge":
            # Logic suggests merge
            action = NodeAction.MERGE
            reasoning = f"Logic suggests merge: {logic_result['reasoning']}"
        elif logic_result and logic_result["action"] == "reject":
            # Logic rejects
            action = NodeAction.REJECT
            reasoning = f"Logic rejects: {logic_result['reasoning']}"
        elif weighted_score >= 0.6:
            # High confidence = accept
            action = NodeAction.ACCEPT
            reasoning = "All experts agree with high confidence"
        else:
            # Low confidence = rethink
            action = NodeAction.RETHINK
            reasoning = f"Low confidence ({weighted_score:.2f}), needs refinement"
        
        passed = action in [NodeAction.ACCEPT, NodeAction.MERGE]
        
        return {
            "action": action.value,
            "passed": passed,
            "score": weighted_score,
            "expert_results": valid_results,
            "reasoning": reasoning,
            "self_correction_feedback": self._generate_feedback(valid_results) if not passed else None
        }
    
    def _generate_feedback(self, expert_results: List[Dict]) -> str:
        """
        Generate self-correction feedback for the generator.
        
        Args:
            expert_results: Results from all experts
            
        Returns:
            Feedback string for next iteration
        """
        feedback_parts = []
        
        for result in expert_results:
            if not result["passed"]:
                expert_name = result["expert"]
                reasoning = result["reasoning"]
                
                if expert_name == ExpertType.HALLUCINATION_HUNTER.value:
                    hallucinations = result.get("hallucinations", [])
                    if hallucinations:
                        claims = ", ".join([h["claim"] for h in hallucinations])
                        feedback_parts.append(f"Remove unsupported claims: {claims}")
                
                elif expert_name == ExpertType.SOURCE_MATCHER.value:
                    feedback_parts.append(f"Better align with source material: {reasoning}")
                
                elif expert_name == ExpertType.LOGIC_EXPERT.value:
                    feedback_parts.append(f"Improve logic: {reasoning}")
        
        return " | ".join(feedback_parts) if feedback_parts else "General refinement needed"
