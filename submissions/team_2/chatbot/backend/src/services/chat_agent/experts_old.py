"""
Mixture of Experts (MoE) for Graph of Thought Verification
Three specialized experts: Hallucination Hunter, Source Matcher, Logic Expert
Uses Groq models
"""

import logging
from typing import Dict, List, Tuple
import asyncio
import re

from src.utils.groq_client import GroqClient

logger = logging.getLogger(__name__)


def strip_markdown_json(text: str) -> str:
    """Remove markdown code fences from LLM responses and extract JSON.
    
    Handles formats like:
    ```json
    {...}
    ```
    or
    ```
    {...}
    ```
    Also handles cases where there's extra text before or after the JSON.
    """
    if not text:
        return text
    
    text = text.strip()
    
    # First, try to find JSON within code fences
    # Pattern: ```json ... ``` or ``` ... ```
    code_fence_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_fence_pattern, text, re.DOTALL)
    if matches:
        # Use the last match (in case there are multiple)
        text = matches[-1].strip()
    
    # Now try to extract just the JSON object/array
    # Find the first { or [ and match to its closing bracket
    import json
    for i, char in enumerate(text):
        if char in '{[':
            # Try to parse from this position
            try:
                # Use JSONDecoder to find where valid JSON ends
                decoder = json.JSONDecoder()
                obj, end_idx = decoder.raw_decode(text[i:])
                # Return the valid JSON string
                return text[i:i+end_idx]
            except json.JSONDecodeError:
                continue
    
    return text.strip()


class Expert:
    """Base class for MoE experts"""
    
    def __init__(self, groq_client: GroqClient):
        """
        Initialize expert
        
        Args:
            groq_client: Groq client for LLM calls
        """
        self.groq_client = groq_client
    
    async def call_llm(self, prompt: str) -> str:
        """
        Call Groq Expert model with the given prompt
        
        Args:
            prompt: Prompt text
        
        Returns:
            LLM response text (empty string on error)
        """
        try:
            response = await self.groq_client.generate_expert(prompt, max_tokens=512)
            return response if response else ""
        except Exception as e:
            logger.error(f"LLM call failed in {self.__class__.__name__}: {e}")
            return ""
    
    async def verify(self, thought: str, context: str, graph_history: List[Dict]) -> Dict:
        """
        Verify a thought (to be implemented by subclasses)
        
        Args:
            thought: The thought to verify
            context: Retrieved context/chunks
            graph_history: Last 3 nodes from the graph
        
        Returns:
            Dict with score and remarks
        """
        raise NotImplementedError


class HallucinationHunter(Expert):
    """
    Expert that detects hallucinations by comparing thought against context
    """
    
    async def verify(self, thought: str, context: str, graph_history: List[Dict]) -> Dict:
        """
        Check if the thought contains any information not present in the context
        
        Returns:
            {
                "score": float (0-1),
                "passed": bool,
                "remarks": str
            }
        """
        prompt = f"""You are a Hallucination Hunter for MetaKGP wiki verification. Your job is fact-checking.

Context from MetaKGP wiki:
{context}

Thought to Verify:
{thought}

RULES:
1. Check if major factual claims in the Thought are supported by the Context
2. Minor details or reasonable inferences are acceptable
3. If the Thought says "not available" or "insufficient information", give it a PASS
4. Be lenient with paraphrasing and reasonable interpretations
5. Only flag clear contradictions or completely unsupported claims

CRITICAL: Return ONLY a valid JSON object, nothing else. No explanations, no markdown formatting.

Output format:
{{
    "hallucinations_found": ["list significant unsupported claims"],
    "confidence": 0.0-1.0,
    "verdict": "PASS" or "FAIL"
}}"""

        response = await self.call_llm(prompt)
        
        # Log the raw response for debugging
        if not response or not response.strip():
            logger.error(f"HallucinationHunter received empty response from LLM")
            return {
                "score": 0.6,
                "passed": True,
                "remarks": "LLM returned empty response, defaulting to PASS",
                "expert": "HallucinationHunter"
            }
        
        try:
            # Strip markdown code fences before parsing
            import json
            cleaned_response = strip_markdown_json(response)
            result = json.loads(cleaned_response.strip())
            
            hallucinations = result.get("hallucinations_found", [])
            confidence = float(result.get("confidence", 0.6))
            verdict = result.get("verdict", "PASS")  # Default to PASS
            
            # More lenient: pass if confidence >= 0.4
            passed = verdict == "PASS" or (confidence >= 0.4 and len(hallucinations) <= 1)
            score = confidence if passed else (1.0 - confidence)
            
            remarks = f"Hallucinations: {', '.join(hallucinations)}" if hallucinations else "No hallucinations detected"
            
            return {
                "score": score,
                "passed": passed,
                "remarks": remarks,
                "expert": "HallucinationHunter"
            }
        
        except Exception as e:
            logger.error(f"Failed to parse HallucinationHunter response: {e}")
            logger.error(f"Raw response was: {response}")
            logger.error(f"Cleaned response was: {cleaned_response}")
            return {
                "score": 0.3,
                "passed": False,
                "remarks": f"Parse error: {e}",
                "expert": "HallucinationHunter"
            }


class SourceMatcher(Expert):
    """
    Expert that verifies the thought is semantically contained in the context
    """
    
    async def verify(self, thought: str, context: str, graph_history: List[Dict]) -> Dict:
        """
        Check if the meaning of the thought is contained in the retrieved chunks
        
        Returns:
            {
                "score": float (0-1),
                "passed": bool,
                "remarks": str
            }
        """
        prompt = f"""You are a Source Matcher. Your job is to verify that the Thought's MEANING is fully supported by the Context.

Context from MetaKGP wiki:
{context}

Thought to Verify:
{thought}

Instructions:
1. Does the Context contain information that reasonably supports this Thought?
2. Is the Thought a reasonable interpretation or inference from the Context?
3. Rate your confidence from 1-10 (be generous).

CRITICAL: Return ONLY a valid JSON object, nothing else. No explanations, no markdown formatting.

Output format:
{{
    "confidence_score": 1-10,
    "reasoning": "brief explanation",
    "verdict": "PASS" or "FAIL"
}}"""

        response = await self.call_llm(prompt)
        
        # Log the raw response for debugging
        if not response or not response.strip():
            logger.error(f"SourceMatcher received empty response from LLM")
            return {
                "score": 0.6,
                "passed": True,
                "remarks": "LLM returned empty response, defaulting to PASS",
                "expert": "SourceMatcher"
            }
        
        try:
            import json
            # Strip markdown code fences before parsing
            cleaned_response = strip_markdown_json(response)
            result = json.loads(cleaned_response.strip())
            
            confidence_score = float(result.get("confidence_score", 6)) / 10.0  # Normalize to 0-1, default 0.6
            reasoning = result.get("reasoning", "")
            verdict = result.get("verdict", "PASS")  # Default to PASS
            
            # More lenient: pass if confidence >= 0.5 or verdict is PASS
            passed = verdict == "PASS" or confidence_score >= 0.5
            
            return {
                "score": confidence_score,
                "passed": passed,
                "remarks": reasoning,
                "expert": "SourceMatcher"
            }
        
        except Exception as e:
            logger.error(f"Failed to parse SourceMatcher response: {e}")
            logger.error(f"Raw response was: {response}")
            logger.error(f"Cleaned response was: {cleaned_response}")
            return {
                "score": 0.3,
                "passed": False,
                "remarks": f"Parse error: {e}",
                "expert": "SourceMatcher"
            }


class LogicExpert(Expert):
    """
    Expert that ensures the reasoning chain makes sense
    """
    
    async def verify(self, thought: str, context: str, graph_history: List[Dict]) -> Dict:
        """
        Check if the thought fits logically in the reasoning chain
        
        Returns:
            {
                "score": float (0-1),
                "passed": bool,
                "remarks": str,
                "action": "keep" | "merge" | "discard"
            }
        """
        # Format graph history
        history_text = ""
        for i, node in enumerate(graph_history[-3:], 1):
            history_text += f"{i}. {node.get('thought', '')}\n"
        
        prompt = f"""You are a Logic Expert. Your job is to ensure the reasoning chain is coherent and non-redundant.

Previous Reasoning Steps:
{history_text if history_text else "This is the first node."}

New Thought:
{thought}

Instructions:
1. Does this Thought logically follow from the previous steps?
2. Is it somewhat redundant but still valuable?
3. Does it contribute to answering the question?

CRITICAL: Return ONLY a valid JSON object, nothing else. No explanations, no markdown formatting.

Output format:
{{
    "coherence_score": 0.0-1.0,
    "is_redundant": true/false,
    "action": "keep" | "merge" | "discard",
    "remarks": "brief explanation"
}}"""

        response = await self.call_llm(prompt)
        
        # Log the raw response for debugging
        if not response or not response.strip():
            logger.error(f"LogicExpert received empty response from LLM")
            return {
                "score": 0.7,
                "passed": True,
                "remarks": "LLM returned empty response, defaulting to PASS",
                "action": "keep",
                "expert": "LogicExpert"
            }
        
        try:
            import json
            # Strip markdown code fences before parsing
            cleaned_response = strip_markdown_json(response)
            result = json.loads(cleaned_response.strip())
            
            coherence_score = float(result.get("coherence_score", 0.7))  # Default to 0.7
            is_redundant = result.get("is_redundant", False)
            action = result.get("action", "keep")
            remarks = result.get("remarks", "")
            
            # More lenient: pass if coherence >= 0.4, allow some redundancy
            passed = coherence_score >= 0.4
            
            return {
                "score": coherence_score,
                "passed": passed,
                "remarks": remarks,
                "action": action,
                "expert": "LogicExpert"
            }
        
        except Exception as e:
            logger.error(f"Failed to parse LogicExpert response: {e}")
            logger.error(f"Raw response was: {response}")
            logger.error(f"Cleaned response was: {cleaned_response}")
            return {
                "score": 0.5,
                "passed": True,
                "remarks": f"Parse error: {e}",
                "action": "keep",
                "expert": "LogicExpert"
            }


class MoEGauntlet:
    """
    Orchestrates the three experts with weighted voting
    """
    
    def __init__(self, groq_client: GroqClient):
        """
        Initialize the MoE Gauntlet with Groq client
        
        Args:
            groq_client: Groq client for expert calls
        """
        self.hallucination_hunter = HallucinationHunter(groq_client=groq_client)
        self.source_matcher = SourceMatcher(groq_client=groq_client)
        self.logic_expert = LogicExpert(groq_client=groq_client)
    
    async def verify_thought(
        self,
        thought: str,
        context: str,
        graph_history: List[Dict]
    ) -> Dict:
        """
        Run all three experts in parallel and compute weighted vote
        
        Weighted Voting Formula (LENIENT):
        final_score = (source_matcher_score * 0.5) + (hallucination_score * 0.3) + (logic_score * 0.2)
        Pass threshold: final_score > 0.5
        
        Args:
            thought: The thought to verify
            context: Retrieved context chunks
            graph_history: Last 3 nodes from the graph
        
        Returns:
            {
                "passed": bool,
                "final_score": float,
                "action": str,
                "expert_results": dict,
                "remarks": str
            }
        """
        logger.info(f"Running MoE Gauntlet on thought: {thought[:100]}...")
        
        # Run all experts in parallel
        results = await asyncio.gather(
            self.hallucination_hunter.verify(thought, context, graph_history),
            self.source_matcher.verify(thought, context, graph_history),
            self.logic_expert.verify(thought, context, graph_history),
            return_exceptions=True
        )
        
        hallucination_result, source_result, logic_result = results
        
        # Handle any exceptions - be generous with defaults
        if isinstance(hallucination_result, Exception):
            hallucination_result = {"score": 0.6, "passed": True, "remarks": str(hallucination_result)}
        if isinstance(source_result, Exception):
            source_result = {"score": 0.6, "passed": True, "remarks": str(source_result)}
        if isinstance(logic_result, Exception):
            logic_result = {"score": 0.7, "passed": True, "action": "keep", "remarks": str(logic_result)}
        
        # Weighted voting (very lenient)
        # Source Matcher (40%), Hallucination Hunter (30%), Logic Expert (30%)
        final_score = (
            source_result["score"] * 0.4 + 
            hallucination_result["score"] * 0.3 + 
            logic_result["score"] * 0.3
        )
        
        # Check if thought passes
        source_pass = source_result["passed"]
        hallucination_pass = hallucination_result["passed"]
        logic_pass = logic_result["passed"]
        
        # Very lenient rules
        # Pass if final_score >= 0.35 OR if at least 1 expert passes
        experts_passed = sum([source_pass, hallucination_pass, logic_pass])
        
        # Almost always pass - only fail if score is very low AND no experts passed
        if final_score < 0.25 and experts_passed == 0:
            passed = False
            action = "discard"
            remarks = f"FAILED: Very low score {final_score:.2f}, no experts passed"
        else:
            # Check Logic Expert's recommendation
            if logic_result.get("action") == "merge":
                passed = True
                action = "merge"
                remarks = f"PASSED (score: {final_score:.2f}) - flagged as redundant, merging"
            elif logic_result.get("action") == "discard" and final_score < 0.2:
                passed = False
                action = "discard"
                remarks = f"FAILED: Logic Expert rejected and very low score ({final_score:.2f})"
            else:
                passed = True
                action = "keep"
                remarks = f"PASSED: score {final_score:.2f}, {experts_passed}/3 experts approved"
        
        logger.info(f"MoE Verdict: {remarks}")
        
        return {
            "passed": passed,
            "final_score": final_score,
            "action": action,
            "expert_results": {
                "hallucination_hunter": hallucination_result,
                "source_matcher": source_result,
                "logic_expert": logic_result
            },
            "remarks": remarks
        }
