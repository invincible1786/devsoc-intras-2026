"""
Mixture of Experts (MoE) - Three Verification Experts
"""

import logging
import json
from typing import Tuple
import asyncio

logger = logging.getLogger(__name__)


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response that may contain markdown or extra text."""
    if not text or not text.strip():
        raise ValueError("Empty response text")
    
    text = text.strip()
    
    # Try to find JSON in code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
    
    # Find first { and parse from there
    json_start = text.find('{')
    if json_start != -1:
        # Try to find matching closing brace
        brace_count = 0
        for j in range(json_start, len(text)):
            if text[j] == '{':
                brace_count += 1
            elif text[j] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found complete JSON object
                    try:
                        return json.loads(text[json_start:j+1])
                    except json.JSONDecodeError:
                        # Try to continue searching
                        pass
    
    # Last resort: try to parse the whole thing
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Return a default error response instead of raising
        logger.warning(f"Could not extract JSON from response: {e}\nText: {text[:200]}")
        return {
            "verdict": "NO",
            "confidence": 0.0,
            "reasoning": f"JSON parsing error: {str(e)[:100]}"
        }


async def source_matcher(claim: str, context: str, groq_client) -> Tuple[bool, float, str]:
    """
    Expert 1: Verify if claim is directly supported by context.
    
    Args:
        claim: The claim to verify
        context: Source text
        groq_client: Groq client
        
    Returns:
        (verdict, confidence, reasoning)
    """
    prompt = f"""You are a Source Matcher. Verify if a claim is supported by context.

CLAIM:
{claim}

SOURCE TEXT:
{context}

TASK: Does the SOURCE TEXT explicitly support this CLAIM?

EVALUATION:
- Check if key facts (names, roles, dates, numbers) in the claim are present in context
- The claim must be DIRECTLY supported, not inferred
- Be STRICT: if something isn't explicit, mark as unsupported

Return ONLY valid JSON:
{{
    "verdict": "YES" or "NO",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}"""

    try:
        response = await groq_client.generate_expert(prompt, max_tokens=512)
        result = extract_json_from_response(response)
        
        verdict = result.get("verdict", "NO") == "YES"
        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("reasoning", "")
        
        return verdict, confidence, reasoning
        
    except Exception as e:
        logger.error(f"Source Matcher error: {e}")
        return False, 0.0, f"Error: {e}"


async def hallucination_hunter(claim: str, context: str, original_query: str, groq_client) -> Tuple[bool, float, str]:
    """
    Expert 2: Detect if the claim invents details not in context.
    
    Args:
        claim: The claim to verify
        context: Source text
        original_query: Original user query
        groq_client: Groq client
        
    Returns:
        (is_hallucinating, confidence, invented_details)
    """
    prompt = f"""You are a Hallucination Hunter. Detect invented information.

ORIGINAL QUERY:
{original_query}

CLAIM/RESPONSE:
{claim}

SCRAPED CONTEXT:
{context}

TASK: Is the CLAIM inventing details NOT present in the context?

WHAT COUNTS AS HALLUCINATION:
- Names not mentioned in context
- Dates or numbers not in context
- Events or facts completely fabricated
- Roles or positions invented

WHAT IS ACCEPTABLE:
- Direct quotes or paraphrases from context
- Information explicitly stated in context

Return ONLY valid JSON:
{{
    "is_hallucinating": true or false,
    "confidence": 0.0 to 1.0,
    "invented_details": "list specific invented details or 'None'"
}}"""

    try:
        response = await groq_client.generate_expert(prompt, max_tokens=512)
        result = extract_json_from_response(response)
        
        is_hallucinating = result.get("is_hallucinating", True)
        confidence = float(result.get("confidence", 0.5))
        details = result.get("invented_details", "Unknown")
        
        return is_hallucinating, confidence, details
        
    except Exception as e:
        logger.error(f"Hallucination Hunter error: {e}")
        return True, 0.5, f"Error: {e}"


async def logic_expert(claim: str, context: str, question: str, groq_client) -> Tuple[bool, float, str]:
    """
    Expert 3: Verify if conclusion logically follows from premises.
    
    Args:
        claim: The conclusion
        context: The premises
        question: The question being answered
        groq_client: Groq client
        
    Returns:
        (is_logical, confidence, reasoning)
    """
    prompt = f"""You are a Logic Expert. Verify logical reasoning.

QUESTION:
{question}

PREMISES (from context):
{context}

CONCLUSION (bot's claim):
{claim}

TASK: Does the CONCLUSION logically follow from the PREMISES?

EVALUATION:
1. Are premises clearly stated?
2. Does conclusion follow logically?
3. Any logical fallacies or leaps?

Return ONLY valid JSON:
{{
    "is_logical": true or false,
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}"""

    try:
        response = await groq_client.generate_expert(prompt, max_tokens=512)
        result = extract_json_from_response(response)
        
        is_logical = result.get("is_logical", False)
        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("reasoning", "")
        
        return is_logical, confidence, reasoning
        
    except Exception as e:
        logger.error(f"Logic Expert error: {e}")
        return False, 0.0, f"Error: {e}"
