"""
Graph of Thoughts Agents: Planner, Execution, Verification, Synthesis
"""

import logging
import json
import asyncio
from typing import List
import numpy as np

from src.services.chat_agent.data_structures import ThoughtNode, ReasoningPath
from src.services.chat_agent.experts import source_matcher, hallucination_hunter, logic_expert

logger = logging.getLogger(__name__)


# ============================================================================
# PLANNER AGENT
# ============================================================================

async def planner_agent(query: str, groq_client) -> List[str]:
    """
    Decompose complex query into simple, searchable sub-questions.
    
    Key responsibilities:
    - Break compound queries into atomic sub-questions
    - Expand acronyms (e.g., TFPS → Technology Film and Photography Society)
    - Ensure each sub-question is independently answerable
    
    Args:
        query: User's original query
        groq_client: Groq client for LLM calls
        
    Returns:
        List of sub-questions
    """
    system_prompt = """You are a Query Planner for MetaKGP wiki (IIT Kharagpur knowledge base).

TASK: Break the user's query into simple, atomic sub-questions.

CRITICAL RULES:
1. Expand ALL acronyms to full names:
   - TFPS → "Technology Film and Photography Society"
   - TSG → "Technology Students' Gymkhana"
   - VP → "Vice President"
   - Gen Sec → "General Secretary"
   
2. Each sub-question must be independently searchable
3. For "current" questions, explicitly mention "2025" or "current"
4. Keep sub-questions simple and direct

EXAMPLES:

Query: "Who is VP of TFPS?"
Output: ["Who is the Vice President of Technology Film and Photography Society?"]

Query: "Who is VP of TFPS and when was it founded?"
Output: [
    "Who is the Vice President of Technology Film and Photography Society?",
    "When was Technology Film and Photography Society founded?"
]

Query: "List current Gen Secs of TSG"
Output: ["Who are the current General Secretaries of Technology Students' Gymkhana in 2025?"]

Return ONLY a valid JSON list of strings, nothing else."""

    prompt = f"""User Query: {query}

Break this into sub-questions following the rules. Return ONLY JSON list."""

    try:
        response = await groq_client.generate_judge(
            system_prompt + "\n\n" + prompt,
            max_tokens=512
        )
        
        # Extract JSON from response
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        plan = json.loads(response)
        
        if not isinstance(plan, list):
            logger.warning(f"Planner returned non-list: {plan}")
            return [query]
        
        logger.info(f"Planner decomposed query into {len(plan)} sub-questions")
        return plan
        
    except Exception as e:
        logger.error(f"Planner error: {e}")
        # Fallback: return original query
        return [query]


# ============================================================================
# EXECUTION AGENT
# ============================================================================

def filter_chunks_by_relevance(docs: List, query: str, embedding_client, top_k: int = 10) -> List:
    """
    Filter retrieved chunks by semantic relevance to the query.
    
    Args:
        docs: List of retrieved documents
        query: The sub-question
        embedding_client: Embedding client
        top_k: Number of chunks to keep
        
    Returns:
        Filtered list of most relevant documents
    """
    try:
        if not docs:
            return []
        
        # Get query embedding
        query_embedding = embedding_client(query)
        if not query_embedding:
            return docs[:top_k]
        
        # Score each document by cosine similarity
        scored_docs = []
        for doc in docs:
            # Get document embedding
            doc_embedding = embedding_client(doc.page_content)
            if doc_embedding:
                # Cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                scored_docs.append((doc, similarity))
        
        # Sort by similarity (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        filtered = [doc for doc, score in scored_docs[:top_k]]
        
        logger.info(f"Filtered {len(docs)} chunks to {len(filtered)} most relevant")
        return filtered
        
    except Exception as e:
        logger.error(f"Error filtering chunks: {e}")
        return docs[:top_k]


async def execution_agent(
    node: ThoughtNode,
    chroma_client,
    embedding_client,
    groq_client
) -> ThoughtNode:
    """
    Generate multiple reasoning paths from different contexts.
    
    Strategy:
    1. Retrieve top 45 chunks from vector DB
    2. Filter to top 10 most relevant chunks
    3. Generate 3 competing reasoning paths:
       - Path 1: Primary source (most relevant chunk)
       - Path 2: Multi-source synthesis (top 3 chunks)
       - Path 3: Temporal filter (if asking about "current")
    
    Args:
        node: ThoughtNode with the sub-question
        chroma_client: ChromaDB client
        embedding_client: Embedding client
        groq_client: Groq client
        
    Returns:
        Updated ThoughtNode with reasoning paths
    """
    logger.info(f"Executing sub-question: {node.question}")
    
    try:
        # Retrieve top 45 chunks
        query_embedding = embedding_client(node.question)
        if not query_embedding:
            logger.error("Failed to generate embedding for query")
            return node
        
        results = chroma_client.collection.query(
            query_embeddings=[query_embedding],
            n_results=30
        )
        
        if not results or not results["documents"] or not results["documents"][0]:
            logger.warning(f"No documents retrieved for: {node.question}")
            return node
        
        # Convert to document objects
        docs = []
        for i, (doc_text, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            docs.append(type('Doc', (), {
                'page_content': doc_text,
                'metadata': metadata
            })())
        
        logger.info(f"Retrieved {len(docs)} chunks from ChromaDB")
        
        # Filter to top 10 most relevant
        filtered_docs = filter_chunks_by_relevance(docs, node.question, embedding_client, top_k=10)
        
        if not filtered_docs:
            logger.warning("No relevant chunks after filtering")
            return node
        
        # Store all context for node
        node.retrieved_context = "\n\n".join([
            f"Source {i+1}: {doc.page_content}" 
            for i, doc in enumerate(filtered_docs[:5])
        ])
        
        # ========== PATH 1: Primary Source ==========
        primary_context = filtered_docs[0].page_content
        primary_source = filtered_docs[0].metadata.get("source_page", "Unknown")
        
        path1_prompt = f"""Answer this question using ONLY the provided context.

QUESTION: {node.question}

CONTEXT:
{primary_context}

RULES:
- Answer directly and concisely
- Only use information from the context
- If the answer is not in the context, say "I don't know"
- Cite the source in your answer

Answer:"""

        path1_response = await groq_client.generate_judge(path1_prompt, max_tokens=512)
        path1_claim = path1_response.strip()
        
        node.reasoning_paths.append(ReasoningPath(
            path_id=0,
            claim=path1_claim,
            context=primary_context,
            source_info=f"Primary Source: {primary_source}"
        ))
        
        logger.info(f"Generated Path 1 (Primary Source)")
        
        # ========== PATH 2: Multi-Source Synthesis ==========
        if len(filtered_docs) >= 3:
            multi_context = "\n\n---\n\n".join([
                f"Source {i+1} ({doc.metadata.get('source_page', 'Unknown')}):\n{doc.page_content}"
                for i, doc in enumerate(filtered_docs[:3])
            ])
            
            path2_prompt = f"""Synthesize an answer from multiple sources.

QUESTION: {node.question}

SOURCES:
{multi_context}

RULES:
- Combine information from all sources
- Only use information explicitly stated
- If sources conflict, mention both viewpoints
- Cite which sources you used

Answer:"""

            path2_response = await groq_client.generate_judge(path2_prompt, max_tokens=512)
            path2_claim = path2_response.strip()
            
            node.reasoning_paths.append(ReasoningPath(
                path_id=1,
                claim=path2_claim,
                context=multi_context,
                source_info="Multi-Source Synthesis"
            ))
            
            logger.info(f"Generated Path 2 (Multi-Source)")
        
        # ========== PATH 3: Temporal Filter (if asking about "current") ==========
        if "current" in node.question.lower() or "2025" in node.question:
            all_context = "\n\n".join([doc.page_content for doc in filtered_docs[:5]])
            
            path3_prompt = f"""Extract ONLY current (2025) information.

QUESTION: {node.question}

CONTEXT:
{all_context}

RULES:
- Look for phrases like "2024-25", "2025", "current"
- Ignore outdated information (2023, 2022, etc.)
- If no current information found, say "No current information available"

Answer:"""

            path3_response = await groq_client.generate_judge(path3_prompt, max_tokens=512)
            path3_claim = path3_response.strip()
            
            node.reasoning_paths.append(ReasoningPath(
                path_id=2,
                claim=path3_claim,
                context=all_context,
                source_info="Temporal Filter (2025)"
            ))
            
            logger.info(f"Generated Path 3 (Temporal Filter)")
        
        return node
        
    except Exception as e:
        logger.error(f"Execution agent error: {e}")
        return node


# ============================================================================
# VERIFICATION AGENT (MoE)
# ============================================================================

async def evaluate_reasoning_path(
    path: ReasoningPath,
    original_query: str,
    question: str,
    groq_client
) -> ReasoningPath:
    """
    Run all three experts on a reasoning path and compute consensus score.
    
    Args:
        path: ReasoningPath to evaluate
        original_query: Original user query
        question: Sub-question being answered
        groq_client: Groq client
        
    Returns:
        Updated ReasoningPath with expert verdicts
    """
    logger.info(f"Evaluating Path {path.path_id}: {path.source_info}")
    
    # Run all three experts in parallel
    results = await asyncio.gather(
        source_matcher(path.claim, path.context, groq_client),
        hallucination_hunter(path.claim, path.context, original_query, groq_client),
        logic_expert(path.claim, path.context, question, groq_client)
    )
    
    # Unpack results
    path.source_match_verdict, path.source_match_conf, path.source_match_reasoning = results[0]
    path.halluc_verdict, path.halluc_conf, path.halluc_details = results[1]
    path.logic_verdict, path.logic_conf, path.logic_reasoning = results[2]
    
    # Calculate consensus score (average confidence)
    path.final_score = (
        path.source_match_conf + 
        path.halluc_conf + 
        path.logic_conf
    ) / 3
    
    # Check if all experts agree (must pass all three)
    path.is_verified = (
        path.source_match_verdict and          # Source found
        (not path.halluc_verdict) and          # No hallucination
        path.logic_verdict and                 # Logically sound
        path.final_score > 0.6                 # High confidence threshold
    )
    
    # Track failure reasons
    if not path.is_verified:
        if not path.source_match_verdict:
            path.failure_reasons.append(f"Source Matcher: {path.source_match_reasoning}")
        if path.halluc_verdict:
            path.failure_reasons.append(f"Hallucination: {path.halluc_details}")
        if not path.logic_verdict:
            path.failure_reasons.append(f"Logic: {path.logic_reasoning}")
    
    # Log expert verdicts
    logger.info(f"  Source Matcher: {'✓' if path.source_match_verdict else '✗'} ({path.source_match_conf:.2f})")
    logger.info(f"  Hallucination: {'✓' if not path.halluc_verdict else '✗'} ({path.halluc_conf:.2f})")
    logger.info(f"  Logic: {'✓' if path.logic_verdict else '✗'} ({path.logic_conf:.2f})")
    logger.info(f"  VERDICT: {'VERIFIED ✓' if path.is_verified else 'REJECTED ✗'} (score: {path.final_score:.2f})")
    
    return path


async def verification_agent(node: ThoughtNode, original_query: str, groq_client) -> ThoughtNode:
    """
    Run MoE verification on all reasoning paths and select the best one.
    
    Args:
        node: ThoughtNode with reasoning paths
        original_query: Original user query
        groq_client: Groq client
        
    Returns:
        Updated ThoughtNode with verified thought
    """
    logger.info(f"Verifying {len(node.reasoning_paths)} reasoning paths for: {node.question}")
    
    if not node.reasoning_paths:
        logger.warning("No reasoning paths to verify")
        return node
    
    # Evaluate all paths in parallel
    evaluated_paths = await asyncio.gather(*[
        evaluate_reasoning_path(path, original_query, node.question, groq_client)
        for path in node.reasoning_paths
    ])
    
    node.reasoning_paths = evaluated_paths
    
    # Rank paths: verified first, then by score
    ranked_paths = sorted(
        node.reasoning_paths,
        key=lambda p: (p.is_verified, p.final_score),
        reverse=True
    )
    
    # Select best path
    best_path = ranked_paths[0]
    node.derived_thought = best_path.claim
    node.verified = best_path.is_verified
    node.score = int(best_path.final_score * 10)
    
    logger.info(f"Selected Path {best_path.path_id}: {best_path.source_info}")
    logger.info(f"  Verified: {node.verified}, Score: {node.score}/10")
    
    return node


# ============================================================================
# SYNTHESIS AGENT
# ============================================================================

async def synthesis_agent(original_query: str, nodes: List[ThoughtNode], groq_client) -> str:
    """
    Combine verified thoughts into a coherent final answer.
    
    Args:
        original_query: Original user query
        nodes: List of ThoughtNodes with verified thoughts
        groq_client: Groq client
        
    Returns:
        Final synthesized answer
    """
    logger.info(f"Synthesizing answer from {len(nodes)} nodes")
    
    # Extract verified facts
    verified_facts = []
    for i, node in enumerate(nodes):
        if node.verified and "I don't know" not in node.derived_thought.lower():
            # Find the verified path
            verified_path = [p for p in node.reasoning_paths if p.is_verified]
            if verified_path:
                verified_facts.append({
                    "question": node.question,
                    "answer": node.derived_thought,
                    "source": verified_path[0].source_info,
                    "confidence": node.score
                })
    
    if not verified_facts:
        return "I couldn't find verified information to answer your question. Please try rephrasing or asking something else."
    
    # Format facts for synthesis
    facts_text = "\n\n".join([
        f"Sub-Question {i+1}: {fact['question']}\n"
        f"Answer: {fact['answer']}\n"
        f"Source: {fact['source']}\n"
        f"Confidence: {fact['confidence']}/10"
        for i, fact in enumerate(verified_facts)
    ])
    
    prompt = f"""You are synthesizing a final answer from verified facts.

ORIGINAL USER QUERY:
{original_query}

VERIFIED FACTS:
{facts_text}

TASK: Create a coherent, natural answer that addresses the user's query.

RULES:
1. Combine all verified facts naturally
2. Include inline citations: (Source: XYZ)
3. If multiple sub-questions, organize logically
4. Keep it concise and direct
5. Only use information from verified facts

Final Answer:"""

    try:
        response = await groq_client.generate_judge(prompt, max_tokens=1024)
        final_answer = response.strip()
        
        logger.info(f"Synthesis complete: {len(final_answer)} chars")
        return final_answer
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        # Fallback: just concatenate answers
        return "\n\n".join([f"{fact['answer']} (Source: {fact['source']})" for fact in verified_facts])
