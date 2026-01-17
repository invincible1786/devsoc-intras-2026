"""
Graph of Thoughts + Mixture of Experts Implementation
Complete implementation following the IMPLEMENTATION_GUIDE.md

This module implements:
1. Graph of Thoughts (GoT) - Multi-step reasoning with sub-question decomposition
2. Mixture of Experts (MoE) - Three specialized verification agents
3. Dynamic Knowledge Graph - NetworkX graph for learned relationships
4. RAG with filtering - Top 45 chunks filtered by relevance
"""

import os
import json
import asyncio
import logging
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import networkx as nx
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger_init = logging.getLogger(__name__)
    logger_init.info(f"Loaded environment variables from {env_path}")
else:
    logger_init = logging.getLogger(__name__)
    logger_init.warning(f".env file not found at {env_path}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ReasoningPath:
    """
    Represents a single reasoning trajectory/hypothesis.
    Multiple paths compete, and MoE selects the best one.
    """
    path_id: int
    claim: str
    context: str
    source_info: str = ""
    
    # Expert evaluations (populated by MoE)
    source_match_verdict: Optional[bool] = None
    source_match_conf: float = 0.0
    source_match_reasoning: str = ""
    
    halluc_verdict: Optional[bool] = None  # True = is hallucinating
    halluc_conf: float = 0.0
    halluc_details: str = ""
    
    logic_verdict: Optional[bool] = None
    logic_conf: float = 0.0
    logic_reasoning: str = ""
    
    # Final verdict
    is_verified: bool = False
    final_score: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class ThoughtNode:
    """
    Represents a single thought/sub-question in the Graph of Thoughts.
    Contains multiple reasoning paths that compete via MoE.
    """
    id: int
    question: str  # The sub-question
    retrieved_context: str = ""  # RAG context from vector DB
    derived_thought: str = ""  # The final answer (from best path)
    verified: bool = False  # Passed MoE verification?
    score: int = 0  # Confidence score (0-10)
    reasoning_paths: List[ReasoningPath] = field(default_factory=list)


# ============================================================================
# STEP 1: PLANNER AGENT (Query Decomposition)
# ============================================================================

def planner_agent(query: str, groq_client) -> List[str]:
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
        response = asyncio.run(groq_client.generate_judge(
            system_prompt + "\n\n" + prompt,
            max_tokens=512
        ))
        
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
# STEP 2: EXECUTION AGENT (Multi-Path Generation)
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
        import numpy as np
        
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


def execution_agent(
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
            n_results=45
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

        path1_response = asyncio.run(groq_client.generate_judge(path1_prompt, max_tokens=512))
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

            path2_response = asyncio.run(groq_client.generate_judge(path2_prompt, max_tokens=512))
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

            path3_response = asyncio.run(groq_client.generate_judge(path3_prompt, max_tokens=512))
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
# STEP 3: MIXTURE OF EXPERTS (MoE) - VERIFICATION
# ============================================================================

def extract_json_from_response(text: str) -> Dict:
    """Extract JSON from LLM response that may contain markdown."""
    import re
    
    if not text or not text.strip():
        raise ValueError("Empty response text")
    
    text = text.strip()
    
    # Try to find JSON in code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    text = text.strip()
    
    # Find first { and parse from there
    for i, char in enumerate(text):
        if char == '{':
            try:
                # Try to find matching closing brace
                brace_count = 0
                for j in range(i, len(text)):
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete JSON object
                            return json.loads(text[i:j+1])
            except json.JSONDecodeError:
                continue
    
    # Last resort: try to parse the whole thing
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not extract JSON from response: {e}\nText: {text[:200]}")


def source_matcher(claim: str, context: str, groq_client) -> Tuple[bool, float, str]:
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
        response = asyncio.run(groq_client.generate_expert(prompt, max_tokens=256))
        result = extract_json_from_response(response)
        
        verdict = result.get("verdict", "NO") == "YES"
        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("reasoning", "")
        
        return verdict, confidence, reasoning
        
    except Exception as e:
        logger.error(f"Source Matcher error: {e}")
        return False, 0.0, f"Error: {e}"


def hallucination_hunter(claim: str, context: str, original_query: str, groq_client) -> Tuple[bool, float, str]:
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
        response = asyncio.run(groq_client.generate_expert(prompt, max_tokens=256))
        result = extract_json_from_response(response)
        
        is_hallucinating = result.get("is_hallucinating", True)
        confidence = float(result.get("confidence", 0.5))
        details = result.get("invented_details", "Unknown")
        
        return is_hallucinating, confidence, details
        
    except Exception as e:
        logger.error(f"Hallucination Hunter error: {e}")
        return True, 0.5, f"Error: {e}"


def logic_expert(claim: str, context: str, question: str, groq_client) -> Tuple[bool, float, str]:
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
        response = asyncio.run(groq_client.generate_expert(prompt, max_tokens=256))
        result = extract_json_from_response(response)
        
        is_logical = result.get("is_logical", False)
        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("reasoning", "")
        
        return is_logical, confidence, reasoning
        
    except Exception as e:
        logger.error(f"Logic Expert error: {e}")
        return False, 0.0, f"Error: {e}"


def evaluate_reasoning_path(
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
    
    # Run all three experts
    path.source_match_verdict, path.source_match_conf, path.source_match_reasoning = \
        source_matcher(path.claim, path.context, groq_client)
    
    path.halluc_verdict, path.halluc_conf, path.halluc_details = \
        hallucination_hunter(path.claim, path.context, original_query, groq_client)
    
    path.logic_verdict, path.logic_conf, path.logic_reasoning = \
        logic_expert(path.claim, path.context, question, groq_client)
    
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


def verification_agent(node: ThoughtNode, original_query: str, groq_client) -> ThoughtNode:
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
    
    # Evaluate all paths
    for path in node.reasoning_paths:
        evaluate_reasoning_path(path, original_query, node.question, groq_client)
    
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
# STEP 4: SYNTHESIS AGENT (Combine Verified Thoughts)
# ============================================================================

def synthesis_agent(original_query: str, nodes: List[ThoughtNode], groq_client) -> str:
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
        response = asyncio.run(groq_client.generate_judge(prompt, max_tokens=1024))
        final_answer = response.strip()
        
        logger.info(f"Synthesis complete: {len(final_answer)} chars")
        return final_answer
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        # Fallback: just concatenate answers
        return "\n\n".join([f"{fact['answer']} (Source: {fact['source']})" for fact in verified_facts])


# ============================================================================
# STEP 5: GRAPH LEARNING (Dynamic Knowledge Graph)
# ============================================================================

class KnowledgeGraph:
    """Persistent NetworkX graph for learned relationships."""
    
    def __init__(self, graph_file: str = "./cache/metakgp_graph.gml"):
        self.graph_file = Path(graph_file)
        self.graph_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or create graph
        if self.graph_file.exists():
            try:
                self.G = nx.read_gml(str(self.graph_file))
                logger.info(f"Loaded graph with {len(self.G.nodes())} nodes")
            except:
                self.G = nx.DiGraph()
                logger.info("Created new graph")
        else:
            self.G = nx.DiGraph()
            logger.info("Created new graph")
    
    def add_verified_thought(self, node: ThoughtNode, previous_node: Optional[ThoughtNode] = None):
        """Add verified thought to the graph."""
        if not node.verified or "I don't know" in node.derived_thought.lower():
            return
        
        # Add node to graph
        self.G.add_node(
            node.question,
            label="Thought",
            answer=node.derived_thought,
            score=node.score,
            timestamp=datetime.now().isoformat()
        )
        
        # Link to previous node (chain of thought)
        if previous_node and previous_node.question in self.G:
            self.G.add_edge(
                previous_node.question,
                node.question,
                relation="leads_to"
            )
        
        logger.info(f"Added node to graph: {node.question[:50]}...")
    
    def save(self):
        """Persist graph to disk."""
        try:
            nx.write_gml(self.G, str(self.graph_file))
            logger.info(f"Saved graph with {len(self.G.nodes())} nodes")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
    
    def get_context(self, query: str, max_nodes: int = 5) -> str:
        """Retrieve relevant context from graph."""
        if not self.G.nodes():
            return ""
        
        # Simple keyword matching (can be enhanced with embeddings)
        query_words = set(query.lower().split())
        
        relevant_nodes = []
        for node, data in self.G.nodes(data=True):
            node_words = set(node.lower().split())
            overlap = len(query_words & node_words)
            if overlap > 0:
                relevant_nodes.append((node, data, overlap))
        
        # Sort by overlap
        relevant_nodes.sort(key=lambda x: x[2], reverse=True)
        
        # Format context
        context_parts = []
        for node, data, _ in relevant_nodes[:max_nodes]:
            context_parts.append(f"Q: {node}\nA: {data.get('answer', 'N/A')}")
        
        return "\n\n".join(context_parts)


# ============================================================================
# MAIN ORCHESTRATION: Graph of Thoughts Pipeline
# ============================================================================

def generate_response_got(
    query: str,
    chroma_client,
    embedding_client,
    groq_client,
    graph: Optional[KnowledgeGraph] = None
) -> str:
    """
    Main Graph of Thoughts + MoE pipeline.
    
    Pipeline:
    1. Planner: Decompose query into sub-questions
    2. Execution: Generate multiple reasoning paths for each sub-question
    3. Verification: MoE evaluates all paths and selects best
    4. Synthesis: Combine verified thoughts into final answer
    5. Graph Learning: Store verified thoughts in knowledge graph
    
    Args:
        query: User's query
        chroma_client: ChromaDB client
        embedding_client: Embedding client
        groq_client: Groq client
        graph: Knowledge graph (optional)
        
    Returns:
        Final answer
    """
    logger.info(f"Processing query: {query}")
    
    # Initialize graph if not provided
    if graph is None:
        graph = KnowledgeGraph()
    
    # Step 1: Plan (decompose query)
    plan = planner_agent(query, groq_client)
    logger.info(f"Plan: {plan}")
    
    # Step 2 & 3: Execute and Verify each sub-question
    nodes = []
    for i, sub_question in enumerate(plan):
        # Create node
        node = ThoughtNode(id=i, question=sub_question)
        
        # Execute (generate reasoning paths)
        node = execution_agent(node, chroma_client, embedding_client, groq_client)
        
        # Verify (MoE)
        node = verification_agent(node, query, groq_client)
        
        nodes.append(node)
        
        # Update graph
        previous_node = nodes[i-1] if i > 0 else None
        graph.add_verified_thought(node, previous_node)
    
    # Save graph
    graph.save()
    
    # Step 4: Synthesize final answer
    final_answer = synthesis_agent(query, nodes, groq_client)
    
    return final_answer


# ============================================================================
# TESTING
# ============================================================================

def test_got_moe():
    """Test the complete GoT + MoE pipeline."""
    
    # Import clients
    from src.utils.chroma_client import MetaKGPChromaClient
    from src.utils.embedding_client import ModalEmbeddingClient
    from src.utils.groq_client import GroqClient
    
    # Initialize clients
    logger.info("Initializing clients...")
    chroma_client = MetaKGPChromaClient()
    embedding_client = ModalEmbeddingClient()
    groq_client = GroqClient()
    graph = KnowledgeGraph()
    
    # Test queries
    test_queries = [
        "Who is the Vice President of Technology Film and Photography Society?",
        "When was TFPS founded?",
        "List the current General Secretaries of TSG for 2025",
    ]
    
    for query in test_queries:
        logger.info(f"\n{'='*80}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*80}\n")
        
        answer = generate_response_got(
            query,
            chroma_client,
            embedding_client,
            groq_client,
            graph
        )
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Final Answer:\n{answer}")
        logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    test_got_moe()
