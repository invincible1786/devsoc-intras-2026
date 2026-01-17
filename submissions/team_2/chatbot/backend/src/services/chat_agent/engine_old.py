"""
Simplified Graph of Thought (GoT) Engine with Llama-3.3-70b
Architecture: Query -> Iterative RAG (up to 3 calls) -> GoT on chunks -> Single MoE round -> Final answer
"""

import logging
import asyncio
from typing import Dict, List, Optional
import json
import httpx

from src.services.chat_agent.experts import MoEGauntlet, strip_markdown_json
from src.utils.embedding_client import ModalEmbeddingClient
from src.utils.groq_client import GroqClient

logger = logging.getLogger(__name__)


class SimplifiedGoTEngine:
    """
    Simplified Graph of Thought reasoning engine with Iterative RAG
    Process: Query -> Iterative RAG (max 3 queries) -> Analyze chunks -> Single MoE verification -> Final answer
    """
    
    def __init__(
        self,
        modal_url: str,
        groq_api_key: str,
        query_api_url: str = "http://localhost:8000/query/search",
        top_k: int = 30
    ):
        """
        Initialize the simplified GoT Engine
        
        Args:
            modal_url: Modal embedding service URL
            groq_api_key: Groq API key
            query_api_url: URL for the RAG query API
            top_k: Number of chunks to retrieve from RAG
        """
        self.modal_url = modal_url
        self.groq_api_key = groq_api_key
        self.query_api_url = query_api_url
        self.top_k = top_k
        
        # Initialize components
        self.embedding_client = ModalEmbeddingClient(modal_url)
        self.groq_client = GroqClient(groq_api_key)
        self.moe_gauntlet = MoEGauntlet(self.groq_client)
        
        # HTTP client for API calls
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        logger.info(f"SimplifiedGoTEngine initialized with top_k={top_k}")
    
    async def query_rag(self, query: str, top_k: Optional[int] = None) -> Dict:
        """
        Query the RAG API to get relevant context
        
        Args:
            query: Search query
            top_k: Number of results to retrieve (uses self.top_k if not specified)
        
        Returns:
            Dict with results
        """
        try:
            k = top_k if top_k is not None else self.top_k
            logger.info(f"Querying RAG: '{query}' with top_k={k}")
            response = await self.http_client.post(
                self.query_api_url,
                json={"query": query, "top_k": k}
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {"results": [], "error": str(e)}
    
    async def call_llm(self, prompt: str, max_tokens: int = 3072) -> str:
        """
        Call Llama-3.3-70b model
        
        Args:
            prompt: Prompt text
            max_tokens: Maximum tokens
        
        Returns:
            LLM response
        """
        try:
            return await self.groq_client.generate_judge(prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""
    
    async def check_relevance(self, query: str) -> Dict:
        """
        Check if the query is related to IIT Kharagpur/MetaKGP
        
        Returns:
            {
                "is_relevant": bool,
                "reasoning": str
            }
        """
        prompt = f"""You are a relevance checker for MetaKGP Wiki (IIT Kharagpur information system).

Question: {query}

Determine if this question could be related to IIT Kharagpur. Be LENIENT:
- Academic terms (GPA, CGPA, grades like "2.2", courses, departments)
- Campus life, facilities, hostels, societies, events
- Abbreviations or short queries that might refer to IIT KGP specific things
- Technical terms that could relate to academics or campus activities

Only mark as irrelevant if it's CLEARLY about something else (e.g., "weather in New York", "recipe for pasta", "movie recommendations").

When in doubt, mark as RELEVANT. Short or ambiguous queries should be marked RELEVANT.

CRITICAL: Return ONLY a valid JSON object, nothing else. No explanations, no markdown formatting.

Output format:
{{
    "is_relevant": true/false,
    "reasoning": "brief explanation"
}}"""
        
        response = await self.call_llm(prompt, max_tokens=256)
        
        try:
            cleaned_response = strip_markdown_json(response)
            result = json.loads(cleaned_response)
            return result
        except:
            # Default to relevant if parsing fails
            return {"is_relevant": True, "reasoning": "Parse error, proceeding"}
    
    async def generate_followup_queries(self, original_query: str, initial_chunks: List[Dict]) -> List[str]:
        """
        Generate follow-up queries based on initial RAG results
        
        Args:
            original_query: The original user query
            initial_chunks: Chunks retrieved from the first RAG call
        
        Returns:
            List of follow-up queries (max 2)
        """
        # Analyze initial chunks to understand what information we got
        chunk_summaries = []
        for i, chunk in enumerate(initial_chunks[:5], 1):
            chunk_summaries.append(f"{i}. {chunk['metadata']['title']}: {chunk['text'][:150]}...")
        
        chunks_preview = "\n".join(chunk_summaries)
        
        prompt = f"""You are analyzing search results to generate follow-up queries for deeper information gathering.

Original Query: {original_query}

Initial Retrieved Information (top 5 chunks):
{chunks_preview}

TASK: Generate 1-2 follow-up queries to gather MORE SPECIFIC and DETAILED information.

Guidelines:
1. If the original query is about events/activities, generate queries about specific event names or details mentioned
2. If it's about a society/club, query for specific initiatives, projects, or achievements
3. If it's about a person, query for their role, contributions, or associated events
4. Make queries more specific than the original (e.g., "events by SWG" → "SWG Hacknight details", "SWG workshops 2023")
5. Use information from the chunks to create targeted queries
6. If chunks already provide comprehensive info, return empty list []

CRITICAL: Return ONLY a valid JSON object, nothing else. No explanations, no markdown formatting.

Output format:
{{
    "followup_queries": ["query 1", "query 2"],
    "reasoning": "why these queries help"
}}"""
        
        try:
            response = await self.call_llm(prompt, max_tokens=512)
            
            # Check if response is empty
            if not response or not response.strip():
                logger.warning("Empty response from LLM for follow-up query generation")
                return []
            
            # Strip markdown and extract JSON
            cleaned_response = strip_markdown_json(response)
            
            if not cleaned_response:
                logger.warning("No valid JSON found in follow-up query response")
                return []
            
            result = json.loads(cleaned_response)
            queries = result.get("followup_queries", [])
            
            # Validate queries are strings
            if not isinstance(queries, list):
                logger.warning(f"Invalid followup_queries format: {type(queries)}")
                return []
            
            # Filter to only string queries
            valid_queries = [q for q in queries if isinstance(q, str) and q.strip()]
            
            logger.info(f"Generated {len(valid_queries)} follow-up queries: {valid_queries}")
            return valid_queries[:2]  # Max 2 follow-up queries
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse follow-up queries JSON: {e}")
            logger.error(f"Raw response: {response[:500]}")
            return []
        except Exception as e:
            logger.error(f"Failed to generate follow-up queries: {e}")
            return []
    
    async def iterative_rag_retrieval(self, query: str, max_iterations: int = 3) -> Dict:
        """
        Perform iterative RAG retrieval with follow-up queries
        
        Args:
            query: Original user query
            max_iterations: Maximum number of RAG calls (default 3)
        
        Returns:
            Dict with aggregated results from all iterations
        """
        all_chunks = []
        seen_texts = set()  # Avoid duplicates
        queries_made = [query]
        
        # First RAG call with original query
        logger.info(f"Iteration 1: Querying with original query")
        rag_results = await self.query_rag(query, top_k=20)
        
        if not rag_results.get("results"):
            return {"results": [], "queries_made": queries_made, "error": rag_results.get("error")}
        
        # Add unique chunks from first call
        for chunk in rag_results["results"]:
            chunk_text = chunk["text"]
            if chunk_text not in seen_texts:
                all_chunks.append(chunk)
                seen_texts.add(chunk_text)
        
        logger.info(f"Iteration 1: Retrieved {len(all_chunks)} unique chunks")
        
        # Generate follow-up queries
        if len(all_chunks) > 0:
            followup_queries = await self.generate_followup_queries(query, all_chunks)
            
            # Perform additional RAG calls for follow-up queries
            for i, followup_query in enumerate(followup_queries[:max_iterations-1], start=2):
                logger.info(f"Iteration {i}: Querying with follow-up query: '{followup_query}'")
                queries_made.append(followup_query)
                
                followup_results = await self.query_rag(followup_query, top_k=15)
                
                if followup_results.get("results"):
                    new_chunks_count = 0
                    for chunk in followup_results["results"]:
                        chunk_text = chunk["text"]
                        if chunk_text not in seen_texts:
                            all_chunks.append(chunk)
                            seen_texts.add(chunk_text)
                            new_chunks_count += 1
                    
                    logger.info(f"Iteration {i}: Added {new_chunks_count} new unique chunks")
        
        # Sort all chunks by score (highest first)
        all_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Iterative RAG complete: {len(queries_made)} queries made, {len(all_chunks)} total unique chunks")
        
        return {
            "results": all_chunks,
            "queries_made": queries_made,
            "total_chunks": len(all_chunks),
            "error": None
        }
    
    async def analyze_chunks(self, query: str, chunks: List[Dict]) -> str:
        """
        Analyze retrieved chunks and extract relevant information
        
        Args:
            query: Original query
            chunks: List of retrieved chunks from RAG
        
        Returns:
            Analyzed thought/summary from chunks
        """
        # Use top 20 chunks since we now have more diverse data from iterative retrieval
        # This gives us ~10k tokens, still within limits
        chunks_to_analyze = chunks[:20]
        
        # Format chunks
        chunks_text = ""
        for i, chunk in enumerate(chunks_to_analyze, 1):
            chunks_text += f"\n--- Chunk {i} (Score: {chunk['score']:.3f}) ---\n"
            chunks_text += f"Title: {chunk['metadata']['title']}\n"
            chunks_text += f"Text: {chunk['text']}\n"
        
        prompt = f"""You are analyzing information from the MetaKGP wiki to answer a question about IIT Kharagpur.

Question: {query}

Retrieved Information (top 20 most relevant chunks from iterative search):
{chunks_text}

INSTRUCTIONS:
1. Analyze ALL the chunks carefully
2. Extract and synthesize relevant information that answers the question
3. Focus on facts, names, dates, numbers, and specific details
4. If some chunks don't contain the answer, acknowledge it
5. Be comprehensive but concise
6. If the information is insufficient, say so clearly
7. Prioritize chunks with higher scores as they are more relevant

Provide a thorough analysis based on these chunks:"""
        
        logger.info(f"Analyzing top {len(chunks_to_analyze)} chunks with LLM")
        return await self.call_llm(prompt, max_tokens=2048)
    
    async def generate_final_answer(self, query: str, analysis: str, verification_result: Dict) -> str:
        """
        Generate the final answer based on analysis and verification
        
        Args:
            query: Original query
            analysis: Analysis from chunks
            verification_result: MoE verification result
        
        Returns:
            Final answer
        """
        verification_remarks = verification_result.get("remarks", "")
        passed = verification_result.get("passed", False)
        final_score = verification_result.get("final_score", 0.0)
        
        prompt = f"""You are synthesizing verified information from the MetaKGP wiki (IIT Kharagpur).

Original Question: {query}

Analysis from Wiki Chunks:
{analysis}

Verification Status:
- Passed: {passed}
- Confidence Score: {final_score:.2f}
- Verification Notes: {verification_remarks}

INSTRUCTIONS:
1. Provide a SHORT, DIRECT answer to the specific question asked
2. ONLY include information that directly answers the question
3. If information is not available, state this clearly and concisely - do NOT provide tangentially related information
4. Avoid unnecessary background, context, or historical information unless specifically asked
5. Be conversational but concise
6. Do not make up information not present in the analysis
7. Keep your response to 2-3 sentences for simple queries

Final Answer:"""
        
        logger.info("Generating final answer")
        return await self.call_llm(prompt, max_tokens=2048)
    
    async def process_query(self, query: str) -> Dict:
        """
        Process a query through the simplified pipeline with iterative RAG
        
        Pipeline:
        1. Iterative RAG retrieval (up to 3 queries for comprehensive data gathering)
        2. Check if query is relevant based on retrieved chunks
        3. Analyze chunks with Graph of Thought reasoning
        4. Run single MoE verification round
        5. Generate final answer
        
        Args:
            query: User query
        
        Returns:
            Dict with final answer and metadata
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Query RAG with iterative retrieval first
            logger.info("Step 1: Querying RAG with iterative retrieval (max 3 iterations)")
            rag_results = await self.iterative_rag_retrieval(query, max_iterations=3)
            
            if not rag_results.get("results"):
                logger.warning("No chunks retrieved from RAG")
                # Only check relevance if we got no chunks
                relevance = await self.check_relevance(query)
                
                if not relevance.get("is_relevant", True):
                    logger.info(f"Query not relevant: {relevance.get('reasoning')}")
                    return {
                        "query": query,
                        "answer": "This question is not related to IIT Kharagpur or MetaKGP. I can only answer questions about IIT Kharagpur campus, academics, facilities, events, societies, and related information from the MetaKGP wiki.",
                        "confidence": 0.0,
                        "chunks_retrieved": 0,
                        "verification_passed": False,
                        "reasoning": relevance.get("reasoning"),
                        "error": None
                    }
                
                # If relevant but no chunks found
                return {
                    "query": query,
                    "answer": "I don't have enough information in the MetaKGP wiki to answer this question. The wiki may not contain details about this topic yet.",
                    "confidence": 0.0,
                    "chunks_retrieved": 0,
                    "verification_passed": False,
                    "reasoning": "No relevant chunks found",
                    "error": rag_results.get("error")
                }
            
            # If we got chunks, skip relevance check (chunks prove relevance)
            chunks = rag_results["results"]
            queries_made = rag_results.get("queries_made", [query])
            logger.info(f"Retrieved {len(chunks)} unique chunks from {len(queries_made)} queries: {queries_made}")
            
            # Format context for MoE
            context_text = "\n\n".join([
                f"[{i+1}] {chunk['text']}" 
                for i, chunk in enumerate(chunks[:10])  # Use top 10 for context
            ])
            
            # Step 2: Analyze chunks with GoT reasoning
            logger.info("Step 2: Analyzing chunks")
            analysis = await self.analyze_chunks(query, chunks)
            
            if not analysis:
                logger.error("Failed to analyze chunks")
                return {
                    "query": query,
                    "answer": "I encountered an error while analyzing the information. Please try again.",
                    "confidence": 0.0,
                    "chunks_retrieved": len(chunks),
                    "verification_passed": False,
                    "reasoning": "Analysis failed",
                    "error": "LLM analysis failed"
                }
            
            # Step 3: Single MoE verification round
            logger.info("Step 3: Running MoE verification")
            verification = await self.moe_gauntlet.verify_thought(
                thought=analysis,
                context=context_text,
                graph_history=[]  # No graph history in simplified version
            )
            
            logger.info(f"Verification result: {verification['remarks']}")
            
            # Step 4: Generate final answer
            logger.info("Step 4: Generating final answer")
            final_answer = await self.generate_final_answer(query, analysis, verification)
            
            if not final_answer:
                logger.error("Failed to generate final answer")
                return {
                    "query": query,
                    "answer": "I encountered an error while generating the answer. Please try again.",
                    "confidence": 0.0,
                    "chunks_retrieved": len(chunks),
                    "verification_passed": verification["passed"],
                    "reasoning": verification["remarks"],
                    "error": "Final answer generation failed"
                }
            
            # Return result
            confidence = verification["final_score"] * 0.9  # Scale down slightly
            
            # Extract query keywords (remove common words)
            query_lower = query.lower()
            common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'at', 'to', 'for', 'on', 'with', 'by', 'from', 'what', 'who', 'where', 'when', 'how', 'why', 'which'}
            query_keywords = [word for word in query_lower.split() if word not in common_words and len(word) > 2]
            
            # Filter chunks that have query keywords in source_page or title
            relevant_chunks = []
            for chunk in chunks:
                source_page = chunk["metadata"]["source_page"].lower()
                title = chunk["metadata"]["title"].lower()
                
                # Check if any query keyword appears in source_page or title
                has_keyword = any(keyword in source_page or keyword in title for keyword in query_keywords)
                
                if has_keyword:
                    relevant_chunks.append(chunk)
            
            # If we have relevant chunks, use them; otherwise fall back to top chunks
            chunks_for_sources = relevant_chunks if relevant_chunks else chunks
            
            # Extract unique sources, prioritizing those with query keywords
            unique_sources = []
            seen_sources = set()
            for chunk in chunks_for_sources[:15]:  # Check top 15 relevant chunks
                source = chunk["metadata"]["source_page"]
                if source not in seen_sources and len(unique_sources) < 5:
                    unique_sources.append({
                        "page": source,
                        "score": chunk["score"]
                    })
                    seen_sources.add(source)
            
            # Format sources as list of page names
            source_pages = [s["page"] for s in unique_sources]
            
            return {
                "query": query,
                "answer": final_answer,
                "confidence": confidence,
                "chunks_retrieved": len(chunks),
                "queries_made": queries_made,
                "verification_passed": verification["passed"],
                "verification_score": verification["final_score"],
                "reasoning": verification["remarks"],
                "sources": source_pages,
                "error": None
            }
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            return {
                "query": query,
                "answer": "I encountered an error while processing your question. Please try again.",
                "confidence": 0.0,
                "chunks_retrieved": 0,
                "verification_passed": False,
                "reasoning": str(e),
                "error": str(e)
            }
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
