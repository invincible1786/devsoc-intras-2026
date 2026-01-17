"""
Graph of Thoughts + Mixture of Experts Engine
Main orchestration for the complete GoT + MoE pipeline
"""

import logging
from typing import Dict, List, Optional
import asyncio

from src.services.chat_agent.data_structures import ThoughtNode
from src.services.chat_agent.agents import (
    planner_agent,
    execution_agent,
    verification_agent,
    synthesis_agent
)
from src.services.chat_agent.knowledge_graph import KnowledgeGraph
from src.utils.chroma_client import MetaKGPChromaClient
from src.utils.embedding_client import ModalEmbeddingClient
from src.utils.groq_client import GroqClient

logger = logging.getLogger(__name__)


class GoTMoEEngine:
    """
    Main Graph of Thoughts + Mixture of Experts engine.
    
    Implements the complete pipeline:
    1. Planner: Decompose query into sub-questions
    2. Execution: Generate multiple reasoning paths (RAG: 45→10 chunks)
    3. Verification: MoE evaluates all paths (3 experts)
    4. Synthesis: Combine verified thoughts
    5. Graph Learning: Store verified thoughts
    """
    
    def __init__(
        self,
        chroma_client: Optional[MetaKGPChromaClient] = None,
        embedding_client: Optional[ModalEmbeddingClient] = None,
        groq_client: Optional[GroqClient] = None,
        graph_file: str = "./cache/metakgp_graph.gml"
    ):
        """
        Initialize the GoT + MoE engine.
        
        Args:
            chroma_client: ChromaDB client (will create if None)
            embedding_client: Embedding client (will create if None)
            groq_client: Groq client (will create if None)
            graph_file: Path to save knowledge graph
        """
        self.chroma_client = chroma_client or MetaKGPChromaClient()
        self.embedding_client = embedding_client or ModalEmbeddingClient()
        self.groq_client = groq_client or GroqClient()
        self.graph = KnowledgeGraph(graph_file)
        
        logger.info("GoTMoEEngine initialized")
    
    async def process_query(self, query: str) -> Dict:
        """
        Process a user query through the complete GoT + MoE pipeline.
        
        Args:
            query: User's query
            
        Returns:
            Dictionary with:
            - answer: Final synthesized answer
            - confidence: Overall confidence score
            - sources: List of sources used
            - reasoning_path: List of sub-questions and answers
            - graph_stats: Knowledge graph statistics
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Plan (decompose query)
            plan = await planner_agent(query, self.groq_client)
            logger.info(f"Plan: {plan}")
            
            # Step 2 & 3: Execute and Verify each sub-question
            nodes = []
            for i, sub_question in enumerate(plan):
                # Create node
                node = ThoughtNode(id=i, question=sub_question)
                
                # Execute (generate reasoning paths with RAG)
                node = await execution_agent(
                    node,
                    self.chroma_client,
                    self.embedding_client,
                    self.groq_client
                )
                
                # Verify (MoE)
                node = await verification_agent(node, query, self.groq_client)
                
                nodes.append(node)
                
                # Update graph
                previous_node = nodes[i-1] if i > 0 else None
                self.graph.add_verified_thought(node, previous_node)
            
            # Save graph
            self.graph.save()
            
            # Step 4: Synthesize final answer
            final_answer = await synthesis_agent(query, nodes, self.groq_client)
            
            # Prepare response
            verified_nodes = [n for n in nodes if n.verified]
            avg_confidence = sum(n.score for n in verified_nodes) / len(verified_nodes) if verified_nodes else 0
            
            sources = []
            reasoning_path = []
            
            for node in nodes:
                reasoning_path.append({
                    "question": node.question,
                    "answer": node.derived_thought,
                    "verified": node.verified,
                    "score": node.score
                })
                
                if node.verified:
                    verified_path = [p for p in node.reasoning_paths if p.is_verified]
                    if verified_path:
                        sources.append(verified_path[0].source_info)
            
            return {
                "answer": final_answer,
                "confidence": avg_confidence / 10.0,  # Normalize to 0-1
                "sources": list(set(sources)),  # Unique sources
                "reasoning_path": reasoning_path,
                "graph_stats": self.graph.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "reasoning_path": [],
                "graph_stats": self.graph.get_stats()
            }
    
    def get_graph_stats(self) -> Dict:
        """Get knowledge graph statistics."""
        return self.graph.get_stats()
