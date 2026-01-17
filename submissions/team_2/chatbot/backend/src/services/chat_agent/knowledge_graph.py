"""
Knowledge Graph for Persistent Thought Storage
"""

import logging
import networkx as nx
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.services.chat_agent.data_structures import ThoughtNode

logger = logging.getLogger(__name__)


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
            except Exception as e:
                logger.warning(f"Failed to load graph: {e}, creating new graph")
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
    
    def get_stats(self) -> dict:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.G.nodes()),
            "total_edges": len(self.G.edges()),
            "graph_file": str(self.graph_file)
        }
