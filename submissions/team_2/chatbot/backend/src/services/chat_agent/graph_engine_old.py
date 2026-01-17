"""
Graph of Thoughts (GoT) Engine with NetworkX
Implements iterative thought expansion with verification, caching, and graph visualization
"""

import logging
import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
import networkx as nx
from pyvis.network import Network

from src.services.chat_agent.moe import MoERouter, NodeAction
from src.utils.groq_client import GroqClient
from src.utils.chroma_client import MetaKGPChromaClient
from src.utils.embedding_client import ModalEmbeddingClient

logger = logging.getLogger(__name__)


class ThoughtNode:
    """Represents a single thought node in the graph"""
    
    def __init__(
        self,
        node_id: str,
        thought: str,
        sub_query: str,
        sources: List[str],
        parent_ids: List[str],
        verification_score: float,
        expert_remarks: str,
        depth: int,
        context: str
    ):
        self.node_id = node_id
        self.thought = thought
        self.sub_query = sub_query
        self.sources = sources
        self.parent_ids = parent_ids
        self.verification_score = verification_score
        self.expert_remarks = expert_remarks
        self.depth = depth
        self.context = context
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary for serialization"""
        return {
            "node_id": self.node_id,
            "thought": self.thought,
            "sub_query": self.sub_query,
            "sources": self.sources,
            "parent_ids": self.parent_ids,
            "verification_score": self.verification_score,
            "expert_remarks": self.expert_remarks,
            "depth": self.depth,
            "context": self.context[:500],  # Truncate for storage
            "timestamp": self.timestamp
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'ThoughtNode':
        """Create node from dictionary"""
        return ThoughtNode(
            node_id=data["node_id"],
            thought=data["thought"],
            sub_query=data["sub_query"],
            sources=data["sources"],
            parent_ids=data["parent_ids"],
            verification_score=data["verification_score"],
            expert_remarks=data["expert_remarks"],
            depth=data["depth"],
            context=data.get("context", "")
        )


class ThoughtCache:
    """
    Tier 1: Semantic caching of verified thoughts in ChromaDB
    Prevents re-computing similar sub-queries
    """
    
    def __init__(self, chroma_client: MetaKGPChromaClient, embedding_client: ModalEmbeddingClient):
        self.chroma_client = chroma_client
        self.embedding_client = embedding_client
        self.cache_collection_name = "verified_thoughts"
        
        # Create cache collection
        try:
            self.cache_collection = self.chroma_client.client.get_or_create_collection(
                name=self.cache_collection_name,
                metadata={"description": "Cached verified thoughts", "hnsw:space": "cosine"}
            )
            logger.info(f"ThoughtCache initialized with collection: {self.cache_collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ThoughtCache: {e}")
            self.cache_collection = None
    
    async def get_cached_thought(self, sub_query: str, threshold: float = 0.1) -> Optional[Dict]:
        """
        Retrieve cached thought if similar query exists.
        
        Args:
            sub_query: The sub-query to search for
            threshold: Distance threshold (lower = more similar)
            
        Returns:
            Cached thought dict or None
        """
        if not self.cache_collection:
            return None
        
        try:
            # Get embedding for sub_query (embedding_client is callable)
            embedding = self.embedding_client(sub_query)
            if not embedding:
                return None
            
            # Query cache
            results = self.cache_collection.query(
                query_embeddings=[embedding],
                n_results=1
            )
            
            if results and results["ids"] and len(results["ids"][0]) > 0:
                distance = results["distances"][0][0]
                
                if distance < threshold:
                    # Cache hit!
                    metadata = results["metadatas"][0][0]
                    logger.info(f"Cache HIT for query: {sub_query[:50]}... (distance: {distance:.4f})")
                    
                    return {
                        "thought": results["documents"][0][0],
                        "sources": json.loads(metadata.get("sources", "[]")),
                        "verification_score": float(metadata.get("verification_score", 0.8)),
                        "expert_remarks": metadata.get("expert_remarks", "Cached result"),
                        "context": metadata.get("context", ""),
                        "cached": True
                    }
            
            logger.info(f"Cache MISS for query: {sub_query[:50]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error querying thought cache: {e}")
            return None
    
    async def store_thought(self, sub_query: str, thought: Dict):
        """
        Store verified thought in cache.
        
        Args:
            sub_query: The sub-query
            thought: Thought dict with all metadata
        """
        if not self.cache_collection:
            return
        
        try:
            # Get embedding (embedding_client is callable)
            embedding = self.embedding_client(sub_query)
            if not embedding:
                return
            
            # Generate ID
            thought_id = hashlib.md5(sub_query.encode()).hexdigest()
            
            # Store in cache
            self.cache_collection.add(
                ids=[thought_id],
                embeddings=[embedding],
                documents=[thought["thought"]],
                metadatas=[{
                    "sub_query": sub_query,
                    "sources": json.dumps(thought.get("sources", [])),
                    "verification_score": str(thought.get("verification_score", 0.0)),
                    "expert_remarks": thought.get("expert_remarks", ""),
                    "context": thought.get("context", "")[:500],
                    "timestamp": datetime.now().isoformat()
                }]
            )
            
            logger.info(f"Stored thought in cache: {sub_query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error storing thought in cache: {e}")


class GraphCache:
    """
    Tier 2: Graph-level caching for entire reasoning paths
    Stores complete NetworkX graphs for similar queries
    """
    
    def __init__(self, cache_dir: str = "./cache/graphs"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"GraphCache initialized at: {self.cache_dir}")
    
    def get_cached_graph(self, query: str) -> Optional[nx.DiGraph]:
        """
        Retrieve cached graph for similar query.
        
        Args:
            query: The query
            
        Returns:
            NetworkX graph or None
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_file = self.cache_dir / f"{query_hash}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct graph
                graph = nx.node_link_graph(data)
                logger.info(f"GraphCache HIT for query: {query[:50]}...")
                return graph
                
            except Exception as e:
                logger.error(f"Error loading cached graph: {e}")
                return None
        
        logger.info(f"GraphCache MISS for query: {query[:50]}...")
        return None
    
    def store_graph(self, query: str, graph: nx.DiGraph):
        """
        Store complete graph in cache.
        
        Args:
            query: The query
            graph: NetworkX graph
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_file = self.cache_dir / f"{query_hash}.json"
        
        try:
            # Convert graph to JSON
            data = nx.node_link_data(graph)
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Stored graph in cache: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error storing graph: {e}")


class GoTEngine:
    """
    Graph of Thoughts Engine with iterative expansion and verification.
    Uses NetworkX for graph management and ChromaDB for caching.
    """
    
    def __init__(
        self,
        groq_client: GroqClient,
        chroma_client: MetaKGPChromaClient,
        embedding_client: ModalEmbeddingClient,
        query_api_url: str = "http://localhost:8000/query/search",
        max_depth: int = 3,
        max_branches: int = 3,
        confidence_threshold: float = 0.85
    ):
        self.groq_client = groq_client
        self.chroma_client = chroma_client
        self.embedding_client = embedding_client
        self.query_api_url = query_api_url
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.confidence_threshold = confidence_threshold
        
        # Initialize MoE
        self.moe_router = MoERouter(groq_client)
        
        # Initialize caching
        self.thought_cache = ThoughtCache(chroma_client, embedding_client)
        self.graph_cache = GraphCache()
        
        # Graph state
        self.graph = nx.DiGraph()
        self.node_counter = 0
        
        logger.info(f"GoTEngine initialized (max_depth={max_depth}, max_branches={max_branches}, threshold={confidence_threshold})")
    
    async def process_query(self, query: str, use_cache: bool = True) -> Dict:
        """
        Main entry point: Process query with Graph of Thoughts reasoning.
        
        Args:
            query: User query
            use_cache: Whether to use cached results
            
        Returns:
            Result dict with answer, graph, and metadata
        """
        logger.info(f"GoT processing query: {query}")
        
        # Check graph cache
        if use_cache:
            cached_graph = self.graph_cache.get_cached_graph(query)
            if cached_graph:
                self.graph = cached_graph
                return await self._synthesize_answer(query)
        
        # Initialize new graph
        self.graph = nx.DiGraph()
        self.node_counter = 0
        
        # Create root node
        root_id = self._generate_node_id()
        self.graph.add_node(root_id, query=query, depth=0, is_root=True)
        
        # Expand graph iteratively
        await self._expand_graph(query, root_id, depth=0)
        
        # Store graph in cache
        if use_cache:
            self.graph_cache.store_graph(query, self.graph)
        
        # Synthesize final answer
        result = await self._synthesize_answer(query)
        
        # Generate visualization
        viz_path = self._visualize_graph(query)
        result["visualization_path"] = viz_path
        
        return result
    
    async def _expand_graph(self, query: str, parent_id: str, depth: int):
        """
        Recursively expand the graph by generating and verifying thoughts.
        
        Args:
            query: Current sub-query or original query
            parent_id: Parent node ID
            depth: Current depth in the graph
        """
        if depth >= self.max_depth:
            logger.info(f"Max depth reached at node {parent_id}")
            return
        
        # At depth 0, try to answer the query directly first
        if depth == 0:
            logger.info("Attempting to answer query directly at root level")
            direct_result = await self._process_thought(query, parent_id, depth + 1)
            
            # If we got a good answer, check if we need more info
            if isinstance(direct_result, dict) and direct_result.get("accepted"):
                child_id = direct_result["node_id"]
                confidence = await self._check_confidence(query, child_id)
                
                if confidence >= self.confidence_threshold:
                    logger.info(f"Direct answer sufficient (confidence: {confidence:.2f})")
                    return
                
                logger.info(f"Direct answer incomplete (confidence: {confidence:.2f}), expanding further")
        
        # Check if we can answer with current knowledge
        if depth > 0:
            confidence = await self._check_confidence(query, parent_id)
            if confidence >= self.confidence_threshold:
                logger.info(f"Confidence threshold met ({confidence:.2f}) at depth {depth}")
                return
        
        # Generate sub-queries/thoughts
        sub_queries = await self._generate_sub_queries(query, parent_id, depth)
        
        if not sub_queries:
            logger.info(f"No more sub-queries at depth {depth}")
            return
        
        # Process sub-queries in parallel (up to max_branches)
        tasks = []
        for sub_query in sub_queries[:self.max_branches]:
            tasks.append(self._process_thought(sub_query, parent_id, depth + 1))
        
        # Wait for all thoughts to be processed
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Continue expansion for accepted nodes
        for result in results:
            if isinstance(result, dict) and result.get("accepted"):
                child_id = result["node_id"]
                await self._expand_graph(sub_query, child_id, depth + 1)
    
    async def _process_thought(self, sub_query: str, parent_id: str, depth: int) -> Dict:
        """
        Process a single thought: retrieve context, verify, and add to graph.
        
        Args:
            sub_query: The sub-query to process
            parent_id: Parent node ID
            depth: Current depth
            
        Returns:
            Result dict
        """
        logger.info(f"Processing thought at depth {depth}: {sub_query[:50]}...")
        
        # Check thought cache first
        cached_thought = await self.thought_cache.get_cached_thought(sub_query)
        
        if cached_thought:
            # Use cached result
            node_id = self._generate_node_id()
            node = ThoughtNode(
                node_id=node_id,
                thought=cached_thought["thought"],
                sub_query=sub_query,
                sources=cached_thought["sources"],
                parent_ids=[parent_id],
                verification_score=cached_thought["verification_score"],
                expert_remarks=cached_thought["expert_remarks"] + " [CACHED]",
                depth=depth,
                context=cached_thought["context"]
            )
            
            self._add_node_to_graph(node, parent_id)
            
            return {"accepted": True, "node_id": node_id, "cached": True}
        
        # Retrieve context from RAG
        context_result = await self._retrieve_context(sub_query, top_k=15)  # Increased from 10 to 15
        
        if not context_result["success"]:
            logger.warning(f"Failed to retrieve context for: {sub_query[:50]}...")
            return {"accepted": False, "reason": "No context retrieved"}
        
        context = context_result["context"]
        sources = context_result["sources"]
        
        # Check if this is a simple factual query (fast path)
        is_simple_query = self._is_simple_factual_query(sub_query)
        
        # Generate thought
        thought = await self._generate_thought(sub_query, context, parent_id)
        
        if not thought:
            return {"accepted": False, "reason": "Failed to generate thought"}
        
        # For simple queries with good context, use lightweight verification
        if is_simple_query and len(context) > 200:
            logger.info(f"Using fast path for simple query: {sub_query[:50]}...")
            # Quick check: does the thought seem to contain relevant information?
            if len(thought) > 20 and not thought.lower().startswith("not found") and not thought.lower().startswith("information not"):
                node_id = self._generate_node_id()
                node = ThoughtNode(
                    node_id=node_id,
                    thought=thought,
                    sub_query=sub_query,
                    sources=sources,
                    parent_ids=[parent_id],
                    verification_score=0.85,  # Good default for fast path
                    expert_remarks="Fast path: Simple factual extraction",
                    depth=depth,
                    context=context
                )
                
                self._add_node_to_graph(node, parent_id)
                await self.thought_cache.store_thought(sub_query, node.to_dict())
                
                return {"accepted": True, "node_id": node_id, "fast_path": True}
        
        # Get parent thoughts for logic checking
        parent_thoughts = self._get_parent_thoughts(parent_id)
        
        # Verify with MoE
        verification = await self.moe_router.verify_thought(
            thought=thought,
            context=context,
            metadata={
                "query": sub_query,
                "parent_thoughts": parent_thoughts,
                "sources": sources,
                "depth": depth,
                "is_simple_query": is_simple_query
            }
        )
        
        action = verification["action"]
        
        if action == NodeAction.ACCEPT.value:
            # Accept: Add to graph
            node_id = self._generate_node_id()
            node = ThoughtNode(
                node_id=node_id,
                thought=thought,
                sub_query=sub_query,
                sources=sources,
                parent_ids=[parent_id],
                verification_score=verification["score"],
                expert_remarks=verification["reasoning"],
                depth=depth,
                context=context
            )
            
            self._add_node_to_graph(node, parent_id)
            
            # Store in cache
            await self.thought_cache.store_thought(sub_query, node.to_dict())
            
            return {"accepted": True, "node_id": node_id}
        
        elif action == NodeAction.MERGE.value:
            # Merge with parent
            logger.info(f"Merging thought with parent {parent_id}")
            self._merge_with_parent(parent_id, thought, verification)
            return {"accepted": False, "merged": True, "parent_id": parent_id}
        
        elif action == NodeAction.RETHINK.value:
            # Rethink: Try once more with feedback
            logger.info(f"Rethinking with feedback: {verification.get('self_correction_feedback')}")
            
            refined_thought = await self._generate_thought(
                sub_query,
                context,
                parent_id,
                feedback=verification.get("self_correction_feedback")
            )
            
            if refined_thought:
                # Re-verify
                reverification = await self.moe_router.verify_thought(
                    thought=refined_thought,
                    context=context,
                    metadata={
                        "query": sub_query,
                        "parent_thoughts": parent_thoughts,
                        "sources": sources,
                        "is_simple_query": is_simple_query
                    }
                )
                
                if reverification["action"] == NodeAction.ACCEPT.value:
                    node_id = self._generate_node_id()
                    node = ThoughtNode(
                        node_id=node_id,
                        thought=refined_thought,
                        sub_query=sub_query,
                        sources=sources,
                        parent_ids=[parent_id],
                        verification_score=reverification["score"],
                        expert_remarks=reverification["reasoning"] + " [REFINED]",
                        depth=depth,
                        context=context
                    )
                    
                    self._add_node_to_graph(node, parent_id)
                    await self.thought_cache.store_thought(sub_query, node.to_dict())
                    
                    return {"accepted": True, "node_id": node_id, "refined": True}
            
            return {"accepted": False, "reason": "Refinement failed"}
        
        else:  # REJECT
            logger.info(f"Thought rejected: {verification['reasoning']}")
            return {"accepted": False, "reason": verification["reasoning"]}
    
    async def _retrieve_context(self, sub_query: str, top_k: int = 10) -> Dict:
        """
        Retrieve context from RAG API with query expansion and intelligent filtering.
        
        Args:
            sub_query: Query to retrieve context for
            top_k: Number of chunks to retrieve (after filtering)
            
        Returns:
            Dict with context, sources, and success flag
        """
        try:
            import httpx
            
            # Expand query to improve RAG retrieval
            expanded_queries = await self._expand_query_for_rag(sub_query)
            
            # Extract entity from query for filtering (e.g., "encore" from "governors of encore")
            query_entity = self._extract_entity_from_query(sub_query)
            
            all_chunks = []
            seen_chunk_ids = set()
            
            # Increase top_k for RAG to get more candidates for filtering
            rag_top_k = min(top_k * 3, 30)  # Get 3x chunks, up to 30
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Query with original and expanded queries
                for query in [sub_query] + expanded_queries[:2]:  # Original + top 2 expansions
                    response = await client.post(
                        self.query_api_url,
                        json={"query": query, "top_k": rag_top_k}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        chunks = data.get("results", [])
                        
                        # Deduplicate chunks
                        for chunk in chunks:
                            chunk_id = chunk.get("chunk_id")
                            if chunk_id and chunk_id not in seen_chunk_ids:
                                seen_chunk_ids.add(chunk_id)
                                all_chunks.append(chunk)
                
                # Filter and re-rank chunks based on entity relevance
                if query_entity and all_chunks:
                    all_chunks = self._filter_and_rerank_chunks(all_chunks, query_entity, sub_query)
                else:
                    # Just sort by score
                    all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
                
                # Take top_k after filtering
                all_chunks = all_chunks[:top_k]
                
                if all_chunks:
                    context = "\n\n".join([f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(all_chunks)])
                    
                    # Extract sources with proper URLs
                    sources = []
                    seen_sources = set()
                    for chunk in all_chunks:
                        metadata = chunk.get("metadata", {})
                        source_page = metadata.get("source_page", "")
                        
                        if source_page and source_page not in seen_sources:
                            # Construct MetaKGP wiki URL from source page name
                            # Replace spaces with underscores for wiki URL format
                            url_encoded = source_page.replace(" ", "_")
                            source_url = f"https://wiki.metakgp.org/w/{url_encoded}"
                            sources.append(source_url)
                            seen_sources.add(source_page)
                    
                    if not sources:
                        sources = ["Unknown"]
                    
                    logger.info(f"Retrieved {len(all_chunks)} chunks after filtering for: {sub_query[:50]}...")
                    
                    return {"success": True, "context": context, "sources": sources, "chunks": all_chunks}
                
                return {"success": False, "context": "", "sources": [], "chunks": []}
                
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return {"success": False, "context": "", "sources": [], "chunks": []}
    
    async def _expand_query_for_rag(self, query: str) -> List[str]:
        """
        Expand query with synonyms and alternative phrasings to improve RAG retrieval.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded/alternative queries
        """
        # Quick synonym expansion without LLM call to save tokens
        expansions = []
        query_lower = query.lower()
        
        # Common MetaKGP synonym mappings
        synonym_map = {
            "governors": ["leadership", "heads", "leaders", "coordinators", "board"],
            "directors": ["heads", "governors", "leadership", "coordinators"],
            "heads": ["directors", "governors", "leadership", "coordinators"],
            "members": ["participants", "students", "team"],
            "founded": ["established", "started", "created", "formation"],
            "founder": ["creator", "establisher", "started by"],
            "events": ["activities", "programs", "competitions"],
            "encore": ["Technology Dramatics Society ENCORE", "English Dramatics", "ETDS"],
            "spring fest": ["SpringFest", "Spring Festival", "SF"],
            "kshitij": ["Kshitij techno-management fest"],
        }
        
        # Extract key terms and expand
        for term, synonyms in synonym_map.items():
            if term in query_lower:
                for synonym in synonyms[:2]:  # Take top 2 synonyms
                    expanded = query_lower.replace(term, synonym)
                    if expanded != query_lower:
                        expansions.append(expanded)
        
        # If query is about a specific role in an organization, try entity-first format
        role_terms = ["governors", "directors", "heads", "members", "coordinators", "founders"]
        for role in role_terms:
            if role in query_lower and " of " in query_lower:
                # "governors of encore" -> "encore governors"
                parts = query_lower.split(" of ")
                if len(parts) == 2:
                    entity_first = f"{parts[1]} {parts[0]}"
                    expansions.append(entity_first)
                    # Also try without the role term
                    expansions.append(parts[1])
                break
        
        return expansions[:3]  # Return top 3 expansions
    
    def _extract_entity_from_query(self, query: str) -> Optional[str]:
        """
        Extract the main entity from a query for filtering chunks.
        
        Args:
            query: The query
            
        Returns:
            Main entity name or None
        """
        query_lower = query.lower()
        
        # Common patterns: "X of Y", "Y's X", "who are the X of Y"
        # Extract Y as the entity
        
        # Pattern: "... of ENTITY"
        if " of " in query_lower:
            parts = query_lower.split(" of ")
            if len(parts) >= 2:
                entity = parts[-1].strip()
                # Clean up common suffixes
                entity = entity.replace("?", "").strip()
                return entity
        
        # Pattern: "ENTITY's ..." or "ENTITY governors"
        # Look for known entities
        known_entities = [
            "encore", "spring fest", "springfest", "kshitij", "rangmanch",
            "business club", "aroma", "spectra", "student welfare group", "swg",
            "debating society", "technology dance society", "prasthanam",
            "technology students gymkhana", "robotics", "aviation society"
        ]
        
        for entity in known_entities:
            if entity in query_lower:
                return entity
        
        return None
    
    def _filter_and_rerank_chunks(self, chunks: List[Dict], entity: str, query: str) -> List[Dict]:
        """
        Filter and re-rank chunks based on entity relevance.
        Uses metadata (title, source_page) and content matching.
        
        Args:
            chunks: List of chunk dicts from RAG
            entity: Main entity to filter for
            query: Original query
            
        Returns:
            Filtered and re-ranked chunks
        """
        entity_lower = entity.lower()
        query_lower = query.lower()
        
        # Score each chunk based on entity relevance
        scored_chunks = []
        for chunk in chunks:
            score = chunk.get("score", 0)
            metadata = chunk.get("metadata", {})
            title = metadata.get("title", "").lower()
            source_page = metadata.get("source_page", "").lower()
            text = chunk.get("text", "").lower()
            
            # Boost factors
            boost = 1.0
            
            # Strong boost: Entity in title (most important)
            if entity_lower in title:
                boost *= 2.5
                logger.debug(f"Title match for '{entity}' in: {title[:50]}")
            
            # Strong boost: Entity in source_page
            if entity_lower in source_page:
                boost *= 2.5
                logger.debug(f"Source page match for '{entity}' in: {source_page[:50]}")
            
            # Medium boost: Entity appears multiple times in text
            entity_count = text.count(entity_lower)
            if entity_count > 0:
                boost *= (1.0 + entity_count * 0.2)  # +20% per occurrence
            
            # Penalty: Wrong entity in title (different society/organization)
            other_entities = ["aroma", "business club", "spectra", "swg", "debating", "prasthanam"]
            if entity_lower not in other_entities:
                other_entities.append(entity_lower)
            
            for other in other_entities:
                if other != entity_lower and other in title:
                    boost *= 0.3  # Heavy penalty for wrong entity
                    logger.debug(f"Wrong entity '{other}' in title: {title[:50]}")
                    break
            
            # Apply boost to score
            boosted_score = score * boost
            
            scored_chunks.append({
                "chunk": chunk,
                "original_score": score,
                "boosted_score": boosted_score,
                "boost_factor": boost
            })
        
        # Sort by boosted score
        scored_chunks.sort(key=lambda x: x["boosted_score"], reverse=True)
        
        # Log top 3 for debugging
        logger.info(f"Top 3 chunks after filtering for '{entity}':")
        for i, sc in enumerate(scored_chunks[:3]):
            title = sc["chunk"].get("metadata", {}).get("title", "Unknown")
            logger.info(f"  {i+1}. {title} (score: {sc['original_score']:.3f} → {sc['boosted_score']:.3f}, boost: {sc['boost_factor']:.2f}x)")
        
        # Return just the chunks
        return [sc["chunk"] for sc in scored_chunks]
    
    def _is_simple_factual_query(self, query: str) -> bool:
        """
        Detect if query is a simple factual lookup (list of people, dates, etc.).
        
        Args:
            query: The query to check
            
        Returns:
            True if simple factual query
        """
        query_lower = query.lower()
        
        # Patterns that indicate simple factual queries
        simple_patterns = [
            # Role-based queries
            "who are the",
            "who is the",
            "list of",
            "names of",
            # Specific role terms
            "governors",
            "directors",
            "heads",
            "members",
            "coordinators",
            "founders",
            "president",
            "secretary",
            # Temporal queries
            "when was",
            "when did",
            "what year",
            # Definition queries
            "what is",
            "what are",
        ]
        
        return any(pattern in query_lower for pattern in simple_patterns)
    
    def _wants_historical_data(self, query: str) -> bool:
        """
        Detect if query is asking for historical or complete list of data.
        
        Args:
            query: The query to check
            
        Returns:
            True if asking for historical/all data
        """
        query_lower = query.lower()
        
        # Patterns that indicate historical/complete data request
        historical_patterns = [
            "history",
            "all governors",
            "all directors",
            "all heads",
            "list of all",
            "over the years",
            "past",
            "previous",
            "since",
            "from the beginning",
            "complete list",
            "through the years",
            "historical",
        ]
        
        return any(pattern in query_lower for pattern in historical_patterns)
    
    async def _generate_thought(
        self,
        sub_query: str,
        context: str,
        parent_id: str,
        feedback: Optional[str] = None
    ) -> str:
        """
        Generate a thought based on sub-query and context.
        
        Args:
            sub_query: The sub-query
            context: Retrieved context
            parent_id: Parent node ID
            feedback: Optional feedback from previous verification
            
        Returns:
            Generated thought text
        """
        parent_thoughts = self._get_parent_thoughts(parent_id)
        history = "\n".join([f"- {t}" for t in parent_thoughts[-2:]]) if parent_thoughts else "[Starting new reasoning]"
        
        feedback_section = f"\n\nPREVIOUS FEEDBACK:\n{feedback}\n\nPlease address this feedback in your response." if feedback else ""
        
        # Check if query wants historical data
        wants_history = self._wants_historical_data(sub_query)
        recency_instruction = "" if wants_history else "2. **CRITICAL: Unless the question asks for 'history', 'all', 'list of all', or 'over the years', provide ONLY the most recent/current information**\n3. For role-based queries (governors, directors, heads, members):\n   - Look for 'current', '2024-25', '2025-26', or the latest year mentioned\n   - Return ONLY the most recent names unless specifically asked for history\n"
        
        prompt = f"""You are extracting factual information from MetaKGP wiki context.

QUESTION: {sub_query}

CONTEXT FROM METAKGP:
{context}
{feedback_section}

INSTRUCTIONS:
1. Find information in the context that directly answers the question
{recency_instruction}4. If the context contains lists with years/sessions, identify the {'complete list' if wants_history else 'LATEST one'}
5. Extract information EXACTLY as stated - copy names and details verbatim
6. If information is not in context, respond: "Information not found in the provided context"
7. Keep your response concise (2-4 sentences)

Extract the facts:"""
        
        try:
            response = await self.groq_client.generate_judge(prompt, max_tokens=512)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating thought: {e}")
            return ""
    
    async def _generate_sub_queries(self, query: str, parent_id: str, depth: int) -> List[str]:
        """
        Generate sub-queries for expanding the graph.
        
        Args:
            query: Current query
            parent_id: Parent node ID
            depth: Current depth
            
        Returns:
            List of sub-queries
        """
        # Get current path
        path_thoughts = self._get_parent_thoughts(parent_id)
        history = "\n".join([f"- {t}" for t in path_thoughts]) if path_thoughts else "[Starting exploration]"
        
        prompt = f"""You are a query decomposition expert for MetaKGP wiki.

ORIGINAL QUESTION: {query}

REASONING SO FAR:
{history}

TASK: Generate {self.max_branches} focused sub-questions to help answer the original question.

RULES:
1. Each sub-question should be specific and direct (e.g., "Who are the governors of ENCORE?" not "Tell me about ENCORE")
2. Focus on what's missing from the reasoning path
3. If the reasoning path seems complete, return empty list []
4. Sub-questions must be answerable from a wiki database

Output valid JSON only:
{{
    "sub_queries": ["sub-question 1", "sub-question 2", ...],
    "reasoning": "brief explanation"
}}"""
        
        try:
            response = await self.groq_client.generate_judge(prompt, max_tokens=512)
            from src.services.chat_agent.moe import strip_markdown_json
            cleaned = strip_markdown_json(response)
            result = json.loads(cleaned)
            
            sub_queries = result.get("sub_queries", [])
            logger.info(f"Generated {len(sub_queries)} sub-queries at depth {depth}")
            
            return sub_queries
            
        except Exception as e:
            logger.error(f"Error generating sub-queries: {e}")
            return []
    
    async def _check_confidence(self, query: str, node_id: str) -> float:
        """
        Check if current knowledge is sufficient to answer the query.
        
        Args:
            query: The query
            node_id: Current node ID
            
        Returns:
            Confidence score (0-1)
        """
        # Get all thoughts in current path
        thoughts = self._get_parent_thoughts(node_id)
        
        if not thoughts:
            return 0.0
        
        combined_knowledge = "\n".join(thoughts)
        
        prompt = f"""Evaluate if the current information is sufficient to answer the question.

QUESTION: {query}

INFORMATION GATHERED:
{combined_knowledge}

TASK: Rate confidence from 0.0 to 1.0 on whether you can answer the question with this information.

SCORING:
- 0.9-1.0: Complete answer with specific details (names, dates, facts)
- 0.7-0.8: Mostly complete, minor details missing
- 0.5-0.6: Partial information
- 0.0-0.4: Insufficient information

Return valid JSON only:
{{
    "confidence": 0.0,
    "can_answer": false,
    "missing_info": "what else is needed"
}}"""
        
        try:
            response = await self.groq_client.generate_expert(prompt, max_tokens=256)
            from src.services.chat_agent.moe import strip_markdown_json
            cleaned = strip_markdown_json(response)
            result = json.loads(cleaned)
            
            return float(result.get("confidence", 0.0))
            
        except Exception as e:
            logger.error(f"Error checking confidence: {e}")
            return 0.0
    
    async def _synthesize_answer(self, query: str) -> Dict:
        """
        Synthesize final answer from the graph.
        
        Args:
            query: Original query
            
        Returns:
            Result dict with answer and metadata
        """
        # Find best path through graph (highest scoring path to leaves)
        leaf_nodes = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        
        if not leaf_nodes:
            return {
                "query": query,
                "answer": "I couldn't generate a complete reasoning path for this query.",
                "confidence": 0.0,
                "graph_stats": self._get_graph_stats()
            }
        
        # Find the best path (highest cumulative verification score)
        best_path = None
        best_score = -1
        
        for leaf in leaf_nodes:
            root = [n for n in self.graph.nodes() if self.graph.nodes[n].get("is_root", False)][0]
            
            try:
                paths = list(nx.all_simple_paths(self.graph, root, leaf))
                
                for path in paths:
                    # Calculate path score
                    scores = [self.graph.nodes[n].get("verification_score", 0.0) for n in path if "verification_score" in self.graph.nodes[n]]
                    
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_path = path
                            
            except nx.NetworkXNoPath:
                continue
        
        if not best_path:
            # Fall back to all nodes
            all_thoughts = [self.graph.nodes[n].get("thought", "") for n in self.graph.nodes() if "thought" in self.graph.nodes[n]]
        else:
            # Get thoughts from best path
            all_thoughts = [self.graph.nodes[n].get("thought", "") for n in best_path if "thought" in self.graph.nodes[n]]
        
        if not all_thoughts:
            return {
                "query": query,
                "answer": "I couldn't find sufficient information to answer this query.",
                "confidence": 0.0,
                "graph_stats": self._get_graph_stats()
            }
        
        # Aggregate thoughts into final answer
        combined_thoughts = "\n\n".join([f"{i+1}. {t}" for i, t in enumerate(all_thoughts)])
        
        # Get all sources
        all_sources = set()
        for node in self.graph.nodes():
            sources = self.graph.nodes[node].get("sources", [])
            all_sources.update(sources)
        
        # Check if query wants historical data
        wants_history = self._wants_historical_data(query)
        recency_instruction = "" if wants_history else "1. **CRITICAL: Unless the question explicitly asks for 'history', 'all', 'list of all', or 'over the years', provide ONLY the most recent/current information**\n2. For role-based queries (governors, directors, heads, members):\n   - If facts contain multiple years/sessions, identify and return ONLY the latest one\n   - Look for 'current', '2024-25', '2025-26', or the most recent year\n"
        
        prompt = f"""Synthesize a direct answer using the verified facts.

QUESTION: {query}

VERIFIED FACTS:
{combined_thoughts}

INSTRUCTIONS:
{recency_instruction}3. Answer directly and concisely
4. If facts contain names and years, format clearly (e.g., "The current governors (2024-25) are: X, Y, Z")
5. If facts don't answer the question, state: "The information was not found"

Provide your answer:"""
        
        try:
            final_answer = await self.groq_client.generate_judge(prompt, max_tokens=1024)
            
            return {
                "query": query,
                "answer": final_answer.strip(),
                "confidence": best_score,
                "sources": list(all_sources),
                "reasoning_path": [self.graph.nodes[n].get("thought", "") for n in best_path] if best_path else all_thoughts,
                "graph_stats": self._get_graph_stats()
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            return {
                "query": query,
                "answer": "Error synthesizing final answer.",
                "confidence": 0.0,
                "graph_stats": self._get_graph_stats()
            }
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        self.node_counter += 1
        return f"node_{self.node_counter:04d}"
    
    def _add_node_to_graph(self, node: ThoughtNode, parent_id: str):
        """Add thought node to graph"""
        self.graph.add_node(
            node.node_id,
            thought=node.thought,
            sub_query=node.sub_query,
            sources=node.sources,
            verification_score=node.verification_score,
            expert_remarks=node.expert_remarks,
            depth=node.depth,
            timestamp=node.timestamp
        )
        
        self.graph.add_edge(parent_id, node.node_id)
        logger.info(f"Added node {node.node_id} to graph (score: {node.verification_score:.2f})")
    
    def _merge_with_parent(self, parent_id: str, thought: str, verification: Dict):
        """Merge redundant thought with parent"""
        if parent_id in self.graph.nodes():
            current_thought = self.graph.nodes[parent_id].get("thought", "")
            merged_thought = f"{current_thought} {thought}"
            
            self.graph.nodes[parent_id]["thought"] = merged_thought
            self.graph.nodes[parent_id]["expert_remarks"] += f" | Merged: {verification['reasoning']}"
    
    def _get_parent_thoughts(self, node_id: str) -> List[str]:
        """Get all parent thoughts for a node"""
        thoughts = []
        
        try:
            root = [n for n in self.graph.nodes() if self.graph.nodes[n].get("is_root", False)][0]
            paths = list(nx.all_simple_paths(self.graph, root, node_id))
            
            if paths:
                path = paths[0]  # Take first path
                
                for n in path:
                    thought = self.graph.nodes[n].get("thought")
                    if thought:
                        thoughts.append(thought)
        except:
            pass
        
        return thoughts
    
    def _get_graph_stats(self) -> Dict:
        """Get graph statistics"""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "max_depth": max([self.graph.nodes[n].get("depth", 0) for n in self.graph.nodes()]) if self.graph.nodes() else 0,
            "avg_verification_score": sum([self.graph.nodes[n].get("verification_score", 0) for n in self.graph.nodes() if "verification_score" in self.graph.nodes[n]]) / max(1, len([n for n in self.graph.nodes() if "verification_score" in self.graph.nodes[n]]))
        }
    
    def _visualize_graph(self, query: str) -> str:
        """
        Generate interactive HTML visualization with Pyvis.
        
        Args:
            query: Query for filename
            
        Returns:
            Path to HTML file
        """
        try:
            # Create Pyvis network
            net = Network(
                height="800px",
                width="100%",
                directed=True,
                notebook=False,
                bgcolor="#1a1a1a",
                font_color="white"
            )
            
            # Configure physics
            net.set_options("""
            {
                "physics": {
                    "enabled": true,
                    "hierarchicalRepulsion": {
                        "centralGravity": 0.0,
                        "springLength": 200,
                        "springConstant": 0.01,
                        "nodeDistance": 150,
                        "damping": 0.09
                    },
                    "solver": "hierarchicalRepulsion"
                },
                "layout": {
                    "hierarchical": {
                        "enabled": true,
                        "direction": "UD",
                        "sortMethod": "directed"
                    }
                }
            }
            """)
            
            # Add nodes
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                
                if node_data.get("is_root"):
                    color = "#4A90E2"
                    label = f"ROOT\n{node_data.get('query', '')[:50]}"
                else:
                    score = node_data.get("verification_score", 0.5)
                    
                    if score >= 0.8:
                        color = "#50C878"  # Green
                    elif score >= 0.6:
                        color = "#FFD700"  # Yellow
                    else:
                        color = "#FF6347"  # Red
                    
                    thought = node_data.get("thought", "")
                    label = f"{node}\n{thought[:60]}..."
                
                title = f"""
                Node: {node}
                Thought: {node_data.get('thought', '')}
                Score: {node_data.get('verification_score', 0.0):.2f}
                Remarks: {node_data.get('expert_remarks', '')}
                """
                
                net.add_node(node, label=label, color=color, title=title, shape="box")
            
            # Add edges
            for edge in self.graph.edges():
                net.add_edge(edge[0], edge[1])
            
            # Save
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
            filename = f"graph_{query_hash}.html"
            filepath = self.graph_cache.cache_dir / filename
            
            net.save_graph(str(filepath))
            logger.info(f"Graph visualization saved to: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
            return ""

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        thought_count = 0
        verified_count = 0
        
        try:
            if self.thought_cache.cache_collection:
                thought_count = self.thought_cache.cache_collection.count()
                verified_count = thought_count  # All cached thoughts are verified
        except Exception:
            pass
        
        # Count cached graphs by counting JSON files in cache directory
        cached_graphs = 0
        try:
            cached_graphs = len(list(self.graph_cache.cache_dir.glob("*.json")))
        except Exception:
            pass
        
        # Calculate cache size
        cache_size_mb = 0.0
        try:
            cache_dir = self.graph_cache.cache_dir
            for f in cache_dir.glob("*.json"):
                cache_size_mb += f.stat().st_size / (1024 * 1024)
            for f in cache_dir.glob("*.html"):
                cache_size_mb += f.stat().st_size / (1024 * 1024)
        except Exception:
            pass
        
        return {
            "total_thoughts": thought_count,
            "verified_thoughts": verified_count,
            "cached_graphs": cached_graphs,
            "cache_size_mb": round(cache_size_mb, 2)
        }

    def clear_caches(self) -> Dict:
        """Clear all caches"""
        results = {"thoughts_cleared": 0, "graphs_cleared": 0}
        
        try:
            # Clear thought cache
            if self.thought_cache.cache_collection:
                count = self.thought_cache.cache_collection.count()
                if count > 0:
                    # Get all IDs and delete
                    all_ids = self.thought_cache.cache_collection.get()["ids"]
                    if all_ids:
                        self.thought_cache.cache_collection.delete(ids=all_ids)
                        results["thoughts_cleared"] = len(all_ids)
        except Exception as e:
            logger.error(f"Error clearing thought cache: {e}")
        
        try:
            # Clear graph cache - count and delete all graph files
            graph_files = list(self.graph_cache.cache_dir.glob("*.json"))
            results["graphs_cleared"] = len(graph_files)
            
            # Delete graph JSON files
            for f in graph_files:
                f.unlink()
            
            # Delete visualization HTML files
            for f in self.graph_cache.cache_dir.glob("*.html"):
                f.unlink()
        except Exception as e:
            logger.error(f"Error clearing graph cache: {e}")
        
        return results