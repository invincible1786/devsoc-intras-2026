"""
Data Structures for Graph of Thoughts + Mixture of Experts
"""

from dataclasses import dataclass, field
from typing import List, Optional


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
