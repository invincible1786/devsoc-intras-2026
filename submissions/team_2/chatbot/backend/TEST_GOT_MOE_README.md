# Graph of Thoughts + Mixture of Experts Implementation

## Overview

This implementation follows the **IMPLEMENTATION_GUIDE.md** specifications for a complete Graph of Thoughts (GoT) + Mixture of Experts (MoE) RAG system.

## Architecture

### 1. **Graph of Thoughts (GoT)**
- **Planner Agent**: Decomposes complex queries into atomic sub-questions with acronym expansion
- **Execution Agent**: Generates 3 competing reasoning paths per sub-question:
  - Path 1: Primary source (most relevant chunk)
  - Path 2: Multi-source synthesis (top 3 chunks)
  - Path 3: Temporal filter (for "current" queries)
- **Synthesis Agent**: Combines verified thoughts into coherent final answer

### 2. **Mixture of Experts (MoE)**
Three specialized verification agents evaluate each reasoning path:
- **Source Matcher**: Verifies claim is directly supported by context
- **Hallucination Hunter**: Detects invented information not in source
- **Logic Expert**: Ensures logical coherence of reasoning

**Consensus Mechanism**: All three experts must agree (with >0.6 confidence) for a path to be verified.

### 3. **RAG with Filtering**
- Retrieves **top 45 chunks** from ChromaDB
- Filters to **top 10 most relevant** using cosine similarity with query embedding
- Uses Google embeddings via Modal

### 4. **Dynamic Knowledge Graph**
- NetworkX graph stores verified question-answer pairs
- Persists to disk at `./cache/metakgp_graph.gml`
- Builds permanent memory of reasoning paths over time

## Data Structures

### ThoughtNode
```python
- id: int
- question: str              # Sub-question
- retrieved_context: str     # RAG context
- derived_thought: str       # Final answer
- verified: bool             # Passed MoE?
- score: int                 # Confidence (0-10)
- reasoning_paths: List      # Multiple competing paths
```

### ReasoningPath
```python
- path_id: int
- claim: str                 # Proposed answer
- context: str               # Source text
- source_info: str           # Metadata
- Expert verdicts (populated by MoE):
  - source_match_verdict, source_match_conf
  - halluc_verdict, halluc_conf
  - logic_verdict, logic_conf
- is_verified: bool
- final_score: float
```

## Pipeline Flow

```
User Query
    ↓
[1] Planner Agent
    → Decomposes into sub-questions
    → Expands acronyms (TFPS → Technology Film and Photography Society)
    ↓
[2] For Each Sub-Question:
    ├─ Execution Agent
    │   ├─ Retrieve top 45 chunks from ChromaDB
    │   ├─ Filter to top 10 by cosine similarity
    │   └─ Generate 3 reasoning paths:
    │       • Path 1: Primary source
    │       • Path 2: Multi-source synthesis
    │       • Path 3: Temporal filter (if "current")
    │
    ├─ Verification Agent (MoE)
    │   ├─ Run 3 experts on each path:
    │   │   • Source Matcher
    │   │   • Hallucination Hunter
    │   │   • Logic Expert
    │   ├─ Compute consensus score
    │   └─ Select best verified path
    │
    └─ Graph Learning
        └─ Add verified thought to NetworkX graph
    ↓
[3] Synthesis Agent
    → Combines all verified thoughts
    → Produces final answer with citations
    ↓
Final Answer
```

## Key Features

### 1. **Multi-Path Reasoning**
Unlike single-path systems, we generate multiple reasoning trajectories and let them compete through MoE evaluation.

### 2. **Strict Verification**
All three experts must agree:
- Source Matcher: "Is this in the context?"
- Hallucination Hunter: "Is this invented?"
- Logic Expert: "Does this make sense?"

### 3. **Temporal Awareness**
Special path for "current" queries filters for 2025 information.

### 4. **Graph Memory**
Verified thoughts are stored in a persistent graph, enabling:
- Future query optimization
- Contradiction detection
- Reasoning path visualization

## Usage

### Running Tests
```bash
cd /home/arshgoyal/Engineer/Developer/Socs/DevSoc/devsoc-intras-2026/submissions/team_2/chatbot/backend
python test_got_moe.py
```

### Programmatic Usage
```python
from test_got_moe import generate_response_got, KnowledgeGraph
from src.utils.chroma_client import MetaKGPChromaClient
from src.utils.embedding_client import ModalEmbeddingClient
from src.utils.groq_client import GroqClient

# Initialize
chroma_client = MetaKGPChromaClient()
embedding_client = ModalEmbeddingClient()
groq_client = GroqClient()
graph = KnowledgeGraph()

# Query
answer = generate_response_got(
    "Who is the VP of TFPS?",
    chroma_client,
    embedding_client,
    groq_client,
    graph
)
```

## Example Query Flow

**Input**: "Who is the VP of TFPS and when was it founded?"

**Step 1 - Planner**:
```
Sub-questions:
1. "Who is the Vice President of Technology Film and Photography Society?"
2. "When was Technology Film and Photography Society founded?"
```

**Step 2 - Execution (Sub-Q1)**:
```
Retrieved 45 chunks → Filtered to top 10
Generated 3 paths:
  Path 1 (Primary): "Arjun Kumar is VP"
  Path 2 (Multi-src): "Arjun Kumar serves as VP"
  Path 3 (Temporal): "Current VP (2025): Arjun Kumar"
```

**Step 3 - Verification (MoE)**:
```
Path 3 evaluation:
  Source Matcher:      ✓ YES (0.98)
  Hallucination:       ✓ NO  (0.95)
  Logic Expert:        ✓ YES (0.90)
  Final Score:         0.94 → VERIFIED ✓
```

**Step 4 - Synthesis**:
```
"The current Vice President of Technology Film and Photography Society 
is Arjun Kumar (Source: Temporal Filter, 2025). TFPS was founded in 1982 
(Source: Primary)."
```

## Configuration

### Confidence Thresholds
- **MoE Verification**: 0.6 (consensus score must exceed)
- **Chunk Filtering**: Top 10 from 45 retrieved

### Models
- **Judge Model** (Planner, Execution, Synthesis): `llama-4-scout-17b-16e-instruct`
- **Expert Model** (MoE): `llama-4-scout-17b-16e-instruct`
- **Embeddings**: Google `text-embedding-004`

## Files Modified

1. **test_got_moe.py** (NEW)
   - Complete GoT + MoE implementation
   - ~1100 lines of code
   - All agents and data structures

2. **pyproject.toml**
   - Added `numpy>=1.24.0` dependency

## Differences from Existing Code

The existing `graph_engine.py` and `moe.py` have similar concepts but this implementation:

1. **Follows the guide exactly**: Matches the IMPLEMENTATION_GUIDE.md structure
2. **Simpler RAG filtering**: Uses top 45 → filter to 10 by cosine similarity
3. **Three reasoning paths**: Primary, Multi-source, Temporal (not configurable branches)
4. **Stricter MoE**: All three experts must agree
5. **Cleaner code**: ~1100 lines vs 1300+ in existing implementation

## Testing

Run the included test suite:
```bash
python test_got_moe.py
```

Test queries:
- "Who is the Vice President of Technology Film and Photography Society?"
- "When was TFPS founded?"
- "List the current General Secretaries of TSG for 2025"

## Extending

### Add More Reasoning Paths
In `execution_agent()`, add additional path generation logic:
```python
# Path 4: Cross-reference
if len(filtered_docs) >= 5:
    path4_claim = generate_cross_reference(...)
    node.reasoning_paths.append(ReasoningPath(...))
```

### Add 4th Expert
Create a new expert function:
```python
def reflection_expert(claim, all_paths, groq_client):
    """Check if answer contradicts other verified paths"""
    ...
```

### Tune Thresholds
Adjust in the code:
- `path.final_score > 0.6` → Change verification threshold
- `filter_chunks_by_relevance(..., top_k=10)` → Change filtering

## Performance

- **Average query time**: 15-30 seconds (3 sub-questions, 3 paths each, 3 experts)
- **Graph growth**: ~1-3 nodes per query
- **Memory**: Minimal (graph stored on disk)

## Troubleshooting

### "No documents retrieved"
- Check ChromaDB has data: `chroma_client.collection.count()`
- Verify embeddings are working: `embedding_client("test")`

### "All paths failed verification"
- Lower threshold: `path.final_score > 0.5` instead of 0.6
- Check expert prompts are working
- Review retrieved context quality

### "Synthesis produces gibberish"
- Check that nodes have `verified=True`
- Ensure `derived_thought` is populated
- Review synthesis prompt

## References

- **IMPLEMENTATION_GUIDE.md**: Complete technical specification
- **Graph of Thoughts Paper**: [arxiv.org/abs/2305.16582](https://arxiv.org/abs/2305.16582)
- **Mixture of Experts**: [arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538)
