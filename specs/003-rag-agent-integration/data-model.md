# Data Model: RAG Agent Integration

**Feature**: 003-rag-agent-integration | **Date**: 2025-12-25

## Entity Overview

This feature adds agent-specific entities while reusing Part 2 retrieval dataclasses.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG Agent Data Flow                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   User Question                                                         │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                         Agent                                   │   │
│   │  ┌──────────────┐  ┌────────────────┐  ┌─────────────────────┐  │   │
│   │  │ Instructions │  │ ModelSettings  │  │  RetrievalTool     │  │   │
│   │  │ (grounding)  │  │ (tool_choice)  │  │  (search_textbook) │  │   │
│   │  └──────────────┘  └────────────────┘  └─────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│        │                                                                │
│        │ invokes                                                        │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    RetrievalTool                                │   │
│   │  ┌──────────────────────┐  ┌────────────────────────────────┐   │   │
│   │  │ search_knowledge_base│  │ GroundingContext              │   │   │
│   │  │ (from Part 2)        │──▶│ (formatted chunks + decision) │   │   │
│   │  └──────────────────────┘  └────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│        │                                                                │
│        │ returns                                                        │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                     AgentResponse                               │   │
│   │  ┌──────────────┐  ┌────────────────┐  ┌─────────────────────┐  │   │
│   │  │    answer    │  │   citations    │  │    tool_calls      │  │   │
│   │  └──────────────┘  └────────────────┘  └─────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## New Entities (Part 3)

### AgentConfig

Configuration for the RAG agent instance.

```python
@dataclass
class AgentConfig:
    """Configuration for the RAG agent."""

    # Model settings
    model: str = "gpt-4o-mini"
    temperature: float = 0.0  # Deterministic for grounding

    # Retrieval settings
    k: int = 5  # Number of chunks to retrieve
    score_threshold: float = 0.3  # Below this = refuse

    # Tool settings
    tool_choice: str = "required"  # Always invoke retrieval

    # Collection
    collection_name: str = "textbook_chunks"
```

**Field Constraints:**
- `k`: 3-8 (per Part 2 validation)
- `score_threshold`: 0.0-1.0
- `tool_choice`: "required" | "auto" | "none"
- `temperature`: 0.0-2.0

---

### GroundingContext

Processed context from retrieval with grounding decision.

```python
@dataclass
class GroundingContext:
    """Context assembled for agent grounding."""

    # Retrieved content
    chunks: list[RetrievedChunk]  # From Part 2
    formatted_text: str  # For agent consumption

    # Grounding decision
    should_refuse: bool
    refusal_reason: str = ""

    # Metadata
    max_score: float = 0.0
    query: str = ""

    @property
    def source_count(self) -> int:
        """Number of unique sources."""
        return len(set(c.source_url for c in self.chunks))
```

**Computed Properties:**
- `source_count`: Unique source URLs
- `has_context`: `len(chunks) > 0 and not should_refuse`

---

### Citation

Individual source citation for transparency.

```python
@dataclass
class Citation:
    """Source citation for a retrieved chunk."""

    source_url: str
    chapter: str
    section: str
    score: float
    chunk_index: int

    def format(self) -> str:
        """Format as inline citation."""
        return f"[Source: {self.source_url} | Chapter: {self.chapter} | Section: {self.section} | Score: {self.score:.2f}]"
```

---

### AgentResponse

Structured response from the agent.

```python
@dataclass
class AgentResponse:
    """Structured response from the RAG agent."""

    # Response content
    answer: str
    citations: list[Citation]

    # Metadata
    query: str
    grounded: bool  # True if answer uses retrieved context
    refused: bool  # True if agent refused to answer

    # Debug info
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    tool_calls: list[str] = field(default_factory=list)

    @property
    def total_time_ms(self) -> float:
        return self.retrieval_time_ms + self.generation_time_ms
```

---

## Reused Entities (from Part 2)

These entities are defined in `backend/main.py` and imported without modification:

### RetrievedChunk (Part 2)

```python
@dataclass
class RetrievedChunk:
    """A chunk retrieved from Qdrant with its metadata and score."""

    text: str
    score: float
    source_url: str
    chunk_index: int
    chunk_id: str
    chapter: str = ""
    section: str = ""
    title: str = ""
```

### RetrievalResult (Part 2)

```python
@dataclass
class RetrievalResult:
    """Collection of retrieved chunks."""

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    collection: str = "textbook_chunks"

    @property
    def count(self) -> int:
        return len(self.chunks)
```

### AssembledContext (Part 2)

```python
@dataclass
class AssembledContext:
    """Formatted context for downstream use."""

    formatted_text: str
    chunk_count: int
    total_chars: int
    sources: list[str] = field(default_factory=list)
```

---

## Entity Relationships

```
AgentConfig (1) ────────── defines ────────── (1) Agent
       │
       │ configures
       ▼
RetrievalTool (1) ──── wraps ──── (1) search_knowledge_base (Part 2)
       │
       │ produces
       ▼
RetrievalResult (1) ──────── contains ──────── (*) RetrievedChunk
       │
       │ transforms to
       ▼
GroundingContext (1) ──────── decides ──────── (1) should_refuse
       │
       │ used by Agent to produce
       ▼
AgentResponse (1) ──────── contains ──────── (*) Citation
```

---

## Validation Rules

### AgentConfig Validation

```python
def validate_config(config: AgentConfig) -> tuple[AgentConfig, Optional[str]]:
    """Validate agent configuration."""
    if config.k < 3 or config.k > 8:
        return config, f"K must be 3-8, got {config.k}"

    if config.score_threshold < 0.0 or config.score_threshold > 1.0:
        return config, f"Score threshold must be 0.0-1.0, got {config.score_threshold}"

    if config.tool_choice not in ("required", "auto", "none"):
        return config, f"Invalid tool_choice: {config.tool_choice}"

    return config, None
```

### Grounding Decision

```python
def decide_grounding(result: RetrievalResult, threshold: float) -> GroundingContext:
    """Decide whether to ground or refuse based on retrieval results."""
    if result.count == 0:
        return GroundingContext(
            chunks=[],
            formatted_text="",
            should_refuse=True,
            refusal_reason="No content retrieved from knowledge base",
            query=result.query
        )

    max_score = max(c.score for c in result.chunks)

    if max_score < threshold:
        return GroundingContext(
            chunks=result.chunks,
            formatted_text="",
            should_refuse=True,
            refusal_reason=f"All retrieved content has low relevance (max score: {max_score:.2f} < threshold: {threshold})",
            max_score=max_score,
            query=result.query
        )

    # Grounding approved
    formatted = format_chunks_for_agent(result.chunks)
    return GroundingContext(
        chunks=result.chunks,
        formatted_text=formatted,
        should_refuse=False,
        max_score=max_score,
        query=result.query
    )
```

---

## Constants

```python
# Grounding thresholds
DEFAULT_SCORE_THRESHOLD = 0.3
MIN_CHUNKS_FOR_ANSWER = 1

# Retrieval defaults
DEFAULT_K = 5
MAX_K = 8
MIN_K = 3

# Response constraints
MAX_ANSWER_LENGTH = 2000  # Characters
MAX_CITATIONS = 5

# Refusal messages
REFUSAL_NO_CONTEXT = "I cannot answer this question based on the textbook content. No relevant information was found."
REFUSAL_LOW_SCORE = "I cannot answer this question with confidence. The retrieved content has low relevance to your question."
REFUSAL_OFF_TOPIC = "This question appears to be outside the scope of the Physical AI & Humanoid Robotics textbook."
```

---

## Notes

- All Part 2 entities remain unchanged
- New entities are additive (no modifications to Part 2)
- `GroundingContext` bridges Part 2 retrieval with Part 3 agent
- `AgentResponse` is the external contract for CLI consumers
