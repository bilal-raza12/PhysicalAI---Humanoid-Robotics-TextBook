# Implementation Plan: RAG Agent Integration

**Branch**: `003-rag-agent-integration` | **Date**: 2025-12-25 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-rag-agent-integration/spec.md`

## Summary

Build an AI agent using the OpenAI Agents SDK that connects to the existing Part 2 retrieval pipeline. The agent answers questions about the Physical AI & Humanoid Robotics textbook using only retrieved context, refuses when content is insufficient, and includes source citations in all responses.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: openai-agents (v0.6+), existing Part 2 deps (qdrant-client, cohere, typer)
**Storage**: Qdrant Cloud (existing collection from Part 1/2)
**Testing**: Manual CLI testing per contract
**Target Platform**: CLI (Windows/Linux/macOS)
**Project Type**: Single backend CLI (extends Part 2)
**Performance Goals**: <5 second response time (including retrieval + generation)
**Constraints**: Must use OpenAI Agents SDK, deterministic tool invocation, grounded responses only
**Scale/Scope**: Single agent, single retrieval tool, CLI interface only

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Accuracy**: Every chatbot answer includes citations to relevant page/section (FR-008)
- [x] **Clarity**: Technical terminology consistent with textbook content
- [x] **Reproducibility**: Agent behavior deterministic via `tool_choice="required"`
- [x] **Security**: No secrets in code (uses environment variables)
- [x] **Free Tier**: OpenAI API, Qdrant Cloud, Cohere API all within free tier limits

## Project Structure

### Documentation (this feature)

```text
specs/003-rag-agent-integration/
├── plan.md              # This file
├── spec.md              # Feature requirements (created)
├── research.md          # Research findings (created)
├── data-model.md        # Entity definitions (created)
├── quickstart.md        # Usage examples (created)
├── contracts/           # Interface contracts
│   └── cli-interface.md # CLI command specification (created)
└── checklists/
    └── requirements.md  # Quality checklist (created)
```

### Source Code (repository root)

```text
backend/
├── main.py              # Extended with agent code (Part 3 additions)
├── pyproject.toml       # Add openai-agents dependency
└── README.md            # Update with ask command docs
```

**Structure Decision**: Single file extension (backend/main.py) consistent with Part 1 and Part 2. All agent code added to existing file to maintain simplicity.

## Complexity Tracking

No constitution violations. Implementation follows minimal complexity approach:
- Single file extension (no new files)
- Wraps existing Part 2 functions (no duplication)
- Standard CLI pattern (consistent with verify, ingest, search commands)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLI (typer)                                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │ verify  │  │ ingest  │  │ search  │  │   ask   │ ◀── NEW (Part 3)   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                     │
└───────┼────────────┼────────────┼────────────┼──────────────────────────┘
        │            │            │            │
        │            │            │            ▼
        │            │            │    ┌───────────────┐
        │            │            │    │     Agent     │
        │            │            │    │ (OpenAI SDK)  │
        │            │            │    └───────┬───────┘
        │            │            │            │
        │            │            │            │ invokes
        │            │            │            ▼
        │            │            │    ┌───────────────┐
        │            │            └───▶│ RetrievalTool │
        │            │                 │ (@function_   │
        │            │                 │   tool)       │
        │            │                 └───────┬───────┘
        │            │                         │
        │            │                         │ wraps
        │            │                         ▼
        │            │              ┌──────────────────┐
        │            └─────────────▶│ Part 2 Functions │
        │                           │ search_knowledge │
        │                           │ _base()          │
        └──────────────────────────▶│ validate_query() │
                                    │ validate_k()     │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │   Qdrant Cloud   │
                                    │  (textbook_      │
                                    │   chunks)        │
                                    └──────────────────┘
```

## Key Technical Decisions

### D1: OpenAI Agents SDK with @function_tool

**Decision**: Use `@function_tool` decorator to wrap retrieval as an agent tool.

**Rationale**:
- Clean decorator pattern for tool definition
- Automatic schema generation from type hints
- Built-in error handling

**Implementation**:
```python
from agents import Agent, function_tool, ModelSettings, Runner

@function_tool
def search_textbook(query: str) -> str:
    """Search the Physical AI & Humanoid Robotics textbook."""
    result, error = search_knowledge_base(query, k=5)
    if error:
        return f"[ERROR] {error}"
    return format_context_for_agent(result)
```

### D2: Deterministic Tool Invocation

**Decision**: Use `ModelSettings(tool_choice="required")` to force retrieval.

**Rationale**:
- Ensures agent ALWAYS queries the knowledge base
- Prevents reliance on model's pre-training knowledge
- Makes behavior predictable and auditable

**Implementation**:
```python
agent = Agent(
    name="textbook_qa",
    instructions=GROUNDING_PROMPT,
    tools=[search_textbook],
    model_settings=ModelSettings(tool_choice="required")
)
```

### D3: Grounding System Prompt

**Decision**: Use explicit grounding instructions in agent system prompt.

**Rationale**:
- Constrains agent to retrieved context only
- Defines clear refusal rules (score < 0.3)
- Mandates source citations

**Key Rules**:
1. ALWAYS use search_textbook tool before answering
2. ONLY use information from retrieved context
3. REFUSE if no relevant content or scores below 0.3
4. Include source citations in every answer

### D4: Score Threshold 0.3

**Decision**: Refuse to answer when all retrieved chunks score below 0.3.

**Rationale**:
- 0.3 is empirically reasonable for cosine similarity with Cohere embeddings
- Prevents answering from weakly related content
- Balances recall (answering when possible) with precision (not hallucinating)

### D5: Wrap Part 2 Functions (No Duplication)

**Decision**: The retrieval tool wraps existing `search_knowledge_base()` from Part 2.

**Rationale**:
- Part 2 is fully tested (41/41 tasks complete)
- Avoids code duplication
- Single source of truth for retrieval logic

## Implementation Phases

### Phase 1: Agent Foundation

- Add openai-agents dependency
- Define AgentConfig dataclass
- Create grounding system prompt constant
- Implement `@function_tool` wrapper for search_knowledge_base

### Phase 2: Agent Implementation

- Create Agent instance with tools and settings
- Implement async `ask_agent()` function
- Add grounding decision logic (should_refuse)
- Format context for agent consumption

### Phase 3: CLI Integration

- Add `ask` command to CLI
- Implement text and JSON output formats
- Add --verbose, --k, --threshold options
- Handle async execution with asyncio.run()

### Phase 4: Response Formatting

- Implement Citation dataclass
- Create AgentResponse structure
- Format source citations consistently
- Add refusal message templates

### Phase 5: Error Handling & Polish

- Handle retrieval errors gracefully
- Handle OpenAI API errors
- Add logging for debugging
- Update README with ask command docs

## Test Strategy

Manual CLI testing per contract (specs/003-rag-agent-integration/contracts/cli-interface.md):

| Test | Command | Expected |
|------|---------|----------|
| TC-001 | `ask "What is ROS 2?"` | Grounded answer with citations |
| TC-002 | `ask "capital of France?"` | Refusal message |
| TC-003 | `ask "ROS 2" --format json` | Valid JSON output |
| TC-004 | `ask "ROS 2" --k 3 --verbose` | 3 chunks shown |
| TC-005 | `ask ""` | Error (empty query) |
| TC-006 | `ask "ROS 2" --verbose` | Timing and scores shown |
| TC-007 | `ask "quantum" --threshold 0.1` | Lower bar for answers |
| TC-008 | `ask "ROS 2" --threshold 0.9` | May refuse (high bar) |

## Dependencies

### Existing (Part 1/2)

- qdrant-client
- cohere
- typer
- tiktoken
- httpx
- beautifulsoup4

### New (Part 3)

- openai-agents

### External Services

- OpenAI API (for agent)
- Cohere API (for embeddings - existing)
- Qdrant Cloud (for vector search - existing)

## Success Criteria Mapping

| Criterion | Implementation |
|-----------|----------------|
| SC-001: 90% accuracy | Grounding prompt + retrieved context |
| SC-002: 100% refusal when irrelevant | Score threshold 0.3 + grounding rules |
| SC-003: <5s response time | Async execution, single retrieval call |
| SC-004: 100% citations | Mandatory in grounding prompt |
| SC-005: 100% tool invocation | tool_choice="required" |
| SC-006: Zero hallucinations | Grounding prompt + context-only responses |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| OpenAI API rate limits | Use gpt-4o-mini (higher limits), implement retry |
| Slow response time | Measure timing, optimize if needed |
| Agent ignores grounding | Test with off-topic questions, validate refusals |
| Part 2 breaking changes | Pin dependencies, integration test on startup |

## Out of Scope (per spec)

- Frontend UI
- FastAPI endpoints
- Streaming responses
- Multi-turn conversation memory
- Custom embedding models
- Performance optimization
