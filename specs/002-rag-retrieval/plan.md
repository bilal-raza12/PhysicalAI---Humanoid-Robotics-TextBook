# Implementation Plan: RAG Retrieval Pipeline

**Branch**: `002-rag-retrieval` | **Date**: 2025-12-25 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-rag-retrieval/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build a semantic retrieval system that extends the existing RAG pipeline (Part 1) with a `search` command. The system accepts natural language queries, embeds them using Cohere embed-english-v3.0 with input_type="search_query", retrieves top-K chunks from Qdrant using cosine similarity, and returns assembled context with full metadata for downstream use.

## Technical Context

**Language/Version**: Python 3.11+ (managed via uv)
**Primary Dependencies**: cohere (embeddings), qdrant-client (vector search), existing Part 1 dependencies
**Storage**: Qdrant Cloud (existing "textbook_chunks" collection from Part 1)
**Testing**: Manual CLI testing (unit tests optional)
**Target Platform**: Local CLI / CI pipeline (cross-platform)
**Project Type**: single (backend-only CLI tool, extending Part 1)
**Performance Goals**: Query response in under 3 seconds
**Constraints**: Cohere trial rate limits (100k tokens/min), Qdrant Cloud free tier
**Scale/Scope**: 268 indexed chunks, K=3-8 results per query

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Accuracy ✅
- **Requirement**: All content must trace back to source sections
- **Compliance**: Every result includes source_url, chapter, section metadata
- **Implementation**: Metadata returned with each chunk enables citation

### Principle II: Clarity ✅
- **Requirement**: Clean structure and readability
- **Compliance**: CLI output clearly formatted with scores and metadata
- **Implementation**: Text and JSON output formats for different use cases

### Principle III: Reproducibility ✅
- **Requirement**: Automated and reproducible workflows
- **Compliance**: Same query returns same results (deterministic retrieval)
- **Implementation**: CLI command with consistent parameters

### Principle IV: Security ✅
- **Requirement**: No secrets in repo, protected data
- **Compliance**: API keys via environment variables
- **Implementation**: .env file for credentials, not committed

### RAG Requirements Alignment ✅
- **Retrieval**: Vector similarity search with cosine distance
- **Context Window**: Chunk metadata included for citation
- **Query Embedding**: search_query input type per Cohere best practices

### Gate Status: **PASSED** - All constitution principles satisfied

## Project Structure

### Documentation (this feature)

```text
specs/002-rag-retrieval/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
│   └── cli-interface.md # CLI contract
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── main.py              # Extended with search command
├── pyproject.toml       # uv package configuration (no changes)
├── .env.example         # Environment variable template (no changes)
└── tests/               # Test files (if added later)
```

**Structure Decision**: Extends existing backend structure from Part 1. New `search` command added to `main.py`. No new files required - retrieval logic integrated into existing codebase.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations detected. Implementation follows minimal complexity principles:
- Single file extension (main.py)
- Reuses existing client initialization functions
- CLI pattern consistent with Part 1 commands

## Architecture Sketch

```text
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
│                 "What is ROS 2?"                             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Query Validation                            │
│            - Non-empty check                                 │
│            - K parameter clamping (3-8)                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                Cohere Embedding                              │
│            - Model: embed-english-v3.0                       │
│            - input_type: "search_query"                      │
│            - Output: 1024-dim vector                         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Qdrant Similarity Search                        │
│            - Collection: textbook_chunks                     │
│            - Distance: cosine                                │
│            - Limit: K results                                │
│            - With payload: true                              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                Context Assembly                              │
│            - Order by score (descending)                     │
│            - Include metadata per chunk                      │
│            - Format: text or JSON                            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output                                    │
│            - Formatted results                               │
│            - Exit code (0/1/2)                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default K | 5 | Balance between context and relevance |
| K Range | 3-8 | Prevents too few/too many results |
| Score Filtering | None | Caller responsibility for flexibility |
| Output Formats | text, json | Human and programmatic use |
| Error Handling | Structured codes | Enables programmatic error handling |

## Phase 0 Artifacts

- **research.md**: Technical decisions on K selection, threshold handling, error handling

## Phase 1 Artifacts

- **data-model.md**: Query, RetrievedChunk, RetrievalResult, AssembledContext entities
- **contracts/cli-interface.md**: search command specification with test cases
- **quickstart.md**: Usage examples and integration scenarios

## Next Steps

Run `/sp.tasks` to generate the implementation task breakdown.
