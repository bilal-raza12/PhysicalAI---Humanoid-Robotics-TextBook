# Implementation Plan: RAG Knowledge Base Construction

**Branch**: `001-textbook-rag-kb` | **Date**: 2025-12-24 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-textbook-rag-kb/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build a RAG knowledge base pipeline that ingests content from a deployed Docusaurus textbook website, processes text into 300-500 token chunks, generates vector embeddings using Cohere embed-english-v3.0, and stores them in Qdrant Cloud for similarity search. All functionality implemented in a single `backend/main.py` entry point using uv Python package manager.

## Technical Context

**Language/Version**: Python 3.11+ (managed via uv)
**Primary Dependencies**: httpx (HTTP client), beautifulsoup4 (HTML parsing), cohere (embeddings), qdrant-client (vector storage), tiktoken (token counting)
**Storage**: Qdrant Cloud (vector database, 1024 dimensions, cosine distance)
**Testing**: pytest (unit and integration tests)
**Target Platform**: Local CLI / CI pipeline (cross-platform)
**Project Type**: single (backend-only CLI tool)
**Performance Goals**: Process all textbook pages in single run, handle rate limits gracefully
**Constraints**: Free-tier Qdrant Cloud limits, Cohere API rate limits, network resilience required
**Scale/Scope**: ~40+ pages, ~1000+ chunks estimated, single Qdrant collection

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Accuracy ✅
- **Requirement**: All content must trace to source sections
- **Compliance**: Pipeline preserves source URL, chapter, section metadata for every chunk
- **Implementation**: Metadata attached to vectors enables citation tracing

### Principle II: Clarity ✅
- **Requirement**: Clean structure and readability
- **Compliance**: Single entry point (`main.py`), clear processing flow, structured output

### Principle III: Reproducibility ✅
- **Requirement**: Automated and reproducible workflows
- **Compliance**: CLI-based pipeline, environment variables for secrets, deterministic chunking
- **Implementation**: Pipeline can be re-run to regenerate knowledge base

### Principle IV: Security ✅
- **Requirement**: No secrets in repo, protected data
- **Compliance**: API keys via environment variables, no user data storage
- **Implementation**: `.env` file for local development, CI secrets for deployment

### RAG Requirements Alignment ✅
- **Chunking**: Deterministic 300-500 tokens with metadata
- **Storage Schema**: Qdrant vectors with source_url, chapter, section, chunk_index payload
- **Retrieval**: Vector similarity search enabled

### Gate Status: **PASSED** - All constitution principles satisfied

## Project Structure

### Documentation (this feature)

```text
specs/001-textbook-rag-kb/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── main.py              # Single entry point for all pipeline functionality
├── pyproject.toml       # uv package configuration
├── .env.example         # Environment variable template
└── tests/
    ├── test_ingestion.py    # URL fetching and content extraction tests
    ├── test_chunking.py     # Chunk size and metadata tests
    ├── test_embedding.py    # Cohere API integration tests
    └── test_storage.py      # Qdrant storage and search tests
```

**Structure Decision**: Single-project backend structure selected per user requirement. All logic in `backend/main.py` with uv package management. Tests organized by pipeline stage.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations detected. Implementation follows minimal complexity principles:
- Single file entry point as requested
- Direct API calls without abstraction layers
- Flat project structure
