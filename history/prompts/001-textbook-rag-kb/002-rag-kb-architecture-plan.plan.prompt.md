---
id: 002
title: RAG KB Architecture Plan
stage: plan
date: 2025-12-24
surface: agent
model: claude-opus-4-5-20251101
feature: 001-textbook-rag-kb
branch: 001-textbook-rag-kb
user: system
command: /sp.plan
labels: ["rag", "architecture", "embedding", "qdrant", "cohere"]
links:
  spec: specs/001-textbook-rag-kb/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-textbook-rag-kb/plan.md
  - specs/001-textbook-rag-kb/research.md
  - specs/001-textbook-rag-kb/data-model.md
  - specs/001-textbook-rag-kb/contracts/cli-interface.md
  - specs/001-textbook-rag-kb/quickstart.md
tests:
  - None (planning phase)
---

## Prompt

RAG Knowledge Base Construction – Part 1

Create:
- Architecture sketch for ingestion → embedding → vector storage
- Section structure for implementation
- Research-concurrent approach and quality validation

Architecture sketch:
- Project root
  - backend/
    - main.py (single entry point for all functionality)
- Docusaurus site URLs → content extraction
- Chunking (300–500 tokens)
- Cohere embeddings
- Qdrant Cloud vector storage

Initial project setup:
- Create a backend/ folder
- Initialize a uv Python package for the project
- All logic implemented in backend/main.py
- Processing flow: fetch URLs → chunk text → generate embeddings → store in Qdrant

Decisions needing documentation:
- URL discovery strategy (manual vs automated)
- Chunk size and overlap tradeoffs
- Cohere embedding model choice
- Qdrant collection schema and distance metric

Testing strategy:
- Validate URL fetching and content extraction
- Verify chunk sizes and metadata
- Confirm embedding dimensions
- Run sample vector search in Qdrant

## Response snapshot

Completed Phase 0 and Phase 1 of the /sp.plan workflow:

**Phase 0 - Research (research.md)**:
- URL Discovery: Sitemap.xml parsing primary, manual fallback
- Chunking: 300-500 tokens with 50-token overlap, paragraph-aware
- Cohere: embed-english-v3.0 (1024 dims), search_document/search_query types
- Qdrant: Cosine distance, textbook_chunks collection, full metadata payload
- Token counting: tiktoken with cl100k_base encoding
- HTML parsing: BeautifulSoup4 with selective content extraction

**Phase 1 - Design**:
- data-model.md: Page, Chunk, Embedding, QdrantPayload entities defined
- contracts/cli-interface.md: CLI commands (ingest, chunk, embed, store, verify, run)
- quickstart.md: Setup instructions and usage examples

**Constitution Check**: All 4 principles satisfied (Accuracy, Clarity, Reproducibility, Security)

## Outcome

- ✅ Impact: Architecture fully documented, ready for /sp.tasks
- 🧪 Tests: N/A (planning phase)
- 📁 Files: 5 files created (plan.md, research.md, data-model.md, cli-interface.md, quickstart.md)
- 🔁 Next prompts: /sp.tasks to generate implementation tasks
- 🧠 Reflection: Single-file architecture (main.py) keeps complexity low while supporting modular CLI

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): Standard /sp.plan workflow
- Next experiment (smallest change to try): Consider adding --dry-run flag for testing without API calls
