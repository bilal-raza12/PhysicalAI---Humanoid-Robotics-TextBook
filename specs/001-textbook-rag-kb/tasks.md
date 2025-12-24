# Tasks: RAG Knowledge Base Construction

**Input**: Design documents from `/specs/001-textbook-rag-kb/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli-interface.md, quickstart.md
**Branch**: `001-textbook-rag-kb`

**Tests**: Not explicitly requested in spec - tasks focus on implementation only.

**Organization**: Tasks grouped by user story. All P1 stories form the core pipeline; US5 (P2) adds verification.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US5)
- All file paths are relative to repository root

## Path Conventions

```text
backend/
├── main.py              # Single entry point
├── pyproject.toml       # uv package config
├── .env.example         # Environment template
└── tests/               # Test files (if added later)
```

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Initialize the backend Python project with uv and configure dependencies

- [x] T001 Create `backend/` directory at repository root
- [x] T002 Initialize uv Python project in `backend/` with `uv init`
- [x] T003 Add dependencies: httpx, beautifulsoup4, cohere, qdrant-client, tiktoken, python-dotenv in `backend/pyproject.toml`
- [x] T004 Create `backend/.env.example` with COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY placeholders
- [x] T005 [P] Add .gitignore entries for `.env`, `*.json` data files, `__pycache__/`

---

## Phase 2: Foundational (Core Data Structures)

**Purpose**: Define data models and CLI framework that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Define PageStatus enum and Page dataclass in `backend/main.py`
- [x] T007 Define Chunk dataclass with metadata fields in `backend/main.py`
- [x] T008 Define EmbeddingResult dataclass in `backend/main.py`
- [x] T009 Implement CLI argument parser with subcommands (ingest, chunk, embed, store, verify, run) in `backend/main.py`
- [x] T010 [P] Implement environment variable loading with python-dotenv in `backend/main.py`
- [x] T011 [P] Implement logging configuration (INFO default, configurable via LOG_LEVEL) in `backend/main.py`
- [x] T012 Implement JSON serialization helpers for Page/Chunk/Embedding dataclasses in `backend/main.py`

**Checkpoint**: Foundation ready - CLI skeleton accepts all commands, data models defined

---

## Phase 3: User Story 1 - Ingest Textbook Content (Priority: P1) 🎯 MVP

**Goal**: Crawl and ingest all public pages from the Docusaurus textbook site

**Independent Test**: Run `python main.py ingest` against the live textbook URL and verify pages.json contains all expected pages with clean text

### Implementation for User Story 1

- [x] T013 [US1] Implement sitemap.xml fetching with httpx in `backend/main.py`
- [x] T014 [US1] Implement sitemap XML parsing to extract all page URLs in `backend/main.py`
- [x] T015 [US1] Implement fallback URL list for when sitemap is unavailable in `backend/main.py`
- [x] T016 [US1] Implement HTML content fetching with retry logic and exponential backoff in `backend/main.py`
- [x] T017 [US1] Implement BeautifulSoup HTML parsing to extract main content (remove nav/sidebar/footer) in `backend/main.py`
- [x] T018 [US1] Implement chapter/section extraction from URL path and headings in `backend/main.py`
- [x] T019 [US1] Implement page title extraction from `<title>` or `<h1>` in `backend/main.py`
- [x] T020 [US1] Implement error handling for 404s, timeouts, and empty pages in `backend/main.py`
- [x] T021 [US1] Implement `ingest` command that fetches all pages and saves to pages.json in `backend/main.py`
- [x] T022 [US1] Add console summary output: pages fetched, errors, skipped in `backend/main.py`

**Checkpoint**: `python main.py ingest` produces pages.json with all textbook content

---

## Phase 4: User Story 2 - Generate Searchable Chunks (Priority: P1)

**Goal**: Split extracted content into 300-500 token chunks with metadata

**Independent Test**: Run `python main.py chunk` on pages.json and verify chunks.json has proper token sizes (300-500) and metadata

### Implementation for User Story 2

- [x] T023 [US2] Implement token counting using tiktoken with cl100k_base encoding in `backend/main.py`
- [x] T024 [US2] Implement paragraph-aware text splitting (split on \n\n) in `backend/main.py`
- [x] T025 [US2] Implement chunk merging logic to reach 300-500 token target in `backend/main.py`
- [x] T026 [US2] Implement 50-token overlap between consecutive chunks in `backend/main.py`
- [x] T027 [US2] Implement chunk metadata assignment (source_url, chapter, section, chunk_index) in `backend/main.py`
- [x] T028 [US2] Implement edge case handling: chunks below minimum, boundary chunks in `backend/main.py`
- [x] T029 [US2] Implement `chunk` command that reads pages.json and outputs chunks.json in `backend/main.py`
- [x] T030 [US2] Add console summary: total chunks, average token count, token distribution in `backend/main.py`

**Checkpoint**: `python main.py chunk` produces chunks.json with properly sized and annotated chunks

---

## Phase 5: User Story 3 - Create Vector Embeddings (Priority: P1)

**Goal**: Generate Cohere embeddings for all chunks with rate limiting

**Independent Test**: Run `python main.py embed` on chunks.json and verify embeddings.json has 1024-dimensional vectors for all chunks

### Implementation for User Story 3

- [x] T031 [US3] Implement Cohere client initialization with API key from environment in `backend/main.py`
- [x] T032 [US3] Implement batch embedding generation (96 texts per request max) in `backend/main.py`
- [x] T033 [US3] Implement embedding call with input_type="search_document" in `backend/main.py`
- [x] T034 [US3] Implement rate limiting with exponential backoff (start at 1s) in `backend/main.py`
- [x] T035 [US3] Implement retry logic for failed API calls in `backend/main.py`
- [x] T036 [US3] Implement embedding dimension validation (must be 1024) in `backend/main.py`
- [x] T037 [US3] Implement `embed` command that reads chunks.json and outputs embeddings.json in `backend/main.py`
- [x] T038 [US3] Add console summary: chunks embedded, API calls made, retries in `backend/main.py`

**Checkpoint**: `python main.py embed` produces embeddings.json with 1024-dim vectors for all chunks

---

## Phase 6: User Story 4 - Store and Search Vectors (Priority: P1)

**Goal**: Store embeddings in Qdrant Cloud and enable similarity search

**Independent Test**: Run `python main.py store` then verify vectors in Qdrant Cloud console

### Implementation for User Story 4

- [x] T039 [US4] Implement Qdrant client initialization with URL and API key from environment in `backend/main.py`
- [x] T040 [US4] Implement collection creation with 1024 dimensions and cosine distance in `backend/main.py`
- [x] T041 [US4] Implement collection recreation logic (--recreate flag) in `backend/main.py`
- [x] T042 [US4] Implement payload construction with source_url, chapter, section, chunk_index, text, title in `backend/main.py`
- [x] T043 [US4] Implement batch upsert of points to Qdrant in `backend/main.py`
- [x] T044 [US4] Implement error handling for connection failures and upsert errors in `backend/main.py`
- [x] T045 [US4] Implement `store` command that reads chunks.json + embeddings.json and upserts to Qdrant in `backend/main.py`
- [x] T046 [US4] Add console summary: points upserted, collection stats in `backend/main.py`

**Checkpoint**: `python main.py store` uploads all vectors to Qdrant Cloud

---

## Phase 7: User Story 5 - Verify Pipeline Integrity (Priority: P2)

**Goal**: Validate the knowledge base works by running sample queries

**Independent Test**: Run `python main.py verify` and confirm relevant results are returned for sample queries

### Implementation for User Story 5

- [x] T047 [US5] Implement query embedding generation with input_type="search_query" in `backend/main.py`
- [x] T048 [US5] Implement similarity search against Qdrant collection in `backend/main.py`
- [x] T049 [US5] Implement result formatting with scores and metadata in `backend/main.py`
- [x] T050 [US5] Implement collection statistics retrieval (vector count, storage size) in `backend/main.py`
- [x] T051 [US5] Implement `verify` command with --query option in `backend/main.py`
- [x] T052 [US5] Add console output: top 5 results with scores, collection stats, pass/fail status in `backend/main.py`

**Checkpoint**: `python main.py verify` returns relevant results and reports collection stats

---

## Phase 8: Full Pipeline Integration

**Goal**: Implement the `run` command that executes all stages sequentially

- [x] T053 Implement `run` command that chains ingest → chunk → embed → store → verify in `backend/main.py`
- [x] T054 Implement progress reporting for each stage in `backend/main.py`
- [x] T055 Implement exit codes per CLI contract (0=success, 1=partial, 2=critical) in `backend/main.py`
- [x] T056 Implement --recreate flag passthrough to store command in `backend/main.py`

**Checkpoint**: `python main.py run` executes complete pipeline end-to-end

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup and improvements

- [x] T057 [P] Add docstrings to all public functions in `backend/main.py`
- [x] T058 [P] Add type hints to all function signatures in `backend/main.py`
- [x] T059 Validate against quickstart.md instructions (run through setup steps) in `backend/`
- [x] T060 Run Black formatter on `backend/main.py`

---

## Dependencies & Execution Order

### Phase Dependencies

```text
Phase 1: Setup ──────────────────┐
                                 │
Phase 2: Foundational ───────────┼─► BLOCKS all user stories
                                 │
    ┌────────────────────────────┘
    │
    ▼
Phase 3: US1 (Ingest) ──────► Phase 4: US2 (Chunk) ──────► Phase 5: US3 (Embed) ──────► Phase 6: US4 (Store)
                                                                                              │
                                                                                              ▼
                                                                                    Phase 7: US5 (Verify)
                                                                                              │
                                                                                              ▼
                                                                                    Phase 8: Integration
                                                                                              │
                                                                                              ▼
                                                                                    Phase 9: Polish
```

### User Story Dependencies

- **US1 (Ingest)**: Depends only on Foundational - produces pages.json
- **US2 (Chunk)**: Depends on US1 - consumes pages.json, produces chunks.json
- **US3 (Embed)**: Depends on US2 - consumes chunks.json, produces embeddings.json
- **US4 (Store)**: Depends on US3 - consumes chunks.json + embeddings.json, uploads to Qdrant
- **US5 (Verify)**: Depends on US4 - queries Qdrant collection

**Note**: This is a data pipeline - stories are sequential by design (each consumes output of previous).

### Parallel Opportunities

Within Setup phase:
- T001-T004: Sequential (project init)
- T005: Parallel (independent file)

Within Foundational phase:
- T006-T009: Sequential (dataclasses before CLI)
- T010, T011: Parallel (independent concerns)
- T012: After T006-T008 (serializes the dataclasses)

Within each User Story:
- Core logic tasks are sequential within story
- Error handling can be parallel with main implementation

Polish phase:
- T057, T058: Parallel (documentation and typing)
- T059, T060: Sequential (validation then formatting)

---

## Parallel Example: Foundational Phase

```bash
# After T009 (CLI parser) is complete, these can run in parallel:
Task: T010 - Implement environment variable loading
Task: T011 - Implement logging configuration
```

---

## Implementation Strategy

### MVP First (P1 Stories Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: US1 (Ingest) → **Test independently**
4. Complete Phase 4: US2 (Chunk) → **Test independently**
5. Complete Phase 5: US3 (Embed) → **Test independently**
6. Complete Phase 6: US4 (Store) → **Test independently**
7. **STOP and VALIDATE**: Full pipeline works with manual stage execution
8. Deploy/demo if ready

### Adding Verification (P2)

9. Complete Phase 7: US5 (Verify) → Automated validation
10. Complete Phase 8: Integration → Single `run` command
11. Complete Phase 9: Polish → Clean code

### Incremental Testing

After each user story phase:
- US1: Run `python main.py ingest` → Check pages.json
- US2: Run `python main.py chunk` → Check chunks.json (token counts, metadata)
- US3: Run `python main.py embed` → Check embeddings.json (vector dimensions)
- US4: Run `python main.py store` → Check Qdrant Cloud console
- US5: Run `python main.py verify` → Check search results

---

## Task Summary

| Phase | Tasks | User Story | Description |
|-------|-------|------------|-------------|
| 1: Setup | T001-T005 | - | Project initialization |
| 2: Foundational | T006-T012 | - | Data models + CLI skeleton |
| 3: US1 | T013-T022 | Ingest | URL discovery + content extraction |
| 4: US2 | T023-T030 | Chunk | Token-based text chunking |
| 5: US3 | T031-T038 | Embed | Cohere API integration |
| 6: US4 | T039-T046 | Store | Qdrant upsert |
| 7: US5 | T047-T052 | Verify | Search validation |
| 8: Integration | T053-T056 | - | Full pipeline `run` command |
| 9: Polish | T057-T060 | - | Code quality |

**Total Tasks**: 60

| Priority | Stories | Tasks |
|----------|---------|-------|
| P1 | US1-US4 | T013-T046 (34 tasks) |
| P2 | US5 | T047-T052 (6 tasks) |
| Shared | Setup/Foundation/Integration/Polish | T001-T012, T053-T060 (20 tasks) |

---

## Notes

- All code goes in single `backend/main.py` as per plan.md
- No separate test files unless explicitly requested later
- Each task should be completable in a single focused session
- Commit after each completed task or logical group
- Run Black formatter before any PR
