# Tasks: RAG Retrieval Pipeline

**Input**: Design documents from `/specs/002-rag-retrieval/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: No automated tests - manual CLI testing per specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Project Type**: Single backend CLI (extends Part 1)
- **Main File**: `backend/main.py` (extending existing file)
- No new files required - retrieval logic integrated into existing codebase

---

## Phase 1: Setup

**Purpose**: Verify Part 1 infrastructure and prepare for retrieval implementation

- [x] T001 Verify Part 1 pipeline complete by running `uv run python main.py verify` in backend/
- [x] T002 [P] Read existing search_vectors function in backend/main.py to understand current implementation
- [x] T003 [P] Create comment block marking retrieval section start in backend/main.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Add RetrievedChunk dataclass in backend/main.py (from data-model.md)
- [x] T005 [P] Add RetrievalResult dataclass in backend/main.py (from data-model.md)
- [x] T006 [P] Add AssembledContext dataclass in backend/main.py (from data-model.md)
- [x] T007 Add validate_query function in backend/main.py (empty check, length check)
- [x] T008 Add validate_k function in backend/main.py (range 3-8 clamping)

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Query Textbook Content (Priority: P1) 🎯 MVP

**Goal**: Enable developers to retrieve textbook content using natural language queries

**Independent Test**: Run `uv run python main.py search --query "What is ROS 2?"` and verify results returned with metadata

### Implementation for User Story 1

- [x] T009 [US1] Add generate_query_embedding function in backend/main.py (input_type="search_query")
- [x] T010 [US1] Add retrieve_chunks function in backend/main.py (calls Qdrant query_points)
- [x] T011 [US1] Add convert_qdrant_result_to_chunk function in backend/main.py (maps payload to RetrievedChunk)
- [x] T012 [US1] Add search_knowledge_base function in backend/main.py (orchestrates embedding + retrieval)
- [x] T013 [US1] Add search CLI subcommand with --query argument in backend/main.py
- [x] T014 [US1] Test TC-001: Basic query returns results with `--query "What is ROS 2?"`

**Checkpoint**: User Story 1 complete - basic search works with default K=5

---

## Phase 4: User Story 2 - Configure Retrieval Parameters (Priority: P1)

**Goal**: Allow developers to configure the number of results (K) returned

**Independent Test**: Run `uv run python main.py search --query "sensors" --k 3` and verify exactly 3 results

### Implementation for User Story 2

- [x] T015 [US2] Add --k argument to search CLI command in backend/main.py
- [x] T016 [US2] Integrate validate_k into search flow in backend/main.py
- [x] T017 [US2] Add K clamping warning message to output in backend/main.py
- [x] T018 [US2] Test TC-002: Custom K returns correct count with `--query "sensors" --k 3`
- [x] T019 [US2] Test TC-003: K clamping works with `--query "sensors" --k 15`

**Checkpoint**: User Story 2 complete - K parameter works correctly

---

## Phase 5: User Story 3 - Assemble Context with Metadata (Priority: P1)

**Goal**: Format retrieved chunks into structured context with full metadata

**Independent Test**: Run search and verify output shows score, source URL, chapter, section for each result

### Implementation for User Story 3

- [x] T020 [US3] Add format_chunk_text function in backend/main.py (creates text block per chunk)
- [x] T021 [US3] Add assemble_context function in backend/main.py (creates AssembledContext)
- [x] T022 [US3] Update search command to show formatted context output in backend/main.py
- [x] T023 [US3] Add context summary footer (chunk count, character count, unique sources) in backend/main.py
- [x] T024 [US3] Test context assembly: verify metadata visible for each result

**Checkpoint**: User Story 3 complete - results show full metadata and context summary

---

## Phase 6: User Story 4 - Handle Empty Results (Priority: P2)

**Goal**: Gracefully handle queries that return no matching content

**Independent Test**: Run `uv run python main.py search --query "quantum cooking techniques"` and verify empty result message

### Implementation for User Story 4

- [x] T025 [US4] Add empty result handling in search_knowledge_base in backend/main.py
- [x] T026 [US4] Add "No matching content found" message for empty results in backend/main.py
- [x] T027 [US4] Add collection-not-found error handling in backend/main.py
- [x] T028 [US4] Add connection error handling with clear message in backend/main.py
- [x] T029 [US4] Test TC-004: Empty query returns error with `--query ""`
- [x] T030 [US4] Test TC-005: Unrelated query returns empty result message

**Checkpoint**: User Story 4 complete - all error cases handled gracefully

---

## Phase 7: Output Formats (Cross-cutting)

**Goal**: Support both text and JSON output formats per CLI contract

- [x] T031 Add --format argument (text/json) to search CLI in backend/main.py
- [x] T032 [P] Add format_search_result_text function in backend/main.py
- [x] T033 [P] Add format_search_result_json function in backend/main.py
- [x] T034 Integrate format selection into search command output in backend/main.py
- [x] T035 Test TC-006: JSON format outputs valid JSON with `--format json`

**Checkpoint**: Both output formats working correctly

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T036 Add --collection argument for custom collection name in backend/main.py
- [x] T037 [P] Update README.md in backend/ with search command documentation
- [x] T038 [P] Run Black formatter on backend/main.py
- [x] T039 Run quickstart.md validation - execute all example commands
- [x] T040 Test TC-007: Connection error handling (with invalid QDRANT_URL)
- [x] T041 Test TC-008: Missing API key handling (unset COHERE_API_KEY)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - US1, US2, US3 are P1 priority - core MVP
  - US4 is P2 priority - can be deferred
- **Output Formats (Phase 7)**: Depends on US1-US3 being functional
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P1)**: Requires US1 search command to exist
- **User Story 3 (P1)**: Requires US1 retrieval to work
- **User Story 4 (P2)**: Requires US1 base infrastructure

### Within Each User Story

- Validation before retrieval
- Retrieval before formatting
- Core implementation before testing
- Story complete before moving to next priority

### Parallel Opportunities

- T002-T003: Can run in parallel (reading vs commenting)
- T005-T006: Dataclasses can be defined in parallel
- T032-T033: Format functions can be written in parallel
- T037-T038: README and formatting can run in parallel

---

## Parallel Example: Phase 2 (Foundational)

```bash
# After T004 completes, these can run in parallel:
Task: "Add RetrievalResult dataclass in backend/main.py" (T005)
Task: "Add AssembledContext dataclass in backend/main.py" (T006)
```

---

## Implementation Strategy

### MVP First (User Stories 1-3)

1. Complete Phase 1: Setup (verify Part 1 works)
2. Complete Phase 2: Foundational (dataclasses and validation)
3. Complete Phase 3: User Story 1 (basic search)
4. **STOP and VALIDATE**: Test basic search independently
5. Complete Phase 4: User Story 2 (K parameter)
6. Complete Phase 5: User Story 3 (context assembly)
7. **CHECKPOINT**: MVP complete - core retrieval functional

### Full Implementation

1. Complete MVP (above)
2. Complete Phase 6: User Story 4 (error handling)
3. Complete Phase 7: Output formats (JSON support)
4. Complete Phase 8: Polish (documentation, cleanup)

### Suggested MVP Scope

- **MVP**: User Stories 1, 2, 3 (P1 priority)
- **Post-MVP**: User Story 4 (P2), Output Formats, Polish

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 (Setup) | T001-T003 | Verify Part 1, prepare file |
| 2 (Foundation) | T004-T008 | Dataclasses and validation |
| 3 (US1) | T009-T014 | Basic search query |
| 4 (US2) | T015-T019 | K parameter |
| 5 (US3) | T020-T024 | Context assembly |
| 6 (US4) | T025-T030 | Empty/error handling |
| 7 (Formats) | T031-T035 | Text/JSON output |
| 8 (Polish) | T036-T041 | Documentation, cleanup |

**Total Tasks**: 41
**MVP Tasks**: 24 (T001-T024)
**Test Cases Covered**: 8 (from CLI contract)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Manual testing per CLI contract test cases
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
