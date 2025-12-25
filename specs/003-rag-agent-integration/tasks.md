# Tasks: RAG Agent Integration

**Input**: Design documents from `/specs/003-rag-agent-integration/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: No automated tests - manual CLI testing per specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Project Type**: Single backend CLI (extends Part 2)
- **Main File**: `backend/main.py` (extending existing file)
- No new files required - agent code integrated into existing codebase

---

## Phase 1: Setup

**Purpose**: Verify Part 2 infrastructure and add agent dependencies

- [x] T001 Verify Part 2 pipeline complete by running `uv run python main.py verify` in backend/
- [x] T002 Verify search command works by running `uv run python main.py search --query "ROS 2"` in backend/
- [x] T003 Add openai-agents dependency with `uv add openai-agents` in backend/
- [x] T004 [P] Verify OpenAI API key is set in environment variables
- [x] T005 [P] Create comment block marking agent section start in backend/main.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Add agent imports (agents.Agent, function_tool, ModelSettings, Runner) in backend/main.py
- [x] T007 [P] Add AgentConfig dataclass in backend/main.py (from data-model.md)
- [x] T008 [P] Add GroundingContext dataclass in backend/main.py (from data-model.md)
- [x] T009 [P] Add Citation dataclass in backend/main.py (from data-model.md)
- [x] T010 [P] Add AgentResponse dataclass in backend/main.py (from data-model.md)
- [x] T011 Add agent constants (DEFAULT_SCORE_THRESHOLD, REFUSAL messages) in backend/main.py
- [x] T012 Add GROUNDING_PROMPT system prompt constant in backend/main.py (from research.md)

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Ask Textbook Questions (Priority: P1) MVP

**Goal**: Enable developers to ask natural language questions and receive grounded answers with citations

**Independent Test**: Run `uv run python main.py ask "What is ROS 2?"` and verify answer with citations

### Implementation for User Story 1

- [x] T013 [US1] Implement format_context_for_agent function in backend/main.py (converts RetrievalResult to agent-consumable string)
- [x] T014 [US1] Implement @function_tool search_textbook wrapper in backend/main.py (wraps search_knowledge_base)
- [x] T015 [US1] Create Agent instance with name, instructions, tools, model_settings in backend/main.py
- [x] T016 [US1] Implement async ask_agent function in backend/main.py (uses Runner.run)
- [x] T017 [US1] Implement format_response_text function in backend/main.py (formats answer with citations)
- [x] T018 [US1] Add ask CLI command with QUESTION argument in backend/main.py
- [x] T019 [US1] Implement asyncio.run wrapper for ask command in backend/main.py
- [x] T020 [US1] Test TC-001: Basic question returns answer with `--query "What is ROS 2?"`

**Checkpoint**: User Story 1 complete - basic grounded Q&A works

---

## Phase 4: User Story 2 - Refuse Unanswerable Questions (Priority: P1)

**Goal**: Agent gracefully refuses to answer when context is insufficient

**Independent Test**: Run `uv run python main.py ask "What is the capital of France?"` and verify refusal message

### Implementation for User Story 2

- [x] T021 [US2] Implement should_refuse function in backend/main.py (checks score threshold)
- [x] T022 [US2] Implement decide_grounding function in backend/main.py (returns GroundingContext)
- [x] T023 [US2] Update search_textbook tool to return refusal signal when should_refuse is True in backend/main.py
- [x] T024 [US2] Update GROUNDING_PROMPT with explicit refusal rules in backend/main.py
- [x] T025 [US2] Add refusal response formatting in format_response_text in backend/main.py
- [x] T026 [US2] Test TC-002: Off-topic question returns refusal with `--query "capital of France?"`

**Checkpoint**: User Story 2 complete - agent refuses unanswerable questions

---

## Phase 5: User Story 3 - Deterministic Tool Invocation (Priority: P2)

**Goal**: Ensure retrieval tool is always invoked before any response

**Independent Test**: Run with `--verbose` and verify tool_calls includes search_textbook

### Implementation for User Story 3

- [x] T027 [US3] Configure ModelSettings with tool_choice="required" in Agent creation in backend/main.py
- [x] T028 [US3] Add tool invocation logging in search_textbook tool in backend/main.py
- [x] T029 [US3] Update AgentResponse to track tool_calls list in backend/main.py
- [x] T030 [US3] Add tool invocation display in verbose output in backend/main.py
- [x] T031 [US3] Test TC-005: Verify tool is called for every question (check verbose output)

**Checkpoint**: User Story 3 complete - deterministic tool invocation verified

---

## Phase 6: User Story 4 - View Source Citations (Priority: P2)

**Goal**: Responses include structured source citations for verification

**Independent Test**: Run ask command and verify citations include URL, chapter, section, score

### Implementation for User Story 4

- [x] T032 [US4] Implement extract_citations function in backend/main.py (creates Citation list from chunks)
- [x] T033 [US4] Format inline citations in agent response in backend/main.py
- [x] T034 [US4] Add Sources section to text output format in backend/main.py
- [x] T035 [US4] Implement format_response_json function in backend/main.py (JSON output format)
- [x] T036 [US4] Add --format argument to ask command (text/json) in backend/main.py
- [x] T037 [US4] Test TC-003: JSON format outputs valid JSON with `--format json`

**Checkpoint**: User Story 4 complete - citations visible and verifiable

---

## Phase 7: CLI Options & Error Handling (Cross-cutting)

**Purpose**: Add remaining CLI options and graceful error handling

- [x] T038 Add --k argument to ask command in backend/main.py
- [x] T039 [P] Add --threshold argument to ask command in backend/main.py
- [x] T040 [P] Add --verbose argument to ask command in backend/main.py
- [x] T041 Implement verbose output format in ask command in backend/main.py
- [x] T042 Add retrieval error handling in search_textbook tool in backend/main.py
- [x] T043 [P] Add OpenAI API error handling in ask_agent function in backend/main.py
- [x] T044 Add empty query validation in ask command in backend/main.py
- [x] T045 Test TC-004: Custom K parameter with `--k 3 --verbose`
- [x] T046 [P] Test TC-005: Empty question returns error with `--query ""`
- [x] T047 [P] Test TC-006: Verbose mode shows details
- [x] T048 Test TC-007: Low threshold with `--threshold 0.1`
- [x] T049 Test TC-008: High threshold with `--threshold 0.9`

**Checkpoint**: All CLI options working, errors handled gracefully

---

## Phase 8: Polish & Documentation

**Purpose**: Documentation and final cleanup

- [x] T050 Update backend/README.md with ask command documentation
- [x] T051 [P] Run Black formatter on backend/main.py
- [x] T052 Run quickstart.md validation - execute all example commands
- [x] T053 Test connection error handling (with invalid QDRANT_URL)
- [x] T054 Test missing API key handling (unset OPENAI_API_KEY)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - US1, US2 are P1 priority - core functionality
  - US3, US4 are P2 priority - enhancements
- **CLI Options (Phase 7)**: Depends on US1-US4 being functional
- **Polish (Phase 8)**: Depends on all phases being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational - Independent of US1 (uses same infrastructure)
- **User Story 3 (P2)**: Requires Agent from US1 to configure tool_choice
- **User Story 4 (P2)**: Requires basic response from US1 to add citations

### Within Each User Story

- Functions before CLI integration
- Core implementation before testing
- Story complete before moving to next priority

### Parallel Opportunities

- T004-T005: Can run in parallel
- T007-T010: All dataclasses can be defined in parallel
- T039-T040: CLI options can be added in parallel
- T043, T045-T047: Tests can run in parallel
- T050-T051: README and formatting can run in parallel

---

## Parallel Example: Phase 2 (Foundational)

```bash
# After T006 completes, these can run in parallel:
Task: "Add AgentConfig dataclass in backend/main.py" (T007)
Task: "Add GroundingContext dataclass in backend/main.py" (T008)
Task: "Add Citation dataclass in backend/main.py" (T009)
Task: "Add AgentResponse dataclass in backend/main.py" (T010)
```

---

## Implementation Strategy

### MVP First (User Stories 1-2)

1. Complete Phase 1: Setup (verify Part 2 works)
2. Complete Phase 2: Foundational (dataclasses and constants)
3. Complete Phase 3: User Story 1 (basic Q&A)
4. **STOP and VALIDATE**: Test basic Q&A independently
5. Complete Phase 4: User Story 2 (refusal)
6. **CHECKPOINT**: MVP complete - grounded Q&A with refusals

### Full Implementation

1. Complete MVP (above)
2. Complete Phase 5: User Story 3 (deterministic tool invocation)
3. Complete Phase 6: User Story 4 (citations)
4. Complete Phase 7: CLI Options (k, threshold, verbose)
5. Complete Phase 8: Polish (documentation, cleanup)

### Suggested MVP Scope

- **MVP**: User Stories 1, 2 (P1 priority) - Tasks T001-T026
- **Post-MVP**: User Stories 3, 4 (P2), CLI Options, Polish

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 (Setup) | T001-T005 | Verify Part 2, add dependencies |
| 2 (Foundation) | T006-T012 | Dataclasses, constants, prompts |
| 3 (US1) | T013-T020 | Basic Q&A with citations |
| 4 (US2) | T021-T026 | Refusal handling |
| 5 (US3) | T027-T031 | Deterministic tool invocation |
| 6 (US4) | T032-T037 | Source citations |
| 7 (CLI) | T038-T049 | Options and error handling |
| 8 (Polish) | T050-T054 | Documentation, cleanup |

**Total Tasks**: 54
**MVP Tasks**: 26 (T001-T026)
**Test Cases Covered**: 8 (from CLI contract)

---

## Notes

- [P] tasks = different files or no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Manual testing per CLI contract test cases
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
