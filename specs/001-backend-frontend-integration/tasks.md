# Tasks: Backend-Frontend Integration

**Feature Branch**: `001-backend-frontend-integration`
**Input**: Design documents from `/specs/001-backend-frontend-integration/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/api.yaml

**Tests**: Not requested in spec - implementation tasks only.

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Paths: `backend/` for Python, `src/` for frontend (per plan.md)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency configuration

- [x] T001 Add FastAPI dependencies to backend/pyproject.toml (fastapi, uvicorn, pydantic)
- [x] T002 [P] Add @openai/chatkit-react dependency to package.json
- [x] T003 [P] Add backendUrl to customFields in docusaurus.config.ts

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Create Pydantic models in backend/models.py (QueryRequest, QueryResponse, Citation, ResponseMetadata, ErrorResponse, SessionResponse, SearchRequest, SearchResponse, RetrievedChunk)
- [x] T005 Create backend/routes.py with FastAPI APIRouter and import structure
- [x] T006 Add CORS middleware configuration to backend/main.py allowing localhost:3000 and 127.0.0.1:3000
- [x] T007 Register router from routes.py in backend/main.py app initialization
- [x] T008 [P] Create ChatWidget component directory structure at src/components/ChatWidget/

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Ask Questions from Textbook UI (Priority: P1) 🎯 MVP

**Goal**: Users can submit questions via ChatKit widget and receive grounded answers with citations

**Independent Test**: Open textbook site, type "What is ROS 2?" in ChatKit interface, verify answer streams in with source citations

### Implementation for User Story 1

- [x] T009 [US1] Implement POST /api/chatkit/session endpoint in backend/routes.py that generates ephemeral client token
- [x] T010 [US1] Implement POST /api/ask endpoint in backend/routes.py that validates QueryRequest, calls ask_agent(), returns QueryResponse
- [x] T011 [US1] Implement POST /api/search endpoint in backend/routes.py that validates SearchRequest, calls search_knowledge_base(), returns SearchResponse
- [x] T012 [US1] Implement POST /api/respond endpoint in backend/routes.py as alias for /api/ask
- [x] T013 [P] [US1] Create ChatWidget/index.tsx with useChatKit hook and getClientSecret callback pointing to /api/chatkit/session
- [x] T014 [P] [US1] Create ChatWidget/styles.module.css with basic styling for widget container
- [x] T015 [US1] Implement ChatKit component in ChatWidget/index.tsx with message rendering and citation display
- [x] T016 [US1] Add empty query validation returning HTTP 400 with EMPTY_QUERY error code in backend/routes.py
- [x] T017 [US1] Add max length validation (1000 chars) returning HTTP 400 with QUERY_TOO_LONG error code in backend/routes.py
- [x] T018 [US1] Format citations in QueryResponse with source_url, chapter, section, and score from agent response
- [x] T019 [US1] Display clickable citation links in ChatWidget/index.tsx message rendering
- [x] T020 [US1] Embed ChatWidget in Docusaurus layout/theme for global visibility across textbook pages

**Checkpoint**: User Story 1 complete - users can ask questions and receive grounded answers with citations

---

## Phase 4: User Story 2 - View Query Loading State (Priority: P1)

**Goal**: Users see loading/typing indicator while waiting for backend response

**Independent Test**: Submit question and observe ChatKit displays loading indicator immediately, then transitions to answer

### Implementation for User Story 2

- [x] T021 [US2] Configure ChatKit loading state handling in ChatWidget/index.tsx useChatKit hook options
- [x] T022 [US2] Ensure ChatKit typing indicator displays within 100ms of query submission
- [x] T023 [US2] Verify loading indicator transitions smoothly to response content

**Checkpoint**: User Stories 1 AND 2 complete - query flow with loading states working

---

## Phase 5: User Story 3 - Handle Backend Unavailability (Priority: P2)

**Goal**: Users see helpful error messages when backend is unavailable and can retry

**Independent Test**: Stop backend server, submit question from ChatKit, verify "Unable to connect to the server" message appears

### Implementation for User Story 3

- [x] T024 [US3] Add try-catch error handling in ChatWidget/index.tsx for fetch failures
- [x] T025 [US3] Display "Unable to connect to the server. Please try again later." for network errors
- [x] T026 [US3] Return HTTP 500 with INTERNAL_ERROR code for unexpected backend errors in backend/routes.py
- [x] T027 [US3] Return HTTP 500 with AGENT_ERROR code when ask_agent() fails in backend/routes.py
- [x] T028 [US3] Return HTTP 500 with RETRIEVAL_ERROR code when search_knowledge_base() fails in backend/routes.py
- [x] T029 [US3] Display user-friendly error message in ChatKit for HTTP 500 responses
- [x] T030 [US3] Enable retry by allowing new submissions after error state

**Checkpoint**: User Stories 1, 2, AND 3 complete - full error handling in place

---

## Phase 6: User Story 4 - Refusal for Off-Topic Questions (Priority: P2)

**Goal**: Agent refusals are displayed clearly in ChatKit conversation

**Independent Test**: Ask off-topic question like "What is the capital of France?", verify refusal message suggests asking about textbook topics

### Implementation for User Story 4

- [x] T031 [US4] Ensure QueryResponse refused field is set correctly from agent response in backend/routes.py
- [x] T032 [US4] Display refusal messages distinctly in ChatKit/index.tsx (e.g., different styling or icon)
- [x] T033 [US4] Include suggestion text to ask about textbook topics when refused=true

**Checkpoint**: All core user stories complete

---

## Phase 7: Additional Endpoints (Priority: P3)

**Goal**: Implement remaining API endpoints from contracts/api.yaml for pipeline operations

**Note**: These endpoints are for administrative/pipeline use, not primary user flow

### Implementation

- [x] T034 [P] Implement POST /api/store endpoint in backend/routes.py for storing embeddings
- [x] T035 [P] Implement POST /api/embed endpoint in backend/routes.py for generating embeddings

**Checkpoint**: All API endpoints from contracts/api.yaml implemented

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and refinements

- [x] T036 Add request/response logging for debugging in backend/routes.py
- [x] T037 Verify CORS headers work correctly between localhost:3000 and localhost:8000
- [x] T038 Run quickstart.md validation (start both servers, test end-to-end flow)
- [x] T039 Validate 10-second response time for typical queries (SC-001)
- [x] T040 Test special character handling in queries
- [x] T041 Test rapid consecutive query behavior (queue or disable input)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational - core MVP
- **User Story 2 (Phase 4)**: Can start after T020 (ChatWidget embedded) - builds on US1
- **User Story 3 (Phase 5)**: Can start after T015 (ChatKit component) - independent error handling
- **User Story 4 (Phase 6)**: Can start after T015 (ChatKit component) - independent refusal handling
- **Additional Endpoints (Phase 7)**: Can run in parallel with user stories after T005 (routes.py created)
- **Polish (Phase 8)**: Depends on all previous phases

### User Story Dependencies

- **US1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **US2 (P1)**: Depends on US1 ChatWidget being created (T015) - Extends loading behavior
- **US3 (P2)**: Can start after T015 - Independent error handling layer
- **US4 (P2)**: Can start after T015 - Independent refusal display layer

### Within Backend Tasks

```
T001 (deps) → T004 (models) → T005 (routes structure) → T009-T012 (endpoints)
                                    ↓
                              T006 (CORS) → T007 (register)
```

### Within Frontend Tasks

```
T002 (chatkit dep) → T008 (directory) → T013-T014 (base files) → T015 (component) → T019-T020 (integration)
       ↓
T003 (config)
```

### Parallel Opportunities

**Setup Phase (parallel)**:
```
T001 (backend deps)
T002 (frontend deps) [P]
T003 (docusaurus config) [P]
```

**Foundational Phase (parallel after T004-T007)**:
```
T008 (ChatWidget directory) [P]
```

**User Story 1 (parallel where marked)**:
```
T013 (ChatWidget index) [P] + T014 (styles) [P]
```

**Additional Endpoints (parallel)**:
```
T034 (/store) [P] + T035 (/embed) [P]
```

---

## Parallel Example: Setup Phase

```bash
# Launch all setup tasks together:
Task: "Add FastAPI dependencies to backend/pyproject.toml"
Task: "Add @openai/chatkit-react dependency to package.json"
Task: "Add backendUrl to customFields in docusaurus.config.ts"
```

## Parallel Example: User Story 3 & 4 After US1

```bash
# After T015 (ChatWidget component), both error handling and refusal display can proceed in parallel:

# US3 stream:
Task: "Add try-catch error handling in ChatWidget/index.tsx"
Task: "Display connection error message"
Task: "Return HTTP 500 errors from backend"

# US4 stream (parallel):
Task: "Set QueryResponse refused field"
Task: "Display refusal messages distinctly"
```

---

## Implementation Strategy

### MVP First (User Story 1 + 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (core query flow)
4. Complete Phase 4: User Story 2 (loading states)
5. **STOP and VALIDATE**: Test US1 + US2 independently
6. Demo: Users can ask questions and see loading states

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. User Story 1 → Test: Ask "What is ROS 2?" → Get answer with citations (MVP!)
3. User Story 2 → Test: See loading indicator → Better UX
4. User Story 3 → Test: Stop backend, see error → Reliability
5. User Story 4 → Test: Ask off-topic, see refusal → Grounding validation
6. Polish → Production-ready

---

## Summary

| Phase | Tasks | Stories | Status |
|-------|-------|---------|--------|
| Setup | T001-T003 | - | Complete |
| Foundational | T004-T008 | - | Complete |
| US1: Ask Questions | T009-T020 | P1 MVP | Complete |
| US2: Loading State | T021-T023 | P1 | Complete |
| US3: Error Handling | T024-T030 | P2 | Complete |
| US4: Refusal Display | T031-T033 | P2 | Complete |
| Additional Endpoints | T034-T035 | P3 | Complete |
| Polish | T036-T041 | - | Complete |

**Total Tasks**: 41
**MVP Tasks (US1+US2)**: 23 (T001-T023)
**Parallel Opportunities**: 8 tasks marked [P]

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently testable after completion
- Backend routes.py is the main implementation file for all endpoints
- ChatWidget/index.tsx is the main frontend implementation file
- Tests not included per spec (no explicit test requirement)
- Commit after each task or logical group
