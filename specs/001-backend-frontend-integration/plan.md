# Implementation Plan: Backend-Frontend Integration

**Branch**: `001-backend-frontend-integration` | **Date**: 2025-12-25 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-backend-frontend-integration/spec.md`

## Summary

Integrate the FastAPI backend with the Docusaurus frontend using OpenAI ChatKit for the chat UI. The backend exposes REST endpoints (`/api/chatkit/session`, `/api/ask`, `/store`, `/embed`, `/search`, `/respond`) that interface with the existing RAG agent. ChatKit provides a batteries-included chat widget that handles UI, loading states, and message rendering, while the backend manages session tokens and routes queries to the grounded Q&A agent.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript/JavaScript (frontend)
**Primary Dependencies**: FastAPI, uvicorn, pydantic (backend); @openai/chatkit-react, Docusaurus (frontend)
**Storage**: Qdrant Cloud (vectors), in-memory session tokens (no persistence)
**Testing**: pytest (backend), manual E2E testing (frontend)
**Target Platform**: Local development (localhost:8000 backend, localhost:3000 frontend)
**Project Type**: Web application (backend + frontend)
**Performance Goals**: 10-second end-to-end query response for 95% of queries
**Constraints**: Free-tier infrastructure, no authentication beyond ChatKit sessions
**Scale/Scope**: Single developer local setup, 1 concurrent user for MVP

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| **I. Accuracy** | PASS | Chatbot answers trace to book sections via citations from RAG agent |
| **II. Clarity** | PASS | Using ChatKit's clean UI; consistent Markdown formatting |
| **III. Reproducibility** | PASS | All endpoints scriptable; environment via .env |
| **IV. Security** | PASS | No secrets in code; session tokens not persisted; minimal logging |

**Technology Stack Alignment**:
| Component | Constitution | Plan | Status |
|-----------|--------------|------|--------|
| Backend API | FastAPI | FastAPI | PASS |
| Vector Database | Qdrant | Qdrant Cloud | PASS |
| AI/Chatbot | OpenAI Agents/ChatKit | ChatKit + OpenAI Agents SDK | PASS |

**Gate Result**: PASS - All principles satisfied. Proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/001-backend-frontend-integration/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (OpenAPI specs)
│   └── api.yaml
└── tasks.md             # Phase 2 output (/sp.tasks command)
```

### Source Code (repository root)

```text
backend/
├── main.py              # FastAPI app initialization + CORS setup
├── routes.py            # All FastAPI route definitions (NEW)
├── .env                 # Environment variables (existing)
└── pyproject.toml       # Dependencies (add fastapi, uvicorn)

src/
├── components/
│   └── ChatWidget/      # ChatKit integration component (NEW)
│       ├── index.tsx
│       └── styles.module.css
├── css/
└── pages/

docusaurus.config.ts     # May need customFields for backend URL
package.json             # Add @openai/chatkit-react dependency
```

**Structure Decision**: Web application pattern selected. Backend code in `/backend/` directory (existing), frontend code in `/src/` directory (existing Docusaurus structure). New `routes.py` file separates route definitions from main.py initialization per user requirement.

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Docusaurus Frontend                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 ChatKit Widget                           │   │
│  │  ┌─────────────┐    ┌──────────────┐   ┌─────────────┐  │   │
│  │  │ useChatKit  │───▶│getClientSecret│──▶│/api/chatkit │  │   │
│  │  │    hook     │    │   callback    │   │  /session   │  │   │
│  │  └─────────────┘    └──────────────┘   └─────────────┘  │   │
│  │         │                                      │         │   │
│  │         ▼                                      ▼         │   │
│  │  ┌─────────────┐                      ┌─────────────┐   │   │
│  │  │  ChatKit    │─────HTTP POST───────▶│ /api/ask    │   │   │
│  │  │  Component  │◀────JSON Response────│  endpoint   │   │   │
│  │  └─────────────┘                      └─────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                    HTTP (localhost:8000)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    routes.py                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ POST         │  │ POST         │  │ POST         │   │   │
│  │  │ /chatkit/    │  │ /ask         │  │ /search      │   │   │
│  │  │   session    │  │              │  │              │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  │         │                 │                 │            │   │
│  │         ▼                 ▼                 ▼            │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │              main.py (existing)                   │   │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │   │   │
│  │  │  │ ask_agent() │  │search_kb()  │  │ Agent    │  │   │   │
│  │  │  │             │  │             │  │ (Part 3) │  │   │   │
│  │  │  └─────────────┘  └─────────────┘  └──────────┘  │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│                    Qdrant Cloud (vectors)                       │
└─────────────────────────────────────────────────────────────────┘
```

## API Routes Design

Per user requirement, all routes defined in `backend/routes.py`:

| Endpoint | Method | Purpose | Calls |
|----------|--------|---------|-------|
| `/api/chatkit/session` | POST | Generate ChatKit client token | OpenAI API |
| `/api/ask` | POST | Main query endpoint for ChatKit | `ask_agent()` |
| `/api/store` | POST | Ingest and store embeddings | `store_vectors()` |
| `/api/embed` | POST | Generate query embeddings | `embed_chunks()` |
| `/api/search` | POST | Retrieve relevant chunks | `search_knowledge_base()` |
| `/api/respond` | POST | Agent-based response (alias) | `ask_agent()` |

## Complexity Tracking

No constitution violations requiring justification.

## Decision Log

| Decision | Choice | Rationale | Alternatives Rejected |
|----------|--------|-----------|----------------------|
| Route separation | Separate `routes.py` | User requirement; clean separation of concerns | All routes in main.py |
| Session tokens | In-memory only | Simplicity for local dev; no persistence needed | Database storage |
| Streaming | Non-streaming (MVP) | Simpler implementation; ChatKit handles loading | SSE streaming |
| Error format | Structured JSON | Consistent API responses; ChatKit can parse | Plain text errors |
| CORS | Allow localhost:3000 | Required for local dev | Open CORS (security risk) |

## Implementation Phases

### Phase 0: Research (Complete)

See [research.md](./research.md) for detailed findings.

### Phase 1: Foundation

1. **Backend Routes** (`routes.py`)
   - Create FastAPI router with all endpoints
   - Implement Pydantic request/response models
   - Add CORS middleware configuration
   - Wire routes to existing functions in main.py

2. **Frontend Integration** (`src/components/ChatWidget/`)
   - Install `@openai/chatkit-react`
   - Create ChatWidget component with useChatKit hook
   - Implement getClientSecret callback
   - Embed widget in Docusaurus layout

3. **Configuration**
   - Add backend URL to docusaurus.config.ts customFields
   - Update pyproject.toml with FastAPI dependencies
   - Create startup scripts for concurrent dev

### Phase 2: Integration & Testing

1. **E2E Flow Validation**
   - Test ChatKit → /ask → agent → response
   - Verify citations display correctly
   - Test error states and refusals

2. **Edge Cases**
   - Empty query validation
   - Backend unavailable handling
   - Timeout behavior

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ChatKit API changes | Low | Medium | Pin package version |
| CORS issues | Medium | Low | Explicit localhost config |
| Agent timeout | Medium | Medium | 30s timeout with retry |
| Port conflicts | Low | Low | Configurable ports |

## Next Steps

1. Run `/sp.tasks` to generate implementation tasks
2. Implement routes.py with Pydantic models
3. Create ChatWidget React component
4. Test end-to-end flow locally
