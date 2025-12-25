# Research: Backend-Frontend Integration

**Feature**: 001-backend-frontend-integration
**Date**: 2025-12-25
**Status**: Complete

## Research Tasks

### 1. ChatKit Integration Pattern

**Question**: How does OpenAI ChatKit integrate with a custom backend (not OpenAI API directly)?

**Finding**: ChatKit is framework-agnostic and works with any backend. The integration pattern uses:
- `useChatKit` hook for configuration
- `getClientSecret` callback for session authentication
- `ChatKit` component for UI rendering

**Decision**: Use custom backend endpoints with ChatKit's callback pattern
**Rationale**: ChatKit handles UI complexity while our backend provides RAG responses
**Alternatives Rejected**: Building custom chat UI (unnecessary complexity)

### 2. Session Token Management

**Question**: How should ChatKit session tokens be managed?

**Finding**: ChatKit requires a `client_secret` returned from `/api/chatkit/session`. The callback receives an `existing` parameter for refresh logic.

**Decision**: Generate ephemeral tokens per session; no persistence required
**Rationale**: Local development only; no need for database storage
**Alternatives Rejected**: Database-backed sessions (overkill for MVP)

### 3. FastAPI Route Structure

**Question**: Should routes be in main.py or separated?

**Finding**: User explicitly requested `routes.py` for all route definitions with `main.py` only for initialization.

**Decision**: Separate `routes.py` file with FastAPI APIRouter
**Rationale**: User requirement; clean separation of concerns
**Alternatives Rejected**: All routes in main.py (user preference)

### 4. CORS Configuration

**Question**: What CORS settings are needed for local development?

**Finding**: Frontend runs on localhost:3000, backend on localhost:8000. Cross-origin requests require CORS.

**Decision**: Allow origins `["http://localhost:3000", "http://127.0.0.1:3000"]`
**Rationale**: Minimal required origins for local dev security
**Alternatives Rejected**: Open CORS `["*"]` (security risk)

### 5. Response Streaming

**Question**: Should responses be streamed or returned complete?

**Finding**: ChatKit supports streaming via response callbacks. However, the existing `ask_agent()` function returns complete responses.

**Decision**: Non-streaming for MVP; return complete responses
**Rationale**: Simpler implementation; ChatKit's loading state handles UX
**Alternatives Rejected**: SSE streaming (requires refactoring agent)

### 6. Error Response Format

**Question**: What error format should the API use?

**Finding**: ChatKit can display error messages from JSON responses. Need consistent error schema.

**Decision**: Structured JSON errors with `code`, `message`, and `details` fields
**Rationale**: Consistent API responses; ChatKit can parse and display
**Alternatives Rejected**: Plain text errors (harder to parse)

## Technology Best Practices

### FastAPI with Pydantic

- Use `BaseModel` for request/response schemas
- Use `HTTPException` for error responses
- Use `APIRouter` for route organization
- Enable automatic OpenAPI documentation

### ChatKit React Integration

- Use `useChatKit` hook at component mount
- Implement `getClientSecret` as async callback
- Handle loading/error states via ChatKit's built-in UI
- Configure backend URL via environment or config

### Docusaurus + React Components

- Create components in `src/components/`
- Use CSS modules for styling
- Import components in layouts or pages
- Use `useDocusaurusContext` for site config access

## Dependencies Identified

### Backend (pyproject.toml)

```toml
[project.dependencies]
fastapi = ">=0.109.0"
uvicorn = ">=0.27.0"
pydantic = ">=2.5.0"
python-dotenv = ">=1.0.0"
# Existing: httpx, tiktoken, beautifulsoup4, openai, cohere, qdrant-client
```

### Frontend (package.json)

```json
{
  "dependencies": {
    "@openai/chatkit-react": "^1.0.0"
  }
}
```

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| ChatKit package version | Use latest stable (^1.0.0) |
| Backend port | 8000 (configurable via .env) |
| Frontend port | 3000 (Docusaurus default) |
| Widget placement | Global (swizzle theme or layout) |

## References

- [ChatKit GitHub](https://github.com/openai/chatkit-js)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docusaurus Custom Components](https://docusaurus.io/docs/creating-pages#react-components)
