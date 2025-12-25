# Feature Specification: Backend-Frontend Integration

**Feature Branch**: `001-backend-frontend-integration`
**Created**: 2025-12-25
**Status**: Draft
**Input**: User description: "Backend–Frontend Integration – Part 4: Connecting the FastAPI backend to the Docusaurus frontend for end-to-end RAG query flow"

## Clarifications

### Session 2025-12-25

- Q: Which UI library should be used for the frontend chat component? → A: OpenAI ChatKit (`@openai/chatkit-react`) from https://github.com/openai/chatkit-js

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ask Questions from the Textbook UI (Priority: P1)

A developer reading the textbook wants to ask questions about the content directly from the book interface. They interact with a ChatKit-powered chat widget and receive an answer grounded in the textbook content, with real-time response streaming displayed alongside their reading.

**Why this priority**: This is the core value proposition - enabling users to get answers without leaving the textbook UI. Without this, there is no integration.

**Independent Test**: Can be fully tested by opening the textbook site, typing "What is ROS 2?" in the ChatKit interface, and verifying an answer streams in with source citations.

**Acceptance Scenarios**:

1. **Given** the user is on any textbook page with the ChatKit widget visible, **When** they type a question and submit, **Then** the question is sent to the backend API and a response streams back within 10 seconds
2. **Given** the backend returns a grounded answer, **When** the response is displayed, **Then** the user sees the answer text and source citations formatted readably in the ChatKit message format
3. **Given** the user submits an empty query, **When** the form is submitted, **Then** the ChatKit component prevents submission and shows a validation message

---

### User Story 2 - View Query Loading State (Priority: P1)

A developer submits a question and sees ChatKit's built-in loading/typing indicator while waiting for the backend response, providing feedback that their request is being processed.

**Why this priority**: Essential for user experience - without loading feedback, users may think the interface is broken.

**Independent Test**: Can be tested by submitting a question and observing that ChatKit displays a loading indicator immediately and transitions to the streamed response.

**Acceptance Scenarios**:

1. **Given** a user submits a question, **When** the request is in flight, **Then** ChatKit displays a typing/loading indicator
2. **Given** the backend responds successfully, **When** the response arrives, **Then** the loading indicator transitions to the streamed answer
3. **Given** the backend returns an error, **When** the error is received, **Then** the loading indicator is replaced with an error message

---

### User Story 3 - Handle Backend Unavailability Gracefully (Priority: P2)

A developer tries to ask a question but the backend is temporarily unavailable. They see a helpful error message in the ChatKit interface and can retry.

**Why this priority**: Important for reliability but secondary to core query flow. Users need to know when something goes wrong.

**Independent Test**: Can be tested by stopping the backend server, submitting a question from the ChatKit interface, and verifying an appropriate error message is shown.

**Acceptance Scenarios**:

1. **Given** the backend server is not running, **When** a user submits a question, **Then** ChatKit displays "Unable to connect to the server. Please try again later."
2. **Given** the backend returns an HTTP error (500), **When** the error is received, **Then** ChatKit displays a user-friendly error message
3. **Given** an error state, **When** the user submits again, **Then** a new request is sent to the backend

---

### User Story 4 - Receive Refusal for Off-Topic Questions (Priority: P2)

A developer asks a question unrelated to the textbook (e.g., "What is the capital of France?"). The agent correctly refuses to answer and explains that the question is outside the textbook scope, displayed in the ChatKit conversation thread.

**Why this priority**: Demonstrates the grounding behavior works end-to-end, important for trust but secondary to core flow.

**Independent Test**: Can be tested by asking an off-topic question and verifying the refusal message is displayed correctly in the ChatKit interface.

**Acceptance Scenarios**:

1. **Given** a question outside textbook scope, **When** the agent refuses to answer, **Then** ChatKit displays the refusal message clearly in the conversation
2. **Given** a refusal response, **When** displayed to the user, **Then** the message suggests asking about topics covered in the textbook

---

### Edge Cases

- What happens when the user submits rapid consecutive queries? ChatKit should queue or disable input while a request is in flight
- What happens when the response is extremely long? ChatKit handles scrollable message areas natively
- What happens when network times out after 30 seconds? Frontend should show timeout error in ChatKit and allow retry
- What happens when CORS is misconfigured? ChatKit should show a connection error message (not a cryptic CORS error)
- What happens when the query contains special characters? Backend should handle encoding correctly

## Requirements *(mandatory)*

### Functional Requirements

#### Backend (FastAPI)

- **FR-001**: System MUST expose a POST endpoint at `/api/chatkit/session` that generates ChatKit client tokens
- **FR-002**: System MUST expose a POST endpoint at `/api/ask` that accepts JSON with a `question` field
- **FR-003**: System MUST validate incoming queries (non-empty, max 1000 characters) and return appropriate error responses
- **FR-004**: System MUST route valid queries to the existing `ask_agent` function and return the agent response
- **FR-005**: System MUST return JSON responses with fields: `answer`, `citations`, `grounded`, `refused`, and `metadata`
- **FR-006**: System MUST enable CORS for local development (allow localhost origins)
- **FR-007**: System MUST return HTTP 400 for invalid requests with descriptive error messages
- **FR-008**: System MUST return HTTP 500 for internal errors with safe error messages (no stack traces)
- **FR-009**: System MUST log all requests and errors for debugging

#### Frontend (Docusaurus + ChatKit)

- **FR-010**: Frontend MUST use OpenAI ChatKit (`@openai/chatkit-react`) for the chat interface
- **FR-011**: Frontend MUST integrate the `ChatKit` component and `useChatKit` hook
- **FR-012**: Frontend MUST implement `getClientSecret` API callback to fetch session tokens from backend
- **FR-013**: Frontend MUST embed the ChatKit widget on textbook pages (global or per-page placement)
- **FR-014**: Frontend MUST display response streaming as messages arrive from the backend
- **FR-015**: Frontend MUST display source citations from the response with links to source URLs
- **FR-016**: Frontend MUST handle and display error states with user-friendly messages
- **FR-017**: Frontend MUST configure backend URL via environment variable or configuration

#### Integration

- **FR-018**: End-to-end query flow MUST complete within 10 seconds for typical questions
- **FR-019**: Frontend and backend MUST communicate over HTTP on configurable ports
- **FR-020**: System MUST support local development mode where both frontend and backend run concurrently

### Key Entities

- **ChatKitSession**: Client session token generated by backend for ChatKit authentication
- **QueryRequest**: The structured request sent from ChatKit to backend containing the user's question
- **QueryResponse**: The structured response from backend containing answer, citations, grounding status, and metadata
- **Citation**: Source reference with URL, chapter, section, and relevance score
- **ErrorResponse**: Structured error with code and user-friendly message

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can submit questions via ChatKit widget and receive grounded answers within 10 seconds for 95% of queries
- **SC-002**: ChatKit loading/typing indicators are visible within 100ms of query submission
- **SC-003**: Error messages are displayed in ChatKit within 2 seconds of backend failure detection
- **SC-004**: Source citations are clickable and link to valid textbook pages
- **SC-005**: The system correctly displays refusal messages for 100% of off-topic questions
- **SC-006**: Both backend and frontend start successfully with a single development command
- **SC-007**: ChatKit widget renders correctly on all textbook pages

## Constraints

- **Backend Framework**: FastAPI (required per user specification)
- **Agent Logic**: Must use existing Part 3 agent (`ask_agent` function in `backend/main.py`)
- **Frontend Platform**: Docusaurus with React components
- **Chat UI Library**: OpenAI ChatKit (`@openai/chatkit-react`) - required per clarification
- **Communication**: HTTP REST API (local development only)
- **Environment**: Local development environment with both services running concurrently

## Out of Scope

- UI/UX design polish (color schemes, typography, advanced layouts beyond ChatKit defaults)
- Authentication or user management (beyond ChatKit session tokens)
- Deployment to production cloud
- Performance optimization (caching, CDN, connection pooling)
- Conversation history or multi-turn chat persistence
- Rate limiting or abuse prevention
- API versioning

## Assumptions

- Part 3 agent implementation (`ask_agent` function) is complete and functional in `backend/main.py`
- Qdrant collection "textbook_chunks" contains indexed textbook content
- Required API keys (OpenAI, Cohere, Qdrant) are configured in environment
- Python 3.11+ environment with uv package manager for backend
- Node.js environment for Docusaurus frontend
- Developer has local access to run both services concurrently
- Default backend port: 8000, frontend port: 3000
- OpenAI API key available for ChatKit session token generation

## Dependencies

- **Part 1**: 001-textbook-rag-kb (knowledge base with embedded chunks)
- **Part 2**: 002-rag-retrieval (search command and retrieval functions)
- **Part 3**: 003-rag-agent-integration (agent with grounded Q&A)
- **External**: OpenAI API, Cohere API, Qdrant Cloud
- **Frontend**: Existing Docusaurus textbook site structure
- **UI Library**: `@openai/chatkit-react` (https://github.com/openai/chatkit-js)
