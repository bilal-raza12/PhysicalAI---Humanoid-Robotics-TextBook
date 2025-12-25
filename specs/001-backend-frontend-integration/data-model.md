# Data Model: Backend-Frontend Integration

**Feature**: 001-backend-frontend-integration
**Date**: 2025-12-25

## Entities

### ChatKitSession

Client session token for ChatKit authentication.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| client_secret | string | Ephemeral token for ChatKit | Required, UUID format |
| created_at | datetime | Token creation timestamp | Auto-generated |
| expires_at | datetime | Token expiration | 1 hour from creation |

**Lifecycle**: Created → Active → Expired (no persistence)

### QueryRequest

Request from ChatKit to backend for Q&A.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| question | string | User's natural language question | Required, 1-1000 chars |

**Validation Rules**:
- `question` must not be empty or whitespace-only
- `question` max length 1000 characters
- Special characters allowed (handled by backend encoding)

### QueryResponse

Response from backend containing grounded answer.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| answer | string | Agent's grounded response | Required |
| citations | Citation[] | Source references | May be empty |
| grounded | boolean | Whether answer uses context | Required |
| refused | boolean | Whether agent refused to answer | Required |
| metadata | ResponseMetadata | Timing and debug info | Required |

### Citation

Source reference for answer grounding.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| source_url | string | URL to textbook page | Valid URL |
| chapter | string | Chapter name | Optional |
| section | string | Section name | Optional |
| score | float | Relevance score | 0.0-1.0 |
| chunk_index | int | Chunk position | >= 0 |

### ResponseMetadata

Timing and debugging information.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| retrieval_time_ms | float | Time for retrieval | >= 0 |
| generation_time_ms | float | Time for generation | >= 0 |
| total_time_ms | float | Total processing time | >= 0 |
| tool_calls | string[] | Tools invoked | May be empty |

### ErrorResponse

Structured error from backend.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| code | string | Error code | Required |
| message | string | Human-readable message | Required |
| details | object | Additional context | Optional |

**Error Codes**:
- `VALIDATION_ERROR`: Invalid request data
- `EMPTY_QUERY`: Question is empty
- `QUERY_TOO_LONG`: Question exceeds 1000 chars
- `AGENT_ERROR`: Agent processing failed
- `RETRIEVAL_ERROR`: Knowledge base query failed
- `INTERNAL_ERROR`: Unexpected server error

### SearchRequest

Request for direct knowledge base search.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| query | string | Search query | Required, 1-1000 chars |
| k | int | Number of results | 3-8, default 5 |

### SearchResponse

Results from knowledge base search.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| query | string | Original query | Required |
| collection | string | Qdrant collection name | Required |
| count | int | Number of results | >= 0 |
| chunks | RetrievedChunk[] | Retrieved chunks | May be empty |

### RetrievedChunk

Single chunk from knowledge base.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| chunk_id | string | Unique identifier | UUID |
| source_url | string | Source page URL | Valid URL |
| text | string | Chunk content | Required |
| score | float | Similarity score | 0.0-1.0 |
| chapter | string | Chapter name | Optional |
| section | string | Section name | Optional |
| chunk_index | int | Position in source | >= 0 |

## Entity Relationships

```text
QueryRequest ──────────────▶ QueryResponse
     │                            │
     │                            ├── Citation[] (0..*)
     │                            └── ResponseMetadata (1)
     │
     ▼
SearchRequest ─────────────▶ SearchResponse
                                  │
                                  └── RetrievedChunk[] (0..*)
```

## State Transitions

### Query Flow States

```text
RECEIVED → VALIDATING → RETRIEVING → GENERATING → COMPLETED
                │            │            │
                ▼            ▼            ▼
           VALIDATION   RETRIEVAL    GENERATION
             ERROR       ERROR         ERROR
```

### Session Token States

```text
CREATED → ACTIVE → EXPIRED
            │
            ▼
         REFRESHED → ACTIVE
```

## Pydantic Models (Python)

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

class Citation(BaseModel):
    source_url: str
    chapter: str = ""
    section: str = ""
    score: float = Field(..., ge=0.0, le=1.0)
    chunk_index: int = Field(..., ge=0)

class ResponseMetadata(BaseModel):
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    tool_calls: list[str] = []

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    grounded: bool
    refused: bool
    metadata: ResponseMetadata

class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None

class SessionResponse(BaseModel):
    client_secret: str
```

## TypeScript Interfaces (Frontend)

```typescript
interface QueryRequest {
  question: string;
}

interface Citation {
  source_url: string;
  chapter: string;
  section: string;
  score: number;
  chunk_index: number;
}

interface ResponseMetadata {
  retrieval_time_ms: number;
  generation_time_ms: number;
  total_time_ms: number;
  tool_calls: string[];
}

interface QueryResponse {
  answer: string;
  citations: Citation[];
  grounded: boolean;
  refused: boolean;
  metadata: ResponseMetadata;
}

interface ErrorResponse {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

interface SessionResponse {
  client_secret: string;
}
```
