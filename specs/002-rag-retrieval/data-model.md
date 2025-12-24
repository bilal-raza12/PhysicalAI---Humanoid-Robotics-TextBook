# Data Model: RAG Retrieval Pipeline

**Feature**: 002-rag-retrieval | **Date**: 2025-12-25 | **Branch**: `002-rag-retrieval`

## Overview

This document defines the data entities for the RAG retrieval pipeline. Since this is a retrieval-only feature (Part 2), we define input/output structures rather than persistent storage models.

---

## Input Entities

### Query

Represents a user's natural language search query.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `text` | string | The natural language query | Required, non-empty, max 1000 chars |
| `k` | int | Number of results to retrieve | Optional, default=5, range 3-8 |

**Validation Rules**:
- `text` must not be empty or whitespace-only
- `k` is clamped to valid range if outside 3-8

---

## Output Entities

### QueryEmbedding

Internal representation of the embedded query vector.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `vector` | float[1024] | Cohere embedding vector | Required, 1024 dimensions |
| `model` | string | Embedding model used | Required, "embed-english-v3.0" |
| `input_type` | string | Cohere input type | Required, "search_query" |

---

### RetrievedChunk

A single chunk retrieved from Qdrant with its metadata and score.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `text` | string | Chunk content | Required |
| `score` | float | Cosine similarity score | Required, 0.0-1.0 |
| `source_url` | string | URL of the source page | Required |
| `chapter` | string | Chapter identifier | Optional |
| `section` | string | Section name | Optional |
| `chunk_index` | int | Position in source page | Required, >= 0 |
| `title` | string | Page title | Optional |
| `chunk_id` | string | Unique chunk identifier | Required |

**Derivation**:
- All fields except `score` come from Qdrant payload
- `score` comes from Qdrant similarity search result

---

### RetrievalResult

Collection of retrieved chunks with assembly metadata.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `query` | string | Original query text | Required |
| `chunks` | RetrievedChunk[] | List of retrieved chunks | Required, 0 to K items |
| `count` | int | Number of chunks returned | Required, >= 0 |
| `collection` | string | Qdrant collection name | Required |

**State**:
- `count == 0`: No matching content found
- `count < k`: Fewer chunks available than requested
- `count == k`: Full result set

---

### ContextBlock

Formatted context for a single chunk (used in context assembly).

| Field | Type | Description |
|-------|------|-------------|
| `header` | string | Formatted header with score and metadata |
| `body` | string | Chunk text content |
| `separator` | string | Visual separator between chunks |

**Format Example**:
```text
[Result 1] Score: 0.89
Source: https://bilal-raza12.github.io/.../ch01-intro-ros2
Chapter: module-1-ros2 | Section: Introduction
---
ROS 2 is the second generation of the Robot Operating System...
```

---

### AssembledContext

Complete formatted context ready for downstream use.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `formatted_text` | string | Full formatted context | Required |
| `chunk_count` | int | Number of chunks included | Required, >= 0 |
| `total_chars` | int | Total character count | Required, >= 0 |
| `sources` | string[] | List of unique source URLs | Required |

---

## Error Entities

### RetrievalError

Represents an error during retrieval.

| Field | Type | Description |
|-------|------|-------------|
| `code` | string | Error code (e.g., "CONNECTION_ERROR", "EMPTY_QUERY") |
| `message` | string | Human-readable error message |
| `details` | string | Optional additional details |

**Error Codes**:
- `EMPTY_QUERY`: Query text was empty or whitespace
- `CONNECTION_ERROR`: Cannot connect to Qdrant or Cohere
- `COLLECTION_NOT_FOUND`: Qdrant collection doesn't exist
- `RATE_LIMIT`: API rate limit exceeded (after retries)

---

## Relationships

```text
Query (input)
    │
    ▼
QueryEmbedding (internal)
    │
    ▼
RetrievedChunk[] (from Qdrant)
    │
    ▼
RetrievalResult
    │
    ▼
AssembledContext (output)
```

---

## Data Flow

1. **Input**: User provides `Query` with text and optional K
2. **Embedding**: Query is embedded to `QueryEmbedding`
3. **Search**: Qdrant returns `RetrievedChunk[]` with scores
4. **Assembly**: Chunks are wrapped in `RetrievalResult`
5. **Formatting**: Result is formatted as `AssembledContext`
6. **Output**: Formatted context returned to caller

---

## In-Memory Data Structures (Python)

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Query:
    """User's search query."""
    text: str
    k: int = 5

@dataclass
class RetrievedChunk:
    """A chunk retrieved from Qdrant."""
    text: str
    score: float
    source_url: str
    chunk_index: int
    chunk_id: str
    chapter: str = ""
    section: str = ""
    title: str = ""

@dataclass
class RetrievalResult:
    """Collection of retrieved chunks."""
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    collection: str = "textbook_chunks"

    @property
    def count(self) -> int:
        return len(self.chunks)

@dataclass
class AssembledContext:
    """Formatted context for downstream use."""
    formatted_text: str
    chunk_count: int
    total_chars: int
    sources: list[str] = field(default_factory=list)
```

---

## Qdrant Collection (Reference from Part 1)

The retrieval pipeline queries the existing `textbook_chunks` collection:

| Field | Qdrant Type | Description |
|-------|-------------|-------------|
| `id` | int | Point ID |
| `vector` | float[1024] | Cohere embedding |
| `payload.source_url` | keyword | Source page URL |
| `payload.chapter` | keyword | Chapter identifier |
| `payload.section` | keyword | Section name |
| `payload.chunk_index` | integer | Position in page |
| `payload.text` | text | Chunk content |
| `payload.title` | keyword | Page title |
| `payload.chunk_id` | keyword | Unique chunk ID |
