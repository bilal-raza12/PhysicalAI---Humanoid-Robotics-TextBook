# Data Model: RAG Knowledge Base Construction

**Feature**: 001-textbook-rag-kb | **Date**: 2025-12-24 | **Branch**: `001-textbook-rag-kb`

## Overview

This document defines the data entities and their relationships for the RAG knowledge base pipeline.

---

## Entities

### Page

Represents a single textbook page fetched from the Docusaurus site.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `url` | string | Full URL of the page | Required, unique, valid URL |
| `raw_html` | string | Original HTML content | Required |
| `clean_text` | string | Extracted main content (no boilerplate) | Required |
| `title` | string | Page title from `<title>` or `<h1>` | Optional |
| `chapter` | string | Chapter identifier (e.g., "ch01", "module-1") | Optional, extracted from URL path |
| `section` | string | Section name from heading hierarchy | Optional |
| `fetched_at` | datetime | Timestamp when page was fetched | Required, ISO 8601 |
| `status` | enum | Processing status: success, error, skipped | Required |

**Validation Rules**:
- URL must be within target domain (`bilal-raza12.github.io`)
- `clean_text` must be non-empty for status=success

**State Transitions**:
- `pending` → `success` (content extracted successfully)
- `pending` → `error` (fetch/parse failure)
- `pending` → `skipped` (no meaningful content)

---

### Chunk

Represents a text segment derived from a Page, sized for embedding.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `chunk_id` | string | Unique identifier (uuid or sequential) | Required, unique |
| `source_url` | string | URL of originating page | Required, references Page.url |
| `text` | string | Chunk text content | Required, non-empty |
| `token_count` | int | Number of tokens in chunk | Required, 300-500 target |
| `chapter` | string | Chapter from source page | Optional |
| `section` | string | Section from source page | Optional |
| `chunk_index` | int | Position within source page (0-indexed) | Required, >= 0 |
| `created_at` | datetime | Timestamp of chunk creation | Required, ISO 8601 |

**Validation Rules**:
- `token_count` should be in 300-500 range (soft constraint)
- `chunk_index` must be sequential per source page
- `text` must not be empty or whitespace-only

**Derivation**:
- Chapter extracted from URL path (e.g., `/docs/module-1-ros2/ch01-intro-ros2` → chapter: "ch01-intro-ros2")
- Section extracted from first heading in chunk or inherited from page

---

### Embedding

Represents the vector embedding of a Chunk.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `embedding_id` | int | Unique identifier (Qdrant point ID) | Required, unique |
| `chunk_id` | string | Reference to source chunk | Required, references Chunk.chunk_id |
| `vector` | float[1024] | Cohere embedding vector | Required, 1024 dimensions |
| `model` | string | Embedding model used | Required, "embed-english-v3.0" |
| `created_at` | datetime | Timestamp of embedding generation | Required, ISO 8601 |

**Validation Rules**:
- `vector` must have exactly 1024 dimensions
- All vector values must be valid floats (no NaN/Inf)

---

### QdrantPayload

The metadata stored alongside each vector in Qdrant.

| Field | Type | Description | Qdrant Type |
|-------|------|-------------|-------------|
| `source_url` | string | Full URL for citation | keyword |
| `chapter` | string | Chapter identifier | keyword |
| `section` | string | Section name | keyword |
| `chunk_index` | int | Position in source page | integer |
| `text` | string | Original chunk text | text |
| `title` | string | Page title | keyword |

**Indexing**:
- `source_url`, `chapter`, `section` indexed as keywords for filtering
- `text` stored for display/debugging but not indexed

---

## Relationships

```
Page (1) ─────< Chunk (many)
                   │
                   │
                   v
            Embedding (1:1)
                   │
                   │
                   v
            QdrantPayload (1:1)
```

- One **Page** produces multiple **Chunks** (chunking process)
- Each **Chunk** has exactly one **Embedding** (embedding generation)
- Each **Embedding** maps to one **QdrantPayload** (vector storage)

---

## In-Memory Data Structures

These are the Python dataclasses used during pipeline execution:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

class PageStatus(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"

@dataclass
class Page:
    url: str
    raw_html: str = ""
    clean_text: str = ""
    title: str = ""
    chapter: str = ""
    section: str = ""
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    status: PageStatus = PageStatus.PENDING
    error_message: Optional[str] = None

@dataclass
class Chunk:
    chunk_id: str
    source_url: str
    text: str
    token_count: int
    chapter: str = ""
    section: str = ""
    chunk_index: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class EmbeddingResult:
    chunk_id: str
    vector: list[float]
    model: str = "embed-english-v3.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
```

---

## Qdrant Collection Configuration

```python
from qdrant_client.models import Distance, VectorParams

COLLECTION_CONFIG = {
    "name": "textbook_chunks",
    "vectors_config": VectorParams(
        size=1024,
        distance=Distance.COSINE
    )
}
```

---

## Data Flow Summary

1. **Ingestion**: Fetch URLs → Create `Page` objects
2. **Extraction**: Parse HTML → Populate `clean_text`, `chapter`, `section`
3. **Chunking**: Split `clean_text` → Create `Chunk` objects (300-500 tokens)
4. **Embedding**: Send chunks to Cohere → Create `EmbeddingResult` objects
5. **Storage**: Upsert to Qdrant → Create points with `QdrantPayload`
6. **Verification**: Query Qdrant → Validate search results return expected content
