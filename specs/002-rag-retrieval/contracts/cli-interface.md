# CLI Contract: RAG Retrieval Pipeline

**Feature**: 002-rag-retrieval | **Date**: 2025-12-25

## Overview

This document defines the CLI interface contract for the RAG retrieval pipeline. The retrieval functionality extends the existing `backend/main.py` CLI.

---

## Command: `search`

### Synopsis

```bash
uv run python main.py search --query "<query_text>" [--k <num>] [--format <format>]
```

### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--query` | string | Yes | - | Natural language search query |
| `--k` | int | No | 5 | Number of results (3-8) |
| `--format` | string | No | "text" | Output format: "text" or "json" |
| `--collection` | string | No | "textbook_chunks" | Qdrant collection name |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Partial success (warnings) |
| 2 | Critical error (connection, empty query) |

---

## Output Format: Text (default)

### Success with Results

```text
==================================================
Search Results
==================================================
Query: "How does ROS 2 handle node communication?"
Collection: textbook_chunks
Results: 5

==================================================

[1] Score: 0.584
Source: https://bilal-raza12.github.io/.../ch01-intro-ros2
Chapter: module-1-ros2
Section: Introduction
---
ROS 2 is the second generation of the Robot Operating System...

--------------------------------------------------

[2] Score: 0.536
Source: https://bilal-raza12.github.io/.../ch02-nodes-topics
Chapter: module-1-ros2
Section: Nodes and Topics
---
Nodes in ROS 2 communicate through a publish-subscribe pattern...

--------------------------------------------------

... (remaining results)

==================================================
Context assembled: 5 chunks, 2847 characters
Sources: 3 unique pages
==================================================
```

### Success with No Results

```text
==================================================
Search Results
==================================================
Query: "quantum computing recipes"
Collection: textbook_chunks
Results: 0

No matching content found in the knowledge base.
==================================================
```

### Error Output

```text
==================================================
Error
==================================================
Code: CONNECTION_ERROR
Message: Cannot connect to Qdrant at https://...
Details: Connection refused

Exit code: 2
==================================================
```

---

## Output Format: JSON

### Success with Results

```json
{
  "status": "success",
  "query": "How does ROS 2 handle node communication?",
  "collection": "textbook_chunks",
  "count": 5,
  "chunks": [
    {
      "rank": 1,
      "score": 0.584,
      "source_url": "https://bilal-raza12.github.io/.../ch01-intro-ros2",
      "chapter": "module-1-ros2",
      "section": "Introduction",
      "title": "Chapter 1: Introduction to ROS 2",
      "chunk_index": 0,
      "chunk_id": "abc123",
      "text": "ROS 2 is the second generation..."
    }
  ],
  "context": {
    "chunk_count": 5,
    "total_chars": 2847,
    "sources": ["https://..."]
  }
}
```

### Success with No Results

```json
{
  "status": "success",
  "query": "quantum computing recipes",
  "collection": "textbook_chunks",
  "count": 0,
  "chunks": [],
  "message": "No matching content found in the knowledge base."
}
```

### Error

```json
{
  "status": "error",
  "code": "CONNECTION_ERROR",
  "message": "Cannot connect to Qdrant at https://...",
  "details": "Connection refused"
}
```

---

## Validation Rules

### Query Validation

| Rule | Action |
|------|--------|
| Empty string | Error: "Query cannot be empty" |
| Whitespace only | Error: "Query cannot be empty" |
| > 1000 chars | Warning + truncate to 1000 |

### K Parameter Validation

| Value | Action |
|-------|--------|
| < 3 | Clamp to 3, log warning |
| > 8 | Clamp to 8, log warning |
| 3-8 | Use as-is |
| Non-integer | Error: "K must be an integer" |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `COHERE_API_KEY` | Yes | Cohere API key for embeddings |
| `QDRANT_URL` | Yes | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Yes | Qdrant API key |

---

## Examples

### Basic Search

```bash
uv run python main.py search --query "What is ROS 2?"
```

### Search with K=3

```bash
uv run python main.py search --query "How do sensors work in Gazebo?" --k 3
```

### JSON Output

```bash
uv run python main.py search --query "reinforcement learning" --format json
```

### Custom Collection

```bash
uv run python main.py search --query "navigation" --collection my_collection
```

---

## Test Cases

### TC-001: Basic Query

**Input**: `--query "What is ROS 2?"`
**Expected**: 5 results with scores, metadata, and text

### TC-002: Custom K

**Input**: `--query "sensors" --k 3`
**Expected**: Exactly 3 results

### TC-003: K Clamping

**Input**: `--query "sensors" --k 15`
**Expected**: 8 results (clamped), warning logged

### TC-004: Empty Query

**Input**: `--query ""`
**Expected**: Exit code 2, error message

### TC-005: No Results

**Input**: `--query "quantum cooking techniques"`
**Expected**: 0 results, "No matching content" message

### TC-006: JSON Format

**Input**: `--query "navigation" --format json`
**Expected**: Valid JSON with all fields

### TC-007: Connection Error

**Precondition**: Invalid QDRANT_URL
**Expected**: Exit code 2, CONNECTION_ERROR

### TC-008: Missing API Key

**Precondition**: No COHERE_API_KEY
**Expected**: Exit code 2, clear error message
