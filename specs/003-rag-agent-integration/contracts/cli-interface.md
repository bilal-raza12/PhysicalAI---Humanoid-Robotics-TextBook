# CLI Interface Contract: RAG Agent

**Feature**: 003-rag-agent-integration | **Date**: 2025-12-25

## Overview

This document defines the CLI contract for the `ask` command that interfaces with the RAG agent.

---

## Command: `ask`

Ask a question about the Physical AI & Humanoid Robotics textbook.

### Syntax

```bash
uv run python main.py ask <QUESTION> [OPTIONS]
```

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `QUESTION` | string | Yes | Natural language question about the textbook |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | text\|json | text | Output format |
| `--k` | int | 5 | Number of chunks to retrieve (3-8) |
| `--threshold` | float | 0.3 | Minimum relevance score (0.0-1.0) |
| `--verbose` | flag | false | Show retrieval details |

---

## Output Formats

### Text Format (default)

**Successful Answer:**
```
Question: What is ROS 2?

Answer:
ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides
libraries and tools to help developers create robot applications, including hardware abstraction,
low-level device control, and message-passing between processes.

Sources:
[1] https://example.com/docs/ros2-intro
    Chapter: Introduction | Section: What is ROS 2? | Score: 0.87

[2] https://example.com/docs/ros2-architecture
    Chapter: Architecture | Section: Core Concepts | Score: 0.72
```

**Refusal (no relevant content):**
```
Question: What is the capital of France?

I cannot answer this question based on the textbook content. No relevant information was found.

Please try rephrasing your question or ask about topics covered in the Physical AI & Humanoid
Robotics textbook, such as:
- ROS 2 and robot software
- Sensors and actuators
- Motion planning
- Computer vision for robotics
```

**Error Response:**
```
Question: What is ROS 2?

Error: Unable to connect to the knowledge base. Please try again later.
```

### JSON Format

**Successful Answer:**
```json
{
  "question": "What is ROS 2?",
  "answer": "ROS 2 (Robot Operating System 2) is a flexible framework...",
  "grounded": true,
  "refused": false,
  "citations": [
    {
      "source_url": "https://example.com/docs/ros2-intro",
      "chapter": "Introduction",
      "section": "What is ROS 2?",
      "score": 0.87,
      "chunk_index": 0
    },
    {
      "source_url": "https://example.com/docs/ros2-architecture",
      "chapter": "Architecture",
      "section": "Core Concepts",
      "score": 0.72,
      "chunk_index": 1
    }
  ],
  "metadata": {
    "retrieval_time_ms": 450,
    "generation_time_ms": 1200,
    "total_time_ms": 1650,
    "tool_calls": ["search_textbook"]
  }
}
```

**Refusal Response:**
```json
{
  "question": "What is the capital of France?",
  "answer": "I cannot answer this question based on the textbook content. No relevant information was found.",
  "grounded": false,
  "refused": true,
  "citations": [],
  "refusal_reason": "No content retrieved from knowledge base",
  "metadata": {
    "retrieval_time_ms": 380,
    "generation_time_ms": 0,
    "total_time_ms": 380,
    "tool_calls": ["search_textbook"]
  }
}
```

**Error Response:**
```json
{
  "question": "What is ROS 2?",
  "answer": null,
  "error": "Unable to connect to the knowledge base. Please try again later.",
  "error_code": "RETRIEVAL_ERROR",
  "grounded": false,
  "refused": false
}
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (answer provided or graceful refusal) |
| 1 | Error (retrieval failure, API error, etc.) |
| 2 | Invalid arguments |

---

## Test Cases

### TC-001: Basic Question Returns Answer

```bash
uv run python main.py ask "What is ROS 2?"
```

**Expected**: Answer with citations, exit code 0

### TC-002: Off-Topic Question Returns Refusal

```bash
uv run python main.py ask "What is the capital of France?"
```

**Expected**: Refusal message, exit code 0

### TC-003: JSON Format Returns Valid JSON

```bash
uv run python main.py ask "What is ROS 2?" --format json | python -m json.tool
```

**Expected**: Valid JSON output, exit code 0

### TC-004: Custom K Parameter Works

```bash
uv run python main.py ask "What is ROS 2?" --k 3 --verbose
```

**Expected**: Exactly 3 chunks in verbose output

### TC-005: Empty Question Returns Error

```bash
uv run python main.py ask ""
```

**Expected**: Error message, exit code 2

### TC-006: Verbose Mode Shows Details

```bash
uv run python main.py ask "What is ROS 2?" --verbose
```

**Expected**: Shows retrieval scores, timing, chunk text

### TC-007: Low Threshold Accepts More Results

```bash
uv run python main.py ask "quantum computing" --threshold 0.1
```

**Expected**: May return answer even with low scores

### TC-008: High Threshold Refuses More

```bash
uv run python main.py ask "What is ROS 2?" --threshold 0.9
```

**Expected**: May refuse if no chunks score above 0.9

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for agent |
| `COHERE_API_KEY` | Yes | Cohere API key for embeddings |
| `QDRANT_URL` | Yes | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud API key |

---

## Verbose Output

When `--verbose` is enabled:

```
Question: What is ROS 2?

=== Retrieval Details ===
Query: What is ROS 2?
K: 5
Threshold: 0.3

Chunks Retrieved: 5
Max Score: 0.87
Min Score: 0.45

[1] Score: 0.87 | https://example.com/docs/ros2-intro
    Chapter: Introduction | Section: What is ROS 2?
    Text: ROS 2 (Robot Operating System 2) is a flexible framework...

[2] Score: 0.72 | https://example.com/docs/ros2-architecture
    Chapter: Architecture | Section: Core Concepts
    Text: The ROS 2 architecture consists of nodes, topics...

...

=== Agent Response ===
Tool Calls: search_textbook
Retrieval Time: 450ms
Generation Time: 1200ms
Total Time: 1650ms

Answer:
ROS 2 (Robot Operating System 2) is a flexible framework...

Sources:
[1] https://example.com/docs/ros2-intro (Score: 0.87)
[2] https://example.com/docs/ros2-architecture (Score: 0.72)
```

---

## Notes

- All questions trigger the retrieval tool (deterministic invocation)
- Refusals are graceful, not errors
- Citations always use the same format for consistency
- Timing is measured separately for retrieval and generation
