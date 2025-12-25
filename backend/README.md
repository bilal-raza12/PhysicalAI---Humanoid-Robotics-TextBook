# RAG Knowledge Base Backend

A CLI tool for building and querying a RAG (Retrieval-Augmented Generation) knowledge base for the Physical AI & Humanoid Robotics textbook.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Cohere API key (for embeddings)
- Qdrant Cloud account (for vector storage)

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Create `.env` file with your credentials:
```bash
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

## Commands

### Part 1: Knowledge Base Pipeline

#### Run Full Pipeline
Ingest, chunk, embed, and store content in one command:
```bash
uv run python main.py run
```

#### Individual Steps
```bash
# Ingest pages from textbook site
uv run python main.py ingest

# Chunk content into 300-500 token segments
uv run python main.py chunk

# Generate Cohere embeddings
uv run python main.py embed

# Store in Qdrant Cloud
uv run python main.py store

# Verify the pipeline
uv run python main.py verify
```

### Part 2: Search Command

Query the knowledge base with natural language:

```bash
# Basic search
uv run python main.py search --query "What is ROS 2?"

# Custom number of results (K must be 3-8)
uv run python main.py search --query "digital twin simulation" --k 3

# JSON output format
uv run python main.py search --query "NVIDIA Isaac" --format json

# Custom collection name
uv run python main.py search --query "sensors" --collection my_collection
```

#### Search Command Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--query` | `-q` | (required) | Natural language search query |
| `--k` | | 5 | Number of results (3-8, auto-clamped) |
| `--format` | `-f` | text | Output format: `text` or `json` |
| `--collection` | | textbook_chunks | Qdrant collection name |

#### Example Output (Text)
```
==================================================
Search Results
==================================================
Query: "What is ROS 2?"
Collection: textbook_chunks
Results: 3

==================================================

[1] Score: 0.584
Source: https://bilal-raza12.github.io/.../ch01-intro-ros2
Chapter: module-1-ros2
Section: Learning Objectives
---
ROS 2 is the second generation of the Robot Operating System...

--------------------------------------------------

==================================================
Context assembled: 3 chunks, 6338 characters
Sources: 1 unique pages
==================================================
```

#### Example Output (JSON)
```json
{
  "status": "success",
  "query": "What is ROS 2?",
  "collection": "textbook_chunks",
  "count": 3,
  "chunks": [
    {
      "rank": 1,
      "score": 0.584,
      "source_url": "https://...",
      "chapter": "module-1-ros2",
      "section": "Learning Objectives",
      "text": "..."
    }
  ],
  "context": {
    "chunk_count": 3,
    "total_chars": 6338,
    "sources": ["https://..."]
  }
}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Partial failure or warning |
| 2 | Error (connection, missing data, invalid input) |

## Architecture

```
Query -> Validate -> Cohere Embed -> Qdrant Search -> Format Output
```

- **Embedding Model**: Cohere embed-english-v3.0 (1024 dimensions)
- **Vector Distance**: Cosine similarity
- **Input Type**: `search_query` for queries, `search_document` for chunks
