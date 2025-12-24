# CLI Interface Contract: RAG Knowledge Base Pipeline

**Feature**: 001-textbook-rag-kb | **Date**: 2025-12-24

## Overview

The pipeline is executed via a single Python script (`backend/main.py`) with command-line arguments to control pipeline stages.

---

## Entry Point

```bash
python backend/main.py <command> [options]
```

---

## Commands

### 1. `ingest` - Fetch and Process Pages

Fetches all pages from the textbook site and extracts content.

```bash
python backend/main.py ingest [--base-url URL] [--output FILE]
```

**Options**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--base-url` | string | `https://bilal-raza12.github.io/PhysicalAI---Humanoid-Robotics-TextBook` | Base URL of Docusaurus site |
| `--output` | string | `pages.json` | Output file for extracted pages |

**Output**:
- JSON file containing list of `Page` objects
- Console summary: pages fetched, errors, skipped

**Exit Codes**:
- `0`: Success
- `1`: Partial failure (some pages failed, others succeeded)
- `2`: Complete failure (no pages fetched)

---

### 2. `chunk` - Split Content into Chunks

Splits extracted page content into chunks suitable for embedding.

```bash
python backend/main.py chunk [--input FILE] [--output FILE] [--min-tokens N] [--max-tokens N] [--overlap N]
```

**Options**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | string | `pages.json` | Input file with extracted pages |
| `--output` | string | `chunks.json` | Output file for chunks |
| `--min-tokens` | int | `300` | Minimum tokens per chunk |
| `--max-tokens` | int | `500` | Maximum tokens per chunk |
| `--overlap` | int | `50` | Token overlap between chunks |

**Output**:
- JSON file containing list of `Chunk` objects
- Console summary: total chunks, average token count

**Exit Codes**:
- `0`: Success
- `1`: Input file not found or invalid
- `2`: No chunks generated

---

### 3. `embed` - Generate Embeddings

Generates vector embeddings for all chunks using Cohere API.

```bash
python backend/main.py embed [--input FILE] [--output FILE] [--batch-size N]
```

**Options**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | string | `chunks.json` | Input file with chunks |
| `--output` | string | `embeddings.json` | Output file for embeddings |
| `--batch-size` | int | `96` | Chunks per API request (max 96) |

**Environment Variables** (Required):
- `COHERE_API_KEY`: Cohere API key

**Output**:
- JSON file containing chunk IDs and embeddings
- Console summary: chunks embedded, API calls made

**Exit Codes**:
- `0`: Success
- `1`: API key missing or invalid
- `2`: Embedding generation failed

---

### 4. `store` - Upload to Qdrant

Stores embeddings and metadata in Qdrant Cloud.

```bash
python backend/main.py store [--chunks FILE] [--embeddings FILE] [--collection NAME] [--recreate]
```

**Options**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--chunks` | string | `chunks.json` | Chunks file for metadata |
| `--embeddings` | string | `embeddings.json` | Embeddings file |
| `--collection` | string | `textbook_chunks` | Qdrant collection name |
| `--recreate` | flag | `false` | Delete and recreate collection |

**Environment Variables** (Required):
- `QDRANT_URL`: Qdrant Cloud URL
- `QDRANT_API_KEY`: Qdrant API key

**Output**:
- Console summary: points upserted, collection stats

**Exit Codes**:
- `0`: Success
- `1`: Connection or auth failure
- `2`: Upsert failed

---

### 5. `verify` - Validate Pipeline

Runs verification queries to ensure the knowledge base is working.

```bash
python backend/main.py verify [--collection NAME] [--query TEXT]
```

**Options**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--collection` | string | `textbook_chunks` | Qdrant collection name |
| `--query` | string | `"What is ROS 2?"` | Sample search query |

**Environment Variables** (Required):
- `QDRANT_URL`, `QDRANT_API_KEY`: Qdrant credentials
- `COHERE_API_KEY`: For query embedding

**Output**:
- Top 5 search results with scores and metadata
- Collection statistics (vector count, storage size)

**Exit Codes**:
- `0`: Verification passed
- `1`: Search returned no results
- `2`: Connection failure

---

### 6. `run` - Full Pipeline Execution

Runs all stages in sequence: ingest → chunk → embed → store → verify.

```bash
python backend/main.py run [--base-url URL] [--collection NAME] [--recreate]
```

**Options**:
- Combines options from all individual commands
- Intermediate files saved to working directory

**Output**:
- Progress updates for each stage
- Final verification results

**Exit Codes**:
- `0`: Complete success
- `1`: Partial failure (pipeline completed with warnings)
- `2`: Critical failure (pipeline aborted)

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `COHERE_API_KEY` | Yes | Cohere API key for embeddings |
| `QDRANT_URL` | Yes | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Yes | Qdrant API key |
| `LOG_LEVEL` | No | Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO) |

---

## Example Usage

```bash
# Full pipeline with default settings
export COHERE_API_KEY="your-key"
export QDRANT_URL="https://xxx.qdrant.io"
export QDRANT_API_KEY="your-key"

python backend/main.py run

# Individual stages with custom options
python backend/main.py ingest --base-url https://example.com/docs
python backend/main.py chunk --min-tokens 250 --max-tokens 450
python backend/main.py embed --batch-size 50
python backend/main.py store --recreate
python backend/main.py verify --query "How do I use Gazebo?"
```

---

## Error Handling

All commands implement:
- Retry with exponential backoff for transient failures
- Graceful degradation (continue on partial failures)
- Detailed error logging to stderr
- JSON output for machine parsing (with `--json` flag)
