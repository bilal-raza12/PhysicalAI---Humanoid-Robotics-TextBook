# Quickstart: RAG Knowledge Base Pipeline

**Feature**: 001-textbook-rag-kb | **Date**: 2025-12-24

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Cohere API key ([get one free](https://dashboard.cohere.com/))
- Qdrant Cloud account ([free tier](https://cloud.qdrant.io/))

---

## 1. Setup

### Clone and Navigate

```bash
cd Physical_ai_&_Humanoid_Robotics
```

### Initialize Python Environment

```bash
cd backend
uv init
uv add httpx beautifulsoup4 cohere qdrant-client tiktoken python-dotenv
```

### Configure Environment

Create `.env` file in `backend/`:

```bash
# backend/.env
COHERE_API_KEY=your-cohere-api-key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

---

## 2. Run the Pipeline

### Full Pipeline (Recommended)

```bash
uv run python main.py run
```

This executes all stages:
1. Fetches pages from the textbook site
2. Chunks content into 300-500 token segments
3. Generates Cohere embeddings
4. Stores vectors in Qdrant
5. Runs verification queries

### Individual Stages

```bash
# Step 1: Ingest pages
uv run python main.py ingest

# Step 2: Chunk content
uv run python main.py chunk

# Step 3: Generate embeddings
uv run python main.py embed

# Step 4: Store in Qdrant
uv run python main.py store

# Step 5: Verify
uv run python main.py verify
```

---

## 3. Verify Success

### Check Collection Stats

```bash
uv run python main.py verify
```

Expected output:
```
Collection: textbook_chunks
Vectors: 847
Storage: 12.3 MB

Sample query: "What is ROS 2?"
Top results:
  1. [0.89] docs/module-1-ros2/ch01-intro-ros2 - "ROS 2 is the second generation..."
  2. [0.85] docs/intro - "This textbook covers ROS 2..."
  3. [0.82] docs/prerequisites - "Before starting, ensure ROS 2..."
```

### Test a Search

```bash
uv run python main.py verify --query "How does Gazebo simulate sensors?"
```

---

## 4. Common Issues

### API Key Missing

```
Error: COHERE_API_KEY not set
Solution: Ensure .env file exists and is properly formatted
```

### Rate Limited

```
Error: Cohere API rate limit exceeded
Solution: Pipeline automatically retries with backoff. Wait and retry.
```

### Qdrant Connection Failed

```
Error: Cannot connect to Qdrant
Solution: Verify QDRANT_URL and QDRANT_API_KEY in .env
```

---

## 5. Next Steps

After successful pipeline execution:

1. **Build retrieval layer**: Use Qdrant search in your application
2. **Integrate with chatbot**: Connect to OpenAI Agents or similar
3. **Set up CI**: Automate re-indexing on content changes

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `uv run python main.py run` | Full pipeline |
| `uv run python main.py verify` | Check collection |
| `uv run python main.py store --recreate` | Reset collection |

| File | Purpose |
|------|---------|
| `pages.json` | Extracted page content |
| `chunks.json` | Text chunks with metadata |
| `embeddings.json` | Vector embeddings |
| `.env` | API credentials |
