# Quickstart: RAG Retrieval Pipeline

**Feature**: 002-rag-retrieval | **Date**: 2025-12-25

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Cohere API key ([dashboard.cohere.com](https://dashboard.cohere.com/))
- Qdrant Cloud account ([cloud.qdrant.io](https://cloud.qdrant.io/))
- **Part 1 Complete**: Knowledge base populated in Qdrant (268 vectors)

---

## 1. Setup

### Verify Environment

```bash
cd backend

# Check .env exists with credentials
cat .env
# Should show:
# COHERE_API_KEY=...
# QDRANT_URL=https://...
# QDRANT_API_KEY=...
```

### Verify Knowledge Base

```bash
uv run python main.py verify
```

Expected output:
```
Collection: textbook_chunks
Points: 268
Verification: PASSED
```

---

## 2. Basic Search

### Text Output (default)

```bash
uv run python main.py search --query "What is ROS 2?"
```

### JSON Output

```bash
uv run python main.py search --query "What is ROS 2?" --format json
```

---

## 3. Configure Results

### Fewer Results (K=3)

```bash
uv run python main.py search --query "How do sensors work?" --k 3
```

### More Results (K=8)

```bash
uv run python main.py search --query "navigation algorithms" --k 8
```

---

## 4. Example Queries

| Query | Expected Topic |
|-------|----------------|
| "What is ROS 2?" | ROS 2 introduction |
| "How does Gazebo simulate sensors?" | Sensor simulation |
| "reinforcement learning for robots" | Isaac Sim, RL training |
| "URDF robot description" | Robot modeling |
| "capstone project requirements" | Final project |

---

## 5. Verify Output Quality

### Check Relevance

```bash
uv run python main.py search --query "How do ROS 2 nodes communicate?"
```

Results should mention:
- Nodes and topics
- Publish/subscribe
- DDS (Data Distribution Service)

### Check Metadata

Each result should include:
- Score (0.0-1.0)
- Source URL
- Chapter
- Section

---

## 6. Handle Edge Cases

### Empty Query

```bash
uv run python main.py search --query ""
# Expected: Error - Query cannot be empty
```

### Non-existent Content

```bash
uv run python main.py search --query "quantum cooking techniques"
# Expected: 0 results, "No matching content found"
```

---

## 7. Troubleshooting

### "Collection does not exist"

Run Part 1 pipeline first:
```bash
uv run python main.py run
```

### "COHERE_API_KEY not set"

Create/update `.env`:
```bash
cp .env.example .env
# Edit .env with your keys
```

### Rate Limit Errors

Wait 60 seconds and retry. The system handles this automatically.

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `search --query "..."` | Search knowledge base |
| `search --query "..." --k 3` | Limit to 3 results |
| `search --query "..." --format json` | JSON output |
| `verify` | Check collection status |

---

## Integration Scenarios

### Scenario 1: Developer Validating Retrieval

```bash
# Run 5 test queries
uv run python main.py search --query "What is ROS 2?"
uv run python main.py search --query "How does Gazebo work?"
uv run python main.py search --query "reinforcement learning"
uv run python main.py search --query "sensor simulation"
uv run python main.py search --query "capstone project"
```

### Scenario 2: Context for LLM

```bash
# Get JSON output for programmatic use
uv run python main.py search --query "How do I set up ROS 2?" --format json > context.json
```

### Scenario 3: Verify Metadata Traceability

```bash
# Search and verify sources are valid URLs
uv run python main.py search --query "ROS 2 installation" --k 3
# Each result should have a clickable source_url
```
