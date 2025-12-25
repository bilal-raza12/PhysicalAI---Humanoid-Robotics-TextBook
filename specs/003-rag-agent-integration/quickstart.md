# Quickstart: RAG Agent Integration

**Feature**: 003-rag-agent-integration | **Date**: 2025-12-25

## Prerequisites

Before using the RAG agent, ensure:

1. **Part 1 Complete**: Knowledge base has embedded chunks in Qdrant
2. **Part 2 Complete**: Search command works (`uv run python main.py search --query "ROS 2"`)
3. **Environment Variables Set**:
   ```bash
   OPENAI_API_KEY=sk-...     # For agent
   COHERE_API_KEY=...        # For embeddings (Part 2)
   QDRANT_URL=...            # For vector search
   QDRANT_API_KEY=...        # For vector search
   ```

## Installation

```bash
cd backend

# Install dependencies (adds openai-agents to existing deps)
uv add openai-agents

# Verify installation
uv run python -c "from agents import Agent; print('OK')"
```

## Basic Usage

### Ask a Question

```bash
# Simple question about the textbook
uv run python main.py ask "What is ROS 2?"

# Expected output:
# Question: What is ROS 2?
#
# Answer:
# ROS 2 (Robot Operating System 2) is a flexible framework for writing robot
# software. It provides libraries and tools to help developers create robot
# applications...
#
# Sources:
# [1] https://example.com/docs/ros2-intro
#     Chapter: Introduction | Section: What is ROS 2? | Score: 0.87
```

### JSON Output

```bash
uv run python main.py ask "What is ROS 2?" --format json
```

Output:
```json
{
  "question": "What is ROS 2?",
  "answer": "ROS 2 is...",
  "grounded": true,
  "refused": false,
  "citations": [...]
}
```

### Verbose Mode

```bash
uv run python main.py ask "What is ROS 2?" --verbose
```

Shows retrieval details, timing, and chunk scores.

## Configuration Options

### Adjust Chunk Count (K)

```bash
# Retrieve 3 chunks instead of default 5
uv run python main.py ask "What is ROS 2?" --k 3

# Retrieve maximum (8) chunks
uv run python main.py ask "What is ROS 2?" --k 8
```

### Adjust Score Threshold

```bash
# Lower threshold (accepts less relevant results)
uv run python main.py ask "quantum computing" --threshold 0.1

# Higher threshold (more strict relevance)
uv run python main.py ask "What is ROS 2?" --threshold 0.8
```

## Expected Behaviors

### Grounded Answers

When relevant content exists:
- Agent retrieves chunks from Qdrant
- Response uses ONLY retrieved context
- Each claim has a source citation
- Citations include URL, chapter, section, score

### Refusal Responses

When content is irrelevant or missing:
```
Question: What is the capital of France?

I cannot answer this question based on the textbook content.
No relevant information was found.

Please try rephrasing your question or ask about topics covered in the
Physical AI & Humanoid Robotics textbook.
```

### Error Handling

```bash
# If Qdrant is unavailable
uv run python main.py ask "What is ROS 2?"
# Output: Error: Unable to connect to the knowledge base.

# If OpenAI API fails
uv run python main.py ask "What is ROS 2?"
# Output: Error: Agent encountered an error. Please try again.
```

## Example Questions

Try these questions to explore the textbook:

```bash
# Core robotics concepts
uv run python main.py ask "What is ROS 2?"
uv run python main.py ask "How do sensors work in robotics?"
uv run python main.py ask "What is motion planning?"

# Specific topics
uv run python main.py ask "How does NVIDIA Isaac work?"
uv run python main.py ask "What are the prerequisites for this textbook?"

# Off-topic (should refuse)
uv run python main.py ask "What is the capital of France?"
uv run python main.py ask "How do I cook pasta?"
```

## Verification Checklist

Run these commands to verify the agent works:

```bash
# 1. Verify Part 2 still works
uv run python main.py search --query "ROS 2"
# Expected: Returns chunks with scores

# 2. Basic agent question
uv run python main.py ask "What is ROS 2?"
# Expected: Grounded answer with citations

# 3. Refusal test
uv run python main.py ask "What is the capital of France?"
# Expected: Refusal message

# 4. JSON output
uv run python main.py ask "What is ROS 2?" --format json | python -m json.tool
# Expected: Valid JSON

# 5. Verbose mode
uv run python main.py ask "What is ROS 2?" --verbose
# Expected: Detailed retrieval info
```

## Troubleshooting

### "OPENAI_API_KEY not set"

```bash
export OPENAI_API_KEY="sk-..."
# Or add to .env file
```

### "No content retrieved"

1. Verify Part 1 completed: `uv run python main.py verify`
2. Check Qdrant connection: `uv run python main.py search --query "test"`
3. Lower threshold: `--threshold 0.1`

### "Agent error"

1. Check OpenAI API key is valid
2. Check API quota/limits
3. Try again (may be transient)

### Slow Responses

- Retrieval: ~500ms (Qdrant + Cohere)
- Generation: ~1-2s (OpenAI)
- Total: ~2-3s typical

If slower, check network latency to cloud services.
