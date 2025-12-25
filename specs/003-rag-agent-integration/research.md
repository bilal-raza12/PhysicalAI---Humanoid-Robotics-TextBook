# Research: RAG Agent Integration

**Feature**: 003-rag-agent-integration | **Date**: 2025-12-25 | **Branch**: `003-rag-agent-integration`

## Research Summary

This document consolidates research findings for key technical decisions in the RAG agent integration with OpenAI Agents SDK.

---

## Decision 1: OpenAI Agents SDK Patterns

### Decision
Use **OpenAI Agents SDK** (openai-agents package) with `@function_tool` decorator for tool definition.

### Rationale
- Official OpenAI agent framework with production-ready patterns
- Simple decorator-based tool definition
- Supports deterministic tool invocation via `ModelSettings(tool_choice="required")`
- Built-in error handling and conversation management

### Implementation Notes

**Agent Initialization:**
```python
from agents import Agent, ModelSettings

# Create agent with system prompt and tools
agent = Agent(
    name="textbook_qa",
    instructions=GROUNDING_PROMPT,
    tools=[search_textbook],
    model_settings=ModelSettings(
        tool_choice="required"  # Forces tool use for every query
    )
)
```

**Tool Definition with @function_tool:**
```python
from agents import function_tool

@function_tool
def search_textbook(query: str) -> str:
    """
    Search the Physical AI & Humanoid Robotics textbook for relevant content.

    Args:
        query: Natural language question about robotics, AI, or related topics

    Returns:
        Retrieved context with source citations, or refusal message
    """
    result, error = search_knowledge_base(query, k=5)
    if error:
        return f"Error: {error}"

    # Format context with citations
    return format_context_with_citations(result)
```

**Running the Agent:**
```python
from agents import Runner

async def ask_question(question: str) -> str:
    result = await Runner.run(agent, question)
    return result.final_output
```

### Key Configuration
- `tool_choice="required"`: Ensures retrieval happens before every response
- Tool return type must be string (will be passed to model as context)
- Agent maintains no conversation history (each question is independent)

---

## Decision 2: Grounding System Prompt

### Decision
Use explicit grounding instructions that enforce context-only responses.

### Rationale
- Prevents hallucination by constraining responses to retrieved content
- Clear refusal rules when context is insufficient
- Mandates source citations for transparency

### Implementation Notes

```python
GROUNDING_PROMPT = """You are a helpful assistant for the Physical AI & Humanoid Robotics textbook.

## CRITICAL RULES

1. **ALWAYS use the search_textbook tool** before answering any question
2. **ONLY use information from retrieved context** - never use your pre-training knowledge
3. **REFUSE to answer** if:
   - No relevant content is found
   - All retrieved chunks have scores below 0.3
   - The question is unrelated to robotics/AI topics
4. **Include source citations** in every answer using this format:
   - [Source: URL | Chapter | Section | Score: X.XX]

## REFUSAL TEMPLATE

When refusing, respond with:
"I cannot answer this question based on the textbook content. The search found no relevant information.
Please try rephrasing your question or ask about topics covered in the Physical AI & Humanoid Robotics textbook."

## CITATION FORMAT

After each statement from retrieved context, cite:
[Source: https://example.com/page | Chapter: X | Section: Y | Score: 0.XX]

## ANSWER STRUCTURE

1. Direct answer to the question (grounded in retrieved content)
2. Supporting details from context
3. Source citations for each point
"""
```

### Refusal Logic
```python
SCORE_THRESHOLD = 0.3  # Below this = refuse to answer

def should_refuse(result: RetrievalResult) -> bool:
    """Determine if agent should refuse to answer."""
    if result.count == 0:
        return True

    # Check if all scores are below threshold
    max_score = max(chunk.score for chunk in result.chunks)
    return max_score < SCORE_THRESHOLD
```

---

## Decision 3: Part 2 Retrieval Integration

### Decision
Wrap existing `search_knowledge_base()` function from Part 2 as an agent tool.

### Rationale
- Reuses tested and verified retrieval logic (41/41 tasks complete)
- Consistent with existing dataclasses (RetrievedChunk, RetrievalResult)
- No duplication of Qdrant/Cohere integration code

### Implementation Notes

**Integration Pattern:**
```python
from main import search_knowledge_base, RetrievalResult, RetrievedChunk

@function_tool
def search_textbook(query: str) -> str:
    """Search textbook and return formatted context."""
    result, error = search_knowledge_base(query, k=5)

    if error:
        return f"[RETRIEVAL_ERROR] {error}"

    if should_refuse(result):
        return "[NO_RELEVANT_CONTEXT] No content found with sufficient relevance."

    return format_context_for_agent(result)
```

**Context Formatting:**
```python
def format_context_for_agent(result: RetrievalResult) -> str:
    """Format retrieved chunks for agent consumption."""
    sections = []

    for i, chunk in enumerate(result.chunks, 1):
        section = f"""
--- Context {i} (Score: {chunk.score:.3f}) ---
Source: {chunk.source_url}
Chapter: {chunk.chapter}
Section: {chunk.section}

{chunk.text}
"""
        sections.append(section)

    return "\n".join(sections)
```

### Existing Functions from Part 2
- `search_knowledge_base(query, k=5)` → `tuple[Optional[RetrievalResult], Optional[str]]`
- `RetrievedChunk`: text, score, source_url, chunk_index, chunk_id, chapter, section, title
- `RetrievalResult`: query, chunks, collection, count property
- `assemble_context(result)` → `AssembledContext`

---

## Decision 4: CLI Interface Design

### Decision
Add `ask` command to existing CLI for agent interaction.

### Rationale
- Consistent with existing CLI structure (verify, ingest, search commands)
- Single entry point for all textbook operations
- Easy to test and demonstrate

### Implementation Notes

**Command Structure:**
```python
@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about the textbook"),
    format: str = typer.Option("text", help="Output format: text or json"),
):
    """Ask a question about the Physical AI & Humanoid Robotics textbook."""
    import asyncio

    response = asyncio.run(ask_agent(question))

    if format == "json":
        print(json.dumps({"question": question, "response": response}))
    else:
        print(response)
```

**Expected Usage:**
```bash
# Basic question
uv run python main.py ask "What is ROS 2?"

# JSON output
uv run python main.py ask "What is ROS 2?" --format json
```

---

## Decision 5: Error Handling Strategy

### Decision
Graceful degradation with user-friendly error messages.

### Rationale
- Agent should never crash on retrieval failures
- Users need actionable feedback
- Errors should be logged for debugging

### Implementation Notes

**Error Categories:**
1. **Retrieval Errors** (Qdrant/Cohere unavailable)
   - Return: "I'm temporarily unable to search the textbook. Please try again later."
   - Log: Full error details

2. **No Relevant Context**
   - Return: Refusal message with suggestion to rephrase
   - Log: Query and max score for analysis

3. **Agent Errors** (OpenAI API issues)
   - Return: "I encountered an error processing your question. Please try again."
   - Log: Full exception

**Implementation:**
```python
async def ask_agent(question: str) -> str:
    """Ask the agent a question with error handling."""
    try:
        result = await Runner.run(agent, question)
        return result.final_output
    except RetrievalError as e:
        logger.error(f"Retrieval error: {e}")
        return "I'm temporarily unable to search the textbook. Please try again later."
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return "I encountered an error processing your question. Please try again."
```

---

## Decision 6: Response Format with Citations

### Decision
Include structured citations in every response.

### Rationale
- Constitution mandates: "Every chatbot answer MUST include citations to relevant page/section"
- Enables verification and trust
- Consistent format aids parsing

### Implementation Notes

**Citation Format:**
```
[Source: https://example.com/docs/ros2 | Chapter: Introduction | Section: What is ROS 2? | Score: 0.85]
```

**Response Structure:**
```
## Answer

[Grounded response text here]

[Source: URL | Chapter | Section | Score]

## Sources Used

1. https://example.com/docs/ros2 (Chapter: Introduction, Section: What is ROS 2?, Score: 0.85)
2. https://example.com/docs/basics (Chapter: Basics, Section: Architecture, Score: 0.72)
```

---

## Open Items

All research questions resolved. No blocking unknowns remain.

## References

- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
- [Part 2 Implementation](../002-rag-retrieval/tasks.md) - 41/41 tasks complete
- [Constitution](../../.specify/memory/constitution.md) - Citation requirements
