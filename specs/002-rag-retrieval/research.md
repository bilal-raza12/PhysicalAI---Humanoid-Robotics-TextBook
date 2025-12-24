# Research: RAG Retrieval Pipeline

**Feature**: 002-rag-retrieval | **Date**: 2025-12-25 | **Branch**: `002-rag-retrieval`

## Research Questions

This document captures technical decisions and research findings for the RAG retrieval pipeline.

---

## RQ-001: Top-K Value Selection Strategy

**Question**: How should the K parameter be handled for optimal retrieval quality?

### Decision

Default K=5 with configurable range 3-8. Values outside range are clamped.

### Rationale

- **K=5 as default**: Provides sufficient context without overwhelming the caller
- **Range 3-8**:
  - Minimum 3 ensures enough context for diverse queries
  - Maximum 8 prevents context window bloat for downstream LLMs
- **Clamping vs Error**: Clamping is more user-friendly than rejecting invalid values

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| Fixed K | Doesn't allow tuning for different use cases |
| K=10+ max | Too many chunks can reduce relevance and increase token costs |
| Dynamic K based on scores | Adds complexity; caller can filter by score if needed |

---

## RQ-002: Similarity Threshold Handling

**Question**: Should low-similarity results be filtered out automatically?

### Decision

No automatic filtering. Return all K results regardless of score. Filtering is caller's responsibility.

### Rationale

- **Transparency**: Caller sees all results and their scores
- **Flexibility**: Different use cases have different relevance thresholds
- **Debugging**: Low scores help identify knowledge gaps
- **Spec alignment**: FR-006 and US4 require returning all results with scores

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| Hard threshold (e.g., >0.5) | Arbitrary; varies by query complexity |
| Dynamic threshold | Adds complexity; no clear algorithm |
| Return empty if all low | User may still want best-effort results |

---

## RQ-003: Metadata Filtering Strategy

**Question**: Should retrieval support filtering by chapter/section?

### Decision

Out of scope for Part 2. Focus on pure similarity search with full metadata returned.

### Rationale

- **Spec alignment**: Out-of-scope section explicitly excludes metadata-based filtering
- **Simplicity**: Part 2 focuses on core retrieval; filtering can be added in Part 3
- **Metadata visibility**: All metadata is returned for downstream filtering

### Future Consideration

If filtering is needed later:
- Add `--chapter` and `--section` CLI flags
- Use Qdrant payload filters in query

---

## RQ-004: Context Assembly Order

**Question**: How should retrieved chunks be ordered in the assembled context?

### Decision

Order by relevance score (descending). Highest similarity first.

### Rationale

- **FR-006 compliance**: Spec requires descending score order
- **LLM context**: Most relevant content at the beginning is more impactful
- **User expectation**: Standard behavior for search results

### Format

```text
[Result 1]
Score: 0.89
Source: https://example.com/docs/chapter1
Chapter: module-1-ros2
Section: Introduction
---
[chunk text]

[Result 2]
...
```

---

## RQ-005: Query Embedding Model Consistency

**Question**: How do we ensure query embeddings match document embeddings?

### Decision

Use identical Cohere configuration:
- Model: `embed-english-v3.0`
- Input type: `search_query` (not `search_document`)
- Dimensions: 1024

### Rationale

- **Cohere best practice**: Use `search_query` for queries, `search_document` for documents
- **Consistency**: Same model ensures compatible vector space
- **Part 1 alignment**: Documents were embedded with `embed-english-v3.0`

### Implementation

```python
# Query embedding
response = cohere_client.embed(
    texts=[query],
    model="embed-english-v3.0",
    input_type="search_query",  # Different from ingestion!
    embedding_types=["float"]
)
```

---

## RQ-006: Rate Limit Handling

**Question**: How should API rate limits be handled during retrieval?

### Decision

Implement exponential backoff with:
- Initial wait: 60 seconds for rate limit (429) errors
- Max retries: 5
- Regular errors: 2^n seconds backoff

### Rationale

- **Cohere trial limit**: 100,000 tokens/minute
- **Single query**: Unlikely to hit limit, but retry logic ensures resilience
- **Consistency**: Same pattern as Part 1 ingestion

---

## RQ-007: Empty/Error Result Handling

**Question**: How should the system communicate empty or error states?

### Decision

Clear structured responses for each scenario:

| Scenario | Response |
|----------|----------|
| No results | Empty list + "No matching content found" message |
| Collection missing | Error + "Collection does not exist" message |
| Connection failure | Error + "Cannot connect to Qdrant" message |
| Empty query | Validation error + "Query cannot be empty" message |

### Rationale

- **Developer experience**: Clear messages enable debugging
- **Programmatic handling**: Structured errors can be caught and processed
- **FR-007, FR-012 compliance**: Spec requires graceful handling and clear messages

---

## Technology Verification

### Qdrant Client Compatibility

**Finding**: qdrant-client >= 1.7 uses `query_points()` instead of `search()`.

```python
# Correct API (qdrant-client >= 1.7)
results = client.query_points(
    collection_name="textbook_chunks",
    query=query_vector,
    limit=k,
    with_payload=True,
).points
```

### Cohere Embedding Response

**Finding**: Access embeddings via `response.embeddings.float_[0]`.

```python
response = cohere_client.embed(
    texts=[query],
    model="embed-english-v3.0",
    input_type="search_query",
    embedding_types=["float"]
)
vector = response.embeddings.float_[0]
```

---

## Summary of Decisions

| Topic | Decision |
|-------|----------|
| Default K | 5 (range 3-8, clamped) |
| Score filtering | None (caller's responsibility) |
| Metadata filtering | Out of scope (future enhancement) |
| Result order | Descending by score |
| Query embedding | `embed-english-v3.0` with `search_query` |
| Rate limiting | 60s backoff for 429, 5 max retries |
| Error handling | Structured messages per scenario |
