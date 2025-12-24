# Feature Specification: RAG Retrieval Pipeline

**Feature Branch**: `002-rag-retrieval`
**Created**: 2025-12-25
**Status**: Draft
**Input**: User description: "RAG Retrieval Pipeline – Part 2: Semantic retrieval of textbook content from Qdrant"

## Summary

Build a semantic retrieval system that queries the textbook knowledge base stored in Qdrant Cloud. This pipeline accepts natural language queries, embeds them using the same Cohere model used during ingestion, retrieves the top-K most relevant chunks, and returns results with full metadata for context assembly.

## User Scenarios & Testing

### User Story 1 - Query Textbook Content (Priority: P1) 🎯 MVP

A developer wants to retrieve relevant textbook content by asking a natural language question. The system embeds the query and finds semantically similar chunks from the vector database.

**Why this priority**: Core functionality - without semantic search, no retrieval is possible.

**Independent Test**: Can be fully tested by running a query command and verifying relevant results are returned with metadata.

**Acceptance Scenarios**:

1. **Given** the textbook knowledge base is populated in Qdrant, **When** the user submits a query "How does ROS 2 handle node communication?", **Then** the system returns top-K chunks related to ROS 2 communication.

2. **Given** a valid query is submitted, **When** results are returned, **Then** each result includes source URL, chapter, section, and relevance score.

3. **Given** a query is submitted, **When** the embedding is generated, **Then** the same Cohere model (embed-english-v3.0) and input_type="search_query" are used.

---

### User Story 2 - Configure Retrieval Parameters (Priority: P1)

A developer wants to control the number of results returned and filter by metadata to get more targeted responses.

**Why this priority**: Essential for tuning retrieval quality and enabling focused searches.

**Independent Test**: Can test by varying K parameter (3-8) and verifying correct number of results returned.

**Acceptance Scenarios**:

1. **Given** the user specifies K=5, **When** a query is executed, **Then** exactly 5 results are returned (or fewer if the collection has fewer matches).

2. **Given** the user specifies K=3, **When** a query is executed, **Then** exactly 3 results are returned.

3. **Given** K is set outside the valid range (e.g., K=10), **When** a query is executed, **Then** the system clamps K to the valid range (3-8) and proceeds.

---

### User Story 3 - Assemble Context with Metadata (Priority: P1)

A developer needs to assemble retrieved chunks into a structured context format suitable for downstream processing (e.g., passing to an LLM for answer generation).

**Why this priority**: Context assembly is critical for making retrieval results usable.

**Independent Test**: Can test by running retrieval and verifying output includes formatted context with all metadata.

**Acceptance Scenarios**:

1. **Given** retrieval returns multiple chunks, **When** context assembly is requested, **Then** the output includes chunk text, source URL, chapter, section, and score for each result.

2. **Given** retrieval results exist, **When** context is assembled, **Then** chunks are ordered by relevance score (highest first).

3. **Given** context is assembled, **When** output is formatted, **Then** each chunk is clearly separated with its metadata visible.

---

### User Story 4 - Handle Empty Results (Priority: P2)

A developer queries for content that doesn't exist in the knowledge base and receives appropriate feedback.

**Why this priority**: Graceful handling of no-result scenarios prevents confusion and enables fallback logic.

**Independent Test**: Can test by querying for content not in the textbook (e.g., "quantum computing recipes").

**Acceptance Scenarios**:

1. **Given** a query has no semantically similar content in the knowledge base, **When** the query is executed, **Then** the system returns an empty result set with a clear indication.

2. **Given** the collection is empty or doesn't exist, **When** a query is attempted, **Then** the system returns an appropriate error message.

3. **Given** results have very low similarity scores, **When** retrieval completes, **Then** all results are still returned (filtering by score is caller's responsibility).

---

### Edge Cases

- What happens when the Qdrant collection is empty? → Return empty results with clear message
- What happens when K exceeds the number of available chunks? → Return all available chunks
- What happens when Cohere API rate limit is hit? → Retry with exponential backoff
- What happens when Qdrant is unreachable? → Return connection error with clear message
- What happens with an empty query string? → Return validation error

## Requirements

### Functional Requirements

- **FR-001**: System MUST embed queries using Cohere embed-english-v3.0 with input_type="search_query"
- **FR-002**: System MUST retrieve chunks from Qdrant using cosine similarity search
- **FR-003**: System MUST return exactly K results (configurable from 3 to 8)
- **FR-004**: System MUST include metadata with each result: source_url, chapter, section, chunk_index, title
- **FR-005**: System MUST include similarity score with each result
- **FR-006**: System MUST return results ordered by similarity score (descending)
- **FR-007**: System MUST handle empty result sets gracefully
- **FR-008**: System MUST validate K is within range 3-8 and clamp if outside
- **FR-009**: System MUST assemble context in a structured format with text and metadata
- **FR-010**: System MUST provide CLI interface for running queries
- **FR-011**: System MUST retry on rate limit errors with exponential backoff
- **FR-012**: System MUST provide clear error messages for connection failures

### Key Entities

- **Query**: User's natural language question to search for
- **QueryEmbedding**: Vector representation of the query (1024 dimensions)
- **RetrievedChunk**: Text chunk with metadata and similarity score
- **RetrievalResult**: Collection of retrieved chunks with context assembly

## Success Criteria

### Measurable Outcomes

- **SC-001**: Queries return results in under 3 seconds (including embedding generation)
- **SC-002**: Results include complete metadata (source URL, chapter, section) for 100% of returned chunks
- **SC-003**: Similarity scores are included for all returned results
- **SC-004**: System correctly returns 0 results for queries with no matching content
- **SC-005**: K parameter correctly limits results (3-8 configurable range)
- **SC-006**: Developers can validate retrieval quality through CLI output

## Constraints

- **Vector DB**: Qdrant Cloud (existing collection from Part 1)
- **Embeddings**: Cohere embed-english-v3.0 (same as ingestion for consistency)
- **Similarity**: Cosine distance (configured in collection)
- **K Range**: 3-8 (configurable)
- **Collection**: "textbook_chunks" (created in Part 1)

## Out of Scope

- Answer generation or LLM response synthesis
- Agent logic or multi-turn conversations
- Web UI or REST API (FastAPI)
- OpenAI Agents SDK integration
- Selected-text-only answering or highlighting
- Re-ranking or hybrid search
- Metadata-based filtering (chapter/section filters)

## Dependencies

- **Part 1 Complete**: Requires the textbook knowledge base to be populated in Qdrant
- **Environment Variables**: COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY
- **Collection**: "textbook_chunks" must exist with indexed vectors

## Assumptions

- The Qdrant collection "textbook_chunks" exists and is populated from Part 1
- API credentials are available in environment variables
- Network connectivity to Cohere API and Qdrant Cloud is stable
- Query length is reasonable (not exceeding model token limits)
