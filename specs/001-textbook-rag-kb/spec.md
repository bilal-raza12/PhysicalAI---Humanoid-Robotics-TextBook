# Feature Specification: RAG Knowledge Base Construction for Textbook Website

**Feature Branch**: `001-textbook-rag-kb`
**Created**: 2025-12-24
**Status**: Draft
**Input**: User description: "RAG Knowledge Base Construction for Textbook Website - Ingesting deployed Docusaurus textbook content, generating embeddings, and storing them in a vector database"

## Clarifications

### Session 2025-12-24

- Q: Which Cohere embedding model should be used? → A: embed-english-v3.0 (1024 dimensions)

## Overview

This feature creates a knowledge base pipeline that ingests content from a deployed Docusaurus textbook website (hosted on GitHub Pages), processes the text into searchable chunks, generates vector embeddings, and stores them in a cloud vector database for similarity search.

**Target Audience**: Developers and reviewers of the RAG pipeline for an AI textbook

**Scope Boundaries**:
- **In Scope**: Content ingestion, text chunking, embedding generation, vector storage, basic verification
- **Out of Scope**: Chatbot/agent logic, retrieval/answering layer, UI integration, FastAPI backend, OpenAI Agents SDK

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ingest Textbook Content (Priority: P1)

As a pipeline operator, I want to crawl and ingest all public pages from the deployed Docusaurus textbook site so that the content is available for processing.

**Why this priority**: Content ingestion is the foundation - without ingested content, no embeddings or search can occur.

**Independent Test**: Can be fully tested by running the ingestion script against the live textbook URL and verifying that all expected pages are captured with clean text extracted.

**Acceptance Scenarios**:

1. **Given** a valid Docusaurus site URL, **When** the ingestion pipeline runs, **Then** all public textbook pages are discovered and their content is extracted
2. **Given** a page with navigation/footer/sidebar content, **When** content is extracted, **Then** only the main content body is captured (boilerplate removed)
3. **Given** a page with code blocks or special formatting, **When** content is extracted, **Then** the text is preserved in a readable format

---

### User Story 2 - Generate Searchable Chunks (Priority: P1)

As a pipeline operator, I want extracted content to be split into appropriately sized chunks so that embeddings capture focused semantic meaning.

**Why this priority**: Chunking quality directly impacts retrieval relevance - too large and context is diluted, too small and meaning is lost.

**Independent Test**: Can be tested by processing sample textbook content and verifying chunk sizes fall within 300-500 token range with proper metadata attached.

**Acceptance Scenarios**:

1. **Given** extracted page content, **When** chunking runs, **Then** each chunk contains 300-500 tokens
2. **Given** a chunk, **When** it is created, **Then** it includes metadata: source URL, chapter, section, and chunk index
3. **Given** content with natural section breaks, **When** chunking runs, **Then** chunks respect paragraph/section boundaries where possible

---

### User Story 3 - Create Vector Embeddings (Priority: P1)

As a pipeline operator, I want each text chunk to be converted into a vector embedding using Cohere so that semantic similarity search is possible.

**Why this priority**: Embeddings are the core representation that enables semantic search - without them, the knowledge base cannot function.

**Independent Test**: Can be tested by processing sample chunks through Cohere API and verifying output vectors have correct dimensions and non-zero values.

**Acceptance Scenarios**:

1. **Given** a text chunk, **When** embedding generation runs, **Then** a vector embedding is produced with correct dimensions for the Cohere model used
2. **Given** multiple chunks, **When** embedding generation runs, **Then** all chunks are processed with rate limiting to respect API constraints
3. **Given** an API failure during embedding, **When** retry logic activates, **Then** failed chunks are retried with exponential backoff

---

### User Story 4 - Store and Search Vectors (Priority: P1)

As a pipeline operator, I want embeddings stored in Qdrant Cloud so that I can perform similarity searches and retrieve relevant textbook content.

**Why this priority**: Vector storage is the destination for all pipeline output - it enables the retrieval that downstream systems will use.

**Independent Test**: Can be tested by inserting sample vectors into Qdrant collection and running similarity queries to verify correct results are returned.

**Acceptance Scenarios**:

1. **Given** generated embeddings with metadata, **When** storage runs, **Then** all vectors are persisted in Qdrant with correct dimensions
2. **Given** stored vectors, **When** a similarity search query is executed, **Then** relevant chunks are returned ranked by similarity
3. **Given** a returned chunk, **When** examining results, **Then** the source URL and location metadata are included for traceability

---

### User Story 5 - Verify Pipeline Integrity (Priority: P2)

As a pipeline operator, I want to verify that the ingestion pipeline completed successfully so that I can confirm the knowledge base is ready for use.

**Why this priority**: Verification provides confidence that the pipeline ran correctly before downstream systems depend on it.

**Independent Test**: Can be tested by running verification checks after pipeline completion and confirming all quality metrics pass.

**Acceptance Scenarios**:

1. **Given** a completed pipeline run, **When** verification runs, **Then** total chunk count and vector count are reported
2. **Given** stored vectors, **When** sample similarity searches are executed, **Then** results return relevant content (not random or empty)
3. **Given** verification results, **When** examining output, **Then** any errors or skipped pages are clearly reported

---

### Edge Cases

- What happens when a page returns a 404 or is temporarily unavailable?
  - The system logs the error, skips the page, and continues processing remaining pages
- What happens when Cohere API rate limits are hit?
  - The system implements exponential backoff and retries, pausing processing as needed
- What happens when Qdrant Cloud connection fails mid-upload?
  - The system retries with backoff; failed vectors are logged for manual review
- What happens when a page contains no meaningful text content?
  - The system skips the page and logs it as empty
- What happens when a chunk falls below minimum token threshold?
  - The system merges it with adjacent content or includes it if at document boundary

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST crawl and discover all public pages from the specified Docusaurus site
- **FR-002**: System MUST extract main content from pages, removing navigation, footer, and sidebar boilerplate
- **FR-003**: System MUST handle pages containing code blocks, tables, and special formatting
- **FR-004**: System MUST split extracted text into chunks of 300-500 tokens each
- **FR-005**: System MUST preserve paragraph and section boundaries when chunking where feasible
- **FR-006**: System MUST attach metadata to each chunk: source URL, chapter, section, chunk index
- **FR-007**: System MUST generate vector embeddings using Cohere embed-english-v3.0 model (1024 dimensions)
- **FR-008**: System MUST handle Cohere API rate limiting with appropriate backoff strategy
- **FR-009**: System MUST store all vectors in Qdrant Cloud with 1024 dimensions
- **FR-010**: System MUST enable similarity search queries against stored vectors
- **FR-011**: System MUST return chunk metadata (source URL, location) with search results
- **FR-012**: System MUST log errors and skipped pages without halting the entire pipeline
- **FR-013**: System MUST provide verification reporting showing ingestion statistics

### Key Entities

- **Page**: A single textbook page with URL, raw HTML content, and extracted clean text
- **Chunk**: A segment of text (300-500 tokens) with metadata including source URL, chapter, section, and chunk index
- **Embedding**: A vector representation of a chunk, with associated chunk ID and metadata
- **Collection**: The Qdrant collection containing all vectors with their payloads (metadata)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of accessible public textbook pages are ingested (excluding error pages)
- **SC-002**: All chunks fall within the 300-500 token size range (with tolerance for boundary chunks)
- **SC-003**: Every stored vector has complete metadata: source URL, chapter, section, chunk index
- **SC-004**: Similarity search for a known textbook topic returns relevant content in top 5 results
- **SC-005**: Pipeline completion produces a verification report with counts and any error summary
- **SC-006**: Retrieved chunks can be traced back to their source URL for user verification
- **SC-007**: System handles temporary network failures without complete pipeline failure

## Constraints

- **Source**: GitHub Pages-deployed Docusaurus site (specific URL to be configured)
- **Embeddings Provider**: Cohere API using embed-english-v3.0 model (1024 dimensions)
- **Vector Database**: Qdrant Cloud Free Tier (capacity limits apply)
- **Metadata Requirements**: Each vector must include source URL, chapter, section, chunk index

## Assumptions

- The Docusaurus site is publicly accessible without authentication
- The site has a discoverable sitemap or predictable URL structure for crawling
- Cohere API credentials will be provided via environment variables
- Qdrant Cloud credentials will be provided via environment variables
- The textbook content is in English
- Network connectivity is generally stable (transient failures handled via retry logic)
- Token counting uses standard tokenization compatible with Cohere's model

## Dependencies

- Access to the deployed Docusaurus textbook URL
- Valid Cohere API key with sufficient quota
- Qdrant Cloud account with collection provisioned
- Network access to both Cohere API and Qdrant Cloud endpoints
