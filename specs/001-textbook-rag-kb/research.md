# Research: RAG Knowledge Base Construction

**Feature**: 001-textbook-rag-kb | **Date**: 2025-12-24 | **Branch**: `001-textbook-rag-kb`

## Research Summary

This document consolidates research findings for the key technical decisions in the RAG knowledge base pipeline.

---

## Decision 1: URL Discovery Strategy

### Decision
Use **sitemap.xml parsing** as primary strategy with **manual URL list fallback**.

### Rationale
- Docusaurus generates sitemap.xml when `@docusaurus/plugin-sitemap` is configured (commonly included in preset-classic)
- Sitemap provides authoritative list of all public pages with metadata
- Target site: `https://bilal-raza12.github.io/PhysicalAI---Humanoid-Robotics-TextBook/sitemap.xml`
- Manual fallback using known doc paths from local `sidebars.ts` if sitemap unavailable

### Alternatives Considered
1. **Web crawler (recursive link following)** - Rejected: Over-engineered, may miss pages, harder to control scope
2. **Parse sidebars.ts directly** - Backup option: Works offline but may miss pages not in sidebar
3. **GitHub API to list docs/ files** - Rejected: Adds API dependency, doesn't reflect published structure

### Implementation Notes
```python
# Primary: Fetch sitemap.xml
sitemap_url = f"{base_url}/sitemap.xml"
response = httpx.get(sitemap_url)
# Parse XML, extract <loc> elements

# Fallback: Use predefined URL list from sidebars analysis
KNOWN_DOC_PATHS = ["docs/intro", "docs/prerequisites", ...]
```

---

## Decision 2: Chunk Size and Overlap

### Decision
- **Chunk size**: 300-500 tokens (target: 400 tokens)
- **Overlap**: 50 tokens (~10-15%)
- **Boundary respect**: Split on paragraph breaks when possible

### Rationale
- 300-500 tokens balances semantic coherence with retrieval precision
- Overlap prevents losing context at chunk boundaries
- Paragraph-aware splitting preserves natural text structure
- Cohere embed-english-v3.0 handles up to 512 tokens optimally

### Alternatives Considered
1. **Fixed character count** - Rejected: Token count more relevant for embedding quality
2. **Sentence-level splitting** - Rejected: Too granular, loses context
3. **No overlap** - Rejected: Risk of losing context at boundaries
4. **Larger chunks (1000+ tokens)** - Rejected: Dilutes semantic focus for retrieval

### Implementation Notes
```python
import tiktoken

# Use cl100k_base tokenizer (compatible with most modern models)
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

# Split strategy:
# 1. Split by paragraph (\n\n)
# 2. Merge paragraphs until ~400 tokens
# 3. Apply 50-token overlap between chunks
```

---

## Decision 3: Cohere Embedding Model

### Decision
Use **Cohere embed-english-v3.0** with 1024-dimensional output.

### Rationale
- 1024 dimensions provides good balance of expressiveness and storage efficiency
- embed-english-v3.0 is current production model (as of late 2024)
- Supports `search_document` and `search_query` input types for asymmetric search
- Free tier: 10M tokens/month, 100 requests/min

### Alternatives Considered
1. **OpenAI text-embedding-3-small** - Rejected: User specified Cohere
2. **Cohere embed-multilingual-v3.0** - Rejected: Textbook is English-only
3. **embed-english-light-v3.0** (384 dims)** - Rejected: Lower quality for technical content

### Implementation Notes
```python
import cohere

co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

# For indexing documents
embeddings = co.embed(
    texts=chunks,
    model="embed-english-v3.0",
    input_type="search_document",
    embedding_types=["float"]
).embeddings.float

# For search queries (at retrieval time)
query_embedding = co.embed(
    texts=[query],
    model="embed-english-v3.0",
    input_type="search_query",
    embedding_types=["float"]
).embeddings.float[0]
```

### Rate Limiting
- Implement exponential backoff starting at 1 second
- Batch up to 96 texts per request (API limit)
- Track API quota usage

---

## Decision 4: Qdrant Collection Schema

### Decision
- **Collection name**: `textbook_chunks`
- **Vector size**: 1024 (Cohere embed-english-v3.0 output)
- **Distance metric**: Cosine similarity

### Rationale
- Cosine similarity standard for normalized embeddings
- Qdrant Cloud free tier: 1GB storage, sufficient for ~40 pages
- Payload includes all required metadata for citation tracing

### Alternatives Considered
1. **Dot product** - Rejected: Requires normalized vectors, cosine handles this automatically
2. **Euclidean distance** - Rejected: Less intuitive for semantic similarity

### Implementation Notes
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)

# Create collection
client.create_collection(
    collection_name="textbook_chunks",
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE
    )
)

# Upsert with payload
points = [
    PointStruct(
        id=idx,
        vector=embedding,
        payload={
            "source_url": url,
            "chapter": chapter,
            "section": section,
            "chunk_index": chunk_idx,
            "text": chunk_text  # For debugging/display
        }
    )
    for idx, (embedding, metadata) in enumerate(zip(embeddings, metadata_list))
]

client.upsert(collection_name="textbook_chunks", points=points)
```

### Free Tier Limits
- 1GB storage
- Unlimited requests
- Single cluster (no replicas)

---

## Decision 5: Token Counting Library

### Decision
Use **tiktoken** with `cl100k_base` encoding.

### Rationale
- Industry standard tokenizer compatible with modern embedding models
- Fast and lightweight
- `cl100k_base` is GPT-4/Claude-compatible encoding
- Close enough to Cohere tokenization for chunking purposes

### Alternatives Considered
1. **Cohere tokenize API** - Rejected: Adds API call overhead for each chunk
2. **transformers tokenizer** - Rejected: Heavy dependency for simple counting
3. **Character-based estimation** - Rejected: Inaccurate for technical content

---

## Decision 6: HTML Parsing Strategy

### Decision
Use **BeautifulSoup4** with selective extraction of main content.

### Rationale
- BeautifulSoup is mature, well-documented, handles malformed HTML
- Docusaurus uses consistent structure: main content in `<article>` or specific CSS class
- Need to strip navigation, sidebar, footer

### Implementation Notes
```python
from bs4 import BeautifulSoup

def extract_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove navigation, sidebar, footer
    for element in soup.select("nav, aside, footer, .navbar, .sidebar"):
        element.decompose()

    # Extract main content
    article = soup.select_one("article, .docMainContainer, main")
    if article:
        return article.get_text(separator="\n", strip=True)

    return soup.get_text(separator="\n", strip=True)
```

---

## Open Items

All clarifications resolved. No blocking unknowns remain.

## References

- [Cohere Embed API](https://docs.cohere.com/reference/embed)
- [Qdrant Python Client](https://qdrant.tech/documentation/python-client/)
- [Docusaurus Sitemap Plugin](https://docusaurus.io/docs/api/plugins/@docusaurus/plugin-sitemap)
- [tiktoken](https://github.com/openai/tiktoken)
