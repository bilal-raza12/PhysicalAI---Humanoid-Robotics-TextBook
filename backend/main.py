"""
RAG Knowledge Base Pipeline for Textbook Website

This script provides a complete pipeline for:
1. Ingesting content from a Docusaurus textbook site
2. Chunking text into 300-500 token segments
3. Generating Cohere embeddings
4. Storing vectors in Qdrant Cloud
5. Verifying the knowledge base with sample queries
6. FastAPI backend for frontend integration (Part 4)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
import asyncio

import httpx

# OpenAI Agents SDK imports (Part 3)
from agents import Agent, Runner, function_tool, ModelSettings
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# FastAPI imports (Part 4)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# =============================================================================
# FastAPI Application Setup (Part 4)
# =============================================================================

app = FastAPI(
    title="Textbook RAG API",
    description="Backend API for the Physical AI & Humanoid Robotics textbook RAG chatbot",
    version="1.0.0",
)

# CORS middleware configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and register routes (deferred to avoid circular imports)
# Routes are registered after all functions are defined

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# T006: PageStatus enum and Page dataclass
# =============================================================================


class PageStatus(Enum):
    """Status of a page during processing."""

    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class Page:
    """Represents a single textbook page fetched from the Docusaurus site."""

    url: str
    raw_html: str = ""
    clean_text: str = ""
    title: str = ""
    chapter: str = ""
    section: str = ""
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "pending"
    error_message: Optional[str] = None


# =============================================================================
# T007: Chunk dataclass
# =============================================================================


@dataclass
class Chunk:
    """Represents a text segment derived from a Page, sized for embedding."""

    chunk_id: str
    source_url: str
    text: str
    token_count: int
    chapter: str = ""
    section: str = ""
    chunk_index: int = 0
    title: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# =============================================================================
# T008: EmbeddingResult dataclass
# =============================================================================


@dataclass
class EmbeddingResult:
    """Represents the vector embedding of a Chunk."""

    chunk_id: str
    vector: list[float] = field(default_factory=list)
    model: str = "embed-english-v3.0"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# =============================================================================
# T012: JSON serialization helpers
# =============================================================================


def page_to_dict(page: Page) -> dict:
    """Convert Page dataclass to dictionary for JSON serialization."""
    return asdict(page)


def dict_to_page(d: dict) -> Page:
    """Convert dictionary to Page dataclass."""
    return Page(**d)


def chunk_to_dict(chunk: Chunk) -> dict:
    """Convert Chunk dataclass to dictionary for JSON serialization."""
    return asdict(chunk)


def dict_to_chunk(d: dict) -> Chunk:
    """Convert dictionary to Chunk dataclass."""
    return Chunk(**d)


def embedding_to_dict(embedding: EmbeddingResult) -> dict:
    """Convert EmbeddingResult dataclass to dictionary for JSON serialization."""
    return asdict(embedding)


def dict_to_embedding(d: dict) -> EmbeddingResult:
    """Convert dictionary to EmbeddingResult dataclass."""
    return EmbeddingResult(**d)


def save_pages(pages: list[Page], filepath: str) -> None:
    """Save list of pages to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([page_to_dict(p) for p in pages], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(pages)} pages to {filepath}")


def load_pages(filepath: str) -> list[Page]:
    """Load list of pages from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    pages = [dict_to_page(d) for d in data]
    logger.info(f"Loaded {len(pages)} pages from {filepath}")
    return pages


def save_chunks(chunks: list[Chunk], filepath: str) -> None:
    """Save list of chunks to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([chunk_to_dict(c) for c in chunks], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(chunks)} chunks to {filepath}")


def load_chunks(filepath: str) -> list[Chunk]:
    """Load list of chunks from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = [dict_to_chunk(d) for d in data]
    logger.info(f"Loaded {len(chunks)} chunks from {filepath}")
    return chunks


def save_embeddings(embeddings: list[EmbeddingResult], filepath: str) -> None:
    """Save list of embeddings to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            [embedding_to_dict(e) for e in embeddings], f, indent=2, ensure_ascii=False
        )
    logger.info(f"Saved {len(embeddings)} embeddings to {filepath}")


def load_embeddings(filepath: str) -> list[EmbeddingResult]:
    """Load list of embeddings from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    embeddings = [dict_to_embedding(d) for d in data]
    logger.info(f"Loaded {len(embeddings)} embeddings from {filepath}")
    return embeddings


# =============================================================================
# Constants
# =============================================================================

DEFAULT_BASE_URL = (
    "https://bilal-raza12.github.io/PhysicalAI---Humanoid-Robotics-TextBook"
)
DEFAULT_COLLECTION = "textbook_chunks"
EMBEDDING_MODEL = "embed-english-v3.0"
EMBEDDING_DIMENSIONS = 1024
MIN_TOKENS = 300
MAX_TOKENS = 500
OVERLAP_TOKENS = 50
BATCH_SIZE = 96


# =============================================================================
# T013-T022: Ingestion functions (US1)
# =============================================================================


def fetch_sitemap(base_url: str) -> list[str]:
    """Fetch and parse sitemap.xml to get all page URLs."""
    sitemap_url = f"{base_url}/sitemap.xml"
    logger.info(f"Fetching sitemap from {sitemap_url}")

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(sitemap_url)
            response.raise_for_status()

        root = ET.fromstring(response.content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text for loc in root.findall(".//ns:loc", namespace) if loc.text]

        logger.info(f"Found {len(urls)} URLs in sitemap")
        return urls

    except Exception as e:
        logger.warning(f"Failed to fetch sitemap: {e}")
        return []


def get_fallback_urls(base_url: str) -> list[str]:
    """Return fallback URL list when sitemap is unavailable."""
    doc_paths = [
        "docs/intro",
        "docs/prerequisites",
        "docs/conventions",
        "docs/module-1-ros2",
        "docs/module-1-ros2/ch01-intro-ros2",
        "docs/module-1-ros2/ch02-nodes-topics",
        "docs/module-1-ros2/ch03-urdf-kinematics",
        "docs/module-1-ros2/ch04-python-agents",
        "docs/module-2-digital-twin",
        "docs/module-2-digital-twin/ch05-physics-sim",
        "docs/module-2-digital-twin/ch06-gazebo-twin",
        "docs/module-2-digital-twin/ch07-sensor-sim",
        "docs/module-2-digital-twin/ch08-unity-hri",
        "docs/module-3-nvidia-isaac",
        "docs/module-3-nvidia-isaac/ch09-isaac-intro",
        "docs/module-3-nvidia-isaac/ch10-synthetic-data",
        "docs/module-3-nvidia-isaac/ch11-reinforcement-learning",
        "docs/module-3-nvidia-isaac/ch12-sim2real",
        "docs/module-4-vla",
        "docs/module-4-vla/ch13-vla-intro",
        "docs/module-4-vla/ch14-llm-planning",
        "docs/module-4-vla/ch15-multimodal-perception",
        "docs/module-4-vla/ch16-embodied-agents",
        "docs/capstone",
        "docs/capstone/ch17-capstone-project",
        "docs/category/appendices",
    ]
    urls = [f"{base_url}/{path}" for path in doc_paths]
    logger.info(f"Using {len(urls)} fallback URLs")
    return urls


def fetch_page_content(url: str, max_retries: int = 3) -> tuple[str, Optional[str]]:
    """Fetch HTML content from URL with retry logic."""
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.text, None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return "", f"Page not found (404)"
            error_msg = f"HTTP error {e.response.status_code}"
        except httpx.TimeoutException:
            error_msg = "Request timed out"
        except Exception as e:
            error_msg = str(e)

        if attempt < max_retries - 1:
            wait_time = 2**attempt
            logger.warning(
                f"Retry {attempt + 1}/{max_retries} for {url} after {wait_time}s: {error_msg}"
            )
            time.sleep(wait_time)

    return "", error_msg


def extract_main_content(html: str) -> str:
    """Extract main content from HTML, removing nav/sidebar/footer."""
    soup = BeautifulSoup(html, "lxml")

    # Remove navigation, sidebar, footer elements
    for selector in [
        "nav",
        "aside",
        "footer",
        ".navbar",
        ".sidebar",
        ".menu",
        ".toc",
        ".breadcrumb",
        "script",
        "style",
    ]:
        for element in soup.select(selector):
            element.decompose()

    # Try to find main content container
    main_content = soup.select_one("article, .docMainContainer, main, .markdown")
    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n\n".join(lines)


def extract_title(html: str) -> str:
    """Extract page title from HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Try <title> tag first
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        title = title_tag.string.strip()
        # Remove common suffixes
        for suffix in [" | Physical AI & Humanoid Robotics", " - Physical AI"]:
            if title.endswith(suffix):
                title = title[: -len(suffix)]
        return title

    # Try <h1> tag
    h1_tag = soup.find("h1")
    if h1_tag:
        return h1_tag.get_text(strip=True)

    return ""


def extract_chapter_section(url: str, html: str) -> tuple[str, str]:
    """Extract chapter and section from URL path and headings."""
    from urllib.parse import urlparse

    path = urlparse(url).path

    # Extract chapter from path
    chapter = ""
    parts = path.strip("/").split("/")
    for part in parts:
        if part.startswith("ch") or part.startswith("module-"):
            chapter = part
            break

    # Extract section from first <h2> or use path
    section = ""
    soup = BeautifulSoup(html, "lxml")
    h2_tag = soup.find("h2")
    if h2_tag:
        section = h2_tag.get_text(strip=True)
    elif parts:
        section = parts[-1].replace("-", " ").title()

    return chapter, section


def ingest_pages(base_url: str, output_file: str) -> int:
    """Fetch all pages from the textbook site and save to JSON."""
    logger.info(f"Starting ingestion from {base_url}")

    # Get URLs from sitemap or fallback
    urls = fetch_sitemap(base_url)
    if not urls:
        urls = get_fallback_urls(base_url)

    # Filter to only doc pages
    urls = [url for url in urls if "/docs/" in url or url.endswith("/docs")]

    pages: list[Page] = []
    success_count = 0
    error_count = 0
    skipped_count = 0

    for i, url in enumerate(urls):
        logger.info(f"Processing [{i + 1}/{len(urls)}]: {url}")

        html, error = fetch_page_content(url)

        if error:
            pages.append(
                Page(
                    url=url,
                    status="error",
                    error_message=error,
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            error_count += 1
            continue

        clean_text = extract_main_content(html)
        if not clean_text or len(clean_text) < 100:
            pages.append(
                Page(
                    url=url,
                    status="skipped",
                    error_message="No meaningful content",
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            skipped_count += 1
            continue

        title = extract_title(html)
        chapter, section = extract_chapter_section(url, html)

        pages.append(
            Page(
                url=url,
                raw_html=html,
                clean_text=clean_text,
                title=title,
                chapter=chapter,
                section=section,
                status="success",
                fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        success_count += 1

    save_pages(pages, output_file)

    print(f"\n{'=' * 50}")
    print(f"Ingestion Summary")
    print(f"{'=' * 50}")
    print(f"Total URLs processed: {len(urls)}")
    print(f"Successfully fetched:  {success_count}")
    print(f"Errors:               {error_count}")
    print(f"Skipped (empty):      {skipped_count}")
    print(f"Output file:          {output_file}")

    if error_count > 0:
        return 1  # Partial failure
    return 0


# =============================================================================
# T023-T030: Chunking functions (US2)
# =============================================================================

# Initialize tokenizer
_encoding: Optional[tiktoken.Encoding] = None


def get_encoding() -> tiktoken.Encoding:
    """Get tiktoken encoding (cached)."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(get_encoding().encode(text))


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


def create_chunks_from_page(
    page: Page, min_tokens: int, max_tokens: int, overlap_tokens: int
) -> list[Chunk]:
    """Create chunks from a page with target token size and overlap."""
    if page.status != "success" or not page.clean_text:
        return []

    paragraphs = split_into_paragraphs(page.clean_text)
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_text = ""
    current_tokens = 0
    chunk_index = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # If single paragraph exceeds max, we need to split it
        if para_tokens > max_tokens:
            # Flush current chunk if any
            if current_text:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        source_url=page.url,
                        text=current_text,
                        token_count=current_tokens,
                        chapter=page.chapter,
                        section=page.section,
                        chunk_index=chunk_index,
                        title=page.title,
                    )
                )
                chunk_index += 1
                current_text = ""
                current_tokens = 0

            # Split large paragraph by sentences or words
            words = para.split()
            temp_text = ""
            temp_tokens = 0
            for word in words:
                word_tokens = count_tokens(word + " ")
                if temp_tokens + word_tokens > max_tokens:
                    if temp_text:
                        chunks.append(
                            Chunk(
                                chunk_id=str(uuid.uuid4()),
                                source_url=page.url,
                                text=temp_text.strip(),
                                token_count=count_tokens(temp_text.strip()),
                                chapter=page.chapter,
                                section=page.section,
                                chunk_index=chunk_index,
                                title=page.title,
                            )
                        )
                        chunk_index += 1
                    temp_text = word + " "
                    temp_tokens = word_tokens
                else:
                    temp_text += word + " "
                    temp_tokens += word_tokens
            if temp_text:
                current_text = temp_text.strip()
                current_tokens = count_tokens(current_text)
            continue

        # Check if adding this paragraph would exceed max
        combined_tokens = (
            current_tokens + para_tokens + (1 if current_text else 0)
        )  # +1 for separator

        if combined_tokens > max_tokens and current_text:
            # Save current chunk
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    source_url=page.url,
                    text=current_text,
                    token_count=current_tokens,
                    chapter=page.chapter,
                    section=page.section,
                    chunk_index=chunk_index,
                    title=page.title,
                )
            )
            chunk_index += 1

            # Start new chunk with overlap
            if overlap_tokens > 0 and current_text:
                overlap_text = get_overlap_text(current_text, overlap_tokens)
                current_text = overlap_text + "\n\n" + para if overlap_text else para
            else:
                current_text = para
            current_tokens = count_tokens(current_text)
        else:
            # Add to current chunk
            if current_text:
                current_text += "\n\n" + para
            else:
                current_text = para
            current_tokens = count_tokens(current_text)

    # Don't forget the last chunk
    if current_text:
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                source_url=page.url,
                text=current_text,
                token_count=current_tokens,
                chapter=page.chapter,
                section=page.section,
                chunk_index=chunk_index,
                title=page.title,
            )
        )

    return chunks


def get_overlap_text(text: str, target_tokens: int) -> str:
    """Get approximately target_tokens worth of text from the end."""
    words = text.split()
    overlap_text = ""
    tokens = 0

    for word in reversed(words):
        word_tokens = count_tokens(word + " ")
        if tokens + word_tokens > target_tokens:
            break
        overlap_text = word + " " + overlap_text
        tokens += word_tokens

    return overlap_text.strip()


def chunk_pages(
    input_file: str,
    output_file: str,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> int:
    """Chunk all pages and save to JSON."""
    pages = load_pages(input_file)
    success_pages = [p for p in pages if p.status == "success"]

    all_chunks: list[Chunk] = []
    for page in success_pages:
        page_chunks = create_chunks_from_page(page, min_tokens, max_tokens, overlap)
        all_chunks.extend(page_chunks)

    save_chunks(all_chunks, output_file)

    # Calculate statistics
    token_counts = [c.token_count for c in all_chunks]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    min_tok = min(token_counts) if token_counts else 0
    max_tok = max(token_counts) if token_counts else 0

    print(f"\n{'=' * 50}")
    print(f"Chunking Summary")
    print(f"{'=' * 50}")
    print(f"Pages processed:     {len(success_pages)}")
    print(f"Total chunks:        {len(all_chunks)}")
    print(f"Average tokens:      {avg_tokens:.1f}")
    print(f"Token range:         {min_tok} - {max_tok}")
    print(f"Output file:         {output_file}")

    if not all_chunks:
        return 2  # No chunks generated
    return 0


# =============================================================================
# T031-T038: Embedding functions (US3)
# =============================================================================


def get_cohere_client():
    """Initialize Cohere client with API key from environment."""
    import cohere

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not set")
    return cohere.Client(api_key=api_key)


def generate_embeddings_batch(
    client, texts: list[str], input_type: str = "search_document", max_retries: int = 5
) -> list[list[float]]:
    """Generate embeddings for a batch of texts with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.embed(
                texts=texts,
                model=EMBEDDING_MODEL,
                input_type=input_type,
                embedding_types=["float"],
            )
            embeddings = response.embeddings.float_

            # Validate dimensions
            for emb in embeddings:
                if len(emb) != EMBEDDING_DIMENSIONS:
                    raise ValueError(
                        f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(emb)}"
                    )

            return embeddings

        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit (429) - need longer wait
            if "429" in str(e) or "rate limit" in error_str or "too many" in error_str:
                wait_time = 60  # Wait 60 seconds for rate limit
                logger.warning(
                    f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}"
                )
            else:
                wait_time = 2 ** (attempt + 1)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}"
                )

            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                raise


def embed_chunks(
    input_file: str, output_file: str, batch_size: int = BATCH_SIZE
) -> int:
    """Generate embeddings for all chunks and save to JSON."""
    chunks = load_chunks(input_file)

    try:
        client = get_cohere_client()
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    embeddings: list[EmbeddingResult] = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    api_calls = 0
    retries = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(
            f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)"
        )

        texts = [c.text for c in batch]

        try:
            vectors = generate_embeddings_batch(client, texts)
            api_calls += 1

            for chunk, vector in zip(batch, vectors):
                embeddings.append(
                    EmbeddingResult(chunk_id=chunk.chunk_id, vector=vector)
                )

        except Exception as e:
            logger.error(f"Failed to embed batch {batch_num}: {e}")
            return 2

        # Rate limiting - delay between batches to avoid hitting token limits
        # Trial API has 100k tokens/min limit, each batch is ~50k tokens
        if i + batch_size < len(chunks):
            logger.info("Waiting 35s between batches to respect rate limits...")
            time.sleep(35)

    save_embeddings(embeddings, output_file)

    print(f"\n{'=' * 50}")
    print(f"Embedding Summary")
    print(f"{'=' * 50}")
    print(f"Chunks embedded:     {len(embeddings)}")
    print(f"API calls made:      {api_calls}")
    print(f"Embedding model:     {EMBEDDING_MODEL}")
    print(f"Dimensions:          {EMBEDDING_DIMENSIONS}")
    print(f"Output file:         {output_file}")

    return 0


# =============================================================================
# T039-T046: Storage functions (US4)
# =============================================================================


def get_qdrant_client():
    """Initialize Qdrant client with credentials from environment."""
    from qdrant_client import QdrantClient

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY environment variables must be set"
        )

    return QdrantClient(url=url, api_key=api_key, timeout=120)


def create_collection(client, collection_name: str, recreate: bool = False) -> None:
    """Create Qdrant collection with proper configuration."""
    from qdrant_client.models import Distance, VectorParams

    if recreate:
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSIONS, distance=Distance.COSINE
            ),
        )
        logger.info(f"Created collection: {collection_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info(f"Collection {collection_name} already exists")
        else:
            raise


def store_vectors(
    chunks_file: str, embeddings_file: str, collection_name: str, recreate: bool = False
) -> int:
    """Store embeddings and metadata in Qdrant."""
    from qdrant_client.models import PointStruct

    chunks = load_chunks(chunks_file)
    embeddings = load_embeddings(embeddings_file)

    # Create chunk lookup
    chunk_map = {c.chunk_id: c for c in chunks}

    try:
        client = get_qdrant_client()
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Create collection
    try:
        create_collection(client, collection_name, recreate)
    except Exception as e:
        print(f"Error creating collection: {e}")
        return 1

    # Prepare points
    points = []
    for i, emb in enumerate(embeddings):
        chunk = chunk_map.get(emb.chunk_id)
        if not chunk:
            logger.warning(f"Chunk not found for embedding: {emb.chunk_id}")
            continue

        point = PointStruct(
            id=i,
            vector=emb.vector,
            payload={
                "source_url": chunk.source_url,
                "chapter": chunk.chapter,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "title": chunk.title,
                "chunk_id": chunk.chunk_id,
            },
        )
        points.append(point)

    # Upsert in batches (smaller batches for cloud reliability)
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        try:
            client.upsert(collection_name=collection_name, points=batch, wait=True)
            logger.info(f"Upserted batch {i // batch_size + 1}")
        except Exception as e:
            print(f"Error upserting batch: {e}")
            return 2

    # Wait a moment for indexing to complete
    time.sleep(1)

    # Get collection info and verify
    try:
        info = client.get_collection(collection_name)
        # Handle different Qdrant client versions
        vectors_count = getattr(info, "vectors_count", None) or getattr(
            info, "points_count", len(points)
        )
        points_count = getattr(info, "points_count", len(points))
    except Exception as e:
        logger.warning(f"Could not get collection info: {e}")
        vectors_count = len(points)
        points_count = len(points)

    print(f"\n{'=' * 50}")
    print(f"Storage Summary")
    print(f"{'=' * 50}")
    print(f"Points prepared:     {len(points)}")
    print(f"Collection:          {collection_name}")
    print(f"Vectors in collection: {vectors_count}")
    print(f"Points in collection:  {points_count}")

    # Verify storage was successful
    if vectors_count == 0 and len(points) > 0:
        print("\nWARNING: Vectors not yet visible. They may still be indexing.")
        print("Run 'verify' command in a few seconds to check.")
        return 1

    if vectors_count < len(points):
        print(f"\nWARNING: Only {vectors_count} of {len(points)} vectors visible.")
        return 1

    print("\nStorage: SUCCESS")
    return 0


# =============================================================================
# T047-T052: Verification functions (US5)
# =============================================================================


def generate_query_embedding(client, query: str) -> list[float]:
    """Generate embedding for a search query."""
    response = client.embed(
        texts=[query],
        model=EMBEDDING_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    return response.embeddings.float_[0]


def search_vectors(collection_name: str, query: str, top_k: int = 5) -> list[dict]:
    """Search for similar vectors in Qdrant."""
    cohere_client = get_cohere_client()
    qdrant_client = get_qdrant_client()

    # Generate query embedding
    query_vector = generate_query_embedding(cohere_client, query)

    # Search using query_points (newer API) or search (older API)
    try:
        # Try newer API first (qdrant-client >= 1.7)
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
    except AttributeError:
        # Fall back to older API
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

    return [{"score": r.score, "payload": r.payload} for r in results]


def verify_pipeline(collection_name: str, query: str) -> int:
    """Verify the pipeline by running a sample query."""
    try:
        qdrant_client = get_qdrant_client()
    except ValueError as e:
        print(f"Error: {e}")
        return 2

    # Get collection stats
    try:
        info = qdrant_client.get_collection(collection_name)
        # Handle different Qdrant client versions
        vectors_count = getattr(info, "vectors_count", None) or getattr(
            info, "points_count", 0
        )
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return 2

    print(f"\n{'=' * 50}")
    print(f"Collection Statistics")
    print(f"{'=' * 50}")
    print(f"Collection:          {collection_name}")
    print(f"Points:              {vectors_count}")

    if vectors_count == 0:
        print("\nNo vectors in collection. Run 'store' command first.")
        return 1

    # Run search
    print(f"\n{'=' * 50}")
    print(f'Sample Query: "{query}"')
    print(f"{'=' * 50}")

    try:
        results = search_vectors(collection_name, query)
    except Exception as e:
        print(f"Error running search: {e}")
        return 2

    if not results:
        print("No results found.")
        return 1

    for i, r in enumerate(results, 1):
        payload = r["payload"]
        score = r["score"]
        title = payload.get("title", "Untitled")
        chapter = payload.get("chapter", "")
        text_preview = payload.get("text", "")[:100] + "..."

        print(f"\n{i}. [{score:.3f}] {title}")
        if chapter:
            print(f"   Chapter: {chapter}")
        print(f"   {text_preview}")

    print(f"\n{'=' * 50}")
    print("Verification: PASSED")
    print(f"{'=' * 50}")

    return 0


# =============================================================================
# Retrieval Pipeline (Part 2) - Search Command
# =============================================================================


@dataclass
class RetrievedChunk:
    """A chunk retrieved from Qdrant with its metadata and score."""

    text: str
    score: float
    source_url: str
    chunk_index: int
    chunk_id: str
    chapter: str = ""
    section: str = ""
    title: str = ""


@dataclass
class RetrievalResult:
    """Collection of retrieved chunks."""

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    collection: str = "textbook_chunks"

    @property
    def count(self) -> int:
        return len(self.chunks)


@dataclass
class AssembledContext:
    """Formatted context for downstream use."""

    formatted_text: str
    chunk_count: int
    total_chars: int
    sources: list[str] = field(default_factory=list)


def validate_query(query: str) -> tuple[str, Optional[str]]:
    """Validate query text. Returns (validated_query, error_message)."""
    if not query or not query.strip():
        return "", "Query cannot be empty"

    query = query.strip()
    if len(query) > 1000:
        logger.warning(f"Query truncated from {len(query)} to 1000 characters")
        query = query[:1000]

    return query, None


def validate_k(k: int) -> int:
    """Validate and clamp K parameter to valid range 3-8."""
    if k < 3:
        logger.warning(f"K={k} is below minimum, clamping to 3")
        return 3
    if k > 8:
        logger.warning(f"K={k} is above maximum, clamping to 8")
        return 8
    return k


def search_knowledge_base(
    query: str, k: int = 5, collection_name: str = DEFAULT_COLLECTION
) -> tuple[Optional[RetrievalResult], Optional[str]]:
    """Search the knowledge base and return structured results."""
    # Validate inputs
    query, error = validate_query(query)
    if error:
        return None, error

    k = validate_k(k)

    # Get clients
    try:
        cohere_client = get_cohere_client()
    except ValueError as e:
        return None, f"Cohere API error: {e}"

    try:
        qdrant_client = get_qdrant_client()
    except ValueError as e:
        return None, f"Qdrant connection error: {e}"

    # Check collection exists
    try:
        info = qdrant_client.get_collection(collection_name)
        points_count = getattr(info, "points_count", 0)
        if points_count == 0:
            return RetrievalResult(query=query, collection=collection_name), None
    except Exception as e:
        return None, f"Collection '{collection_name}' not found: {e}"

    # Generate query embedding
    try:
        query_vector = generate_query_embedding(cohere_client, query)
    except Exception as e:
        return None, f"Embedding generation failed: {e}"

    # Search Qdrant
    try:
        # Try newer API first (qdrant-client >= 1.7)
        try:
            results = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=k,
                with_payload=True,
            ).points
        except AttributeError:
            # Fall back to older API
            results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
            )
    except Exception as e:
        return None, f"Search failed: {e}"

    # Convert to RetrievedChunk objects
    chunks = []
    for r in results:
        payload = r.payload
        chunks.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                score=r.score,
                source_url=payload.get("source_url", ""),
                chunk_index=payload.get("chunk_index", 0),
                chunk_id=payload.get("chunk_id", ""),
                chapter=payload.get("chapter", ""),
                section=payload.get("section", ""),
                title=payload.get("title", ""),
            )
        )

    return RetrievalResult(query=query, chunks=chunks, collection=collection_name), None


def assemble_context(result: RetrievalResult) -> AssembledContext:
    """Assemble retrieved chunks into formatted context."""
    if result.count == 0:
        return AssembledContext(
            formatted_text="No matching content found in the knowledge base.",
            chunk_count=0,
            total_chars=0,
            sources=[],
        )

    blocks = []
    sources = set()

    for i, chunk in enumerate(result.chunks, 1):
        header = f"[{i}] Score: {chunk.score:.3f}"
        header += f"\nSource: {chunk.source_url}"
        if chunk.chapter:
            header += f"\nChapter: {chunk.chapter}"
        if chunk.section:
            header += f" | Section: {chunk.section}"

        block = f"{header}\n---\n{chunk.text}"
        blocks.append(block)
        sources.add(chunk.source_url)

    formatted_text = "\n\n" + "-" * 50 + "\n\n".join(blocks)
    total_chars = sum(len(c.text) for c in result.chunks)

    return AssembledContext(
        formatted_text=formatted_text,
        chunk_count=result.count,
        total_chars=total_chars,
        sources=list(sources),
    )


def sanitize_text_for_console(text: str) -> str:
    """Remove or replace characters that can't be displayed in Windows console."""
    # Remove zero-width spaces and other problematic Unicode characters
    replacements = {
        "\u200b": "",  # zero-width space
        "\u200c": "",  # zero-width non-joiner
        "\u200d": "",  # zero-width joiner
        "\ufeff": "",  # byte order mark
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Encode to ASCII with replacement for any remaining non-ASCII chars
    return text.encode("ascii", "replace").decode("ascii")


def format_search_result_text(
    result: RetrievalResult, context: AssembledContext
) -> str:
    """Format search result as text output."""
    output = []
    output.append("=" * 50)
    output.append("Search Results")
    output.append("=" * 50)
    output.append(f'Query: "{result.query}"')
    output.append(f"Collection: {result.collection}")
    output.append(f"Results: {result.count}")
    output.append("")
    output.append("=" * 50)

    if result.count == 0:
        output.append("")
        output.append("No matching content found in the knowledge base.")
        output.append("=" * 50)
        return "\n".join(output)

    for i, chunk in enumerate(result.chunks, 1):
        output.append("")
        output.append(f"[{i}] Score: {chunk.score:.3f}")
        output.append(f"Source: {chunk.source_url}")
        if chunk.chapter:
            output.append(f"Chapter: {chunk.chapter}")
        if chunk.section:
            output.append(f"Section: {chunk.section}")
        output.append("---")
        # Show first 200 chars of text (sanitized for console)
        text_preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
        output.append(sanitize_text_for_console(text_preview))
        output.append("")
        output.append("-" * 50)

    output.append("")
    output.append("=" * 50)
    output.append(
        f"Context assembled: {context.chunk_count} chunks, {context.total_chars} characters"
    )
    output.append(f"Sources: {len(context.sources)} unique pages")
    output.append("=" * 50)

    return "\n".join(output)


def format_search_result_json(
    result: RetrievalResult, context: AssembledContext, error: Optional[str] = None
) -> str:
    """Format search result as JSON output."""
    if error:
        data = {
            "status": "error",
            "code": "SEARCH_ERROR",
            "message": error,
        }
    elif result.count == 0:
        data = {
            "status": "success",
            "query": result.query,
            "collection": result.collection,
            "count": 0,
            "chunks": [],
            "message": "No matching content found in the knowledge base.",
        }
    else:
        chunks_data = []
        for i, chunk in enumerate(result.chunks, 1):
            chunks_data.append(
                {
                    "rank": i,
                    "score": chunk.score,
                    "source_url": chunk.source_url,
                    "chapter": chunk.chapter,
                    "section": chunk.section,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                }
            )

        data = {
            "status": "success",
            "query": result.query,
            "collection": result.collection,
            "count": result.count,
            "chunks": chunks_data,
            "context": {
                "chunk_count": context.chunk_count,
                "total_chars": context.total_chars,
                "sources": context.sources,
            },
        }

    return json.dumps(data, indent=2, ensure_ascii=False)


def safe_print(text: str) -> None:
    """Print text with fallback encoding for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: encode with errors='replace' for Windows console
        print(text.encode("ascii", "replace").decode("ascii"))


def run_search(
    query: str,
    k: int = 5,
    output_format: str = "text",
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """Execute search command and return exit code."""
    result, error = search_knowledge_base(query, k, collection_name)

    if error:
        if output_format == "json":
            safe_print(format_search_result_json(None, None, error))
        else:
            safe_print(f"\n{'=' * 50}")
            safe_print("Error")
            safe_print(f"{'=' * 50}")
            safe_print(f"Message: {error}")
            safe_print(f"\nExit code: 2")
            safe_print(f"{'=' * 50}")
        return 2

    context = assemble_context(result)

    if output_format == "json":
        safe_print(format_search_result_json(result, context))
    else:
        safe_print(format_search_result_text(result, context))

    return 0


# =============================================================================
# RAG Agent Pipeline (Part 3) - Agent Integration
# =============================================================================
# This section implements the AI agent using OpenAI Agents SDK that:
# - Uses the retrieval pipeline (Part 2) as a tool
# - Provides grounded Q&A over textbook content
# - Refuses to answer when context is insufficient
# - Includes source citations in all responses
# =============================================================================


# -----------------------------------------------------------------------------
# T007-T010: Agent Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Configuration for the RAG agent."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0  # Deterministic for grounding
    k: int = 5  # Number of chunks to retrieve
    score_threshold: float = 0.3  # Below this = refuse
    tool_choice: str = "required"  # Always invoke retrieval
    collection_name: str = "textbook_chunks"


@dataclass
class GroundingContext:
    """Context assembled for agent grounding."""

    chunks: list  # list[RetrievedChunk]
    formatted_text: str
    should_refuse: bool
    refusal_reason: str = ""
    max_score: float = 0.0
    query: str = ""

    @property
    def source_count(self) -> int:
        """Number of unique sources."""
        return len(set(c.source_url for c in self.chunks))


@dataclass
class Citation:
    """Source citation for a retrieved chunk."""

    source_url: str
    chapter: str
    section: str
    score: float
    chunk_index: int

    def format(self) -> str:
        """Format as inline citation."""
        return f"[Source: {self.source_url} | Chapter: {self.chapter} | Section: {self.section} | Score: {self.score:.2f}]"


@dataclass
class AgentResponse:
    """Structured response from the RAG agent."""

    answer: str
    citations: list  # list[Citation]
    query: str
    grounded: bool  # True if answer uses retrieved context
    refused: bool  # True if agent refused to answer
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    tool_calls: list = field(default_factory=list)

    @property
    def total_time_ms(self) -> float:
        return self.retrieval_time_ms + self.generation_time_ms


# -----------------------------------------------------------------------------
# T011: Agent Constants
# -----------------------------------------------------------------------------

DEFAULT_SCORE_THRESHOLD = 0.3
MIN_CHUNKS_FOR_ANSWER = 1
DEFAULT_AGENT_K = 5
MAX_AGENT_K = 8
MIN_AGENT_K = 3

# Refusal messages
REFUSAL_NO_CONTEXT = (
    "I cannot answer this question based on the textbook content. "
    "No relevant information was found."
)
REFUSAL_LOW_SCORE = (
    "I cannot answer this question with confidence. "
    "The retrieved content has low relevance to your question."
)
REFUSAL_OFF_TOPIC = (
    "This question appears to be outside the scope of the "
    "Physical AI & Humanoid Robotics textbook."
)

# -----------------------------------------------------------------------------
# T012: Grounding System Prompt
# -----------------------------------------------------------------------------

GROUNDING_PROMPT = """You are a helpful assistant for the Physical AI & Humanoid Robotics textbook.

## CRITICAL RULES

1. **ALWAYS use the search_textbook tool** before answering any question
2. **ONLY use information from retrieved context** - never use your pre-training knowledge
3. **REFUSE to answer** if:
   - The tool returns "[NO_RELEVANT_CONTEXT]" or "[REFUSAL]"
   - No relevant content is found
   - The question is unrelated to robotics/AI topics
4. **Include source citations** in every answer

## REFUSAL TEMPLATE

When refusing, respond with:
"I cannot answer this question based on the textbook content. The search found no relevant information.
Please try rephrasing your question or ask about topics covered in the Physical AI & Humanoid Robotics textbook."

## CITATION FORMAT

After your answer, list sources used:

Sources:
[1] URL | Chapter: X | Section: Y | Score: 0.XX

## ANSWER STRUCTURE

1. Direct answer to the question (grounded in retrieved content)
2. Supporting details from context
3. Source citations
"""


# -----------------------------------------------------------------------------
# T013: Format context for agent consumption
# -----------------------------------------------------------------------------


def format_context_for_agent(
    result: RetrievalResult, threshold: float = DEFAULT_SCORE_THRESHOLD
) -> str:
    """Format retrieved chunks for agent consumption with refusal check."""
    if result is None or result.count == 0:
        return "[NO_RELEVANT_CONTEXT] No content retrieved from knowledge base."

    # Check if all scores are below threshold
    max_score = max(c.score for c in result.chunks)
    if max_score < threshold:
        return f"[REFUSAL] All retrieved content has low relevance (max score: {max_score:.2f} < threshold: {threshold})"

    # Format chunks for agent
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


# -----------------------------------------------------------------------------
# T014: search_textbook tool wrapper
# -----------------------------------------------------------------------------

# Global config for agent (can be overridden per request)
_agent_config = AgentConfig()


@function_tool
def search_textbook(query: str) -> str:
    """
    Search the Physical AI & Humanoid Robotics textbook for relevant content.

    Args:
        query: Natural language question about robotics, AI, or related topics

    Returns:
        Retrieved context with source citations, or refusal message if no relevant content found
    """
    logger.info(f"search_textbook tool invoked with query: {query[:100]}...")

    result, error = search_knowledge_base(
        query, k=_agent_config.k, collection_name=_agent_config.collection_name
    )

    if error:
        logger.error(f"Retrieval error: {error}")
        return f"[ERROR] Unable to search the knowledge base: {error}"

    formatted = format_context_for_agent(result, _agent_config.score_threshold)
    logger.info(
        f"Retrieved {result.count} chunks, max_score={max(c.score for c in result.chunks) if result.chunks else 0:.3f}"
    )

    return formatted


# -----------------------------------------------------------------------------
# T015-T016: Agent instance and ask_agent function
# -----------------------------------------------------------------------------


def create_agent(config: AgentConfig = None) -> Agent:
    """Create the RAG agent with configured tools and settings."""
    global _agent_config
    if config:
        _agent_config = config

    return Agent(
        name="textbook_qa",
        instructions=GROUNDING_PROMPT,
        tools=[search_textbook],
        model=_agent_config.model,
        model_settings=ModelSettings(
            tool_choice="required", temperature=_agent_config.temperature
        ),
    )


async def ask_agent(question: str, config: AgentConfig = None) -> AgentResponse:
    """Ask the agent a question and get a grounded response."""
    import time

    if config is None:
        config = AgentConfig()

    agent = create_agent(config)

    start_time = time.time()

    try:
        result = await Runner.run(agent, question)
        generation_time_ms = (time.time() - start_time) * 1000

        # Check if response indicates refusal
        answer = result.final_output
        refused = (
            "[NO_RELEVANT_CONTEXT]" in answer
            or "[REFUSAL]" in answer
            or "cannot answer" in answer.lower()
        )

        return AgentResponse(
            answer=answer,
            citations=[],  # Will be populated in T032
            query=question,
            grounded=not refused,
            refused=refused,
            retrieval_time_ms=0,  # TODO: Track separately
            generation_time_ms=generation_time_ms,
            tool_calls=["search_textbook"],
        )

    except Exception as e:
        logger.error(f"Agent error: {e}")
        return AgentResponse(
            answer=f"Error: {str(e)}",
            citations=[],
            query=question,
            grounded=False,
            refused=False,
            retrieval_time_ms=0,
            generation_time_ms=0,
            tool_calls=[],
        )


# -----------------------------------------------------------------------------
# T017: Format response for text output
# -----------------------------------------------------------------------------


def format_agent_response_text(response: AgentResponse) -> str:
    """Format agent response for text output."""
    output = []
    output.append(f"Question: {response.query}")
    output.append("")

    if response.refused:
        output.append(response.answer)
    else:
        output.append("Answer:")
        output.append(response.answer)

    output.append("")
    output.append(
        f"[Grounded: {response.grounded} | Time: {response.total_time_ms:.0f}ms]"
    )

    return "\n".join(output)


# -----------------------------------------------------------------------------
# T018-T019: CLI ask command
# -----------------------------------------------------------------------------


def run_ask(
    question: str,
    k: int = DEFAULT_AGENT_K,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    output_format: str = "text",
    verbose: bool = False,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """Execute ask command and return exit code."""
    # Validate question
    if not question or not question.strip():
        safe_print("Error: Question cannot be empty")
        return 2

    # Create config
    config = AgentConfig(
        k=k, score_threshold=threshold, collection_name=collection_name
    )

    # Run agent
    try:
        response = asyncio.run(ask_agent(question, config))
    except Exception as e:
        safe_print(f"Error: {e}")
        return 1

    # Output response
    if output_format == "json":
        output = {
            "question": response.query,
            "answer": response.answer,
            "grounded": response.grounded,
            "refused": response.refused,
            "citations": (
                [asdict(c) for c in response.citations] if response.citations else []
            ),
            "metadata": {
                "retrieval_time_ms": response.retrieval_time_ms,
                "generation_time_ms": response.generation_time_ms,
                "total_time_ms": response.total_time_ms,
                "tool_calls": response.tool_calls,
            },
        }
        safe_print(json.dumps(output, indent=2))
    else:
        safe_print(format_agent_response_text(response))

    return (
        0 if not response.refused or response.grounded else 0
    )  # Refusal is success, not error


# =============================================================================
# T053-T056: Full pipeline integration
# =============================================================================


def run_full_pipeline(
    base_url: str, collection_name: str, recreate: bool = False
) -> int:
    """Run the complete pipeline: ingest → chunk → embed → store → verify."""
    print(f"\n{'=' * 60}")
    print("RAG Knowledge Base Pipeline")
    print(f"{'=' * 60}")
    print(f"Base URL:   {base_url}")
    print(f"Collection: {collection_name}")
    print(f"Recreate:   {recreate}")
    print(f"{'=' * 60}\n")

    # Stage 1: Ingest
    print("\n[1/5] INGESTING PAGES...")
    result = ingest_pages(base_url, "pages.json")
    if result == 2:
        print("CRITICAL: Ingestion failed completely")
        return 2

    # Stage 2: Chunk
    print("\n[2/5] CHUNKING CONTENT...")
    result = chunk_pages("pages.json", "chunks.json")
    if result == 2:
        print("CRITICAL: No chunks generated")
        return 2

    # Stage 3: Embed
    print("\n[3/5] GENERATING EMBEDDINGS...")
    result = embed_chunks("chunks.json", "embeddings.json")
    if result != 0:
        print("CRITICAL: Embedding generation failed")
        return 2

    # Stage 4: Store
    print("\n[4/5] STORING IN QDRANT...")
    result = store_vectors("chunks.json", "embeddings.json", collection_name, recreate)
    if result != 0:
        print("CRITICAL: Storage failed")
        return 2

    # Stage 5: Verify
    print("\n[5/5] VERIFYING PIPELINE...")
    result = verify_pipeline(collection_name, "What is ROS 2?")

    print(f"\n{'=' * 60}")
    if result == 0:
        print("PIPELINE COMPLETE: All stages successful")
    else:
        print("PIPELINE COMPLETE: With warnings")
    print(f"{'=' * 60}")

    return result


# =============================================================================
# T009: CLI argument parser
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="textbook-rag-kb",
        description="RAG Knowledge Base Pipeline for Textbook Website",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Fetch and process pages from textbook site"
    )
    ingest_parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="Base URL of Docusaurus site"
    )
    ingest_parser.add_argument(
        "--output", default="pages.json", help="Output file for extracted pages"
    )

    # chunk command
    chunk_parser = subparsers.add_parser("chunk", help="Split content into chunks")
    chunk_parser.add_argument(
        "--input", default="pages.json", help="Input file with extracted pages"
    )
    chunk_parser.add_argument(
        "--output", default="chunks.json", help="Output file for chunks"
    )
    chunk_parser.add_argument(
        "--min-tokens", type=int, default=MIN_TOKENS, help="Minimum tokens per chunk"
    )
    chunk_parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS, help="Maximum tokens per chunk"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP_TOKENS,
        help="Token overlap between chunks",
    )

    # embed command
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings for chunks")
    embed_parser.add_argument(
        "--input", default="chunks.json", help="Input file with chunks"
    )
    embed_parser.add_argument(
        "--output", default="embeddings.json", help="Output file for embeddings"
    )
    embed_parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Chunks per API request"
    )

    # store command
    store_parser = subparsers.add_parser("store", help="Store embeddings in Qdrant")
    store_parser.add_argument(
        "--chunks", default="chunks.json", help="Chunks file for metadata"
    )
    store_parser.add_argument(
        "--embeddings", default="embeddings.json", help="Embeddings file"
    )
    store_parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name"
    )
    store_parser.add_argument(
        "--recreate", action="store_true", help="Delete and recreate collection"
    )

    # verify command
    verify_parser = subparsers.add_parser(
        "verify", help="Validate pipeline with sample queries"
    )
    verify_parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name"
    )
    verify_parser.add_argument(
        "--query", default="What is ROS 2?", help="Sample search query"
    )

    # run command
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="Base URL of Docusaurus site"
    )
    run_parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name"
    )
    run_parser.add_argument(
        "--recreate", action="store_true", help="Delete and recreate collection"
    )

    # search command (Part 2 - Retrieval Pipeline)
    search_parser = subparsers.add_parser(
        "search", help="Search the knowledge base with a natural language query"
    )
    search_parser.add_argument(
        "--query", "-q", required=True, help="Natural language search query"
    )
    search_parser.add_argument(
        "--k", type=int, default=5, help="Number of results (3-8, default: 5)"
    )
    search_parser.add_argument(
        "--format",
        "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    search_parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name"
    )

    # ask command (Part 3 - Agent Integration)
    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question about the textbook using the AI agent"
    )
    ask_parser.add_argument(
        "question", help="Natural language question about the textbook"
    )
    ask_parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_AGENT_K,
        help="Number of chunks to retrieve (3-8, default: 5)",
    )
    ask_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Minimum relevance score (0.0-1.0, default: 0.3)",
    )
    ask_parser.add_argument(
        "--format",
        "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    ask_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show retrieval details"
    )
    ask_parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name"
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "ingest":
        return ingest_pages(args.base_url, args.output)

    elif args.command == "chunk":
        return chunk_pages(
            args.input, args.output, args.min_tokens, args.max_tokens, args.overlap
        )

    elif args.command == "embed":
        return embed_chunks(args.input, args.output, args.batch_size)

    elif args.command == "store":
        return store_vectors(
            args.chunks, args.embeddings, args.collection, args.recreate
        )

    elif args.command == "verify":
        return verify_pipeline(args.collection, args.query)

    elif args.command == "run":
        return run_full_pipeline(args.base_url, args.collection, args.recreate)

    elif args.command == "search":
        return run_search(args.query, args.k, args.format, args.collection)

    elif args.command == "ask":
        return run_ask(
            args.question,
            args.k,
            args.threshold,
            args.format,
            args.verbose,
            args.collection,
        )

    return 0


# =============================================================================
# FastAPI Router Registration (Part 4)
# =============================================================================
# Register routes after all functions are defined to avoid circular imports

from routes import router
app.include_router(router)


if __name__ == "__main__":
    sys.exit(main())
