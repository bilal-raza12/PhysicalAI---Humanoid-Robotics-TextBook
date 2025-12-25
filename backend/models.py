"""
Pydantic models for Backend-Frontend Integration API.

This module defines all request/response schemas for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


# =============================================================================
# Session Models
# =============================================================================


class SessionResponse(BaseModel):
    """Response containing ChatKit session token."""
    client_secret: str = Field(..., description="Ephemeral token for ChatKit authentication")


# =============================================================================
# Query Models
# =============================================================================


class QueryRequest(BaseModel):
    """Request to ask a question to the RAG agent."""
    question: str = Field(..., min_length=1, max_length=1000, description="Natural language question about the textbook")


class Citation(BaseModel):
    """Source reference for answer grounding."""
    source_url: str = Field(..., description="URL to textbook page")
    chapter: str = Field(default="", description="Chapter name")
    section: str = Field(default="", description="Section name")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    chunk_index: int = Field(..., ge=0, description="Chunk position in source")


class ResponseMetadata(BaseModel):
    """Timing and debugging information."""
    retrieval_time_ms: float = Field(default=0.0, ge=0, description="Time for retrieval in milliseconds")
    generation_time_ms: float = Field(default=0.0, ge=0, description="Time for generation in milliseconds")
    total_time_ms: float = Field(default=0.0, ge=0, description="Total processing time in milliseconds")
    tool_calls: list[str] = Field(default_factory=list, description="Tools invoked during processing")


class QueryResponse(BaseModel):
    """Response from the RAG agent containing grounded answer."""
    answer: str = Field(..., description="Agent's grounded response")
    citations: list[Citation] = Field(default_factory=list, description="Source references for the answer")
    grounded: bool = Field(..., description="Whether answer uses retrieved context")
    refused: bool = Field(..., description="Whether agent refused to answer")
    metadata: ResponseMetadata = Field(..., description="Timing and debug info")


# =============================================================================
# Error Models
# =============================================================================


class ErrorResponse(BaseModel):
    """Structured error response."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(default=None, description="Additional error context")


# =============================================================================
# Search Models
# =============================================================================


class SearchRequest(BaseModel):
    """Request for direct knowledge base search."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    k: int = Field(default=5, ge=3, le=8, description="Number of results to return")


class RetrievedChunk(BaseModel):
    """Single chunk from knowledge base."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    source_url: str = Field(..., description="Source page URL")
    text: str = Field(..., description="Chunk content")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    chapter: str = Field(default="", description="Chapter name")
    section: str = Field(default="", description="Section name")
    chunk_index: int = Field(default=0, ge=0, description="Position in source")


class SearchResponse(BaseModel):
    """Results from knowledge base search."""
    query: str = Field(..., description="Original search query")
    collection: str = Field(..., description="Qdrant collection name")
    count: int = Field(..., ge=0, description="Number of results found")
    chunks: list[RetrievedChunk] = Field(default_factory=list, description="Retrieved chunks")


# =============================================================================
# Pipeline Models (for /store and /embed endpoints)
# =============================================================================


class StoreRequest(BaseModel):
    """Request to store embeddings in Qdrant."""
    chunks_file: str = Field(..., description="Path to chunks JSON file")
    embeddings_file: str = Field(..., description="Path to embeddings JSON file")
    collection: str = Field(default="textbook_chunks", description="Qdrant collection name")
    recreate: bool = Field(default=False, description="Whether to recreate the collection")


class StoreResponse(BaseModel):
    """Response from storing embeddings."""
    status: str = Field(..., description="Status: success or error")
    stored_count: int = Field(default=0, description="Number of vectors stored")
    collection: str = Field(..., description="Collection name used")


class EmbedRequest(BaseModel):
    """Request to generate embeddings."""
    chunks_file: str = Field(..., description="Path to chunks JSON file")
    output_file: str = Field(default="embeddings.json", description="Output file for embeddings")


class EmbedResponse(BaseModel):
    """Response from generating embeddings."""
    status: str = Field(..., description="Status: success or error")
    embedded_count: int = Field(default=0, description="Number of chunks embedded")
    output_file: str = Field(..., description="Path to output file")
