"""
FastAPI route definitions for Backend-Frontend Integration.

All API endpoints are defined here and registered via APIRouter.
Routes call existing functions from main.py for actual processing.
"""

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from models import (
    QueryRequest,
    QueryResponse,
    Citation,
    ResponseMetadata,
    ErrorResponse,
    SessionResponse,
    SearchRequest,
    SearchResponse,
    RetrievedChunk,
    StoreRequest,
    StoreResponse,
    EmbedRequest,
    EmbedResponse,
)

# Import functions from main.py
from main import (
    ask_agent,
    search_knowledge_base,
    store_vectors,
    embed_chunks,
    AgentConfig,
)

logger = logging.getLogger(__name__)

# Create API router with /api prefix
router = APIRouter(prefix="/api")


# =============================================================================
# Session Endpoints
# =============================================================================


@router.post("/chatkit/session", response_model=SessionResponse, tags=["Session"])
async def create_chatkit_session() -> SessionResponse:
    """
    Generate a ChatKit client session token.

    Returns an ephemeral token for ChatKit authentication.
    Tokens are not persisted (in-memory only for local dev).
    """
    try:
        # Generate ephemeral session token
        # Format: ck_sess_{uuid} for ChatKit compatibility
        client_secret = f"ck_sess_{uuid.uuid4().hex}"

        logger.info(f"Created ChatKit session: {client_secret[:20]}...")

        return SessionResponse(client_secret=client_secret)

    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="INTERNAL_ERROR",
                message="Failed to create session token"
            ).model_dump()
        )


# =============================================================================
# Query Endpoints
# =============================================================================


@router.post("/ask", response_model=QueryResponse, tags=["Query"])
async def ask_question(request: QueryRequest) -> QueryResponse:
    """
    Ask a question to the RAG agent.

    Validates the query, invokes the grounded Q&A agent,
    and returns a structured response with citations.
    """
    start_time = time.time()

    # Validation is handled by Pydantic, but add explicit checks for error codes
    question = request.question.strip()

    # Log request
    logger.info(f"POST /api/ask - Question: {question[:100]}...")

    if not question:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                code="EMPTY_QUERY",
                message="Question cannot be empty"
            ).model_dump()
        )

    if len(question) > 1000:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                code="QUERY_TOO_LONG",
                message="Question exceeds maximum length of 1000 characters"
            ).model_dump()
        )

    try:
        # Create agent config
        config = AgentConfig(k=5, score_threshold=0.3)

        # Call the agent
        retrieval_start = time.time()
        response = await ask_agent(question, config)
        generation_time_ms = (time.time() - start_time) * 1000

        # Build citations from agent response (if available)
        citations = []
        if hasattr(response, 'citations') and response.citations:
            for c in response.citations:
                citations.append(Citation(
                    source_url=c.source_url if hasattr(c, 'source_url') else str(c.get('source_url', '')),
                    chapter=c.chapter if hasattr(c, 'chapter') else str(c.get('chapter', '')),
                    section=c.section if hasattr(c, 'section') else str(c.get('section', '')),
                    score=c.score if hasattr(c, 'score') else float(c.get('score', 0.0)),
                    chunk_index=c.chunk_index if hasattr(c, 'chunk_index') else int(c.get('chunk_index', 0))
                ))

        result = QueryResponse(
            answer=response.answer,
            citations=citations,
            grounded=response.grounded,
            refused=response.refused,
            metadata=ResponseMetadata(
                retrieval_time_ms=response.retrieval_time_ms,
                generation_time_ms=response.generation_time_ms,
                total_time_ms=response.total_time_ms,
                tool_calls=response.tool_calls
            )
        )

        # Log response
        logger.info(f"POST /api/ask - Response: grounded={result.grounded}, refused={result.refused}, time={result.metadata.total_time_ms:.0f}ms")

        return result

    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="AGENT_ERROR",
                message="Failed to process question",
                details={"error": str(e)}
            ).model_dump()
        )


@router.post("/respond", response_model=QueryResponse, tags=["Query"])
async def generate_response(request: QueryRequest) -> QueryResponse:
    """
    Alias for /ask endpoint.

    Routes to the same agent for grounded response generation.
    """
    return await ask_question(request)


# =============================================================================
# Search Endpoints
# =============================================================================


@router.post("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search_knowledge_base_endpoint(request: SearchRequest) -> SearchResponse:
    """
    Search the textbook knowledge base.

    Performs semantic search and returns relevant chunks
    with their metadata and similarity scores.
    """
    query = request.query.strip()

    # Log request
    logger.info(f"POST /api/search - Query: {query[:100]}..., k={request.k}")

    if not query:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                code="EMPTY_QUERY",
                message="Search query cannot be empty"
            ).model_dump()
        )

    if len(query) > 1000:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                code="QUERY_TOO_LONG",
                message="Query exceeds maximum length of 1000 characters"
            ).model_dump()
        )

    try:
        result, error = search_knowledge_base(query, k=request.k)

        if error:
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    code="RETRIEVAL_ERROR",
                    message=error
                ).model_dump()
            )

        # Convert to response model
        chunks = []
        if result and result.chunks:
            for chunk in result.chunks:
                chunks.append(RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    source_url=chunk.source_url,
                    text=chunk.text,
                    score=chunk.score,
                    chapter=chunk.chapter,
                    section=chunk.section,
                    chunk_index=chunk.chunk_index
                ))

        response = SearchResponse(
            query=query,
            collection=result.collection if result else "textbook_chunks",
            count=len(chunks),
            chunks=chunks
        )

        # Log response
        logger.info(f"POST /api/search - Response: count={response.count}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="RETRIEVAL_ERROR",
                message="Failed to search knowledge base",
                details={"error": str(e)}
            ).model_dump()
        )


# =============================================================================
# Pipeline Endpoints
# =============================================================================


@router.post("/store", response_model=StoreResponse, tags=["Pipeline"])
async def store_embeddings_endpoint(request: StoreRequest) -> StoreResponse:
    """
    Store embeddings in Qdrant.

    Takes chunks and embeddings files and upserts vectors
    to the specified collection.
    """
    try:
        result = store_vectors(
            request.chunks_file,
            request.embeddings_file,
            request.collection,
            request.recreate
        )

        if result != 0:
            return StoreResponse(
                status="error",
                stored_count=0,
                collection=request.collection
            )

        return StoreResponse(
            status="success",
            stored_count=0,  # Would need to track actual count
            collection=request.collection
        )

    except Exception as e:
        logger.error(f"Store error: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="INTERNAL_ERROR",
                message="Failed to store embeddings",
                details={"error": str(e)}
            ).model_dump()
        )


@router.post("/embed", response_model=EmbedResponse, tags=["Pipeline"])
async def generate_embeddings_endpoint(request: EmbedRequest) -> EmbedResponse:
    """
    Generate embeddings for chunks.

    Reads chunks from file and generates Cohere embeddings,
    saving results to the output file.
    """
    try:
        result = embed_chunks(request.chunks_file, request.output_file)

        if result != 0:
            return EmbedResponse(
                status="error",
                embedded_count=0,
                output_file=request.output_file
            )

        return EmbedResponse(
            status="success",
            embedded_count=0,  # Would need to track actual count
            output_file=request.output_file
        )

    except Exception as e:
        logger.error(f"Embed error: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="INTERNAL_ERROR",
                message="Failed to generate embeddings",
                details={"error": str(e)}
            ).model_dump()
        )
