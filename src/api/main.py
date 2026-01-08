"""FastAPI application for Ethereum Protocol Intelligence System."""

from contextlib import asynccontextmanager
from datetime import datetime

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..embeddings.voyage_embedder import VoyageEmbedder
from ..generation.cited_generator import CitedGenerator
from ..generation.simple_generator import SimpleGenerator
from ..generation.validated_generator import ValidatedGenerator
from ..retrieval.simple_retriever import SimpleRetriever
from ..storage.pg_vector_store import PgVectorStore

logger = structlog.get_logger()

# Global instances
store: PgVectorStore | None = None
embedder: VoyageEmbedder | None = None
retriever: SimpleRetriever | None = None
simple_generator: SimpleGenerator | None = None
cited_generator: CitedGenerator | None = None
validated_generator: ValidatedGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global store, embedder, retriever, simple_generator, cited_generator, validated_generator

    logger.info("starting_application")

    # Initialize components
    store = PgVectorStore()
    await store.connect()

    embedder = VoyageEmbedder()
    retriever = SimpleRetriever(embedder=embedder, store=store)
    simple_generator = SimpleGenerator(retriever=retriever)
    cited_generator = CitedGenerator(retriever=retriever)

    # Validated generator is optional (requires NLI model)
    try:
        validated_generator = ValidatedGenerator(retriever=retriever)
        logger.info("validated_generator_initialized")
    except Exception as e:
        logger.warning("validated_generator_not_available", error=str(e))
        validated_generator = None

    yield

    # Cleanup
    await store.close()
    logger.info("application_shutdown")


app = FastAPI(
    title="Ethereum Protocol Intelligence API",
    description="RAG system for Ethereum protocol documentation",
    version="0.1.0",
    lifespan=lifespan,
)


# Request/Response models

class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., description="The question to answer")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of chunks to retrieve")
    mode: str = Field(
        default="cited",
        description="Generation mode: 'simple' (Phase 0), 'cited' (Phase 1), 'validated' (Phase 2)"
    )
    validate: bool = Field(
        default=True,
        description="Run NLI validation (only for 'validated' mode)"
    )


class EvidenceSource(BaseModel):
    """Evidence source in response."""
    document_id: str
    section: str | None
    similarity: float


class QueryResponse(BaseModel):
    """Query response model."""
    query: str
    response: str
    sources: list[EvidenceSource]
    mode: str
    model: str
    input_tokens: int
    output_tokens: int

    # Validation fields (Phase 2)
    total_claims: int | None = None
    supported_claims: int | None = None
    support_ratio: float | None = None
    is_trustworthy: bool | None = None
    validation_report: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    chunks_count: int
    documents_count: int


class StatsResponse(BaseModel):
    """Statistics response."""
    total_documents: int
    total_chunks: int
    database_connected: bool


# Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        chunks = await store.count_chunks()
        docs = await store.count_documents()
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            chunks_count=chunks,
            documents_count=docs,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    try:
        return StatsResponse(
            total_documents=await store.count_documents(),
            total_chunks=await store.count_chunks(),
            database_connected=store.pool is not None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the system with a question about Ethereum protocol.

    Modes:
    - simple: Basic RAG (Phase 0) - fast, no citations
    - cited: With citations (Phase 1) - includes source references
    - validated: With NLI validation (Phase 2) - verifies claims against evidence
    """
    try:
        if request.mode == "simple":
            # Phase 0: Simple generation
            result = await simple_generator.generate(
                query=request.query,
                top_k=request.top_k,
            )

            sources = [
                EvidenceSource(
                    document_id=r.chunk.document_id,
                    section=r.chunk.section_path,
                    similarity=r.similarity,
                )
                for r in result.retrieval.results
            ]

            return QueryResponse(
                query=request.query,
                response=result.response,
                sources=sources,
                mode="simple",
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

        elif request.mode == "cited":
            # Phase 1: With citations
            result = await cited_generator.generate(
                query=request.query,
                top_k=request.top_k,
            )

            sources = [
                EvidenceSource(
                    document_id=r.chunk.document_id,
                    section=r.chunk.section_path,
                    similarity=r.similarity,
                )
                for r in result.retrieval.results
            ]

            return QueryResponse(
                query=request.query,
                response=result.response_with_citations,
                sources=sources,
                mode="cited",
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

        elif request.mode == "validated":
            # Phase 2: With validation
            if validated_generator is None:
                raise HTTPException(
                    status_code=501,
                    detail="Validated mode not available (NLI model not loaded)"
                )

            result = await validated_generator.generate(
                query=request.query,
                top_k=request.top_k,
                validate=request.validate,
            )

            sources = [
                EvidenceSource(
                    document_id=r.chunk.document_id,
                    section=r.chunk.section_path,
                    similarity=r.similarity,
                )
                for r in result.retrieval.results
            ]

            validation_report = None
            if request.validate:
                validation_report = validated_generator.get_validation_report(result)

            return QueryResponse(
                query=request.query,
                response=result.validated_response,
                sources=sources,
                mode="validated",
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_claims=result.total_claims,
                supported_claims=result.supported_claims,
                support_ratio=result.support_ratio,
                is_trustworthy=result.is_trustworthy,
                validation_report=validation_report,
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown mode: {request.mode}. Use 'simple', 'cited', or 'validated'."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("query_failed", query=request.query[:50])
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/eip/{eip_number}")
async def get_eip(eip_number: int):
    """Get EIP metadata and chunks."""
    document_id = f"eip-{eip_number}"

    doc = await store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"EIP-{eip_number} not found")

    chunks = await store.get_chunks_by_document(document_id)

    return {
        "document_id": document_id,
        "eip_number": doc["eip_number"],
        "title": doc["title"],
        "status": doc["status"],
        "type": doc["type"],
        "category": doc["category"],
        "author": doc["author"],
        "requires": doc["requires"],
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "section": c.section_path,
                "token_count": c.token_count,
                "preview": c.content[:200] + "..." if len(c.content) > 200 else c.content,
            }
            for c in chunks
        ],
    }


@app.get("/search")
async def search(
    q: str,
    top_k: int = 10,
    document_filter: str | None = None,
):
    """Search for relevant chunks."""
    result = await retriever.retrieve(
        query=q,
        top_k=top_k,
        document_filter=document_filter,
    )

    return {
        "query": q,
        "total_results": len(result.results),
        "total_tokens": result.total_tokens,
        "results": [
            {
                "document_id": r.chunk.document_id,
                "section": r.chunk.section_path,
                "similarity": r.similarity,
                "content": r.chunk.content,
            }
            for r in result.results
        ],
    }
