"""FastAPI application for Ethereum Protocol Intelligence System."""

import os
from contextlib import asynccontextmanager
from datetime import datetime

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..agents import AgentBudget, ReactAgent, RetrievalTool
from ..config import DEFAULT_MODEL
from ..embeddings.voyage_embedder import VoyageEmbedder
from ..generation.cited_generator import CitedGenerator
from ..generation.simple_generator import SimpleGenerator
from ..generation.validated_generator import ValidatedGenerator
from ..graph import DependencyTraverser, FalkorDBStore
from ..retrieval.simple_retriever import SimpleRetriever
from ..storage.pg_vector_store import PgVectorStore

logger = structlog.get_logger()

# Global instances
store: PgVectorStore | None = None
embedder: VoyageEmbedder | None = None
retriever: SimpleRetriever | None = None
retrieval_tool: RetrievalTool | None = None
simple_generator: SimpleGenerator | None = None
cited_generator: CitedGenerator | None = None
validated_generator: ValidatedGenerator | None = None
graph_store: FalkorDBStore | None = None
dependency_traverser: DependencyTraverser | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global store, embedder, retriever, retrieval_tool
    global simple_generator, cited_generator, validated_generator
    global graph_store, dependency_traverser

    logger.info("starting_application")

    # Initialize components
    store = PgVectorStore()
    await store.connect()

    embedder = VoyageEmbedder()
    retriever = SimpleRetriever(embedder=embedder, store=store)
    retrieval_tool = RetrievalTool(simple_retriever=retriever, default_limit=5)
    simple_generator = SimpleGenerator(retriever=retriever)
    cited_generator = CitedGenerator(retriever=retriever)

    # Validated generator is optional (requires NLI model)
    try:
        validated_generator = ValidatedGenerator(retriever=retriever)
        logger.info("validated_generator_initialized")
    except Exception as e:
        logger.warning("validated_generator_not_available", error=str(e))
        validated_generator = None

    # Initialize graph store (optional - continues if unavailable)
    try:
        graph_store = FalkorDBStore(
            host=os.environ.get("FALKORDB_HOST", "localhost"),
            port=int(os.environ.get("FALKORDB_PORT", "6379")),
        )
        graph_store.connect()
        dependency_traverser = DependencyTraverser(graph_store)
        logger.info("graph_store_initialized")
    except Exception as e:
        logger.warning("graph_store_not_available", error=str(e))
        graph_store = None
        dependency_traverser = None

    yield

    # Cleanup
    if graph_store:
        graph_store.close()
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
        description="Generation mode: 'simple', 'cited', 'validated', 'agentic', or 'graph'"
    )
    validate: bool = Field(
        default=True,
        description="Run NLI validation (only for 'validated' mode)"
    )
    max_retrievals: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Max retrieval attempts for 'agentic' mode"
    )
    include_dependencies: bool = Field(
        default=True,
        description="Include EIP dependencies in 'graph' mode"
    )


class EvidenceSource(BaseModel):
    """Evidence source in response."""
    document_id: str
    section: str | None
    similarity: float
    # Enriched metadata
    title: str | None = None
    author: str | None = None
    url: str | None = None


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

    # Agentic fields (Phase 8)
    llm_calls: int | None = None
    retrieval_count: int | None = None
    reasoning_chain: list[str] | None = None
    termination_reason: str | None = None

    # Graph fields (Phase 4)
    related_eips: list[int] | None = None
    dependency_chain: list[int] | None = None


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


async def enrich_sources(sources: list[EvidenceSource]) -> list[EvidenceSource]:
    """Enrich sources with document metadata (title, author, url)."""
    import json

    if not sources:
        return sources

    # Get unique document IDs
    doc_ids = list({s.document_id for s in sources})

    # Fetch document metadata
    doc_metadata = {}
    async with store.connection() as conn:
        rows = await conn.fetch(
            """
            SELECT document_id, title, author, metadata
            FROM documents
            WHERE document_id = ANY($1)
            """,
            doc_ids,
        )
        for row in rows:
            raw_metadata = row["metadata"]
            # Handle metadata as string or dict
            if isinstance(raw_metadata, str):
                metadata = json.loads(raw_metadata) if raw_metadata else {}
            else:
                metadata = raw_metadata or {}
            url = metadata.get("url")

            # Generate URL if not stored
            if not url:
                if row["document_id"].startswith("ethresearch-topic-"):
                    topic_id = metadata.get("topic_id")
                    slug = metadata.get("slug", "topic")
                    if topic_id:
                        url = f"https://ethresear.ch/t/{slug}/{topic_id}"
                elif row["document_id"].startswith("eip-"):
                    eip_num = row["document_id"].replace("eip-", "")
                    url = f"https://eips.ethereum.org/EIPS/eip-{eip_num}"

            doc_metadata[row["document_id"]] = {
                "title": row["title"],
                "author": row["author"],
                "url": url,
            }

    # Enrich sources
    enriched = []
    for s in sources:
        meta = doc_metadata.get(s.document_id, {})
        enriched.append(
            EvidenceSource(
                document_id=s.document_id,
                section=s.section,
                similarity=s.similarity,
                title=meta.get("title"),
                author=meta.get("author"),
                url=meta.get("url"),
            )
        )
    return enriched


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the system with a question about Ethereum protocol.

    Modes:
    - simple: Basic RAG (Phase 0) - fast, no citations
    - cited: With citations (Phase 1) - includes source references
    - validated: With NLI validation (Phase 2) - verifies claims against evidence
    - agentic: ReAct agent (Phase 8) - iterative retrieval with reasoning chain
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
            sources = await enrich_sources(sources)

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
            sources = await enrich_sources(sources)

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

        elif request.mode == "agentic":
            # Phase 8: Agentic retrieval with ReAct loop
            budget = AgentBudget(
                max_retrievals=request.max_retrievals,
                max_tokens=10000,
                max_llm_calls=request.max_retrievals * 3,
            )

            agent = ReactAgent(
                retrieval_tool=retrieval_tool,
                budget=budget,
                enable_reflection=True,
                enable_backtracking=True,
            )

            result = await agent.run(request.query)

            sources = [
                EvidenceSource(
                    document_id=r.get("document_id", "unknown"),
                    section=r.get("section_path"),
                    similarity=r.get("similarity", 0.0),
                )
                for r in result.retrievals
            ]

            reasoning_chain = [
                f"{t.action.value}: {t.content[:200]}"
                for t in result.thoughts
            ]

            return QueryResponse(
                query=request.query,
                response=result.answer,
                sources=sources,
                mode="agentic",
                model=DEFAULT_MODEL,
                input_tokens=result.total_tokens_retrieved,
                output_tokens=0,
                llm_calls=result.llm_calls,
                retrieval_count=result.retrieval_count,
                reasoning_chain=reasoning_chain,
                termination_reason=result.termination_reason,
            )

        elif request.mode == "graph":
            # Phase 4: Graph-augmented retrieval
            if graph_store is None or dependency_traverser is None:
                raise HTTPException(
                    status_code=503,
                    detail="Graph database not available"
                )

            # First do standard cited generation
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

            # Extract EIP numbers from sources and get graph context
            related_eips: list[int] = []
            dependency_chain: list[int] = []

            if request.include_dependencies:
                import re
                eip_numbers = set()
                for source in sources:
                    match = re.match(r"eip-(\d+)", source.document_id)
                    if match:
                        eip_numbers.add(int(match.group(1)))

                # Get dependencies for each mentioned EIP
                for eip_num in eip_numbers:
                    try:
                        deps = dependency_traverser.get_dependencies(eip_num, max_depth=2)
                        related_eips.extend(deps.direct_dependencies)
                        related_eips.extend(deps.direct_dependents)
                        if deps.all_dependencies:
                            dependency_chain.extend(
                                [d.eip_number for d in deps.all_dependencies]
                            )
                    except Exception as e:
                        logger.debug("graph_lookup_failed", eip=eip_num, error=str(e))

                # Deduplicate while preserving order
                related_eips = list(dict.fromkeys(related_eips))
                dependency_chain = list(dict.fromkeys(dependency_chain))

            return QueryResponse(
                query=request.query,
                response=result.response_with_citations,
                sources=sources,
                mode="graph",
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                related_eips=related_eips if related_eips else None,
                dependency_chain=dependency_chain if dependency_chain else None,
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown mode: {request.mode}. Use 'simple', 'cited', 'validated', 'agentic', or 'graph'."
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


# Graph endpoints

@app.get("/eip/{eip_number}/dependencies")
async def get_eip_dependencies(eip_number: int, depth: int = 3):
    """Get EIP dependency chain (what this EIP requires)."""
    if dependency_traverser is None:
        raise HTTPException(status_code=503, detail="Graph database not available")

    try:
        result = dependency_traverser.get_dependencies(eip_number, max_depth=depth)
        return {
            "eip": eip_number,
            "direct_dependencies": result.direct_dependencies,
            "all_dependencies": [
                {
                    "eip": d.eip_number,
                    "title": d.title,
                    "status": d.status,
                    "depth": d.depth,
                }
                for d in result.all_dependencies
            ],
            "supersedes": result.supersedes,
            "superseded_by": result.superseded_by,
            "depth": depth,
        }
    except Exception as e:
        logger.exception("dependency_lookup_failed", eip=eip_number)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/eip/{eip_number}/dependents")
async def get_eip_dependents(eip_number: int, depth: int = 3):
    """Get EIPs that depend on this one."""
    if dependency_traverser is None:
        raise HTTPException(status_code=503, detail="Graph database not available")

    try:
        result = dependency_traverser.get_dependencies(eip_number, max_depth=depth)
        return {
            "eip": eip_number,
            "direct_dependents": result.direct_dependents,
            "all_dependents": [
                {
                    "eip": d.eip_number,
                    "title": d.title,
                    "status": d.status,
                    "depth": d.depth,
                }
                for d in result.all_dependents
            ],
            "depth": depth,
        }
    except Exception as e:
        logger.exception("dependents_lookup_failed", eip=eip_number)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/eip/{eip_number}/tree")
async def get_eip_dependency_tree(eip_number: int, depth: int = 3):
    """Get a tree visualization of EIP dependencies."""
    if dependency_traverser is None:
        raise HTTPException(status_code=503, detail="Graph database not available")

    try:
        tree = dependency_traverser.get_dependency_tree(eip_number, max_depth=depth)
        return tree
    except Exception as e:
        logger.exception("tree_lookup_failed", eip=eip_number)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/graph/stats")
async def get_graph_stats():
    """Get graph statistics."""
    if graph_store is None:
        raise HTTPException(status_code=503, detail="Graph database not available")

    try:
        node_count = graph_store.count_nodes()
        relationship_counts = graph_store.count_relationships()
        return {
            "nodes": node_count,
            "relationships": relationship_counts,
            "total_relationships": sum(relationship_counts.values()),
        }
    except Exception as e:
        logger.exception("graph_stats_failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/graph/most-depended")
async def get_most_depended_upon(limit: int = 10):
    """Get the most depended-upon EIPs."""
    if dependency_traverser is None:
        raise HTTPException(status_code=503, detail="Graph database not available")

    try:
        results = dependency_traverser.get_most_depended_upon(limit=limit)
        return {
            "most_depended": [
                {"eip": eip, "dependent_count": count}
                for eip, count in results
            ]
        }
    except Exception as e:
        logger.exception("most_depended_failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
