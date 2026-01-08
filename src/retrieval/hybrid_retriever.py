"""Hybrid Retriever - Combine BM25 and vector search using Reciprocal Rank Fusion."""

from dataclasses import dataclass

import structlog

from ..embeddings.voyage_embedder import VoyageEmbedder
from ..filters.metadata_filter import MetadataFilter, MetadataQuery
from ..storage.pg_vector_store import PgVectorStore, SearchResult, StoredChunk
from .bm25_retriever import BM25Result, BM25Retriever

logger = structlog.get_logger()


@dataclass
class HybridResult:
    """A search result from hybrid retrieval with RRF score."""

    chunk: StoredChunk
    rrf_score: float
    vector_rank: int | None
    bm25_rank: int | None
    vector_similarity: float | None
    bm25_score: float | None


@dataclass
class HybridRetrievalResult:
    """Complete result from hybrid retrieval."""

    query: str
    results: list[HybridResult]
    total_tokens: int
    vector_results_count: int
    bm25_results_count: int


class HybridRetriever:
    """Hybrid retriever combining BM25 and vector search using RRF.

    Reciprocal Rank Fusion (RRF) formula:
        RRF_score(d) = sum(1 / (k + rank_i(d)))

    where k is typically 60 (prevents high-ranked documents from dominating).

    The hybrid approach benefits from:
    - BM25: Exact keyword matching, handles rare terms well
    - Vector: Semantic similarity, handles synonyms and paraphrasing
    """

    def __init__(
        self,
        embedder: VoyageEmbedder,
        store: PgVectorStore,
        bm25_retriever: BM25Retriever | None = None,
        rrf_k: int = 60,
        default_limit: int = 10,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        """Initialize hybrid retriever.

        Args:
            embedder: Voyage embedder for query embedding
            store: PostgreSQL vector store
            bm25_retriever: Optional pre-configured BM25 retriever
            rrf_k: RRF constant k (default: 60)
            default_limit: Default number of results to return
            vector_weight: Weight for vector search RRF scores
            bm25_weight: Weight for BM25 RRF scores
        """
        self.embedder = embedder
        self.store = store
        self.bm25_retriever = bm25_retriever or BM25Retriever(store)
        self.rrf_k = rrf_k
        self.default_limit = default_limit
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.metadata_filter = MetadataFilter()

    async def retrieve(
        self,
        query: str,
        limit: int | None = None,
        metadata_query: MetadataQuery | None = None,
        vector_limit: int | None = None,
        bm25_limit: int | None = None,
    ) -> HybridRetrievalResult:
        """Retrieve documents using hybrid BM25 + vector search with RRF.

        Args:
            query: Natural language query
            limit: Number of final results to return
            metadata_query: Optional metadata filters
            vector_limit: Number of results to fetch from vector search (default: 2x limit)
            bm25_limit: Number of results to fetch from BM25 (default: 2x limit)

        Returns:
            HybridRetrievalResult with fused results
        """
        k = limit or self.default_limit
        v_limit = vector_limit or (k * 2)
        b_limit = bm25_limit or (k * 2)

        # Run both searches in parallel
        query_embedding = self.embedder.embed_query(query)

        # Vector search with optional metadata filter
        vector_results = await self._vector_search_with_filter(
            query_embedding, v_limit, metadata_query
        )

        # BM25 search with optional metadata filter
        bm25_results = await self.bm25_retriever.search(
            query, limit=b_limit, metadata_query=metadata_query
        )

        # Fuse results using RRF
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        # Take top k
        top_results = fused[:k]

        total_tokens = sum(r.chunk.token_count for r in top_results)

        logger.info(
            "hybrid_retrieval",
            query=query[:50],
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
            fused_count=len(fused),
            returned_count=len(top_results),
            total_tokens=total_tokens,
        )

        return HybridRetrievalResult(
            query=query,
            results=top_results,
            total_tokens=total_tokens,
            vector_results_count=len(vector_results),
            bm25_results_count=len(bm25_results),
        )

    async def _vector_search_with_filter(
        self,
        query_embedding: list[float],
        limit: int,
        metadata_query: MetadataQuery | None,
    ) -> list[SearchResult]:
        """Perform vector search with optional metadata filtering."""
        if metadata_query is None or metadata_query.is_empty():
            return await self.store.search(query_embedding, limit=limit)

        # Build filtered vector search query
        doc_subquery, doc_params = self.metadata_filter.build_document_ids_subquery(
            metadata_query, param_offset=2
        )

        if not doc_subquery:
            return await self.store.search(query_embedding, limit=limit)

        async with self.store.connection() as conn:
            query = f"""
                SELECT
                    id, chunk_id, document_id, content, token_count,
                    chunk_index, section_path, embedding, created_at,
                    1 - (embedding <=> $1) as similarity
                FROM chunks
                WHERE document_id IN ({doc_subquery})
                ORDER BY embedding <=> $1
                LIMIT ${len(doc_params) + 2}
            """

            params = [query_embedding, *doc_params, limit]
            rows = await conn.fetch(query, *params)

            results = []
            for row in rows:
                chunk = StoredChunk(
                    id=row["id"],
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    token_count=row["token_count"],
                    chunk_index=row["chunk_index"],
                    section_path=row["section_path"],
                    embedding=list(row["embedding"]) if row["embedding"] is not None else [],
                    created_at=row["created_at"],
                )
                results.append(SearchResult(chunk=chunk, similarity=row["similarity"]))

            return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[BM25Result],
    ) -> list[HybridResult]:
        """Fuse results from vector and BM25 search using RRF.

        RRF_score(d) = sum(weight_i / (k + rank_i(d)))
        """
        # Build chunk_id -> ranks mapping
        chunk_data: dict[str, dict] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.chunk.chunk_id
            chunk_data[chunk_id] = {
                "chunk": result.chunk,
                "vector_rank": rank,
                "vector_similarity": result.similarity,
                "bm25_rank": None,
                "bm25_score": None,
            }

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result.chunk.chunk_id
            if chunk_id in chunk_data:
                chunk_data[chunk_id]["bm25_rank"] = rank
                chunk_data[chunk_id]["bm25_score"] = result.rank
            else:
                chunk_data[chunk_id] = {
                    "chunk": result.chunk,
                    "vector_rank": None,
                    "vector_similarity": None,
                    "bm25_rank": rank,
                    "bm25_score": result.rank,
                }

        hybrid_results = []
        for _, data in chunk_data.items():
            rrf_score = 0.0

            if data["vector_rank"] is not None:
                rrf_score += self.vector_weight / (self.rrf_k + data["vector_rank"])

            if data["bm25_rank"] is not None:
                rrf_score += self.bm25_weight / (self.rrf_k + data["bm25_rank"])

            hybrid_results.append(
                HybridResult(
                    chunk=data["chunk"],
                    rrf_score=rrf_score,
                    vector_rank=data["vector_rank"],
                    bm25_rank=data["bm25_rank"],
                    vector_similarity=data["vector_similarity"],
                    bm25_score=data["bm25_score"],
                )
            )

        # Sort by RRF score descending
        hybrid_results.sort(key=lambda x: x.rrf_score, reverse=True)

        return hybrid_results

    def format_context(
        self,
        results: list[HybridResult],
        max_tokens: int | None = None,
    ) -> str:
        """Format retrieved chunks into context string."""
        context_parts = []
        token_count = 0

        for result in results:
            chunk = result.chunk

            if max_tokens and token_count + chunk.token_count > max_tokens:
                break

            source = chunk.document_id.upper()
            if chunk.section_path:
                source = f"{source} ({chunk.section_path})"

            context_parts.append(f"[Source: {source}]\n{chunk.content}\n")
            token_count += chunk.token_count

        return "\n---\n".join(context_parts)

    def format_context_with_citations(
        self,
        results: list[HybridResult],
        max_tokens: int | None = None,
    ) -> tuple[str, list[dict]]:
        """Format context with citation metadata.

        Returns:
            Tuple of (context_string, citation_list)
        """
        context_parts = []
        citations = []
        token_count = 0

        for i, result in enumerate(results):
            chunk = result.chunk
            citation_id = f"[{i + 1}]"

            if max_tokens and token_count + chunk.token_count > max_tokens:
                break

            citation = {
                "id": citation_id,
                "document_id": chunk.document_id,
                "section": chunk.section_path or "Full document",
                "chunk_id": chunk.chunk_id,
                "rrf_score": result.rrf_score,
                "vector_rank": result.vector_rank,
                "bm25_rank": result.bm25_rank,
            }
            citations.append(citation)

            context_parts.append(
                f"{citation_id} [{chunk.document_id.upper()}"
                f"{' - ' + chunk.section_path if chunk.section_path else ''}]\n"
                f"{chunk.content}"
            )
            token_count += chunk.token_count

        context = "\n\n---\n\n".join(context_parts)
        return context, citations
