"""BM25 Retriever - Full-text search using PostgreSQL tsvector."""

from dataclasses import dataclass

import structlog

from ..filters.metadata_filter import MetadataFilter, MetadataQuery
from ..storage.pg_vector_store import PgVectorStore, StoredChunk

logger = structlog.get_logger()


@dataclass
class BM25Result:
    """A search result from BM25 full-text search."""

    chunk: StoredChunk
    rank: float
    headline: str | None = None


class BM25Retriever:
    """Full-text search retriever using PostgreSQL tsvector/tsquery.

    Uses PostgreSQL's built-in full-text search with ts_rank for scoring.
    The chunks table should have a GIN index on to_tsvector('english', content).
    """

    def __init__(
        self,
        store: PgVectorStore,
        default_limit: int = 20,
    ):
        self.store = store
        self.default_limit = default_limit
        self.metadata_filter = MetadataFilter()

    async def search(
        self,
        query: str,
        limit: int | None = None,
        metadata_query: MetadataQuery | None = None,
        include_headline: bool = False,
    ) -> list[BM25Result]:
        """Search for chunks matching the query using full-text search.

        Args:
            query: Natural language search query
            limit: Maximum number of results (default: 20)
            metadata_query: Optional metadata filters
            include_headline: Whether to include highlighted snippets

        Returns:
            List of BM25Result sorted by relevance rank (highest first)
        """
        k = limit or self.default_limit

        # Build the tsquery from natural language
        # plainto_tsquery handles natural language -> tsquery conversion
        tsquery = "plainto_tsquery('english', $1)"

        # Build headline selection if requested
        headline_select = ""
        if include_headline:
            headline_select = f"""
                ts_headline(
                    'english',
                    content,
                    {tsquery},
                    'StartSel=<mark>, StopSel=</mark>, MaxWords=50, MinWords=25'
                ) as headline,
            """
        else:
            headline_select = "NULL as headline,"

        # Start building the query
        base_query = f"""
            SELECT
                c.id, c.chunk_id, c.document_id, c.content, c.token_count,
                c.chunk_index, c.section_path, c.embedding, c.created_at,
                {headline_select}
                ts_rank(to_tsvector('english', c.content), {tsquery}) as rank
            FROM chunks c
        """

        # Build metadata filter if provided
        params = [query]
        param_offset = 2
        where_conditions = [f"to_tsvector('english', c.content) @@ {tsquery}"]

        if metadata_query and not metadata_query.is_empty():
            # Join with documents table for metadata filtering
            doc_subquery, doc_params = self.metadata_filter.build_document_ids_subquery(
                metadata_query, param_offset
            )
            if doc_subquery:
                where_conditions.append(f"c.document_id IN ({doc_subquery})")
                params.extend(doc_params)
                param_offset += len(doc_params)

        where_clause = " AND ".join(where_conditions)

        full_query = f"""
            {base_query}
            WHERE {where_clause}
            ORDER BY rank DESC
            LIMIT ${param_offset}
        """
        params.append(k)

        async with self.store.connection() as conn:
            rows = await conn.fetch(full_query, *params)

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
                results.append(
                    BM25Result(
                        chunk=chunk,
                        rank=float(row["rank"]),
                        headline=row["headline"],
                    )
                )

            logger.debug(
                "bm25_search",
                query=query[:50],
                num_results=len(results),
                top_rank=results[0].rank if results else 0,
            )

            return results

    async def search_with_phrases(
        self,
        query: str,
        phrases: list[str] | None = None,
        limit: int | None = None,
        metadata_query: MetadataQuery | None = None,
    ) -> list[BM25Result]:
        """Search with support for exact phrase matching.

        Allows combining free-text search with specific phrase requirements.

        Args:
            query: Natural language search query
            phrases: Exact phrases that must appear (ANDed together)
            limit: Maximum number of results
            metadata_query: Optional metadata filters

        Returns:
            List of BM25Result
        """
        k = limit or self.default_limit

        # Build combined tsquery
        tsquery_parts = ["plainto_tsquery('english', $1)"]
        params = [query]
        param_offset = 2

        if phrases:
            for phrase in phrases:
                # phraseto_tsquery handles exact phrase matching
                tsquery_parts.append(f"phraseto_tsquery('english', ${param_offset})")
                params.append(phrase)
                param_offset += 1

        # Combine with AND
        combined_tsquery = " && ".join(tsquery_parts)

        base_query = f"""
            SELECT
                c.id, c.chunk_id, c.document_id, c.content, c.token_count,
                c.chunk_index, c.section_path, c.embedding, c.created_at,
                NULL as headline,
                ts_rank(to_tsvector('english', c.content), {combined_tsquery}) as rank
            FROM chunks c
        """

        where_conditions = [f"to_tsvector('english', c.content) @@ ({combined_tsquery})"]

        if metadata_query and not metadata_query.is_empty():
            doc_subquery, doc_params = self.metadata_filter.build_document_ids_subquery(
                metadata_query, param_offset
            )
            if doc_subquery:
                where_conditions.append(f"c.document_id IN ({doc_subquery})")
                params.extend(doc_params)
                param_offset += len(doc_params)

        where_clause = " AND ".join(where_conditions)

        full_query = f"""
            {base_query}
            WHERE {where_clause}
            ORDER BY rank DESC
            LIMIT ${param_offset}
        """
        params.append(k)

        async with self.store.connection() as conn:
            rows = await conn.fetch(full_query, *params)

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
                results.append(
                    BM25Result(
                        chunk=chunk,
                        rank=float(row["rank"]),
                        headline=None,
                    )
                )

            return results
