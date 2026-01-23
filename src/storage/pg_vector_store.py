"""PostgreSQL Vector Store - Store and retrieve chunks using pgvector."""

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg
import structlog
from pgvector.asyncpg import register_vector

from ..embeddings.voyage_embedder import EmbeddedChunk

logger = structlog.get_logger()


@dataclass
class StoredChunk:
    """A chunk as stored in the database."""
    id: int
    chunk_id: str
    document_id: str
    content: str
    token_count: int
    chunk_index: int
    section_path: str | None
    embedding: list[float]
    created_at: datetime


@dataclass
class SearchResult:
    """A search result with similarity score."""
    chunk: StoredChunk
    similarity: float


class PgVectorStore:
    """PostgreSQL + pgvector storage for chunks and embeddings."""

    def __init__(
        self,
        database_url: str | None = None,
        embedding_dim: int = 1024,
    ):
        self.database_url = database_url or os.environ.get(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/eth_protocol"
        )
        self.embedding_dim = embedding_dim
        self.pool: asyncpg.Pool | None = None

    async def connect(self):
        """Initialize connection pool."""
        # First, create a connection to enable the extension
        conn = await asyncpg.connect(self.database_url)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await conn.close()

        # Now create the pool with vector type registration
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=10,
            init=self._init_connection,
        )
        logger.info("connected_to_database")

    async def _init_connection(self, conn: asyncpg.Connection):
        """Initialize each connection with pgvector."""
        await register_vector(conn)

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("closed_database_connection")

    @asynccontextmanager
    async def connection(self):
        """Get a connection from the pool."""
        async with self.pool.acquire() as conn:
            yield conn

    async def initialize_schema(self):
        """Create tables and indexes."""
        async with self.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_id VARCHAR(512) UNIQUE NOT NULL,
                    document_id VARCHAR(128) NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    section_path VARCHAR(256),
                    embedding vector({self.embedding_dim}),
                    git_commit VARCHAR(64),
                    created_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{{}}'
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(128) UNIQUE NOT NULL,
                    eip_number INTEGER,
                    title VARCHAR(512),
                    status VARCHAR(64),
                    type VARCHAR(64),
                    category VARCHAR(64),
                    author TEXT,
                    created_date VARCHAR(32),
                    requires INTEGER[],
                    git_commit VARCHAR(64),
                    raw_content TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    document_type VARCHAR(64) DEFAULT 'eip',
                    source VARCHAR(128),
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Migration for existing databases - add new columns if missing
            await conn.execute("""
                ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_type VARCHAR(64) DEFAULT 'eip';
                ALTER TABLE documents ADD COLUMN IF NOT EXISTS source VARCHAR(128);
                ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS chunks_document_id_idx
                ON chunks (document_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
                ON chunks USING gin (to_tsvector('english', content))
            """)

            logger.info("initialized_schema")

    async def store_embedded_chunks(
        self,
        embedded_chunks: list[EmbeddedChunk],
        git_commit: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Store embedded chunks in the database.

        Args:
            embedded_chunks: List of chunks with embeddings to store
            git_commit: Optional git commit hash for versioning
            metadata: Optional metadata dict to store with each chunk
        """
        if not embedded_chunks:
            return

        metadata_json = json.dumps(metadata or {})

        async with self.connection() as conn:
            # Use a transaction for batch insert
            async with conn.transaction():
                for ec in embedded_chunks:
                    # Remove null bytes that can occur in PDF extraction
                    clean_content = ec.chunk.content.replace("\x00", "") if ec.chunk.content else ""
                    # Truncate section_path to fit VARCHAR(256)
                    section_path = ec.chunk.section_path
                    if section_path and len(section_path) > 250:
                        section_path = section_path[:247] + "..."
                    # Merge chunk-specific metadata with provided metadata
                    chunk_meta = getattr(ec.chunk, "metadata", {}) or {}
                    merged_meta = {**(metadata or {}), **chunk_meta}
                    await conn.execute(
                        """
                        INSERT INTO chunks (
                            chunk_id, document_id, content, token_count,
                            chunk_index, section_path, embedding, git_commit, metadata
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            git_commit = EXCLUDED.git_commit,
                            metadata = EXCLUDED.metadata
                        """,
                        ec.chunk.chunk_id,
                        ec.chunk.document_id,
                        clean_content,
                        ec.chunk.token_count,
                        ec.chunk.chunk_index,
                        section_path,
                        ec.embedding,
                        git_commit,
                        json.dumps(merged_meta),
                    )

        logger.info("stored_chunks", count=len(embedded_chunks))

    async def store_document(
        self,
        document_id: str,
        eip_number: int,
        title: str,
        status: str,
        type_: str,
        category: str | None,
        author: str,
        created_date: str,
        requires: list[int],
        raw_content: str,
        git_commit: str,
    ):
        """Store document metadata."""
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO documents (
                    document_id, eip_number, title, status, type, category,
                    author, created_date, requires, raw_content, git_commit
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (document_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    status = EXCLUDED.status,
                    type = EXCLUDED.type,
                    category = EXCLUDED.category,
                    author = EXCLUDED.author,
                    requires = EXCLUDED.requires,
                    raw_content = EXCLUDED.raw_content,
                    git_commit = EXCLUDED.git_commit,
                    updated_at = NOW()
                """,
                document_id,
                eip_number,
                title,
                status,
                type_,
                category,
                author,
                created_date,
                requires,
                raw_content,
                git_commit,
            )

    async def store_generic_document(
        self,
        document_id: str,
        document_type: str,
        title: str,
        source: str,
        raw_content: str,
        metadata: dict[str, Any] | None = None,
        author: str | None = None,
        git_commit: str | None = None,
    ) -> None:
        """Store a generic document (forum post, transcript, paper, spec).

        Args:
            document_id: Unique identifier (e.g., "ethresearch-topic-1234")
            document_type: One of: forum_topic, acd_transcript, arxiv_paper, consensus_spec, execution_spec
            title: Document title
            source: Source identifier (e.g., "ethresearch", "ethereum/pm")
            raw_content: Full text content
            metadata: Additional JSON metadata
            author: Author name(s)
            git_commit: Git commit hash if from a repo
        """
        # Remove null bytes that can occur in PDF extraction
        clean_content = raw_content.replace("\x00", "") if raw_content else ""
        clean_title = title.replace("\x00", "") if title else ""
        clean_author = author.replace("\x00", "") if author else None

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO documents (
                    document_id, title, author, raw_content, git_commit,
                    document_type, source, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (document_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    author = EXCLUDED.author,
                    raw_content = EXCLUDED.raw_content,
                    git_commit = EXCLUDED.git_commit,
                    document_type = EXCLUDED.document_type,
                    source = EXCLUDED.source,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                document_id,
                clean_title,
                clean_author,
                clean_content,
                git_commit,
                document_type,
                source,
                json.dumps(metadata or {}),
            )

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        document_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks using cosine similarity."""
        async with self.connection() as conn:
            if document_filter:
                rows = await conn.fetch(
                    """
                    SELECT
                        id, chunk_id, document_id, content, token_count,
                        chunk_index, section_path, embedding, created_at,
                        1 - (embedding <=> $1) as similarity
                    FROM chunks
                    WHERE document_id = $3
                    ORDER BY embedding <=> $1
                    LIMIT $2
                    """,
                    query_embedding,
                    limit,
                    document_filter,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        id, chunk_id, document_id, content, token_count,
                        chunk_index, section_path, embedding, created_at,
                        1 - (embedding <=> $1) as similarity
                    FROM chunks
                    ORDER BY embedding <=> $1
                    LIMIT $2
                    """,
                    query_embedding,
                    limit,
                )

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
                    embedding=list(row["embedding"]),
                    created_at=row["created_at"],
                )
                results.append(SearchResult(
                    chunk=chunk,
                    similarity=row["similarity"],
                ))

            return results

    async def get_document(self, document_id: str) -> dict | None:
        """Get document metadata."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM documents WHERE document_id = $1
                """,
                document_id,
            )
            if row:
                return dict(row)
            return None

    async def get_chunks_by_document(self, document_id: str) -> list[StoredChunk]:
        """Get all chunks for a document."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id, chunk_id, document_id, content, token_count,
                    chunk_index, section_path, embedding, created_at
                FROM chunks
                WHERE document_id = $1
                ORDER BY chunk_index
                """,
                document_id,
            )

            return [
                StoredChunk(
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
                for row in rows
            ]

    async def count_chunks(self) -> int:
        """Count total chunks in database."""
        async with self.connection() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM chunks")
            return row["count"]

    async def count_documents(self) -> int:
        """Count total documents in database."""
        async with self.connection() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM documents")
            return row["count"]

    async def delete_document(self, document_id: str):
        """Delete a document and its chunks."""
        async with self.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM chunks WHERE document_id = $1",
                    document_id,
                )
                await conn.execute(
                    "DELETE FROM documents WHERE document_id = $1",
                    document_id,
                )

    async def clear_all(self):
        """Clear all data (for testing)."""
        async with self.connection() as conn:
            await conn.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
            logger.warning("cleared_all_data")

    async def reindex_embeddings(self):
        """Rebuild the vector similarity index.

        IVFFlat indexes need rebuilding after bulk inserts for new data
        to be properly indexed. Call this after ingestion completes.
        """
        async with self.connection() as conn:
            await conn.execute("REINDEX INDEX chunks_embedding_idx")
            logger.info("reindexed_embeddings")
