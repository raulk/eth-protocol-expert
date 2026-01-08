"""PostgreSQL Schema Migrations - Manage database schema updates."""

from dataclasses import dataclass

import asyncpg
import structlog

logger = structlog.get_logger()


@dataclass
class Migration:
    """A database migration."""

    version: int
    name: str
    sql: str


# Schema migrations for the eth-protocol-expert database
MIGRATIONS = [
    Migration(
        version=1,
        name="add_superseded_by_and_replaces",
        sql="""
        -- Add superseded_by column to documents table
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS superseded_by INTEGER;

        -- Add replaces column to documents table (for EIPs that replace others)
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS replaces INTEGER[];

        -- Add index for superseded_by lookups
        CREATE INDEX IF NOT EXISTS documents_superseded_by_idx
        ON documents (superseded_by) WHERE superseded_by IS NOT NULL;

        -- Add description field for longer EIP descriptions
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS description TEXT;

        -- Add last_call_deadline for EIPs in Last Call status
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS last_call_deadline VARCHAR(32);

        -- Add withdrawal_reason for withdrawn EIPs
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS withdrawal_reason TEXT;
        """,
    ),
    Migration(
        version=2,
        name="add_full_text_search_config",
        sql="""
        -- Create a custom text search configuration for EIP content
        -- This is optional but can improve search quality for technical content

        -- Add a tsvector column for precomputed search vectors (performance optimization)
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector tsvector;

        -- Create trigger to auto-update search_vector on insert/update
        CREATE OR REPLACE FUNCTION chunks_search_vector_update() RETURNS trigger AS $$
        BEGIN
            NEW.search_vector := to_tsvector('english', COALESCE(NEW.content, ''));
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql;

        -- Drop trigger if exists, then create
        DROP TRIGGER IF EXISTS chunks_search_vector_trigger ON chunks;

        CREATE TRIGGER chunks_search_vector_trigger
        BEFORE INSERT OR UPDATE OF content ON chunks
        FOR EACH ROW EXECUTE FUNCTION chunks_search_vector_update();

        -- Create GIN index on the search_vector column
        CREATE INDEX IF NOT EXISTS chunks_search_vector_idx
        ON chunks USING gin (search_vector);

        -- Backfill existing rows
        UPDATE chunks SET search_vector = to_tsvector('english', content)
        WHERE search_vector IS NULL;
        """,
    ),
    Migration(
        version=3,
        name="add_document_metadata_indexes",
        sql="""
        -- Add indexes for common metadata queries

        -- Index for status filtering (Final, Draft, etc.)
        CREATE INDEX IF NOT EXISTS documents_status_idx ON documents (status);

        -- Index for type filtering (Standards Track, Meta, Informational)
        CREATE INDEX IF NOT EXISTS documents_type_idx ON documents (type);

        -- Index for category filtering (Core, ERC, etc.)
        CREATE INDEX IF NOT EXISTS documents_category_idx ON documents (category)
        WHERE category IS NOT NULL;

        -- Index for EIP number lookups
        CREATE INDEX IF NOT EXISTS documents_eip_number_idx ON documents (eip_number);

        -- GIN index for requires array containment queries
        CREATE INDEX IF NOT EXISTS documents_requires_idx ON documents USING gin (requires);

        -- Full-text search on author field
        CREATE INDEX IF NOT EXISTS documents_author_fts_idx
        ON documents USING gin (to_tsvector('english', author));
        """,
    ),
]


class SchemaMigrator:
    """Manage database schema migrations."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    async def ensure_migrations_table(self, conn: asyncpg.Connection):
        """Create migrations tracking table if it doesn't exist."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name VARCHAR(256) NOT NULL,
                applied_at TIMESTAMP DEFAULT NOW()
            )
        """)

    async def get_applied_versions(self, conn: asyncpg.Connection) -> set[int]:
        """Get set of already applied migration versions."""
        rows = await conn.fetch("SELECT version FROM schema_migrations")
        return {row["version"] for row in rows}

    async def apply_migration(
        self,
        conn: asyncpg.Connection,
        migration: Migration,
    ):
        """Apply a single migration."""
        async with conn.transaction():
            await conn.execute(migration.sql)
            await conn.execute(
                """
                INSERT INTO schema_migrations (version, name)
                VALUES ($1, $2)
                """,
                migration.version,
                migration.name,
            )

        logger.info(
            "applied_migration",
            version=migration.version,
            name=migration.name,
        )

    async def migrate(self) -> list[Migration]:
        """Apply all pending migrations.

        Returns:
            List of migrations that were applied
        """
        conn = await asyncpg.connect(self.database_url)
        try:
            await self.ensure_migrations_table(conn)
            applied = await self.get_applied_versions(conn)

            applied_migrations = []
            for migration in MIGRATIONS:
                if migration.version not in applied:
                    await self.apply_migration(conn, migration)
                    applied_migrations.append(migration)

            if applied_migrations:
                logger.info(
                    "migrations_complete",
                    applied_count=len(applied_migrations),
                )
            else:
                logger.debug("no_pending_migrations")

            return applied_migrations

        finally:
            await conn.close()

    async def get_current_version(self) -> int | None:
        """Get the current schema version."""
        conn = await asyncpg.connect(self.database_url)
        try:
            await self.ensure_migrations_table(conn)
            row = await conn.fetchrow(
                "SELECT MAX(version) as version FROM schema_migrations"
            )
            return row["version"] if row else None
        finally:
            await conn.close()

    async def pending_migrations(self) -> list[Migration]:
        """Get list of migrations that haven't been applied yet."""
        conn = await asyncpg.connect(self.database_url)
        try:
            await self.ensure_migrations_table(conn)
            applied = await self.get_applied_versions(conn)

            return [m for m in MIGRATIONS if m.version not in applied]
        finally:
            await conn.close()


async def run_migrations(database_url: str) -> list[Migration]:
    """Convenience function to run all pending migrations.

    Args:
        database_url: PostgreSQL connection string

    Returns:
        List of applied migrations
    """
    migrator = SchemaMigrator(database_url)
    return await migrator.migrate()
