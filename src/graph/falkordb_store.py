"""FalkorDB Store - Connection and query wrapper for graph database."""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import structlog
from falkordb import FalkorDB, Graph

logger = structlog.get_logger()


@dataclass
class QueryResult:
    """Result from a graph query."""

    result_set: list[list[Any]]
    nodes_created: int
    relationships_created: int
    nodes_deleted: int
    relationships_deleted: int
    properties_set: int
    run_time_ms: float


class FalkorDBStore:
    """FalkorDB connection and query wrapper.

    Provides a clean interface for graph operations on EIP relationships.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        password: str | None = None,
        graph_name: str = "eip_graph",
    ):
        self.host = host or os.environ.get("FALKORDB_HOST", "localhost")
        self.port = port or int(os.environ.get("FALKORDB_PORT", "6379"))
        self.password = password or os.environ.get("FALKORDB_PASSWORD")
        self.graph_name = graph_name

        self._db: FalkorDB | None = None
        self._graph: Graph | None = None

    def connect(self):
        """Establish connection to FalkorDB."""
        if self.password:
            self._db = FalkorDB(
                host=self.host,
                port=self.port,
                password=self.password,
            )
        else:
            self._db = FalkorDB(host=self.host, port=self.port)

        self._graph = self._db.select_graph(self.graph_name)
        logger.info(
            "connected_to_falkordb",
            host=self.host,
            port=self.port,
            graph=self.graph_name,
        )

    def close(self):
        """Close connection to FalkorDB."""
        self._db = None
        self._graph = None
        logger.info("closed_falkordb_connection")

    @contextmanager
    def connection(self):
        """Context manager for graph operations."""
        if self._graph is None:
            self.connect()
        try:
            yield self._graph
        finally:
            pass  # Keep connection open for reuse

    @property
    def graph(self) -> Graph:
        """Get the graph instance, connecting if necessary."""
        if self._graph is None:
            self.connect()
        return self._graph

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> QueryResult:
        """Execute a Cypher query.

        Args:
            cypher: Cypher query string
            params: Optional query parameters

        Returns:
            QueryResult with results and statistics
        """
        result = self.graph.query(cypher, params=params or {})

        return QueryResult(
            result_set=result.result_set,
            nodes_created=result.nodes_created,
            relationships_created=result.relationships_created,
            nodes_deleted=result.nodes_deleted,
            relationships_deleted=result.relationships_deleted,
            properties_set=result.properties_set,
            run_time_ms=result.run_time_ms,
        )

    def initialize_schema(self):
        """Create indexes for efficient graph queries."""
        # Create index on EIP number for fast lookups
        try:
            self.query("CREATE INDEX FOR (e:EIP) ON (e.number)")
            logger.info("created_eip_number_index")
        except Exception as e:
            # Index may already exist
            if "already indexed" not in str(e).lower():
                logger.warning("failed_to_create_index", error=str(e))

        # Create index on EIP status for filtering
        try:
            self.query("CREATE INDEX FOR (e:EIP) ON (e.status)")
            logger.info("created_eip_status_index")
        except Exception as e:
            if "already indexed" not in str(e).lower():
                logger.warning("failed_to_create_index", error=str(e))

    def create_eip_node(
        self,
        number: int,
        title: str,
        status: str,
        type_: str,
        category: str | None = None,
    ) -> QueryResult:
        """Create an EIP node in the graph.

        Uses MERGE to avoid duplicates.
        """
        cypher = """
            MERGE (e:EIP {number: $number})
            ON CREATE SET
                e.title = $title,
                e.status = $status,
                e.type = $type,
                e.category = $category
            ON MATCH SET
                e.title = $title,
                e.status = $status,
                e.type = $type,
                e.category = $category
            RETURN e
        """
        return self.query(
            cypher,
            params={
                "number": number,
                "title": title,
                "status": status,
                "type": type_,
                "category": category,
            },
        )

    def create_requires_relationship(
        self,
        from_eip: int,
        to_eip: int,
    ) -> QueryResult:
        """Create a REQUIRES relationship between EIPs.

        from_eip REQUIRES to_eip means from_eip depends on to_eip.
        """
        cypher = """
            MATCH (from:EIP {number: $from_eip})
            MATCH (to:EIP {number: $to_eip})
            MERGE (from)-[r:REQUIRES]->(to)
            RETURN from, r, to
        """
        return self.query(
            cypher,
            params={"from_eip": from_eip, "to_eip": to_eip},
        )

    def create_supersedes_relationship(
        self,
        from_eip: int,
        to_eip: int,
    ) -> QueryResult:
        """Create a SUPERSEDES relationship between EIPs.

        from_eip SUPERSEDES to_eip means from_eip replaces to_eip.
        Note: This is the reverse direction of the 'superseded-by' frontmatter field.
        """
        cypher = """
            MATCH (from:EIP {number: $from_eip})
            MATCH (to:EIP {number: $to_eip})
            MERGE (from)-[r:SUPERSEDES]->(to)
            RETURN from, r, to
        """
        return self.query(
            cypher,
            params={"from_eip": from_eip, "to_eip": to_eip},
        )

    def create_replaces_relationship(
        self,
        from_eip: int,
        to_eip: int,
    ) -> QueryResult:
        """Create a REPLACES relationship between EIPs.

        from_eip REPLACES to_eip.
        """
        cypher = """
            MATCH (from:EIP {number: $from_eip})
            MATCH (to:EIP {number: $to_eip})
            MERGE (from)-[r:REPLACES]->(to)
            RETURN from, r, to
        """
        return self.query(
            cypher,
            params={"from_eip": from_eip, "to_eip": to_eip},
        )

    def get_eip(self, number: int) -> dict[str, Any] | None:
        """Get an EIP node by number."""
        result = self.query(
            "MATCH (e:EIP {number: $number}) RETURN e",
            params={"number": number},
        )
        if result.result_set:
            node = result.result_set[0][0]
            return dict(node.properties)
        return None

    def get_direct_dependencies(self, eip_number: int) -> list[int]:
        """Get EIPs that this EIP directly requires."""
        result = self.query(
            """
            MATCH (e:EIP {number: $number})-[:REQUIRES]->(dep:EIP)
            RETURN dep.number
            """,
            params={"number": eip_number},
        )
        return [row[0] for row in result.result_set]

    def get_direct_dependents(self, eip_number: int) -> list[int]:
        """Get EIPs that directly require this EIP."""
        result = self.query(
            """
            MATCH (dep:EIP)-[:REQUIRES]->(e:EIP {number: $number})
            RETURN dep.number
            """,
            params={"number": eip_number},
        )
        return [row[0] for row in result.result_set]

    def get_superseded_by(self, eip_number: int) -> list[int]:
        """Get EIPs that supersede this EIP."""
        result = self.query(
            """
            MATCH (newer:EIP)-[:SUPERSEDES]->(e:EIP {number: $number})
            RETURN newer.number
            """,
            params={"number": eip_number},
        )
        return [row[0] for row in result.result_set]

    def get_supersedes(self, eip_number: int) -> list[int]:
        """Get EIPs that this EIP supersedes."""
        result = self.query(
            """
            MATCH (e:EIP {number: $number})-[:SUPERSEDES]->(older:EIP)
            RETURN older.number
            """,
            params={"number": eip_number},
        )
        return [row[0] for row in result.result_set]

    def count_nodes(self) -> int:
        """Count total EIP nodes in the graph."""
        result = self.query("MATCH (e:EIP) RETURN count(e)")
        return result.result_set[0][0] if result.result_set else 0

    def count_relationships(self) -> dict[str, int]:
        """Count relationships by type."""
        counts = {}
        for rel_type in ["REQUIRES", "SUPERSEDES", "REPLACES"]:
            result = self.query(
                f"MATCH ()-[r:{rel_type}]->() RETURN count(r)"
            )
            counts[rel_type] = result.result_set[0][0] if result.result_set else 0
        return counts

    def clear_graph(self):
        """Delete all nodes and relationships (for testing)."""
        self.query("MATCH (n) DETACH DELETE n")
        logger.warning("cleared_graph_data")
