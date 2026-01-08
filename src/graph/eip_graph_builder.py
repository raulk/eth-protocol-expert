"""EIP Graph Builder - Build graph from EIP frontmatter metadata."""

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from ..ingestion.eip_parser import ParsedEIP
from .falkordb_store import FalkorDBStore

logger = structlog.get_logger()


@dataclass
class EIPNode:
    """Represents an EIP node in the graph."""

    number: int
    title: str
    status: str
    type: str
    category: str | None = None


@dataclass
class EIPRelationship:
    """Represents a relationship between EIPs."""

    from_eip: int
    to_eip: int
    relationship_type: str  # REQUIRES, SUPERSEDES, REPLACES


@dataclass
class GraphBuildResult:
    """Result of building the EIP graph."""

    nodes_created: int
    relationships_created: int
    requires_count: int
    supersedes_count: int
    replaces_count: int
    errors: list[str] = field(default_factory=list)


class EIPGraphBuilder:
    """Build the EIP dependency graph from parsed EIP data.

    Extracts relationships from frontmatter:
    - requires: EIP-X requires EIP-Y
    - superseded-by: EIP-X is superseded by EIP-Y (stored as Y SUPERSEDES X)
    - replaces: EIP-X replaces EIP-Y (if present)
    """

    def __init__(self, store: FalkorDBStore):
        self.store = store

    def build_from_eips(self, parsed_eips: list[ParsedEIP]) -> GraphBuildResult:
        """Build the graph from a list of parsed EIPs.

        Args:
            parsed_eips: List of ParsedEIP objects with frontmatter data

        Returns:
            GraphBuildResult with statistics
        """
        result = GraphBuildResult(
            nodes_created=0,
            relationships_created=0,
            requires_count=0,
            supersedes_count=0,
            replaces_count=0,
        )

        # First pass: create all nodes
        for eip in parsed_eips:
            try:
                node_result = self.store.create_eip_node(
                    number=eip.eip_number,
                    title=eip.title,
                    status=eip.status,
                    type_=eip.type,
                    category=eip.category,
                )
                result.nodes_created += node_result.nodes_created
            except Exception as e:
                error_msg = f"Failed to create node for EIP-{eip.eip_number}: {e}"
                result.errors.append(error_msg)
                logger.warning("node_creation_failed", eip=eip.eip_number, error=str(e))

        logger.info("created_eip_nodes", count=result.nodes_created)

        # Second pass: create relationships
        for eip in parsed_eips:
            try:
                rels = self._create_relationships_for_eip(eip)
                result.relationships_created += rels["total"]
                result.requires_count += rels["requires"]
                result.supersedes_count += rels["supersedes"]
                result.replaces_count += rels["replaces"]
            except Exception as e:
                error_msg = f"Failed to create relationships for EIP-{eip.eip_number}: {e}"
                result.errors.append(error_msg)
                logger.warning(
                    "relationship_creation_failed", eip=eip.eip_number, error=str(e)
                )

        logger.info(
            "created_relationships",
            total=result.relationships_created,
            requires=result.requires_count,
            supersedes=result.supersedes_count,
            replaces=result.replaces_count,
        )

        return result

    def _create_relationships_for_eip(self, eip: ParsedEIP) -> dict[str, int]:
        """Create all relationships for a single EIP."""
        counts = {"requires": 0, "supersedes": 0, "replaces": 0, "total": 0}

        # Handle 'requires' field
        for required_eip in eip.requires:
            try:
                rel_result = self.store.create_requires_relationship(
                    from_eip=eip.eip_number,
                    to_eip=required_eip,
                )
                counts["requires"] += rel_result.relationships_created
                counts["total"] += rel_result.relationships_created
            except Exception as e:
                logger.debug(
                    "requires_relationship_failed",
                    from_eip=eip.eip_number,
                    to_eip=required_eip,
                    error=str(e),
                )

        # Handle 'superseded-by' field (reverse direction)
        superseded_by = self._parse_eip_references(
            eip.frontmatter.get("superseded-by")
        )
        for superseding_eip in superseded_by:
            try:
                # Note: superseded-by means the OTHER EIP supersedes THIS one
                rel_result = self.store.create_supersedes_relationship(
                    from_eip=superseding_eip,
                    to_eip=eip.eip_number,
                )
                counts["supersedes"] += rel_result.relationships_created
                counts["total"] += rel_result.relationships_created
            except Exception as e:
                logger.debug(
                    "supersedes_relationship_failed",
                    from_eip=superseding_eip,
                    to_eip=eip.eip_number,
                    error=str(e),
                )

        # Handle 'replaces' field (if present)
        replaces = self._parse_eip_references(eip.frontmatter.get("replaces"))
        for replaced_eip in replaces:
            try:
                rel_result = self.store.create_replaces_relationship(
                    from_eip=eip.eip_number,
                    to_eip=replaced_eip,
                )
                counts["replaces"] += rel_result.relationships_created
                counts["total"] += rel_result.relationships_created
            except Exception as e:
                logger.debug(
                    "replaces_relationship_failed",
                    from_eip=eip.eip_number,
                    to_eip=replaced_eip,
                    error=str(e),
                )

        return counts

    def _parse_eip_references(self, value: Any) -> list[int]:
        """Parse EIP references from various frontmatter formats.

        Handles:
        - Single integer: 1559
        - List of integers: [1559, 4844]
        - String with EIP numbers: "EIP-1559, EIP-4844"
        - Comma-separated numbers: "1559, 4844"
        """
        if value is None:
            return []

        if isinstance(value, int):
            return [value]

        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, int):
                    result.append(item)
                elif isinstance(item, str):
                    result.extend(self._extract_numbers(item))
            return result

        if isinstance(value, str):
            return self._extract_numbers(value)

        return []

    def _extract_numbers(self, text: str) -> list[int]:
        """Extract EIP numbers from a string."""
        # Match patterns like "1559", "EIP-1559", "eip1559"
        matches = re.findall(r"(?:eip[- ]?)?(\d+)", text, re.IGNORECASE)
        return [int(m) for m in matches]

    def add_single_eip(self, eip: ParsedEIP) -> dict[str, int]:
        """Add a single EIP and its relationships to the graph.

        Useful for incremental updates.
        """
        # Create node
        self.store.create_eip_node(
            number=eip.eip_number,
            title=eip.title,
            status=eip.status,
            type_=eip.type,
            category=eip.category,
        )

        # Create relationships
        return self._create_relationships_for_eip(eip)

    def get_nodes(self) -> list[EIPNode]:
        """Get all EIP nodes from the graph."""
        result = self.store.query(
            """
            MATCH (e:EIP)
            RETURN e.number, e.title, e.status, e.type, e.category
            ORDER BY e.number
            """
        )
        return [
            EIPNode(
                number=row[0],
                title=row[1],
                status=row[2],
                type=row[3],
                category=row[4],
            )
            for row in result.result_set
        ]

    def get_relationships(
        self, relationship_type: str | None = None
    ) -> list[EIPRelationship]:
        """Get all relationships, optionally filtered by type."""
        if relationship_type:
            cypher = f"""
                MATCH (from:EIP)-[r:{relationship_type}]->(to:EIP)
                RETURN from.number, to.number, type(r)
                ORDER BY from.number
            """
        else:
            cypher = """
                MATCH (from:EIP)-[r]->(to:EIP)
                RETURN from.number, to.number, type(r)
                ORDER BY from.number
            """

        result = self.store.query(cypher)
        return [
            EIPRelationship(
                from_eip=row[0],
                to_eip=row[1],
                relationship_type=row[2],
            )
            for row in result.result_set
        ]

    def get_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the current graph."""
        return {
            "node_count": self.store.count_nodes(),
            "relationship_counts": self.store.count_relationships(),
        }
