"""Selective Traverser - Traverse graph following only high-confidence edges (Phase 11)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

import structlog

from src.graph.falkordb_store import FalkorDBStore
from src.graph.relationship_inferrer import RelationshipType

logger = structlog.get_logger()


class TraversalDirection(Enum):
    """Direction for traversal."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


@dataclass
class TraversalConfig:
    """Configuration for selective traversal."""

    min_confidence: float = 0.7
    max_depth: int = 3
    include_explicit: bool = True
    relationship_types: list[RelationshipType] | None = None


@dataclass
class RelationshipStep:
    """A single step in a traversal path."""

    from_eip: int
    to_eip: int
    relationship_type: str
    confidence: float
    is_inferred: bool


@dataclass
class TraversalResult:
    """Result from a selective traversal."""

    eip_number: int
    path: list[int]
    confidence: float
    relationship_chain: list[RelationshipStep]
    depth: int

    @property
    def path_string(self) -> str:
        """Human-readable path representation."""
        if not self.relationship_chain:
            return str(self.eip_number)

        parts = [str(self.relationship_chain[0].from_eip)]
        for step in self.relationship_chain:
            rel_type = step.relationship_type.replace("_", " ")
            parts.append(f"-[{rel_type}]->")
            parts.append(str(step.to_eip))
        return " ".join(parts)


@dataclass
class TraversalSummary:
    """Summary of a complete traversal operation."""

    root_eip: int
    direction: TraversalDirection
    results: list[TraversalResult] = field(default_factory=list)
    explicit_count: int = 0
    inferred_count: int = 0
    max_depth_reached: int = 0

    def get_all_eips(self) -> set[int]:
        """Get all EIP numbers found during traversal."""
        eips = {self.root_eip}
        for result in self.results:
            eips.update(result.path)
        return eips


class SelectiveTraverser:
    """Traverse the EIP graph following only high-confidence edges.

    Combines explicit relationships (from frontmatter) and inferred
    relationships (from LLM analysis) while respecting confidence thresholds.
    """

    EXPLICIT_RELATIONSHIP_TYPES: ClassVar[set[str]] = {
        "REQUIRES",
        "SUPERSEDES",
        "REPLACES",
        "DISCUSSES",
    }

    INFERRED_RELATIONSHIP_TYPES: ClassVar[set[str]] = {
        "CONFLICTS_WITH",
        "ADDRESSES_CONCERN",
        "INSPIRED_BY",
        "BUILDS_ON",
        "ALTERNATIVE_TO",
        "RELATED_TO",
    }

    def __init__(
        self,
        store: FalkorDBStore,
        config: TraversalConfig | None = None,
    ):
        self.store = store
        self.config = config or TraversalConfig()

    def traverse(
        self,
        eip_number: int,
        direction: str | TraversalDirection = "both",
    ) -> TraversalSummary:
        """Traverse the graph from a starting EIP.

        Args:
            eip_number: Starting EIP number
            direction: "outgoing", "incoming", or "both"

        Returns:
            TraversalSummary with all found paths
        """
        if isinstance(direction, str):
            direction = TraversalDirection(direction)

        summary = TraversalSummary(
            root_eip=eip_number,
            direction=direction,
        )

        visited: set[int] = {eip_number}
        queue: list[tuple[int, list[int], list[RelationshipStep], float]] = [
            (eip_number, [eip_number], [], 1.0)
        ]

        while queue:
            current, path, chain, path_confidence = queue.pop(0)
            current_depth = len(chain)

            if current_depth >= self.config.max_depth:
                continue

            neighbors = self._get_neighbors(current, direction, path_confidence)

            for neighbor_eip, step in neighbors:
                if neighbor_eip in visited:
                    continue

                new_confidence = path_confidence * step.confidence
                if new_confidence < self.config.min_confidence:
                    continue

                visited.add(neighbor_eip)
                new_path = [*path, neighbor_eip]
                new_chain = [*chain, step]

                result = TraversalResult(
                    eip_number=neighbor_eip,
                    path=new_path,
                    confidence=new_confidence,
                    relationship_chain=new_chain,
                    depth=len(new_chain),
                )
                summary.results.append(result)

                if step.is_inferred:
                    summary.inferred_count += 1
                else:
                    summary.explicit_count += 1

                summary.max_depth_reached = max(summary.max_depth_reached, len(new_chain))

                if len(new_chain) < self.config.max_depth:
                    queue.append((neighbor_eip, new_path, new_chain, new_confidence))

        logger.debug(
            "traversal_complete",
            root_eip=eip_number,
            direction=direction.value,
            results=len(summary.results),
            explicit=summary.explicit_count,
            inferred=summary.inferred_count,
        )

        return summary

    def find_path(
        self,
        from_eip: int,
        to_eip: int,
        max_depth: int | None = None,
    ) -> TraversalResult | None:
        """Find a path between two EIPs following high-confidence edges.

        Args:
            from_eip: Starting EIP
            to_eip: Target EIP
            max_depth: Maximum path length

        Returns:
            TraversalResult if path found, None otherwise
        """
        depth = max_depth or self.config.max_depth
        visited: set[int] = {from_eip}
        queue: list[tuple[int, list[int], list[RelationshipStep], float]] = [
            (from_eip, [from_eip], [], 1.0)
        ]

        while queue:
            current, path, chain, path_confidence = queue.pop(0)

            if current == to_eip:
                return TraversalResult(
                    eip_number=to_eip,
                    path=path,
                    confidence=path_confidence,
                    relationship_chain=chain,
                    depth=len(chain),
                )

            if len(chain) >= depth:
                continue

            neighbors = self._get_neighbors(current, TraversalDirection.OUTGOING, path_confidence)

            for neighbor_eip, step in neighbors:
                if neighbor_eip in visited:
                    continue

                new_confidence = path_confidence * step.confidence
                if new_confidence < self.config.min_confidence:
                    continue

                visited.add(neighbor_eip)
                new_path = [*path, neighbor_eip]
                new_chain = [*chain, step]
                queue.append((neighbor_eip, new_path, new_chain, new_confidence))

        return None

    def get_related_by_type(
        self,
        eip_number: int,
        relationship_type: RelationshipType,
        direction: TraversalDirection = TraversalDirection.BOTH,
    ) -> list[TraversalResult]:
        """Get EIPs related by a specific relationship type.

        Args:
            eip_number: Starting EIP
            relationship_type: Type of relationship to follow
            direction: Direction to traverse

        Returns:
            List of related EIPs with confidence scores
        """
        original_types = self.config.relationship_types
        self.config.relationship_types = [relationship_type]

        try:
            summary = self.traverse(eip_number, direction)
            return [
                r
                for r in summary.results
                if any(
                    s.relationship_type.lower() == relationship_type.value.lower()
                    for s in r.relationship_chain
                )
            ]
        finally:
            self.config.relationship_types = original_types

    def _get_neighbors(
        self,
        eip_number: int,
        direction: TraversalDirection,
        current_confidence: float,
    ) -> list[tuple[int, RelationshipStep]]:
        """Get neighboring EIPs with confidence filtering."""
        neighbors: list[tuple[int, RelationshipStep]] = []

        if self.config.include_explicit:
            explicit = self._get_explicit_neighbors(eip_number, direction)
            neighbors.extend(explicit)

        inferred = self._get_inferred_neighbors(eip_number, direction, current_confidence)
        neighbors.extend(inferred)

        return neighbors

    def _get_explicit_neighbors(
        self,
        eip_number: int,
        direction: TraversalDirection,
    ) -> list[tuple[int, RelationshipStep]]:
        """Get neighbors via explicit relationships."""
        neighbors = []

        rel_type_filter = ""
        if self.config.relationship_types:
            type_names = [r.value.upper() for r in self.config.relationship_types]
            explicit_types = [t for t in type_names if t in self.EXPLICIT_RELATIONSHIP_TYPES]
            if explicit_types:
                rel_type_filter = f"WHERE type(r) IN {explicit_types}"
            else:
                return []

        if direction in (TraversalDirection.OUTGOING, TraversalDirection.BOTH):
            cypher = f"""
                MATCH (e:EIP {{number: $number}})-[r]->(neighbor:EIP)
                {rel_type_filter}
                RETURN neighbor.number, type(r)
            """
            result = self.store.query(cypher, params={"number": eip_number})
            for row in result.result_set:
                step = RelationshipStep(
                    from_eip=eip_number,
                    to_eip=row[0],
                    relationship_type=row[1],
                    confidence=1.0,
                    is_inferred=False,
                )
                neighbors.append((row[0], step))

        if direction in (TraversalDirection.INCOMING, TraversalDirection.BOTH):
            cypher = f"""
                MATCH (neighbor:EIP)-[r]->(e:EIP {{number: $number}})
                {rel_type_filter}
                RETURN neighbor.number, type(r)
            """
            result = self.store.query(cypher, params={"number": eip_number})
            for row in result.result_set:
                step = RelationshipStep(
                    from_eip=row[0],
                    to_eip=eip_number,
                    relationship_type=row[1],
                    confidence=1.0,
                    is_inferred=False,
                )
                neighbors.append((row[0], step))

        return neighbors

    def _get_inferred_neighbors(
        self,
        eip_number: int,
        direction: TraversalDirection,
        current_confidence: float,
    ) -> list[tuple[int, RelationshipStep]]:
        """Get neighbors via inferred relationships with confidence filtering."""
        neighbors = []
        min_stored_confidence = self.config.min_confidence / max(current_confidence, 0.01)

        rel_type_filter = ""
        if self.config.relationship_types:
            type_names = [r.value.upper() for r in self.config.relationship_types]
            inferred_types = [t for t in type_names if t in self.INFERRED_RELATIONSHIP_TYPES]
            if inferred_types:
                rel_type_filter = f"AND r.relationship_type IN {inferred_types}"
            else:
                return []

        if direction in (TraversalDirection.OUTGOING, TraversalDirection.BOTH):
            cypher = f"""
                MATCH (e:EIP {{number: $number}})-[r:INFERRED]->(neighbor:EIP)
                WHERE r.confidence >= $min_confidence
                {rel_type_filter}
                RETURN neighbor.number, r.relationship_type, r.confidence
            """
            result = self.store.query(
                cypher,
                params={"number": eip_number, "min_confidence": min_stored_confidence},
            )
            for row in result.result_set:
                step = RelationshipStep(
                    from_eip=eip_number,
                    to_eip=row[0],
                    relationship_type=row[1],
                    confidence=row[2],
                    is_inferred=True,
                )
                neighbors.append((row[0], step))

        if direction in (TraversalDirection.INCOMING, TraversalDirection.BOTH):
            cypher = f"""
                MATCH (neighbor:EIP)-[r:INFERRED]->(e:EIP {{number: $number}})
                WHERE r.confidence >= $min_confidence
                {rel_type_filter}
                RETURN neighbor.number, r.relationship_type, r.confidence
            """
            result = self.store.query(
                cypher,
                params={"number": eip_number, "min_confidence": min_stored_confidence},
            )
            for row in result.result_set:
                step = RelationshipStep(
                    from_eip=row[0],
                    to_eip=eip_number,
                    relationship_type=row[1],
                    confidence=row[2],
                    is_inferred=True,
                )
                neighbors.append((row[0], step))

        return neighbors
