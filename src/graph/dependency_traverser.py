"""Dependency Traverser - Traverse EIP requires/supersedes relationships."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from .falkordb_store import FalkorDBStore

logger = structlog.get_logger()


class TraversalDirection(Enum):
    """Direction for dependency traversal."""

    UPSTREAM = "upstream"  # EIPs this one depends on
    DOWNSTREAM = "downstream"  # EIPs that depend on this one
    BOTH = "both"


@dataclass
class DependencyNode:
    """A node in the dependency tree."""

    eip_number: int
    title: str | None = None
    status: str | None = None
    depth: int = 0
    relationship_type: str | None = None  # How we reached this node


@dataclass
class DependencyResult:
    """Result of a dependency traversal."""

    root_eip: int
    direction: TraversalDirection
    max_depth: int

    # Direct dependencies
    direct_dependencies: list[int] = field(default_factory=list)
    direct_dependents: list[int] = field(default_factory=list)

    # Transitive (all, including direct)
    all_dependencies: list[DependencyNode] = field(default_factory=list)
    all_dependents: list[DependencyNode] = field(default_factory=list)

    # Supersedes relationships
    supersedes: list[int] = field(default_factory=list)
    superseded_by: list[int] = field(default_factory=list)

    def get_related_eips(self) -> set[int]:
        """Get all related EIP numbers."""
        related = set()
        related.update(dep.eip_number for dep in self.all_dependencies)
        related.update(dep.eip_number for dep in self.all_dependents)
        related.update(self.supersedes)
        related.update(self.superseded_by)
        related.discard(self.root_eip)
        return related


class DependencyTraverser:
    """Traverse EIP dependency relationships.

    Supports:
    - Direct dependencies (one hop)
    - Transitive dependencies (full chain)
    - Supersedes relationships
    - Configurable depth limits
    """

    def __init__(self, store: FalkorDBStore, max_depth: int = 10):
        self.store = store
        self.max_depth = max_depth

    def get_dependencies(
        self,
        eip_number: int,
        direction: TraversalDirection = TraversalDirection.BOTH,
        max_depth: int | None = None,
        include_supersedes: bool = True,
    ) -> DependencyResult:
        """Get all dependencies for an EIP.

        Args:
            eip_number: The EIP to analyze
            direction: Which direction to traverse
            max_depth: Maximum depth to traverse (default: instance max_depth)
            include_supersedes: Include supersedes relationships

        Returns:
            DependencyResult with all found relationships
        """
        depth = max_depth or self.max_depth

        result = DependencyResult(
            root_eip=eip_number,
            direction=direction,
            max_depth=depth,
        )

        # Get direct relationships first
        result.direct_dependencies = self.store.get_direct_dependencies(eip_number)
        result.direct_dependents = self.store.get_direct_dependents(eip_number)

        # Traverse based on direction
        if direction in (TraversalDirection.UPSTREAM, TraversalDirection.BOTH):
            result.all_dependencies = self._traverse_upstream(eip_number, depth)

        if direction in (TraversalDirection.DOWNSTREAM, TraversalDirection.BOTH):
            result.all_dependents = self._traverse_downstream(eip_number, depth)

        # Get supersedes relationships
        if include_supersedes:
            result.supersedes = self.store.get_supersedes(eip_number)
            result.superseded_by = self.store.get_superseded_by(eip_number)

        logger.debug(
            "traversed_dependencies",
            eip=eip_number,
            direct_deps=len(result.direct_dependencies),
            direct_dependents=len(result.direct_dependents),
            all_deps=len(result.all_dependencies),
            all_dependents=len(result.all_dependents),
        )

        return result

    def _traverse_upstream(
        self,
        eip_number: int,
        max_depth: int,
    ) -> list[DependencyNode]:
        """Traverse upstream (EIPs this one depends on)."""
        cypher = f"""
            MATCH path = (e:EIP {{number: $number}})-[:REQUIRES*1..{max_depth}]->(dep:EIP)
            WITH dep, min(length(path)) as depth
            RETURN DISTINCT dep.number, dep.title, dep.status, depth
            ORDER BY depth, dep.number
        """

        result = self.store.query(cypher, params={"number": eip_number})

        return [
            DependencyNode(
                eip_number=row[0],
                title=row[1],
                status=row[2],
                depth=row[3],
                relationship_type="REQUIRES",
            )
            for row in result.result_set
        ]

    def _traverse_downstream(
        self,
        eip_number: int,
        max_depth: int,
    ) -> list[DependencyNode]:
        """Traverse downstream (EIPs that depend on this one)."""
        cypher = f"""
            MATCH path = (dep:EIP)-[:REQUIRES*1..{max_depth}]->(e:EIP {{number: $number}})
            WITH dep, min(length(path)) as depth
            RETURN DISTINCT dep.number, dep.title, dep.status, depth
            ORDER BY depth, dep.number
        """

        result = self.store.query(cypher, params={"number": eip_number})

        return [
            DependencyNode(
                eip_number=row[0],
                title=row[1],
                status=row[2],
                depth=row[3],
                relationship_type="REQUIRES",
            )
            for row in result.result_set
        ]

    def get_dependency_chain(
        self,
        from_eip: int,
        to_eip: int,
    ) -> list[int] | None:
        """Find the shortest dependency path between two EIPs.

        Returns:
            List of EIP numbers in the path, or None if no path exists
        """
        cypher = """
            MATCH path = shortestPath(
                (from:EIP {number: $from_eip})-[:REQUIRES*]->(to:EIP {number: $to_eip})
            )
            RETURN [node in nodes(path) | node.number]
        """

        result = self.store.query(
            cypher,
            params={"from_eip": from_eip, "to_eip": to_eip},
        )

        if result.result_set:
            return result.result_set[0][0]
        return None

    def get_dependency_tree(
        self,
        eip_number: int,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """Get a tree representation of dependencies.

        Returns a nested dict suitable for visualization.
        """
        eip_data = self.store.get_eip(eip_number)
        if not eip_data:
            return {"error": f"EIP-{eip_number} not found"}

        return self._build_tree(eip_number, max_depth, visited=set())

    def _build_tree(
        self,
        eip_number: int,
        remaining_depth: int,
        visited: set[int],
    ) -> dict[str, Any]:
        """Recursively build dependency tree."""
        if eip_number in visited or remaining_depth <= 0:
            return {"eip": eip_number, "truncated": True}

        visited.add(eip_number)

        eip_data = self.store.get_eip(eip_number)
        dependencies = self.store.get_direct_dependencies(eip_number)

        tree = {
            "eip": eip_number,
            "title": eip_data.get("title") if eip_data else None,
            "status": eip_data.get("status") if eip_data else None,
            "dependencies": [],
        }

        for dep_number in dependencies:
            child_tree = self._build_tree(
                dep_number,
                remaining_depth - 1,
                visited.copy(),
            )
            tree["dependencies"].append(child_tree)

        return tree

    def find_common_dependencies(
        self,
        eip_numbers: list[int],
    ) -> list[int]:
        """Find EIPs that all given EIPs depend on.

        Useful for finding foundational EIPs.
        """
        if not eip_numbers:
            return []

        if len(eip_numbers) == 1:
            deps = self._traverse_upstream(eip_numbers[0], self.max_depth)
            return [d.eip_number for d in deps]

        # Get dependencies for all EIPs
        dep_sets = []
        for eip in eip_numbers:
            deps = self._traverse_upstream(eip, self.max_depth)
            dep_sets.append(set(d.eip_number for d in deps))

        # Find intersection
        common = dep_sets[0]
        for dep_set in dep_sets[1:]:
            common = common.intersection(dep_set)

        return sorted(common)

    def get_most_depended_upon(self, limit: int = 10) -> list[tuple[int, int]]:
        """Get EIPs with the most dependents.

        Returns:
            List of (eip_number, dependent_count) tuples
        """
        cypher = """
            MATCH (dep:EIP)-[:REQUIRES]->(e:EIP)
            WITH e.number as eip, count(dep) as dependents
            ORDER BY dependents DESC
            LIMIT $limit
            RETURN eip, dependents
        """

        result = self.store.query(cypher, params={"limit": limit})
        return [(row[0], row[1]) for row in result.result_set]

    def get_leaf_eips(self) -> list[int]:
        """Get EIPs with no dependencies (leaf nodes)."""
        cypher = """
            MATCH (e:EIP)
            WHERE NOT (e)-[:REQUIRES]->()
            RETURN e.number
            ORDER BY e.number
        """

        result = self.store.query(cypher)
        return [row[0] for row in result.result_set]

    def get_root_eips(self) -> list[int]:
        """Get EIPs with no dependents (root nodes)."""
        cypher = """
            MATCH (e:EIP)
            WHERE NOT ()-[:REQUIRES]->(e)
            RETURN e.number
            ORDER BY e.number
        """

        result = self.store.query(cypher)
        return [row[0] for row in result.result_set]
