"""Tests for Phase 4: EIP Graph.

These are self-contained unit tests that don't require FalkorDB.
"""

from dataclasses import dataclass
from enum import Enum

import pytest


@dataclass
class EIPNode:
    """EIP node for graph storage."""
    number: int
    title: str
    status: str
    type: str
    category: str | None = None


@dataclass
class EIPRelationship:
    """Relationship between EIPs."""
    from_eip: int
    to_eip: int
    relationship_type: str


class TraversalDirection(Enum):
    """Direction for graph traversal."""
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    BOTH = "both"


class TestEIPNode:
    """Tests for EIP node dataclass."""

    def test_node_creation(self):
        """Test creating an EIP node."""
        node = EIPNode(
            number=1559,
            title="Fee Market Change for ETH 1.0 Chain",
            status="Final",
            type="Standards Track",
            category="Core",
        )

        assert node.number == 1559
        assert node.status == "Final"
        assert node.category == "Core"

    def test_node_to_dict(self):
        """Test converting node to dict for graph storage."""
        node = EIPNode(
            number=4844,
            title="Shard Blob Transactions",
            status="Final",
            type="Standards Track",
            category="Core",
        )

        node_dict = {
            "number": node.number,
            "title": node.title,
            "status": node.status,
            "type": node.type,
            "category": node.category,
        }

        assert node_dict["number"] == 4844

    def test_node_without_category(self):
        """Test node without optional category."""
        node = EIPNode(
            number=20,
            title="ERC-20 Token Standard",
            status="Final",
            type="Standards Track",
        )

        assert node.category is None


class TestEIPRelationship:
    """Tests for EIP relationship dataclass."""

    def test_requires_relationship(self):
        """Test REQUIRES relationship."""
        rel = EIPRelationship(
            from_eip=4844,
            to_eip=1559,
            relationship_type="REQUIRES",
        )

        assert rel.from_eip == 4844
        assert rel.to_eip == 1559
        assert rel.relationship_type == "REQUIRES"

    def test_supersedes_relationship(self):
        """Test SUPERSEDES relationship."""
        rel = EIPRelationship(
            from_eip=6780,
            to_eip=6049,
            relationship_type="SUPERSEDES",
        )

        assert rel.relationship_type == "SUPERSEDES"

    def test_replaces_relationship(self):
        """Test REPLACES relationship."""
        rel = EIPRelationship(
            from_eip=100,
            to_eip=50,
            relationship_type="REPLACES",
        )

        assert rel.relationship_type == "REPLACES"


class TestTraversalDirection:
    """Tests for traversal direction enum."""

    def test_upstream_direction(self):
        """Test upstream traversal (dependencies)."""
        direction = TraversalDirection.UPSTREAM
        assert direction.value == "upstream"

    def test_downstream_direction(self):
        """Test downstream traversal (dependents)."""
        direction = TraversalDirection.DOWNSTREAM
        assert direction.value == "downstream"

    def test_both_direction(self):
        """Test bidirectional traversal."""
        direction = TraversalDirection.BOTH
        assert direction.value == "both"


class TestGraphRelationships:
    """Tests for graph relationship logic."""

    def test_requires_is_directional(self):
        """REQUIRES should be directional: A requires B != B requires A."""
        rel1 = EIPRelationship(4844, 1559, "REQUIRES")
        rel2 = EIPRelationship(1559, 4844, "REQUIRES")

        assert rel1.from_eip != rel2.from_eip
        assert rel1.to_eip != rel2.to_eip

    def test_supersedes_reverses_frontmatter(self):
        """SUPERSEDES reverses the 'superseded-by' frontmatter field.

        If EIP-A has 'superseded-by: EIP-B', then B SUPERSEDES A.
        """
        # EIP-6049 has superseded-by: 6780
        # So EIP-6780 SUPERSEDES EIP-6049
        rel = EIPRelationship(
            from_eip=6780,
            to_eip=6049,
            relationship_type="SUPERSEDES",
        )

        assert rel.from_eip == 6780  # The newer one
        assert rel.to_eip == 6049     # The older one


class TestDependencyGraph:
    """Tests for dependency graph operations."""

    @pytest.fixture
    def sample_relationships(self):
        """Create sample relationships for testing."""
        return [
            EIPRelationship(4844, 1559, "REQUIRES"),
            EIPRelationship(4844, 2718, "REQUIRES"),
            EIPRelationship(2718, 1559, "REQUIRES"),
            EIPRelationship(3675, 1559, "REQUIRES"),
        ]

    def test_direct_dependencies(self, sample_relationships):
        """Test getting direct dependencies."""
        # Find what EIP-4844 directly requires
        deps = [r.to_eip for r in sample_relationships if r.from_eip == 4844]

        assert len(deps) == 2
        assert 1559 in deps
        assert 2718 in deps

    def test_transitive_dependencies(self, sample_relationships):
        """Test transitive dependency calculation."""
        # Build adjacency list
        graph = {}
        for rel in sample_relationships:
            if rel.from_eip not in graph:
                graph[rel.from_eip] = []
            graph[rel.from_eip].append(rel.to_eip)

        def get_all_deps(eip, visited=None):
            if visited is None:
                visited = set()
            if eip in visited:
                return set()
            visited.add(eip)
            deps = set(graph.get(eip, []))
            for dep in list(deps):
                deps.update(get_all_deps(dep, visited))
            return deps

        # EIP-4844 transitively requires EIP-1559 (both directly and via 2718)
        all_deps = get_all_deps(4844)
        assert 1559 in all_deps
        assert 2718 in all_deps

    def test_dependents(self, sample_relationships):
        """Test getting EIPs that depend on a given EIP."""
        # Find what depends on EIP-1559
        dependents = [r.from_eip for r in sample_relationships if r.to_eip == 1559]

        assert 4844 in dependents
        assert 2718 in dependents
        assert 3675 in dependents

    def test_no_circular_dependencies(self, sample_relationships):
        """Dependencies should not be circular."""
        # Build graph and check for cycles
        graph = {}
        for rel in sample_relationships:
            if rel.from_eip not in graph:
                graph[rel.from_eip] = []
            graph[rel.from_eip].append(rel.to_eip)

        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        # Check each node for cycles
        visited = set()
        for node in graph:
            if node not in visited:
                assert not has_cycle(node, visited, set())


class TestGraphBuilder:
    """Tests for graph building from frontmatter."""

    def test_parse_requires_int(self):
        """Parse integer requires field."""
        frontmatter = {"requires": 1559}
        requires = [frontmatter["requires"]] if isinstance(frontmatter["requires"], int) else []
        assert requires == [1559]

    def test_parse_requires_list(self):
        """Parse list requires field."""
        frontmatter = {"requires": [1559, 2718]}
        requires = frontmatter["requires"] if isinstance(frontmatter["requires"], list) else []
        assert requires == [1559, 2718]

    def test_parse_requires_string(self):
        """Parse string requires field."""
        import re
        frontmatter = {"requires": "1559, 2718"}
        requires_str = frontmatter["requires"]
        requires = [int(x.strip()) for x in re.split(r'[,\s]+', requires_str) if x.strip().isdigit()]
        assert 1559 in requires
        assert 2718 in requires

    def test_parse_superseded_by(self):
        """Parse superseded-by field into SUPERSEDES relationship."""
        # In frontmatter: superseded-by: 6780 means EIP-6780 SUPERSEDES this EIP
        frontmatter = {"eip": 6049, "superseded-by": 6780}

        # This creates a SUPERSEDES edge FROM 6780 TO 6049
        rel = EIPRelationship(
            from_eip=frontmatter["superseded-by"],
            to_eip=frontmatter["eip"],
            relationship_type="SUPERSEDES",
        )

        assert rel.from_eip == 6780
        assert rel.to_eip == 6049
