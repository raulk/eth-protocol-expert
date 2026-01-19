"""Dependency View - Graph visualization data for EIP dependencies."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from src.graph.dependency_traverser import DependencyTraverser, TraversalDirection
from src.graph.falkordb_store import FalkorDBStore

logger = structlog.get_logger()


class NodeType(Enum):
    """Type of node in the dependency graph."""

    ROOT = "root"
    DEPENDENCY = "dependency"
    DEPENDENT = "dependent"
    SUPERSEDES = "supersedes"
    SUPERSEDED_BY = "superseded_by"


class LayoutHint(Enum):
    """Hints for graph layout algorithms."""

    HIERARCHICAL = "hierarchical"
    FORCE_DIRECTED = "force_directed"
    RADIAL = "radial"
    TREE = "tree"


@dataclass
class GraphNode:
    """A node in the dependency graph."""

    id: str
    label: str
    type: NodeType
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type.value,
            "metadata": self.metadata,
        }


@dataclass
class GraphEdge:
    """An edge in the dependency graph."""

    source: str
    target: str
    relationship: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
            "weight": self.weight,
        }


@dataclass
class DependencyView:
    """Graph visualization data for dependencies."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    layout_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "layout_hints": self.layout_hints,
            "statistics": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
            },
        }

    def to_d3_format(self) -> dict[str, Any]:
        """Convert to D3.js compatible format."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "group": n.type.value,
                    **n.metadata,
                }
                for n in self.nodes
            ],
            "links": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.relationship,
                    "value": e.weight,
                }
                for e in self.edges
            ],
        }

    def to_cytoscape_format(self) -> dict[str, Any]:
        """Convert to Cytoscape.js compatible format."""
        elements = []

        for node in self.nodes:
            elements.append(
                {
                    "data": {
                        "id": node.id,
                        "label": node.label,
                        "type": node.type.value,
                        **node.metadata,
                    },
                    "group": "nodes",
                }
            )

        for edge in self.edges:
            elements.append(
                {
                    "data": {
                        "source": edge.source,
                        "target": edge.target,
                        "relationship": edge.relationship,
                        "weight": edge.weight,
                    },
                    "group": "edges",
                }
            )

        return {"elements": elements}


class DependencyViewBuilder:
    """Build graph visualization data for EIP dependencies.

    Uses the existing DependencyTraverser to gather relationships and
    formats them for visualization libraries like D3.js or Cytoscape.
    """

    def __init__(
        self,
        store: FalkorDBStore,
        traverser: DependencyTraverser | None = None,
    ):
        self.store = store
        self.traverser = traverser or DependencyTraverser(store)

    def build(
        self,
        eip_number: int,
        depth: int = 3,
        include_supersedes: bool = True,
        layout: LayoutHint = LayoutHint.HIERARCHICAL,
    ) -> DependencyView:
        """Build a dependency view for an EIP.

        Args:
            eip_number: The root EIP to analyze
            depth: How many levels of dependencies to include
            include_supersedes: Include supersedes relationships
            layout: Suggested layout algorithm

        Returns:
            DependencyView with nodes, edges, and layout hints
        """
        result = self.traverser.get_dependencies(
            eip_number=eip_number,
            direction=TraversalDirection.BOTH,
            max_depth=depth,
            include_supersedes=include_supersedes,
        )

        view = DependencyView()

        root_eip_data = self.store.get_eip(eip_number)
        root_node = GraphNode(
            id=f"eip-{eip_number}",
            label=f"EIP-{eip_number}",
            type=NodeType.ROOT,
            metadata={
                "title": root_eip_data.get("title") if root_eip_data else None,
                "status": root_eip_data.get("status") if root_eip_data else None,
                "eip_number": eip_number,
            },
        )
        view.nodes.append(root_node)

        node_ids = {f"eip-{eip_number}"}

        for dep in result.all_dependencies:
            node_id = f"eip-{dep.eip_number}"
            if node_id not in node_ids:
                node_ids.add(node_id)
                view.nodes.append(
                    GraphNode(
                        id=node_id,
                        label=f"EIP-{dep.eip_number}",
                        type=NodeType.DEPENDENCY,
                        metadata={
                            "title": dep.title,
                            "status": dep.status,
                            "depth": dep.depth,
                            "eip_number": dep.eip_number,
                        },
                    )
                )

        for dep in result.all_dependents:
            node_id = f"eip-{dep.eip_number}"
            if node_id not in node_ids:
                node_ids.add(node_id)
                view.nodes.append(
                    GraphNode(
                        id=node_id,
                        label=f"EIP-{dep.eip_number}",
                        type=NodeType.DEPENDENT,
                        metadata={
                            "title": dep.title,
                            "status": dep.status,
                            "depth": dep.depth,
                            "eip_number": dep.eip_number,
                        },
                    )
                )

        if include_supersedes:
            for sup_num in result.supersedes:
                node_id = f"eip-{sup_num}"
                if node_id not in node_ids:
                    node_ids.add(node_id)
                    sup_data = self.store.get_eip(sup_num)
                    view.nodes.append(
                        GraphNode(
                            id=node_id,
                            label=f"EIP-{sup_num}",
                            type=NodeType.SUPERSEDES,
                            metadata={
                                "title": sup_data.get("title") if sup_data else None,
                                "status": sup_data.get("status") if sup_data else None,
                                "eip_number": sup_num,
                            },
                        )
                    )

            for sup_num in result.superseded_by:
                node_id = f"eip-{sup_num}"
                if node_id not in node_ids:
                    node_ids.add(node_id)
                    sup_data = self.store.get_eip(sup_num)
                    view.nodes.append(
                        GraphNode(
                            id=node_id,
                            label=f"EIP-{sup_num}",
                            type=NodeType.SUPERSEDED_BY,
                            metadata={
                                "title": sup_data.get("title") if sup_data else None,
                                "status": sup_data.get("status") if sup_data else None,
                                "eip_number": sup_num,
                            },
                        )
                    )

        root_id = f"eip-{eip_number}"
        for dep_num in result.direct_dependencies:
            view.edges.append(
                GraphEdge(
                    source=root_id,
                    target=f"eip-{dep_num}",
                    relationship="REQUIRES",
                    weight=1.0,
                )
            )

        for dep in result.all_dependencies:
            if dep.depth > 1:
                parent_deps = self.store.get_direct_dependents(dep.eip_number)
                for parent in parent_deps:
                    if f"eip-{parent}" in node_ids:
                        view.edges.append(
                            GraphEdge(
                                source=f"eip-{parent}",
                                target=f"eip-{dep.eip_number}",
                                relationship="REQUIRES",
                                weight=1.0 / dep.depth,
                            )
                        )
                        break

        for dep_num in result.direct_dependents:
            view.edges.append(
                GraphEdge(
                    source=f"eip-{dep_num}",
                    target=root_id,
                    relationship="REQUIRES",
                    weight=1.0,
                )
            )

        for dep in result.all_dependents:
            if dep.depth > 1:
                deps_of_dep = self.store.get_direct_dependencies(dep.eip_number)
                for child in deps_of_dep:
                    if f"eip-{child}" in node_ids:
                        view.edges.append(
                            GraphEdge(
                                source=f"eip-{dep.eip_number}",
                                target=f"eip-{child}",
                                relationship="REQUIRES",
                                weight=1.0 / dep.depth,
                            )
                        )
                        break

        if include_supersedes:
            for sup_num in result.supersedes:
                view.edges.append(
                    GraphEdge(
                        source=root_id,
                        target=f"eip-{sup_num}",
                        relationship="SUPERSEDES",
                        weight=0.8,
                    )
                )

            for sup_num in result.superseded_by:
                view.edges.append(
                    GraphEdge(
                        source=f"eip-{sup_num}",
                        target=root_id,
                        relationship="SUPERSEDES",
                        weight=0.8,
                    )
                )

        view.layout_hints = self._build_layout_hints(view, layout, eip_number)

        logger.info(
            "built_dependency_view",
            root_eip=eip_number,
            depth=depth,
            nodes=len(view.nodes),
            edges=len(view.edges),
        )

        return view

    def build_multi_eip_view(
        self,
        eip_numbers: list[int],
        depth: int = 2,
    ) -> DependencyView:
        """Build a dependency view for multiple EIPs.

        Useful for visualizing how multiple EIPs relate to each other.
        """
        view = DependencyView()
        node_ids: set[str] = set()
        edge_set: set[tuple[str, str, str]] = set()

        for eip_num in eip_numbers:
            single_view = self.build(eip_num, depth=depth)

            for node in single_view.nodes:
                if node.id not in node_ids:
                    node_ids.add(node.id)
                    if node.type == NodeType.ROOT and eip_num != int(
                        node.metadata.get("eip_number", 0)
                    ):
                        node.type = NodeType.DEPENDENCY
                    view.nodes.append(node)

            for edge in single_view.edges:
                edge_key = (edge.source, edge.target, edge.relationship)
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    view.edges.append(edge)

        for node in view.nodes:
            if node.metadata.get("eip_number") in eip_numbers:
                node.type = NodeType.ROOT

        view.layout_hints = {
            "algorithm": LayoutHint.FORCE_DIRECTED.value,
            "root_nodes": [f"eip-{n}" for n in eip_numbers],
            "charge": -300,
            "link_distance": 100,
        }

        return view

    def _build_layout_hints(
        self,
        view: DependencyView,
        layout: LayoutHint,
        root_eip: int,
    ) -> dict[str, Any]:
        """Build layout configuration hints."""
        hints: dict[str, Any] = {
            "algorithm": layout.value,
            "root_node": f"eip-{root_eip}",
        }

        if layout == LayoutHint.HIERARCHICAL:
            hints["direction"] = "TB"
            hints["level_separation"] = 100
            hints["node_separation"] = 50

        elif layout == LayoutHint.FORCE_DIRECTED:
            hints["charge"] = -200
            hints["link_distance"] = 80
            hints["collision_radius"] = 30

        elif layout == LayoutHint.RADIAL:
            hints["radius_step"] = 100
            hints["angle_range"] = 360

        elif layout == LayoutHint.TREE:
            hints["orientation"] = "vertical"
            hints["sibling_spacing"] = 40

        dep_count = sum(1 for n in view.nodes if n.type == NodeType.DEPENDENCY)
        dependent_count = sum(1 for n in view.nodes if n.type == NodeType.DEPENDENT)

        hints["statistics"] = {
            "dependencies": dep_count,
            "dependents": dependent_count,
            "total_nodes": len(view.nodes),
        }

        return hints
