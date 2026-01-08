"""Graph module - FalkorDB-based EIP relationship graph (Phase 4+6)."""

from .cross_reference import CrossReferenceBuilder, EIPMention
from .dependency_traverser import DependencyResult, DependencyTraverser
from .eip_graph_builder import EIPGraphBuilder, EIPNode, EIPRelationship
from .falkordb_store import FalkorDBStore
from .forum_graph import ForumGraphBuilder, ForumNode, ReplyEdge

__all__ = [
    "CrossReferenceBuilder",
    "DependencyResult",
    "DependencyTraverser",
    "EIPGraphBuilder",
    "EIPMention",
    "EIPNode",
    "EIPRelationship",
    "FalkorDBStore",
    "ForumGraphBuilder",
    "ForumNode",
    "ReplyEdge",
]
