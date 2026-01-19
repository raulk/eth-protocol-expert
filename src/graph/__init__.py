"""Graph module - FalkorDB-based EIP relationship graph (Phase 4+6+10+11+13)."""

from .citation_graph import CitationEdge, CitationGraph, PaperNode
from .confidence_scorer import CalibrationData, ConfidenceFactors, ConfidenceScorer
from .cross_reference import CrossReferenceBuilder, EIPMention
from .dependency_traverser import DependencyResult, DependencyTraverser
from .eip_graph_builder import EIPGraphBuilder, EIPNode, EIPRelationship
from .falkordb_store import FalkorDBStore
from .forum_graph import ForumGraphBuilder, ForumNode, ReplyEdge
from .incremental_updater import (
    ChangeType,
    CorpusChange,
    IncrementalUpdater,
    InferenceMetadata,
    UpdateResult,
)
from .relationship_inferrer import (
    InferredRelationship,
    RelationshipInferrer,
    RelationshipType,
)
from .selective_traverser import (
    RelationshipStep,
    SelectiveTraverser,
    TraversalConfig,
    TraversalDirection,
    TraversalResult,
    TraversalSummary,
)
from .spec_impl_linker import SpecImplLink, SpecImplLinker

__all__ = [
    "CalibrationData",
    "ChangeType",
    "CitationEdge",
    "CitationGraph",
    "ConfidenceFactors",
    "ConfidenceScorer",
    "CorpusChange",
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
    "IncrementalUpdater",
    "InferenceMetadata",
    "InferredRelationship",
    "PaperNode",
    "RelationshipInferrer",
    "RelationshipStep",
    "RelationshipType",
    "ReplyEdge",
    "SelectiveTraverser",
    "SpecImplLink",
    "SpecImplLinker",
    "TraversalConfig",
    "TraversalDirection",
    "TraversalResult",
    "TraversalSummary",
    "UpdateResult",
]
