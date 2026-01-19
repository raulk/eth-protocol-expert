"""Concept Resolution module for Ethereum Protocol Intelligence (Phase 7).

This module provides:
- AliasTable: Bidirectional mapping between canonical concepts and aliases
- AliasExtractor: Automatic extraction of aliases from EIP content
- QueryExpander: Query expansion with synonyms and aliases
- ConceptResolver: Main entry point combining all functionality
"""

from src.concepts.alias_extractor import AliasExtractor, ExtractedAlias
from src.concepts.alias_table import AliasEntry, AliasTable
from src.concepts.concept_resolver import ConceptResolver, ResolvedQuery
from src.concepts.query_expander import ExpandedQuery, QueryExpander

__all__ = [
    "AliasEntry",
    "AliasExtractor",
    "AliasTable",
    "ConceptResolver",
    "ExpandedQuery",
    "ExtractedAlias",
    "QueryExpander",
    "ResolvedQuery",
]
