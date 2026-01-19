"""Concept Resolver - Main entry point for concept resolution (Phase 7)."""

from dataclasses import dataclass, field
from pathlib import Path

from src.concepts.alias_extractor import AliasExtractor, ExtractedAlias
from src.concepts.alias_table import AliasEntry, AliasTable
from src.concepts.query_expander import ExpandedQuery, QueryExpander


@dataclass
class ResolvedQuery:
    """Result of resolving concepts in a query."""

    original: str
    canonical_concepts: list[str] = field(default_factory=list)
    expanded_terms: list[str] = field(default_factory=list)
    eip_numbers: list[int] = field(default_factory=list)
    term_mappings: dict[str, str] = field(default_factory=dict)

    @property
    def has_resolutions(self) -> bool:
        return len(self.canonical_concepts) > 0

    def get_all_terms(self) -> list[str]:
        """Get all terms including original, canonical, and expanded."""
        terms = [self.original]
        terms.extend(self.canonical_concepts)
        terms.extend(self.expanded_terms)
        return list(dict.fromkeys(terms))

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "canonical_concepts": self.canonical_concepts,
            "expanded_terms": self.expanded_terms,
            "eip_numbers": self.eip_numbers,
            "term_mappings": self.term_mappings,
        }


class ConceptResolver:
    """Main entry point for concept resolution.

    Combines alias table lookup, alias extraction, and query expansion
    to provide comprehensive concept resolution for Ethereum protocol queries.
    """

    def __init__(
        self,
        alias_table: AliasTable | None = None,
        alias_file: Path | str | None = None,
    ) -> None:
        if alias_table:
            self._alias_table = alias_table
        elif alias_file:
            self._alias_table = AliasTable.from_file(alias_file)
        else:
            self._alias_table = AliasTable()

        self._extractor = AliasExtractor()
        self._expander = QueryExpander(self._alias_table)

    @property
    def alias_table(self) -> AliasTable:
        return self._alias_table

    def resolve_query(self, query: str) -> ResolvedQuery:
        """Resolve concepts in a query.

        Args:
            query: The user's question or search query

        Returns:
            ResolvedQuery with canonical concepts, expanded terms, and EIP numbers
        """
        expanded = self._expander.expand(query)

        return ResolvedQuery(
            original=query,
            canonical_concepts=expanded.canonical_terms,
            expanded_terms=expanded.expanded_terms,
            eip_numbers=expanded.eip_numbers,
            term_mappings=expanded.term_mappings,
        )

    def resolve_term(self, term: str) -> str | None:
        """Resolve a single term to its canonical form."""
        return self._alias_table.resolve(term)

    def get_aliases(self, canonical: str) -> list[str]:
        """Get all aliases for a canonical concept."""
        return self._alias_table.get_aliases(canonical)

    def get_eip_numbers(self, term: str) -> list[int]:
        """Get EIP numbers associated with a term."""
        return self._alias_table.get_eip_numbers(term)

    def add_alias(self, canonical: str, alias: str) -> None:
        """Add an alias for a canonical concept."""
        self._alias_table.add_alias(canonical, alias)

    def learn_from_eip(self, eip_content: str, eip_number: int) -> list[ExtractedAlias]:
        """Extract and learn aliases from EIP content.

        Automatically adds extracted aliases to the alias table.

        Returns:
            List of extracted aliases
        """
        extracted = self._extractor.extract_detailed(eip_content, eip_number)

        for alias_info in extracted:
            self._alias_table.add_alias(alias_info.canonical, alias_info.alias)

        return extracted

    def get_entry(self, canonical: str) -> AliasEntry | None:
        """Get the full entry for a canonical concept."""
        return self._alias_table.get_entry(canonical)

    def save(self, path: Path | str) -> None:
        """Save the alias table to a file."""
        self._alias_table.save(path)

    def load(self, path: Path | str) -> None:
        """Load aliases from a file (merges with existing)."""
        self._alias_table.load(path)

    def expand_for_retrieval(self, query: str) -> ExpandedQuery:
        """Expand a query for retrieval purposes.

        Returns the full ExpandedQuery object for more detailed control.
        """
        return self._expander.expand(query)
