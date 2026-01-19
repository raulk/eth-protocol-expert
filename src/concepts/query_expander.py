"""Query Expander - Expand queries with aliases (Phase 7)."""

import re
from dataclasses import dataclass, field

from src.concepts.alias_table import AliasTable


@dataclass
class ExpandedQuery:
    """Result of query expansion with aliases."""

    original: str
    expanded_terms: list[str] = field(default_factory=list)
    canonical_terms: list[str] = field(default_factory=list)
    eip_numbers: list[int] = field(default_factory=list)
    term_mappings: dict[str, str] = field(default_factory=dict)

    @property
    def has_expansions(self) -> bool:
        return len(self.expanded_terms) > 0 or len(self.canonical_terms) > 0

    def get_search_terms(self) -> list[str]:
        """Get all terms to use for search (original + expanded + canonical)."""
        terms = [self.original]
        terms.extend(self.expanded_terms)
        terms.extend(self.canonical_terms)
        return list(dict.fromkeys(terms))

    def get_eip_filter(self) -> list[int] | None:
        """Get EIP numbers for filtering, if any."""
        return self.eip_numbers if self.eip_numbers else None


class QueryExpander:
    """Expand queries by resolving aliases and adding related terms.

    Given a query like "What is proto-danksharding?", expands it to include:
    - Canonical form: "EIP-4844"
    - Related aliases: "blob transactions", "danksharding-lite"
    """

    EIP_PATTERN = re.compile(r"\b(?:eip|erc)-?(\d+)\b", re.IGNORECASE)
    WORD_BOUNDARY_PATTERN = re.compile(r"\b([a-z][\w-]*(?:\s+[a-z][\w-]*)*)\b", re.IGNORECASE)

    def __init__(self, alias_table: AliasTable | None = None) -> None:
        self.alias_table = alias_table or AliasTable()

    def expand(self, query: str, alias_table: AliasTable | None = None) -> ExpandedQuery:
        """Expand a query with aliases and canonical terms.

        Args:
            query: The user's original query
            alias_table: Optional alias table override

        Returns:
            ExpandedQuery with original, expanded terms, and canonical terms
        """
        table = alias_table or self.alias_table
        result = ExpandedQuery(original=query)
        seen_canonicals: set[str] = set()
        seen_terms: set[str] = set()
        seen_eips: set[int] = set()

        for match in self.EIP_PATTERN.finditer(query):
            eip_num = int(match.group(1))
            canonical = f"EIP-{eip_num}"
            seen_eips.add(eip_num)

            if canonical not in seen_canonicals:
                seen_canonicals.add(canonical)
                result.canonical_terms.append(canonical)
                result.term_mappings[match.group(0)] = canonical

                for alias in table.get_aliases(canonical):
                    if alias.lower() not in seen_terms:
                        seen_terms.add(alias.lower())
                        result.expanded_terms.append(alias)

        query_lower = query.lower()
        for entry in table.all_entries():
            for alias in entry.aliases:
                if alias.lower() in query_lower:
                    canonical = entry.canonical_id
                    if canonical not in seen_canonicals:
                        seen_canonicals.add(canonical)
                        result.canonical_terms.append(canonical)
                        result.term_mappings[alias] = canonical

                        for eip_num in entry.eip_numbers:
                            if eip_num not in seen_eips:
                                seen_eips.add(eip_num)

                        for other_alias in entry.aliases:
                            if (
                                other_alias.lower() != alias.lower()
                                and other_alias.lower() not in seen_terms
                            ):
                                seen_terms.add(other_alias.lower())
                                result.expanded_terms.append(other_alias)

            if entry.canonical_name.lower() in query_lower:
                canonical = entry.canonical_id
                if canonical not in seen_canonicals:
                    seen_canonicals.add(canonical)
                    result.canonical_terms.append(canonical)
                    result.term_mappings[entry.canonical_name] = canonical

                    for eip_num in entry.eip_numbers:
                        if eip_num not in seen_eips:
                            seen_eips.add(eip_num)

                    for alias in entry.aliases:
                        if alias.lower() not in seen_terms:
                            seen_terms.add(alias.lower())
                            result.expanded_terms.append(alias)

        result.eip_numbers = sorted(seen_eips)

        return result

    def expand_for_search(self, query: str) -> list[str]:
        """Convenience method that returns just the search terms."""
        expanded = self.expand(query)
        return expanded.get_search_terms()

    def resolve_term(self, term: str) -> str | None:
        """Resolve a single term to its canonical form."""
        return self.alias_table.resolve(term)

    def get_related_terms(self, term: str) -> list[str]:
        """Get all related terms for a given term."""
        canonical = self.alias_table.resolve(term)
        if not canonical:
            return []
        return self.alias_table.get_aliases(canonical)
