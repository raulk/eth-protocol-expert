"""Alias Extractor - Extract aliases from corpus automatically (Phase 7)."""

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class ExtractedAlias:
    """An alias extracted from corpus text."""

    canonical: str
    alias: str
    source_eip: int | None
    pattern_type: str
    context: str


class AliasExtractor:
    """Extract concept aliases from EIP content using pattern matching.

    Detects patterns like:
    - "also known as X"
    - "commonly called X"
    - "referred to as X"
    - Parenthetical aliases: "EIP-4844 (proto-danksharding)"
    - Acronym definitions: "Account Abstraction (AA)"
    """

    ALIAS_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        (
            r"(?:also|sometimes)\s+(?:known|referred\s+to)\s+as\s+[\"']?([^\"'\.,]+)[\"']?",
            "known_as",
        ),
        (r"commonly\s+called\s+[\"']?([^\"'\.,]+)[\"']?", "commonly_called"),
        (r"referred\s+to\s+as\s+[\"']?([^\"'\.,]+)[\"']?", "referred_as"),
        (r"nicknamed?\s+[\"']?([^\"'\.,]+)[\"']?", "nicknamed"),
        (r"dubbed\s+[\"']?([^\"'\.,]+)[\"']?", "dubbed"),
        (r"\((?:the\s+)?\"([^\"]+)\"\)", "parenthetical_quoted"),
        (r"\(a\.?k\.?a\.?\s+([^)]+)\)", "aka"),
    ]

    EIP_PARENTHETICAL_PATTERN: ClassVar[str] = r"EIP-?(\d+)\s*\(([^)]+)\)"
    ACRONYM_PATTERN: ClassVar[str] = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\(([A-Z]{2,})\)"
    EIP_NUMBER_PATTERN: ClassVar[str] = r"EIP-?(\d+)"

    def __init__(self) -> None:
        self._compiled_patterns: list[tuple[re.Pattern, str]] = [
            (re.compile(pattern, re.IGNORECASE), name) for pattern, name in self.ALIAS_PATTERNS
        ]
        self._eip_parenthetical = re.compile(self.EIP_PARENTHETICAL_PATTERN, re.IGNORECASE)
        self._acronym_pattern = re.compile(self.ACRONYM_PATTERN)
        self._eip_number_pattern = re.compile(self.EIP_NUMBER_PATTERN, re.IGNORECASE)

    def extract_from_eip(self, eip_content: str, eip_number: int) -> list[tuple[str, str]]:
        """Extract aliases from EIP content.

        Returns list of (canonical, alias) tuples.
        """
        extracted = self._extract_all(eip_content, eip_number)
        return [(alias_info.canonical, alias_info.alias) for alias_info in extracted]

    def extract_detailed(self, eip_content: str, eip_number: int) -> list[ExtractedAlias]:
        """Extract aliases with full metadata."""
        return self._extract_all(eip_content, eip_number)

    def _extract_all(self, content: str, eip_number: int) -> list[ExtractedAlias]:
        results: list[ExtractedAlias] = []
        canonical = f"EIP-{eip_number}"
        seen_aliases: set[str] = set()

        for match in self._eip_parenthetical.finditer(content):
            matched_eip = int(match.group(1))
            alias = match.group(2).strip()

            if matched_eip == eip_number and self._is_valid_alias(alias):
                alias_lower = alias.lower()
                if alias_lower not in seen_aliases:
                    seen_aliases.add(alias_lower)
                    results.append(
                        ExtractedAlias(
                            canonical=canonical,
                            alias=alias,
                            source_eip=eip_number,
                            pattern_type="eip_parenthetical",
                            context=match.group(0),
                        )
                    )

        for pattern, pattern_name in self._compiled_patterns:
            for match in pattern.finditer(content):
                alias = match.group(1).strip()
                if self._is_valid_alias(alias):
                    alias_lower = alias.lower()
                    if alias_lower not in seen_aliases:
                        seen_aliases.add(alias_lower)
                        results.append(
                            ExtractedAlias(
                                canonical=canonical,
                                alias=alias,
                                source_eip=eip_number,
                                pattern_type=pattern_name,
                                context=self._extract_context(content, match.start(), match.end()),
                            )
                        )

        for match in self._acronym_pattern.finditer(content):
            full_name = match.group(1)
            acronym = match.group(2)
            if self._is_valid_acronym(full_name, acronym):
                for variant in [full_name, acronym]:
                    variant_lower = variant.lower()
                    if variant_lower not in seen_aliases:
                        seen_aliases.add(variant_lower)
                        results.append(
                            ExtractedAlias(
                                canonical=canonical,
                                alias=variant,
                                source_eip=eip_number,
                                pattern_type="acronym_definition",
                                context=match.group(0),
                            )
                        )

        return results

    def _is_valid_alias(self, alias: str) -> bool:
        if len(alias) < 2 or len(alias) > 100:
            return False

        if re.match(r"^[\d\s]+$", alias):
            return False

        stop_words = {"this", "that", "the", "a", "an", "it", "its", "which", "what", "where"}
        if alias.lower() in stop_words:
            return False

        if alias.lower().startswith(("see ", "note ", "i.e.", "e.g.")):
            return False

        return True

    def _is_valid_acronym(self, full_name: str, acronym: str) -> bool:
        if len(acronym) < 2 or len(acronym) > 8:
            return False

        words = full_name.split()
        if len(words) < len(acronym):
            return False

        expected_acronym = "".join(word[0].upper() for word in words if word[0].isupper())
        return acronym == expected_acronym or len(acronym) >= 2

    def _extract_context(self, content: str, start: int, end: int, window: int = 50) -> str:
        context_start = max(0, start - window)
        context_end = min(len(content), end + window)
        return content[context_start:context_end]

    def extract_cross_references(
        self, content: str, source_eip: int
    ) -> list[tuple[int, str | None]]:
        """Extract references to other EIPs with optional context.

        Returns list of (eip_number, context) tuples.
        """
        results: list[tuple[int, str | None]] = []
        seen_eips: set[int] = set()

        for match in self._eip_parenthetical.finditer(content):
            eip_num = int(match.group(1))
            if eip_num != source_eip and eip_num not in seen_eips:
                seen_eips.add(eip_num)
                context = match.group(2).strip()
                results.append((eip_num, context))

        for match in self._eip_number_pattern.finditer(content):
            eip_num = int(match.group(1))
            if eip_num != source_eip and eip_num not in seen_eips:
                seen_eips.add(eip_num)
                results.append((eip_num, None))

        return results
