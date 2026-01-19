"""Tests for Concept Resolution module (Phase 7)."""

import tempfile
from pathlib import Path

import pytest

from src.concepts.alias_extractor import AliasExtractor
from src.concepts.alias_table import AliasEntry, AliasTable
from src.concepts.concept_resolver import ConceptResolver
from src.concepts.query_expander import QueryExpander


class TestAliasEntry:
    def test_to_dict_roundtrip(self) -> None:
        entry = AliasEntry(
            canonical_id="EIP-4844",
            canonical_name="Proto-Danksharding",
            aliases=["blobs", "blob transactions"],
            eip_numbers=[4844],
        )
        data = entry.to_dict()
        restored = AliasEntry.from_dict(data)

        assert restored.canonical_id == entry.canonical_id
        assert restored.canonical_name == entry.canonical_name
        assert restored.aliases == entry.aliases
        assert restored.eip_numbers == entry.eip_numbers


class TestAliasTable:
    def test_resolve_canonical(self) -> None:
        table = AliasTable()
        assert table.resolve("EIP-4844") == "EIP-4844"
        assert table.resolve("eip-4844") == "EIP-4844"
        assert table.resolve("eip4844") == "EIP-4844"

    def test_resolve_alias(self) -> None:
        table = AliasTable()
        assert table.resolve("proto-danksharding") == "EIP-4844"
        assert table.resolve("blob transactions") == "EIP-4844"
        assert table.resolve("the merge") == "EIP-3675"
        assert table.resolve("account abstraction") == "EIP-4337"

    def test_resolve_unknown(self) -> None:
        table = AliasTable()
        assert table.resolve("unknown-concept") is None

    def test_get_aliases(self) -> None:
        table = AliasTable()
        aliases = table.get_aliases("EIP-4844")
        assert "proto-danksharding" in aliases
        assert "blob transactions" in aliases

    def test_add_alias(self) -> None:
        table = AliasTable()
        table.add_alias("EIP-4844", "new-alias")
        assert table.resolve("new-alias") == "EIP-4844"
        assert "new-alias" in table.get_aliases("EIP-4844")

    def test_add_alias_new_entry(self) -> None:
        table = AliasTable()
        table.add_alias("EIP-9999", "test-alias")
        assert table.resolve("test-alias") == "EIP-9999"
        assert table.resolve("eip-9999") == "EIP-9999"

    def test_get_eip_numbers(self) -> None:
        table = AliasTable()
        assert table.get_eip_numbers("proto-danksharding") == [4844]
        assert table.get_eip_numbers("the merge") == [3675]

    def test_save_load(self) -> None:
        table = AliasTable()
        table.add_alias("EIP-4844", "custom-alias")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            table.save(path)
            assert path.exists()

            new_table = AliasTable.from_file(path)
            assert new_table.resolve("custom-alias") == "EIP-4844"
            assert new_table.resolve("proto-danksharding") == "EIP-4844"
        finally:
            path.unlink(missing_ok=True)

    def test_erc_alias_resolution(self) -> None:
        table = AliasTable()
        assert table.resolve("erc20") == "EIP-20"
        assert table.resolve("erc-20") == "EIP-20"
        assert table.resolve("erc721") == "EIP-721"


class TestAliasExtractor:
    def test_extract_parenthetical(self) -> None:
        extractor = AliasExtractor()
        content = "EIP-4844 (proto-danksharding) introduces blob transactions."
        results = extractor.extract_from_eip(content, 4844)

        aliases = [alias for _, alias in results]
        assert "proto-danksharding" in aliases

    def test_extract_known_as(self) -> None:
        extractor = AliasExtractor()
        content = (
            "This proposal, also known as 'the merge', transitions Ethereum to proof of stake."
        )
        results = extractor.extract_from_eip(content, 3675)

        aliases = [alias for _, alias in results]
        assert any("merge" in alias.lower() for alias in aliases)

    def test_extract_commonly_called(self) -> None:
        extractor = AliasExtractor()
        content = "The standard is commonly called 'NFT standard' in the community."
        results = extractor.extract_from_eip(content, 721)

        aliases = [alias for _, alias in results]
        assert any("nft" in alias.lower() for alias in aliases)

    def test_extract_acronym(self) -> None:
        extractor = AliasExtractor()
        content = "Account Abstraction (AA) allows for smart contract wallets."
        results = extractor.extract_detailed(content, 4337)

        aliases = [r.alias for r in results]
        assert "Account Abstraction" in aliases or "AA" in aliases

    def test_extract_cross_references(self) -> None:
        extractor = AliasExtractor()
        content = "This EIP depends on EIP-1559 (fee market) and references EIP-2930."
        refs = extractor.extract_cross_references(content, 4844)

        eip_nums = [num for num, _ in refs]
        assert 1559 in eip_nums
        assert 2930 in eip_nums

    def test_filter_invalid_aliases(self) -> None:
        extractor = AliasExtractor()
        content = "Also known as 'the' something, referred to as 'a'."
        results = extractor.extract_from_eip(content, 1234)

        aliases = [alias for _, alias in results]
        assert "the" not in aliases
        assert "a" not in aliases


class TestQueryExpander:
    def test_expand_eip_number(self) -> None:
        expander = QueryExpander()
        result = expander.expand("What is EIP-4844?")

        assert "EIP-4844" in result.canonical_terms
        assert any("blob" in term.lower() for term in result.expanded_terms)
        assert 4844 in result.eip_numbers

    def test_expand_alias(self) -> None:
        expander = QueryExpander()
        result = expander.expand("What is proto-danksharding?")

        assert "EIP-4844" in result.canonical_terms
        assert result.has_expansions

    def test_expand_multiple(self) -> None:
        expander = QueryExpander()
        result = expander.expand("Compare EIP-1559 and EIP-4844")

        assert "EIP-1559" in result.canonical_terms
        assert "EIP-4844" in result.canonical_terms
        assert 1559 in result.eip_numbers
        assert 4844 in result.eip_numbers

    def test_get_search_terms(self) -> None:
        expander = QueryExpander()
        result = expander.expand("What is the merge?")
        terms = result.get_search_terms()

        assert result.original in terms
        assert "EIP-3675" in terms

    def test_no_expansion_unknown(self) -> None:
        expander = QueryExpander()
        result = expander.expand("What is the weather today?")

        assert not result.has_expansions
        assert len(result.canonical_terms) == 0


class TestConceptResolver:
    def test_resolve_query_with_alias(self) -> None:
        resolver = ConceptResolver()
        result = resolver.resolve_query("How does proto-danksharding work?")

        assert result.has_resolutions
        assert "EIP-4844" in result.canonical_concepts
        assert 4844 in result.eip_numbers

    def test_resolve_query_with_eip(self) -> None:
        resolver = ConceptResolver()
        result = resolver.resolve_query("Explain EIP-1559 fee mechanism")

        assert "EIP-1559" in result.canonical_concepts
        assert 1559 in result.eip_numbers

    def test_resolve_single_term(self) -> None:
        resolver = ConceptResolver()
        assert resolver.resolve_term("account abstraction") == "EIP-4337"
        assert resolver.resolve_term("nfts") == "EIP-721"

    def test_get_aliases(self) -> None:
        resolver = ConceptResolver()
        aliases = resolver.get_aliases("EIP-4844")
        assert len(aliases) > 0
        assert "proto-danksharding" in aliases

    def test_learn_from_eip(self) -> None:
        resolver = ConceptResolver()
        content = """
        EIP-9999 (Super Scaling) introduces a new mechanism.
        This proposal, also known as "hyper mode", improves throughput.
        """
        extracted = resolver.learn_from_eip(content, 9999)

        assert len(extracted) > 0
        assert resolver.resolve_term("super scaling") == "EIP-9999"

    def test_save_load(self) -> None:
        resolver = ConceptResolver()
        resolver.add_alias("EIP-4844", "my-custom-alias")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            resolver.save(path)

            new_resolver = ConceptResolver(alias_file=path)
            assert new_resolver.resolve_term("my-custom-alias") == "EIP-4844"
        finally:
            path.unlink(missing_ok=True)

    def test_to_dict(self) -> None:
        resolver = ConceptResolver()
        result = resolver.resolve_query("What is EIP-4844?")
        data = result.to_dict()

        assert "original" in data
        assert "canonical_concepts" in data
        assert "eip_numbers" in data
        assert data["original"] == "What is EIP-4844?"

    def test_expand_for_retrieval(self) -> None:
        resolver = ConceptResolver()
        expanded = resolver.expand_for_retrieval("Tell me about blobs")

        assert expanded.original == "Tell me about blobs"
        assert "EIP-4844" in expanded.canonical_terms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
