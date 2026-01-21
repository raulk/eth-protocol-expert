"""Tests for span extraction edge cases with markdown, code, and tables."""

import pytest

from src.evidence.evidence_span import EvidenceSpan
from src.evidence.span_selector import MarkdownSpanExtractor, SpanSelector


class TestMarkdownSpanExtractor:
    """Test markdown-aware span extraction."""

    @pytest.fixture
    def extractor(self):
        return MarkdownSpanExtractor()

    def test_code_block_extraction(self, extractor):
        """Code blocks should be extracted as atomic units."""
        text = """Some text before.

```solidity
function transfer(address to, uint256 amount) public {
    balances[msg.sender] -= amount;
    balances[to] += amount;
}
```

Some text after."""

        evidence = EvidenceSpan.from_chunk(
            document_id="test-doc",
            chunk_id="chunk-1",
            content=text,
        )

        spans = extractor.extract_structural_spans(evidence)

        # Should find the code block
        code_spans = [s for s in spans if "```" in s.span_text]
        assert len(code_spans) >= 1

        # Code block should be complete
        assert "function transfer" in code_spans[0].span_text
        assert "balances[to] += amount" in code_spans[0].span_text

    def test_table_extraction(self, extractor):
        """Tables should be extracted as atomic units."""
        text = """Gas costs for operations:

| Operation | Gas Cost | Description |
|-----------|----------|-------------|
| SSTORE | 20000 | Storage write |
| SLOAD | 800 | Storage read |
| CALL | 700 | External call |

More text after the table."""

        evidence = EvidenceSpan.from_chunk(
            document_id="test-doc",
            chunk_id="chunk-1",
            content=text,
        )

        spans = extractor.extract_structural_spans(evidence)

        # Should find the table
        table_spans = [s for s in spans if "|" in s.span_text and "SSTORE" in s.span_text]
        assert len(table_spans) >= 1

        # Table should include all rows
        table_text = table_spans[0].span_text
        assert "SSTORE" in table_text
        assert "SLOAD" in table_text
        assert "CALL" in table_text

    def test_code_block_position_detection(self, extractor):
        """Positions inside code blocks should be detected."""
        text = """Before code.

```python
x = 1
y = 2
```

After code."""

        code_start = text.find("```python")
        code_middle = text.find("x = 1")
        code_end = text.find("```\n\nAfter") + 3
        after_code = text.find("After code")

        assert extractor.is_inside_code_block(text, code_start)
        assert extractor.is_inside_code_block(text, code_middle)
        assert not extractor.is_inside_code_block(text, after_code)

    def test_nested_code_in_list(self, extractor):
        """Code blocks nested in lists should be handled."""
        text = """Steps:
- Step 1: Do this
- Step 2: Run this code:
  ```bash
  npm install
  npm run build
  ```
- Step 3: Verify"""

        evidence = EvidenceSpan.from_chunk(
            document_id="test-doc",
            chunk_id="chunk-1",
            content=text,
        )

        spans = extractor.extract_structural_spans(evidence)
        code_spans = [s for s in spans if "npm install" in s.span_text]

        # Should extract the code block
        assert len(code_spans) >= 1


class TestSpanSelector:
    """Test span selection for claims."""

    @pytest.fixture
    def selector(self):
        return SpanSelector(min_span_chars=30, max_span_chars=200)

    def test_small_span_unchanged(self, selector):
        """Small spans should be returned as-is."""
        text = "EIP-1559 introduces a base fee that is burned."
        evidence = EvidenceSpan.from_chunk(
            document_id="eip-1559",
            chunk_id="chunk-1",
            content=text,
        )

        candidates = selector._generate_candidate_spans(evidence)

        # Single small span should be returned unchanged
        assert len(candidates) == 1
        assert candidates[0].span_text == text

    def test_large_span_split(self, selector):
        """Large spans should be split into candidates."""
        text = """The base fee is adjusted based on the gas used in the previous block.
If the previous block used more than the target gas, the base fee increases.
If the previous block used less than the target gas, the base fee decreases.
This mechanism aims to keep blocks around 50% full on average.
The base fee is burned, removing ETH from circulation.
This creates deflationary pressure on the ETH supply."""

        evidence = EvidenceSpan.from_chunk(
            document_id="eip-1559",
            chunk_id="chunk-1",
            content=text,
        )

        candidates = selector._generate_candidate_spans(evidence)

        # Should generate multiple candidates
        assert len(candidates) > 1

        # All candidates should meet minimum size
        for c in candidates:
            assert len(c.span_text) >= selector.min_span_chars

    def test_keyword_scoring(self, selector):
        """Keyword scoring should rank relevant spans higher."""
        claim = "The base fee is burned in EIP-1559."
        spans = [
            EvidenceSpan.from_chunk(
                document_id="eip-1559",
                chunk_id="chunk-1",
                content="The priority fee goes to validators.",
            ),
            EvidenceSpan.from_chunk(
                document_id="eip-1559",
                chunk_id="chunk-2",
                content="The base fee is burned, removing ETH from circulation.",
            ),
            EvidenceSpan.from_chunk(
                document_id="eip-1559",
                chunk_id="chunk-3",
                content="Gas limits are set by block producers.",
            ),
        ]

        scored = selector._score_with_keywords(claim, spans, top_n=3)

        # The span about base fee burning should rank highest
        assert "burned" in scored[0].span.span_text.lower()

    def test_sentence_boundaries_respected(self, selector):
        """Span extraction should respect sentence boundaries."""
        text = """First sentence about gas. Second sentence about fees. Third sentence about blocks."""
        evidence = EvidenceSpan.from_chunk(
            document_id="test-doc",
            chunk_id="chunk-1",
            content=text,
        )

        sentence_spans = selector._extract_sentence_spans(evidence)

        # Each span should contain complete sentences
        for span in sentence_spans:
            # Should end with punctuation or be the end of text
            text = span.span_text.strip()
            if text:
                assert text[-1] in '.!?' or text == evidence.span_text.strip()


class TestCharacterOffsets:
    """Test character offset handling."""

    def test_offset_extraction(self):
        """Offsets should correctly identify text positions."""
        full_text = "The base fee is 15 gwei. The tip is 2 gwei."
        span = EvidenceSpan.from_substring(
            document_id="test",
            chunk_id="c1",
            full_content=full_text,
            start_offset=0,
            end_offset=24,
        )

        assert span.span_text == "The base fee is 15 gwei."
        assert span.start_offset == 0
        assert span.end_offset == 24

    def test_offset_validation(self):
        """Span validation should use character offsets."""
        full_text = "Original content here."
        span = EvidenceSpan.from_substring(
            document_id="test",
            chunk_id="c1",
            full_content=full_text,
            start_offset=0,
            end_offset=len(full_text),
        )

        # Same content should validate
        assert span.validate_against(full_text)

        # Modified content should fail validation
        modified = "Changed content here."
        assert not span.validate_against(modified)

    def test_hash_consistency(self):
        """Span hash should be consistent for same content."""
        content = "Test content for hashing."

        span1 = EvidenceSpan.from_chunk(
            document_id="test",
            chunk_id="c1",
            content=content,
        )

        span2 = EvidenceSpan.from_chunk(
            document_id="test",
            chunk_id="c2",
            content=content,
        )

        assert span1.span_hash == span2.span_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
