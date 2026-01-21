"""Span Selector - Select optimal evidence spans for claims using cross-encoder."""

import re
from dataclasses import dataclass

import structlog

from .evidence_span import EvidenceSpan

logger = structlog.get_logger()


@dataclass
class ScoredSpan:
    """A span with its relevance score."""

    span: EvidenceSpan
    score: float
    rank: int


class SpanSelector:
    """Select optimal evidence spans for claims using cross-encoder reranking.

    For each claim, finds the minimal text span that provides the strongest
    support, rather than using entire chunks. This improves NLI precision
    by focusing validation on the most relevant text.
    """

    def __init__(
        self,
        reranker=None,
        min_span_chars: int = 50,
        max_span_chars: int = 500,
        overlap_chars: int = 50,
    ):
        """Initialize span selector.

        Args:
            reranker: CohereReranker instance (optional, for cross-encoder scoring)
            min_span_chars: Minimum span size in characters
            max_span_chars: Maximum span size in characters
            overlap_chars: Overlap between candidate spans
        """
        self.reranker = reranker
        self.min_span_chars = min_span_chars
        self.max_span_chars = max_span_chars
        self.overlap_chars = overlap_chars

    def select_best_spans(
        self,
        claim_text: str,
        evidence_spans: list[EvidenceSpan],
        top_n: int = 3,
    ) -> list[ScoredSpan]:
        """Select the best evidence spans for a claim.

        Args:
            claim_text: The claim to find evidence for
            evidence_spans: Full evidence spans to search within
            top_n: Number of best spans to return

        Returns:
            List of ScoredSpan sorted by relevance score
        """
        if not evidence_spans:
            return []

        # Generate candidate sub-spans from each evidence span
        candidates = []
        for evidence in evidence_spans:
            sub_spans = self._generate_candidate_spans(evidence)
            candidates.extend(sub_spans)

        if not candidates:
            # Return original spans if no candidates generated
            return [
                ScoredSpan(span=e, score=0.5, rank=i) for i, e in enumerate(evidence_spans[:top_n])
            ]

        # Score candidates using cross-encoder if available
        if self.reranker:
            scored = self._score_with_reranker(claim_text, candidates, top_n)
        else:
            # Fall back to keyword overlap scoring
            scored = self._score_with_keywords(claim_text, candidates, top_n)

        logger.debug(
            "selected_spans",
            claim=claim_text[:50],
            candidates=len(candidates),
            selected=len(scored),
            top_score=scored[0].score if scored else 0,
        )

        return scored

    def _generate_candidate_spans(
        self,
        evidence: EvidenceSpan,
    ) -> list[EvidenceSpan]:
        """Generate candidate sub-spans from an evidence span."""
        text = evidence.span_text

        if len(text) <= self.max_span_chars:
            # Text is small enough to use as-is
            return [evidence]

        candidates = []

        # Strategy 1: Sentence-based spans
        sentence_spans = self._extract_sentence_spans(evidence)
        candidates.extend(sentence_spans)

        # Strategy 2: Sliding window spans
        window_spans = self._extract_window_spans(evidence)
        candidates.extend(window_spans)

        # Strategy 3: Paragraph-based spans
        paragraph_spans = self._extract_paragraph_spans(evidence)
        candidates.extend(paragraph_spans)

        return candidates

    def _extract_sentence_spans(
        self,
        evidence: EvidenceSpan,
    ) -> list[EvidenceSpan]:
        """Extract sentence-based spans."""
        text = evidence.span_text
        sentences = re.split(r"(?<=[.!?])\s+", text)

        spans = []
        offset = 0
        current_span_start = 0
        current_text = ""

        for sentence in sentences:
            if len(current_text) + len(sentence) <= self.max_span_chars:
                current_text += sentence + " "
            else:
                if len(current_text.strip()) >= self.min_span_chars:
                    spans.append(
                        EvidenceSpan.from_substring(
                            document_id=evidence.document_id,
                            chunk_id=evidence.chunk_id,
                            full_content=text,
                            start_offset=current_span_start,
                            end_offset=current_span_start + len(current_text.strip()),
                            section_path=evidence.section_path,
                            git_commit=evidence.git_commit,
                        )
                    )
                current_span_start = offset
                current_text = sentence + " "

            offset += len(sentence) + 1

        # Add remaining text
        if len(current_text.strip()) >= self.min_span_chars:
            spans.append(
                EvidenceSpan.from_substring(
                    document_id=evidence.document_id,
                    chunk_id=evidence.chunk_id,
                    full_content=text,
                    start_offset=current_span_start,
                    end_offset=min(current_span_start + len(current_text.strip()), len(text)),
                    section_path=evidence.section_path,
                    git_commit=evidence.git_commit,
                )
            )

        return spans

    def _extract_window_spans(
        self,
        evidence: EvidenceSpan,
    ) -> list[EvidenceSpan]:
        """Extract sliding window spans."""
        text = evidence.span_text
        spans = []
        step = self.max_span_chars - self.overlap_chars

        for start in range(0, len(text) - self.min_span_chars, step):
            end = min(start + self.max_span_chars, len(text))

            # Adjust to word boundaries
            if end < len(text):
                space_idx = text.rfind(" ", start, end)
                if space_idx > start + self.min_span_chars:
                    end = space_idx

            if end - start >= self.min_span_chars:
                spans.append(
                    EvidenceSpan.from_substring(
                        document_id=evidence.document_id,
                        chunk_id=evidence.chunk_id,
                        full_content=text,
                        start_offset=start,
                        end_offset=end,
                        section_path=evidence.section_path,
                        git_commit=evidence.git_commit,
                    )
                )

        return spans

    def _extract_paragraph_spans(
        self,
        evidence: EvidenceSpan,
    ) -> list[EvidenceSpan]:
        """Extract paragraph-based spans."""
        text = evidence.span_text
        paragraphs = re.split(r"\n\n+", text)

        spans = []
        offset = 0

        for para in paragraphs:
            para = para.strip()
            if len(para) >= self.min_span_chars:
                # Find actual position in text
                start = text.find(para, offset)
                if start >= 0:
                    end = start + len(para)
                    if len(para) <= self.max_span_chars:
                        spans.append(
                            EvidenceSpan.from_substring(
                                document_id=evidence.document_id,
                                chunk_id=evidence.chunk_id,
                                full_content=text,
                                start_offset=start,
                                end_offset=end,
                                section_path=evidence.section_path,
                                git_commit=evidence.git_commit,
                            )
                        )
                    offset = end

        return spans

    def _score_with_reranker(
        self,
        claim_text: str,
        candidates: list[EvidenceSpan],
        top_n: int,
    ) -> list[ScoredSpan]:
        """Score candidate spans using cross-encoder reranker."""
        documents = [c.span_text for c in candidates]

        response = self.reranker.client.rerank(
            query=claim_text,
            documents=documents,
            model=self.reranker.model,
            top_n=min(top_n, len(candidates)),
        )

        scored = []
        for i, result in enumerate(response.results):
            scored.append(
                ScoredSpan(
                    span=candidates[result.index],
                    score=result.relevance_score,
                    rank=i + 1,
                )
            )

        return scored

    def _score_with_keywords(
        self,
        claim_text: str,
        candidates: list[EvidenceSpan],
        top_n: int,
    ) -> list[ScoredSpan]:
        """Score candidate spans using keyword overlap (fallback)."""
        claim_words = set(claim_text.lower().split())

        scored = []
        for candidate in candidates:
            span_words = set(candidate.span_text.lower().split())
            overlap = len(claim_words & span_words)
            total = len(claim_words | span_words)
            score = overlap / total if total > 0 else 0
            scored.append(ScoredSpan(span=candidate, score=score, rank=0))

        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, s in enumerate(scored[:top_n]):
            s.rank = i + 1

        return scored[:top_n]


class MarkdownSpanExtractor:
    """Extract spans from markdown content while respecting structure.

    Handles code blocks, tables, headers, and lists properly.
    """

    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")
    TABLE_PATTERN = re.compile(r"(\|[^\n]+\|\n)+")
    HEADER_PATTERN = re.compile(r"^#+\s+.+$", re.MULTILINE)
    LIST_PATTERN = re.compile(r"^[-*]\s+.+$", re.MULTILINE)

    def extract_structural_spans(
        self,
        evidence: EvidenceSpan,
    ) -> list[EvidenceSpan]:
        """Extract spans that respect markdown structure."""
        text = evidence.span_text
        spans = []

        # Extract code blocks as atomic units
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            spans.append(
                EvidenceSpan.from_substring(
                    document_id=evidence.document_id,
                    chunk_id=evidence.chunk_id,
                    full_content=text,
                    start_offset=match.start(),
                    end_offset=match.end(),
                    section_path=evidence.section_path,
                    git_commit=evidence.git_commit,
                )
            )

        # Extract tables as atomic units
        for match in self.TABLE_PATTERN.finditer(text):
            spans.append(
                EvidenceSpan.from_substring(
                    document_id=evidence.document_id,
                    chunk_id=evidence.chunk_id,
                    full_content=text,
                    start_offset=match.start(),
                    end_offset=match.end(),
                    section_path=evidence.section_path,
                    git_commit=evidence.git_commit,
                )
            )

        return spans

    def is_inside_code_block(self, text: str, position: int) -> bool:
        """Check if a position is inside a code block."""
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            if match.start() <= position < match.end():
                return True
        return False

    def is_inside_table(self, text: str, position: int) -> bool:
        """Check if a position is inside a table."""
        for match in self.TABLE_PATTERN.finditer(text):
            if match.start() <= position < match.end():
                return True
        return False
