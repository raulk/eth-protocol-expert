"""Quality Scorer - Assess PDF extraction quality."""

import re
from dataclasses import dataclass, field
from typing import ClassVar

import structlog

from .pdf_extractor import PDFContent, PDFSection

logger = structlog.get_logger()


@dataclass
class QualityScore:
    """Quality assessment for extracted PDF content."""

    overall: float  # 0-1 composite score
    text_quality: float  # 0-1 text extraction quality
    structure_quality: float  # 0-1 structure preservation
    issues: list[str] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)


class QualityScorer:
    """Score the quality of PDF extraction.

    Detects common extraction issues:
    - OCR artifacts and garbled text
    - Missing sections
    - Poor structure preservation
    - Encoding problems
    """

    # Common OCR artifact patterns
    OCR_ARTIFACTS: ClassVar[list[str]] = [
        r"[^\x00-\x7F]{3,}",  # Long sequences of non-ASCII
        r"[\ufffd]{2,}",  # Unicode replacement characters
        r"[|l1I]{4,}",  # Common OCR confusion patterns
        r"\s{3,}",  # Excessive whitespace
        r"[^\w\s]{5,}",  # Long symbol sequences
    ]

    # Expected sections in academic papers
    EXPECTED_SECTIONS: ClassVar[list[str]] = [
        "abstract",
        "introduction",
        "conclusion",
        "references",
    ]

    def score(self, pdf_content: PDFContent) -> QualityScore:
        """Compute quality score for extracted PDF content.

        Args:
            pdf_content: Extracted PDF content to score

        Returns:
            QualityScore with overall score and breakdown
        """
        issues = []
        details = {}

        # Check text quality
        text_quality, text_issues = self.check_text_extraction(pdf_content.full_text)
        details["text_extraction"] = text_quality
        issues.extend(text_issues)

        # Check structure quality
        structure_quality, struct_issues = self.check_structure(pdf_content.sections)
        details["structure"] = structure_quality
        issues.extend(struct_issues)

        # Check content completeness
        completeness, complete_issues = self._check_completeness(pdf_content)
        details["completeness"] = completeness
        issues.extend(complete_issues)

        # Check encoding
        encoding_quality, enc_issues = self._check_encoding(pdf_content.full_text)
        details["encoding"] = encoding_quality
        issues.extend(enc_issues)

        # Compute overall score (weighted average)
        overall = (
            text_quality * 0.4
            + structure_quality * 0.3
            + completeness * 0.2
            + encoding_quality * 0.1
        )

        logger.debug(
            "scored_pdf_quality",
            title=pdf_content.title[:50] if pdf_content.title else "untitled",
            overall=round(overall, 3),
            issues=len(issues),
        )

        return QualityScore(
            overall=overall,
            text_quality=text_quality,
            structure_quality=structure_quality,
            issues=issues,
            details=details,
        )

    def check_text_extraction(self, content: str) -> tuple[float, list[str]]:
        """Check quality of text extraction.

        Args:
            content: Full text content

        Returns:
            Tuple of (score 0-1, list of issues)
        """
        if not content:
            return 0.0, ["No text content extracted"]

        issues = []
        penalties = 0.0

        # Check for OCR artifacts
        total_artifacts = 0
        for pattern in self.OCR_ARTIFACTS:
            matches = re.findall(pattern, content)
            total_artifacts += len(matches)

        artifact_ratio = total_artifacts / max(len(content), 1) * 1000
        if artifact_ratio > 10:
            issues.append(f"High OCR artifact density: {artifact_ratio:.1f} per 1000 chars")
            penalties += min(0.3, artifact_ratio / 100)

        # Check word density (real text should have reasonable word count)
        words = re.findall(r"\b\w+\b", content)
        word_density = len(words) / max(len(content), 1) * 100

        if word_density < 10:
            issues.append(f"Low word density: {word_density:.1f}%")
            penalties += 0.2
        elif word_density > 25:
            issues.append(f"Unusual word density: {word_density:.1f}%")
            penalties += 0.1

        # Check for garbled text (random character sequences)
        garbled_pattern = r"[bcdfghjklmnpqrstvwxz]{6,}"  # No vowels
        garbled = re.findall(garbled_pattern, content.lower())
        if len(garbled) > 5:
            issues.append(f"Possible garbled text: {len(garbled)} suspicious sequences")
            penalties += 0.2

        # Check sentence structure
        sentences = re.split(r"[.!?]+", content)
        valid_sentences = sum(
            1 for s in sentences if 10 < len(s.split()) < 100 and re.search(r"[a-zA-Z]", s)
        )
        sentence_ratio = valid_sentences / max(len(sentences), 1)

        if sentence_ratio < 0.3:
            issues.append(f"Low valid sentence ratio: {sentence_ratio:.1%}")
            penalties += 0.2

        return max(0.0, 1.0 - penalties), issues

    def check_structure(self, sections: list[PDFSection]) -> tuple[float, list[str]]:
        """Check quality of structure preservation.

        Args:
            sections: List of extracted sections

        Returns:
            Tuple of (score 0-1, list of issues)
        """
        if not sections:
            return 0.0, ["No sections extracted"]

        issues = []
        score = 1.0

        # Check for expected sections
        section_names = [s.heading.lower() for s in sections]
        missing_sections = []
        for expected in self.EXPECTED_SECTIONS:
            found = any(expected in name for name in section_names)
            if not found:
                missing_sections.append(expected)

        if missing_sections:
            penalty = len(missing_sections) * 0.1
            score -= penalty
            issues.append(f"Missing expected sections: {', '.join(missing_sections)}")

        # Check section content quality
        empty_sections = sum(1 for s in sections if len(s.content.strip()) < 50)
        if empty_sections > 0:
            score -= empty_sections * 0.05
            issues.append(f"{empty_sections} sections have minimal content")

        # Check section ordering (abstract should be early, references late)
        section_order = {s.heading.lower(): i for i, s in enumerate(sections)}
        if "abstract" in section_order and "references" in section_order:
            if section_order["abstract"] > section_order["references"]:
                score -= 0.2
                issues.append("Section order appears incorrect")

        return max(0.0, score), issues

    def _check_completeness(self, pdf_content: PDFContent) -> tuple[float, list[str]]:
        """Check content completeness."""
        issues = []
        score = 1.0

        # Check total content length
        total_chars = len(pdf_content.full_text)
        expected_min = pdf_content.page_count * 500  # ~500 chars per page minimum

        if total_chars < expected_min:
            ratio = total_chars / max(expected_min, 1)
            score -= (1 - ratio) * 0.3
            issues.append(
                f"Content seems sparse: {total_chars} chars for {pdf_content.page_count} pages"
            )

        # Check references
        if not pdf_content.references:
            score -= 0.1
            issues.append("No references extracted")
        elif len(pdf_content.references) < 5:
            score -= 0.05
            issues.append(f"Few references extracted: {len(pdf_content.references)}")

        # Check title
        if not pdf_content.title or pdf_content.title == "Untitled":
            score -= 0.1
            issues.append("No title extracted")

        return max(0.0, score), issues

    def _check_encoding(self, content: str) -> tuple[float, list[str]]:
        """Check for encoding issues."""
        issues = []
        score = 1.0

        # Check for replacement characters
        replacement_count = content.count("\ufffd")
        if replacement_count > 10:
            score -= min(0.3, replacement_count / 100)
            issues.append(f"Encoding issues: {replacement_count} replacement characters")

        # Check for null bytes or control characters
        control_chars = sum(1 for c in content if ord(c) < 32 and c not in "\n\r\t")
        if control_chars > 10:
            score -= min(0.2, control_chars / 100)
            issues.append(f"Control characters found: {control_chars}")

        # Check for mojibake patterns (common encoding corruption)
        mojibake_patterns = [
            r"Ã©",  # e acute in UTF-8 read as Latin-1
            r"Ã¨",  # e grave
            r"â€™",  # right single quote
            r'â€"',  # em dash
        ]
        mojibake_count = sum(len(re.findall(p, content)) for p in mojibake_patterns)
        if mojibake_count > 5:
            score -= min(0.2, mojibake_count / 50)
            issues.append(f"Possible encoding corruption: {mojibake_count} mojibake patterns")

        return max(0.0, score), issues

    def is_acceptable(self, score: QualityScore, threshold: float = 0.6) -> bool:
        """Check if quality score meets acceptable threshold.

        Args:
            score: Quality score to check
            threshold: Minimum acceptable overall score

        Returns:
            True if score meets threshold
        """
        return score.overall >= threshold

    def get_improvement_suggestions(self, score: QualityScore) -> list[str]:
        """Get suggestions for improving extraction quality.

        Args:
            score: Quality score with issues

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        if score.text_quality < 0.6:
            suggestions.append(
                "Text extraction quality is low. Consider using OCR-specific extraction "
                "or a different PDF library."
            )

        if score.structure_quality < 0.6:
            suggestions.append(
                "Document structure was not well preserved. The PDF may have non-standard "
                "formatting or be image-based."
            )

        if score.details.get("encoding", 1.0) < 0.8:
            suggestions.append(
                "Encoding issues detected. Try re-extracting with explicit UTF-8 handling "
                "or check the original PDF encoding."
            )

        if score.details.get("completeness", 1.0) < 0.7:
            suggestions.append(
                "Content extraction seems incomplete. The PDF may have protected content "
                "or use complex layouts."
            )

        if not suggestions:
            suggestions.append("Quality is acceptable. No major improvements needed.")

        return suggestions
