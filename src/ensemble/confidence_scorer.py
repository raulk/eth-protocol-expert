"""Confidence Scorer - Score retrieval confidence for ensemble decisions (Phase 12)."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import structlog

logger = structlog.get_logger()


@dataclass
class ConfidenceFactors:
    """Factors contributing to retrieval confidence score."""

    relevance_scores: list[float] = field(default_factory=list)
    coverage: float = 0.0
    diversity: float = 0.0
    source_authority: float = 0.0
    query_specificity: float = 0.0
    result_count: int = 0


@dataclass
class RetrievalConfidence:
    """Retrieval confidence assessment result."""

    score: float
    factors: ConfidenceFactors
    recommendation: str
    is_sufficient: bool


class RetrievalConfidenceScorer:
    """Score confidence of retrieval results for ensemble decision-making.

    Evaluates retrieval quality based on:
    - Relevance: How relevant are the retrieved chunks to the query
    - Coverage: Do results cover all aspects of the query
    - Diversity: Are sources diverse (not all from same document)
    - Authority: Are sources from authoritative documents (EIPs vs forum posts)
    """

    AUTHORITY_WEIGHTS: ClassVar[dict[str, float]] = {
        "eip": 1.0,
        "erc": 1.0,
        "execution-specs": 0.95,
        "consensus-specs": 0.95,
        "ethereum-research": 0.8,
        "eth-magicians": 0.7,
        "acd-transcript": 0.75,
        "arxiv": 0.85,
    }

    def __init__(
        self,
        relevance_threshold: float = 0.6,
        min_results: int = 3,
        diversity_weight: float = 0.2,
        authority_weight: float = 0.2,
    ):
        self.relevance_threshold = relevance_threshold
        self.min_results = min_results
        self.diversity_weight = diversity_weight
        self.authority_weight = authority_weight

    def score(
        self,
        query: str,
        results: list[Any],
        relevance_scores: list[float] | None = None,
    ) -> RetrievalConfidence:
        """Score the confidence of retrieval results.

        Args:
            query: The original query
            results: List of retrieval results (HybridResult, SearchResult, etc.)
            relevance_scores: Optional pre-computed relevance scores

        Returns:
            RetrievalConfidence with overall score and factors
        """
        if not results:
            factors = ConfidenceFactors(result_count=0)
            return RetrievalConfidence(
                score=0.0,
                factors=factors,
                recommendation="No results retrieved. Consider broadening the query.",
                is_sufficient=False,
            )

        scores = relevance_scores or self._extract_relevance_scores(results)

        relevance_score = self._calculate_relevance_score(scores)
        coverage_score = self._calculate_coverage_score(query, results)
        diversity_score = self._calculate_diversity_score(results)
        authority_score = self._calculate_authority_score(results)
        specificity_score = self._calculate_query_specificity(query)

        factors = ConfidenceFactors(
            relevance_scores=scores,
            coverage=coverage_score,
            diversity=diversity_score,
            source_authority=authority_score,
            query_specificity=specificity_score,
            result_count=len(results),
        )

        overall_score = self._calculate_overall_score(factors)
        recommendation = self._generate_recommendation(factors, overall_score)
        is_sufficient = self.is_sufficient(
            RetrievalConfidence(
                score=overall_score,
                factors=factors,
                recommendation=recommendation,
                is_sufficient=True,
            )
        )

        logger.debug(
            "retrieval_confidence_scored",
            query=query[:50],
            overall_score=f"{overall_score:.3f}",
            relevance=f"{relevance_score:.3f}",
            coverage=f"{coverage_score:.3f}",
            diversity=f"{diversity_score:.3f}",
            authority=f"{authority_score:.3f}",
            result_count=len(results),
        )

        return RetrievalConfidence(
            score=overall_score,
            factors=factors,
            recommendation=recommendation,
            is_sufficient=is_sufficient,
        )

    def is_sufficient(
        self,
        confidence: RetrievalConfidence,
        threshold: float = 0.7,
    ) -> bool:
        """Check if retrieval confidence meets threshold.

        Args:
            confidence: RetrievalConfidence to evaluate
            threshold: Minimum acceptable confidence score

        Returns:
            True if confidence is sufficient
        """
        if confidence.factors.result_count < self.min_results:
            return False

        if not confidence.factors.relevance_scores:
            return False

        top_relevance = max(confidence.factors.relevance_scores[:3], default=0)
        if top_relevance < self.relevance_threshold:
            return False

        return confidence.score >= threshold

    def suggest_improvement(self, confidence: RetrievalConfidence) -> str:
        """Suggest how to improve retrieval based on confidence factors.

        Args:
            confidence: RetrievalConfidence to analyze

        Returns:
            Suggestion string for improvement
        """
        suggestions = []

        factors = confidence.factors

        if factors.result_count < self.min_results:
            suggestions.append("Try broader search terms or relax filters")

        if factors.relevance_scores:
            avg_relevance = sum(factors.relevance_scores) / len(factors.relevance_scores)
            if avg_relevance < self.relevance_threshold:
                suggestions.append(
                    "Rephrase query to be more specific to Ethereum protocol concepts"
                )

        if factors.coverage < 0.5:
            suggestions.append(
                "Query may contain multiple concepts - consider decomposing into sub-queries"
            )

        if factors.diversity < 0.3:
            suggestions.append(
                "Results are concentrated in few sources - consider expanding source types"
            )

        if factors.source_authority < 0.5:
            suggestions.append("Prioritize authoritative sources like EIPs and execution specs")

        if not suggestions:
            if confidence.score < 0.8:
                suggestions.append("Consider using ensemble models for more thorough analysis")
            else:
                suggestions.append("Retrieval quality is good")

        return "; ".join(suggestions)

    def _extract_relevance_scores(self, results: list[Any]) -> list[float]:
        """Extract relevance scores from various result types."""
        scores = []

        for result in results:
            score = None

            if hasattr(result, "rrf_score"):
                score = min(result.rrf_score * 30, 1.0)
            elif hasattr(result, "similarity"):
                score = result.similarity
            elif hasattr(result, "score"):
                score = result.score
            elif hasattr(result, "relevance"):
                score = result.relevance

            if score is not None:
                scores.append(float(score))
            else:
                scores.append(0.5)

        return scores

    def _calculate_relevance_score(self, scores: list[float]) -> float:
        """Calculate weighted relevance score (top results weighted higher)."""
        if not scores:
            return 0.0

        weights = [1.0 / (i + 1) for i in range(len(scores))]
        total_weight = sum(weights)

        weighted_sum = sum(s * w for s, w in zip(scores, weights, strict=True))
        return weighted_sum / total_weight

    def _calculate_coverage_score(self, query: str, results: list[Any]) -> float:
        """Estimate how well results cover query concepts."""
        query_terms = set(query.lower().split())
        query_terms -= {
            "what",
            "how",
            "why",
            "when",
            "which",
            "is",
            "are",
            "the",
            "a",
            "an",
            "in",
            "of",
            "to",
            "for",
        }

        if not query_terms:
            return 1.0

        covered_terms = set()
        for result in results:
            content = self._get_content(result).lower()
            for term in query_terms:
                if term in content:
                    covered_terms.add(term)

        return len(covered_terms) / len(query_terms) if query_terms else 1.0

    def _calculate_diversity_score(self, results: list[Any]) -> float:
        """Calculate source diversity (unique documents / total results)."""
        if not results:
            return 0.0

        unique_docs = set()
        for result in results:
            doc_id = self._get_document_id(result)
            if doc_id:
                unique_docs.add(doc_id)

        if len(unique_docs) <= 1:
            return 0.2

        diversity_ratio = len(unique_docs) / len(results)
        return min(diversity_ratio * 1.5, 1.0)

    def _calculate_authority_score(self, results: list[Any]) -> float:
        """Calculate average source authority."""
        if not results:
            return 0.0

        authority_scores = []
        for result in results:
            doc_id = self._get_document_id(result) or ""
            doc_lower = doc_id.lower()

            authority = 0.5
            for source_type, weight in self.AUTHORITY_WEIGHTS.items():
                if source_type in doc_lower:
                    authority = weight
                    break

            authority_scores.append(authority)

        return sum(authority_scores) / len(authority_scores)

    def _calculate_query_specificity(self, query: str) -> float:
        """Estimate query specificity (more specific = better retrieval expected)."""
        words = query.split()
        word_count = len(words)

        has_eip_ref = any("eip" in w.lower() for w in words)
        has_technical_terms = any(
            term in query.lower()
            for term in ["gas", "block", "transaction", "validator", "beacon", "blob", "calldata"]
        )

        base_score = min(word_count / 15, 1.0)

        if has_eip_ref:
            base_score = min(base_score + 0.2, 1.0)
        if has_technical_terms:
            base_score = min(base_score + 0.1, 1.0)

        return base_score

    def _calculate_overall_score(self, factors: ConfidenceFactors) -> float:
        """Calculate overall confidence score from factors."""
        if not factors.relevance_scores:
            return 0.0

        relevance_weight = 0.4
        coverage_weight = 0.2
        diversity_weight = self.diversity_weight
        authority_weight = self.authority_weight

        relevance = self._calculate_relevance_score(factors.relevance_scores)

        score = (
            relevance * relevance_weight
            + factors.coverage * coverage_weight
            + factors.diversity * diversity_weight
            + factors.source_authority * authority_weight
        )

        if factors.result_count < self.min_results:
            score *= factors.result_count / self.min_results

        return min(score, 1.0)

    def _generate_recommendation(self, factors: ConfidenceFactors, score: float) -> str:
        """Generate recommendation based on confidence analysis."""
        if score >= 0.85:
            return "High confidence - single model response should be sufficient"

        if score >= 0.7:
            return "Good confidence - consider ensemble for critical queries"

        if score >= 0.5:
            return "Moderate confidence - ensemble recommended for better coverage"

        return "Low confidence - ensemble strongly recommended; consider query reformulation"

    def _get_content(self, result: Any) -> str:
        """Extract content from various result types."""
        if hasattr(result, "chunk"):
            if hasattr(result.chunk, "content"):
                return result.chunk.content
        if hasattr(result, "content"):
            return result.content
        if hasattr(result, "text"):
            return result.text
        return ""

    def _get_document_id(self, result: Any) -> str | None:
        """Extract document ID from various result types."""
        if hasattr(result, "chunk"):
            if hasattr(result.chunk, "document_id"):
                return result.chunk.document_id
        if hasattr(result, "document_id"):
            return result.document_id
        if hasattr(result, "doc_id"):
            return result.doc_id
        return None
