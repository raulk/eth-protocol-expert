"""Confidence Scorer - Score and calibrate inferred relationship confidence (Phase 11)."""

from dataclasses import dataclass, field
from typing import Any

import structlog

from src.graph.relationship_inferrer import InferredRelationship, RelationshipType

logger = structlog.get_logger()


@dataclass
class ConfidenceFactors:
    """Factors contributing to relationship confidence."""

    evidence_strength: float = 0.0
    semantic_similarity: float = 0.0
    community_support: float = 0.0
    recency: float = 0.0

    def weighted_average(
        self,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Calculate weighted average of all factors."""
        default_weights = {
            "evidence_strength": 0.4,
            "semantic_similarity": 0.25,
            "community_support": 0.2,
            "recency": 0.15,
        }
        w = weights or default_weights

        total = (
            self.evidence_strength * w["evidence_strength"]
            + self.semantic_similarity * w["semantic_similarity"]
            + self.community_support * w["community_support"]
            + self.recency * w["recency"]
        )

        return min(1.0, max(0.0, total))


@dataclass
class CalibrationData:
    """Data for calibrating confidence scores."""

    known_positive: list[tuple[int, int, RelationshipType]] = field(default_factory=list)
    known_negative: list[tuple[int, int]] = field(default_factory=list)
    adjustment_factor: float = 1.0


class ConfidenceScorer:
    """Score and calibrate confidence for inferred relationships.

    Combines multiple signals:
    - Evidence quality from the inference
    - Semantic similarity between EIPs
    - Community support (forum discussion overlap)
    - Recency of the EIPs
    """

    def __init__(
        self,
        base_weights: dict[str, float] | None = None,
    ):
        self.weights = base_weights or {
            "evidence_strength": 0.4,
            "semantic_similarity": 0.25,
            "community_support": 0.2,
            "recency": 0.15,
        }
        self.calibration = CalibrationData()

    def score(
        self,
        relationship: InferredRelationship,
        semantic_similarity: float | None = None,
        community_support: float | None = None,
        recency_score: float | None = None,
    ) -> float:
        """Calculate overall confidence score for a relationship.

        Args:
            relationship: The inferred relationship to score
            semantic_similarity: Optional precomputed semantic similarity (0-1)
            community_support: Optional community support score (0-1)
            recency_score: Optional recency factor (0-1, 1 = very recent)

        Returns:
            Calibrated confidence score between 0 and 1
        """
        factors = ConfidenceFactors(
            evidence_strength=self._score_evidence(relationship),
            semantic_similarity=semantic_similarity or 0.5,
            community_support=community_support or 0.5,
            recency=recency_score or 0.5,
        )

        raw_score = factors.weighted_average(self.weights)
        calibrated_score = self._apply_calibration(
            raw_score, relationship.source_eip, relationship.target_eip
        )

        logger.debug(
            "scored_relationship",
            source_eip=relationship.source_eip,
            target_eip=relationship.target_eip,
            relationship_type=relationship.relationship_type.value,
            raw_score=raw_score,
            calibrated_score=calibrated_score,
            factors=vars(factors),
        )

        return calibrated_score

    def calibrate(
        self,
        known_relationships: list[dict[str, Any]],
    ) -> None:
        """Calibrate the scorer using known relationships.

        Args:
            known_relationships: List of dicts with keys:
                - source_eip: int
                - target_eip: int
                - relationship_type: str (from RelationshipType)
                - is_valid: bool
        """
        positives = []
        negatives = []

        for rel in known_relationships:
            source = rel.get("source_eip")
            target = rel.get("target_eip")
            is_valid = rel.get("is_valid", True)

            if is_valid:
                rel_type_str = rel.get("relationship_type", "related_to")
                try:
                    rel_type = RelationshipType(rel_type_str)
                except ValueError:
                    rel_type = RelationshipType.RELATED_TO
                positives.append((source, target, rel_type))
            else:
                negatives.append((source, target))

        self.calibration = CalibrationData(
            known_positive=positives,
            known_negative=negatives,
            adjustment_factor=self._compute_adjustment(positives, negatives),
        )

        logger.info(
            "calibration_complete",
            positive_samples=len(positives),
            negative_samples=len(negatives),
            adjustment_factor=self.calibration.adjustment_factor,
        )

    def get_high_confidence(
        self,
        relationships: list[InferredRelationship],
        threshold: float = 0.7,
    ) -> list[InferredRelationship]:
        """Filter relationships to only those with high confidence.

        Args:
            relationships: List of inferred relationships
            threshold: Minimum confidence score (default 0.7)

        Returns:
            Filtered list of high-confidence relationships
        """
        high_conf = []
        for rel in relationships:
            score = self.score(rel)
            if score >= threshold:
                high_conf.append(rel)

        logger.debug(
            "filtered_high_confidence",
            total=len(relationships),
            high_confidence=len(high_conf),
            threshold=threshold,
        )

        return high_conf

    def _score_evidence(self, relationship: InferredRelationship) -> float:
        """Score the quality of evidence for a relationship."""
        base_score = relationship.confidence

        evidence_count = len(relationship.evidence)
        evidence_bonus = min(evidence_count * 0.05, 0.2)

        reasoning_length = len(relationship.reasoning)
        reasoning_bonus = min(reasoning_length / 500, 0.1)

        return min(1.0, base_score + evidence_bonus + reasoning_bonus)

    def _apply_calibration(
        self,
        raw_score: float,
        source_eip: int,
        target_eip: int,
    ) -> float:
        """Apply calibration adjustments to raw score."""
        for neg_source, neg_target in self.calibration.known_negative:
            if neg_source == source_eip and neg_target == target_eip:
                return raw_score * 0.3

        for pos_source, pos_target, _ in self.calibration.known_positive:
            if pos_source == source_eip and pos_target == target_eip:
                return min(1.0, raw_score * 1.2)

        return raw_score * self.calibration.adjustment_factor

    def _compute_adjustment(
        self,
        positives: list[tuple[int, int, RelationshipType]],
        negatives: list[tuple[int, int]],
    ) -> float:
        """Compute global adjustment factor from calibration data."""
        if not positives and not negatives:
            return 1.0

        ratio = len(positives) / max(len(positives) + len(negatives), 1)
        return 0.8 + (ratio * 0.4)
