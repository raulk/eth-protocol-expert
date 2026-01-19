"""Incremental Updater - Re-infer relationships on corpus changes (Phase 11)."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from src.graph.falkordb_store import FalkorDBStore
from src.graph.relationship_inferrer import InferredRelationship, RelationshipInferrer

logger = structlog.get_logger()


class ChangeType(Enum):
    """Type of corpus change."""

    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"


@dataclass
class CorpusChange:
    """Represents a change to the EIP corpus."""

    change_type: ChangeType
    eip_number: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    content_hash: str | None = None
    previous_hash: str | None = None


@dataclass
class InferenceMetadata:
    """Metadata for stored inferences."""

    source_eip: int
    target_eip: int
    relationship_type: str
    confidence: float
    inferred_at: datetime
    source_content_hash: str | None
    target_content_hash: str | None


@dataclass
class UpdateResult:
    """Result of an incremental update."""

    change: CorpusChange
    invalidated_count: int = 0
    revalidated_count: int = 0
    new_inferences: list[InferredRelationship] = field(default_factory=list)
    processing_time_ms: float = 0.0


class IncrementalUpdater:
    """Update inferred relationships when the EIP corpus changes.

    Tracks when inferences were made and invalidates them when
    source content changes. Supports incremental re-inference.
    """

    def __init__(
        self,
        inferrer: RelationshipInferrer,
        store: FalkorDBStore,
    ):
        self.inferrer = inferrer
        self.store = store

    async def update_on_change(
        self,
        change: CorpusChange,
        content: str | None = None,
        related_eips: list[tuple[int, str]] | None = None,
    ) -> UpdateResult:
        """Update inferences when an EIP changes.

        Args:
            change: The corpus change that occurred
            content: New content of the changed EIP (for ADDED/MODIFIED)
            related_eips: List of (eip_number, content) for EIPs to compare against

        Returns:
            UpdateResult with statistics and new inferences
        """
        start_time = datetime.utcnow()
        result = UpdateResult(change=change)

        if change.change_type == ChangeType.REMOVED:
            result.invalidated_count = self._invalidate_for_eip(change.eip_number)
            logger.info(
                "invalidated_relationships_on_removal",
                eip=change.eip_number,
                count=result.invalidated_count,
            )

        elif change.change_type in (ChangeType.ADDED, ChangeType.MODIFIED):
            result.invalidated_count = self._invalidate_for_eip(change.eip_number)

            if content and related_eips:
                for target_eip, target_content in related_eips:
                    if target_eip == change.eip_number:
                        continue

                    new_rels = await self.inferrer.infer_relationships(
                        source_eip=change.eip_number,
                        target_eip=target_eip,
                        source_content=content,
                        target_content=target_content,
                    )

                    for rel in new_rels:
                        self._store_inference(rel, change.content_hash)
                        result.new_inferences.append(rel)

            logger.info(
                "updated_relationships_on_change",
                eip=change.eip_number,
                change_type=change.change_type.value,
                invalidated=result.invalidated_count,
                new_inferences=len(result.new_inferences),
            )

        end_time = datetime.utcnow()
        result.processing_time_ms = (end_time - start_time).total_seconds() * 1000

        return result

    async def revalidate_affected(
        self,
        eip_number: int,
        eip_content: str,
        related_eips: list[tuple[int, str]],
    ) -> list[InferredRelationship]:
        """Re-infer relationships for an EIP and its related EIPs.

        Args:
            eip_number: The EIP to revalidate
            eip_content: Current content of the EIP
            related_eips: List of (eip_number, content) for related EIPs

        Returns:
            List of newly inferred relationships
        """
        self._invalidate_for_eip(eip_number)

        all_inferences: list[InferredRelationship] = []

        for target_eip, target_content in related_eips:
            if target_eip == eip_number:
                continue

            inferences = await self.inferrer.infer_relationships(
                source_eip=eip_number,
                target_eip=target_eip,
                source_content=eip_content,
                target_content=target_content,
            )

            for rel in inferences:
                self._store_inference(rel)
                all_inferences.append(rel)

            reverse_inferences = await self.inferrer.infer_relationships(
                source_eip=target_eip,
                target_eip=eip_number,
                source_content=target_content,
                target_content=eip_content,
            )

            for rel in reverse_inferences:
                self._store_inference(rel)
                all_inferences.append(rel)

        logger.info(
            "revalidated_relationships",
            eip=eip_number,
            related_count=len(related_eips),
            new_inferences=len(all_inferences),
        )

        return all_inferences

    def get_stale_inferences(
        self,
        max_age_days: int = 30,
    ) -> list[InferenceMetadata]:
        """Get inferences that are older than the specified age.

        Args:
            max_age_days: Maximum age in days

        Returns:
            List of stale inference metadata
        """
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

        cypher = """
            MATCH (source:EIP)-[r:INFERRED]->(target:EIP)
            WHERE r.inferred_at < $cutoff
            RETURN source.number, target.number, r.relationship_type,
                   r.confidence, r.inferred_at, r.source_hash, r.target_hash
        """

        result = self.store.query(
            cypher,
            params={"cutoff": cutoff_date.isoformat()},
        )

        stale = []
        for row in result.result_set:
            inferred_at = row[4]
            if isinstance(inferred_at, str):
                inferred_at = datetime.fromisoformat(inferred_at)

            stale.append(
                InferenceMetadata(
                    source_eip=row[0],
                    target_eip=row[1],
                    relationship_type=row[2],
                    confidence=row[3],
                    inferred_at=inferred_at,
                    source_content_hash=row[5],
                    target_content_hash=row[6],
                )
            )

        logger.debug(
            "found_stale_inferences",
            max_age_days=max_age_days,
            count=len(stale),
        )

        return stale

    def get_inference_stats(self) -> dict[str, Any]:
        """Get statistics about stored inferences."""
        count_query = """
            MATCH ()-[r:INFERRED]->()
            RETURN count(r), avg(r.confidence)
        """
        count_result = self.store.query(count_query)

        type_query = """
            MATCH ()-[r:INFERRED]->()
            RETURN r.relationship_type, count(r)
        """
        type_result = self.store.query(type_query)

        type_counts = {row[0]: row[1] for row in type_result.result_set}

        total = count_result.result_set[0][0] if count_result.result_set else 0
        avg_confidence = count_result.result_set[0][1] if count_result.result_set else 0

        return {
            "total_inferences": total,
            "average_confidence": avg_confidence,
            "by_type": type_counts,
        }

    def _invalidate_for_eip(self, eip_number: int) -> int:
        """Remove all inferences involving an EIP."""
        cypher = """
            MATCH (e:EIP {number: $number})-[r:INFERRED]-()
            DELETE r
            RETURN count(r)
        """
        result = self.store.query(cypher, params={"number": eip_number})

        count = result.result_set[0][0] if result.result_set else 0
        return count

    def _store_inference(
        self,
        relationship: InferredRelationship,
        content_hash: str | None = None,
    ) -> None:
        """Store an inferred relationship in the graph database."""
        cypher = """
            MATCH (source:EIP {number: $source_eip})
            MATCH (target:EIP {number: $target_eip})
            MERGE (source)-[r:INFERRED {relationship_type: $rel_type}]->(target)
            ON CREATE SET
                r.confidence = $confidence,
                r.evidence = $evidence,
                r.reasoning = $reasoning,
                r.inferred_at = $inferred_at,
                r.model_version = $model_version,
                r.source_hash = $source_hash
            ON MATCH SET
                r.confidence = $confidence,
                r.evidence = $evidence,
                r.reasoning = $reasoning,
                r.inferred_at = $inferred_at,
                r.model_version = $model_version,
                r.source_hash = $source_hash
            RETURN r
        """

        self.store.query(
            cypher,
            params={
                "source_eip": relationship.source_eip,
                "target_eip": relationship.target_eip,
                "rel_type": relationship.relationship_type.value,
                "confidence": relationship.confidence,
                "evidence": relationship.evidence,
                "reasoning": relationship.reasoning,
                "inferred_at": relationship.inferred_at.isoformat(),
                "model_version": relationship.model_version,
                "source_hash": content_hash,
            },
        )
