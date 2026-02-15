"""Relationship Inferrer - LLM-based inference of implicit EIP relationships (Phase 11)."""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import anthropic
import structlog

from src.config import DEFAULT_MODEL

logger = structlog.get_logger()


class RelationshipType(Enum):
    """Types of inferred relationships between EIPs."""

    CONFLICTS_WITH = "conflicts_with"
    ADDRESSES_CONCERN = "addresses_concern"
    INSPIRED_BY = "inspired_by"
    BUILDS_ON = "builds_on"
    ALTERNATIVE_TO = "alternative_to"
    RELATED_TO = "related_to"


@dataclass
class InferredRelationship:
    """An inferred relationship between two EIPs."""

    source_eip: int
    target_eip: int
    relationship_type: RelationshipType
    confidence: float
    evidence: list[str]
    reasoning: str
    inferred_at: datetime = field(default_factory=datetime.utcnow)
    model_version: str = ""


class RelationshipInferrer:
    """Infer implicit relationships between EIPs using LLM analysis.

    Analyzes EIP content to detect relationships not explicitly stated
    in frontmatter, such as conceptual dependencies, conflicts, and
    alternative approaches.
    """

    INFERENCE_PROMPT = """Analyze these two Ethereum Improvement Proposals and identify any implicit relationships.

## EIP-{source_eip}
{source_content}

## EIP-{target_eip}
{target_content}

Identify relationships between EIP-{source_eip} and EIP-{target_eip}. Consider:
- CONFLICTS_WITH: Technical incompatibility or contradictory approaches
- ADDRESSES_CONCERN: One addresses a concern or limitation raised about the other
- INSPIRED_BY: Conceptual inspiration without explicit reference
- BUILDS_ON: Extends concepts without explicit "requires" dependency
- ALTERNATIVE_TO: Different approach to same problem
- RELATED_TO: General conceptual relationship

For each relationship found, provide:
1. The relationship type
2. Confidence score (0.0-1.0)
3. Specific evidence from the text (quotes or paraphrases)
4. Reasoning for the inference

Return ONLY a JSON array of relationships:
[
  {{
    "relationship_type": "builds_on" | "conflicts_with" | "addresses_concern" | "inspired_by" | "alternative_to" | "related_to",
    "confidence": 0.0-1.0,
    "evidence": ["quote 1", "quote 2"],
    "reasoning": "explanation of why this relationship exists"
  }}
]

If no meaningful relationships exist, return an empty array: []
Focus on substantive relationships with confidence >= 0.5."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model

        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("no_api_key_provided", component="RelationshipInferrer")

    async def infer_relationships(
        self,
        source_eip: int,
        target_eip: int,
        source_content: str,
        target_content: str,
    ) -> list[InferredRelationship]:
        """Infer relationships between two EIPs based on their content.

        Args:
            source_eip: The source EIP number
            target_eip: The target EIP number
            source_content: Full content of the source EIP
            target_content: Full content of the target EIP

        Returns:
            List of inferred relationships with confidence scores and evidence
        """
        if not self.client:
            logger.error("no_client_available", method="infer_relationships")
            return []

        prompt = self.INFERENCE_PROMPT.format(
            source_eip=source_eip,
            target_eip=target_eip,
            source_content=self._truncate_content(source_content),
            target_content=self._truncate_content(target_content),
        )

        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            result_text = response.content[0].text
            relationships = self._parse_inference_response(result_text, source_eip, target_eip)

            logger.debug(
                "inferred_relationships",
                source_eip=source_eip,
                target_eip=target_eip,
                count=len(relationships),
            )

            return relationships

        except anthropic.APIError as e:
            logger.error(
                "inference_api_error",
                source_eip=source_eip,
                target_eip=target_eip,
                error=str(e),
            )
            return []

    async def infer_from_corpus(
        self,
        eip_pairs: list[tuple[int, int, str, str]],
    ) -> list[InferredRelationship]:
        """Infer relationships for multiple EIP pairs.

        Args:
            eip_pairs: List of (source_eip, target_eip, source_content, target_content) tuples

        Returns:
            All inferred relationships across the corpus
        """
        all_relationships: list[InferredRelationship] = []

        for source_eip, target_eip, source_content, target_content in eip_pairs:
            relationships = await self.infer_relationships(
                source_eip, target_eip, source_content, target_content
            )
            all_relationships.extend(relationships)

        logger.info(
            "corpus_inference_complete",
            pairs_analyzed=len(eip_pairs),
            relationships_found=len(all_relationships),
        )

        return all_relationships

    def _truncate_content(self, content: str, max_chars: int = 8000) -> str:
        """Truncate content to fit within token limits."""
        if len(content) <= max_chars:
            return content
        return content[:max_chars] + "\n\n[Content truncated...]"

    def _parse_inference_response(
        self,
        response_text: str,
        source_eip: int,
        target_eip: int,
    ) -> list[InferredRelationship]:
        """Parse LLM response into InferredRelationship objects."""
        try:
            json_match = re.search(r"\[[\s\S]*\]", response_text)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response_text)

            if not isinstance(parsed, list):
                parsed = [parsed]

            relationships = []
            for item in parsed:
                rel_type_str = item.get("relationship_type", "related_to").lower()
                try:
                    rel_type = RelationshipType(rel_type_str)
                except ValueError:
                    rel_type = RelationshipType.RELATED_TO

                confidence = float(item.get("confidence", 0.5))
                evidence = item.get("evidence", [])
                if isinstance(evidence, str):
                    evidence = [evidence]

                reasoning = item.get("reasoning", "")

                if confidence >= 0.3:
                    relationships.append(
                        InferredRelationship(
                            source_eip=source_eip,
                            target_eip=target_eip,
                            relationship_type=rel_type,
                            confidence=confidence,
                            evidence=evidence,
                            reasoning=reasoning,
                            model_version=self.model,
                        )
                    )

            return relationships

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(
                "failed_to_parse_inference",
                error=str(e),
                response=response_text[:500],
            )
            return []
