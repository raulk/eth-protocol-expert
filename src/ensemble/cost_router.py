"""Cost Router - Route queries by complexity and cost to appropriate model tier (Phase 12)."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import structlog

from src.config import MODEL_BALANCED, MODEL_FAST, MODEL_POWERFUL
from src.routing.query_classifier import ClassificationResult, QueryClassifier, QueryType

logger = structlog.get_logger()


class ModelTier(Enum):
    """Model tier classification by capability and cost."""

    FAST = "fast"
    BALANCED = "balanced"
    POWERFUL = "powerful"


@dataclass
class RoutingDecision:
    """Result from cost-based routing decision."""

    model_tier: ModelTier
    model_id: str
    estimated_cost: float
    reasoning: str
    query_classification: ClassificationResult | None = None


@dataclass
class ModelConfig:
    """Configuration for a model including pricing."""

    model_id: str
    tier: ModelTier
    input_cost_per_1k: float
    output_cost_per_1k: float
    max_output_tokens: int = 4096


class CostRouter:
    """Route queries to appropriate model tier based on complexity and cost.

    Routing logic:
    - Simple factual queries -> FAST (haiku) - quick, cheap
    - Standard queries -> BALANCED (sonnet) - good balance
    - Complex synthesis, multi-hop -> POWERFUL (opus) - best quality
    """

    MODEL_CONFIGS: ClassVar[dict[ModelTier, ModelConfig]] = {
        ModelTier.FAST: ModelConfig(
            model_id=MODEL_FAST,
            tier=ModelTier.FAST,
            input_cost_per_1k=0.0008,
            output_cost_per_1k=0.004,
            max_output_tokens=8192,
        ),
        ModelTier.BALANCED: ModelConfig(
            model_id=MODEL_BALANCED,
            tier=ModelTier.BALANCED,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            max_output_tokens=16384,
        ),
        ModelTier.POWERFUL: ModelConfig(
            model_id=MODEL_POWERFUL,
            tier=ModelTier.POWERFUL,
            input_cost_per_1k=0.015,
            output_cost_per_1k=0.075,
            max_output_tokens=16384,
        ),
    }

    COMPLEXITY_TIER_MAP: ClassVar[dict[QueryType, ModelTier]] = {
        QueryType.SIMPLE: ModelTier.FAST,
        QueryType.COMPLEX: ModelTier.BALANCED,
        QueryType.MULTI_HOP: ModelTier.POWERFUL,
    }

    def __init__(
        self,
        default_tier: ModelTier = ModelTier.BALANCED,
        query_classifier: QueryClassifier | None = None,
        api_key: str | None = None,
    ):
        self.default_tier = default_tier
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.query_classifier = query_classifier or QueryClassifier(
            api_key=self.api_key,
            use_llm=False,
        )

    async def route(
        self,
        query: str,
        context_tokens: int = 0,
        force_tier: ModelTier | None = None,
    ) -> RoutingDecision:
        """Route query to appropriate model tier.

        Args:
            query: The user's query
            context_tokens: Estimated context tokens for cost calculation
            force_tier: Optional override to force a specific tier

        Returns:
            RoutingDecision with selected model and reasoning
        """
        if force_tier:
            config = self.MODEL_CONFIGS[force_tier]
            estimated_cost = self.estimate_cost(
                config.model_id,
                context_tokens + len(query.split()) * 2,
                config.max_output_tokens // 2,
            )
            return RoutingDecision(
                model_tier=force_tier,
                model_id=config.model_id,
                estimated_cost=estimated_cost,
                reasoning=f"Forced to {force_tier.value} tier",
            )

        classification = await self.query_classifier.classify(query)
        tier = self._determine_tier(classification, context_tokens)
        config = self.MODEL_CONFIGS[tier]

        input_tokens = context_tokens + len(query.split()) * 2
        estimated_output = self._estimate_output_tokens(classification)
        estimated_cost = self.estimate_cost(config.model_id, input_tokens, estimated_output)

        reasoning = self._build_reasoning(classification, tier, context_tokens)

        logger.info(
            "query_routed",
            query=query[:50],
            tier=tier.value,
            model=config.model_id,
            query_type=classification.query_type.value,
            estimated_cost=f"${estimated_cost:.6f}",
        )

        return RoutingDecision(
            model_tier=tier,
            model_id=config.model_id,
            estimated_cost=estimated_cost,
            reasoning=reasoning,
            query_classification=classification,
        )

    def route_sync(self, query: str, context_tokens: int = 0) -> RoutingDecision:
        """Synchronous routing using rule-based classification only."""
        classification = self.query_classifier.classify_sync(query)
        tier = self._determine_tier(classification, context_tokens)
        config = self.MODEL_CONFIGS[tier]

        input_tokens = context_tokens + len(query.split()) * 2
        estimated_output = self._estimate_output_tokens(classification)
        estimated_cost = self.estimate_cost(config.model_id, input_tokens, estimated_output)

        reasoning = self._build_reasoning(classification, tier, context_tokens)

        return RoutingDecision(
            model_tier=tier,
            model_id=config.model_id,
            estimated_cost=estimated_cost,
            reasoning=reasoning,
            query_classification=classification,
        )

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a given model and token counts.

        Args:
            model: Model ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        config = self._get_config_by_model_id(model)
        if not config:
            config = self.MODEL_CONFIGS[self.default_tier]

        input_cost = (input_tokens / 1000) * config.input_cost_per_1k
        output_cost = (output_tokens / 1000) * config.output_cost_per_1k

        return input_cost + output_cost

    def get_model_for_tier(self, tier: ModelTier) -> str:
        """Get the model ID for a given tier."""
        return self.MODEL_CONFIGS[tier].model_id

    def _determine_tier(
        self,
        classification: ClassificationResult,
        context_tokens: int,
    ) -> ModelTier:
        """Determine appropriate tier based on classification and context size."""
        base_tier = self.COMPLEXITY_TIER_MAP.get(
            classification.query_type,
            self.default_tier,
        )

        if context_tokens > 50000:
            if base_tier == ModelTier.FAST:
                return ModelTier.BALANCED
            return base_tier

        if classification.confidence < 0.5:
            if base_tier == ModelTier.FAST:
                return ModelTier.BALANCED

        return base_tier

    def _estimate_output_tokens(self, classification: ClassificationResult) -> int:
        """Estimate expected output tokens based on query complexity."""
        if classification.query_type == QueryType.SIMPLE:
            return 500
        if classification.query_type == QueryType.COMPLEX:
            return 1500
        return 2500

    def _build_reasoning(
        self,
        classification: ClassificationResult,
        tier: ModelTier,
        context_tokens: int,
    ) -> str:
        """Build human-readable reasoning for the routing decision."""
        parts = [
            f"Query classified as {classification.query_type.value}",
            f"(confidence: {classification.confidence:.2f})",
        ]

        if context_tokens > 0:
            parts.append(f"with {context_tokens:,} context tokens")

        parts.append(f"-> routed to {tier.value} tier")

        if classification.needs_decomposition:
            parts.append(
                f"(needs decomposition into ~{classification.estimated_sub_questions} sub-questions)"
            )

        return " ".join(parts)

    def _get_config_by_model_id(self, model_id: str) -> ModelConfig | None:
        """Get model config by model ID."""
        for config in self.MODEL_CONFIGS.values():
            if config.model_id == model_id:
                return config
        return None
