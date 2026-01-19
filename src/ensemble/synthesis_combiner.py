"""Synthesis Combiner - Combine outputs from multiple models (Phase 12)."""

import os
import re
from dataclasses import dataclass, field

import anthropic
import structlog

from src.ensemble.multi_model_runner import ModelRun

logger = structlog.get_logger()


@dataclass
class CombinedOutput:
    """Combined output from multiple model runs."""

    final_response: str
    contributing_models: list[str]
    confidence: float
    citations: list[dict] = field(default_factory=list)
    strategy_used: str = ""
    total_cost: float = 0.0
    synthesis_cost: float = 0.0


class SynthesisCombiner:
    """Combine outputs from multiple models into a single response.

    Strategies:
    - best: Select the highest quality response
    - merge: Combine unique information from all responses
    - vote: Use consensus across responses for factual claims
    """

    def __init__(
        self,
        api_key: str | None = None,
        synthesis_model: str = "claude-sonnet-4-20250514",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.synthesis_model = synthesis_model

        if self.api_key:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        else:
            self.client = None

    async def combine(
        self,
        outputs: list[ModelRun],
        strategy: str = "best",
        query: str | None = None,
    ) -> CombinedOutput:
        """Combine multiple model outputs using specified strategy.

        Args:
            outputs: List of ModelRun results to combine
            strategy: Combination strategy - "best", "merge", or "vote"
            query: Original query (used for merge/vote strategies)

        Returns:
            CombinedOutput with final synthesized response
        """
        valid_outputs = [o for o in outputs if o.response and not o.error]

        if not valid_outputs:
            return CombinedOutput(
                final_response="Unable to generate response - all models failed.",
                contributing_models=[],
                confidence=0.0,
                strategy_used=strategy,
            )

        if len(valid_outputs) == 1:
            output = valid_outputs[0]
            return CombinedOutput(
                final_response=output.response,
                contributing_models=[output.model_id],
                confidence=output.quality_score or 0.7,
                strategy_used="single",
                total_cost=output.cost,
            )

        total_cost = sum(o.cost for o in valid_outputs)

        if strategy == "best":
            result = await self.select_best(valid_outputs)
            result.total_cost = total_cost
            return result

        if strategy == "merge":
            if not query:
                logger.warning("merge_strategy_without_query")
                return await self.select_best(valid_outputs)

            result = await self.merge_responses(valid_outputs, query)
            result.total_cost = total_cost + result.synthesis_cost
            return result

        if strategy == "vote":
            if not query:
                logger.warning("vote_strategy_without_query")
                return await self.select_best(valid_outputs)

            result = await self._vote_consensus(valid_outputs, query)
            result.total_cost = total_cost + result.synthesis_cost
            return result

        logger.warning("unknown_strategy", strategy=strategy)
        return await self.select_best(valid_outputs)

    async def select_best(self, outputs: list[ModelRun]) -> CombinedOutput:
        """Select the highest quality response.

        Uses quality_score if available, otherwise uses heuristics.
        """
        scored_outputs = []

        for output in outputs:
            if output.quality_score is not None:
                score = output.quality_score
            else:
                score = self._heuristic_quality(output.response)

            scored_outputs.append((output, score))

        scored_outputs.sort(key=lambda x: x[1], reverse=True)
        best_output, best_score = scored_outputs[0]

        logger.info(
            "best_output_selected",
            model=best_output.model_id,
            score=f"{best_score:.3f}",
            candidates=len(outputs),
        )

        return CombinedOutput(
            final_response=best_output.response,
            contributing_models=[best_output.model_id],
            confidence=best_score,
            strategy_used="best",
            total_cost=best_output.cost,
        )

    async def merge_responses(
        self,
        outputs: list[ModelRun],
        query: str,
    ) -> CombinedOutput:
        """Merge unique information from multiple responses.

        Uses an LLM to synthesize a comprehensive response that combines
        the best elements from each model's output.
        """
        if not self.client:
            logger.warning("merge_without_client")
            return await self.select_best(outputs)

        responses_text = "\n\n".join(
            f"=== Response from {o.model_id} ===\n{o.response}" for o in outputs
        )

        merge_prompt = f"""You are synthesizing multiple AI responses into a single comprehensive answer.

Original Question: {query}

Multiple responses to combine:
{responses_text}

Instructions:
1. Identify unique, valuable information from each response
2. Combine into a single coherent answer
3. Resolve any contradictions by preferring more detailed/specific information
4. Maintain proper citations if present in any response
5. Do not add information not present in any response
6. Structure the response clearly

Synthesized Response:"""

        try:
            response = await self.client.messages.create(
                model=self.synthesis_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": merge_prompt}],
            )

            merged_response = response.content[0].text
            synthesis_cost = self._estimate_synthesis_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

            contributing_models = [o.model_id for o in outputs]

            avg_quality = sum(
                o.quality_score or self._heuristic_quality(o.response) for o in outputs
            ) / len(outputs)
            confidence = min(avg_quality + 0.1, 1.0)

            logger.info(
                "responses_merged",
                model_count=len(outputs),
                synthesis_model=self.synthesis_model,
                synthesis_cost=f"${synthesis_cost:.6f}",
            )

            return CombinedOutput(
                final_response=merged_response,
                contributing_models=contributing_models,
                confidence=confidence,
                strategy_used="merge",
                synthesis_cost=synthesis_cost,
            )

        except anthropic.APIError as e:
            logger.error("merge_failed", error=str(e))
            return await self.select_best(outputs)

    async def _vote_consensus(
        self,
        outputs: list[ModelRun],
        query: str,
    ) -> CombinedOutput:
        """Build consensus response by voting on claims.

        Extracts claims from each response and includes only those
        that appear in multiple responses.
        """
        if not self.client:
            logger.warning("vote_without_client")
            return await self.select_best(outputs)

        responses_text = "\n\n".join(
            f"=== Response {i + 1} ({o.model_id}) ===\n{o.response}" for i, o in enumerate(outputs)
        )

        vote_prompt = f"""You are building a consensus answer from multiple AI responses.

Original Question: {query}

Multiple responses:
{responses_text}

Instructions:
1. Identify factual claims that appear in MULTIPLE responses (consensus claims)
2. Include only claims that at least 2 responses agree on
3. If responses contradict each other on a point, note the disagreement
4. Structure the consensus answer clearly
5. Indicate confidence level for each claim based on agreement

Consensus Response:"""

        try:
            response = await self.client.messages.create(
                model=self.synthesis_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": vote_prompt}],
            )

            consensus_response = response.content[0].text
            synthesis_cost = self._estimate_synthesis_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

            contributing_models = [o.model_id for o in outputs]

            logger.info(
                "consensus_built",
                model_count=len(outputs),
                synthesis_model=self.synthesis_model,
            )

            return CombinedOutput(
                final_response=consensus_response,
                contributing_models=contributing_models,
                confidence=0.85,
                strategy_used="vote",
                synthesis_cost=synthesis_cost,
            )

        except anthropic.APIError as e:
            logger.error("vote_failed", error=str(e))
            return await self.select_best(outputs)

    def _heuristic_quality(self, response: str) -> float:
        """Estimate response quality using heuristics."""
        if not response or len(response.strip()) < 50:
            return 0.2

        score = 0.5

        words = len(response.split())
        if 100 <= words <= 1500:
            score += 0.15
        elif words > 50:
            score += 0.1

        has_structure = any(
            marker in response
            for marker in ["\n\n", "\n1.", "\n- ", "First,", "Second,", "Additionally"]
        )
        if has_structure:
            score += 0.1

        technical_terms = [
            "EIP",
            "gas",
            "block",
            "transaction",
            "validator",
            "beacon",
            "blob",
            "calldata",
            "consensus",
            "execution",
        ]
        term_count = sum(1 for term in technical_terms if term.lower() in response.lower())
        score += min(term_count * 0.02, 0.15)

        citation_pattern = r"\[\d+\]|\[Source:"
        if re.search(citation_pattern, response):
            score += 0.1

        return min(score, 1.0)

    def _estimate_synthesis_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate synthesis cost based on model."""
        input_cost_per_1k = 0.003
        output_cost_per_1k = 0.015

        return (input_tokens / 1000 * input_cost_per_1k) + (
            output_tokens / 1000 * output_cost_per_1k
        )
