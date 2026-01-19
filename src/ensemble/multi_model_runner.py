"""Multi-Model Runner - Run multiple models in parallel or cascade (Phase 12)."""

import asyncio
import os
import time
from dataclasses import dataclass

import anthropic
import structlog

from src.ensemble.cost_router import CostRouter, ModelTier

logger = structlog.get_logger()


@dataclass
class ModelRun:
    """Result from a single model run."""

    model_id: str
    response: str
    tokens_used: int
    latency_ms: float
    cost: float
    input_tokens: int = 0
    output_tokens: int = 0
    quality_score: float | None = None
    error: str | None = None


class MultiModelRunner:
    """Run multiple models in parallel or cascade for ensemble responses.

    Supports three execution modes:
    - Single: Run one model
    - Parallel: Run multiple models simultaneously, combine results
    - Cascade: Start with fast model, escalate if quality insufficient
    """

    def __init__(
        self,
        models: list[str] | None = None,
        api_key: str | None = None,
        cost_router: CostRouter | None = None,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.cost_router = cost_router or CostRouter(api_key=self.api_key)

        self.models = models or [
            self.cost_router.get_model_for_tier(ModelTier.FAST),
            self.cost_router.get_model_for_tier(ModelTier.BALANCED),
            self.cost_router.get_model_for_tier(ModelTier.POWERFUL),
        ]

        self.max_tokens = max_tokens

    async def run_single(
        self,
        model: str,
        prompt: str,
        context: str,
        system_prompt: str | None = None,
    ) -> ModelRun:
        """Run a single model and return results.

        Args:
            model: Model ID to use
            prompt: User prompt/question
            context: Retrieved context to include
            system_prompt: Optional system prompt

        Returns:
            ModelRun with response and metrics
        """
        start_time = time.perf_counter()

        full_prompt = self._build_prompt(prompt, context)

        try:
            messages = [{"role": "user", "content": full_prompt}]

            kwargs = {
                "model": model,
                "max_tokens": self.max_tokens,
                "messages": messages,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            response = await self.client.messages.create(**kwargs)

            latency_ms = (time.perf_counter() - start_time) * 1000
            response_text = response.content[0].text

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            cost = self.cost_router.estimate_cost(model, input_tokens, output_tokens)

            logger.info(
                "model_run_complete",
                model=model,
                latency_ms=f"{latency_ms:.1f}",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=f"${cost:.6f}",
            )

            return ModelRun(
                model_id=model,
                response=response_text,
                tokens_used=total_tokens,
                latency_ms=latency_ms,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except anthropic.APIError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error("model_run_failed", model=model, error=str(e))

            return ModelRun(
                model_id=model,
                response="",
                tokens_used=0,
                latency_ms=latency_ms,
                cost=0.0,
                error=str(e),
            )

    async def run_parallel(
        self,
        prompt: str,
        context: str,
        models: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> list[ModelRun]:
        """Run multiple models in parallel.

        Args:
            prompt: User prompt/question
            context: Retrieved context to include
            models: Optional list of models (defaults to self.models)
            system_prompt: Optional system prompt

        Returns:
            List of ModelRun results from all models
        """
        models_to_run = models or self.models

        logger.info(
            "parallel_run_started",
            models=models_to_run,
            prompt_length=len(prompt),
            context_length=len(context),
        )

        tasks = [self.run_single(model, prompt, context, system_prompt) for model in models_to_run]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_runs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "parallel_run_exception",
                    model=models_to_run[i],
                    error=str(result),
                )
                successful_runs.append(
                    ModelRun(
                        model_id=models_to_run[i],
                        response="",
                        tokens_used=0,
                        latency_ms=0,
                        cost=0.0,
                        error=str(result),
                    )
                )
            else:
                successful_runs.append(result)

        total_cost = sum(r.cost for r in successful_runs)
        total_tokens = sum(r.tokens_used for r in successful_runs)

        logger.info(
            "parallel_run_complete",
            model_count=len(successful_runs),
            successful=sum(1 for r in successful_runs if not r.error),
            total_cost=f"${total_cost:.6f}",
            total_tokens=total_tokens,
        )

        return successful_runs

    async def run_cascade(
        self,
        prompt: str,
        context: str,
        quality_threshold: float = 0.7,
        system_prompt: str | None = None,
    ) -> ModelRun:
        """Run models in cascade, escalating if quality is insufficient.

        Starts with the fastest/cheapest model and escalates to more powerful
        models only if the response quality doesn't meet the threshold.

        Args:
            prompt: User prompt/question
            context: Retrieved context to include
            quality_threshold: Minimum quality score to accept (0-1)
            system_prompt: Optional system prompt

        Returns:
            ModelRun from the first model that meets quality threshold
        """
        cascade_order = [
            self.cost_router.get_model_for_tier(ModelTier.FAST),
            self.cost_router.get_model_for_tier(ModelTier.BALANCED),
            self.cost_router.get_model_for_tier(ModelTier.POWERFUL),
        ]

        logger.info(
            "cascade_run_started",
            quality_threshold=quality_threshold,
            cascade_order=cascade_order,
        )

        cumulative_cost = 0.0
        cumulative_latency = 0.0

        for i, model in enumerate(cascade_order):
            run = await self.run_single(model, prompt, context, system_prompt)

            if run.error:
                logger.warning(
                    "cascade_model_failed",
                    model=model,
                    error=run.error,
                    escalating=i < len(cascade_order) - 1,
                )
                continue

            cumulative_cost += run.cost
            cumulative_latency += run.latency_ms

            quality_score = self._assess_quality(run.response, prompt, context)
            run.quality_score = quality_score

            logger.info(
                "cascade_quality_assessed",
                model=model,
                quality_score=f"{quality_score:.3f}",
                threshold=quality_threshold,
                meets_threshold=quality_score >= quality_threshold,
            )

            if quality_score >= quality_threshold:
                run.cost = cumulative_cost
                run.latency_ms = cumulative_latency
                return run

            if i < len(cascade_order) - 1:
                logger.info(
                    "cascade_escalating",
                    from_model=model,
                    to_model=cascade_order[i + 1],
                    reason=f"quality {quality_score:.3f} < threshold {quality_threshold}",
                )

        final_run = run
        final_run.cost = cumulative_cost
        final_run.latency_ms = cumulative_latency

        logger.info(
            "cascade_complete",
            final_model=final_run.model_id,
            total_cost=f"${cumulative_cost:.6f}",
            total_latency_ms=f"{cumulative_latency:.1f}",
        )

        return final_run

    def _build_prompt(self, prompt: str, context: str) -> str:
        """Build full prompt with context."""
        if not context:
            return prompt

        return f"""Based on the following context, answer the question.

<context>
{context}
</context>

Question: {prompt}

Provide a clear, accurate answer based on the context. Cite specific information from the context when making claims."""

    def _assess_quality(
        self,
        response: str,
        prompt: str,
        context: str,
    ) -> float:
        """Assess response quality (heuristic-based).

        Quality factors:
        - Response length (not too short)
        - Contains relevant terms from query
        - References context appropriately
        - Well-structured
        """
        if not response or len(response.strip()) < 50:
            return 0.2

        score = 0.5

        words = response.split()
        if 100 <= len(words) <= 1500:
            score += 0.1
        elif len(words) > 50:
            score += 0.05

        query_terms = set(prompt.lower().split())
        query_terms -= {"what", "how", "why", "when", "which", "is", "are", "the", "a"}
        response_lower = response.lower()

        matched = sum(1 for term in query_terms if term in response_lower)
        if query_terms:
            term_coverage = matched / len(query_terms)
            score += term_coverage * 0.2

        has_structure = any(
            marker in response
            for marker in ["\n\n", "1.", "- ", "* ", "First", "Second", "However", "Additionally"]
        )
        if has_structure:
            score += 0.1

        hedging_phrases = ["I don't know", "I cannot", "I'm not sure", "unclear", "no information"]
        if not any(phrase.lower() in response_lower for phrase in hedging_phrases):
            score += 0.1

        return min(score, 1.0)
