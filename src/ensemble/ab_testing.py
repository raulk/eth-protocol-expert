"""A/B Testing Framework - Compare response strategies (Phase 9)."""

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import structlog

logger = structlog.get_logger()


class AssignmentStrategy(Enum):
    """Strategy for assigning users/queries to variants."""

    RANDOM = "random"
    DETERMINISTIC = "deterministic"
    WEIGHTED = "weighted"


@dataclass
class Variant:
    """A variant in an A/B test."""

    name: str
    weight: float = 0.5
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "config": self.config,
        }


@dataclass
class Assignment:
    """Assignment of a query to a variant."""

    variant: Variant
    experiment_id: str
    assignment_key: str
    timestamp: str


@dataclass
class ExperimentResult:
    """Result of a single experiment trial."""

    experiment_id: str
    variant_name: str
    query: str
    response_quality: float | None = None
    latency_ms: float | None = None
    cost: float | None = None
    user_feedback: int | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "variant_name": self.variant_name,
            "query": self.query,
            "response_quality": self.response_quality,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "user_feedback": self.user_feedback,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class Experiment:
    """An A/B test experiment."""

    id: str
    name: str
    variants: list[Variant]
    strategy: AssignmentStrategy = AssignmentStrategy.DETERMINISTIC
    status: str = "active"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "variants": [v.to_dict() for v in self.variants],
            "strategy": self.strategy.value,
            "status": self.status,
            "created_at": self.created_at,
            "description": self.description,
        }


class ABTestingFramework:
    """Framework for running A/B tests on response strategies.

    Supports testing different:
    - Model tiers (fast vs balanced vs powerful)
    - Retrieval strategies (simple vs hybrid vs graph)
    - Generation modes (cited vs validated vs agentic)
    - Prompt variations
    """

    DEFAULT_EXPERIMENT_DIR: ClassVar[Path] = Path("data/experiments")

    def __init__(
        self,
        experiment_dir: Path | str | None = None,
        default_strategy: AssignmentStrategy = AssignmentStrategy.DETERMINISTIC,
    ):
        self.experiment_dir = (
            Path(experiment_dir) if experiment_dir else self.DEFAULT_EXPERIMENT_DIR
        )
        self.default_strategy = default_strategy
        self._experiments: dict[str, Experiment] = {}
        self._results: list[ExperimentResult] = []

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        name: str,
        variants: list[Variant],
        strategy: AssignmentStrategy | None = None,
        description: str = "",
    ) -> Experiment:
        """Create a new A/B test experiment.

        Args:
            name: Human-readable experiment name
            variants: List of variants to test
            strategy: Assignment strategy (default: deterministic)
            description: Optional description

        Returns:
            Created Experiment
        """
        experiment_id = (
            f"{name.lower().replace(' ', '_')}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        )

        total_weight = sum(v.weight for v in variants)
        for v in variants:
            v.weight = v.weight / total_weight

        experiment = Experiment(
            id=experiment_id,
            name=name,
            variants=variants,
            strategy=strategy or self.default_strategy,
            description=description,
        )

        self._experiments[experiment_id] = experiment

        self._save_experiment(experiment)

        logger.info(
            "experiment_created",
            experiment_id=experiment_id,
            name=name,
            variants=[v.name for v in variants],
            strategy=experiment.strategy.value,
        )

        return experiment

    def assign(
        self,
        experiment_id: str,
        assignment_key: str,
    ) -> Assignment | None:
        """Assign a key (user/session/query) to a variant.

        Args:
            experiment_id: ID of the experiment
            assignment_key: Key to use for assignment (e.g., query hash, session ID)

        Returns:
            Assignment or None if experiment not found/inactive
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment or experiment.status != "active":
            return None

        variant = self._select_variant(experiment, assignment_key)

        return Assignment(
            variant=variant,
            experiment_id=experiment_id,
            assignment_key=assignment_key,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def assign_query(
        self,
        experiment_id: str,
        query: str,
    ) -> Assignment | None:
        """Assign a query to a variant using query hash.

        Args:
            experiment_id: ID of the experiment
            query: The query string

        Returns:
            Assignment or None
        """
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        return self.assign(experiment_id, query_hash)

    def record_result(
        self,
        experiment_id: str,
        variant_name: str,
        query: str,
        response_quality: float | None = None,
        latency_ms: float | None = None,
        cost: float | None = None,
        user_feedback: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentResult:
        """Record the result of an experiment trial.

        Args:
            experiment_id: ID of the experiment
            variant_name: Name of the variant used
            query: The query that was processed
            response_quality: Optional quality score (0-1)
            latency_ms: Optional latency in milliseconds
            cost: Optional cost in USD
            user_feedback: Optional user feedback (-1, 0, 1)
            metadata: Optional additional metadata

        Returns:
            Recorded ExperimentResult
        """
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_name=variant_name,
            query=query,
            response_quality=response_quality,
            latency_ms=latency_ms,
            cost=cost,
            user_feedback=user_feedback,
            metadata=metadata or {},
        )

        self._results.append(result)

        self._save_result(result)

        logger.debug(
            "experiment_result_recorded",
            experiment_id=experiment_id,
            variant=variant_name,
            quality=response_quality,
        )

        return result

    def get_results(
        self,
        experiment_id: str,
    ) -> list[ExperimentResult]:
        """Get all results for an experiment."""
        return [r for r in self._results if r.experiment_id == experiment_id]

    def analyze(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        """Analyze experiment results.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Analysis with per-variant statistics
        """
        results = self.get_results(experiment_id)
        if not results:
            return {"error": "No results found", "experiment_id": experiment_id}

        by_variant: dict[str, list[ExperimentResult]] = {}
        for r in results:
            by_variant.setdefault(r.variant_name, []).append(r)

        analysis = {
            "experiment_id": experiment_id,
            "total_trials": len(results),
            "variants": {},
        }

        for variant_name, variant_results in by_variant.items():
            qualities = [
                r.response_quality for r in variant_results if r.response_quality is not None
            ]
            latencies = [r.latency_ms for r in variant_results if r.latency_ms is not None]
            costs = [r.cost for r in variant_results if r.cost is not None]
            feedbacks = [r.user_feedback for r in variant_results if r.user_feedback is not None]

            analysis["variants"][variant_name] = {
                "trials": len(variant_results),
                "quality": {
                    "mean": sum(qualities) / len(qualities) if qualities else None,
                    "min": min(qualities) if qualities else None,
                    "max": max(qualities) if qualities else None,
                    "count": len(qualities),
                },
                "latency": {
                    "mean": sum(latencies) / len(latencies) if latencies else None,
                    "min": min(latencies) if latencies else None,
                    "max": max(latencies) if latencies else None,
                    "count": len(latencies),
                },
                "cost": {
                    "total": sum(costs) if costs else None,
                    "mean": sum(costs) / len(costs) if costs else None,
                    "count": len(costs),
                },
                "feedback": {
                    "positive": sum(1 for f in feedbacks if f > 0),
                    "neutral": sum(1 for f in feedbacks if f == 0),
                    "negative": sum(1 for f in feedbacks if f < 0),
                    "count": len(feedbacks),
                },
            }

        return analysis

    def determine_winner(
        self,
        experiment_id: str,
        metric: str = "quality",
        min_trials: int = 30,
    ) -> dict[str, Any]:
        """Determine the winning variant.

        Args:
            experiment_id: ID of the experiment
            metric: Metric to optimize ('quality', 'latency', 'cost')
            min_trials: Minimum trials per variant to declare winner

        Returns:
            Winner determination with confidence
        """
        analysis = self.analyze(experiment_id)
        if "error" in analysis:
            return analysis

        variants = analysis["variants"]

        eligible = {name: data for name, data in variants.items() if data["trials"] >= min_trials}

        if not eligible:
            return {
                "winner": None,
                "reason": f"No variant has {min_trials}+ trials",
                "analysis": analysis,
            }

        if metric == "quality":
            scores = {
                name: data["quality"]["mean"]
                for name, data in eligible.items()
                if data["quality"]["mean"] is not None
            }
            winner = max(scores.keys(), key=lambda k: scores[k]) if scores else None
            maximize = True
        elif metric == "latency":
            scores = {
                name: data["latency"]["mean"]
                for name, data in eligible.items()
                if data["latency"]["mean"] is not None
            }
            winner = min(scores.keys(), key=lambda k: scores[k]) if scores else None
            maximize = False
        elif metric == "cost":
            scores = {
                name: data["cost"]["mean"]
                for name, data in eligible.items()
                if data["cost"]["mean"] is not None
            }
            winner = min(scores.keys(), key=lambda k: scores[k]) if scores else None
            maximize = False
        else:
            return {"error": f"Unknown metric: {metric}"}

        if not winner:
            return {
                "winner": None,
                "reason": f"No data for metric '{metric}'",
                "analysis": analysis,
            }

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=maximize)
        if len(sorted_scores) >= 2:
            diff = abs(sorted_scores[0][1] - sorted_scores[1][1])
            relative_diff = diff / abs(sorted_scores[0][1]) if sorted_scores[0][1] else 0
            confidence = min(relative_diff * 100, 99)
        else:
            confidence = 50

        return {
            "winner": winner,
            "metric": metric,
            "score": scores[winner],
            "confidence": confidence,
            "all_scores": scores,
            "analysis": analysis,
        }

    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an active experiment."""
        if experiment_id in self._experiments:
            self._experiments[experiment_id].status = "stopped"
            self._save_experiment(self._experiments[experiment_id])
            logger.info("experiment_stopped", experiment_id=experiment_id)
            return True
        return False

    def load_experiments(self) -> int:
        """Load experiments from disk.

        Returns:
            Number of experiments loaded
        """
        count = 0
        experiments_file = self.experiment_dir / "experiments.json"

        if experiments_file.exists():
            try:
                data = json.loads(experiments_file.read_text())
                for exp_data in data.get("experiments", []):
                    exp = Experiment(
                        id=exp_data["id"],
                        name=exp_data["name"],
                        variants=[
                            Variant(
                                name=v["name"],
                                weight=v["weight"],
                                config=v.get("config", {}),
                            )
                            for v in exp_data["variants"]
                        ],
                        strategy=AssignmentStrategy(exp_data.get("strategy", "deterministic")),
                        status=exp_data.get("status", "active"),
                        created_at=exp_data.get("created_at", ""),
                        description=exp_data.get("description", ""),
                    )
                    self._experiments[exp.id] = exp
                    count += 1
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("failed_to_load_experiments", error=str(e))

        results_file = self.experiment_dir / "results.jsonl"
        if results_file.exists():
            try:
                with results_file.open() as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self._results.append(
                                ExperimentResult(
                                    experiment_id=data["experiment_id"],
                                    variant_name=data["variant_name"],
                                    query=data["query"],
                                    response_quality=data.get("response_quality"),
                                    latency_ms=data.get("latency_ms"),
                                    cost=data.get("cost"),
                                    user_feedback=data.get("user_feedback"),
                                    timestamp=data.get("timestamp", ""),
                                    metadata=data.get("metadata", {}),
                                )
                            )
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("failed_to_load_results", error=str(e))

        logger.info("experiments_loaded", count=count, results=len(self._results))
        return count

    def _select_variant(
        self,
        experiment: Experiment,
        assignment_key: str,
    ) -> Variant:
        """Select variant based on experiment strategy."""
        if experiment.strategy == AssignmentStrategy.RANDOM:
            return random.choices(
                experiment.variants,
                weights=[v.weight for v in experiment.variants],
            )[0]

        if experiment.strategy == AssignmentStrategy.DETERMINISTIC:
            key_hash = int(hashlib.sha256(assignment_key.encode()).hexdigest()[:8], 16)
            bucket = (key_hash % 1000) / 1000.0

            cumulative = 0.0
            for variant in experiment.variants:
                cumulative += variant.weight
                if bucket < cumulative:
                    return variant
            return experiment.variants[-1]

        if experiment.strategy == AssignmentStrategy.WEIGHTED:
            return random.choices(
                experiment.variants,
                weights=[v.weight for v in experiment.variants],
            )[0]

        return experiment.variants[0]

    def _save_experiment(self, experiment: Experiment) -> None:
        """Save experiment to disk."""
        experiments_file = self.experiment_dir / "experiments.json"

        existing = {"experiments": []}
        if experiments_file.exists():
            try:
                existing = json.loads(experiments_file.read_text())
            except json.JSONDecodeError:
                pass

        existing["experiments"] = [
            e for e in existing.get("experiments", []) if e.get("id") != experiment.id
        ]
        existing["experiments"].append(experiment.to_dict())

        experiments_file.write_text(json.dumps(existing, indent=2))

    def _save_result(self, result: ExperimentResult) -> None:
        """Save result to disk (append to JSONL)."""
        results_file = self.experiment_dir / "results.jsonl"
        with results_file.open("a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
