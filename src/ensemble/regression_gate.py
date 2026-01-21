"""Regression Gate - Prevent accuracy regression in deployments (Phase 9)."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import structlog

logger = structlog.get_logger()


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time."""

    timestamp: str
    accuracy: float | None = None
    recall_at_k: float | None = None
    citation_accuracy: float | None = None
    latency_p50_ms: float | None = None
    latency_p99_ms: float | None = None
    cost_per_query: float | None = None
    eval_count: int = 0
    version: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "accuracy": self.accuracy,
            "recall_at_k": self.recall_at_k,
            "citation_accuracy": self.citation_accuracy,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "cost_per_query": self.cost_per_query,
            "eval_count": self.eval_count,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetricSnapshot":
        return cls(
            timestamp=data.get("timestamp", ""),
            accuracy=data.get("accuracy"),
            recall_at_k=data.get("recall_at_k"),
            citation_accuracy=data.get("citation_accuracy"),
            latency_p50_ms=data.get("latency_p50_ms"),
            latency_p99_ms=data.get("latency_p99_ms"),
            cost_per_query=data.get("cost_per_query"),
            eval_count=data.get("eval_count", 0),
            version=data.get("version", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GateResult:
    """Result of a regression gate check."""

    passed: bool
    regressions: list[str]
    improvements: list[str]
    warnings: list[str]
    baseline_version: str
    current_version: str
    details: dict = field(default_factory=dict)


class RegressionGate:
    """Gate that blocks deployments if metrics regress beyond thresholds.

    Tracks key metrics over time and prevents deploying changes that
    would significantly worsen accuracy, recall, or other quality metrics.
    """

    DEFAULT_THRESHOLDS: ClassVar[dict[str, float]] = {
        "accuracy": 0.02,  # Allow 2% regression
        "recall_at_k": 0.03,  # Allow 3% regression
        "citation_accuracy": 0.05,  # Allow 5% regression
        "latency_p50_ms": 0.20,  # Allow 20% increase
        "latency_p99_ms": 0.30,  # Allow 30% increase
        "cost_per_query": 0.15,  # Allow 15% increase
    }

    LOWER_IS_BETTER: ClassVar[set[str]] = {
        "latency_p50_ms",
        "latency_p99_ms",
        "cost_per_query",
    }

    def __init__(
        self,
        baseline_path: Path | str = "data/eval/baseline_metrics.json",
        history_path: Path | str = "data/eval/metrics_history.jsonl",
        thresholds: dict[str, float] | None = None,
    ):
        self.baseline_path = Path(baseline_path)
        self.history_path = Path(history_path)
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()

        self._baseline: MetricSnapshot | None = None
        self._history: list[MetricSnapshot] = []

        self._load_baseline()
        self._load_history()

    def check(
        self,
        current: MetricSnapshot,
        baseline: MetricSnapshot | None = None,
    ) -> GateResult:
        """Check if current metrics pass the regression gate.

        Args:
            current: Current metric snapshot to evaluate
            baseline: Optional baseline to compare against (uses stored baseline if None)

        Returns:
            GateResult with pass/fail and details
        """
        baseline = baseline or self._baseline

        if not baseline:
            logger.warning("no_baseline_for_regression_check")
            return GateResult(
                passed=True,
                regressions=[],
                improvements=[],
                warnings=["No baseline available - gate passed by default"],
                baseline_version="none",
                current_version=current.version,
            )

        regressions = []
        improvements = []
        warnings = []
        details = {}

        metrics_to_check = [
            ("accuracy", current.accuracy, baseline.accuracy),
            ("recall_at_k", current.recall_at_k, baseline.recall_at_k),
            ("citation_accuracy", current.citation_accuracy, baseline.citation_accuracy),
            ("latency_p50_ms", current.latency_p50_ms, baseline.latency_p50_ms),
            ("latency_p99_ms", current.latency_p99_ms, baseline.latency_p99_ms),
            ("cost_per_query", current.cost_per_query, baseline.cost_per_query),
        ]

        for metric_name, current_val, baseline_val in metrics_to_check:
            if current_val is None or baseline_val is None:
                continue

            threshold = self.thresholds.get(metric_name, 0.05)
            lower_is_better = metric_name in self.LOWER_IS_BETTER

            if lower_is_better:
                change = (current_val - baseline_val) / baseline_val if baseline_val else 0
                regressed = change > threshold
                improved = change < -threshold * 0.5
            else:
                change = (baseline_val - current_val) / baseline_val if baseline_val else 0
                regressed = change > threshold
                improved = change < -threshold * 0.5

            details[metric_name] = {
                "baseline": baseline_val,
                "current": current_val,
                "change_pct": change * 100,
                "threshold_pct": threshold * 100,
                "regressed": regressed,
            }

            if regressed:
                direction = "increased" if lower_is_better else "decreased"
                regressions.append(
                    f"{metric_name} {direction} by {abs(change) * 100:.1f}% "
                    f"(threshold: {threshold * 100:.1f}%)"
                )
            elif improved:
                direction = "decreased" if lower_is_better else "increased"
                improvements.append(f"{metric_name} {direction} by {abs(change) * 100:.1f}%")

        if current.eval_count < 20:
            warnings.append(
                f"Low eval count ({current.eval_count}) - results may not be statistically significant"
            )

        passed = len(regressions) == 0

        logger.info(
            "regression_gate_check",
            passed=passed,
            regressions=len(regressions),
            improvements=len(improvements),
            baseline_version=baseline.version,
            current_version=current.version,
        )

        return GateResult(
            passed=passed,
            regressions=regressions,
            improvements=improvements,
            warnings=warnings,
            baseline_version=baseline.version,
            current_version=current.version,
            details=details,
        )

    def set_baseline(
        self,
        snapshot: MetricSnapshot,
        force: bool = False,
    ) -> bool:
        """Set a new baseline for regression checks.

        Args:
            snapshot: Metrics snapshot to use as baseline
            force: If True, overwrite existing baseline

        Returns:
            True if baseline was set
        """
        if self._baseline and not force:
            result = self.check(snapshot, self._baseline)
            if not result.passed:
                logger.warning(
                    "cannot_set_regressed_baseline",
                    regressions=result.regressions,
                )
                return False

        self._baseline = snapshot
        self._save_baseline()

        logger.info(
            "baseline_set",
            version=snapshot.version,
            accuracy=snapshot.accuracy,
            recall_at_k=snapshot.recall_at_k,
        )

        return True

    def record_snapshot(self, snapshot: MetricSnapshot) -> None:
        """Record a metrics snapshot to history."""
        self._history.append(snapshot)
        self._save_snapshot(snapshot)

        logger.debug(
            "snapshot_recorded",
            version=snapshot.version,
            accuracy=snapshot.accuracy,
        )

    def get_trend(
        self,
        metric: str,
        last_n: int = 10,
    ) -> list[tuple[str, float]]:
        """Get trend of a metric over recent history.

        Args:
            metric: Name of metric to track
            last_n: Number of recent snapshots to include

        Returns:
            List of (timestamp, value) tuples
        """
        trend = []
        for snapshot in self._history[-last_n:]:
            value = getattr(snapshot, metric, None)
            if value is not None:
                trend.append((snapshot.timestamp, value))
        return trend

    def get_baseline(self) -> MetricSnapshot | None:
        """Get the current baseline."""
        return self._baseline

    def _load_baseline(self) -> None:
        """Load baseline from disk."""
        if self.baseline_path.exists():
            try:
                data = json.loads(self.baseline_path.read_text())
                self._baseline = MetricSnapshot.from_dict(data)
                logger.info("baseline_loaded", version=self._baseline.version)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("failed_to_load_baseline", error=str(e))

    def _save_baseline(self) -> None:
        """Save baseline to disk."""
        if self._baseline:
            self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self.baseline_path.write_text(json.dumps(self._baseline.to_dict(), indent=2))

    def _load_history(self) -> None:
        """Load metrics history from disk."""
        if self.history_path.exists():
            try:
                with self.history_path.open() as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self._history.append(MetricSnapshot.from_dict(data))
                logger.info("history_loaded", count=len(self._history))
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("failed_to_load_history", error=str(e))

    def _save_snapshot(self, snapshot: MetricSnapshot) -> None:
        """Append snapshot to history file."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a") as f:
            f.write(json.dumps(snapshot.to_dict()) + "\n")


class DeploymentGate:
    """High-level gate for deployment decisions.

    Combines regression checking with other deployment safeguards.
    """

    def __init__(
        self,
        regression_gate: RegressionGate | None = None,
        require_tests_pass: bool = True,
        min_eval_count: int = 50,
    ):
        self.regression_gate = regression_gate or RegressionGate()
        self.require_tests_pass = require_tests_pass
        self.min_eval_count = min_eval_count

    def can_deploy(
        self,
        metrics: MetricSnapshot,
        tests_passed: bool = True,
        manual_override: bool = False,
    ) -> tuple[bool, list[str]]:
        """Check if deployment can proceed.

        Args:
            metrics: Current metrics snapshot
            tests_passed: Whether tests passed
            manual_override: Force deployment despite gates

        Returns:
            Tuple of (can_deploy, list of blocking reasons)
        """
        if manual_override:
            logger.warning("deployment_gate_overridden")
            return True, ["Manual override enabled"]

        blockers = []

        if self.require_tests_pass and not tests_passed:
            blockers.append("Tests failed")

        if metrics.eval_count < self.min_eval_count:
            blockers.append(
                f"Insufficient eval count ({metrics.eval_count} < {self.min_eval_count})"
            )

        regression_result = self.regression_gate.check(metrics)
        if not regression_result.passed:
            blockers.extend([f"Regression: {r}" for r in regression_result.regressions])

        can_deploy = len(blockers) == 0

        logger.info(
            "deployment_gate_check",
            can_deploy=can_deploy,
            blockers=blockers,
            version=metrics.version,
        )

        return can_deploy, blockers


def create_snapshot_from_eval(
    eval_results: dict,
    version: str = "",
) -> MetricSnapshot:
    """Create a MetricSnapshot from evaluation results.

    Args:
        eval_results: Dict with evaluation metrics
        version: Optional version identifier

    Returns:
        MetricSnapshot populated from eval results
    """
    return MetricSnapshot(
        timestamp=datetime.now(UTC).isoformat(),
        accuracy=eval_results.get("accuracy"),
        recall_at_k=eval_results.get("recall_at_10") or eval_results.get("recall_at_k"),
        citation_accuracy=eval_results.get("citation_accuracy"),
        latency_p50_ms=eval_results.get("latency_p50_ms"),
        latency_p99_ms=eval_results.get("latency_p99_ms"),
        cost_per_query=eval_results.get("cost_per_query"),
        eval_count=eval_results.get("eval_count", 0),
        version=version,
        metadata={"source": "eval_run"},
    )
