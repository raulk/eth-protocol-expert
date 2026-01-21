"""Confidence Calibrator - Calibrate confidence scores using isotonic regression (Phase 9)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class CalibrationPoint:
    """A single point for calibration: (predicted_confidence, actual_accuracy)."""

    predicted: float
    actual: float
    query_type: str | None = None
    timestamp: str | None = None


@dataclass
class CalibrationResult:
    """Result of confidence calibration."""

    original_confidence: float
    calibrated_confidence: float
    adjustment: float
    calibration_method: str


class IsotonicCalibrator:
    """Calibrate confidence scores using isotonic regression.

    Isotonic regression ensures the calibrated scores are monotonically
    increasing while minimizing squared error. This corrects systematic
    over/under-confidence without changing ranking.
    """

    MIN_POINTS_FOR_CALIBRATION: ClassVar[int] = 20

    def __init__(
        self,
        calibration_data: list[CalibrationPoint] | None = None,
    ):
        self._points: list[CalibrationPoint] = calibration_data or []
        self._sorted_predictions: list[float] = []
        self._calibrated_values: list[float] = []
        self._is_fitted = False

    def add_point(
        self,
        predicted: float,
        actual: float,
        query_type: str | None = None,
    ) -> None:
        """Add a calibration point from evaluation.

        Args:
            predicted: Model's predicted confidence (0-1)
            actual: Actual accuracy (0 or 1 for binary, 0-1 for graded)
            query_type: Optional query type for stratified calibration
        """
        from datetime import UTC, datetime

        self._points.append(
            CalibrationPoint(
                predicted=predicted,
                actual=actual,
                query_type=query_type,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )
        self._is_fitted = False

    def fit(self) -> bool:
        """Fit the isotonic regression model on collected points.

        Returns:
            True if fitting succeeded, False if insufficient data
        """
        if len(self._points) < self.MIN_POINTS_FOR_CALIBRATION:
            logger.warning(
                "insufficient_calibration_data",
                points=len(self._points),
                required=self.MIN_POINTS_FOR_CALIBRATION,
            )
            return False

        predictions = np.array([p.predicted for p in self._points])
        actuals = np.array([p.actual for p in self._points])

        sort_idx = np.argsort(predictions)
        sorted_preds = predictions[sort_idx]
        sorted_actuals = actuals[sort_idx]

        calibrated = self._isotonic_regression(sorted_preds, sorted_actuals)

        self._sorted_predictions = sorted_preds.tolist()
        self._calibrated_values = calibrated.tolist()
        self._is_fitted = True

        logger.info(
            "calibrator_fitted",
            points=len(self._points),
            unique_predictions=len(set(sorted_preds)),
        )

        return True

    def calibrate(self, confidence: float) -> CalibrationResult:
        """Calibrate a confidence score.

        Args:
            confidence: Raw confidence score (0-1)

        Returns:
            CalibrationResult with original and calibrated scores
        """
        if not self._is_fitted or not self._sorted_predictions:
            return CalibrationResult(
                original_confidence=confidence,
                calibrated_confidence=confidence,
                adjustment=0.0,
                calibration_method="uncalibrated",
            )

        calibrated = self._interpolate(confidence)

        return CalibrationResult(
            original_confidence=confidence,
            calibrated_confidence=calibrated,
            adjustment=calibrated - confidence,
            calibration_method="isotonic",
        )

    def get_calibration_curve(
        self,
        n_bins: int = 10,
    ) -> dict[str, list[float]]:
        """Get calibration curve data for visualization.

        Args:
            n_bins: Number of bins for the curve

        Returns:
            Dict with 'predicted', 'actual', and 'count' lists
        """
        if not self._points:
            return {"predicted": [], "actual": [], "count": []}

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_predictions = []
        bin_actuals = []
        bin_counts = []

        predictions = np.array([p.predicted for p in self._points])
        actuals = np.array([p.actual for p in self._points])

        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])

            if np.sum(mask) > 0:
                bin_predictions.append(float(np.mean(predictions[mask])))
                bin_actuals.append(float(np.mean(actuals[mask])))
                bin_counts.append(int(np.sum(mask)))

        return {
            "predicted": bin_predictions,
            "actual": bin_actuals,
            "count": bin_counts,
        }

    def expected_calibration_error(self) -> float:
        """Calculate Expected Calibration Error (ECE).

        Lower is better. Perfect calibration = 0.
        """
        curve = self.get_calibration_curve(n_bins=10)
        if not curve["predicted"]:
            return 1.0

        predictions = np.array(curve["predicted"])
        actuals = np.array(curve["actual"])
        counts = np.array(curve["count"])

        total = np.sum(counts)
        if total == 0:
            return 1.0

        weights = counts / total
        ece = np.sum(weights * np.abs(predictions - actuals))

        return float(ece)

    def save(self, path: Path | str) -> None:
        """Save calibration state to file."""
        path = Path(path)
        data = {
            "points": [
                {
                    "predicted": p.predicted,
                    "actual": p.actual,
                    "query_type": p.query_type,
                    "timestamp": p.timestamp,
                }
                for p in self._points
            ],
            "sorted_predictions": self._sorted_predictions,
            "calibrated_values": self._calibrated_values,
            "is_fitted": self._is_fitted,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        logger.info("calibrator_saved", path=str(path), points=len(self._points))

    def load(self, path: Path | str) -> bool:
        """Load calibration state from file.

        Returns:
            True if loaded successfully
        """
        path = Path(path)
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text())
            self._points = [
                CalibrationPoint(
                    predicted=p["predicted"],
                    actual=p["actual"],
                    query_type=p.get("query_type"),
                    timestamp=p.get("timestamp"),
                )
                for p in data["points"]
            ]
            self._sorted_predictions = data.get("sorted_predictions", [])
            self._calibrated_values = data.get("calibrated_values", [])
            self._is_fitted = data.get("is_fitted", False)

            logger.info("calibrator_loaded", path=str(path), points=len(self._points))
            return True
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("calibrator_load_failed", path=str(path), error=str(e))
            return False

    def _isotonic_regression(
        self,
        sorted_x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Pool Adjacent Violators Algorithm (PAVA) for isotonic regression."""
        n = len(y)
        result = y.copy().astype(float)
        weight = np.ones(n)

        i = 0
        while i < n - 1:
            if result[i] > result[i + 1]:
                pooled_value = (result[i] * weight[i] + result[i + 1] * weight[i + 1]) / (
                    weight[i] + weight[i + 1]
                )
                result[i] = pooled_value
                result[i + 1] = pooled_value
                weight[i] = weight[i] + weight[i + 1]
                weight[i + 1] = weight[i]

                j = i
                while j > 0 and result[j - 1] > result[j]:
                    pooled = (result[j - 1] * weight[j - 1] + result[j] * weight[j]) / (
                        weight[j - 1] + weight[j]
                    )
                    result[j - 1] = pooled
                    result[j] = pooled
                    weight[j - 1] = weight[j - 1] + weight[j]
                    weight[j] = weight[j - 1]
                    j -= 1
            i += 1

        return result

    def _interpolate(self, x: float) -> float:
        """Interpolate calibrated value for a given prediction."""
        if not self._sorted_predictions:
            return x

        preds = self._sorted_predictions
        vals = self._calibrated_values

        if x <= preds[0]:
            return vals[0]
        if x >= preds[-1]:
            return vals[-1]

        for i in range(len(preds) - 1):
            if preds[i] <= x <= preds[i + 1]:
                if preds[i + 1] == preds[i]:
                    return vals[i]
                t = (x - preds[i]) / (preds[i + 1] - preds[i])
                return vals[i] + t * (vals[i + 1] - vals[i])

        return x


class ConfidenceCalibrationManager:
    """Manage confidence calibration across query types.

    Maintains separate calibrators for different query types and provides
    unified calibration interface.
    """

    def __init__(
        self,
        calibration_dir: Path | str = "data/calibration",
    ):
        self.calibration_dir = Path(calibration_dir)
        self._global_calibrator = IsotonicCalibrator()
        self._type_calibrators: dict[str, IsotonicCalibrator] = {}

    def record_outcome(
        self,
        predicted_confidence: float,
        actual_accuracy: float,
        query_type: str | None = None,
    ) -> None:
        """Record an outcome for calibration.

        Args:
            predicted_confidence: Model's confidence score (0-1)
            actual_accuracy: Whether response was correct (0 or 1)
            query_type: Optional query type for stratified calibration
        """
        self._global_calibrator.add_point(predicted_confidence, actual_accuracy, query_type)

        if query_type:
            if query_type not in self._type_calibrators:
                self._type_calibrators[query_type] = IsotonicCalibrator()
            self._type_calibrators[query_type].add_point(
                predicted_confidence, actual_accuracy, query_type
            )

    def fit_all(self) -> dict[str, bool]:
        """Fit all calibrators.

        Returns:
            Dict mapping calibrator name to fit success status
        """
        results = {"global": self._global_calibrator.fit()}

        for query_type, calibrator in self._type_calibrators.items():
            results[query_type] = calibrator.fit()

        return results

    def calibrate(
        self,
        confidence: float,
        query_type: str | None = None,
    ) -> CalibrationResult:
        """Calibrate a confidence score.

        Uses type-specific calibrator if available and fitted,
        otherwise falls back to global calibrator.
        """
        if query_type and query_type in self._type_calibrators:
            calibrator = self._type_calibrators[query_type]
            if calibrator._is_fitted:
                return calibrator.calibrate(confidence)

        return self._global_calibrator.calibrate(confidence)

    def get_ece_scores(self) -> dict[str, float]:
        """Get Expected Calibration Error for all calibrators."""
        scores = {"global": self._global_calibrator.expected_calibration_error()}

        for query_type, calibrator in self._type_calibrators.items():
            scores[query_type] = calibrator.expected_calibration_error()

        return scores

    def save_all(self) -> None:
        """Save all calibrators to disk."""
        self._global_calibrator.save(self.calibration_dir / "global_calibration.json")

        for query_type, calibrator in self._type_calibrators.items():
            safe_name = query_type.replace("/", "_").replace(" ", "_")
            calibrator.save(self.calibration_dir / f"{safe_name}_calibration.json")

    def load_all(self) -> dict[str, bool]:
        """Load all calibrators from disk.

        Returns:
            Dict mapping calibrator name to load success status
        """
        results = {
            "global": self._global_calibrator.load(self.calibration_dir / "global_calibration.json")
        }

        if self.calibration_dir.exists():
            for path in self.calibration_dir.glob("*_calibration.json"):
                if path.name == "global_calibration.json":
                    continue
                query_type = path.stem.replace("_calibration", "")
                calibrator = IsotonicCalibrator()
                if calibrator.load(path):
                    self._type_calibrators[query_type] = calibrator
                    results[query_type] = True

        return results
