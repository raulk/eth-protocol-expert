"""Ensemble module for multi-model orchestration (Phases 9 & 12).

Phase 9 additions:
- ConfidenceCalibrator: Isotonic regression for confidence calibration
- CircuitBreaker: Halt generation for low-evidence cases
- ABTestingFramework: A/B testing for response strategies
- RegressionGate: Prevent accuracy regression in deployments
"""

from src.ensemble.ab_testing import (
    ABTestingFramework,
    Assignment,
    AssignmentStrategy,
    Experiment,
    ExperimentResult,
    Variant,
)
from src.ensemble.circuit_breaker import (
    BreakReason,
    CircuitBreaker,
    CircuitDecision,
    CircuitState,
    LowEvidenceHandler,
)
from src.ensemble.conditional_trigger import ConditionalTrigger, EnsembleDecision
from src.ensemble.confidence_calibrator import (
    CalibrationPoint,
    CalibrationResult,
    ConfidenceCalibrationManager,
    IsotonicCalibrator,
)
from src.ensemble.confidence_scorer import (
    ConfidenceFactors,
    RetrievalConfidence,
    RetrievalConfidenceScorer,
)
from src.ensemble.cost_router import CostRouter, ModelConfig, ModelTier, RoutingDecision
from src.ensemble.multi_model_runner import ModelRun, MultiModelRunner
from src.ensemble.regression_gate import (
    DeploymentGate,
    GateResult,
    MetricSnapshot,
    RegressionGate,
    create_snapshot_from_eval,
)
from src.ensemble.synthesis_combiner import CombinedOutput, SynthesisCombiner

__all__ = [
    # A/B Testing
    "ABTestingFramework",
    "Assignment",
    "AssignmentStrategy",
    # Circuit Breaker
    "BreakReason",
    # Confidence Calibration
    "CalibrationPoint",
    "CalibrationResult",
    "CircuitBreaker",
    "CircuitDecision",
    "CircuitState",
    # Synthesis Combiner
    "CombinedOutput",
    # Conditional Trigger
    "ConditionalTrigger",
    "ConfidenceCalibrationManager",
    # Confidence Scoring
    "ConfidenceFactors",
    # Cost Router
    "CostRouter",
    # Regression Gate
    "DeploymentGate",
    "EnsembleDecision",
    "Experiment",
    "ExperimentResult",
    "GateResult",
    "IsotonicCalibrator",
    "LowEvidenceHandler",
    "MetricSnapshot",
    "ModelConfig",
    # Multi-Model Runner
    "ModelRun",
    "ModelTier",
    "MultiModelRunner",
    "RegressionGate",
    "RetrievalConfidence",
    "RetrievalConfidenceScorer",
    "RoutingDecision",
    "SynthesisCombiner",
    "Variant",
    "create_snapshot_from_eval",
]
