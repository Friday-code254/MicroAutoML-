"""
MicroAutoML-Agent — Custom Exceptions

All custom exceptions are defined here. Use these instead of bare Python
built-ins so that callers can catch specific failure categories.
"""

from __future__ import annotations

from automl.core.enums import FailureType, LeakageSeverity


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class MicroAutoMLError(Exception):
    """Base exception for all MicroAutoML errors."""


# ---------------------------------------------------------------------------
# Data / schema
# ---------------------------------------------------------------------------


class DataLoadError(MicroAutoMLError):
    """Raised when the input dataset cannot be loaded or parsed."""


class SchemaValidationError(MicroAutoMLError):
    """Raised when an `ExperimentSpec` or other schema object fails validation."""

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class UnsupportedTaskError(MicroAutoMLError):
    """Raised when the detected task type is not supported."""


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


class LeakageError(MicroAutoMLError):
    """Raised when a critical data leakage is detected and cannot be auto-remedied."""

    def __init__(self, message: str, severity: LeakageSeverity = LeakageSeverity.HIGH) -> None:
        super().__init__(message)
        self.severity = severity


class PreSplitFitError(LeakageError):
    """Raised when a transformer is detected to be fit before the train/test split."""

    def __init__(self, transformer_name: str) -> None:
        super().__init__(
            f"Transformer '{transformer_name}' was fit before the train/test split. "
            "This causes data leakage. All transforms must be fit inside CV folds.",
            severity=LeakageSeverity.CRITICAL,
        )
        self.transformer_name = transformer_name


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class ExperimentError(MicroAutoMLError):
    """Raised when an experiment fails to execute."""

    def __init__(
        self,
        message: str,
        failure_type: FailureType = FailureType.UNKNOWN,
        experiment_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.experiment_id = experiment_id


class ExperimentTimeoutError(ExperimentError):
    """Raised when an experiment exceeds its wall-clock budget."""

    def __init__(self, experiment_id: str, timeout_seconds: float) -> None:
        super().__init__(
            f"Experiment '{experiment_id}' exceeded timeout of {timeout_seconds:.1f}s.",
            failure_type=FailureType.TIMEOUT,
            experiment_id=experiment_id,
        )
        self.timeout_seconds = timeout_seconds


class ExperimentMemoryError(ExperimentError):
    """Raised when an experiment exceeds the memory budget."""

    def __init__(self, experiment_id: str, peak_gb: float, budget_gb: float) -> None:
        super().__init__(
            f"Experiment '{experiment_id}' used {peak_gb:.2f} GB RAM, "
            f"exceeding budget of {budget_gb:.2f} GB.",
            failure_type=FailureType.MEMORY,
            experiment_id=experiment_id,
        )
        self.peak_gb = peak_gb
        self.budget_gb = budget_gb


class MaxRepairsExceededError(ExperimentError):
    """Raised when an experiment has been repaired the maximum allowed times."""

    def __init__(self, experiment_id: str, max_repairs: int = 2) -> None:
        super().__init__(
            f"Experiment '{experiment_id}' exceeded max repairs ({max_repairs}). Abandoning.",
            failure_type=FailureType.UNKNOWN,
            experiment_id=experiment_id,
        )
        self.max_repairs = max_repairs


# ---------------------------------------------------------------------------
# Compiler / registry
# ---------------------------------------------------------------------------


class ComponentNotFoundError(MicroAutoMLError):
    """Raised when a requested component is not registered."""

    def __init__(self, component_name: str, component_type: str = "component") -> None:
        super().__init__(
            f"{component_type.capitalize()} '{component_name}' is not registered. "
            "Check the component registry."
        )
        self.component_name = component_name
        self.component_type = component_type


class IncompatibleComponentError(MicroAutoMLError):
    """Raised when two components cannot be combined in a pipeline."""

    def __init__(self, component_a: str, component_b: str, reason: str = "") -> None:
        msg = f"Components '{component_a}' and '{component_b}' are incompatible."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg)


class PipelineCompilationError(MicroAutoMLError):
    """Raised when the pipeline compiler cannot produce a valid pipeline."""


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------


class ResourceDeniedError(MicroAutoMLError):
    """Raised when the resource governor denies an experiment before it starts."""

    def __init__(self, reason: str, estimated_gb: float | None = None) -> None:
        super().__init__(f"Resource denied: {reason}")
        self.estimated_gb = estimated_gb


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchLockedError(MicroAutoMLError):
    """Raised when an attempt is made to run a new experiment after search is locked."""


class FinalTestAlreadyUsedError(MicroAutoMLError):
    """
    Raised when the final holdout test set is accessed more than once.
    This is a hard invariant — the test set must be evaluated exactly once.
    """

    def __init__(self) -> None:
        super().__init__(
            "The final holdout test set has already been evaluated. "
            "It cannot be accessed again. This is a non-negotiable invariant."
        )


class SaturationError(MicroAutoMLError):
    """Raised when the search is declared saturated and cannot continue."""


# ---------------------------------------------------------------------------
# Memory / checkpoint
# ---------------------------------------------------------------------------


class CheckpointError(MicroAutoMLError):
    """Raised when a checkpoint cannot be saved or loaded."""


class RunNotFoundError(MicroAutoMLError):
    """Raised when a run ID does not exist in the database."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run '{run_id}' not found in the experiment database.")
        self.run_id = run_id


# ---------------------------------------------------------------------------
# SLM / planner
# ---------------------------------------------------------------------------


class SLMUnavailableError(MicroAutoMLError):
    """Raised when the SLM is requested but not available. Falls back to rules."""


class HypothesisSchemaError(MicroAutoMLError):
    """Raised when SLM output does not conform to the hypothesis schema."""


class NoHypothesesError(MicroAutoMLError):
    """Raised when the planner produces zero valid hypotheses."""


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class MetricLockedError(MicroAutoMLError):
    """Raised when an attempt is made to change the primary metric after it is locked."""

    def __init__(self) -> None:
        super().__init__(
            "The primary evaluation metric is locked. It cannot be changed "
            "after the first experiment is run."
        )
