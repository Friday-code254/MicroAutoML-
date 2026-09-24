"""
MicroAutoML-Agent — Compatibility Checker

Validates an ExperimentSpec against the ComponentRegistry *before* the compiler
builds the pipeline. This is the first gate in the compilation pipeline.

Without this check, the search controller could queue a classifier spec for a
regression task, and the failure would only surface at runtime — wasting compute
budget.  The CompatibilityChecker makes this rejection instant and zero-cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from automl.core.enums import TaskType
from automl.core.schemas import ExperimentSpec
from automl.registry.registry import ComponentRegistry


@dataclass
class CompatibilityReport:
    """Result of a compatibility check."""
    passed: bool
    model_name: str
    task_type: TaskType | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


class CompatibilityChecker:
    """
    Validates ExperimentSpec compatibility with the registry and the active task.

    Checks performed:
    1. Model is registered in ComponentRegistry.
    2. Model supports the detected task type (e.g., classifier ≠ regression task).
    3. (Extendable) Encoder ↔ dtype compatibility and transform ordering.
    """

    def __init__(self, registry: ComponentRegistry | None = None) -> None:
        self.registry = registry or ComponentRegistry()

    def check(
        self,
        spec: ExperimentSpec,
        task_type: TaskType | None = None,
    ) -> CompatibilityReport:
        """
        Check spec compatibility.

        Parameters
        ----------
        spec : ExperimentSpec
            The experiment to validate.
        task_type : TaskType | None
            The active task type from EvidencePack. If None, task-level checks
            are skipped (used in tests / pre-Phase 1 contexts).

        Returns
        -------
        CompatibilityReport
            `passed=True` if all checks clear; `passed=False` with populated
            `errors` list otherwise.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check 1: Model is registered
        model_spec = self.registry.get_model_spec(spec.model_name)
        if model_spec is None:
            errors.append(
                f"Model '{spec.model_name}' is not registered in the ComponentRegistry. "
                f"Available: {sorted(self.registry.list_model_names())}"
            )
            return CompatibilityReport(
                passed=False,
                model_name=spec.model_name,
                task_type=task_type,
                errors=errors,
                warnings=warnings,
            )

        # Check 2: Model supports the task type
        if task_type is not None:
            if task_type not in model_spec.supported_tasks:
                errors.append(
                    f"Model '{spec.model_name}' does not support task '{task_type.value}'. "
                    f"Supported: {[t.value for t in model_spec.supported_tasks]}"
                )

        # Check 3: Hyperparameter keys are not obviously wrong
        if spec.hyperparameters:
            # We can't verify all keys without inspecting the sklearn class, but we can
            # catch common mistakes like passing 'n_estimators' to a linear model.
            if model_spec.family.value == "linear" and "n_estimators" in spec.hyperparameters:
                warnings.append(
                    f"'{spec.model_name}' is a linear model but spec has 'n_estimators'. "
                    "This parameter will be ignored."
                )

        return CompatibilityReport(
            passed=len(errors) == 0,
            model_name=spec.model_name,
            task_type=task_type,
            errors=errors,
            warnings=warnings,
        )

    def assert_compatible(
        self,
        spec: ExperimentSpec,
        task_type: TaskType | None = None,
    ) -> None:
        """
        Like check() but raises ValueError on failure.
        Use this in the compiler where you want a hard fail.
        """
        report = self.check(spec, task_type)
        if not report:
            raise ValueError(
                f"Incompatible ExperimentSpec for experiment {spec.experiment_id}: "
                + "; ".join(report.errors)
            )
