"""
MicroAutoML-Agent — Invariants Enforcement Layer

Strict runtime assertions guaranteeing the central claims of the research architecture.
These checks actively crash the system if an invariant is violated, rather than
relying on developer discipline or silent fails.
"""

from __future__ import annotations

import logging
from typing import Any

from automl.core.enums import SaturationState
from automl.core.schemas import EvidencePack, ExperimentSpec, SearchState, ResourceBudget
from automl.core.exceptions import MetricLockedError

logger = logging.getLogger(__name__)


class InvariantViolation(Exception):
    """Raised when a core system invariant is breached."""
    pass


def assert_metric_locked(evidence: EvidencePack) -> None:
    """INVARIANT: The primary metric must be explicitly locked before ANY experiments run."""
    if not evidence.metric_locked or evidence.primary_metric is None:
        raise InvariantViolation(
            "CRITICAL: Metric was not locked in EvidencePack. "
            "Baselines and Search cannot proceed until the metric is locked."
        )


def assert_test_not_used(evidence: EvidencePack) -> None:
    """INVARIANT: The test set can only be evaluated EXACTLY ONCE at the end of the run."""
    if evidence.split and evidence.split.test_set_used:
        raise InvariantViolation(
            "CRITICAL: The holdout test set has already been evaluated. "
            "Multiple evaluations are strictly forbidden to prevent test-set optimization."
        )


def assert_valid_parent(spec: ExperimentSpec, tree: Any) -> None:
    """INVARIANT: Every non-baseline experiment MUST derive from an existing node."""
    if spec.operator.value != "baseline" and spec.parent_id is None:
        raise InvariantViolation(f"CRITICAL: Non-baseline experiment {spec.experiment_id} has no parent_id.")
    if spec.parent_id and spec.parent_id != "BASELINE":
        # Check if parent exists in tree
        if not tree.get_node(spec.parent_id):
            raise InvariantViolation(
                f"CRITICAL: Parent node {spec.parent_id} for experiment {spec.experiment_id} does not exist in the tree."
            )


def assert_repair_limit(spec: ExperimentSpec, max_repairs: int = 2) -> None:
    """INVARIANT: A failed experiment branch can only be repaired up to a strict limit."""
    # We enforce this via a counter that should be injected in spec.repair_count
    repair_count = getattr(spec, "repair_count", 0)
    if repair_count > max_repairs:
        raise InvariantViolation(
            f"CRITICAL: Experiment {spec.experiment_id} has exceeded the max repair limit ({max_repairs})."
        )


def assert_resource_budget(state: SearchState) -> None:
    """INVARIANT: The overall time budget must not be exceeded before launching new work."""
    if state.budget_remaining_seconds <= 0:
        raise InvariantViolation(
            f"CRITICAL: Time budget exhausted. Remaining: {state.budget_remaining_seconds}s. "
            "Cannot launch new experiments."
        )


def assert_unique_experiment(spec: ExperimentSpec, state: SearchState) -> None:
    """INVARIANT: The same experiment ID cannot be executed twice."""
    if spec.experiment_id in state.completed_experiments or spec.experiment_id in state.failed_experiments:
        raise InvariantViolation(
            f"CRITICAL: Experiment {spec.experiment_id} has already been executed."
        )


def assert_seed_recorded(result: Any) -> None:
    """INVARIANT: Every executed experiment must explicitly record its random seeds."""
    if not hasattr(result, "random_seeds") or not isinstance(result.random_seeds, dict):
        raise InvariantViolation(
            f"CRITICAL: Experiment {result.experiment_id} did not record random seeds."
        )
