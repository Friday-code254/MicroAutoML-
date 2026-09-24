"""
MicroAutoML-Agent — Repair Policies
Deterministic heuristics mapping FailureTypes to modified ExperimentSpecs.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from automl.core.enums import FailureType
from automl.core.schemas import ExperimentSpec

logger = logging.getLogger(__name__)


class RepairPolicies:
    """Provides methods to modify an ExperimentSpec to fix a specific FailureType."""

    @classmethod
    def apply_policy(cls, spec: ExperimentSpec, failure_type: FailureType) -> tuple[ExperimentSpec, str]:
        """
        Applies a repair policy based on failure type.
        Returns:
            (repaired_spec, applied_action_description)
        """
        new_spec = copy.deepcopy(spec)
        action_taken = "No repair policy defined."

        if failure_type == FailureType.MEMORY:
            # Policy: Reduce model complexity or data fraction
            if "n_estimators" in new_spec.hyperparameters:
                new_val = max(10, new_spec.hyperparameters["n_estimators"] // 2)
                new_spec.hyperparameters["n_estimators"] = new_val
                action_taken = f"Reduced n_estimators to {new_val}"
            elif "max_depth" in new_spec.hyperparameters and new_spec.hyperparameters["max_depth"] is not None:
                new_val = max(3, new_spec.hyperparameters["max_depth"] // 2)
                new_spec.hyperparameters["max_depth"] = new_val
                action_taken = f"Reduced max_depth to {new_val}"
            else:
                action_taken = "Failed to find memory parameter to reduce."
                
        elif failure_type == FailureType.TIMEOUT:
            # Policy: Same as memory (reduce iterations)
            if "max_iter" in new_spec.hyperparameters:
                new_val = max(50, new_spec.hyperparameters["max_iter"] // 2)
                new_spec.hyperparameters["max_iter"] = new_val
                action_taken = f"Reduced max_iter to {new_val}"
            elif "n_estimators" in new_spec.hyperparameters:
                new_val = max(10, new_spec.hyperparameters["n_estimators"] // 2)
                new_spec.hyperparameters["n_estimators"] = new_val
                action_taken = f"Reduced n_estimators to {new_val}"
            else:
                action_taken = "Failed to find timeout parameter to reduce."
                
        elif failure_type == FailureType.TYPE:
            # Policy: TYPE errors usually mean unencoded strings reached an estimator.
            # We explicitly inject an OrdinalEncoder as the first preprocessing step.
            has_encoder = any(p.get("name") == "OrdinalEncoder" for p in new_spec.preprocessing)
            if not has_encoder:
                new_spec.preprocessing.insert(0, {
                    "name": "OrdinalEncoder",
                    "params": {"handle_unknown": "use_encoded_value", "unknown_value": -1}
                })
                action_taken = "Injected OrdinalEncoder to handle categorical/string types."
            else:
                action_taken = "OrdinalEncoder already present, cannot repair TYPE error."
                
        elif failure_type == FailureType.NUMERICAL:
            # Policy: Often caused by lack of scaling or clipping
            # Try adding StandardScaler if not present
            has_scaler = any(p.get("name") == "StandardScaler" for p in new_spec.preprocessing)
            if not has_scaler:
                new_spec.preprocessing.append({
                    "name": "StandardScaler",
                    "params": {}
                })
                action_taken = "Appended StandardScaler to fix numerical instability."
            else:
                action_taken = "StandardScaler already present, cannot repair NUMERICAL error."

        elif failure_type == FailureType.SHAPE:
            # Policy: Shape mismatch often caused by faulty feature selection or generation.
            # Remove the last preprocessing or feature selection step.
            if new_spec.feature_selection:
                new_spec.feature_selection = {}
                action_taken = "Removed feature selection step to fix SHAPE error."
            elif new_spec.preprocessing:
                removed = new_spec.preprocessing.pop()
                action_taken = f"Removed last preprocessing step ({removed.get('name')}) to fix SHAPE error."
            else:
                action_taken = "No steps to remove for SHAPE error."

        else:
            action_taken = f"No automated repair for {failure_type.value}."

        return new_spec, action_taken
