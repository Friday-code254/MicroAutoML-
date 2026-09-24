"""
MicroAutoML-Agent — Repair Engine
Coordinates the repair sequence: Classify -> Policy -> Modify Spec.
"""

from __future__ import annotations

import logging

from automl.core.enums import OperatorType
from automl.core.schemas import ExperimentResult, ExperimentSpec, FailureReport, _new_id
from automl.repair.classifier import FailureClassifier
from automl.repair.policies import RepairPolicies

logger = logging.getLogger(__name__)


class RepairEngine:
    """Manages automated recovery of failed experiments."""

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries
        self.classifier = FailureClassifier()

    def attempt_repair(
        self,
        failed_spec: ExperimentSpec,
        result: ExperimentResult
    ) -> tuple[ExperimentSpec | None, FailureReport]:
        """
        Takes a failed spec and its result, classifies the failure, and attempts a repair.
        Returns the repaired spec (if allowed) and the failure report.
        """
        # 1. Classify
        failure_type = result.failure_type
        if failure_type is None:
            failure_type = self.classifier.classify(
                exception_type="UnknownException", 
                error_message=result.error_message or "", 
                traceback_str=result.error_traceback or ""
            )

        report = FailureReport(
            experiment_id=failed_spec.experiment_id,
            failure_type=failure_type,
            error_message=result.error_message or "",
            traceback=result.error_traceback or "",
            original_spec=failed_spec,
            repair_count=failed_spec.repair_count
        )

        # 2. Check limits
        if failed_spec.repair_count >= self.max_retries:
            logger.warning(f"Experiment {failed_spec.experiment_id} failed {failed_spec.repair_count} times. Max retries reached.")
            report.repair_applied = "Max retries reached."
            return None, report

        # 3. Apply Policy
        repaired_spec, action = RepairPolicies.apply_policy(failed_spec, failure_type)
        
        # 4. Finalize repaired spec
        # Give it a new ID so we track it separately in the tree, but keep same parent!
        new_id = _new_id("EXP")
        repaired_spec.experiment_id = new_id
        repaired_spec.operator = OperatorType.REPAIR
        repaired_spec.hypothesis = f"[REPAIR] {action} | Originally: {failed_spec.hypothesis}"
        
        report.repaired_spec = repaired_spec
        report.repair_applied = action
        report.repair_successful = action != "No repair policy defined." and "Failed to find" not in action and "mock implementation" not in action
        
        # In a real setup, we might also want to mark the failure as not successfully repaired if it's a mock action
        if "mock implementation" in action or "No automated repair" in action:
            report.repair_successful = False
            return None, report

        logger.info(f"Generated repair for {failed_spec.experiment_id}: {action}")
        return repaired_spec, report
