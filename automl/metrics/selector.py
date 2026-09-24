"""
MicroAutoML-Agent — Metric Selector

Automatically selects the best primary optimization metric based on the 
TaskProfile and dataset evidence, unless explicitly overridden by the user.

Rules:
- Binary Classification (imbalanced): ROC_AUC or LOG_LOSS
- Binary Classification (balanced): ROC_AUC or ACCURACY
- Multiclass Classification: LOG_LOSS or MACRO_F1
- Regression: RMSE or MAE
"""

from __future__ import annotations

from automl.core.enums import MetricDirection, MetricName, TaskType
from automl.core.schemas import TaskProfile, FastProfile


class MetricSelector:
    """
    Selects the primary evaluation metric and its optimization direction.
    """

    def select(
        self,
        task: TaskProfile,
        fast_profile: FastProfile | None = None,
        force_metric: str | None = None,
    ) -> tuple[MetricName, MetricDirection]:
        """
        Returns (metric_name, metric_direction)
        """
        # 1. Force override
        if force_metric:
            try:
                name = MetricName(force_metric.lower())
                return name, self._get_direction(name)
            except ValueError:
                pass  # Fall back to default if invalid

        # 2. Binary Classification
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            # If highly imbalanced, ROC AUC is safer than log loss (which can blow up)
            imbalance = fast_profile.target_class_imbalance if fast_profile else None
            if imbalance is not None and imbalance < 0.10:
                return MetricName.ROC_AUC, MetricDirection.HIGHER_IS_BETTER
            return MetricName.LOG_LOSS, MetricDirection.LOWER_IS_BETTER

        # 3. Multiclass Classification
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return MetricName.LOG_LOSS, MetricDirection.LOWER_IS_BETTER

        # 4. Regression
        if task.task_type == TaskType.REGRESSION:
            # We default to RMSE for regression
            return MetricName.RMSE, MetricDirection.LOWER_IS_BETTER

        # Fallback
        return MetricName.RMSE, MetricDirection.LOWER_IS_BETTER

    def _get_direction(self, name: MetricName) -> MetricDirection:
        if name in (MetricName.ROC_AUC, MetricName.ACCURACY, MetricName.F1, MetricName.R2):
            return MetricDirection.HIGHER_IS_BETTER
        return MetricDirection.LOWER_IS_BETTER
