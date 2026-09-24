"""
MicroAutoML-Agent — Metric Utilities

Shared functions for comparing scores according to their metric direction.
"""

from automl.core.enums import MetricDirection


def is_better(new_score: float, old_score: float, direction: MetricDirection) -> bool:
    """
    Returns True if new_score is strictly better than old_score
    according to the specified direction.
    """
    if direction == MetricDirection.HIGHER_IS_BETTER:
        return new_score > old_score
    return new_score < old_score

