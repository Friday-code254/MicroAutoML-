"""
MicroAutoML-Agent — Regression Metrics

Evaluates predictions for regression tasks.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import logging

logger = logging.getLogger(__name__)

from automl.core.enums import MetricName


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    primary_metric: MetricName,
) -> dict[str, float]:
    """
    Compute all regression metrics, returning a dictionary of scores.
    The primary_metric is guaranteed to be included.
    """
    scores = {}
    
    # 1. RMSE (Root Mean Squared Error)
    try:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        scores[MetricName.RMSE.value] = rmse
    except Exception as e:
        logger.warning(f"Failed to compute metric: {e}")
        
    # 2. MAE (Mean Absolute Error)
    try:
        scores[MetricName.MAE.value] = float(mean_absolute_error(y_true, y_pred))
    except Exception as e:
        logger.warning(f"Failed to compute metric: {e}")
        
    # 3. R2 Score
    try:
        scores[MetricName.R2.value] = float(r2_score(y_true, y_pred))
    except Exception as e:
        logger.warning(f"Failed to compute metric: {e}")
        
    # Verification of primary metric
    if primary_metric.value not in scores:
        raise RuntimeError(f"Failed to compute primary metric {primary_metric.value}.")
        
    return scores
