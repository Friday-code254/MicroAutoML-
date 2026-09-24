"""
MicroAutoML-Agent — Classification Metrics

Evaluates predictions for binary and multiclass tasks.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    brier_score_loss
)
import logging

logger = logging.getLogger(__name__)

from automl.core.enums import MetricName, TaskType


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None,
    primary_metric: MetricName,
    task_type: TaskType,
    labels: list | None = None,
) -> dict[str, float]:
    """
    Compute all classification metrics, returning a dictionary of scores.
    The primary_metric is guaranteed to be included.
    """
    scores = {}
    has_proba = y_pred_proba is not None
    is_multiclass = task_type == TaskType.MULTICLASS_CLASSIFICATION
    
    pos_label = 1
    if not is_multiclass:
        unique_labels = np.unique(y_true)
        if len(unique_labels) == 2 and 1 not in unique_labels:
            pos_label = unique_labels[1]
            y_true = (y_true == pos_label).astype(int)
            y_pred = (y_pred == pos_label).astype(int)
    
    # 1. Accuracy
    try:
        scores[MetricName.ACCURACY.value] = float(accuracy_score(y_true, y_pred))
    except Exception as e:
        logger.warning(f"Failed to compute metric: {e}")
        
    # 2. F1 Score
    try:
        average_method = "macro" if is_multiclass else "binary"
        scores[MetricName.F1.value] = float(f1_score(y_true, y_pred, average=average_method))
    except Exception as e:
        logger.warning(f"Failed to compute metric: {e}")
        
    # 3. ROC AUC (requires proba)
    if has_proba:
        try:
            if is_multiclass:
                scores[MetricName.ROC_AUC.value] = float(
                    roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro", labels=labels)
                )
            else:
                proba_pos = y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba
                scores[MetricName.ROC_AUC.value] = float(roc_auc_score(y_true, proba_pos))
        except Exception:
            pass
            
    # 4. Log Loss (requires proba)
    if has_proba:
        try:
            scores[MetricName.LOG_LOSS.value] = float(log_loss(y_true, y_pred_proba, labels=labels))
        except Exception:
            pass
            
        try:
            # Brier score is only implemented for binary classification in sklearn natively
            if not is_multiclass:
                proba_pos = y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba
                scores[MetricName.BRIER_SCORE.value] = float(brier_score_loss(y_true, proba_pos))
        except Exception:
            pass
            
    # 5. Precision & Recall
    try:
        average_method = "macro" if is_multiclass else "binary"
        scores[MetricName.PRECISION.value] = float(precision_score(y_true, y_pred, average=average_method, zero_division=0))
        scores[MetricName.RECALL.value] = float(recall_score(y_true, y_pred, average=average_method, zero_division=0))
    except Exception as e:
        logger.warning(f"Failed to compute metric: {e}")
            
    # Verification of primary metric
    if primary_metric.value not in scores:
        if primary_metric in (MetricName.LOG_LOSS, MetricName.ROC_AUC) and not has_proba:
            # Fallback to accuracy if probabilities are missing
            if MetricName.ACCURACY.value in scores:
                logger.warning(f"Falling back to accuracy for {primary_metric.value} due to missing probabilities.")
                scores[primary_metric.value] = scores[MetricName.ACCURACY.value]
            else:
                raise ValueError(f"Primary metric {primary_metric.value} requires probabilities, but model did not output them.")
        else:
            raise RuntimeError(f"Failed to compute primary metric {primary_metric.value}.")
            
    return scores
