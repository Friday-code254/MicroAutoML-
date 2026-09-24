"""
MicroAutoML-Agent — Final Holdout Evaluator
Evaluates the final selected model(s) exactly once against the untouched test set.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from automl.core.enums import SaturationState, TaskType
from automl.core.schemas import SearchState, EvidencePack
from automl.core.exceptions import FinalTestAlreadyUsedError
from automl.metrics.classification import evaluate_classification
from automl.metrics.regression import evaluate_regression

logger = logging.getLogger(__name__)


class FinalHoldoutEvaluator:
    """Safeguards the test set and performs the one-time final evaluation."""

    def __init__(self) -> None:
        self._test_used = False

    def evaluate(
        self,
        state: SearchState,
        final_model_selection: dict[str, Any],
        X_test: Any,
        y_test: Any,
        fitted_pipelines: dict[str, Any],
        evidence: EvidencePack
    ) -> float:
        if self._test_used or (evidence.split and evidence.split.test_set_used):
            raise FinalTestAlreadyUsedError()
            
        if state.saturation_state not in (SaturationState.SATURATED, SaturationState.BUDGET_EXHAUSTED):
            raise RuntimeError(
                "CRITICAL VIOLATION: Evaluating test set before search is officially saturated! "
                f"Current state: {state.saturation_state.value}"
            )
            
        self._test_used = True
        if evidence.split:
            evidence.split.test_set_used = True
        # Note: We should ideally persist state here but that happens in the orchestrator
            
        logger.info(f"LOCKING SEARCH. Evaluating final model selection: {final_model_selection['selection_id']}")

        selection_type = final_model_selection["type"]
        weights = final_model_selection["weights"]
        
        is_classification = evidence.task.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        
        predictions = None
        probabilities = None
        
        for exp_id, weight in weights.items():
            pipeline = fitted_pipelines[exp_id]
            
            preds = pipeline.predict(X_test)
            
            if is_classification:
                # Store hard predictions (preserves string labels)
                predictions = preds
                
                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(X_test)
                    if probabilities is None:
                        probabilities = np.zeros_like(probs, dtype=float)
                    probabilities += probs * weight
            else:
                # Accumulate numeric predictions for regression
                if predictions is None:
                    predictions = np.zeros_like(preds, dtype=float)
                predictions += preds * weight
            
        # Score using central metrics
        primary_metric = evidence.primary_metric
        
        if primary_metric is None:
            raise RuntimeError(
                "CRITICAL VIOLATION: Cannot evaluate test set: primary_metric is None. "
                "Metric selection must occur before final evaluation."
            )
        
        try:
            if is_classification:
                final_preds = predictions
                    
                scores = evaluate_classification(
                    y_test, 
                    final_preds, 
                    probabilities, 
                    primary_metric=primary_metric,
                    task_type=evidence.task.task_type
                )
            else:
                scores = evaluate_regression(y_test, predictions, primary_metric=primary_metric)
                
            logger.info("--- Final Holdout Metrics ---")
            for m, val in scores.items():
                logger.info(f"  {m}: {val:.4f}")
                
            final_score = scores.get(primary_metric.value, 0.0)
            return float(final_score)
            
        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")
            raise
