"""
MicroAutoML-Agent — Final Model Selector
Picks the absolute best model (or ensemble) to deploy.
"""

from __future__ import annotations

import logging
from typing import Any

from automl.core.schemas import MetricDirection

logger = logging.getLogger(__name__)


class FinalModelSelector:
    """Evaluates the top single model against the ensemble blend to pick the final deployment candidate."""

    def __init__(self, metric_direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER) -> None:
        self.metric_direction = metric_direction

    def select_final(
        self,
        best_single_id: str,
        best_single_score: float,
        ensemble_score: float | None = None,
        ensemble_weights: dict[str, float] | None = None,
        ensemble_overhead_penalty: float = 0.01  # Demand a 1% absolute gain to justify ensemble complexity
    ) -> dict[str, Any]:
        """
        Chooses between the best single model and the ensemble.
        
        Returns a dictionary describing the selection:
            - type: "single" or "ensemble"
            - selection_id: experiment_id (if single) or "ENSEMBLE"
            - weights: dict of experiment_id -> weight (if ensemble)
            - final_cv_score: float
        """
        selection = {
            "type": "single",
            "selection_id": best_single_id,
            "weights": {best_single_id: 1.0},
            "final_cv_score": best_single_score
        }

        if ensemble_score is None or ensemble_weights is None or len(ensemble_weights) < 2:
            logger.info("No valid ensemble available. Selecting best single model.")
            return selection

        from automl.application.metric_utils import is_better
        
        # Check if ensemble beats single model by the penalty margin
        is_ensemble_better = False
        target_score = best_single_score + ensemble_overhead_penalty if self.metric_direction == MetricDirection.HIGHER_IS_BETTER else best_single_score - ensemble_overhead_penalty
        
        if is_better(ensemble_score, target_score, self.metric_direction):
            is_ensemble_better = True

        if is_ensemble_better:
            logger.info(f"Ensemble score ({ensemble_score:.4f}) beats single model ({best_single_score:.4f}) by margin. Selecting ensemble.")
            selection = {
                "type": "ensemble",
                "selection_id": "ENSEMBLE",
                "weights": ensemble_weights,
                "final_cv_score": ensemble_score
            }
        else:
            logger.info(f"Ensemble score ({ensemble_score:.4f}) did not beat single model ({best_single_score:.4f}) by margin. Selecting single.")

        return selection
