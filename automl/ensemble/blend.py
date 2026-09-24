"""
MicroAutoML-Agent — Diversity & Blend Selection
Selects complementary candidates and computes weighted blends.
"""

from __future__ import annotations

import logging
import numpy as np

from automl.core.enums import MetricDirection
from automl.critic.diversity import DiversityAnalyzer

logger = logging.getLogger(__name__)


class BlendSelector:
    """Selects complementary candidates and generates blended predictions."""

    def __init__(self, max_ensemble_size: int = 3, correlation_threshold: float = 0.85) -> None:
        self.max_ensemble_size = max_ensemble_size
        self.diversity_analyzer = DiversityAnalyzer(correlation_threshold=correlation_threshold)

    def select_ensemble(self, candidates: dict[str, dict], direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER) -> list[str]:
        """
        Selects a subset of complementary candidates for ensembling.
        
        Args:
            candidates: Dict mapping experiment_id -> dict with keys:
                - 'oof_preds': np.ndarray
                - 'cv_score_mean': float
                - 'cv_score_std': float
            direction: MetricDirection for sorting
                
        Returns:
            List of selected experiment_ids.
        """
        from functools import cmp_to_key
        from automl.application.metric_utils import is_better
        
        def compare_candidates(k1, k2):
            if is_better(candidates[k1]['cv_score_mean'], candidates[k2]['cv_score_mean'], direction):
                return -1
            elif is_better(candidates[k2]['cv_score_mean'], candidates[k1]['cv_score_mean'], direction):
                return 1
            return 0
            
        sorted_ids = sorted(
            candidates.keys(), 
            key=cmp_to_key(compare_candidates)
        )
        
        if not sorted_ids:
            return []
            
        # Start with the best model
        selected = [sorted_ids[0]]
        
        # Greedily add complementary models
        for candidate_id in sorted_ids[1:]:
            if len(selected) >= self.max_ensemble_size:
                break
                
            candidate_oof = candidates[candidate_id]['oof_preds']
            
            # Check if complementary to ALL currently selected models
            is_complementary = True
            for sel_id in selected:
                sel_oof = candidates[sel_id]['oof_preds']
                if not self.diversity_analyzer.are_complementary(sel_oof, candidate_oof):
                    is_complementary = False
                    break
                    
            if is_complementary:
                selected.append(candidate_id)
                
        return selected

    def compute_weights(self, selected_candidates: dict[str, dict]) -> dict[str, float]:
        """
        Computes simple inverse-variance weights for the selected candidates.
        """
        weights = {}
        total_inv_var = 0.0
        
        for exp_id, data in selected_candidates.items():
            # Add small epsilon to prevent div by zero
            std = max(data['cv_score_std'], 1e-6)
            inv_var = 1.0 / (std ** 2)
            weights[exp_id] = inv_var
            total_inv_var += inv_var
            
        # Normalize
        for exp_id in weights:
            weights[exp_id] /= total_inv_var
            
        return weights

    def blend_predictions(self, predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
        """
        Blends predictions using the computed weights.
        """
        blended = None
        for exp_id, weight in weights.items():
            preds = predictions[exp_id]
            if blended is None:
                blended = np.zeros_like(preds, dtype=float)
            blended += preds * weight
            
        return blended
