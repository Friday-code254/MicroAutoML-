"""
MicroAutoML-Agent — Diversity Analyzer
Evaluates Out-Of-Fold (OOF) prediction correlation between candidates.
"""

from __future__ import annotations

import numpy as np


class DiversityAnalyzer:
    """Computes correlation between Out-Of-Fold predictions to find complementary models."""

    def __init__(self, correlation_threshold: float = 0.85) -> None:
        """
        Args:
            correlation_threshold: Pairs with correlation below this are considered complementary.
        """
        self.correlation_threshold = correlation_threshold

    def are_complementary(self, oof_preds_1: np.ndarray, oof_preds_2: np.ndarray) -> bool:
        """Returns True if the two sets of predictions are sufficiently uncorrelated."""
        if len(oof_preds_1) != len(oof_preds_2):
            return False
            
        if len(oof_preds_1) == 0:
            return False

        # Flatten in case of probas (N, C) -> (N * C,)
        p1 = oof_preds_1.flatten()
        p2 = oof_preds_2.flatten()
        
        # If either is constant, correlation is undefined (0 variance)
        if np.std(p1) == 0 or np.std(p2) == 0:
            return False

        correlation = np.corrcoef(p1, p2)[0, 1]
        
        return bool(correlation < self.correlation_threshold)
