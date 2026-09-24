"""
MicroAutoML-Agent — Stability Analyzer
Detects fold-specific failures, single-fold outliers, and consistent degradation.
"""

from __future__ import annotations

import numpy as np

from automl.core.schemas import MetricDirection


class StabilityAnalyzer:
    """Analyzes per-fold score variance."""

    def __init__(
        self,
        metric_direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
        max_std_threshold: float = 0.05,
        outlier_z_threshold: float = 2.0
    ) -> None:
        self.metric_direction = metric_direction
        self.max_std_threshold = max_std_threshold
        self.outlier_z_threshold = outlier_z_threshold

    def analyze_folds(self, fold_scores: list[float]) -> tuple[bool, list[str]]:
        """
        Analyzes the fold scores.
        Returns:
            is_stable (bool): False if heavily unstable or contains outliers.
            flags (list[str]): List of specific instability flags.
        """
        if not fold_scores or len(fold_scores) < 2:
            return True, []
            
        scores = np.array(fold_scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        flags = []
        is_stable = True
        
        # 1. High Variance
        if std_score > self.max_std_threshold:
            flags.append("UNSTABLE")
            is_stable = False
            
        # 2. Outlier Detection (Z-score)
        if std_score > 0:
            z_scores = np.abs((scores - mean_score) / std_score)
            if np.any(z_scores > self.outlier_z_threshold):
                flags.append("FOLD_OUTLIER")
                is_stable = False
                
        return is_stable, flags
