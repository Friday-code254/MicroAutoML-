"""
Unit tests for Phase 10: Ensemble Engine
"""

import numpy as np
import pytest

from automl.ensemble.oof import OOFPredictor
from automl.ensemble.blend import BlendSelector
from automl.ensemble.selection import FinalModelSelector
from automl.core.schemas import MetricDirection


def test_oof_predictor_aggregation():
    n_samples = 10
    
    # 3 folds
    fold_indices = [
        (np.array([3, 4, 5, 6, 7, 8, 9]), np.array([0, 1, 2])),
        (np.array([0, 1, 2, 6, 7, 8, 9]), np.array([3, 4, 5])),
        (np.array([0, 1, 2, 3, 4, 5, 9]), np.array([6, 7, 8])),
    ]
    
    fold_preds = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6]),
        np.array([0.7, 0.8, 0.9])
    ]
    
    oof = OOFPredictor.aggregate_oof_predictions(n_samples, fold_indices, fold_preds)
    
    # Sample 9 was not in any val_idx. It should be filled with the mean of the others.
    mean_val = np.mean([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    assert len(oof) == 10
    assert oof[0] == pytest.approx(0.1)
    assert oof[4] == pytest.approx(0.5)
    assert oof[8] == pytest.approx(0.9)
    assert oof[9] == pytest.approx(mean_val)


def test_blend_selector():
    selector = BlendSelector(max_ensemble_size=2, correlation_threshold=0.85)
    
    # A and B are identical (corr 1.0)
    # C is different
    candidates = {
        "A": {"oof_preds": np.array([1, 0, 1, 1, 0]), "cv_score_mean": 0.90, "cv_score_std": 0.05},
        "B": {"oof_preds": np.array([1, 0, 1, 1, 0]), "cv_score_mean": 0.88, "cv_score_std": 0.06},
        "C": {"oof_preds": np.array([0, 1, 0, 0, 1]), "cv_score_mean": 0.85, "cv_score_std": 0.04},
    }
    
    selected = selector.select_ensemble(candidates)
    
    # A should be selected first (highest score)
    # B is skipped (correlated with A)
    # C is selected (not correlated with A)
    assert selected == ["A", "C"]
    
    # Check weights calculation
    weights = selector.compute_weights({
        "A": candidates["A"],
        "C": candidates["C"]
    })
    
    # C has lower std (0.04) vs A (0.05), so C should have higher weight
    assert weights["C"] > weights["A"]
    assert pytest.approx(weights["A"] + weights["C"]) == 1.0


def test_final_model_selector():
    selector = FinalModelSelector(metric_direction=MetricDirection.HIGHER_IS_BETTER)
    
    # 1. Ensemble beats single model significantly
    res1 = selector.select_final(
        best_single_id="A", best_single_score=0.90,
        ensemble_score=0.92, ensemble_weights={"A": 0.5, "C": 0.5},
        ensemble_overhead_penalty=0.01
    )
    assert res1["type"] == "ensemble"
    
    # 2. Ensemble only beats by 0.005, which is < penalty (0.01)
    res2 = selector.select_final(
        best_single_id="A", best_single_score=0.90,
        ensemble_score=0.905, ensemble_weights={"A": 0.5, "C": 0.5},
        ensemble_overhead_penalty=0.01
    )
    assert res2["type"] == "single"
    assert res2["selection_id"] == "A"
