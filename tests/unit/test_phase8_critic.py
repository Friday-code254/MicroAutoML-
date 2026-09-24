"""
Unit tests for Phase 8: Critic & Attribution
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from automl.core.enums import OperatorType, Fidelity
from automl.core.schemas import (
    ExperimentResult, 
    ExperimentSpec, 
    MetricDirection, 
    ResourceUsageRecord
)
from automl.critic.critic import NumericalCritic
from automl.critic.stability import StabilityAnalyzer
from automl.critic.attribution import AttributionEngine
from automl.critic.diversity import DiversityAnalyzer
from automl.search.tree import ExperimentTree


def test_stability_analyzer_stable():
    analyzer = StabilityAnalyzer(max_std_threshold=0.05)
    # Very stable scores
    fold_scores = [0.90, 0.91, 0.89, 0.90, 0.92]
    
    is_stable, flags = analyzer.analyze_folds(fold_scores)
    assert is_stable is True
    assert len(flags) == 0


def test_stability_analyzer_unstable():
    analyzer = StabilityAnalyzer(max_std_threshold=0.05)
    # High variance
    fold_scores = [0.90, 0.70, 0.85, 0.95, 0.60]
    
    is_stable, flags = analyzer.analyze_folds(fold_scores)
    assert is_stable is False
    assert "UNSTABLE" in flags


def test_numerical_critic_overfitting():
    critic = NumericalCritic(overfit_threshold=0.10, metric_direction=MetricDirection.HIGHER_IS_BETTER)
    
    result = ExperimentResult(
        experiment_id="EXP_1",
        status="success",
        train_score=0.99,
        cv_score_mean=0.80, # 19% drop, should trigger overfitting
        fold_scores=[0.80, 0.81, 0.79],
        resource_usage=ResourceUsageRecord(peak_ram_gb=1.0, runtime_seconds=10.0)
    )
    
    report = critic.analyze(result)
    assert report.is_overfitting is True
    assert "OVERFITTING" in report.critic_flags


def test_numerical_critic_resources():
    critic = NumericalCritic(memory_heavy_gb=4.0, slow_seconds=300.0)
    
    result = ExperimentResult(
        experiment_id="EXP_1",
        status="success",
        train_score=0.90,
        cv_score_mean=0.88,
        fold_scores=[0.88, 0.89, 0.87],
        resource_usage=ResourceUsageRecord(peak_ram_gb=5.0, runtime_seconds=400.0)
    )
    
    report = critic.analyze(result)
    assert report.is_memory_heavy is True
    assert report.is_slow is True
    assert "MEMORY_HEAVY" in report.critic_flags
    assert "SLOW" in report.critic_flags


def test_attribution_engine():
    tree = ExperimentTree()
    
    # Root
    tree.insert_node(ExperimentSpec(experiment_id="BASE", operator=OperatorType.BASELINE))
    tree.update_node_result(ExperimentResult(
        experiment_id="BASE", status="success", cv_score_mean=0.80,
        resource_usage=ResourceUsageRecord(runtime_seconds=10, peak_ram_gb=1.0)
    ))
    
    # Child
    tree.insert_node(ExperimentSpec(experiment_id="CHILD", parent_id="BASE", operator=OperatorType.FEATURE_ENGINEERING, hypothesis="Added feature A"))
    tree.update_node_result(ExperimentResult(
        experiment_id="CHILD", status="success", cv_score_mean=0.85,
        resource_usage=ResourceUsageRecord(runtime_seconds=15, peak_ram_gb=1.5)
    ))
    
    engine = AttributionEngine(metric_direction=MetricDirection.HIGHER_IS_BETTER)
    attrs = engine.attribute_branch("CHILD", tree)
    
    assert len(attrs) == 1
    assert attrs[0]["parent_id"] == "BASE"
    assert attrs[0]["child_id"] == "CHILD"
    assert attrs[0]["score_delta"] == pytest.approx(0.05)
    assert attrs[0]["runtime_delta"] == pytest.approx(5.0)
    assert attrs[0]["ram_delta"] == pytest.approx(0.5)
    assert attrs[0]["is_improvement"] is True


def test_diversity_analyzer():
    analyzer = DiversityAnalyzer(correlation_threshold=0.85)
    
    # Identical predictions (correlation 1.0)
    p1 = np.array([1, 0, 1, 1, 0])
    p2 = np.array([1, 0, 1, 1, 0])
    assert analyzer.are_complementary(p1, p2) is False
    
    # Different predictions
    p3 = np.array([0, 1, 0, 0, 1])
    assert analyzer.are_complementary(p1, p3) is True
