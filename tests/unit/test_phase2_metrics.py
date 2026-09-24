"""
Phase 2 Tests — Metric Engine & Baseline Hub

Tests:
- MetricSelector assigns correct metrics for binary, multiclass, regression
- MetricSelector handles class imbalance fallback for binary
- Classification metrics compute correctly (accuracy, log loss, roc auc, f1)
- Regression metrics compute correctly (rmse, mae, r2)
- Metrics gracefully fallback to default score if probability/true label issues occur
- BaselineHub generates dummy, linear, and tree specs with correct model names for task
"""

from __future__ import annotations

import numpy as np
import pytest

from automl.core.enums import DatasetStructure, MetricDirection, MetricName, OperatorType, TaskType
from automl.core.schemas import FastProfile, TaskProfile
from automl.metrics import MetricSelector, evaluate_classification, evaluate_regression
from automl.models.baseline_hub import BaselineHub


class TestMetricSelector:
    def test_binary_classification_balanced(self) -> None:
        task = TaskProfile(
            task_type=TaskType.BINARY_CLASSIFICATION,
            dataset_structure=DatasetStructure.IID,
            target_column="target",
            n_classes=2,
        )
        fast_profile = FastProfile(target_class_imbalance=0.4) # balanced
        selector = MetricSelector()
        metric, direction = selector.select(task, fast_profile)
        assert metric == MetricName.LOG_LOSS
        assert direction == MetricDirection.LOWER_IS_BETTER

    def test_binary_classification_imbalanced(self) -> None:
        task = TaskProfile(
            task_type=TaskType.BINARY_CLASSIFICATION,
            dataset_structure=DatasetStructure.IID,
            target_column="target",
            n_classes=2,
        )
        fast_profile = FastProfile(target_class_imbalance=0.05) # highly imbalanced
        selector = MetricSelector()
        metric, direction = selector.select(task, fast_profile)
        assert metric == MetricName.ROC_AUC
        assert direction == MetricDirection.HIGHER_IS_BETTER

    def test_multiclass_classification(self) -> None:
        task = TaskProfile(
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            dataset_structure=DatasetStructure.IID,
            target_column="target",
            n_classes=3,
        )
        selector = MetricSelector()
        metric, direction = selector.select(task)
        assert metric == MetricName.LOG_LOSS
        assert direction == MetricDirection.LOWER_IS_BETTER

    def test_regression(self) -> None:
        task = TaskProfile(
            task_type=TaskType.REGRESSION,
            dataset_structure=DatasetStructure.IID,
            target_column="target",
        )
        selector = MetricSelector()
        metric, direction = selector.select(task)
        assert metric == MetricName.RMSE
        assert direction == MetricDirection.LOWER_IS_BETTER

    def test_force_metric(self) -> None:
        task = TaskProfile(
            task_type=TaskType.BINARY_CLASSIFICATION,
            dataset_structure=DatasetStructure.IID,
            target_column="target",
        )
        selector = MetricSelector()
        metric, direction = selector.select(task, force_metric="accuracy")
        assert metric == MetricName.ACCURACY
        assert direction == MetricDirection.HIGHER_IS_BETTER


class TestClassificationMetrics:
    def test_binary_evaluation(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_pred_proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.4, 0.6], [0.3, 0.7]])
        
        scores = evaluate_classification(
            y_true, y_pred, y_pred_proba, primary_metric=MetricName.ROC_AUC, task_type=TaskType.BINARY_CLASSIFICATION
        )
        
        assert MetricName.ACCURACY.value in scores
        assert MetricName.F1.value in scores
        assert MetricName.ROC_AUC.value in scores
        assert MetricName.LOG_LOSS.value in scores
        assert scores[MetricName.ACCURACY.value] == 0.75

    def test_fallback_without_proba(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        scores = evaluate_classification(
            y_true, y_pred, y_pred_proba=None, primary_metric=MetricName.LOG_LOSS, task_type=TaskType.BINARY_CLASSIFICATION
        )
        
        # Log loss and ROC AUC require proba, so they shouldn't be computed normally, 
        # but the fallback mechanism guarantees primary_metric is present.
        assert MetricName.LOG_LOSS.value in scores
        assert scores[MetricName.LOG_LOSS.value] == 0.75 # Defaulted to accuracy fallback


class TestRegressionMetrics:
    def test_regression_evaluation(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        
        scores = evaluate_regression(
            y_true, y_pred, primary_metric=MetricName.RMSE
        )
        
        assert MetricName.RMSE.value in scores
        assert MetricName.MAE.value in scores
        assert MetricName.R2.value in scores
        
        assert scores[MetricName.MAE.value] == pytest.approx(0.15)


class TestBaselineHub:
    def test_binary_baselines(self) -> None:
        task = TaskProfile(
            task_type=TaskType.BINARY_CLASSIFICATION,
            dataset_structure=DatasetStructure.IID,
            target_column="target",
            n_classes=2,
        )
        hub = BaselineHub()
        specs = hub.generate_baselines(task)
        
        assert len(specs) == 5
        model_names = [s.model_name for s in specs]
        assert "DummyClassifier" in model_names
        assert "LogisticRegression" in model_names
        assert "RandomForestClassifier" in model_names
        assert "ExtraTreesClassifier" in model_names
        assert "HistGradientBoostingClassifier" in model_names
        
        for spec in specs:
            assert spec.operator == OperatorType.BASELINE
            assert spec.parent_id is None

    def test_regression_baselines(self) -> None:
        task = TaskProfile(
            task_type=TaskType.REGRESSION,
            dataset_structure=DatasetStructure.IID,
            target_column="target",
        )
        hub = BaselineHub()
        specs = hub.generate_baselines(task)
        
        assert len(specs) == 5
        model_names = [s.model_name for s in specs]
        assert "DummyRegressor" in model_names
        assert "Ridge" in model_names
        assert "RandomForestRegressor" in model_names
        assert "ExtraTreesRegressor" in model_names
        assert "HistGradientBoostingRegressor" in model_names
