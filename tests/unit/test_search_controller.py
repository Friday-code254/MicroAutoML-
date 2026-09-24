"""
Unit tests for SearchController (The Brain)
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock

from automl.core.enums import SaturationState, FailureType, MetricDirection, ExperimentStatus
from automl.core.schemas import SearchState, EvidencePack, TaskProfile, ExperimentSpec, ExperimentResult, HypothesisSet, Hypothesis
from automl.search.controller import SearchController
from automl.search.tree import ExperimentTree
from automl.memory.experiments import ExperimentStore
from automl.core.config import RunConfig
from automl.application.run_context import RunContext


@pytest.fixture
def base_context():
    config = RunConfig.cpu_default()
    # Speed up tests
    config.search.max_repairs_per_experiment = 1
    
    state = SearchState(run_id="TEST_RUN")
    tree = ExperimentTree()
    
    # Mock store
    store = MagicMock(spec=ExperimentStore)
    
    evidence = EvidencePack(
        task=TaskProfile(task_type="binary_classification"),
        primary_metric="roc_auc",
        metric_direction=MetricDirection.HIGHER_IS_BETTER
    )
    
    context = RunContext(
        config=config,
        store=store,
        evidence=evidence,
        state=state,
        tree=tree,
        dataset_id="mock_ds"
    )
    return context


def test_search_controller_saturation_immediate(base_context):
    """Test that the loop exits immediately if saturation is reached on the first check."""
    planner = MagicMock()
    selector = MagicMock()
    runner = MagicMock()
    governor = MagicMock()
    detector = MagicMock()
    strategist = MagicMock()
    
    # Force immediate saturation
    detector.check_saturation.return_value = SaturationState.SATURATED
    
    controller = SearchController(planner, selector, runner, governor, detector, strategist)
    
    X = pd.DataFrame({"A": [1, 2]})
    y = pd.Series([0, 1])
    
    controller.search(
        X_train=X, 
        y_train=y, 
        evidence=base_context.evidence, 
        base_model_name="MockModel", 
        base_hyperparameters={}, 
        context=base_context
    )
    
    # Verify planner was never called
    planner.generate.assert_not_called()
    assert base_context.state.saturation_state == SaturationState.SATURATED


def test_search_controller_successful_loop(base_context):
    """Test a single successful iteration of the search loop that then saturates."""
    planner = MagicMock()
    selector = MagicMock()
    runner = MagicMock()
    governor = MagicMock()
    detector = MagicMock()
    strategist = MagicMock()
    
    # Cycle 1: SEARCHING, Cycle 2: SATURATED
    detector.check_saturation.side_effect = [SaturationState.SEARCHING, SaturationState.SATURATED]
    
    # Mock Planner
    mock_hyp = Hypothesis(hypothesis="Mock hyp 1", experiment_type="hpo")
    hyp_set = HypothesisSet(planning_round=1, hypotheses=[mock_hyp])
    planner.generate.return_value = hyp_set
    
    # Mock Selector
    mock_spec = ExperimentSpec(
        experiment_id="EXP_1",
        model_name="MockModel",
        hyperparameters={},
        features=[],
        hypothesis="Mock hyp 1"
    )
    selector.select.return_value = [mock_spec]
    
    # Mock Runner
    mock_result = ExperimentResult(
        experiment_id="EXP_1",
        status=ExperimentStatus.SUCCESS,
        cv_score_mean=0.9,
        cv_score_std=0.01,
        train_time_seconds=1.0
    )
    runner.run.return_value = mock_result
    
    controller = SearchController(planner, selector, runner, governor, detector, strategist)
    
    X = pd.DataFrame({"A": [1, 2]})
    y = pd.Series([0, 1])
    
    controller.search(
        X_train=X, 
        y_train=y, 
        evidence=base_context.evidence, 
        base_model_name="MockModel", 
        base_hyperparameters={}, 
        context=base_context
    )
    
    # Verify executions
    assert planner.generate.call_count == 1
    assert selector.select.call_count == 1
    assert runner.run.call_count == 1
    
    # Verify state updates
    assert "EXP_1" in base_context.state.completed_experiments
    assert base_context.state.incumbent_score == 0.9
    assert base_context.state.incumbent_experiment_id == "EXP_1"


def test_search_controller_self_healing(base_context):
    """Test that a failed experiment triggers the self-healing repair policy loop."""
    planner = MagicMock()
    selector = MagicMock()
    runner = MagicMock()
    governor = MagicMock()
    detector = MagicMock()
    strategist = MagicMock()
    
    detector.check_saturation.side_effect = [SaturationState.SEARCHING, SaturationState.SATURATED]
    
    mock_spec = ExperimentSpec(experiment_id="EXP_FAIL", model_name="MockModel", hyperparameters={}, features=[])
    mock_hyp = Hypothesis(hypothesis="Mock hyp 1", experiment_type="repair")
    planner.generate.return_value = HypothesisSet(planning_round=1, hypotheses=[mock_hyp])
    selector.select.return_value = [mock_spec]
    
    # First run fails, second run succeeds
    fail_result = ExperimentResult(
        experiment_id="EXP_FAIL",
        status=ExperimentStatus.FAILED,
        error_message="OOM Error",
        error_traceback="Exception: MemoryError"
    )
    success_result = ExperimentResult(
        experiment_id="EXP_REPAIRED",
        status=ExperimentStatus.SUCCESS,
        cv_score_mean=0.85
    )
    runner.run.side_effect = [fail_result, success_result]
    
    controller = SearchController(planner, selector, runner, governor, detector, strategist)
    
    # Mock the repair policies to return a repaired spec
    repaired_spec = ExperimentSpec(experiment_id="EXP_REPAIRED", model_name="MockModel", hyperparameters={"repaired": True}, features=[])
    controller.repair_policies.apply_policy = MagicMock(return_value=(repaired_spec, "action"))
    controller.failure_classifier.classify = MagicMock(return_value=FailureType.MEMORY)
    
    X = pd.DataFrame({"A": [1, 2]})
    y = pd.Series([0, 1])
    
    controller.search(
        X_train=X, y_train=y, evidence=base_context.evidence,
        base_model_name="Mock", base_hyperparameters={}, context=base_context
    )
    
    # Runner should be called twice (original + 1 repair)
    assert runner.run.call_count == 2
    assert base_context.state.failed_experiments == ["EXP_FAIL"]
    assert "EXP_REPAIRED" in base_context.state.completed_experiments
    assert base_context.state.incumbent_score == 0.85
