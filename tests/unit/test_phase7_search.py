"""
Unit tests for Phase 7: Autonomous Search Controller
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from automl.core.enums import BranchStatus, Fidelity, SaturationState, OperatorType
from automl.core.schemas import (
    ExperimentSpec,
    ExperimentResult,
    ResourceUsageRecord,
    SearchNode,
    SearchState,
    MetricDirection
)
from automl.search.tree import ExperimentTree
from automl.search.strategies import SearchStrategist
from automl.search.saturation import SaturationDetector
from automl.search.controller import SearchController


@pytest.fixture
def empty_tree():
    return ExperimentTree()


def test_tree_insertion_and_lineage(empty_tree):
    # Root node
    spec1 = ExperimentSpec(
        experiment_id="EXP_01",
        parent_id=None,
        operator=OperatorType.BASELINE,
        fidelity=Fidelity.CHEAP
    )
    node1 = empty_tree.insert_node(spec1)
    
    assert node1.depth == 0
    assert "EXP_01" in empty_tree.root_nodes
    
    # Child node
    spec2 = ExperimentSpec(
        experiment_id="EXP_02",
        parent_id="EXP_01",
        operator=OperatorType.HPO,
        fidelity=Fidelity.CHEAP
    )
    node2 = empty_tree.insert_node(spec2)
    
    assert node2.depth == 1
    assert "EXP_02" in empty_tree.nodes["EXP_01"].children
    
    # Branch history
    history = empty_tree.get_branch_history("EXP_02")
    assert [n.experiment_id for n in history] == ["EXP_01", "EXP_02"]


def test_tree_result_update_and_incumbent(empty_tree):
    spec1 = ExperimentSpec(experiment_id="EXP_01", fidelity=Fidelity.FULL)
    empty_tree.insert_node(spec1)
    
    result1 = ExperimentResult(
        experiment_id="EXP_01",
        status="success",
        cv_score_mean=0.85,
        cv_score_std=0.01,
        resource_usage=ResourceUsageRecord(runtime_seconds=10.0, peak_ram_gb=1.0)
    )
    empty_tree.update_node_result(result1)
    
    incumbent = empty_tree.get_incumbent()
    assert incumbent is not None
    assert incumbent.experiment_id == "EXP_01"
    assert incumbent.score == 0.85


def test_strategist_pruning(empty_tree):
    # Setup: Incumbent is 0.90
    spec_inc = ExperimentSpec(experiment_id="EXP_INC", fidelity=Fidelity.FULL)
    empty_tree.insert_node(spec_inc)
    empty_tree.update_node_result(ExperimentResult(experiment_id="EXP_INC", status="success", cv_score_mean=0.90))
    
    # Setup: New node is 0.80 (more than 5% worse than 0.90, threshold is 0.855)
    spec_bad = ExperimentSpec(experiment_id="EXP_BAD", parent_id="EXP_INC", fidelity=Fidelity.CHEAP)
    empty_tree.insert_node(spec_bad)
    empty_tree.update_node_result(ExperimentResult(experiment_id="EXP_BAD", status="success", cv_score_mean=0.80))
    
    strategist = SearchStrategist(prune_threshold_pct=0.05, metric_direction=MetricDirection.HIGHER_IS_BETTER)
    strategist.evaluate_tree(empty_tree)
    
    assert empty_tree.nodes["EXP_BAD"].status == BranchStatus.PRUNED


def test_strategist_promotion(empty_tree):
    # Setup: Incumbent is 0.90
    spec_inc = ExperimentSpec(experiment_id="EXP_INC", fidelity=Fidelity.FULL)
    empty_tree.insert_node(spec_inc)
    empty_tree.update_node_result(ExperimentResult(experiment_id="EXP_INC", status="success", cv_score_mean=0.90))
    
    # Setup: New node is 0.89 (within 2% of 0.90, threshold is 0.882)
    spec_promising = ExperimentSpec(experiment_id="EXP_PROM", fidelity=Fidelity.CHEAP)
    empty_tree.insert_node(spec_promising)
    empty_tree.update_node_result(ExperimentResult(experiment_id="EXP_PROM", status="success", cv_score_mean=0.89))
    
    strategist = SearchStrategist(promotion_threshold_pct=0.02, metric_direction=MetricDirection.HIGHER_IS_BETTER)
    strategist.evaluate_tree(empty_tree)
    
    assert empty_tree.nodes["EXP_PROM"].status == BranchStatus.PROMISING
    
    promising = empty_tree.get_promising_candidates(Fidelity.CHEAP)
    assert len(promising) == 1
    assert promising[0].experiment_id == "EXP_PROM"


def test_saturation_detector_plateau():
    state = SearchState(run_id="RUN_1")
    state.consecutive_no_gain = 5  # Reached patience
    state.budget_total_seconds = 1000
    state.budget_elapsed_seconds = 100
    
    tree = ExperimentTree()
    
    detector = SaturationDetector(plateau_patience=5)
    
    new_state = detector.check_saturation(state, tree)
    assert new_state == SaturationState.PLATEAU_DETECTED
    
    state.consecutive_no_gain = 10  # Double patience
    new_state = detector.check_saturation(state, tree)
    assert new_state == SaturationState.SATURATED


def test_saturation_detector_budget():
    state = SearchState(run_id="RUN_1")
    state.consecutive_no_gain = 0
    state.budget_total_seconds = 1000
    state.budget_elapsed_seconds = 950  # 5% remaining (warning threshold is 10%)
    
    tree = ExperimentTree()
    detector = SaturationDetector(budget_warning_pct=0.10)
    
    new_state = detector.check_saturation(state, tree)
    assert new_state == SaturationState.PLATEAU_DETECTED
    
    state.budget_elapsed_seconds = 1000  # 0% remaining
    new_state = detector.check_saturation(state, tree)
    assert new_state == SaturationState.BUDGET_EXHAUSTED
