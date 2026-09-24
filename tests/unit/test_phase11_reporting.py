"""
Unit tests for Phase 11: Reporting Subsystem
"""

import numpy as np
import pytest
from rich.table import Table

from automl.core.enums import OperatorType, SaturationState
from automl.core.schemas import ExperimentResult, SearchState
from automl.reporting.leaderboard import Leaderboard
from automl.reporting.plots import PlotGenerator
from automl.reporting.report import ReportGenerator


def test_leaderboard_generation():
    lb = Leaderboard()
    
    results = [
        ExperimentResult(
            experiment_id="EXP_1", status="success", cv_score_mean=0.85, cv_score_std=0.01,
            operator=OperatorType.BASELINE
        ),
        ExperimentResult(
            experiment_id="EXP_2", status="success", cv_score_mean=0.90, cv_score_std=0.02,
            operator=OperatorType.HPO
        ),
        ExperimentResult(
            experiment_id="EXP_3", status="failed", error_message="Boom"
        )
    ]
    
    table = lb.generate_table(results)
    
    assert isinstance(table, Table)
    assert len(table.columns) == 6
    # 2 successful rows
    assert len(table.rows) == 2
    # EXP_2 should be first because 0.90 > 0.85
    assert "EXP_2" in str(table.columns[1]._cells[0])


def test_plots_generation():
    results = [
        ExperimentResult(experiment_id="EXP_1", status="success", cv_score_mean=0.80, runtime_seconds=5, peak_ram_gb=1.0),
        ExperimentResult(experiment_id="EXP_2", status="success", cv_score_mean=0.85, runtime_seconds=10, peak_ram_gb=1.5),
        ExperimentResult(experiment_id="EXP_3", status="success", cv_score_mean=0.82, runtime_seconds=7, peak_ram_gb=1.2),
    ]
    
    lc_b64 = PlotGenerator.plot_learning_curve(results)
    assert lc_b64 is not None
    assert lc_b64.startswith("iVBORw0K") # Typical PNG base64 header
    
    ru_b64 = PlotGenerator.plot_resource_usage(results)
    assert ru_b64 is not None
    assert ru_b64.startswith("iVBORw0K")
    
    corr_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    div_b64 = PlotGenerator.plot_diversity_heatmap(corr_matrix, ["A", "B"])
    assert div_b64 is not None


def test_report_generation():
    generator = ReportGenerator()
    
    state = SearchState(run_id="RUN_1", saturation_state=SaturationState.SATURATED)
    results = [
        ExperimentResult(
            experiment_id="EXP_1", status="success", cv_score_mean=0.95,
            operator=OperatorType.BASELINE
        )
    ]
    selection = {"type": "single", "selection_id": "EXP_1"}
    
    html = generator.generate_html(state, results, selection, final_test_score=0.92)
    
    assert "<html>" in html
    assert "MicroAutoML Run Report: RUN_1" in html
    assert "0.9200" in html # Final test score
    assert "EXP_1" in html
