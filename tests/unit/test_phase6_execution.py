"""
Phase 6 Tests — Multi-Fidelity Runner & Resource Governor

Tests fidelity constraints, resource budget logic, timeout wrappers,
and early stopping injections.
"""

from __future__ import annotations

import time
import pytest
import numpy as np
import pandas as pd

from automl.core.enums import Fidelity, ResourceDecision, TaskType, OperatorType
from automl.core.schemas import ExperimentSpec, TaskProfile, EvidencePack, DatasetFingerprint
from automl.core.config import RunConfig
from automl.execution.resource import ResourceGovernor
from automl.execution.fidelity import FidelityManager
from automl.execution.timeout import run_with_guards, TimeoutException
from automl.execution.early_stop import EarlyStoppingManager
from automl.execution.runner import ExperimentRunner


def test_resource_governor() -> None:
    gov = ResourceGovernor(max_memory_mb=100.0, max_threads=4)
    spec = ExperimentSpec(
        parent_id="EXP_BASE",
        operator=OperatorType.HPO,
        model_name="RandomForestClassifier"
    )
    
    # 1000 rows, 10 cols, RF (multiplier=8) => 1000 * 10 * 8 bytes = 0.076 MB * 8 = ~0.6MB
    # Should easily pass
    prof = gov.evaluate(spec, num_rows=1000, num_cols=10)
    assert prof.decision == ResourceDecision.ALLOWED
    assert prof.recommended_threads == 4
    
    # 10M rows, 100 cols => 10M * 100 * 8 bytes = 8000 MB * 8 = 64000 MB
    # Should definitely fail 100MB limit
    prof_huge = gov.evaluate(spec, num_rows=10000000, num_cols=100)
    assert prof_huge.decision == ResourceDecision.DENIED_MEMORY


def test_fidelity_manager() -> None:
    mgr = FidelityManager()
    
    smoke = mgr.get_constraints(Fidelity.SMOKE)
    assert smoke.cv_folds == 1
    assert getattr(smoke, "use_holdout", False) is True
    assert smoke.sample_fraction == 0.05
    
    full = mgr.get_constraints(Fidelity.FULL)
    assert full.cv_folds == 5
    assert getattr(full, "use_holdout", False) is False
    assert full.sample_fraction == 1.0


def _fast_func() -> str:
    return "done"

def _slow_func() -> None:
    import time
    time.sleep(5.0)

def test_timeout() -> None:
    res = run_with_guards(_fast_func, timeout_seconds=5.0)
    assert res == "done"
    
    with pytest.raises(TimeoutException):
        run_with_guards(_slow_func, timeout_seconds=1.0)


def test_early_stopping_manager() -> None:
    mgr = EarlyStoppingManager()
    X_val, y_val = [1, 2], [0, 1]

    # get_fit_params now always returns {} — eval_set injection was removed (pipeline safety fix)
    spec_xgb = ExperimentSpec(parent_id="EXP_BASE", operator=OperatorType.HPO, model_name="XGBClassifier")
    params_xgb = mgr.get_fit_params(spec_xgb, X_val, y_val)
    assert params_xgb == {}  # eval_set injection removed — see early_stop.py for rationale

    # RF also returns empty (no early stopping support)
    spec_rf = ExperimentSpec(parent_id="EXP_BASE", operator=OperatorType.HPO, model_name="RandomForestClassifier")
    params_rf = mgr.get_fit_params(spec_rf, X_val, y_val)
    assert params_rf == {}

    # HistGB supports safe constructor-level early stopping (internal validation fraction)
    spec_hgb = ExperimentSpec(parent_id="EXP_BASE", operator=OperatorType.HPO, model_name="HistGradientBoostingClassifier")
    overrides = mgr.get_constructor_overrides(spec_hgb)
    assert "validation_fraction" in overrides
    assert "n_iter_no_change" in overrides



def test_experiment_runner() -> None:
    # A lightweight test for the runner to ensure it connects components properly
    # Using DummyClassifier to avoid heavy training
    run_config = RunConfig.development()
    runner = ExperimentRunner(run_config=run_config)  # Bug fix: RunConfig threaded in
    
    spec = ExperimentSpec(
        parent_id="EXP_BASE",
        operator=OperatorType.BASELINE,
        model_name="DummyClassifier",
        fidelity=Fidelity.SMOKE
    )
    
    # Create simple data
    X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
    y = pd.Series(np.random.randint(0, 2, size=100))
    
    evidence = EvidencePack()
    evidence.task = TaskProfile(task_type=TaskType.BINARY_CLASSIFICATION)
    evidence.fingerprint = DatasetFingerprint(task_type="binary_classification")
    
    result = runner.run(spec, X, y, evidence)
    
    assert result.status.value == "success"
    assert result.cv_score_mean is not None
    # Bug fixes verified:
    assert result.experiment_id == spec.experiment_id  # Not "EXP_UNKNOWN"
    assert result.experiment_id != "EXP_UNKNOWN"
    assert result.operator == OperatorType.BASELINE  # operator is preserved
