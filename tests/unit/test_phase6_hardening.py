"""
Phase 6 Hardening Tests

Validates process isolation, memory guards, test data isolation, and holdout splits.
"""
import os
import time
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from automl.core.enums import DatasetStructure, Fidelity, TaskType, OperatorType, ExperimentStatus, FailureType, SplitType
from automl.core.schemas import ExperimentSpec, EvidencePack, TaskProfile, SplitProfile
from automl.execution.runner import ExperimentRunner
from automl.execution.timeout import run_with_guards, TimeoutException
from automl.models.baseline_hub import BaselineHub


# =====================================================================
# Guards & Subprocess Isolation Tests (timeout.py)
# =====================================================================

def _infinite_loop(*args, **kwargs):
    while True:
        time.sleep(0.1)

def test_timeout_kills_training_process():
    """Verify runaway processes are killed."""
    start = time.time()
    with pytest.raises(TimeoutException):
        run_with_guards(_infinite_loop, timeout_seconds=2.0)
    elapsed = time.time() - start
    assert 1.9 < elapsed < 4.0


def _memory_hog(*args, **kwargs):
    # Rapidly allocate memory
    _list = []
    for _ in range(100):
        _list.append(b"A" * 1024 * 1024 * 10)  # 10 MB per iteration
        time.sleep(0.05)

def test_memory_limit_kills_training_process():
    """Verify greedy processes are killed."""
    with pytest.raises(MemoryError):
        run_with_guards(_memory_hog, timeout_seconds=5.0, max_memory_mb=50.0)


# =====================================================================
# Split & Holdout Correctness (runner.py)
# =====================================================================

def test_cheap_fidelity_does_not_use_kfold_1():
    """Verify CHEAP uses holdout splits, avoiding n_splits=1 KFold crash."""
    runner = ExperimentRunner()
    # Mock constraints to CHEAP
    f_constraints = runner.fidelity_mgr.get_constraints(Fidelity.CHEAP)
    assert f_constraints.use_holdout is True

    # Generate dummy data
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.randint(0, 2, 100))
    
    evidence = EvidencePack()
    evidence.task = TaskProfile(task_type=TaskType.BINARY_CLASSIFICATION)
    
    cv, groups = runner._get_cv_splitter(
        evidence, f_constraints, DatasetStructure.IID, X, y, seed=42
    )
    
    # Must not be KFold (which crashes with n_splits=1)
    from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
    assert isinstance(cv, (StratifiedShuffleSplit, ShuffleSplit))
    assert cv.get_n_splits() == 1


def test_group_holdout_respects_groups():
    runner = ExperimentRunner()
    f_constraints = runner.fidelity_mgr.get_constraints(Fidelity.CHEAP)
    
    groups = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    X = pd.DataFrame({"feat": np.random.rand(15), "grp": groups})
    y = pd.Series(np.random.randint(0, 2, 15))
    
    evidence = EvidencePack()
    evidence.task = TaskProfile(task_type=TaskType.BINARY_CLASSIFICATION, group_column="grp")
    
    cv, grp_arr = runner._get_cv_splitter(
        evidence, f_constraints, DatasetStructure.GROUPED, X, y, seed=42
    )
    
    # Should be GroupShuffleSplit
    from sklearn.model_selection import GroupShuffleSplit
    assert isinstance(cv, GroupShuffleSplit)
    assert cv.get_n_splits() == 1
    
    train_idx, val_idx = next(cv.split(X, y, groups=grp_arr))
    
    train_groups = set(grp_arr[train_idx])
    val_groups = set(grp_arr[val_idx])
    assert train_groups.isdisjoint(val_groups)


def test_temporal_holdout_respects_time_order():
    runner = ExperimentRunner()
    f_constraints = runner.fidelity_mgr.get_constraints(Fidelity.CHEAP)
    
    X = pd.DataFrame({"feat": np.random.rand(20)})
    y = pd.Series(np.random.randint(0, 2, 20))
    
    evidence = EvidencePack()
    
    cv, grp_arr = runner._get_cv_splitter(
        evidence, f_constraints, DatasetStructure.TEMPORAL, X, y, seed=42
    )
    
    assert hasattr(cv, "__call__")
    
    splits = list(cv(X, y, groups=None))
    assert len(splits) == 1
    train_idx, val_idx = splits[0]
    
    # 80/20 chronological split
    assert len(train_idx) == 16
    assert len(val_idx) == 4
    assert max(train_idx) < min(val_idx)


# =====================================================================
# Runner Leakage & Injection Guarantees
# =====================================================================

@patch("automl.execution.runner.run_with_guards")
def test_train_val_slice_excludes_test_rows(mock_run):
    """Test data MUST be explicitly sliced out before the runner does anything."""
    mock_run.return_value = {"roc_auc": 0.8}
    
    runner = ExperimentRunner()
    
    # 30 rows: 0-23 are train/val, 24-29 are test
    X_full = pd.DataFrame(np.random.rand(30, 3))
    y_full = pd.Series([0, 1] * 15)
    
    evidence = EvidencePack()
    evidence.task = TaskProfile(task_type=TaskType.BINARY_CLASSIFICATION)
    evidence.split = SplitProfile(
        split_type=SplitType.STRATIFIED_KFOLD,
        n_folds=3,
        test_fraction=0.2,
        test_indices=list(range(24, 30)),
        train_val_indices=list(range(24)),
        test_set_used=False
    )
    
    spec = ExperimentSpec(
        operator=OperatorType.BASELINE,
        model_name="DummyClassifier",
        fidelity=Fidelity.CHEAP
    )
    
    # Run
    runner.run(spec, X_full, y_full, evidence)
    
    # Verify run_with_guards was called with exactly 24 rows in X_train/X_val combined
    # Inspect arguments passed to worker
    assert mock_run.called
    args = mock_run.call_args.kwargs.get("args") or mock_run.call_args.args[1]
    # args: (pipeline, X_train, y_train, X_val, y_val, fit_params, is_class, metric)
    X_train_worker = args[1]
    X_val_worker = args[3]
    y_train_worker = args[2]
    
    # Check that indices don't include 24 or 29
    all_worker_idx = X_train_worker.index.tolist() + X_val_worker.index.tolist()
    assert 24 not in all_worker_idx
    assert 29 not in all_worker_idx
    
    # Check y was sliced identically
    assert set(X_train_worker.index) == set(y_train_worker.index)


def test_y_is_sliced_with_X():
    """Dummy test for explicit tracking; test_train_val_slice_excludes_test_rows tests this."""
    pass


@patch("automl.execution.runner.run_with_guards")
def test_thread_limit_reaches_estimator(mock_run):
    """Verify recommended_threads is pushed into hyperparameters and os.environ."""
    mock_run.return_value = {"roc_auc": 0.8}
    runner = ExperimentRunner()
    
    X = pd.DataFrame(np.random.rand(10, 3))
    y = pd.Series(np.random.randint(0, 2, 10))
    
    evidence = EvidencePack()
    evidence.task = TaskProfile(task_type=TaskType.BINARY_CLASSIFICATION)
    
    spec = ExperimentSpec(
        operator=OperatorType.BASELINE,
        model_name="RandomForestClassifier",
        fidelity=Fidelity.CHEAP
    )
    
    # Force the resource governor to recommend exactly 4 threads
    runner.resource_gov.max_threads = 4
    runner.run(spec, X, y, evidence)
    
    # Hyperparams should have n_jobs=4 injected
    assert spec.hyperparameters.get("n_jobs") == 4
    
    # Environment variables should be set
    assert os.environ.get("OMP_NUM_THREADS") == "4"
    assert os.environ.get("MKL_NUM_THREADS") == "4"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "4"


# =====================================================================
# Full Integration (Baselines x Fidelities)
# =====================================================================

def test_full_baselines_run_at_cheap_fidelity():
    """Integration: Can we run all baselines in CHEAP mode? (This broke previously)"""
    runner = ExperimentRunner()
    hub = BaselineHub()
    
    # Small realistic dataset
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f_{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    
    task = TaskProfile(task_type=TaskType.BINARY_CLASSIFICATION)
    evidence = EvidencePack(task=task)
    
    specs = hub.generate_baselines(task)
    
    for spec in specs:
        spec.fidelity = Fidelity.CHEAP
        res = runner.run(spec, X, y, evidence)
        assert res.status == ExperimentStatus.SUCCESS, f"{spec.model_name} failed: {res.error_traceback}"


def test_full_baselines_run_at_medium_fidelity():
    """Integration: Can we run all baselines in MEDIUM mode?"""
    runner = ExperimentRunner()
    hub = BaselineHub()
    
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f_{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    
    task = TaskProfile(task_type=TaskType.BINARY_CLASSIFICATION)
    evidence = EvidencePack(task=task)
    
    specs = hub.generate_baselines(task)
    
    for spec in specs:
        spec.fidelity = Fidelity.MEDIUM
        res = runner.run(spec, X, y, evidence)
        assert res.status == ExperimentStatus.SUCCESS, f"{spec.model_name} failed: {res.error_traceback}"
