"""
Unit tests for Phase 9: Experiment Memory & Meta-Learning
"""

import os
import sqlite3
import pytest
from pathlib import Path

from automl.core.enums import FailureType
from automl.core.schemas import (
    DatasetFingerprint, 
    ExperimentResult, 
    ExperimentSpec, 
    FailureReport, 
    SearchState
)
from automl.memory.database import DatabaseManager
from automl.memory.experiments import ExperimentStore
from automl.memory.meta_prior import MetaPrior


@pytest.fixture
def db_manager(tmp_path):
    """Provides a fresh, temporary SQLite database."""
    db_file = tmp_path / "test_microautoml.db"
    manager = DatabaseManager(db_path=db_file)
    yield manager
    # Clean up (usually handled by tmp_path, but explicitly close)
    # The context manager in DatabaseManager handles closing per query.
    if db_file.exists():
        try:
            os.remove(db_file)
        except PermissionError:
            pass # Windows might keep the WAL open


def test_database_initialization(db_manager):
    """Test that the schema is created correctly."""
    with db_manager.get_connection() as conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        table_names = {t["name"] for t in tables}
        
        assert "datasets" in table_names
        assert "experiments" in table_names
        assert "failures" in table_names
        assert "checkpoints" in table_names
        assert "lessons" in table_names


def test_experiment_store(db_manager):
    store = ExperimentStore(db_manager)
    
    run_id = "RUN_001"
    dataset_id = "DS_001"
    
    # 1. Dataset Fingerprint
    fingerprint = DatasetFingerprint(n_samples=1000, n_features=10)
    store.save_dataset_fingerprint(dataset_id, fingerprint)
    
    # 2. Experiment Spec
    spec = ExperimentSpec(experiment_id="EXP_001", model_name="Dummy")
    store.save_experiment_spec(run_id, dataset_id, spec)
    
    # 3. Experiment Result
    result = ExperimentResult(experiment_id="EXP_001", status="success", cv_score_mean=0.95)
    store.save_experiment_result(result)
    
    # 4. Failure Report
    report = FailureReport(experiment_id="EXP_002", failure_type=FailureType.MEMORY)
    store.save_failure_report(run_id, report)
    
    # 5. Checkpoints (SearchState)
    state = SearchState(run_id=run_id)
    state.completed_experiments.append("EXP_001")
    store.save_checkpoint(dataset_id, state)
    
    loaded_state = store.load_checkpoint(run_id)
    assert loaded_state is not None
    assert loaded_state.run_id == run_id
    assert "EXP_001" in loaded_state.completed_experiments


def test_meta_prior(db_manager):
    prior = MetaPrior(db_manager)
    
    fingerprint = DatasetFingerprint(n_samples=5000, n_features=20)
    hypothesis = "Use TargetEncoding for categorical variables."
    
    # Initially neutral
    assert prior.get_prior(fingerprint, hypothesis) == 0.5
    
    # Record 3 successes and 1 failure
    prior.record_outcome(fingerprint, hypothesis, is_improvement=True)
    prior.record_outcome(fingerprint, hypothesis, is_improvement=True)
    prior.record_outcome(fingerprint, hypothesis, is_improvement=False)
    prior.record_outcome(fingerprint, hypothesis, is_improvement=True)
    
    # Prior should be 0.75 (3/4)
    assert prior.get_prior(fingerprint, hypothesis) == 0.75
    
    # Unseen hypothesis should still be neutral
    assert prior.get_prior(fingerprint, "Some other hypothesis") == 0.5
