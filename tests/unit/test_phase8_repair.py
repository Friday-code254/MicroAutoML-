"""
Unit tests for Phase 8: Failure Classifier & Repair Engine
"""

import pytest

from automl.core.enums import FailureType
from automl.core.schemas import ExperimentResult, ExperimentSpec
from automl.repair.classifier import FailureClassifier
from automl.repair.repair import RepairEngine


def test_failure_classifier():
    classifier = FailureClassifier()
    
    # Exact exception match
    assert classifier.classify("MemoryError", "", "") == FailureType.MEMORY
    assert classifier.classify("TimeoutError", "", "") == FailureType.TIMEOUT
    
    # Regex match on exception type / message
    assert classifier.classify("ValueError", "Found array with dim 3. Expected <= 2", "") == FailureType.SHAPE
    assert classifier.classify("ValueError", "could not convert string to float: 'Male'", "") == FailureType.TYPE
    assert classifier.classify("ValueError", "Input contains NaN, infinity or a value too large", "") == FailureType.NUMERICAL
    
    # Unknown
    assert classifier.classify("RuntimeError", "Something weird happened", "") == FailureType.UNKNOWN


def test_repair_engine_timeout():
    engine = RepairEngine()
    
    spec = ExperimentSpec(
        experiment_id="EXP_1",
        model_name="RandomForestClassifier",
        hyperparameters={"n_estimators": 200, "max_depth": 10},
        hypothesis="Original"
    )
    
    result = ExperimentResult(
        experiment_id="EXP_1",
        status="failed",
        error_message="took too long",
        failure_type=None  # Force classifier to run
    )
    
    repaired_spec, report = engine.attempt_repair(spec, result)
    
    assert report.failure_type == FailureType.TIMEOUT
    assert report.repair_successful is True
    assert repaired_spec is not None
    assert repaired_spec.experiment_id != "EXP_1"
    assert repaired_spec.hyperparameters["n_estimators"] == 100  # Halved


def test_repair_engine_max_retries():
    engine = RepairEngine(max_retries=2)
    
    spec = ExperimentSpec(experiment_id="EXP_1", repair_count=2)
    result = ExperimentResult(
        experiment_id="EXP_1",
        status="failed",
        failure_type=FailureType.MEMORY
    )
    
    repaired_spec, report = engine.attempt_repair(spec, result)
    
    assert repaired_spec is None
    assert report.repair_applied == "Max retries reached."
