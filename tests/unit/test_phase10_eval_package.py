"""
Unit tests for Phase 10: Final Evaluation & Packaging
"""

import os
import shutil
import pickle
import pytest
from pathlib import Path
import numpy as np

from automl.core.enums import SaturationState, TaskType, MetricName
from automl.core.schemas import SearchState, EvidencePack, TaskProfile
from automl.execution.final_eval import FinalHoldoutEvaluator, FinalTestAlreadyUsedError
from automl.execution.packager import ModelPackager

# A mock sklearn-like pipeline
class MockPipeline:
    def predict(self, X):
        return np.ones(len(X))


def test_final_evaluator_once_only():
    evaluator = FinalHoldoutEvaluator()
    
    state = SearchState(run_id="RUN_1", saturation_state=SaturationState.SATURATED)
    selection = {"type": "single", "selection_id": "EXP_1", "weights": {"EXP_1": 1.0}}
    
    X_test = [1, 2, 3]
    y_test = np.array([1, 1, 1])
    
    pipelines = {"EXP_1": MockPipeline()}
    
    evidence = EvidencePack(
        task=TaskProfile(task_type=TaskType.BINARY_CLASSIFICATION),
        primary_metric=MetricName.ACCURACY
    )
    
    # First eval should succeed (accuracy 1.0)
    score = evaluator.evaluate(state, selection, X_test, y_test, pipelines, evidence)
    assert score == 1.0
    
    # Second eval should hard crash
    with pytest.raises(FinalTestAlreadyUsedError):
        evaluator.evaluate(state, selection, X_test, y_test, pipelines, evidence)


def test_model_packager(tmp_path):
    output_dir = tmp_path / "final_model"
    packager = ModelPackager(output_dir=output_dir)
    
    pipeline = MockPipeline()
    metadata = {"seed": 42, "run_id": "RUN_1"}
    schema = {"feature1": "float"}
    
    packager.package(pipeline, metadata, schema)
    
    # Check files exist
    assert (output_dir / "model.pkl").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "feature_schema.json").exists()
    assert (output_dir / "inference.py").exists()
    
    # Check model is loadable
    with open(output_dir / "model.pkl", "rb") as f:
        loaded = pickle.load(f)
        
    assert isinstance(loaded, MockPipeline)
