"""
Phase 4 Integration Tests

Verifies that the BaselineHub successfully generates ExperimentSpecs 
that can be compiled by the PipelineCompiler via the ComponentRegistry.
"""

import pytest
from sklearn.pipeline import Pipeline

from automl.models.baseline_hub import BaselineHub
from automl.core.schemas import TaskProfile
from automl.core.enums import TaskType, DatasetStructure
from automl.compiler.compiler import PipelineCompiler


def test_baseline_hub_to_compiler_integration_classification():
    hub = BaselineHub()
    task = TaskProfile(task_type=TaskType.BINARY_CLASSIFICATION)
    specs = hub.generate_baselines(task)
    
    assert len(specs) == 5  # Dummy, Linear, RF, ET, HistGB
    
    compiler = PipelineCompiler()
    
    for spec in specs:
        # Pass task_type so CompatibilityChecker validates task compatibility
        pipeline = compiler.compile(spec, task_type=TaskType.BINARY_CLASSIFICATION)
        assert isinstance(pipeline, Pipeline)
        assert pipeline.steps[-1][0] == "estimator"


def test_baseline_hub_to_compiler_integration_regression():
    hub = BaselineHub()
    task = TaskProfile(task_type=TaskType.REGRESSION)
    specs = hub.generate_baselines(task)
    
    assert len(specs) == 5  # Dummy, Linear, RF, ET, HistGB
    
    compiler = PipelineCompiler()
    
    for spec in specs:
        pipeline = compiler.compile(spec, task_type=TaskType.REGRESSION)
        assert isinstance(pipeline, Pipeline)
        assert pipeline.steps[-1][0] == "estimator"
