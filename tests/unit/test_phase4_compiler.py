"""
Phase 4 Tests — Component Registry & Pipeline Compiler

Tests the dynamic instantiation of models/transformers and the end-to- natural
compilation of an ExperimentSpec into a scikit-learn Pipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from automl.core.schemas import ExperimentSpec
from automl.registry.registry import ComponentRegistry
from automl.compiler.compiler import PipelineCompiler
from automl.core.enums import OperatorType


class TestComponentRegistry:
    @pytest.fixture
    def registry(self) -> ComponentRegistry:
        return ComponentRegistry()

    def test_registry_loads_models(self, registry: ComponentRegistry) -> None:
        spec = registry.get_model_spec("RandomForestClassifier")
        assert spec.name == "RandomForestClassifier"
        assert spec.class_path == "sklearn.ensemble.RandomForestClassifier"
        
    def test_registry_loads_transforms(self, registry: ComponentRegistry) -> None:
        spec = registry.get_transform_spec("StandardScaler")
        assert spec.name == "StandardScaler"
        
    def test_build_model_default_params(self, registry: ComponentRegistry) -> None:
        model = registry.build_model("RandomForestClassifier")
        assert isinstance(model, RandomForestClassifier)
        assert model.random_state == 42  # From default_params
        
    def test_build_model_override_params(self, registry: ComponentRegistry) -> None:
        model = registry.build_model("RandomForestClassifier", {"n_estimators": 50})
        assert model.n_estimators == 50
        assert model.random_state == 42  # default preserved


class TestPipelineCompiler:
    @pytest.fixture
    def compiler(self) -> PipelineCompiler:
        return PipelineCompiler()

    def test_compile_minimal_spec(self, compiler: PipelineCompiler) -> None:
        spec = ExperimentSpec(
            operator=OperatorType.BASELINE,
            model_name="DummyClassifier"
        )
        pipeline = compiler.compile(spec)
        
        # Base preprocessor + Estimator
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "base_preprocessor"
        assert pipeline.steps[1][0] == "estimator"
        
    def test_compile_full_spec(self, compiler: PipelineCompiler) -> None:
        spec = ExperimentSpec(
            operator=OperatorType.FEATURE_ENGINEERING,
            parent_id="EXP_123",
            model_name="RandomForestClassifier",
            features=["Log1pTransformer"],
            preprocessing=[
                {"name": "StandardScaler", "columns": ["age", "income"]},
                {"name": "SimpleImputer", "params": {"strategy": "mean"}}
            ],
            feature_selection={"name": "DropFeatures", "params": {"features_to_drop": ["id"]}},
            hyperparameters={"n_estimators": 200}
        )
        pipeline = compiler.compile(spec)
        
        # Steps:
        # 1. feat_Log1pTransformer
        # 2. base_preprocessor
        # 3. prep_StandardScaler_0 (ColumnTransformer)
        # 4. prep_SimpleImputer_1 (SimpleImputer)
        # 5. feature_selection (DropFeatures)
        # 6. estimator (RandomForestClassifier)
        assert len(pipeline.steps) == 6
        
        assert pipeline.steps[0][0] == "feat_Log1pTransformer"
        assert pipeline.steps[1][0] == "base_preprocessor"
        
        assert pipeline.steps[2][0] == "prep_StandardScaler_0"
        from sklearn.compose import ColumnTransformer
        assert isinstance(pipeline.steps[2][1], ColumnTransformer)
        
        assert pipeline.steps[3][0] == "prep_SimpleImputer_1"
        assert pipeline.steps[4][0] == "feature_selection"
        assert pipeline.steps[5][0] == "estimator"
        
        rf = pipeline.steps[5][1]
        assert rf.n_estimators == 200
        assert rf.random_state == 42
        
    def test_compile_fails_on_missing_model(self, compiler: PipelineCompiler) -> None:
        spec = ExperimentSpec(
            operator=OperatorType.BASELINE,
            model_name="NonExistentModel"
        )
        # CompatibilityChecker fires first (before compiler tries to build the estimator)
        with pytest.raises((ValueError, RuntimeError)):
            compiler.compile(spec)
