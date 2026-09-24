"""
MicroAutoML-Agent — Pipeline Compiler

Maps a declarative ExperimentSpec into an executable scikit-learn Pipeline.
Uses the ComponentRegistry to dynamically instantiate objects.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

from automl.core.enums import TaskType
from automl.core.schemas import ExperimentSpec
from automl.registry.registry import ComponentRegistry
from automl.compiler.compatibility import CompatibilityChecker

logger = logging.getLogger(__name__)


class PipelineCompiler:
    """
    Compiles an ExperimentSpec JSON-like configuration into a fully executable
    sklearn Pipeline object.
    """
    
    def __init__(self, registry: ComponentRegistry | None = None, random_seed: int = 42) -> None:
        self.registry = registry or ComponentRegistry()
        self.checker = CompatibilityChecker(self.registry)
        self.random_seed = random_seed

    def compile(self, spec: ExperimentSpec, task_type: TaskType | None = None) -> Pipeline:
        """
        Builds the pipeline sequence:
        1. Compatibility check (fast, zero-cost guard)
        2. Feature Generators (Phase 3 transforms)
        3. Preprocessing (Imputers, Encoders, Scalers)
        4. Feature Selection
        5. Estimator
        """
        # 0. Compatibility gate — raises ValueError immediately on invalid specs
        self.checker.assert_compatible(spec, task_type)
        
        steps = []
        
        # 1. Feature Generators
        if spec.features:
            generator_steps = []
            for feat_name in spec.features:
                try:
                    transformer = self.registry.build_transform(feat_name)
                    generator_steps.append((feat_name, transformer))
                except Exception as e:
                    raise ValueError(f"Could not build feature generator '{feat_name}': {e}") from e
                    
            if generator_steps:
                for step_name, transformer in generator_steps:
                    # 3.5b & 3.5c: Target only numerical columns, and keep remainder
                    ct = ColumnTransformer(
                        transformers=[(step_name, transformer, make_column_selector(dtype_include=np.number))],
                        remainder="passthrough",
                        verbose_feature_names_out=False
                    )
                    try:
                        ct.set_output(transform="pandas")
                    except Exception as e:
                        logger.debug(f"Could not set_output to pandas: {e}")
                    steps.append((f"feat_{step_name}", ct))
                    
        # 2. Manual Preprocessing
        if spec.preprocessing:
            for i, p_config in enumerate(spec.preprocessing):
                name = p_config.get("name")
                params = p_config.get("params", {})
                columns = p_config.get("columns", None)
                
                if not name:
                    continue
                    
                try:
                    transformer = self.registry.build_transform(name, params)
                    
                    if columns is not None:
                        # Wrap in a ColumnTransformer to apply only to specific columns
                        # remainder="passthrough" ensures other columns are not dropped
                        step_name = f"prep_{name}_{i}"
                        col_transformer = ColumnTransformer(
                            transformers=[(name, transformer, columns)],
                            remainder="passthrough",
                            verbose_feature_names_out=False
                        )
                        try:
                            col_transformer.set_output(transform="pandas")
                        except Exception:
                            pass
                        steps.append((step_name, col_transformer))
                    else:
                        steps.append((f"prep_{name}_{i}", transformer))
                        
                except Exception as e:
                    raise ValueError(f"Could not build preprocessor '{name}': {e}") from e

        # 3. Base Preprocessing (Auto-Injected)
        # Prevents mixed-type string crashes by ensuring all features are numeric and imputed
        numeric_transformer = SimpleImputer(strategy="median")
        categorical_transformer = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="missing"),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        )
        base_preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, make_column_selector(dtype_exclude=["object", "category", "string", "datetime64", "datetime64[ns]", "datetime64[ns, UTC]"])),
                ("cat", categorical_transformer, make_column_selector(dtype_include=["object", "category", "string"]))
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )
        try:
            base_preprocessor.set_output(transform="pandas")
        except Exception:
            pass
        steps.append(("base_preprocessor", base_preprocessor))

        # 4. Feature Selection

        if spec.feature_selection:
            name = spec.feature_selection.get("name")
            params = spec.feature_selection.get("params", {})
            if name:
                try:
                    selector = self.registry.build_transform(name, params)
                    steps.append(("feature_selection", selector))
                except Exception as e:
                    raise ValueError(f"Could not build selector '{name}': {e}") from e

        # 4. Estimator
        try:
            estimator = self.registry.build_model(
                spec.model_name, 
                spec.hyperparameters,
                random_seed=self.random_seed
            )
            steps.append(("estimator", estimator))
        except Exception as e:
            raise RuntimeError(f"Failed to build estimator '{spec.model_name}': {e}") from e

        if not steps:
            raise ValueError("Pipeline has no steps (missing estimator).")

        return Pipeline(steps)
