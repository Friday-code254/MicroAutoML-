"""
MicroAutoML-Agent — Component Registry

Singleton registry holding available models and transformers.
Provides dynamic importing and lookup capabilities.
"""

from __future__ import annotations

import importlib
from typing import Any

from automl.core.enums import TaskType
from automl.registry.model_specs import ALL_MODELS, ModelSpec
from automl.registry.transform_specs import ALL_TRANSFORMS, TransformSpec


class ComponentRegistry:
    """
    Central registry for models and transformers.
    Dynamically imports classes from their paths to avoid heavy initialization times
    and hard dependencies if certain packages (like xgboost/lightgbm) are missing.
    """
    
    _instance: ComponentRegistry | None = None
    
    def __new__(cls) -> ComponentRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self.models: dict[str, ModelSpec] = {m.name: m for m in ALL_MODELS}
        self.transforms: dict[str, TransformSpec] = {t.name: t for t in ALL_TRANSFORMS}

    def get_model_spec(self, name: str) -> ModelSpec | None:
        """Returns the ModelSpec for the given name, or None if not registered."""
        return self.models.get(name)

    def require_model_spec(self, name: str) -> ModelSpec:
        """Returns the ModelSpec or raises ValueError if not registered."""
        spec = self.models.get(name)
        if spec is None:
            raise ValueError(f"Model '{name}' not found in registry.")
        return spec

    def list_model_names(self) -> list[str]:
        """Returns all registered model names."""
        return list(self.models.keys())

    def get_transform_spec(self, name: str) -> TransformSpec:
        if name not in self.transforms:
            raise ValueError(f"Transform '{name}' not found in registry.")
        return self.transforms[name]

    def get_models_for_task(self, task: TaskType) -> list[ModelSpec]:
        """Returns all models that support the given task type."""
        return [m for m in self.models.values() if task in m.supported_tasks]

    def instantiate(self, class_path: str, **kwargs: Any) -> Any:
        """
        Dynamically imports a class from a string path (e.g., 'sklearn.ensemble.RandomForestClassifier')
        and instantiates it with kwargs.
        """
        module_path, class_name = class_path.rsplit(".", 1)
        
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        except ImportError as e:
            raise ImportError(f"Could not import {class_path}. Is the package installed?") from e
        except AttributeError as e:
            raise AttributeError(f"Class {class_name} not found in module {module_path}") from e
            
        return cls(**kwargs)

    def build_model(self, name: str, params: dict[str, Any] | None = None, random_seed: int | None = None) -> Any:
        """Instantiates a model by name, combining default params with provided params."""
        spec = self.require_model_spec(name)
        final_params = spec.default_params.copy()
        if params:
            final_params.update(params)
            
        if random_seed is not None and "random_state" in final_params:
            final_params["random_state"] = random_seed
            
        return self.instantiate(spec.class_path, **final_params)

    def build_transform(self, name: str, params: dict[str, Any] | None = None) -> Any:
        """Instantiates a transformer by name."""
        spec = self.get_transform_spec(name)
        final_params = params.copy() if params else {}
        
        # Enforce pandas compatibility for OneHotEncoder
        if name == "OneHotEncoder":
            final_params["sparse_output"] = False
            
        return self.instantiate(spec.class_path, **final_params)
