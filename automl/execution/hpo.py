"""
MicroAutoML-Agent — HPO Engine (Optuna)
Runs a Hyperparameter Optimization study on a given ExperimentSpec.
"""
import logging
from typing import Any
import pandas as pd
import numpy as np

import optuna
from optuna.samplers import TPESampler

from automl.core.schemas import ExperimentSpec
from automl.core.enums import MetricDirection
from automl.registry.model_specs import ALL_MODELS, HyperparameterSpace

logger = logging.getLogger(__name__)

class OptunaOptimizer:
    def __init__(self, n_trials: int = 10, timeout: float = 300.0, metric_direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER):
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric_direction = metric_direction
        
        # Suppress Optuna logging spam
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optimize(
        self, 
        spec: ExperimentSpec, 
        objective_fn: callable
    ) -> dict[str, Any]:
        """
        Runs an Optuna study to find the best hyperparameters.
        `objective_fn` should accept a dict of hyperparameters and return a float score.
        """
        model_spec = next((m for m in ALL_MODELS if m.name == spec.model_name), None)
        if not model_spec or not model_spec.search_space:
            logger.warning(f"No search space defined for {spec.model_name}. Skipping HPO.")
            return spec.hyperparameters
            
        direction = "maximize" if self.metric_direction == MetricDirection.HIGHER_IS_BETTER else "minimize"
        study = optuna.create_study(direction=direction, sampler=TPESampler(seed=42))
        
        def _optuna_objective(trial: optuna.Trial) -> float:
            hp_dict = {}
            for param in model_spec.search_space:
                if param.type == "int":
                    hp_dict[param.name] = trial.suggest_int(param.name, int(param.low), int(param.high), log=param.log)
                elif param.type == "float":
                    hp_dict[param.name] = trial.suggest_float(param.name, param.low, param.high, log=param.log)
                elif param.type == "categorical":
                    hp_dict[param.name] = trial.suggest_categorical(param.name, param.choices)
                    
            # Merge with existing non-search space params if any
            final_hps = {**spec.hyperparameters, **hp_dict}
            return objective_fn(final_hps)
            
        logger.info(f"Starting Optuna study for {spec.model_name} ({self.n_trials} trials)")
        study.optimize(_optuna_objective, n_trials=self.n_trials, timeout=self.timeout)
        
        logger.info(f"HPO completed. Best score: {study.best_value:.4f}")
        return {**spec.hyperparameters, **study.best_params}
