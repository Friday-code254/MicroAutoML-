"""
MicroAutoML-Agent — Baseline Hub

Generates standardized baseline experiments to establish performance floors
before the autonomous search begins.

Baseline portfolio (5 models — all zero optional dependencies):
  1. Dummy           — absolute floor (majority class / mean)
  2. Linear          — LogisticRegression / Ridge
  3. RandomForest    — shallow RF (max_depth=5, 50 trees)
  4. ExtraTrees      — shallow ET (max_depth=5, 50 trees) — often better than RF at no extra cost
  5. HistGradientBoosting — modern gradient boosting, always available in sklearn

CatBoost / LightGBM / XGBoost are registered in the ComponentRegistry but NOT baselined
here — they are optional dependencies and will be routed by Phase 7's conditional model
selector when the environment has them installed.
"""

from __future__ import annotations

from automl.core.enums import Fidelity, OperatorType, TaskType
from automl.core.schemas import ExperimentSpec, TaskProfile


class BaselineHub:
    """
    Generates ExperimentSpec objects for the standard 5-model baseline portfolio.
    All models are from sklearn — zero optional dependencies.
    """

    def generate_baselines(self, task: TaskProfile) -> list[ExperimentSpec]:
        """Generate baseline experiments for the given task."""
        is_classification = task.task_type in (
            TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION
        )
        specs: list[ExperimentSpec] = []

        # 1. Dummy — absolute floor (CHEAP fidelity to ensure fair comparison with other baselines)
        specs.append(ExperimentSpec(
            model_name="DummyClassifier" if is_classification else "DummyRegressor",
            operator=OperatorType.BASELINE,
            parent_id=None,
            fidelity=Fidelity.CHEAP,
            hyperparameters={},
        ))

        # 2. Linear — LogisticRegression / Ridge (CHEAP)
        specs.append(ExperimentSpec(
            model_name="LogisticRegression" if is_classification else "Ridge",
            operator=OperatorType.BASELINE,
            parent_id=None,
            fidelity=Fidelity.CHEAP,
            hyperparameters={},
        ))

        # 3. RandomForest — shallow default (CHEAP)
        specs.append(ExperimentSpec(
            model_name="RandomForestClassifier" if is_classification else "RandomForestRegressor",
            operator=OperatorType.BASELINE,
            parent_id=None,
            fidelity=Fidelity.CHEAP,
            hyperparameters={"max_depth": 5, "n_estimators": 50},
        ))

        # 4. ExtraTrees — often stronger than RF at same cost (CHEAP)
        specs.append(ExperimentSpec(
            model_name="ExtraTreesClassifier" if is_classification else "ExtraTreesRegressor",
            operator=OperatorType.BASELINE,
            parent_id=None,
            fidelity=Fidelity.CHEAP,
            hyperparameters={"max_depth": 5, "n_estimators": 50},
        ))

        # 5. HistGradientBoosting — modern sklearn GBM, handles missing natively (CHEAP)
        specs.append(ExperimentSpec(
            model_name=(
                "HistGradientBoostingClassifier" if is_classification
                else "HistGradientBoostingRegressor"
            ),
            operator=OperatorType.BASELINE,
            parent_id=None,
            fidelity=Fidelity.CHEAP,
            hyperparameters={},
        ))

        return specs

