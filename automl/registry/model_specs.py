"""
MicroAutoML-Agent — Model Specifications

Defines the ModelSpec contract and pre-registers standard algorithms.
Includes their parameter search spaces (Optuna-compatible bounds).
"""

from __future__ import annotations

from typing import Any, Callable
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator

from automl.core.enums import ModelFamily, TaskType


class HyperparameterSpace(BaseModel):
    """Defines the search space for a single hyperparameter."""
    name: str
    type: str  # "int", "float", "categorical"
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None
    log: bool = False
    default: Any = None


class ModelSpec(BaseModel):
    """
    Specification for a machine learning model.
    Contains everything needed to instantiate it dynamically without hardcoded imports.
    """
    name: str
    family: ModelFamily
    supported_tasks: list[TaskType]
    
    # We store the fully qualified class string, e.g., "sklearn.ensemble.RandomForestClassifier"
    # The compiler will dynamically import this.
    class_path: str
    
    # Default parameters that should always be passed (e.g., random_state, n_jobs)
    default_params: dict[str, Any] = Field(default_factory=dict)
    
    # Hyperparameter search space
    search_space: list[HyperparameterSpace] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pre-registered Model Specifications
# ---------------------------------------------------------------------------

# 1. Random Forest Classifier
rf_classifier = ModelSpec(
    name="RandomForestClassifier",
    family=ModelFamily.TREE_ENSEMBLE,
    supported_tasks=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
    class_path="sklearn.ensemble.RandomForestClassifier",
    default_params={"random_state": 42},
    search_space=[
        HyperparameterSpace(name="n_estimators", type="int", low=50, high=500, default=100),
        HyperparameterSpace(name="max_depth", type="int", low=3, high=30, default=None),
        HyperparameterSpace(name="min_samples_split", type="int", low=2, high=20, default=2),
        HyperparameterSpace(name="min_samples_leaf", type="int", low=1, high=20, default=1),
        HyperparameterSpace(name="max_features", type="categorical", choices=["sqrt", "log2", None], default="sqrt"),
    ]
)

# 2. Random Forest Regressor
rf_regressor = ModelSpec(
    name="RandomForestRegressor",
    family=ModelFamily.TREE_ENSEMBLE,
    supported_tasks=[TaskType.REGRESSION],
    class_path="sklearn.ensemble.RandomForestRegressor",
    default_params={"random_state": 42},
    search_space=rf_classifier.search_space
)

# 3. HistGradientBoosting Classifier (Native sklearn LightGBM equivalent)
hist_gb_classifier = ModelSpec(
    name="HistGradientBoostingClassifier",
    family=ModelFamily.GRADIENT_BOOSTING,
    supported_tasks=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
    class_path="sklearn.ensemble.HistGradientBoostingClassifier",
    default_params={"random_state": 42},
    search_space=[
        HyperparameterSpace(name="learning_rate", type="float", low=1e-3, high=1.0, log=True, default=0.1),
        HyperparameterSpace(name="max_iter", type="int", low=50, high=1000, default=100),
        HyperparameterSpace(name="max_depth", type="int", low=3, high=20, default=None),
        HyperparameterSpace(name="min_samples_leaf", type="int", low=10, high=100, default=20),
        HyperparameterSpace(name="l2_regularization", type="float", low=1e-8, high=10.0, log=True, default=0.0),
    ]
)

# 4. HistGradientBoosting Regressor
hist_gb_regressor = ModelSpec(
    name="HistGradientBoostingRegressor",
    family=ModelFamily.GRADIENT_BOOSTING,
    supported_tasks=[TaskType.REGRESSION],
    class_path="sklearn.ensemble.HistGradientBoostingRegressor",
    default_params={"random_state": 42},
    search_space=hist_gb_classifier.search_space
)

# 5. Ridge Classifier
ridge_classifier = ModelSpec(
    name="RidgeClassifier",
    family=ModelFamily.LINEAR,
    supported_tasks=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
    class_path="sklearn.linear_model.RidgeClassifier",
    default_params={"random_state": 42},
    search_space=[
        HyperparameterSpace(name="alpha", type="float", low=1e-3, high=1e3, log=True, default=1.0)
    ]
)

# 6. Ridge Regressor
ridge_regressor = ModelSpec(
    name="Ridge",
    family=ModelFamily.LINEAR,
    supported_tasks=[TaskType.REGRESSION],
    class_path="sklearn.linear_model.Ridge",
    default_params={"random_state": 42},
    search_space=[
        HyperparameterSpace(name="alpha", type="float", low=1e-3, high=1e3, log=True, default=1.0)
    ]
)

# 7. ElasticNet Regressor
elasticnet_regressor = ModelSpec(
    name="ElasticNet",
    family=ModelFamily.LINEAR,
    supported_tasks=[TaskType.REGRESSION],
    class_path="sklearn.linear_model.ElasticNet",
    default_params={"random_state": 42},
    search_space=[
        HyperparameterSpace(name="alpha", type="float", low=1e-4, high=1e2, log=True, default=1.0),
        HyperparameterSpace(name="l1_ratio", type="float", low=0.0, high=1.0, default=0.5)
    ]
)

# 8. Dummy Classifier
dummy_classifier = ModelSpec(
    name="DummyClassifier",
    family=ModelFamily.DUMMY,
    supported_tasks=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
    class_path="sklearn.dummy.DummyClassifier",
    default_params={"strategy": "prior"},
    search_space=[]
)

# 9. Dummy Regressor
dummy_regressor = ModelSpec(
    name="DummyRegressor",
    family=ModelFamily.DUMMY,
    supported_tasks=[TaskType.REGRESSION],
    class_path="sklearn.dummy.DummyRegressor",
    default_params={"strategy": "mean"},
    search_space=[]
)

# 10. LightGBM Classifier (Requires pip install lightgbm)
lgbm_classifier = ModelSpec(
    name="LGBMClassifier",
    family=ModelFamily.GRADIENT_BOOSTING,
    supported_tasks=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
    class_path="lightgbm.LGBMClassifier",
    default_params={"random_state": 42, "n_jobs": -1},
    search_space=[
        HyperparameterSpace(name="learning_rate", type="float", low=1e-3, high=1.0, log=True, default=0.1),
        HyperparameterSpace(name="n_estimators", type="int", low=50, high=1000, default=100),
        HyperparameterSpace(name="num_leaves", type="int", low=15, high=255, default=31),
        HyperparameterSpace(name="min_child_samples", type="int", low=5, high=100, default=20),
        HyperparameterSpace(name="reg_alpha", type="float", low=1e-8, high=10.0, log=True, default=0.0),
        HyperparameterSpace(name="reg_lambda", type="float", low=1e-8, high=10.0, log=True, default=0.0),
    ]
)

# 11. LightGBM Regressor
lgbm_regressor = ModelSpec(
    name="LGBMRegressor",
    family=ModelFamily.GRADIENT_BOOSTING,
    supported_tasks=[TaskType.REGRESSION],
    class_path="lightgbm.LGBMRegressor",
    default_params={"random_state": 42, "n_jobs": -1},
    search_space=lgbm_classifier.search_space
)

# 12. XGBoost Classifier (Requires pip install xgboost)
xgb_classifier = ModelSpec(
    name="XGBClassifier",
    family=ModelFamily.GRADIENT_BOOSTING,
    supported_tasks=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
    class_path="xgboost.XGBClassifier",
    default_params={"random_state": 42, "n_jobs": -1, "use_label_encoder": False, "eval_metric": "logloss"},
    search_space=[
        HyperparameterSpace(name="learning_rate", type="float", low=1e-3, high=1.0, log=True, default=0.1),
        HyperparameterSpace(name="n_estimators", type="int", low=50, high=1000, default=100),
        HyperparameterSpace(name="max_depth", type="int", low=3, high=10, default=6),
        HyperparameterSpace(name="min_child_weight", type="float", low=1, high=20, default=1),
        HyperparameterSpace(name="reg_alpha", type="float", low=1e-8, high=10.0, log=True, default=0.0),
        HyperparameterSpace(name="reg_lambda", type="float", low=1e-8, high=10.0, log=True, default=1.0),
    ]
)

# 13. XGBoost Regressor
xgb_regressor = ModelSpec(
    name="XGBRegressor",
    family=ModelFamily.GRADIENT_BOOSTING,
    supported_tasks=[TaskType.REGRESSION],
    class_path="xgboost.XGBRegressor",
    default_params={"random_state": 42, "n_jobs": -1},
    search_space=xgb_classifier.search_space
)


# 14. Logistic Regression
logistic_regression = ModelSpec(
    name="LogisticRegression",
    family=ModelFamily.LINEAR,
    supported_tasks=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
    class_path="sklearn.linear_model.LogisticRegression",
    default_params={"random_state": 42, "max_iter": 1000},
    search_space=[
        HyperparameterSpace(name="C", type="float", low=1e-4, high=1e4, log=True, default=1.0)
    ]
)

# 15. ExtraTrees Classifier
et_classifier = ModelSpec(
    name="ExtraTreesClassifier",
    family=ModelFamily.TREE_ENSEMBLE,
    supported_tasks=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
    class_path="sklearn.ensemble.ExtraTreesClassifier",
    default_params={"random_state": 42},
    search_space=rf_classifier.search_space
)

# 16. ExtraTrees Regressor
et_regressor = ModelSpec(
    name="ExtraTreesRegressor",
    family=ModelFamily.TREE_ENSEMBLE,
    supported_tasks=[TaskType.REGRESSION],
    class_path="sklearn.ensemble.ExtraTreesRegressor",
    default_params={"random_state": 42},
    search_space=rf_classifier.search_space
)

# 17. CatBoost Classifier
catboost_classifier = ModelSpec(
    name="CatBoostClassifier",
    family=ModelFamily.GRADIENT_BOOSTING,
    supported_tasks=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
    class_path="catboost.CatBoostClassifier",
    default_params={"random_state": 42, "verbose": 0, "thread_count": -1},
    search_space=[
        HyperparameterSpace(name="learning_rate", type="float", low=1e-3, high=1.0, log=True, default=0.1),
        HyperparameterSpace(name="iterations", type="int", low=50, high=1000, default=100),
        HyperparameterSpace(name="depth", type="int", low=3, high=10, default=6),
        HyperparameterSpace(name="l2_leaf_reg", type="float", low=1e-8, high=10.0, log=True, default=3.0),
    ]
)

# 18. CatBoost Regressor
catboost_regressor = ModelSpec(
    name="CatBoostRegressor",
    family=ModelFamily.GRADIENT_BOOSTING,
    supported_tasks=[TaskType.REGRESSION],
    class_path="catboost.CatBoostRegressor",
    default_params={"random_state": 42, "verbose": 0, "thread_count": -1},
    search_space=catboost_classifier.search_space
)


ALL_MODELS = [
    rf_classifier, rf_regressor,
    et_classifier, et_regressor,
    hist_gb_classifier, hist_gb_regressor,
    logistic_regression, ridge_classifier, ridge_regressor,
    elasticnet_regressor,
    dummy_classifier, dummy_regressor,
    lgbm_classifier, lgbm_regressor,
    xgb_classifier, xgb_regressor,
    catboost_classifier, catboost_regressor,
]
