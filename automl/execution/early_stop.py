"""
MicroAutoML-Agent — Early Stopping Manager

Configures early stopping parameters for gradient boosting models during training.

Design note (important):
  `eval_set` cannot safely be injected via Pipeline.fit(**fit_params) because the
  estimator is the *last* step inside a sklearn Pipeline. By the time the estimator
  receives the training data it has already been transformed by preprocessing steps,
  but a raw `X_val` passed as eval_set has NOT been transformed. This causes a
  representation mismatch and silently produces wrong early-stopping signals.

  Safe alternatives:
  - Set `early_stopping_rounds` / `n_iter_no_change` as constructor params on the
    estimator — these trigger internal validation splits, fully inside the fitted
    preprocessing pipeline, with no mismatch.
  - For proper held-out eval_set: Phase 8 will use a pipeline-aware wrapper that
    transforms X_val through the fitted pipeline steps before passing it.

  For now: only init-param-based early stopping is injected here.
"""

from __future__ import annotations

from typing import Any

from automl.core.schemas import ExperimentSpec


class EarlyStoppingManager:
    """
    Configures early stopping for gradient boosting models.

    Only injects parameters that are safe to use inside a sklearn Pipeline
    (i.e., constructor-level params or fit_params that don't reference raw data).
    """

    # Models that support early_stopping_rounds as a fit param (NOT eval_set — see module docstring)
    _EARLY_STOP_MODELS = frozenset({
        "XGBClassifier", "XGBRegressor",
        "LGBMClassifier", "LGBMRegressor",
        "CatBoostClassifier", "CatBoostRegressor",
    })

    def get_fit_params(self, spec: ExperimentSpec, X_val: Any = None, y_val: Any = None) -> dict[str, Any]:
        """
        Returns fit_params safe to pass to Pipeline.fit().

        IMPORTANT: Does NOT inject eval_set. See module docstring.
        """
        # No safe fit_params to inject at this stage.
        # eval_set requires pipeline-aware wrapping (Phase 8).
        return {}

    def get_constructor_overrides(self, spec: ExperimentSpec) -> dict[str, Any]:
        """
        Returns hyperparameter overrides to apply at estimator construction time.
        These are merged into spec.hyperparameters before the compiler builds the pipeline.

        Safe because these operate entirely on train-fold data within the pipeline.
        """
        overrides: dict[str, Any] = {}

        if spec.model_name in ("XGBClassifier", "XGBRegressor"):
            overrides["early_stopping_rounds"] = None  # Disabled until eval_set fix in Phase 8
        elif spec.model_name in ("LGBMClassifier", "LGBMRegressor"):
            pass  # LGBM early stopping also requires eval_set; deferred to Phase 8
        elif spec.model_name in ("HistGradientBoostingClassifier", "HistGradientBoostingRegressor"):
            # HistGB supports internal validation via constructor params — safe
            overrides["validation_fraction"] = 0.1
            overrides["n_iter_no_change"] = 10
        elif spec.model_name in ("CatBoostClassifier", "CatBoostRegressor"):
            pass  # CatBoost early stopping requires eval_set; deferred to Phase 8

        return overrides

