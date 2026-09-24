"""
MicroAutoML-Agent — Numerical Feature Transformers

Provides scikit-learn compatible transformers for numerical feature engineering.
Includes wrappers around numpy functions designed to be pipeline-safe and
compatible with pandas DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class BaseNumericTransformer(BaseEstimator, TransformerMixin):
    """Base class for simple element-wise numeric transformations."""
    
    def __init__(self) -> None:
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None) -> BaseNumericTransformer:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)
            self.n_features_in_ = X.shape[1]
        else:
            self.n_features_in_ = X.shape[1]
        return self

    def get_feature_names_out(self, input_features: np.ndarray | list[str] | None = None) -> np.ndarray:
        if input_features is not None:
            return np.array([f"{self._prefix}_{col}" for col in input_features])
        if self.feature_names_in_ is not None:
            return np.array([f"{self._prefix}_{col}" for col in self.feature_names_in_])
        return np.array([f"{self._prefix}_x{i}" for i in range(self.n_features_in_)])
        
    @property
    def _prefix(self) -> str:
        return "trans"


class Log1pTransformer(BaseNumericTransformer):
    """Applies np.log1p (log(1 + x)) to numerical features. Safely handles negatives by clipping."""
    
    @property
    def _prefix(self) -> str:
        return "log1p"

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        check_is_fitted(self, "n_features_in_")
        is_df = isinstance(X, pd.DataFrame)
        X_vals = X.values if is_df else X
        
        # Clip at 0 to avoid NaNs from log1p of negative numbers
        res = np.log1p(np.clip(X_vals.astype(float), 0, None))
        
        if is_df:
            return pd.DataFrame(res, columns=self.get_feature_names_out(), index=X.index)
        return res


class SqrtTransformer(BaseNumericTransformer):
    """Applies np.sqrt. Safely handles negatives by taking absolute value first or clipping."""
    
    @property
    def _prefix(self) -> str:
        return "sqrt"

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        check_is_fitted(self, "n_features_in_")
        is_df = isinstance(X, pd.DataFrame)
        X_vals = X.values if is_df else X
        
        res = np.sqrt(np.clip(X_vals.astype(float), 0, None))
        
        if is_df:
            return pd.DataFrame(res, columns=self.get_feature_names_out(), index=X.index)
        return res


class SquareTransformer(BaseNumericTransformer):
    """Applies np.square."""
    
    @property
    def _prefix(self) -> str:
        return "sq"

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        check_is_fitted(self, "n_features_in_")
        is_df = isinstance(X, pd.DataFrame)
        X_vals = X.values if is_df else X
        
        res = np.square(X_vals.astype(float))
        
        if is_df:
            return pd.DataFrame(res, columns=self.get_feature_names_out(), index=X.index)
        return res


class ClipTransformer(BaseNumericTransformer):
    """Clips features to the 1st and 99th percentiles learned from the training data."""
    
    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> None:
        super().__init__()
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    @property
    def _prefix(self) -> str:
        return "clip"

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None) -> ClipTransformer:
        super().fit(X, y)
        is_df = isinstance(X, pd.DataFrame)
        X_vals = X.values if is_df else X
        
        self.lower_bounds_ = np.nanpercentile(X_vals.astype(float), self.lower_percentile, axis=0)
        self.upper_bounds_ = np.nanpercentile(X_vals.astype(float), self.upper_percentile, axis=0)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        is_df = isinstance(X, pd.DataFrame)
        X_vals = X.values if is_df else X
        
        res = np.clip(X_vals.astype(float), self.lower_bounds_, self.upper_bounds_)
        
        if is_df:
            return pd.DataFrame(res, columns=self.get_feature_names_out(), index=X.index)
        return res


class RankTransformer(BaseNumericTransformer):
    """Converts continuous features to their rank (percentile) representation."""
    
    @property
    def _prefix(self) -> str:
        return "rank"

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None) -> RankTransformer:
        super().fit(X, y)
        is_df = isinstance(X, pd.DataFrame)
        X_vals = X.values if is_df else X
        
        # Store sorted values for each feature to compute percentiles at transform time
        self.sorted_features_ = []
        for i in range(X_vals.shape[1]):
            col = X_vals[:, i].astype(float)
            col = col[~np.isnan(col)]
            col.sort()
            self.sorted_features_.append(col)
            
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        check_is_fitted(self, "sorted_features_")
        is_df = isinstance(X, pd.DataFrame)
        X_vals = X.values if is_df else X
        
        res = np.zeros_like(X_vals, dtype=float)
        for i in range(X_vals.shape[1]):
            col = X_vals[:, i].astype(float)
            # Use searchsorted to find rank, then divide by N to get percentile
            n = len(self.sorted_features_[i])
            if n > 1:
                ranks = np.searchsorted(self.sorted_features_[i], col)
                res[:, i] = ranks / (n - 1)
            else:
                res[:, i] = 0.5
                
        if is_df:
            return pd.DataFrame(res, columns=self.get_feature_names_out(), index=X.index)
        return res
