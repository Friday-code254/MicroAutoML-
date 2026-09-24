"""
MicroAutoML-Agent — Feature Selection

Provides transformers for dropping or keeping features, as well as dynamic 
filters like Variance and Correlation filters. Note that many selection 
decisions are made statically by the Planner based on the EvidencePack, 
and enacted here via DropFeatures/KeepFeatures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.validation import check_is_fitted


class KeepFeatures(BaseEstimator, TransformerMixin):
    """Keeps only the specified columns, dropping all others."""
    
    def __init__(self, features_to_keep: list[str]) -> None:
        self.features_to_keep = features_to_keep
        
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> KeepFeatures:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("KeepFeatures requires a pandas DataFrame.")
        self.feature_names_in_ = np.array(X.columns)
        self.valid_features_ = [f for f in self.features_to_keep if f in X.columns]
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "_is_fitted")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("KeepFeatures requires a pandas DataFrame.")
        return X[self.valid_features_].copy()

    def get_feature_names_out(self, input_features: np.ndarray | list[str] | None = None) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        return np.array(self.valid_features_)


class DropFeatures(BaseEstimator, TransformerMixin):
    """Drops the specified columns, keeping all others."""
    
    def __init__(self, features_to_drop: list[str]) -> None:
        self.features_to_drop = features_to_drop
        
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> DropFeatures:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("DropFeatures requires a pandas DataFrame.")
        self.feature_names_in_ = np.array(X.columns)
        self.valid_features_ = [f for f in X.columns if f not in self.features_to_drop]
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "_is_fitted")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("DropFeatures requires a pandas DataFrame.")
        return X[self.valid_features_].copy()

    def get_feature_names_out(self, input_features: np.ndarray | list[str] | None = None) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        return np.array(self.valid_features_)


class HighCorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Dynamically drops one feature from pairs with absolute correlation > threshold.
    Computed on the training set during fit.
    """
    
    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold
        
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> HighCorrelationFilter:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("HighCorrelationFilter requires a pandas DataFrame.")
            
        self.feature_names_in_ = np.array(X.columns)
        
        # Only compute on numeric columns
        num_cols = X.select_dtypes(include=np.number).columns
        corr_matrix = X[num_cols].corr().abs()
        
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        
        self.features_to_drop_ = to_drop
        self.valid_features_ = [f for f in X.columns if f not in self.features_to_drop_]
        
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "_is_fitted")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("HighCorrelationFilter requires a pandas DataFrame.")
        return X[self.valid_features_].copy()

    def get_feature_names_out(self, input_features: np.ndarray | list[str] | None = None) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        return np.array(self.valid_features_)
