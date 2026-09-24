"""
MicroAutoML-Agent — Categorical Encoders

Provides custom encoders for categorical features. Standard encoders 
like OneHotEncoder, OrdinalEncoder, and TargetEncoder are imported 
from sklearn.preprocessing directly by the pipeline compiler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features by replacing them with their normalized frequency 
    (proportion) in the training set.
    Unknown categories during transform are assigned a frequency of 0.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        self.frequencies_: dict[int, dict[Any, float]] = {}
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None) -> FrequencyEncoder:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)
            self.n_features_in_ = X.shape[1]
            X_vals = X.values
        else:
            self.n_features_in_ = X.shape[1]
            X_vals = X

        n_samples = X_vals.shape[0]
        
        for i in range(self.n_features_in_):
            col = X_vals[:, i]
            # Pandas value_counts with normalize=True gives proportion
            val_counts = pd.Series(col).value_counts(normalize=True, dropna=False)
            self.frequencies_[i] = val_counts.to_dict()
            
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        check_is_fitted(self, "frequencies_")
        is_df = isinstance(X, pd.DataFrame)
        X_vals = X.values if is_df else X
        
        res = np.zeros(X_vals.shape, dtype=float)
        
        for i in range(self.n_features_in_):
            col = pd.Series(X_vals[:, i])
            # Map values to frequencies, fill unknown with 0.0
            freqs = col.map(self.frequencies_[i]).fillna(0.0).values
            res[:, i] = freqs
            
        if is_df:
            cols = [f"freq_{c}" for c in self.feature_names_in_] if self.feature_names_in_ is not None else None
            return pd.DataFrame(res, columns=cols, index=X.index)
        return res

    def get_feature_names_out(self, input_features: np.ndarray | list[str] | None = None) -> np.ndarray:
        if input_features is not None:
            return np.array([f"freq_{col}" for col in input_features])
        if self.feature_names_in_ is not None:
            return np.array([f"freq_{col}" for col in self.feature_names_in_])
        return np.array([f"freq_x{i}" for i in range(self.n_features_in_)])


class CategoricalPassthrough(BaseEstimator, TransformerMixin):
    """
    Passes categorical features through untouched, but ensures they are 
    cast to string or 'category' dtype, which is required for native 
    categorical support in CatBoost, LightGBM, and HistGradientBoosting.
    """
    
    def __init__(self, cast_to: str = "str", **kwargs: Any) -> None:
        """cast_to: 'str' or 'category'"""
        self.cast_to = cast_to
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None) -> CategoricalPassthrough:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)
            self.n_features_in_ = X.shape[1]
        else:
            self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        check_is_fitted(self, "_is_fitted")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"x{i}" for i in range(self.n_features_in_)])
            
        X = X.fillna("__MISSING__")
            
        if self.cast_to == "category":
            return X.astype("category")
        return X.astype(str)

    def get_feature_names_out(self, input_features: np.ndarray | list[str] | None = None) -> np.ndarray:
        if input_features is not None:
            return np.array(input_features)
        if self.feature_names_in_ is not None:
            return np.array(self.feature_names_in_)
        return np.array([f"cat_x{i}" for i in range(self.n_features_in_)])
