"""
MicroAutoML-Agent — Interaction Features

Provides transformers for creating interaction features (A+B, A-B, A*B, A/B)
between selected numeric candidate pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class NumericInteractions(BaseEstimator, TransformerMixin):
    """
    Creates arithmetic interactions for explicitly specified column pairs.
    Pairs are passed as a list of tuples: [("col1", "col2"), ("col3", "col4")]
    """
    
    def __init__(self, interaction_pairs: list[tuple[str, str]]) -> None:
        self.interaction_pairs = interaction_pairs
        
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None) -> NumericInteractions:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("NumericInteractions requires a pandas DataFrame with column names.")
            
        self.feature_names_in_ = np.array(X.columns)
        self.n_features_in_ = X.shape[1]
        
        # Verify all pairs exist
        for col1, col2 in self.interaction_pairs:
            if col1 not in self.feature_names_in_ or col2 not in self.feature_names_in_:
                raise ValueError(f"Interaction pair ({col1}, {col2}) not found in input features.")
                
        self.out_feature_names_ = []
        for col1, col2 in self.interaction_pairs:
            self.out_feature_names_.extend([
                f"{col1}_add_{col2}",
                f"{col1}_sub_{col2}",
                f"{col1}_mul_{col2}",
                f"{col1}_div_{col2}"
            ])
            
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "_is_fitted")
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("NumericInteractions requires a pandas DataFrame.")
            
        out_cols = {}
        for col1, col2 in self.interaction_pairs:
            c1 = X[col1].astype(float)
            c2 = X[col2].astype(float)
            
            out_cols[f"{col1}_add_{col2}"] = c1 + c2
            out_cols[f"{col1}_sub_{col2}"] = c1 - c2
            out_cols[f"{col1}_mul_{col2}"] = c1 * c2
            
            # Safe division: add epsilon, or handle 0
            with np.errstate(divide='ignore', invalid='ignore'):
                div = np.where(c2 == 0, 0, c1 / c2)
            out_cols[f"{col1}_div_{col2}"] = div
            
        return pd.DataFrame(out_cols, index=X.index)

    def get_feature_names_out(self, input_features: np.ndarray | list[str] | None = None) -> np.ndarray:
        check_is_fitted(self, "out_feature_names_")
        return np.array(self.out_feature_names_)
