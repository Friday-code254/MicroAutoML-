"""
MicroAutoML-Agent — Datetime Features

Provides transformers for extracting useful components from datetime features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Any


class DatetimeExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features from datetime columns: year, month, day, 
    day_of_week, hour, is_weekend, and elapsed time since epoch.
    """
    
    def __init__(
        self, 
        extract_year: bool = True,
        extract_month: bool = True,
        extract_day: bool = True,
        extract_dow: bool = True,
        extract_hour: bool = True,
        extract_is_weekend: bool = True,
        extract_elapsed: bool = True,
        **kwargs: Any
    ) -> None:
        self.extract_year = extract_year
        self.extract_month = extract_month
        self.extract_day = extract_day
        self.extract_dow = extract_dow
        self.extract_hour = extract_hour
        self.extract_is_weekend = extract_is_weekend
        self.extract_elapsed = extract_elapsed
        
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.out_feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None) -> DatetimeExtractor:
        if not isinstance(X, pd.DataFrame):
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = np.array([f"dt_{i}" for i in range(self.n_features_in_)])
        else:
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = np.array(X.columns)
            
        self.out_feature_names_ = []
        for col in self.feature_names_in_:
            if self.extract_year:
                self.out_feature_names_.append(f"{col}_year")
            if self.extract_month:
                self.out_feature_names_.append(f"{col}_month")
            if self.extract_day:
                self.out_feature_names_.append(f"{col}_day")
            if self.extract_dow:
                self.out_feature_names_.append(f"{col}_dow")
            if self.extract_hour:
                self.out_feature_names_.append(f"{col}_hour")
            if self.extract_is_weekend:
                self.out_feature_names_.append(f"{col}_is_weekend")
            if self.extract_elapsed:
                self.out_feature_names_.append(f"{col}_elapsed")
                
        self.epoch_ = pd.Timestamp("1970-01-01")
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        check_is_fitted(self, "_is_fitted")
        
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X_df = X
            
        out_cols = {}
        for col in self.feature_names_in_:
            s = pd.to_datetime(X_df[col], errors='coerce')
            
            if self.extract_year:
                out_cols[f"{col}_year"] = s.dt.year.astype(float)
            if self.extract_month:
                out_cols[f"{col}_month"] = s.dt.month.astype(float)
            if self.extract_day:
                out_cols[f"{col}_day"] = s.dt.day.astype(float)
            if self.extract_dow:
                out_cols[f"{col}_dow"] = s.dt.dayofweek.astype(float)
            if self.extract_hour:
                out_cols[f"{col}_hour"] = s.dt.hour.astype(float)
            if self.extract_is_weekend:
                out_cols[f"{col}_is_weekend"] = (s.dt.dayofweek >= 5).astype(float)
            if self.extract_elapsed:
                if s.dt.tz is not None:
                    s = s.dt.tz_localize(None)
                out_cols[f"{col}_elapsed"] = (s - self.epoch_).dt.total_seconds().astype(float)
                
        res = pd.DataFrame(out_cols, index=X_df.index)
        
        # If input was numpy, return numpy
        if not isinstance(X, pd.DataFrame):
            return res.values
        return res

    def get_feature_names_out(self, input_features: np.ndarray | list[str] | None = None) -> np.ndarray:
        check_is_fitted(self, "out_feature_names_")
        return np.array(self.out_feature_names_)
