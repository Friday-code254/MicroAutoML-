"""
MicroAutoML-Agent — Aggregation Features

Provides transformers for group-by aggregations on categorical keys.
Always computed safely on the train fold and joined to the validation/test fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class GroupAggregationTransformer(BaseEstimator, TransformerMixin):
    """
    Groups by a categorical 'group_col' and computes aggregations 
    (mean, median, std, min, max, count) on a 'numeric_col'.
    """
    
    def __init__(
        self, 
        group_col: str | None = None, 
        numeric_col: str | None = None,
        aggs: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        # Accept **kwargs to prevent TypeError if LLM hallucinates parameters like "grouping"
        self.group_col = group_col or kwargs.get("grouping")
        self.numeric_col = numeric_col or kwargs.get("numeric") or kwargs.get("value_col")
        self.aggs = aggs or ["mean", "std", "min", "max", "count"]
        
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None) -> GroupAggregationTransformer:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("GroupAggregationTransformer requires a pandas DataFrame.")
            
        # Resiliency: If columns were omitted or hallucinated, auto-select them
        if not self.group_col or self.group_col not in X.columns:
            # Pick first categorical/object column
            cats = X.select_dtypes(include=["object", "category", "string"]).columns
            if len(cats) > 0:
                self.group_col = cats[0]
            else:
                self.group_col = X.columns[0]
                
        if not self.numeric_col or self.numeric_col not in X.columns:
            # Pick first numerical column that is not the group_col
            nums = X.select_dtypes(include=["number"]).columns
            nums = [c for c in nums if c != self.group_col]
            if len(nums) > 0:
                self.numeric_col = nums[0]
            else:
                self.numeric_col = X.columns[-1]
            
        self.feature_names_in_ = np.array(X.columns)
        
        # Compute agg map
        # e.g., {'mean': group_col -> mean_val}
        grouped = X.groupby(self.group_col)[self.numeric_col]
        self.agg_maps_ = {}
        
        if "mean" in self.aggs:
            self.agg_maps_["mean"] = grouped.mean().to_dict()
        if "median" in self.aggs:
            self.agg_maps_["median"] = grouped.median().to_dict()
        if "std" in self.aggs:
            self.agg_maps_["std"] = grouped.std().to_dict()
        if "min" in self.aggs:
            self.agg_maps_["min"] = grouped.min().to_dict()
        if "max" in self.aggs:
            self.agg_maps_["max"] = grouped.max().to_dict()
        if "count" in self.aggs:
            self.agg_maps_["count"] = grouped.count().to_dict()
            
        # Global fallbacks for unknown groups
        self.global_fallbacks_ = {}
        global_series = X[self.numeric_col]
        
        if "mean" in self.aggs:
            self.global_fallbacks_["mean"] = global_series.mean()
        if "median" in self.aggs:
            self.global_fallbacks_["median"] = global_series.median()
        if "std" in self.aggs:
            self.global_fallbacks_["std"] = global_series.std()
        if "min" in self.aggs:
            self.global_fallbacks_["min"] = global_series.min()
        if "max" in self.aggs:
            self.global_fallbacks_["max"] = global_series.max()
        if "count" in self.aggs:
            self.global_fallbacks_["count"] = 1.0 # 1 count for unknown group
            
        self.out_feature_names_ = [f"{self.group_col}_{self.numeric_col}_{agg}" for agg in self.aggs]
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "_is_fitted")
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("GroupAggregationTransformer requires a pandas DataFrame.")
            
        out_cols = {}
        groups = X[self.group_col]
        
        for agg in self.aggs:
            name = f"{self.group_col}_{self.numeric_col}_{agg}"
            # map and fillna with global fallback
            out_cols[name] = groups.map(self.agg_maps_[agg]).fillna(self.global_fallbacks_[agg])
            
        return pd.DataFrame(out_cols, index=X.index)

    def get_feature_names_out(self, input_features: np.ndarray | list[str] | None = None) -> np.ndarray:
        check_is_fitted(self, "out_feature_names_")
        return np.array(self.out_feature_names_)
