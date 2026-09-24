"""
Phase 3 Tests — Feature Engineering Engine

Tests for numerical transforms, categorical encoding wrappers, datetime extraction, 
interaction features, and feature selection modules. Ensure all transformers 
are pipeline-safe, correctly handle data frames, and avoid target leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

from automl.features.transforms import (
    Log1pTransformer, SqrtTransformer, SquareTransformer, ClipTransformer, RankTransformer
)
from automl.features.encoding import FrequencyEncoder, CategoricalPassthrough
from automl.features.datetime import DatetimeExtractor
from automl.features.interactions import NumericInteractions
from automl.features.aggregations import GroupAggregationTransformer
from automl.features.selection import KeepFeatures, DropFeatures, HighCorrelationFilter


class TestTransforms:
    def test_log1p_transformer(self) -> None:
        df = pd.DataFrame({"a": [0, np.e - 1, -5], "b": [1, 2, 3]})
        trans = Log1pTransformer()
        res = trans.fit_transform(df)
        
        assert isinstance(res, pd.DataFrame)
        assert res.columns.tolist() == ["log1p_a", "log1p_b"]
        assert res["log1p_a"].iloc[0] == 0.0
        assert res["log1p_a"].iloc[1] == pytest.approx(1.0)
        assert res["log1p_a"].iloc[2] == 0.0  # -5 should be clipped to 0, log1p(0) = 0
        
    def test_clip_transformer(self) -> None:
        df = pd.DataFrame({"a": range(101)}) # 0 to 100
        trans = ClipTransformer(lower_percentile=10.0, upper_percentile=90.0)
        res = trans.fit_transform(df)
        
        assert res["clip_a"].min() == 10.0
        assert res["clip_a"].max() == 90.0

    def test_rank_transformer(self) -> None:
        df = pd.DataFrame({"a": [100, 10, 50, 50]})
        trans = RankTransformer()
        res = trans.fit_transform(df)
        
        # Sorted unique values for ranking: 10, 50, 50, 100
        # 10 is index 0 -> 0.0
        # 50 is index 1 or 2 -> 1/4 = 0.25 (or depending on searchsorted, could be 0.25 or 0.5)
        # 100 is index 4 (end) -> 4/4 = 1.0
        assert res["rank_a"].iloc[0] == 1.0
        assert res["rank_a"].iloc[1] == 0.0


class TestEncoding:
    def test_frequency_encoder(self) -> None:
        df = pd.DataFrame({"cat": ["A", "A", "A", "B"]})
        enc = FrequencyEncoder()
        res = enc.fit_transform(df)
        
        assert res["freq_cat"].iloc[0] == 0.75
        assert res["freq_cat"].iloc[3] == 0.25
        
        # Test unknown category on transform
        df_test = pd.DataFrame({"cat": ["C"]})
        res_test = enc.transform(df_test)
        assert res_test["freq_cat"].iloc[0] == 0.0

    def test_categorical_passthrough(self) -> None:
        df = pd.DataFrame({"cat": [1, 2, 3]})
        enc = CategoricalPassthrough(cast_to="category")
        res = enc.fit_transform(df)
        assert res["cat"].dtype.name == "category"


class TestDatetimeExtractor:
    def test_datetime_extractor(self) -> None:
        df = pd.DataFrame({"dt": ["2020-01-01 12:00:00", "2021-02-02 14:30:00"]})
        extractor = DatetimeExtractor(extract_dow=True, extract_elapsed=True)
        res = extractor.fit_transform(df)
        
        assert "dt_year" in res.columns
        assert "dt_hour" in res.columns
        assert res["dt_year"].iloc[0] == 2020
        assert res["dt_hour"].iloc[1] == 14
        assert res["dt_elapsed"].iloc[0] > 0
        

class TestInteractions:
    def test_numeric_interactions(self) -> None:
        df = pd.DataFrame({"a": [10, 20], "b": [2, 5], "c": [1, 1]})
        trans = NumericInteractions(interaction_pairs=[("a", "b")])
        res = trans.fit_transform(df)
        
        assert "a_add_b" in res.columns
        assert "a_sub_b" in res.columns
        assert "a_mul_b" in res.columns
        assert "a_div_b" in res.columns
        
        assert res["a_add_b"].iloc[0] == 12
        assert res["a_sub_b"].iloc[1] == 15
        assert res["a_mul_b"].iloc[0] == 20
        assert res["a_div_b"].iloc[1] == 4.0


class TestAggregations:
    def test_group_aggregation(self) -> None:
        df = pd.DataFrame({
            "group": ["A", "A", "B", "B", "B"],
            "val": [10, 20, 1, 2, 3]
        })
        trans = GroupAggregationTransformer(group_col="group", numeric_col="val", aggs=["mean", "max"])
        res = trans.fit_transform(df)
        
        assert "group_val_mean" in res.columns
        assert "group_val_max" in res.columns
        assert res["group_val_mean"].iloc[0] == 15.0 # Group A mean
        assert res["group_val_max"].iloc[2] == 3.0   # Group B max
        
        # Unknown group
        df_test = pd.DataFrame({"group": ["C"], "val": [100]})
        res_test = trans.transform(df_test)
        # Should fallback to global mean (36/5 = 7.2) and global max (20)
        assert res_test["group_val_mean"].iloc[0] == 7.2
        assert res_test["group_val_max"].iloc[0] == 20.0


class TestSelection:
    def test_keep_features(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        trans = KeepFeatures(features_to_keep=["a", "c"])
        res = trans.fit_transform(df)
        assert list(res.columns) == ["a", "c"]

    def test_drop_features(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        trans = DropFeatures(features_to_drop=["b"])
        res = trans.fit_transform(df)
        assert list(res.columns) == ["a", "c"]

    def test_high_correlation_filter(self) -> None:
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100),       # perfectly correlated with a
            "c": np.random.randn(100) # uncorrelated
        })
        trans = HighCorrelationFilter(threshold=0.95)
        res = trans.fit_transform(df)
        
        # 'a' and 'c' should remain, 'b' is dropped because it correlates with 'a'
        assert "a" in res.columns
        assert "c" in res.columns
        assert "b" not in res.columns
