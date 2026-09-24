"""
Unit tests for GroupAggregationTransformer
"""
import pytest
import numpy as np
import pandas as pd

from automl.features.aggregations import GroupAggregationTransformer


def test_group_aggregation_basic():
    """Test standard aggregations on a simple dataframe."""
    df = pd.DataFrame({
        "category": ["A", "A", "B", "B", "C"],
        "value": [10, 20, 100, 200, 500]
    })
    
    transformer = GroupAggregationTransformer(
        group_col="category", 
        numeric_col="value",
        aggs=["mean", "sum", "max"] # sum should fail safely or be ignored since it's not in the default supported list? Wait, if we pass sum, it will KeyError if we didn't implement it. Let's stick to the supported ones.
    )
    # The current transformer supports: mean, median, std, min, max, count.
    transformer = GroupAggregationTransformer(
        group_col="category", 
        numeric_col="value",
        aggs=["mean", "count"]
    )
    
    transformed = transformer.fit_transform(df)
    
    # Check shapes
    assert transformed.shape == (5, 2)
    assert list(transformed.columns) == ["category_value_mean", "category_value_count"]
    
    # Check A
    assert transformed["category_value_mean"].iloc[0] == 15.0
    assert transformed["category_value_count"].iloc[0] == 2.0
    
    # Check C
    assert transformed["category_value_mean"].iloc[4] == 500.0
    assert transformed["category_value_count"].iloc[4] == 1.0


def test_group_aggregation_auto_inference():
    """Test that it correctly infers categorical and numeric columns if omitted."""
    df = pd.DataFrame({
        "cat1": ["X", "X", "Y"],
        "num1": [1.0, 3.0, 5.0]
    })
    
    # Don't specify columns
    transformer = GroupAggregationTransformer(aggs=["mean"])
    transformed = transformer.fit_transform(df)
    
    assert transformer.group_col == "cat1"
    assert transformer.numeric_col == "num1"
    assert transformed["cat1_num1_mean"].iloc[0] == 2.0
    assert transformed["cat1_num1_mean"].iloc[2] == 5.0


def test_group_aggregation_global_fallback():
    """Test that unseen categories in transform() get global fallbacks instead of NaNs."""
    train_df = pd.DataFrame({
        "group": ["G1", "G1"],
        "val": [10.0, 20.0]
    })
    
    test_df = pd.DataFrame({
        "group": ["G1", "G2"],  # G2 is unseen
        "val": [15.0, 999.0]
    })
    
    transformer = GroupAggregationTransformer(group_col="group", numeric_col="val", aggs=["mean", "count"])
    transformer.fit(train_df)
    
    transformed = transformer.transform(test_df)
    
    # G1 should get training mean (15.0) and count (2.0)
    assert transformed["group_val_mean"].iloc[0] == 15.0
    assert transformed["group_val_count"].iloc[0] == 2.0
    
    # G2 should get global fallback: global mean is 15.0, count fallback is always 1.0
    assert transformed["group_val_mean"].iloc[1] == 15.0
    assert transformed["group_val_count"].iloc[1] == 1.0


def test_group_aggregation_missing_values():
    """Test resiliency to missing values in the numeric column during fit and transform."""
    df = pd.DataFrame({
        "grp": ["A", "A", "B", "B"],
        "val": [10.0, np.nan, 20.0, 40.0]
    })
    
    transformer = GroupAggregationTransformer(group_col="grp", numeric_col="val", aggs=["mean"])
    transformed = transformer.fit_transform(df)
    
    # pandas groupby mean ignores NaNs by default
    assert transformed["grp_val_mean"].iloc[0] == 10.0
    assert transformed["grp_val_mean"].iloc[2] == 30.0

def test_group_aggregation_wrong_type():
    """Test that it raises ValueError if passed a numpy array instead of dataframe."""
    transformer = GroupAggregationTransformer(aggs=["mean"])
    with pytest.raises(ValueError, match="GroupAggregationTransformer requires a pandas DataFrame"):
        transformer.fit(np.array([[1, 2], [3, 4]]))
