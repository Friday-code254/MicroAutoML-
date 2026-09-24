"""
MicroAutoML-Agent — Transform Specifications

Defines the TransformSpec contract and pre-registers standard preprocessing 
and feature engineering components (from Phase 3 and sklearn).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TransformSpec(BaseModel):
    """
    Specification for a transformer component.
    """
    name: str
    class_path: str
    is_feature_generator: bool = False
    is_preprocessing: bool = False
    is_selector: bool = False


# 1. Core Preprocessing (sklearn)
imputer_simple = TransformSpec(
    name="SimpleImputer",
    class_path="sklearn.impute.SimpleImputer",
    is_preprocessing=True
)

scaler_standard = TransformSpec(
    name="StandardScaler",
    class_path="sklearn.preprocessing.StandardScaler",
    is_preprocessing=True
)

scaler_robust = TransformSpec(
    name="RobustScaler",
    class_path="sklearn.preprocessing.RobustScaler",
    is_preprocessing=True
)

scaler_minmax = TransformSpec(
    name="MinMaxScaler",
    class_path="sklearn.preprocessing.MinMaxScaler",
    is_preprocessing=True
)

encoder_onehot = TransformSpec(
    name="OneHotEncoder",
    class_path="sklearn.preprocessing.OneHotEncoder",
    is_preprocessing=True
)

encoder_ordinal = TransformSpec(
    name="OrdinalEncoder",
    class_path="sklearn.preprocessing.OrdinalEncoder",
    is_preprocessing=True
)

encoder_target = TransformSpec(
    name="TargetEncoder",
    class_path="sklearn.preprocessing.TargetEncoder",
    is_preprocessing=True
)

# 2. Custom Transforms (Phase 3)
trans_log1p = TransformSpec(
    name="Log1pTransformer",
    class_path="automl.features.transforms.Log1pTransformer",
    is_feature_generator=True
)

trans_sqrt = TransformSpec(
    name="SqrtTransformer",
    class_path="automl.features.transforms.SqrtTransformer",
    is_feature_generator=True
)

trans_square = TransformSpec(
    name="SquareTransformer",
    class_path="automl.features.transforms.SquareTransformer",
    is_feature_generator=True
)

trans_clip = TransformSpec(
    name="ClipTransformer",
    class_path="automl.features.transforms.ClipTransformer",
    is_feature_generator=True
)

trans_rank = TransformSpec(
    name="RankTransformer",
    class_path="automl.features.transforms.RankTransformer",
    is_feature_generator=True
)

enc_frequency = TransformSpec(
    name="FrequencyEncoder",
    class_path="automl.features.encoding.FrequencyEncoder",
    is_feature_generator=True
)

enc_passthrough = TransformSpec(
    name="CategoricalPassthrough",
    class_path="automl.features.encoding.CategoricalPassthrough",
    is_preprocessing=True
)

feat_datetime = TransformSpec(
    name="DatetimeExtractor",
    class_path="automl.features.datetime.DatetimeExtractor",
    is_feature_generator=True
)

feat_interactions = TransformSpec(
    name="NumericInteractions",
    class_path="automl.features.interactions.NumericInteractions",
    is_feature_generator=True
)

feat_aggregations = TransformSpec(
    name="GroupAggregationTransformer",
    class_path="automl.features.aggregations.GroupAggregationTransformer",
    is_feature_generator=True
)

sel_keep = TransformSpec(
    name="KeepFeatures",
    class_path="automl.features.selection.KeepFeatures",
    is_selector=True
)

sel_drop = TransformSpec(
    name="DropFeatures",
    class_path="automl.features.selection.DropFeatures",
    is_selector=True
)

sel_correlation = TransformSpec(
    name="HighCorrelationFilter",
    class_path="automl.features.selection.HighCorrelationFilter",
    is_selector=True
)


ALL_TRANSFORMS = [
    imputer_simple, scaler_standard, scaler_robust, scaler_minmax,
    encoder_onehot, encoder_ordinal, encoder_target,
    trans_log1p, trans_sqrt, trans_square, trans_clip, trans_rank,
    enc_frequency, enc_passthrough,
    feat_datetime, feat_interactions, feat_aggregations,
    sel_keep, sel_drop, sel_correlation
]
