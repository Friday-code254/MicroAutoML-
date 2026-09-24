"""
MicroAutoML-Agent — Rule Engine

Deterministic hypothesis generation based on data evidence.
Guarantees a safe baseline of feature engineering regardless of SLM capabilities.
"""

from __future__ import annotations

from automl.core.enums import FeatureDtype, OperatorType, TaskType
from automl.core.schemas import EvidencePack, Hypothesis


class RuleEngine:
    """Generates standard machine learning hypotheses based on data shape and statistics."""
    
    def generate_hypotheses(self, evidence: EvidencePack) -> list[Hypothesis]:
        hypotheses = []
        
        # We look at the feature map and fast profile
        num_features = []
        cat_features = []
        high_card_cat = []
        skewed_features = []
        missing_num = []
        missing_cat = []
        dt_features = []
        
        for name, info in evidence.feature_map.items():
            if info.dtype == FeatureDtype.NUMERICAL:
                num_features.append(name)
                if info.missing_ratio > 0:
                    missing_num.append(name)
                if info.skew and abs(info.skew) > 1.5:
                    skewed_features.append(name)
            elif info.dtype == FeatureDtype.CATEGORICAL:
                cat_features.append(name)
                if info.missing_ratio > 0:
                    missing_cat.append(name)
                if info.cardinality > 20:
                    high_card_cat.append(name)
            elif info.dtype == FeatureDtype.DATETIME:
                dt_features.append(name)
                
        # Rule 1: Numeric Imputation
        if missing_num:
            hypotheses.append(Hypothesis(
                hypothesis="Impute missing numeric values using median to handle potential outliers.",
                evidence=[f"{len(missing_num)} numeric features contain missing values."],
                experiment_type=OperatorType.PREPROCESSING_CHANGE,
                changes=["Add SimpleImputer(strategy='median') for numeric columns"],
                candidate_pipeline={
                    "preprocessing": [{
                        "name": "SimpleImputer",
                        "columns": missing_num,
                        "params": {"strategy": "median"}
                    }]
                },
                expected_gain=0.05,
                expected_cost=0.1,
                risk=["Median imputation might destroy signal if missingness is not MAR."],
                success_condition="Model handles missing data without crashing and improves CV score.",
                source="rules"
            ))

        # Rule 2: Categorical Imputation
        if missing_cat:
            hypotheses.append(Hypothesis(
                hypothesis="Impute missing categorical values with a constant 'missing' category.",
                evidence=[f"{len(missing_cat)} categorical features contain missing values."],
                experiment_type=OperatorType.PREPROCESSING_CHANGE,
                changes=["Add SimpleImputer(strategy='constant', fill_value='missing') for categorical columns"],
                candidate_pipeline={
                    "preprocessing": [{
                        "name": "SimpleImputer",
                        "columns": missing_cat,
                        "params": {"strategy": "constant", "fill_value": "missing"}
                    }]
                },
                expected_gain=0.03,
                expected_cost=0.1,
                risk=["Adds a new category that might be rare."],
                success_condition="Model handles missing data without crashing.",
                source="rules"
            ))

        # Rule 3: Target Encoding for High Cardinality
        if high_card_cat:
            hypotheses.append(Hypothesis(
                hypothesis="Apply Target Encoding to high cardinality categoricals to reduce feature space explosion.",
                evidence=[f"{len(high_card_cat)} categorical features have cardinality > 20."],
                experiment_type=OperatorType.FEATURE_ENGINEERING,
                changes=["Add TargetEncoder for high cardinality features"],
                candidate_pipeline={
                    "preprocessing": [{
                        "name": "TargetEncoder",
                        "columns": high_card_cat,
                        "params": {"target_type": "continuous" if evidence.task and evidence.task.task_type == "regression" else "binary"}
                    }]
                },
                expected_gain=0.1,
                expected_cost=0.2,
                risk=["Target encoding can cause overfitting if smoothing is insufficient."],
                success_condition="Validation score improves without train score diverging significantly.",
                source="rules"
            ))

        # Rule 4: One-Hot Encoding for Low Cardinality
        low_card_cat = [c for c in cat_features if c not in high_card_cat]
        if low_card_cat:
            hypotheses.append(Hypothesis(
                hypothesis="Apply One-Hot Encoding to low cardinality categoricals.",
                evidence=[f"{len(low_card_cat)} categorical features have low cardinality."],
                experiment_type=OperatorType.FEATURE_ENGINEERING,
                changes=["Add OneHotEncoder for low cardinality features"],
                candidate_pipeline={
                    "preprocessing": [{
                        "name": "OneHotEncoder",
                        "columns": low_card_cat,
                        "params": {"handle_unknown": "ignore", "sparse_output": False}
                    }]
                },
                expected_gain=0.05,
                expected_cost=0.3,
                risk=["Increases dimensionality."],
                success_condition="Linear and Tree models can process the categorical features.",
                source="rules"
            ))

        # Rule 5: Skew reduction
        if skewed_features:
            hypotheses.append(Hypothesis(
                hypothesis="Apply Log1p transform to highly skewed numeric features to stabilize variance.",
                evidence=[f"{len(skewed_features)} numeric features have skew magnitude > 1.5."],
                experiment_type=OperatorType.FEATURE_ENGINEERING,
                changes=["Apply Log1pTransformer to skewed features"],
                candidate_pipeline={
                    "preprocessing": [{
                        "name": "Log1pTransformer",
                        "columns": skewed_features,
                        "params": {}
                    }]
                },
                expected_gain=0.08,
                expected_cost=0.2,
                risk=["Log1p might not be appropriate if features have negative values (though transformer clips them)."],
                success_condition="Linear models improve due to normalized distributions.",
                source="rules"
            ))

        # Rule 6: Datetime Extraction
        if dt_features:
            hypotheses.append(Hypothesis(
                hypothesis="Extract temporal components (year, month, dow, hour) from datetime features.",
                evidence=[f"Dataset contains {len(dt_features)} datetime features."],
                experiment_type=OperatorType.FEATURE_ENGINEERING,
                changes=["Apply DatetimeExtractor"],
                candidate_pipeline={
                    "preprocessing": [{
                        "name": "DatetimeExtractor",
                        "columns": dt_features,
                        "params": {"extract_year": True, "extract_month": True, "extract_dow": True, "extract_hour": True}
                    }]
                },
                expected_gain=0.15,
                expected_cost=0.3,
                risk=["Increases dimensionality, tree models might struggle with cyclic patterns without cyclic encoding."],
                success_condition="Tree models capture temporal effects.",
                source="rules"
            ))

        # Rule 7: Model Selection (CatBoost for Categorical Heavy)
        if evidence.fingerprint.categorical_fraction > 0.3 and evidence.fingerprint.nonlinear_signal > 0.3:
            is_classification = evidence.task and evidence.task.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
            model_name = "CatBoostClassifier" if is_classification else "CatBoostRegressor"
            hypotheses.append(Hypothesis(
                hypothesis=f"Use {model_name} since dataset has high categorical fraction and nonlinear signal.",
                evidence=[
                    f"Categorical fraction: {evidence.fingerprint.categorical_fraction:.2f}",
                    f"Nonlinear signal: {evidence.fingerprint.nonlinear_signal:.2f}"
                ],
                experiment_type=OperatorType.MODEL_CHANGE,
                changes=[f"Change estimator to {model_name}"],
                candidate_pipeline={
                    "model_name": model_name
                },
                expected_gain=0.2,
                expected_cost=0.4,
                risk=["CatBoost can be slower to train than LightGBM."],
                success_condition="Model outperforms simpler trees without heavy preprocessing.",
                source="rules"
            ))

        # Rule 8: Feature Selection (High Redundancy)
        if evidence.fingerprint.redundancy > 0.5:
            hypotheses.append(Hypothesis(
                hypothesis="Apply HighCorrelationFilter to reduce redundancy and multicollinearity.",
                evidence=[f"Dataset redundancy is high ({evidence.fingerprint.redundancy:.2f})."],
                experiment_type=OperatorType.FEATURE_SELECTION,
                changes=["Add HighCorrelationFilter(threshold=0.85)"],
                candidate_pipeline={
                    "feature_selection": {
                        "name": "HighCorrelationFilter",
                        "params": {"threshold": 0.85}
                    }
                },
                expected_gain=0.05,
                expected_cost=0.2,
                risk=["Might drop features that contain independent predictive signal in interactions."],
                success_condition="Model maintains performance with fewer features.",
                source="rules"
            ))

        # Rule 9: HPO for Small/Medium Data
        if evidence.fingerprint.size_label in ("small", "medium"):
            hypotheses.append(Hypothesis(
                hypothesis="Perform Hyperparameter Optimization (HPO).",
                evidence=[f"Dataset size is {evidence.fingerprint.size_label}, HPO is computationally feasible."],
                experiment_type=OperatorType.HPO,
                changes=["Tune model hyperparameters"],
                candidate_pipeline={},  # Leaves model as is, but flags for HPO
                expected_gain=0.1,
                expected_cost=0.8,
                risk=["High compute cost, potential overfitting on small validation sets."],
                success_condition="Optimized params improve over defaults.",
                source="rules"
            ))
            
        return hypotheses
