"""
MicroAutoML-Agent — SLM Prompts

Contains carefully structured prompts designed specifically for the Qwen3-4B model.
Includes strict JSON format constraints and few-shot examples to compensate for 
the smaller model's limited zero-shot reasoning capacity.
"""

from __future__ import annotations

import json
from typing import Any

from automl.core.schemas import DatasetFingerprint, Hypothesis, OperatorType


# ---------------------------------------------------------------------------
# Planner Prompt (Hypothesis Generation)
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are an expert AutoML Feature Engineer. Your task is to generate exactly ONE novel machine learning hypothesis based on the dataset fingerprint provided.

RESPONSIBILITY BOUNDARIES (STRICT):
You CAN: Generate hypotheses, suggest unexplored directions, and reflect on past evidence.
You CANNOT: Calculate metrics, execute arbitrary code, inspect test labels, choose arbitrary non-whitelisted libraries, alter resource limits, or declare the search successful.
You CANNOT: Use any column names that are not explicitly provided in the `all_features` list. DO NOT hallucinate features.

You MUST output valid JSON only, exactly matching the schema below.
DO NOT wrap the JSON in markdown blocks like ```json.
DO NOT include any conversational text.

JSON SCHEMA:
{
  "hypothesis": "String describing the idea",
  "evidence": ["String 1", "String 2"],
  "experiment_type": "feature_engineering",
  "changes": ["Change 1", "Change 2"],
  "candidate_pipeline": {
    "preprocessing": [
      {"name": "TransformerName", "columns": ["col_name"], "params": {}}
    ]
  },
  "expected_gain": 0.05,
  "expected_cost": 0.1,
  "information_value": 0.8,
  "failure_risk": 0.2,
  "risk": ["Risk 1"],
  "success_condition": "String"
}

Allowed Transformers for `candidate_pipeline.preprocessing`:
- Log1pTransformer, SqrtTransformer, SquareTransformer, ClipTransformer, RankTransformer
- FrequencyEncoder, CategoricalPassthrough
- DatetimeExtractor, NumericInteractions, GroupAggregationTransformer
- StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, TargetEncoder, SimpleImputer

Parameter Definitions:
- DatetimeExtractor accepts booleans: extract_year, extract_month, extract_day, extract_dow, extract_hour, extract_is_weekend, extract_elapsed. DO NOT use an 'extract' list. Output columns are named `{col}_year`, `{col}_month`, `{col}_hour`, `{col}_dow`, etc. (e.g. {"extract_hour": true})
- NumericInteractions accepts: interaction_pairs: list[list[str]] (e.g. {"interaction_pairs": [["col_a", "col_b"]]})
"""

PLANNER_FEW_SHOT = """
EXAMPLE INPUT:
{"size_label": "medium", "numeric_fraction": 0.8, "categorical_fraction": 0.2, "nonlinear_signal": 0.45, "redundancy": 0.6, "missing_fraction": 0.0, "high_cardinality_fraction": 0.0, "task_type": "binary_classification"}
All Features: ["age", "income", "credit_score", "region", "tenure"]
Top Features by Mutual Information: ["age", "income", "credit_score"]

EXAMPLE OUTPUT:
{
  "hypothesis": "Create interactions between top features (age, income, credit_score) to capture non-linear combinations.",
  "evidence": ["Dataset has high nonlinear signal (0.45) and no missing values.", "Tree models benefit from explicit interaction terms."],
  "experiment_type": "feature_engineering",
  "changes": ["Apply NumericInteractions to top 3 features"],
  "candidate_pipeline": {
    "preprocessing": [
      {
        "name": "NumericInteractions",
        "columns": ["age", "income", "credit_score"],
        "params": {"interaction_pairs": [["age", "income"], ["income", "credit_score"]]}
      }
    ]
  },
  "expected_gain": 0.04,
  "expected_cost": 0.2,
  "information_value": 0.75,
  "failure_risk": 0.15,
  "risk": ["Increases dimensionality which might cause overfitting on medium size data."],
  "success_condition": "CV score improves compared to baseline without interactions."
}
"""

def build_planner_prompt(fingerprint: DatasetFingerprint, top_features: list[str], all_features: list[str]) -> str:
    """Builds the prompt string for hypothesis generation."""
    input_data = {
        "fingerprint": fingerprint.model_dump(include={
            "size_label", "numeric_fraction", "categorical_fraction", 
            "nonlinear_signal", "redundancy", "missing_fraction", 
            "high_cardinality_fraction", "task_type"
        }),
        "all_features": all_features,
        "top_features": top_features
    }
    
    return f"{PLANNER_SYSTEM_PROMPT}\n\n{PLANNER_FEW_SHOT}\n\nYOUR TASK:\nINPUT:\n{json.dumps(input_data)}\n\nOUTPUT:\n"


# ---------------------------------------------------------------------------
# Explainer Prompt (Result Interpretation)
# ---------------------------------------------------------------------------

EXPLAINER_SYSTEM_PROMPT = """You are an expert AutoML Analyst. Explain why an experiment succeeded or failed compared to its parent.
Output ONLY a short, concise paragraph of text. No JSON.
"""

def build_explainer_prompt(hypothesis_text: str, score_delta: float, runtime_delta: float) -> str:
    direction = "succeeded (improved score)" if score_delta > 0 else "failed (worsened score)"
    return (
        f"{EXPLAINER_SYSTEM_PROMPT}\n\n"
        f"Hypothesis tested: {hypothesis_text}\n"
        f"Result: The experiment {direction} by {score_delta:.4f} units.\n"
        f"Runtime impact: {runtime_delta:.2f} seconds.\n\n"
        f"Explain in 1-2 sentences why this likely happened."
    )


# ---------------------------------------------------------------------------
# Repair Prompt
# ---------------------------------------------------------------------------

REPAIR_SYSTEM_PROMPT = """You are an expert AutoML Debugger. An experiment failed with an error.
You must suggest a repair in valid JSON format.
DO NOT wrap the JSON in markdown blocks.

JSON SCHEMA:
{
  "explanation": "Why it failed",
  "repair_action": "String description of repair",
  "modified_hyperparameters": {"param_name": "new_value"}
}
"""

def build_repair_prompt(error_msg: str, traceback: str, current_config: dict[str, Any]) -> str:
    return (
        f"{REPAIR_SYSTEM_PROMPT}\n\n"
        f"Current Config:\n{json.dumps(current_config)}\n\n"
        f"Error:\n{error_msg}\n\n"
        f"OUTPUT:\n"
    )
