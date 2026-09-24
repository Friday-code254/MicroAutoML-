"""
MicroAutoML-Agent — Hypothesis Selector

Evaluates a HypothesisSet, ranks them by expected utility, and converts 
the top candidates into executable ExperimentSpecs.
"""

from __future__ import annotations

from typing import Any

from automl.core.schemas import ExperimentSpec, HypothesisSet, ResourceBudget
from automl.core.enums import OperatorType


class HypothesisSelector:
    """Ranks hypotheses and converts them to experiment specs."""
    
    def __init__(self) -> None:
        pass

    def select(
        self, 
        hypothesis_set: HypothesisSet, 
        top_k: int, 
        parent_id: str, 
        base_model_name: str,
        base_hyperparameters: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        explore_ratio: float = 0.5
    ) -> list[ExperimentSpec]:
        """
        Ranks the hypotheses in the set and returns up to top_k ExperimentSpecs.
        """
        if not hypothesis_set.hypotheses:
            return []
            
        history_configs = history or []
        
        # 1. Score hypotheses
        # utility = evidence_support + expected_improvement + uncertainty + novelty - compute_cost - repetition_penalty
        for hyp in hypothesis_set.hypotheses:
            # Base metrics
            evidence_support = 0.5 if hyp.evidence else 0.1
            expected_improvement = hyp.expected_gain
            compute_cost = hyp.expected_cost
            failure_risk = hyp.failure_risk
            
            # Exploration vs Exploitation balance
            # High explore_ratio boosts utility of exploratory, high-information hypotheses
            information_value = hyp.information_value * explore_ratio
            uncertainty = explore_ratio * (0.5 if hyp.source == "slm" else 0.1)
            novelty = explore_ratio * 0.5
            
            # Repetition Penalty
            repetition_penalty = 0.0
            for past_cfg in history_configs:
                # Naive similarity: same model and same operator
                if (past_cfg.get("model_name") == hyp.candidate_pipeline.get("model_name", base_model_name) and 
                    past_cfg.get("operator") == hyp.experiment_type):
                    repetition_penalty += 0.5
            
            # Full Utility Formula
            # Exploitative (improves model) vs Exploratory (teaches us about dataset)
            hyp.utility_score = (
                evidence_support 
                + expected_improvement 
                + uncertainty 
                + novelty 
                + information_value
                - compute_cost 
                - repetition_penalty
                - failure_risk
            )
            
        # 2. Sort by utility
        ranked = sorted(hypothesis_set.hypotheses, key=lambda h: h.utility_score, reverse=True)
        top_candidates = ranked[:top_k]
        
        # 3. Convert to ExperimentSpec
        specs = []
        for hyp in top_candidates:
            # We construct the pipeline by merging the hypothesis changes onto the base model
            prep = hyp.candidate_pipeline.get("preprocessing", [])
            feats = hyp.candidate_pipeline.get("features", [])
            sel = hyp.candidate_pipeline.get("feature_selection", {})
            hpos = hyp.candidate_pipeline.get("hyperparameters", base_hyperparameters or {})
            
            # Note: if it's a MODEL_CHANGE hypothesis, it might specify a different model
            model_name = hyp.candidate_pipeline.get("model_name", base_model_name)
            
            spec = ExperimentSpec(
                parent_id=parent_id,
                hypothesis_id=hyp.hypothesis_id,
                hypothesis=hyp.hypothesis,
                operator=hyp.experiment_type,
                model_name=model_name,
                preprocessing=prep,
                features=feats,
                feature_selection=sel,
                hyperparameters=hpos,
                success_condition=hyp.success_condition,
                expected_gain=hyp.expected_gain,
                resource_budget=ResourceBudget() # Default for now
            )
            specs.append(spec)
            
        return specs
