"""
Phase 5 Tests — Planner (Rule Engine + SLM + Hypothesis Generator)

Tests deterministic hypothesis generation and the parsing of SLM responses,
as well as the ranking and conversion logic of the HypothesisSelector.
"""

from __future__ import annotations

import json
import pytest

from automl.core.schemas import EvidencePack, FeatureInfo, TaskProfile, Hypothesis
from automl.core.enums import FeatureDtype, TaskType, OperatorType
from automl.core.config import SLMConfig
from automl.planner.rules import RuleEngine
from automl.planner.prompts import build_planner_prompt, build_explainer_prompt, build_repair_prompt
from automl.planner.hypothesis import HypothesisGenerator
from automl.planner.selector import HypothesisSelector


class TestRuleEngine:
    def test_missing_numeric_rule(self) -> None:
        evidence = EvidencePack()
        evidence.feature_map["age"] = FeatureInfo(
            feature="age", dtype=FeatureDtype.NUMERICAL, missing_ratio=0.1
        )
        
        engine = RuleEngine()
        hyps = engine.generate_hypotheses(evidence)
        
        # Should generate numeric imputation rule
        assert any("SimpleImputer" in h.changes[0] and "median" in h.changes[0] for h in hyps)
        
    def test_high_cardinality_rule(self) -> None:
        evidence = EvidencePack()
        evidence.task = TaskProfile(task_type=TaskType.BINARY_CLASSIFICATION)
        evidence.feature_map["zipcode"] = FeatureInfo(
            feature="zipcode", dtype=FeatureDtype.CATEGORICAL, cardinality=100
        )
        
        engine = RuleEngine()
        hyps = engine.generate_hypotheses(evidence)
        
        # Should generate TargetEncoder rule
        assert any("TargetEncoder" in h.changes[0] for h in hyps)
        
    def test_skew_rule(self) -> None:
        evidence = EvidencePack()
        evidence.feature_map["income"] = FeatureInfo(
            feature="income", dtype=FeatureDtype.NUMERICAL, skew=2.5
        )
        
        engine = RuleEngine()
        hyps = engine.generate_hypotheses(evidence)
        
        assert any("Log1pTransformer" in h.changes[0] for h in hyps)


class TestPrompts:
    def test_planner_prompt(self) -> None:
        evidence = EvidencePack()
        prompt = build_planner_prompt(evidence.fingerprint, ["feat1", "feat2"])
        assert "feat1" in prompt
        assert "JSON SCHEMA" in prompt
        
    def test_explainer_prompt(self) -> None:
        prompt = build_explainer_prompt("Test hypothesis", 0.05, 1.2)
        assert "succeeded" in prompt
        assert "1-2 sentences" in prompt
        
    def test_repair_prompt(self) -> None:
        prompt = build_repair_prompt("KeyError: 'foo'", "traceback string", {"model": "RF"})
        assert "KeyError" in prompt
        assert "JSON SCHEMA" in prompt


class TestHypothesisGenerator:
    def test_generator_with_rule_engine(self) -> None:
        evidence = EvidencePack()
        evidence.feature_map["age"] = FeatureInfo(
            feature="age", dtype=FeatureDtype.NUMERICAL, missing_ratio=0.1
        )
        config = SLMConfig(enabled=False)
        generator = HypothesisGenerator(slm_config=config)
        
        hset = generator.generate(evidence, round_num=1)
        assert len(hset.hypotheses) > 0
        assert hset.planning_round == 1
        assert hset.hypotheses[0].source == "rules"
        
    def test_generator_with_slm(self) -> None:
        evidence = EvidencePack()
        config = SLMConfig(enabled=True)
        
        # Mock SLM response
        mock_json = {
            "hypothesis": "SLM test",
            "evidence": [],
            "experiment_type": "feature_engineering",
            "changes": [],
            "candidate_pipeline": {"model_name": "SLM_Mock_Model"},
            "expected_gain": 0.1,
            "expected_cost": 0.1,
            "risk": [],
            "success_condition": ""
        }
        
        def mock_invoke(prompt: str) -> str:
            return f"```json\n{json.dumps(mock_json)}\n```"
            
        generator = HypothesisGenerator(slm_config=config, slm_invoke_func=mock_invoke)
        
        # We temporarily disable the blanket catch in test to see the error
        raw_response = mock_invoke("")
        parsed = generator._parse_slm_json(raw_response)
        parsed["source"] = "slm"
        try:
            h = Hypothesis.model_validate(parsed)
        except Exception as e:
            pytest.fail(f"Validation failed: {e}")
            
        hset = generator.generate(evidence)
        
        assert any(h.source == "slm" for h in hset.hypotheses)


class TestHypothesisSelector:
    def test_selector_ranking_and_conversion(self) -> None:
        evidence = EvidencePack()
        evidence.feature_map["age"] = FeatureInfo(
            feature="age", dtype=FeatureDtype.NUMERICAL, missing_ratio=0.1
        )
        
        config = SLMConfig(enabled=False)
        generator = HypothesisGenerator(slm_config=config)
        hset = generator.generate(evidence)
        
        # Inject custom expected_gain/costs to test ranking
        if len(hset.hypotheses) >= 1:
            hset.hypotheses[0].expected_gain = 0.5
            hset.hypotheses[0].expected_cost = 0.1 # Utility = 5.0
            
        selector = HypothesisSelector()
        specs = selector.select(
            hypothesis_set=hset, 
            top_k=2, 
            parent_id="EXP_BASE",
            base_model_name="RandomForestClassifier"
        )
        
        assert len(specs) > 0
        assert specs[0].parent_id == "EXP_BASE"
        assert specs[0].model_name == "RandomForestClassifier"
        assert specs[0].expected_gain == 0.5
