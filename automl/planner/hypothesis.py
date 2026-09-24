"""
MicroAutoML-Agent — Hypothesis Generator

Integrates deterministic rules and (optionally) the SLM to generate a set of testable 
hypotheses for the current state of the dataset.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from automl.core.schemas import EvidencePack, Hypothesis, HypothesisSet
from automl.core.config import SLMConfig
from automl.planner.rules import RuleEngine
from automl.planner.prompts import build_planner_prompt

logger = logging.getLogger(__name__)


class HypothesisGenerator:
    """Generates Hypotheses by merging Rule Engine outputs and SLM generation."""
    
    def __init__(
        self, 
        slm_config: SLMConfig, 
        slm_invoke_func: Callable[[str], str] | None = None
    ) -> None:
        self.rule_engine = RuleEngine()
        self.slm_config = slm_config
        self.slm_invoke_func = slm_invoke_func

    def generate(self, evidence: EvidencePack, round_num: int = 1) -> HypothesisSet:
        """Generates a combined set of hypotheses."""
        hypotheses: list[Hypothesis] = []
        
        # 1. Rule Engine (Always runs, extremely safe and fast)
        logger.info("Generating hypotheses via Rule Engine...")
        rule_hyps = self.rule_engine.generate_hypotheses(evidence)
        hypotheses.extend(rule_hyps)
        
        # 2. SLM Planner (If enabled and caller provided an invoke function)
        if self.slm_config.enabled and self.slm_invoke_func:
            logger.info(f"Generating hypotheses via SLM ({self.slm_config.model})...")
            try:
                # Get top features from evidence (naive heuristic for the prompt)
                top_features = []
                if evidence.deep_analysis and evidence.deep_analysis.mutual_info_to_target:
                    sorted_mi = sorted(
                        evidence.deep_analysis.mutual_info_to_target.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    top_features = [k for k, v in sorted_mi[:5]]
                    
                all_features = list(evidence.feature_map.keys())
                prompt = build_planner_prompt(evidence.fingerprint, top_features, all_features)
                
                logger.info(f"--- SLM Prompt ---\n{prompt}\n------------------")
                raw_response = self.slm_invoke_func(prompt)
                logger.info(f"--- SLM Response ---\n{raw_response}\n--------------------")
                
                # Attempt to parse
                parsed = self._parse_slm_json(raw_response)
                if parsed:
                    # Validate against schema
                    parsed["source"] = "slm"
                    hyp = Hypothesis.model_validate(parsed)
                    hypotheses.append(hyp)
                    logger.info("Successfully parsed SLM hypothesis.")
                    
            except Exception as e:
                logger.error(f"SLM hypothesis generation failed: {e}")
                # Degrade gracefully: Rule Engine hypotheses still go through.
                pass
                
        # 3. Deduplicate and return
        unique_hyps = self._deduplicate(hypotheses)
        return HypothesisSet(
            hypotheses=unique_hyps,
            planning_round=round_num
        )

    def _parse_slm_json(self, response: str) -> dict | None:
        """Safely extracts JSON from the SLM response."""
        response = response.strip()
        # Remove markdown blocks if the model ignored instructions
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
            
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return None

    def _deduplicate(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Deduplicates hypotheses that have the exact same candidate_pipeline."""
        seen_configs = set()
        unique = []
        for hyp in hypotheses:
            config_str = json.dumps(hyp.candidate_pipeline, sort_keys=True)
            if config_str not in seen_configs:
                seen_configs.add(config_str)
                unique.append(hyp)
        return unique
