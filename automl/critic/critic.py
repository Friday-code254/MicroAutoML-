"""
MicroAutoML-Agent — Numerical Critic
Analyzes experiment results deterministically for overfitting, instability, and resource hogs.
"""

from __future__ import annotations

import logging
from typing import Callable

from automl.core.config import SLMConfig
from automl.core.schemas import CriticReport, ExperimentResult, MetricDirection
from automl.critic.stability import StabilityAnalyzer

logger = logging.getLogger(__name__)


class NumericalCritic:
    """Deterministically inspects results to identify structural flaws."""

    def __init__(
        self,
        slm_config: SLMConfig | None = None,
        slm_invoke_func: Callable[[str], str] | None = None,
        metric_direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
        overfit_threshold: float = 0.10,
        memory_heavy_gb: float = 4.0,
        slow_seconds: float = 300.0,
    ) -> None:
        self.slm_config = slm_config
        self.slm_invoke_func = slm_invoke_func
        self.metric_direction = metric_direction
        self.overfit_threshold = overfit_threshold
        self.memory_heavy_gb = memory_heavy_gb
        self.slow_seconds = slow_seconds
        
        self.stability = StabilityAnalyzer(metric_direction=metric_direction)

    def analyze(self, result: ExperimentResult) -> CriticReport:
        """Analyzes a single result and generates a CriticReport."""
        report = CriticReport(experiment_id=result.experiment_id)
        
        if result.status != "success":
            # Failed experiments don't get numerical critique
            report.critic_flags.append("FAILED")
            return report

        # 1. Stability Check (Variance across folds)
        is_stable, stability_flags = self.stability.analyze_folds(result.fold_scores)
        report.is_stable = is_stable
        report.critic_flags.extend(stability_flags)
        
        # 2. Overfit Check
        if result.train_score is not None and result.cv_score_mean is not None:
            # Overfit ratio = (Train - Val) / Train
            if self.metric_direction == MetricDirection.HIGHER_IS_BETTER:
                train_val_gap = result.train_score - result.cv_score_mean
                # Handle edge cases where train score is 0 or negative
                if result.train_score > 0:
                    report.overfit_ratio = train_val_gap / result.train_score
            else: # LOWER_IS_BETTER
                train_val_gap = result.cv_score_mean - result.train_score
                if result.cv_score_mean > 0:
                    report.overfit_ratio = train_val_gap / result.cv_score_mean
                    
            if report.overfit_ratio > self.overfit_threshold:
                report.is_overfitting = True
                report.critic_flags.append("OVERFITTING")
                
        # 3. Resource Check
        if result.peak_ram_gb > self.memory_heavy_gb:
            report.is_memory_heavy = True
            report.critic_flags.append("MEMORY_HEAVY")
            
        if result.runtime_seconds > self.slow_seconds:
            report.is_slow = True
            report.critic_flags.append("SLOW")

        # 4. LLM Reflection (Optional)
        if self.slm_config and self.slm_invoke_func:
            report.reflection = self._generate_reflection(result, report)

        return report

    def _generate_reflection(self, result: ExperimentResult, report: CriticReport) -> str:
        """Invokes the SLM to explain *why* the hypothesis succeeded or failed."""
        prompt = f"""
        Analyze this experiment result:
        Experiment ID: {result.experiment_id}
        Parent ID: {result.parent_id}
        CV Score: {result.cv_score_mean} +/- {result.cv_score_std}
        Flags: {report.critic_flags}
        
        Why did this happen? What evidence supports it? What should be tested next?
        """
        try:
            return self.slm_invoke_func(prompt)
        except Exception as e:
            logger.error(f"Failed to generate reflection: {e}")
            return "Reflection failed."
