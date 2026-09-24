"""
MicroAutoML-Agent — Saturation Detector
Monitors search progress and detects when the search has plateaued or exhausted its budget.
"""

from __future__ import annotations

from automl.core.enums import SaturationState
from automl.core.schemas import MetricDirection, SearchState
from automl.search.tree import ExperimentTree


class SaturationDetector:
    """Evaluates whether the search should transition to SATURATING or LOCKED state."""

    def __init__(
        self,
        plateau_patience: int = 5,
        min_gain_threshold: float = 0.001,
        budget_warning_pct: float = 0.10,
        metric_direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    ) -> None:
        self.plateau_patience = plateau_patience
        self.min_gain_threshold = min_gain_threshold
        self.budget_warning_pct = budget_warning_pct
        self.metric_direction = metric_direction

    def check_saturation(
        self, 
        state: SearchState, 
        tree: ExperimentTree
    ) -> SaturationState:
        """
        Evaluate current state and tree to determine saturation level.
        Returns the new SaturationState.
        """
        # 1. Check strict budget limits (LOCKED)
        if state.budget_fraction_remaining <= 0.0:
            return SaturationState.BUDGET_EXHAUSTED
            
        # 2. Check plateau condition (SATURATING)
        # We track `consecutive_no_gain` in the SearchState. 
        # If the search has gone `plateau_patience` experiments without 
        # improving the incumbent by at least `min_gain_threshold`, we are saturating.
        if state.consecutive_no_gain >= self.plateau_patience:
            # If we're already saturating and we still have no gain for another patience cycle, we lock.
            # E.g. patience=5. At 5 we saturate. At 10 we lock.
            if state.consecutive_no_gain >= (self.plateau_patience * 2):
                return SaturationState.SATURATED
            return SaturationState.PLATEAU_DETECTED
            
        # 3. Check nearing budget end (SATURATING)
        if state.budget_fraction_remaining <= self.budget_warning_pct:
            return SaturationState.PLATEAU_DETECTED
            
        return SaturationState.SEARCHING
