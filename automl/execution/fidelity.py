"""
MicroAutoML-Agent — Fidelity Manager

Translates the declarative Fidelity enum into concrete runtime constraints:
- CV folds
- Dataset downsampling fraction
"""

from __future__ import annotations

from dataclasses import dataclass
from automl.core.enums import Fidelity


@dataclass
class FidelityConstraints:
    """Concrete execution constraints for a given fidelity level."""
    cv_folds: int
    sample_fraction: float
    use_holdout: bool = False


class FidelityManager:
    """Maps Fidelity enums to execution constraints, using RunConfig when available."""
    
    def __init__(self, run_config: "RunConfig | None" = None) -> None:
        # Pull fractions and folds from config if available; else use safe defaults.
        if run_config is not None:
            fc = run_config.fidelity
            smoke_frac = fc.smoke_fraction
            cheap_frac = fc.cheap_fraction
            medium_frac = fc.medium_fraction
            full_frac = fc.full_fraction
            cheap_folds = fc.cheap_cv_folds
            medium_folds = fc.medium_cv_folds
            full_folds = fc.full_cv_folds
        else:
            smoke_frac, cheap_frac, medium_frac, full_frac = 0.05, 0.30, 0.70, 1.0
            cheap_folds, medium_folds, full_folds = 2, 3, 5

        self._map = {
            Fidelity.SMOKE: FidelityConstraints(cv_folds=1, sample_fraction=smoke_frac, use_holdout=True),
            Fidelity.CHEAP: FidelityConstraints(cv_folds=cheap_folds, sample_fraction=cheap_frac, use_holdout=(cheap_folds == 1)),
            Fidelity.MEDIUM: FidelityConstraints(cv_folds=medium_folds, sample_fraction=medium_frac),
            Fidelity.FULL: FidelityConstraints(cv_folds=full_folds, sample_fraction=full_frac),
        }

    def get_constraints(self, fidelity: Fidelity) -> FidelityConstraints:
        """Returns constraints for the given fidelity level."""
        if fidelity not in self._map:
            # Fallback to CHEAP if unknown
            return self._map[Fidelity.CHEAP]
        return self._map[fidelity]
