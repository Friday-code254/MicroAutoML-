"""
MicroAutoML-Agent — Real-time Leaderboard
Uses `rich` to draw a dynamic table of experiments.
"""

from __future__ import annotations

import logging
from typing import Any

from rich.console import Console
from rich.table import Table

from automl.core.enums import MetricDirection
from automl.core.schemas import ExperimentResult

logger = logging.getLogger(__name__)


class Leaderboard:
    """Manages the terminal leaderboard."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def generate_table(self, results: list[ExperimentResult], title: str = "Search Leaderboard", direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER) -> Table:
        """
        Builds a rich Table of the best experiments.
        """
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("ID", style="blue")
        table.add_column("Operator", style="green")
        table.add_column("CV Score", justify="right")
        table.add_column("Runtime (s)", justify="right")
        table.add_column("RAM (GB)", justify="right")

        from functools import cmp_to_key
        from automl.application.metric_utils import is_better
        
        def compare_results(r1, r2):
            if is_better(r1.cv_score_mean, r2.cv_score_mean, direction):
                return -1 # r1 goes before r2
            elif is_better(r2.cv_score_mean, r1.cv_score_mean, direction):
                return 1
            return 0
            
        valid_results = [r for r in results if r.cv_score_mean is not None and r.status == "success"]
        valid_results.sort(key=cmp_to_key(compare_results))

        for i, res in enumerate(valid_results):
            # Only show top 10
            if i >= 10:
                break
                
            table.add_row(
                str(i + 1),
                res.experiment_id,
                res.operator.name if res.operator else "UNKNOWN",
                f"{res.cv_score_mean:.4f} ± {res.cv_score_std:.4f}" if res.cv_score_std else f"{res.cv_score_mean:.4f}",
                f"{res.runtime_seconds:.1f}",
                f"{res.peak_ram_gb:.2f}"
            )
            
        return table

    def print(self, results: list[ExperimentResult], direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER) -> None:
        """Prints the leaderboard to the terminal."""
        table = self.generate_table(results, direction=direction)
        self.console.print(table)
