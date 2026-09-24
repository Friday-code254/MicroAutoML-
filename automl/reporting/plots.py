"""
MicroAutoML-Agent — Reporting Plots
Generates matplotlib/seaborn visualizations for the HTML report.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from automl.core.schemas import SearchState, ExperimentResult

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generates base64-encoded PNG plots from experiment data."""

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        """Converts a matplotlib figure to a base64 string and closes the figure."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    @classmethod
    def plot_learning_curve(cls, results: list[ExperimentResult]) -> str | None:
        """Plots CV Score vs Experiment Timeline."""
        if not results:
            return None
            
        valid_results = [r for r in results if r.status == "success" and r.cv_score_mean is not None]
        if not valid_results:
            return None
            
        # Sort by completion time (or just rely on list order if timestamps are missing)
        def get_time(r): return r.completed_at.timestamp() if r.completed_at else 0
        valid_results.sort(key=get_time)
        
        scores = [r.cv_score_mean for r in valid_results]
        best_scores = np.maximum.accumulate(scores) # assuming HIGHER_IS_BETTER for plot
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(scores) + 1), scores, 'o-', alpha=0.3, label="Experiment Score")
        ax.plot(range(1, len(scores) + 1), best_scores, 'r-', linewidth=2, label="Best So Far")
        ax.set_title("Search Trajectory")
        ax.set_xlabel("Experiment Count")
        ax.set_ylabel("CV Score")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return cls._fig_to_base64(fig)

    @classmethod
    def plot_resource_usage(cls, results: list[ExperimentResult]) -> str | None:
        """Plots RAM and Runtime distributions."""
        valid_results = [r for r in results if r.status == "success"]
        if not valid_results:
            return None
            
        runtimes = [r.runtime_seconds for r in valid_results]
        rams = [r.peak_ram_gb for r in valid_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        sns.histplot(runtimes, kde=True, ax=ax1, color="blue")
        ax1.set_title("Runtime Distribution (s)")
        ax1.set_xlabel("Seconds")
        
        sns.histplot(rams, kde=True, ax=ax2, color="green")
        ax2.set_title("Peak RAM Distribution (GB)")
        ax2.set_xlabel("Gigabytes")
        
        plt.tight_layout()
        return cls._fig_to_base64(fig)
        
    @classmethod
    def plot_diversity_heatmap(cls, correlation_matrix: np.ndarray, labels: list[str]) -> str | None:
        """Plots the OOF prediction correlation heatmap."""
        if correlation_matrix is None or len(labels) == 0:
            return None
            
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1, 
                    xticklabels=labels, yticklabels=labels, ax=ax, fmt=".2f")
        ax.set_title("Model Diversity (Prediction Correlation)")
        plt.tight_layout()
        
        return cls._fig_to_base64(fig)
