"""
MicroAutoML-Agent — Report Generator
Generates a standalone HTML report summarizing the AutoML run.
"""

from __future__ import annotations

import logging
from typing import Any

from automl.core.enums import MetricDirection
from automl.core.schemas import SearchState, ExperimentResult
from automl.reporting.plots import PlotGenerator

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates an HTML report string."""

    def __init__(self) -> None:
        pass

    def generate_html(
        self,
        state: SearchState,
        results: list[ExperimentResult],
        direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
        final_selection: dict[str, Any] | None = None,
        final_test_score: float | None = None
    ) -> str:
        """
        Builds the standalone HTML report.
        """
        # Generate plots
        learning_curve_b64 = PlotGenerator.plot_learning_curve(results)
        resource_b64 = PlotGenerator.plot_resource_usage(results)
        
        # Build HTML
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>MicroAutoML Report</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }",
            "h1, h2, h3 { color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }",
            ".metric { background: #f8f9fa; border-left: 4px solid #007bff; padding: 15px; margin: 10px 0; border-radius: 4px; }",
            ".metric-value { font-size: 24px; font-weight: bold; color: #007bff; }",
            "img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; padding: 5px; margin: 15px 0; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }",
            "th { background-color: #f8f9fa; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>MicroAutoML Run Report: {state.run_id}</h1>"
        ]
        
        # Summary Section
        html.append("<h2>Run Summary</h2>")
        status = state.saturation_state.value
        html.append(f"<div class='metric'>Status: <span class='metric-value'>{status}</span></div>")
        
        if final_test_score is not None:
            html.append(f"<div class='metric'>Final Test Score: <span class='metric-value'>{final_test_score:.4f}</span></div>")
            
        if final_selection:
            html.append(f"<h3>Final Selection</h3>")
            html.append(f"<p>Type: <strong>{final_selection['type']}</strong></p>")
            html.append(f"<p>ID(s): <strong>{final_selection['selection_id']}</strong></p>")
        
        # Trajectory Section
        html.append("<h2>Search Trajectory</h2>")
        if learning_curve_b64:
            html.append(f"<img src='data:image/png;base64,{learning_curve_b64}' alt='Learning Curve' />")
        else:
            html.append("<p>Not enough data for learning curve.</p>")
            
        # Resource Section
        html.append("<h2>Resource Usage</h2>")
        if resource_b64:
            html.append(f"<img src='data:image/png;base64,{resource_b64}' alt='Resource Usage' />")
        else:
            html.append("<p>Not enough data for resource plots.</p>")
            
        # Leaderboard Table
        html.append("<h2>Top 10 Models</h2>")
        html.append("<table>")
        html.append("<tr><th>Rank</th><th>ID</th><th>Operator</th><th>Score</th><th>Runtime (s)</th><th>RAM (GB)</th></tr>")
        
        from functools import cmp_to_key
        from automl.application.metric_utils import is_better
        
        def compare_results(r1, r2):
            if is_better(r1.cv_score_mean, r2.cv_score_mean, direction):
                return -1
            elif is_better(r2.cv_score_mean, r1.cv_score_mean, direction):
                return 1
            return 0
            
        valid_results = [r for r in results if r.cv_score_mean is not None and r.status == "success"]
        valid_results.sort(key=cmp_to_key(compare_results))
        
        for i, res in enumerate(valid_results[:10]):
            html.append(
                f"<tr>"
                f"<td>{i+1}</td>"
                f"<td>{res.experiment_id}</td>"
                f"<td>{res.operator.name}</td>"
                f"<td>{res.cv_score_mean:.4f}</td>"
                f"<td>{res.runtime_seconds:.1f}</td>"
                f"<td>{res.peak_ram_gb:.2f}</td>"
                f"</tr>"
            )
        html.append("</table>")
        
        # Footer
        html.append("</body></html>")
        
        return "\n".join(html)
