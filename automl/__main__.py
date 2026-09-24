"""
MicroAutoML-Agent — CLI Entrypoint
Provides the `microautoml` command-line interface.
"""

import click
import logging
import sys
import warnings

# Suppress noisy scikit-learn warnings that bypass logging
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Basic logging setup for CLI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

@click.group()
def cli():
    """MicroAutoML Autonomous Search Engine."""
    pass

@cli.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="Path to training data (CSV)")
@click.option("--target", required=True, help="Target column name")
def profile(data, target):
    """Run data intelligence only."""
    click.echo(f"Profiling {data} for target {target}...")
    # Hook to Phase 1 data profiler
    click.echo("Done (Mock).")

@cli.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="Path to training data (CSV)")
@click.option("--target", required=True, help="Target column name")
@click.option("--budget", type=int, default=3600, help="Max experiments budget in seconds")
@click.option("--memory-budget", type=float, default=16.0, help="Max RAM (GB)")
def run(data, target, budget, memory_budget):
    """Execute the full autonomous run."""
    click.echo(f"Running autonomous search on {data}")
    click.echo(f"Target: {target} | Budget: {budget}s | Memory: {memory_budget}GB")
    
    from automl.core.config import RunConfig
    from automl.application.orchestrator import RunOrchestrator
    
    from pathlib import Path
    
    config = RunConfig()
    config.dataset_path = Path(data)
    config.target_column = target
    config.search.time_budget_seconds = budget
    config.resource.total_ram_gb = memory_budget
    
    try:
        orchestrator = RunOrchestrator(config)
        orchestrator.execute()
        click.echo("Search completed successfully.")
    except Exception as e:
        click.echo(f"Run failed: {e}")
        sys.exit(1)

@cli.command()
@click.option("--run-id", required=True, help="Run ID to resume")
@click.option("--data", required=True, type=click.Path(exists=True), help="Path to training data (CSV)")
def resume(run_id, data):
    """Resume a search from a SQLite checkpoint."""
    click.echo(f"Resuming run {run_id} from database...")
    from automl.core.config import RunConfig
    from automl.application.orchestrator import RunOrchestrator
    
    from pathlib import Path
    
    config = RunConfig.development()
    config.dataset_path = Path(data)
    
    try:
        orchestrator = RunOrchestrator(config)
        orchestrator.resume(run_id)
        click.echo("Resume completed successfully.")
    except Exception as e:
        click.echo(f"Resume failed: {e}")
        sys.exit(1)

@cli.command()
def benchmark():
    """Execute the standard benchmark suite."""
    click.echo("Running benchmark suite...")
    # This would import benchmark.run
    click.echo("Benchmark completed (Mock).")

@cli.command()
@click.option("--run-id", required=True, help="Run ID to inspect")
def report(run_id):
    """Regenerate the HTML report for a run."""
    click.echo(f"Generating report for {run_id}...")
    # Would pull state and results from Memory store and call ReportGenerator
    click.echo(f"Report saved to report_{run_id}.html (Mock)")

if __name__ == "__main__":
    cli()
