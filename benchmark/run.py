"""
MicroAutoML-Agent — Benchmark Runner
Executes fixed datasets against standard baselines to obtain real numbers.
"""

import logging
import time
import argparse
from pathlib import Path

from automl.application.orchestrator import RunOrchestrator
from automl.core.config import RunConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_benchmark(dataset_path: str, target_column: str, budget_seconds: int = 3600, disable_saturation: bool = False):
    """Executes a real benchmark run on a dataset."""
    print(f"\n{'='*50}")
    print(f"=== Benchmarking {dataset_path} ===")
    print(f"{'='*50}\n")
    
    config = RunConfig.cpu_default()
    config.dataset_path = Path(dataset_path)
    config.target_column = target_column
    config.search.time_budget_seconds = budget_seconds
    
    if disable_saturation:
        config.search.plateau_window = 9999
        config.search.min_gain_threshold = -1.0
        print("NOTE: Saturation logic (plateau window) is disabled for this run.\n")
        
    orchestrator = RunOrchestrator(config)
    
    t0 = time.perf_counter()
    try:
        final_score = orchestrator.execute()
    except Exception as e:
        print(f"CRITICAL BENCHMARK FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
    elapsed = time.perf_counter() - t0
    
    print(f"\n{'='*50}")
    print(f"=== Benchmark Complete ===")
    print(f"Dataset: {dataset_path}")
    print(f"Target: {target_column}")
    print(f"Final Test Score: {final_score:.4f}")
    print(f"Elapsed Time: {elapsed:.1f}s")
    print(f"{'='*50}\n")
    
    return final_score, elapsed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str, help="Path to the dataset CSV")
    parser.add_argument("--target", required=True, type=str, help="Target column name")
    parser.add_argument("--budget", type=int, default=1800, help="Time budget in seconds")
    parser.add_argument("--disable-saturation", action="store_true", help="Force the search to use the full time budget")
    args = parser.parse_args()
    
    run_benchmark(args.data, args.target, args.budget, args.disable_saturation)
