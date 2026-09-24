import pandas as pd
from sklearn.datasets import make_classification
import os

from automl.application.orchestrator import RunOrchestrator
from automl.core.config import RunConfig

def main():
    print("Generating dummy dataset...")
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    
    os.makedirs("data", exist_ok=True)
    dataset_path = "data/dummy_clf.csv"
    df.to_csv(dataset_path, index=False)
    
    config = RunConfig.development()
    config.dataset_path = dataset_path
    config.target_column = "target"
    config.search.time_budget_seconds = 60  # Short budget for test
    config.search.max_repairs_per_experiment = 1
    
    orchestrator = RunOrchestrator(config)
    print("Starting Orchestrator execute...")
    orchestrator.execute()
    print("Done!")

if __name__ == "__main__":
    main()
