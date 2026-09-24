# MicroAutoML-Agent (v0.1)

MicroAutoML is an autonomous, agentic machine learning framework designed for tabular datasets. It replaces brute-force hyperparameter optimization (HPO) with a reasoning-driven loop powered by Small Language Models (SLMs) and deterministic rule engines.

## Features

- **Agentic Planning**: Analyzes data and formulates explicit, testable hypotheses (e.g., "The target is highly skewed; applying Log1p to continuous features should stabilize variance"). Uses local SLMs via Ollama to generate feature engineering ideas dynamically.
- **Multi-Fidelity Execution**: Tests risky hypotheses on tiny data slices first, saving compute for proven configurations.
- **Self-Healing Pipelines**: Automatically traps errors and rewrites the experiment config on the fly. Dynamically repairs LLM hallucinations (e.g. invalid keyword arguments) without crashing.
- **Resource Governance**: Hard-enforced limits on CPU threads and memory using `threadpoolctl` and process governance.
- **Crash Recovery**: Fully backed by an atomic SQLite memory store. Interrupt a run and resume exactly where it left off.
- **Zero Leakage**: A strict firewall prevents any test-set inspection until the final model is explicitly selected and the search is locked. The final packaged `Pipeline` is physically saved as a `joblib` `.pkl` file.

## Quickstart

### Prerequisites

You must have `uv` (the fast Python package installer) and `ollama` installed on your machine.
Ensure you pull a local LLM model for the agent to use:
```bash
ollama pull qwen:0.5b
# or 
ollama pull qwen2.5:3b
```

### Installation

```bash
git clone https://github.com/yourusername/MicroAutoML.git
cd MicroAutoML
uv sync
```

### Usage

Run the fully autonomous engine:

```bash
microautoml run --data data/Churn.csv --target Churn --budget 3600 --memory-budget 8.0
```

The system will:
1. Profile the dataset and split it strictly into train/val/test folds.
2. Run baseline models (Dummy, LogisticRegression, HistGradientBoosting, XGBoost, CatBoost).
3. Connect to Ollama and iteratively generate/evaluate hypotheses on feature engineering.
4. Stop when the search space saturates.
5. Lock the search, evaluate the test set once, and save `final_pipeline.pkl` into the `runs/` directory.

Resume a paused run:
```bash
microautoml resume --run-id RUN_XXX
```

## Architecture

MicroAutoML is composed of 11 fully isolated phases, ranging from Data Profiling to Final Model Packaging.
The architecture explicitly defends against Data Leakage through the `FinalHoldoutEvaluator` which locks the holdout test set until the agent formally completes its search loop.

## License
MIT
