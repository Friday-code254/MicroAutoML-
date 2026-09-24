# MicroAutoML Research Protocol

This document defines the formal research protocol used to evaluate the MicroAutoML framework. 
To ensure fair, reproducible benchmarking, all components and rules described here are permanently frozen for the duration of the evaluation.

## 1. Research Questions (RQs)

1. **RQ1**: Can an evidence-driven LLM planner discover higher-performing pipelines than Random Search or fixed portfolios under the same compute budget?
2. **RQ2**: Does adaptive hypothesis selection (balancing expected gain, exploration, and cost) reach saturation with fewer experiments?
3. **RQ3**: Does multi-fidelity execution (allocating cheap runs to speculative ideas and full runs to mature ones) yield a higher score-per-unit-compute?
4. **RQ4**: Does the self-healing Critic Engine successfully rescue experiments that would otherwise crash (e.g., OOM), allowing the search to explore more complex state spaces safely?
5. **RQ5**: Can the Meta-Prior engine accelerate search convergence on new datasets by leveraging memory from past similar dataset fingerprints?

## 2. Benchmark Rules

- **Compute Budget**: Exactly 3600 seconds (1 Hour) of wall-clock time per dataset.
- **Hardware**: All benchmarks must be run on identical hardware specs (documented in the resulting metadata).
- **Data Splitting**: A fixed, pre-allocated 80/20 train/test split. The 20% test set is absolutely untouchable by the framework until `SEARCH_LOCKED`.
- **Random Seeds**: The global seed must be fixed at `42` for all random operations (splitting, initialization, sampling).
- **SLM Backend**: The evaluation uses a fixed Local/API backend (e.g., Qwen3-4B or Gemini Flash) for the LLM Planner.

## 3. Ablation Configurations

The core benchmark script (`benchmark/run.py`) executes the system under the following ablations:

- **Config A**: Random Search (No Planning, No Rules).
- **Config B**: Planner Active, but Multi-Fidelity disabled (all runs are FULL).
- **Config C**: Multi-Fidelity Active, but Self-Healing disabled (crashing models are pruned immediately).
- **Config D**: Full MicroAutoML Framework.

## 4. Evaluation Datasets

The framework will be evaluated on the following open-source benchmark datasets:
- **Classification**: Adult Income (Binary), Titanic (Binary/Imbalanced), Iris (Multiclass).
- **Regression**: California Housing, Bike Sharing.
