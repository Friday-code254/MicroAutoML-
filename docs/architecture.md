# MicroAutoML Architecture

MicroAutoML is an autonomous machine learning system designed around an agentic execution loop. It combines Large Language Models (LLMs), deterministic rule engines, multi-fidelity resource constraints, and a self-healing critic to autonomously discover high-performing machine learning pipelines.

## The 12-Phase Execution Loop

The system operates strictly across 12 isolated phases:

1. **Phase 0: Core Infrastructure**: Enums, schemas, and configurations.
2. **Phase 1: Data Intelligence (Profiler)**: Calculates dataset fingerprints, checks class imbalance, and runs a strict leakage auditor to prevent data leaks.
3. **Phase 2: Metrics & Baselines**: Selects the optimization metric and runs simple dummy baselines to anchor the search.
4. **Phase 3: Feature Engine**: Registers robust transformers (e.g., Target Encoding, Log1p) ensuring all transformations occur securely within cross-validation boundaries.
5. **Phase 4: Component Registry & Compiler**: Converts abstract hypotheses into valid `sklearn.Pipeline` objects, strictly filtering out unapproved operators.
6. **Phase 5: LLM Planner & Hypothesis Selector**: Uses a dual-engine (deterministic rules + LLM) to generate hypotheses. A utility function selects the next experiment balancing exploration, expected gain, and compute cost.
7. **Phase 6: Multi-Fidelity Runner**: Safely executes untrusted pipelines within isolated subprocesses to enforce strict RAM and Time budgets, managing folds based on the assigned fidelity level.
8. **Phase 7: Search Controller**: Manages the Experiment DAG (Directed Acyclic Graph), prunes failing branches, and halts the search when saturation is detected.
9. **Phase 8: Critic & Self-Healing**: Analyzes fold stability and automatically classifies and repairs failed experiments (e.g., reducing `n_estimators` on Memory Errors).
10. **Phase 9: Experiment Memory**: Backs all states and checkpoints to SQLite, allowing runs to be paused, resumed, and aggregated to build a cross-dataset Meta-Prior.
11. **Phase 10: Ensemble & Selection**: Combines diverse candidates using Out-Of-Fold (OOF) prediction correlation, evaluates against a strictly protected holdout set, and packages the final model for independent deployment.
12. **Phase 11 & 12: UI and Reporting**: Generates a rich CLI, standalone HTML reports, and executes benchmark/ablation protocols.

## Design Invariants

- **No Executable LLM Code**: The compiler strictly builds pipelines from a whitelisted registry. The LLM only outputs JSON configurations.
- **Zero Test Leakage**: The test set is locked until `SEARCH_LOCKED=True`. It is evaluated exactly once.
- **Hard Resource Limits**: Subprocesses enforce `timeout` and `max_memory` via OS-level signals.
