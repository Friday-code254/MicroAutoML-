"""
MicroAutoML-Agent — Test Set Firewall

Actively blocks the test set from reaching the ExperimentRunner.
Enforces the strict Phase 12 invariant: no test rows during search.
"""

from __future__ import annotations

import logging

import pandas as pd

from automl.core.schemas import EvidencePack
from automl.application.invariants import InvariantViolation

logger = logging.getLogger(__name__)


class TestBoundaryGuard:
    """Enforces the boundary between training and testing data."""

    @staticmethod
    def slice_for_search(X: pd.DataFrame, y: pd.Series, evidence: EvidencePack) -> tuple[pd.DataFrame, pd.Series]:
        """
        Slices the dataset to EXCLUDE test rows.
        Returns the (X_train_val, y_train_val) slice.
        Raises InvariantViolation if the test split is corrupted.
        """
        if not evidence.split:
            logger.warning("No split strategy found in EvidencePack. Assuming all data is train/val.")
            return X, y

        train_val_idx = evidence.split.train_val_indices
        test_idx = evidence.split.test_indices

        if not train_val_idx:
            raise InvariantViolation("CRITICAL: train_val_indices is empty!")

        # Verify no overlap
        overlap = set(train_val_idx).intersection(set(test_idx))
        if overlap:
            raise InvariantViolation(f"CRITICAL: {len(overlap)} rows overlap between train_val and test splits!")

        logger.debug(f"Applying TestBoundaryGuard: {len(train_val_idx)} train/val rows, {len(test_idx)} test rows.")
        return X.iloc[train_val_idx], y.iloc[train_val_idx]

    @staticmethod
    def slice_for_final_test(X: pd.DataFrame, y: pd.Series, evidence: EvidencePack) -> tuple[pd.DataFrame, pd.Series]:
        """
        Slices the dataset to ONLY include test rows for the ONE-TIME final evaluation.
        """
        if not evidence.split or not evidence.split.test_indices:
            raise InvariantViolation("CRITICAL: Cannot evaluate test set; no test indices found.")

        test_idx = evidence.split.test_indices
        logger.info(f"Extracting test slice: {len(test_idx)} rows.")
        return X.iloc[test_idx], y.iloc[test_idx]

