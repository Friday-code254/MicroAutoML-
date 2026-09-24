"""
MicroAutoML-Agent — OOF Predictor
Aggregates Out-Of-Fold predictions from cross-validation to support ensemble blending.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class OOFPredictor:
    """Manages the generation and alignment of out-of-fold predictions."""

    @staticmethod
    def aggregate_oof_predictions(
        n_samples: int,
        fold_indices: list[tuple[np.ndarray, np.ndarray]],
        fold_predictions: list[np.ndarray]
    ) -> np.ndarray:
        """
        Reconstructs a single out-of-fold prediction array aligned to the original training data indices.
        
        Args:
            n_samples: Total number of samples in the original training set.
            fold_indices: List of (train_idx, val_idx) arrays from the CV splitter.
            fold_predictions: List of prediction arrays corresponding to each val_idx.
            
        Returns:
            np.ndarray of shape (n_samples,) or (n_samples, n_classes) containing OOF predictions.
        """
        if not fold_indices or not fold_predictions:
            raise ValueError("fold_indices and fold_predictions must not be empty.")
            
        if len(fold_indices) != len(fold_predictions):
            raise ValueError("Number of fold indices must match number of fold predictions.")

        # Inspect the shape of the first prediction to determine output shape
        first_pred = fold_predictions[0]
        
        # Force dtype to float to prevent fractional averages from being truncated to int
        if first_pred.ndim > 1:
            oof = np.zeros((n_samples, first_pred.shape[1]), dtype=float)
        else:
            oof = np.zeros(n_samples, dtype=float)
            
        counts = np.zeros(n_samples, dtype=int)

        for (_, val_idx), preds in zip(fold_indices, fold_predictions):
            if len(val_idx) != len(preds):
                raise ValueError(f"Mismatched lengths in fold: len(val_idx)={len(val_idx)}, len(preds)={len(preds)}")
            
            # Accumulate (in case of overlapping folds like RepeatedKFold)
            oof[val_idx] += preds
            counts[val_idx] += 1
            
        # Average where counts > 1, leave 0 where count == 0 (though ideally CV covers all samples)
        valid = counts > 0
        if first_pred.ndim > 1:
            oof[valid] = oof[valid] / counts[valid][:, None]
        else:
            oof[valid] = oof[valid] / counts[valid]
            
        # Fill any unpredicted samples with mean (fallback)
        missing = counts == 0
        if np.any(missing):
            logger.warning(f"{np.sum(missing)} samples missing from OOF predictions. Filling with mean.")
            if first_pred.ndim > 1:
                oof[missing] = np.mean(oof[valid], axis=0)
            else:
                oof[missing] = np.mean(oof[valid])

        return oof
