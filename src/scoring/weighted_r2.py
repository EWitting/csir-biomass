"""Weighted R² scorer for biomass competition."""
from typing import Any

import numpy as np

from src.scoring.scorer import Scorer


class WeightedR2(Scorer):
    """Weighted R² scorer for multi-output regression.
    
    Computes R² for each target and returns weighted average.
    Target order: [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
    Weights: [0.1, 0.1, 0.1, 0.2, 0.5]
    """

    def __init__(self, name: str = "weighted_r2") -> None:
        """Initialize the scorer."""
        super().__init__(name)
        
        # Weights for each target (in alphabetical order)
        # Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g
        self.weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float32)

    def __call__(
        self,
        y_true: np.ndarray[Any, Any],
        y_pred: np.ndarray[Any, Any],
        **kwargs: Any,
    ) -> float:
        """Calculate weighted R² score.

        :param y_true: Ground truth values (n_samples, 5)
        :param y_pred: Predicted values (n_samples, 5)
        :param kwargs: Additional arguments (ignored)
        :return: Weighted R² score
        """
        # Ensure correct shape
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        # Compute R² for each target
        r2_scores = []
        for i in range(y_true.shape[1]):
            r2 = self._compute_r2(y_true[:, i], y_pred[:, i])
            r2_scores.append(r2)
        
        r2_scores = np.array(r2_scores)
        
        # Compute weighted average
        weighted_r2 = np.sum(self.weights * r2_scores)
        
        return float(weighted_r2)
    
    def _compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² for a single target.

        :param y_true: Ground truth values
        :param y_pred: Predicted values
        :return: R² score
        """
        # Residual sum of squares
        ss_res = np.sum((y_true - y_pred) ** 2)
        
        # Total sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        # R² = 1 - (SS_res / SS_tot)
        if ss_tot == 0:
            # If all values are the same, R² is undefined; return 0
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        
        return float(r2)

