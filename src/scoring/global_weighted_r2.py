"""Global Weighted R² scorer for biomass competition.

This implements the actual competition metric where all samples are combined
into a single list with per-row weights, as opposed to computing R² per target
and averaging.
"""
from typing import Any, Optional
import inspect

import numpy as np
import pandas as pd

from src.scoring.scorer import Scorer


class GlobalWeightedR2(Scorer):
    """Global Weighted R² scorer for multi-output regression.
    
    Computes a single global R² where all (image, target) pairs are in one list,
    each weighted by its target type. This matches the actual competition metric.
    
    Target order: [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
    Weights: [0.1, 0.1, 0.1, 0.2, 0.5]
    
    When verbose=True, also logs:
    - Old per-target averaged R²
    - Global R² broken down by target, species, and state
    """

    def __init__(
        self,
        name: str = "global_weighted_r2",
        verbose: bool = True,
        metadata: Optional[pd.DataFrame] = None,
        metadata_path: Optional[str] = None,
        log_old_metric: bool = True,
    ) -> None:
        """Initialize the scorer.
        
        :param name: Name of the scorer (for the NEW global metric)
        :param verbose: If True, log detailed breakdowns
        :param metadata: DataFrame with columns: State, Species (indexed by sample)
        :param metadata_path: Path to train.csv to load metadata (if metadata not provided)
        :param log_old_metric: If True, also log old per-target averaged R² as "weighted_r2"
        """
        super().__init__(name)
        
        # Weights for each target (in alphabetical order)
        self.target_names = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'GDM_g', 'Dry_Total_g']
        self.weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float32)
        
        self.verbose = verbose
        self.log_old_metric = log_old_metric
        
        # Load metadata if path provided
        if metadata is None and metadata_path is not None:
            from pathlib import Path
            from src.setup.setup_data import setup_metadata
            metadata = setup_metadata(Path(metadata_path))
        
        self.metadata = metadata
        
        # For accessing parent frame to log (hacky but user approved!)
        self._logger = None

    def __call__(
        self,
        y_true: np.ndarray[Any, Any],
        y_pred: np.ndarray[Any, Any],
        **kwargs: Any,
    ) -> float:
        """Calculate global weighted R² score.

        :param y_true: Ground truth values (n_samples, 5)
        :param y_pred: Predicted values (n_samples, 5)
        :param kwargs: Additional arguments (can include 'epoch', 'indices', 'logger')
        :return: Global weighted R² score
        """
        # Extract metadata from kwargs
        epoch = kwargs.get('epoch', None)
        indices = kwargs.get('indices', None)
        logger = kwargs.get('logger', None)
        
        # Ensure correct shape
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        # Compute global R²
        global_r2 = self._compute_global_r2(y_true, y_pred)
        
        # If verbose, compute and log detailed breakdowns
        if self.verbose and logger is not None:
            self._log_verbose_metrics(y_true, y_pred, global_r2, epoch, indices, logger)
        
        return float(global_r2)
    
    def _compute_global_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute global weighted R².
        
        All (sample, target) pairs are combined into one list, each weighted
        by its target type.
        
        :param y_true: Ground truth values (n_samples, n_targets)
        :param y_pred: Predicted values (n_samples, n_targets)
        :return: Global R² score
        """
        n_samples, n_targets = y_true.shape
        
        # Flatten to one long list
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Create per-row weights (repeat weights for each sample)
        weights_flat = np.tile(self.weights, n_samples)
        
        # Compute weighted global mean
        weighted_mean = np.sum(weights_flat * y_true_flat) / np.sum(weights_flat)
        
        # Compute weighted SS_tot and SS_res
        ss_tot = np.sum(weights_flat * (y_true_flat - weighted_mean) ** 2)
        ss_res = np.sum(weights_flat * (y_true_flat - y_pred_flat) ** 2)
        
        # R² = 1 - (SS_res / SS_tot)
        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        
        return float(r2)
    
    def _compute_per_target_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute R² for each target separately (old method).
        
        :param y_true: Ground truth values (n_samples, n_targets)
        :param y_pred: Predicted values (n_samples, n_targets)
        :return: Tuple of (per-target R² array, weighted average)
        """
        r2_scores = []
        for i in range(y_true.shape[1]):
            r2 = self._compute_r2(y_true[:, i], y_pred[:, i])
            r2_scores.append(r2)
        
        r2_scores = np.array(r2_scores)
        weighted_avg = np.sum(self.weights * r2_scores)
        
        return r2_scores, float(weighted_avg)
    
    def _compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute standard R² for a single target.
        
        :param y_true: Ground truth values
        :param y_pred: Predicted values
        :return: R² score
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return float(1 - (ss_res / ss_tot))
    
    def _compute_global_r2_subset(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Compute global R² for a subset of samples.
        
        :param y_true: Ground truth values (n_samples, n_targets)
        :param y_pred: Predicted values (n_samples, n_targets)
        :param mask: Boolean mask for samples to include
        :return: Global R² for the subset
        """
        if not np.any(mask):
            return 0.0
        
        return self._compute_global_r2(y_true[mask], y_pred[mask])
    
    def _log_verbose_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        global_r2: float,
        epoch: Optional[int],
        indices: Optional[np.ndarray],
        logger: Any,
    ) -> None:
        """Log detailed metric breakdowns.
        
        :param y_true: Ground truth values
        :param y_pred: Predicted values
        :param global_r2: The global R² score
        :param epoch: Current epoch number
        :param indices: Sample indices (for metadata lookup)
        :param logger: Logger object with log_to_external method
        """
        # Prepare metrics dict
        metrics: dict[str, Any] = {}
        
        # Always include epoch if available - this is critical for proper W&B tracking
        if epoch is not None:
            # Use the same wrapping pattern as in TorchTrainer
            if hasattr(logger, 'wrap_log'):
                metrics[logger.wrap_log("epoch")] = epoch
            else:
                metrics["epoch"] = epoch
        
        # Helper to wrap metric names with logger (includes fold suffix if present)
        def wrap_metric(name: str) -> str:
            if hasattr(logger, 'wrap_log'):
                return logger.wrap_log(name)
            return name
        
        # 1. Log old method (per-target averaged R²) as main "weighted_r2" metric
        # This keeps old runs comparable on W&B
        if self.log_old_metric:
            per_target_r2, old_weighted_avg = self._compute_per_target_r2(y_true, y_pred)
            metrics[wrap_metric("Validation/weighted_r2")] = old_weighted_avg
        
        # 2. Log per-target global R²
        for i, target_name in enumerate(self.target_names):
            # Create mask for this target (all samples, specific target)
            target_r2 = self._compute_r2(y_true[:, i], y_pred[:, i])
            metrics[wrap_metric(f"Validation/R2_target/{target_name}")] = target_r2
        
        # 3. Log per-species breakdown (if metadata available)
        if self.metadata is not None and indices is not None:
            metadata_subset = self.metadata.iloc[indices]
            
            if 'Species' in metadata_subset.columns:
                for species in metadata_subset['Species'].unique():
                    species_mask = metadata_subset['Species'].values == species
                    if np.any(species_mask):
                        species_r2 = self._compute_global_r2_subset(y_true, y_pred, species_mask)
                        # Clean species name for logging (replace spaces with underscores)
                        species_clean = str(species).replace(' ', '_')
                        metric_name = wrap_metric(f"Validation/R2_species/{species_clean}")
                        metrics[metric_name] = species_r2
                        
                        # Define metric with epoch tracking on first encounter
                        if hasattr(logger, 'external_define_metric'):
                            try:
                                logger.external_define_metric(metric_name, wrap_metric("epoch"))
                            except Exception:
                                pass  # Metric might already be defined, that's ok
        
        # 4. Log per-state breakdown (if metadata available)
        if self.metadata is not None and indices is not None:
            metadata_subset = self.metadata.iloc[indices]
            
            if 'State' in metadata_subset.columns:
                for state in metadata_subset['State'].unique():
                    state_mask = metadata_subset['State'].values == state
                    if np.any(state_mask):
                        state_r2 = self._compute_global_r2_subset(y_true, y_pred, state_mask)
                        metric_name = wrap_metric(f"Validation/R2_state/{state}")
                        metrics[metric_name] = state_r2
                        
                        # Define metric with epoch tracking on first encounter
                        if hasattr(logger, 'external_define_metric'):
                            try:
                                logger.external_define_metric(metric_name, wrap_metric("epoch"))
                            except Exception:
                                pass  # Metric might already be defined, that's ok
        
        # Log all metrics at once
        if hasattr(logger, 'log_to_external'):
            logger.log_to_external(message=metrics)
    
    def __repr__(self) -> str:
        """Return a deterministic string representation of the scorer."""
        weights_str = np.array2string(self.weights, separator=', ', precision=4)
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"weights={weights_str}, "
            f"verbose={self.verbose})"
        )

