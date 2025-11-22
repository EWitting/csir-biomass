"""Tabular model trainer for sklearn-compatible models like AutoGluon."""

import functools
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from epochlib.data import Data
from epochlib.training.training_block import TrainingBlock

from src.modules.logging.logger import Logger


@dataclass
class TabularTrainer(TrainingBlock, Logger):
    """Trainer for sklearn-compatible tabular models.

    This trainer wraps any sklearn-compatible model (fit/predict interface)
    including AutoGluon, TabPFN, XGBoost, LightGBM, etc.

    The trainer handles:
    - Converting numpy arrays to pandas DataFrames (required by AutoGluon)
    - Passing groups for stratified bagging/cross-validation
    - Saving/loading fitted models
    - Flexible configuration via kwargs dictionaries

    Parameters
    ----------
    model : functools.partial
        Partial function that creates the model with init kwargs
    model_name : str
        Name of the model for logging and saving
    fit_kwargs : dict
        Keyword arguments to pass to model.fit()
    predict_kwargs : dict
        Keyword arguments to pass to model.predict()
    to_predict : str
        What to predict on: 'validation', 'all', or 'none'
    use_groups : bool
        Whether to pass groups to the fit method
    convert_to_dataframe : bool
        Whether to convert numpy arrays to pandas DataFrames
    """

    model: functools.partial
    model_name: str = "TabularModel"
    fit_kwargs: dict[str, Any] = field(default_factory=dict)
    predict_kwargs: dict[str, Any] = field(default_factory=dict)
    to_predict: str = "validation"  # 'validation', 'all', 'none'
    use_groups: bool = True
    convert_to_dataframe: bool = True
    compute_aggregates: bool = False  # If True, train on 3 base targets and compute 2 aggregates

    # Internal state
    fitted_model: Any = field(default=None, init=False, repr=False)
    fitted_models: list[Any] = field(default_factory=list, init=False, repr=False)  # For multi-target
    trained_models_directory: Path = field(
        default=Path("models/trained"), init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize the trainer."""
        super().__post_init__()
        self.trained_models_directory.mkdir(parents=True, exist_ok=True)

    def _to_dataframe(
        self, x: npt.NDArray[np.float32], prefix: str = "feat"
    ) -> pd.DataFrame:
        """Convert numpy array to pandas DataFrame.

        Parameters
        ----------
        x : np.ndarray
            Input features, shape (n_samples, n_features)
        prefix : str
            Prefix for column names

        Returns
        -------
        pd.DataFrame
            DataFrame with columns named feat_0, feat_1, ...
        """
        if not self.convert_to_dataframe:
            return x

        n_features = x.shape[1] if len(x.shape) > 1 else 1
        columns = [f"{prefix}_{i}" for i in range(n_features)]
        return pd.DataFrame(x, columns=columns)

    def custom_train(
        self,
        x: npt.NDArray[np.float32] | Data,
        y: npt.NDArray[np.float32] | Data,
        **kwargs: Any,
    ) -> tuple[npt.NDArray[np.float32] | None, npt.NDArray[np.float32] | None]:
        """Train the tabular model.

        Parameters
        ----------
        x : np.ndarray
            Input features, shape (n_samples, n_features)
        y : np.ndarray
            Target values, shape (n_samples, n_targets)
        **kwargs : Any
            Additional arguments:
            - train_indices: Indices for training data
            - validation_indices: Indices for validation data
            - groups: Group labels for stratified splitting/bagging
            - save_model: Whether to save the model
            - fold: Fold number

        Returns
        -------
        tuple[np.ndarray | None, np.ndarray | None]
            Predictions and labels (or None if to_predict='none')
        """
        # Extract indices from kwargs
        train_indices = kwargs.get("train_indices", None)
        validation_indices = kwargs.get("validation_indices", None)
        groups = kwargs.get("groups", None)

        if train_indices is None or validation_indices is None:
            raise ValueError("train_indices and validation_indices must be provided in kwargs")

        self.log_to_terminal(
            f"Training {self.model_name} on {len(train_indices)} samples"
        )

        # Convert indices to arrays if needed
        train_indices = np.array(train_indices)
        validation_indices = np.array(validation_indices)

        # Get train data
        X_train = x[train_indices]
        y_train = y[train_indices]

        # Convert to DataFrame if needed
        X_train_df = self._to_dataframe(X_train)

        # Check if this is AutoGluon by inspecting the partial function
        is_autogluon = False
        try:
            # Check if the model is AutoGluon TabularPredictor
            if hasattr(self.model, 'func'):
                model_class = self.model.func
                is_autogluon = 'TabularPredictor' in str(model_class)
        except Exception:
            pass

        # For AutoGluon, we need to create a single DataFrame with features, target(s), and groups
        if is_autogluon and self.convert_to_dataframe:
            self.log_to_terminal("Preparing AutoGluon DataFrame with target and groups")

            # Handle multi-target case
            # AutoGluon only supports single target, so we need to handle this carefully
            if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                if self.compute_aggregates:
                    # Expected target order: [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
                    # Train 3 separate models for base components, compute aggregates later
                    self.log_to_terminal(
                        f"AutoGluon multi-target: Training 3 separate models for base targets "
                        f"(Clover, Dead, Green). Will compute aggregates (GDM, Total) from predictions."
                    )

                    target_names = ["Clover", "Dead", "Green"]
                    self.fitted_models = []

                    for target_idx, target_name in enumerate(target_names):
                        self.log_to_terminal(f"\n{'='*80}")
                        self.log_to_terminal(f"Training model {target_idx+1}/3 for {target_name}")
                        self.log_to_terminal(f"{'='*80}")

                        # Create DataFrame for this target
                        X_train_target = X_train_df.copy()
                        X_train_target['target'] = y_train[:, target_idx]

                        # Add groups column if provided
                        if self.use_groups and groups is not None:
                            train_groups = groups[train_indices]
                            X_train_target['group_id'] = train_groups
                            if target_idx == 0:  # Only log once
                                self.log_to_terminal(
                                    f"Found {len(np.unique(train_groups))} unique groups in training data"
                                )
                                self.log_to_terminal(
                                    "Groups will be used for stratified bagging in AutoGluon"
                                )

                        # Initialize model (groups is passed in __init__ via partial)
                        model = self.model()

                        # Prepare fit kwargs
                        fit_kwargs = self.fit_kwargs.copy()

                        # Fit the model
                        self.log_to_terminal(f"Fitting {target_name} model...")
                        model.fit(X_train_target, **fit_kwargs)

                        self.fitted_models.append(model)
                        self.log_to_terminal(f"{target_name} model training completed")

                    # Set fitted_model to first model for compatibility
                    self.fitted_model = self.fitted_models[0]

                else:
                    self.log_to_warning(
                        f"AutoGluon does not support multi-target regression. "
                        f"Found {y_train.shape[1]} targets. "
                        f"Will train on first target only. "
                        f"Consider enabling compute_aggregates=True to train on base components."
                    )
                    # Use only the first target
                    y_train_single = y_train[:, 0]

                    # Add target column to DataFrame
                    X_train_df['target'] = y_train_single

                    # Add groups column if provided
                    if self.use_groups and groups is not None:
                        train_groups = groups[train_indices]
                        X_train_df['group_id'] = train_groups
                        self.log_to_terminal(
                            f"Found {len(np.unique(train_groups))} unique groups in training data"
                        )
                        self.log_to_terminal(
                            "Groups will be used for stratified bagging in AutoGluon"
                        )

                    # Initialize and fit single model
                    self.log_to_terminal(f"Initializing {self.model_name}")
                    self.fitted_model = self.model()
                    fit_kwargs = self.fit_kwargs.copy()
                    self.log_to_terminal("Fitting model...")
                    self.fitted_model.fit(X_train_df, **fit_kwargs)
            else:
                # Single target case
                y_train_single = y_train.flatten() if len(y_train.shape) > 1 else y_train

                # Add target column to DataFrame
                X_train_df['target'] = y_train_single

                # Add groups column if provided
                if self.use_groups and groups is not None:
                    train_groups = groups[train_indices]
                    X_train_df['group_id'] = train_groups
                    self.log_to_terminal(
                        f"Found {len(np.unique(train_groups))} unique groups in training data"
                    )
                    self.log_to_terminal(
                        "Groups will be used for stratified bagging in AutoGluon"
                    )

                # Initialize and fit single model
                self.log_to_terminal(f"Initializing {self.model_name}")
                self.fitted_model = self.model()
                fit_kwargs = self.fit_kwargs.copy()
                self.log_to_terminal("Fitting model...")
                self.fitted_model.fit(X_train_df, **fit_kwargs)

        else:
            # Standard sklearn-compatible models
            # Initialize model
            self.log_to_terminal(f"Initializing {self.model_name}")
            self.fitted_model = self.model()

            # Prepare fit kwargs
            fit_kwargs = self.fit_kwargs.copy()

            # For sklearn models, groups can be passed to fit via fit_kwargs
            if self.use_groups and groups is not None:
                train_groups = groups[train_indices]
                self.log_to_terminal(
                    f"Found {len(np.unique(train_groups))} unique groups in training data"
                )

            # Fit the model
            self.log_to_terminal("Fitting model...")

            # Handle different fit signatures
            try:
                self.fitted_model.fit(X_train_df, y_train, **fit_kwargs)
            except TypeError:
                # Some models might not accept all kwargs
                self.log_to_warning(
                    "Model doesn't accept all fit_kwargs, trying with fewer arguments"
                )
                self.fitted_model.fit(X_train_df, y_train)

        self.log_to_terminal("Model training completed")

        # Save model if requested
        if kwargs.get("save_model", False):
            self._save_model()

        # Make predictions based on to_predict setting
        predictions = None
        labels = None

        if self.to_predict == "validation" and len(validation_indices) > 0:
            self.log_to_terminal("Predicting on validation set")
            X_val = self._to_dataframe(x[validation_indices])
            predictions = self.custom_predict(X_val)
            labels = y[validation_indices]

        elif self.to_predict == "all":
            self.log_to_terminal("Predicting on all data")
            X_all = self._to_dataframe(x)
            predictions = self.custom_predict(X_all)
            labels = y

        return predictions, labels

    def custom_predict(
        self,
        x: npt.NDArray[np.float32] | Data | pd.DataFrame,
        **pred_args: Any,
    ) -> npt.NDArray[np.float32]:
        """Predict using the fitted model.

        Parameters
        ----------
        x : np.ndarray or pd.DataFrame
            Input features
        **pred_args : Any
            Additional arguments to pass to predict()

        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples, n_targets)
            If compute_aggregates=True, returns all 5 targets with aggregates computed
        """
        if self.fitted_model is None and not self.fitted_models:
            raise ValueError("Model has not been trained yet. Call train() first.")

        # Convert to DataFrame if needed and input is numpy array
        if isinstance(x, np.ndarray):
            x_df = self._to_dataframe(x)
        else:
            x_df = x

        # Merge predict_kwargs with pred_args
        kwargs = {**self.predict_kwargs, **pred_args}

        # Handle multi-model case (compute_aggregates=True)
        if self.fitted_models and self.compute_aggregates:
            self.log_to_terminal("Predicting with 3 base models and computing aggregates")

            # Predict with each model
            base_predictions = []
            for model in self.fitted_models:
                preds = model.predict(x_df, **kwargs)

                # Convert to numpy if needed
                if isinstance(preds, pd.DataFrame) or isinstance(preds, pd.Series):
                    preds = preds.values

                # Ensure 1D
                if len(preds.shape) > 1:
                    preds = preds.flatten()

                base_predictions.append(preds)

            # Stack base predictions: [Clover, Dead, Green]
            clover = base_predictions[0].reshape(-1, 1)  # (N, 1)
            dead = base_predictions[1].reshape(-1, 1)    # (N, 1)
            green = base_predictions[2].reshape(-1, 1)   # (N, 1)

            # Compute aggregates
            gdm = green + clover                         # GDM = Green + Clover
            total = green + clover + dead                # Total = Green + Clover + Dead

            # Concatenate in expected order: [Clover, Dead, Green, GDM, Total]
            predictions = np.concatenate([clover, dead, green, gdm, total], axis=1)

            self.log_to_terminal(f"Generated predictions with shape {predictions.shape} (5 targets)")

            return predictions.astype(np.float32)

        else:
            # Single model case
            predictions = self.fitted_model.predict(x_df, **kwargs)

            # Convert to numpy if needed
            if isinstance(predictions, pd.DataFrame) or isinstance(predictions, pd.Series):
                predictions = predictions.values

            # Ensure 2D shape for consistency
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)

            return predictions.astype(np.float32)

    def _save_model(self, path: Path | None = None) -> None:
        """Save the fitted model.

        For AutoGluon, uses its save method.
        For other sklearn models, uses pickle.

        Parameters
        ----------
        path : Path, optional
            Path to save the model. If None, uses default path.
        """
        if self.fitted_model is None:
            self.log_to_warning("No fitted model to save")
            return

        if path is None:
            path = self.trained_models_directory / f"{self.get_hash()}"

        # Check if this is AutoGluon (has a save method)
        if hasattr(self.fitted_model, "save"):
            # AutoGluon saves to a directory
            save_path = str(path)
            self.log_to_terminal(f"Saving AutoGluon model to {save_path}")
            self.fitted_model.save(save_path)
        else:
            # Use pickle for sklearn-compatible models
            save_path = path.with_suffix(".pkl")
            self.log_to_terminal(f"Saving model to {save_path}")
            with open(save_path, "wb") as f:
                pickle.dump(self.fitted_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_model(self, path: Path | None = None) -> None:
        """Load a fitted model.

        Parameters
        ----------
        path : Path, optional
            Path to load the model from. If None, uses default path.
        """
        if path is None:
            path = self.trained_models_directory / f"{self.get_hash()}"

        # Try loading as AutoGluon first
        if path.exists() and path.is_dir():
            self.log_to_terminal(f"Loading AutoGluon model from {path}")
            # Import AutoGluon only when needed
            try:
                from autogluon.tabular import TabularPredictor

                self.fitted_model = TabularPredictor.load(str(path))
                return
            except ImportError:
                self.log_to_warning("AutoGluon not available, trying pickle")

        # Try loading as pickle
        pkl_path = path.with_suffix(".pkl")
        if pkl_path.exists():
            self.log_to_terminal(f"Loading model from {pkl_path}")
            with open(pkl_path, "rb") as f:
                self.fitted_model = pickle.load(f)
            return

        raise FileNotFoundError(f"Model not found at {path} or {pkl_path}")
