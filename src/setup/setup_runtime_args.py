"""File containing functions related to setting up runtime arguments for pipelines."""
from typing import Any

from epochlib.ensemble import EnsemblePipeline
from epochlib.model import ModelPipeline

from src.utils.logger import logger


def setup_train_args(
    pipeline: ModelPipeline | EnsemblePipeline,
    cache_args: dict[str, Any],
    train_indices: list[int],
    test_indices: list[int],
    fold: int = -1,
    *,
    save_model: bool = False,
    save_model_preds: bool = False,
    groups: Any = None,
) -> dict[str, Any]:
    """Set train arguments for pipeline.

    :param pipeline: Pipeline to receive arguments
    :param cache_args: Caching arguments
    :param train_indices: Train indices
    :param test_indices: Test indices
    :param fold: Fold number if it exists
    :param save_model: Whether to save the model to File
    :param save_model_preds: Whether to save the model predictions
    :param groups: Group labels for stratified bagging/CV
    :return: Dictionary containing arguments
    """
    x_sys = {
        "cache_args": cache_args,
    }

    main_trainer = {
        "train_indices": train_indices,
        "validation_indices": test_indices,
        "save_model": save_model,
    }

    if fold > -1:
        main_trainer["fold"] = fold

    # Add groups if provided
    if groups is not None:
        main_trainer["groups"] = groups

    train_sys = {
        "MainTrainer": main_trainer,
    }

    # Also add train args to TabularTrainer and MLPTrainer if they exist
    if hasattr(pipeline, 'train_sys') and hasattr(pipeline.train_sys, 'get_steps'):
        for step in pipeline.train_sys.get_steps():
            step_name = step.__class__.__name__
            if step_name in ["TabularTrainer", "MLPTrainer"]:
                # TabularTrainer and MLPTrainer need the same args as MainTrainer
                trainer_args = {
                    "train_indices": train_indices,
                    "validation_indices": test_indices,
                    "save_model": save_model,
                }
                if fold > -1:
                    trainer_args["fold"] = fold
                if groups is not None:
                    trainer_args["groups"] = groups
                train_sys[step_name] = trainer_args

    if save_model_preds:
        train_sys["cache_args"] = cache_args

    pred_sys: dict[str, Any] = {}

    train_args = {
        "x_sys": x_sys,
        "train_sys": train_sys,
        "pred_sys": pred_sys,
    }

    if isinstance(pipeline, EnsemblePipeline):
        train_args = {
            "ModelPipeline": train_args,
        }

    return train_args


def setup_pred_args(pipeline: ModelPipeline | EnsemblePipeline) -> dict[str, Any]:
    """Set train arguments for pipeline.

    :param pipeline: Pipeline to receive arguments
    :return: Dictionary containing arguments
    """
    # pred_args = {
    #     "train_sys": {
    #         "MainTrainer": {
    #             # "batch_size": 16,
    #             # "model_folds": cfg.model_folds,
    #         },
    #     },
    # }
    pred_args: dict[str, Any] = {}

    if isinstance(pipeline, EnsemblePipeline):
        pred_args = {
            "ModelPipeline": pred_args,
        }

    return pred_args
