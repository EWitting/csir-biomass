"""File containing all methods related to processing and formatting data before it is inserted into a pipeline.

- Since these methods are very competition specific none have been implemented here yet.
- Usually you'll have one data setup for training and another for making submissions.
"""
from pathlib import Path
from typing import Any 
import pandas as pd
import numpy as np

from src.utils.logger import logger


def setup_train_x_data(path: Path) -> np.ndarray:
    """Create train x data for pipeline.
    
    Returns image file paths as strings. Images are loaded during training for memory efficiency.
    Each unique image appears once.

    :param path: Path to train.csv
    :return: Array of image paths (n_images,) with dtype object
    """
    logger.info("Loading training data from train.csv")
    
    df = pd.read_csv(path)
    
    # Get unique images (each image has 5 rows for 5 targets)
    unique_images = df[['image_path']].drop_duplicates().reset_index(drop=True)
    
    # Convert to full paths - use dtype=object for string arrays
    raw_path = path.parent
    image_paths = unique_images['image_path'].apply(lambda x: str(raw_path / x)).to_numpy(dtype=object)
    
    logger.info(f"Loaded {len(image_paths)} unique training images")
    
    return image_paths


def setup_train_y_data(path: Path) -> np.ndarray:
    """Create train y data for pipeline.
    
    Returns targets in shape (n_images, 5) for the 5 biomass components:
    [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]

    :param path: Path to train.csv
    :return: Target array (n_images, 5)
    """
    logger.info("Loading target data from train.csv")
    
    df = pd.read_csv(path)
    
    # Target order (alphabetical for consistency)
    target_names = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'GDM_g', 'Dry_Total_g']
    
    # Group by image and pivot to get all 5 targets per image
    targets_df = df.pivot_table(
        index='image_path',
        columns='target_name',
        values='target',
        aggfunc='first'
    )
    
    # Ensure consistent ordering
    targets = targets_df[target_names].to_numpy(dtype=np.float32)
    
    logger.info(f"Loaded targets with shape {targets.shape}")
    logger.info(f"Target order: {target_names}")
    
    return targets


def setup_inference_data(path: Path) -> np.ndarray:
    """Create data for inference with pipeline.
    
    Returns image file paths for test set.

    :param path: Path to test.csv
    :return: Array of image paths
    """
    logger.info("Loading test data from test.csv")
    
    df = pd.read_csv(path)
    
    # Get unique images
    unique_images = df[['image_path']].drop_duplicates().reset_index(drop=True)
    
    # Convert to full paths
    raw_path = path.parent
    image_paths = unique_images['image_path'].apply(lambda x: str(raw_path / x)).to_numpy()
    
    logger.info(f"Loaded {len(image_paths)} test images")
    
    return image_paths


def setup_splitter_data(path: Path) -> np.ndarray:
    """Create data for splitter.
    
    Returns image paths for stratification/grouping in cross-validation.

    :param path: Path to train.csv
    :return: Array of image identifiers for stratification
    """
    df = pd.read_csv(path)
    
    # Return unique image paths (one per image)
    unique_images = df[['image_path']].drop_duplicates().reset_index(drop=True)
    
    return unique_images['image_path'].to_numpy()
