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


def setup_metadata(path: Path) -> pd.DataFrame | None:
    """Load metadata for verbose scoring.
    
    Returns DataFrame with State, Species, and other metadata for each image.
    Returns None if metadata columns are not available (e.g., during inference).
    
    :param path: Path to CSV file (train.csv or test.csv)
    :return: DataFrame with metadata (indexed by image order), or None if columns missing
    """
    logger.info(f"Attempting to load metadata from {path.name}")
    
    df = pd.read_csv(path)
    
    # Check if metadata columns exist (they won't in test.csv)
    metadata_cols = ['image_path', 'State', 'Species', 'Sampling_Date', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    missing_cols = [col for col in metadata_cols if col not in df.columns]
    
    if missing_cols:
        logger.info(f"Metadata columns not available (missing: {missing_cols}). Skipping metadata loading (this is normal during inference).")
        return None
    
    # Get metadata for unique images (take first row per image)
    metadata_df = df.groupby('image_path').first().reset_index()
    
    # Select relevant columns
    metadata = metadata_df[metadata_cols].copy()
    
    logger.info(f"Loaded metadata for {len(metadata)} images")
    logger.info(f"States: {metadata['State'].unique()}")
    logger.info(f"Species: {metadata['Species'].unique()}")

    return metadata


def setup_train_groups(path: Path) -> np.ndarray | None:
    """Create group labels for stratified splitting and bagging.

    Groups are created by combining season (derived from Sampling_Date) and State.
    This ensures that samples from the same geographical and temporal context
    are kept together during cross-validation and bagging.

    :param path: Path to train.csv
    :return: Array of group IDs (n_images,), or None if groups cannot be created
    """
    logger.info("Creating group labels for stratified splitting")

    df = pd.read_csv(path)

    # Check if required columns exist
    required_cols = ['image_path', 'State', 'Sampling_Date']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.warning(f"Cannot create groups (missing columns: {missing_cols}). Groups will not be used.")
        return None

    # Get unique images with their metadata
    unique_df = df.groupby('image_path').first().reset_index()

    # Extract season from Sampling_Date
    from src.modules.splitters.season_state_splitter import month_to_season
    unique_df['season'] = pd.to_datetime(unique_df['Sampling_Date']).dt.month.map(month_to_season)

    # Create season-state combination group labels
    unique_df['season_state_group'] = unique_df['season'] + '_' + unique_df['State']

    # Convert to numeric labels
    group_labels, unique_groups = pd.factorize(unique_df['season_state_group'])

    logger.info(f"Created {len(unique_groups)} unique groups")
    logger.info(f"Groups: {list(unique_groups)}")

    # Count samples per group
    group_counts = pd.Series(group_labels).value_counts().sort_index()
    logger.info(f"Samples per group: {dict(zip(unique_groups, group_counts.values))}")

    return group_labels.astype(np.int32)