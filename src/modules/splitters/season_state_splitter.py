"""Season-State based splitter for cross-validation.

Groups samples by combinations of season and state to ensure proper
cross-validation split that respects temporal and geographical boundaries.
"""
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def month_to_season(month: int) -> str:
    """Maps month (int) to season (str) for Southern Hemisphere (Australia).

    :param month: Month number (1-12)
    :return: Season name
    """
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:  # [12, 1, 2]
        return 'Winter'


class SeasonStateSplitter:
    """Cross-validation splitter based on season-state combinations.

    This splitter groups images by their season (derived from Sampling_Date)
    and State, then uses GroupKFold to ensure that all images from the same
    season-state combination are in the same fold.

    This prevents data leakage across temporal and geographical boundaries.

    :param data_path: Path to train.csv file containing metadata
    :param n_splits: Number of folds for cross-validation
    :param shuffle: Whether to shuffle groups before splitting
    :param random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        data_path: str | Path,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> None:
        """Initialize the SeasonStateSplitter.

        :param data_path: Path to train.csv file
        :param n_splits: Number of folds
        :param shuffle: Whether to shuffle the groups
        :param random_state: Random seed
        """
        self.data_path = Path(data_path)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        # Load metadata and create groups
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata and create season-state groups for each image."""
        # Read the CSV file
        df = pd.read_csv(self.data_path)

        # Get unique images with their metadata (each image appears 5 times for 5 targets)
        unique_df = df.groupby('image_path').first().reset_index()

        # Extract season from Sampling_Date
        unique_df['season'] = pd.to_datetime(unique_df['Sampling_Date']).dt.month.map(month_to_season)

        # Create season-state combination group labels
        unique_df['season_state_group'] = unique_df['season'] + '_' + unique_df['State']

        # Convert groups to numeric labels for GroupKFold
        group_labels, unique_groups = pd.factorize(unique_df['season_state_group'])

        # Store groups and image order
        self.groups = group_labels
        self.image_paths = unique_df['image_path'].values
        self.unique_groups = unique_groups

        # Log information about the groups
        from src.utils.logger import logger
        logger.info(f"SeasonStateSplitter: Found {len(unique_groups)} unique season-state combinations")
        logger.info(f"Groups: {list(unique_groups)}")

        # Count samples per group
        group_counts = pd.Series(self.groups).value_counts().sort_index()
        logger.info(f"Samples per group: {dict(zip(unique_groups, group_counts.values))}")

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices split by season-state groups.

        :param X: Input data (not used, but required for sklearn compatibility)
        :param y: Target data (not used, but required for sklearn compatibility)
        :param groups: Group labels (not used, we use our own)
        :yield: Train and test indices for each fold
        """
        # Use GroupKFold with our season-state groups
        gkf = GroupKFold(n_splits=self.n_splits)

        # Create dummy X array with correct length
        dummy_X = np.zeros(len(self.groups))

        # Generate splits using the groups
        for train_idx, test_idx in gkf.split(dummy_X, groups=self.groups):
            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Return the number of splitting iterations.

        :param X: Input data (not used)
        :param y: Target data (not used)
        :param groups: Group labels (not used)
        :return: Number of splits
        """
        return self.n_splits
