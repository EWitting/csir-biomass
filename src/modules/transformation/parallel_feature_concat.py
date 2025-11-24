"""Parallel feature extractor that concatenates embeddings from multiple models."""
from dataclasses import dataclass
from typing import Any

import numpy as np

from epochlib.core.parallel_transforming_system import ParallelTransformingSystem
from src.modules.logging.logger import Logger


@dataclass
class ParallelFeatureConcat(ParallelTransformingSystem, Logger):
    """Run multiple feature extractors in parallel and concatenate their outputs.

    This system allows you to extract features from multiple foundation models
    simultaneously and combine them into a single feature vector.

    Each feature extractor is cached independently, so if you've already computed
    DINOv2 features, adding a SigLIP extractor will only compute the new features.

    The concatenated result is also cached separately.

    Parameters
    ----------
    steps : list
        List of feature extractors (e.g., FrozenDinoV2, FrozenSigLIP)
    weights : list[float], optional
        Weights for each extractor (currently unused, all weighted equally)

    Example
    -------
    ```yaml
    x_sys:
      steps:
        - _target_: src.modules.transformation.parallel_feature_concat.ParallelFeatureConcat
          steps:
            - _target_: src.modules.transformation.frozen_dinov2.FrozenDinoV2
              model_name: "dinov2_vitb14"
            - _target_: src.modules.transformation.frozen_siglip.FrozenSigLIP
              model_name: "vit_base_patch16_siglip_224"
    ```

    This would produce features of shape (N, 768 + 768) = (N, 1536)
    """

    def concat(
        self,
        original_data: np.ndarray | None,
        data_to_concat: np.ndarray,
        weight: float = 1.0,
    ) -> np.ndarray:
        """Concatenate feature embeddings along the feature dimension.

        Parameters
        ----------
        original_data : np.ndarray | None
            Previously concatenated features, or None if this is the first
        data_to_concat : np.ndarray
            New features to concatenate, shape (N, embedding_dim)
        weight : float
            Weight for the new features (currently unused)

        Returns
        -------
        np.ndarray
            Concatenated features, shape (N, total_embedding_dim)
        """
        if original_data is None:
            # First extractor
            return data_to_concat

        # Concatenate along feature dimension (axis=1)
        # original_data: (N, dim1), data_to_concat: (N, dim2)
        # result: (N, dim1 + dim2)
        concatenated = np.concatenate([original_data, data_to_concat], axis=1)

        self.log_to_debug(
            f"Concatenated features: {original_data.shape} + {data_to_concat.shape} = {concatenated.shape}"
        )

        return concatenated
