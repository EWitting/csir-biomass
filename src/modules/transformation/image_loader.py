"""Image loading and preprocessing transformation."""
from typing import Optional

import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


class ImageLoader(VerboseTransformationBlock):
    """Load and preprocess images for training.
    
    Converts image paths to tensors with resizing and normalization.
    """
    
    def __init__(
        self,
        size: int = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        compute_stats: bool = False,
    ) -> None:
        """Initialize the image loader.
        
        :param size: Target size for images
        :param mean: Mean for normalization
        :param std: Std for normalization
        :param compute_stats: Whether to compute mean/std from data
        """
        super().__init__()
        self.size = size
        self.mean = mean
        self.std = std
        self.compute_stats = compute_stats
    
    def custom_fit(self, x: npt.NDArray[np.str_], y: Optional[npt.NDArray] = None) -> None:
        """Fit the transformer.
        
        Optionally computes dataset statistics if compute_stats=True.

        :param x: Array of image file paths
        :param y: Ignored
        """
        if self.compute_stats:
            self.log_to_terminal(f"Computing dataset mean and std from {len(x)} images...")
            self.mean, self.std = self._compute_dataset_stats(x)
            self.log_to_terminal(f"Dataset mean: {self.mean}")
            self.log_to_terminal(f"Dataset std: {self.std}")
        else:
            self.log_to_terminal(f"Using provided mean: {self.mean}, std: {self.std}")
    
    def custom_transform(self, x: npt.NDArray[np.str_]) -> npt.NDArray[np.float32]:
        """Transform image paths to normalized tensors.

        :param x: Array of image file paths
        :return: Array of image tensors (N, C, H, W)
        """
        self.log_to_terminal(f"Loading and preprocessing {len(x)} images...")
        
        images = []
        for img_path in tqdm(x, desc="Loading images", disable=False):
            img = self._load_and_preprocess_image(img_path)
            images.append(img)
        
        # Stack into array
        images_array = np.stack(images, axis=0)
        
        # Ensure dtype is float32
        images_array = images_array.astype(np.float32)
        
        self.log_to_terminal(f"Loaded images with shape {images_array.shape}, dtype={images_array.dtype}")
        
        return images_array
    
    def _load_and_preprocess_image(self, img_path: str) -> npt.NDArray[np.float32]:
        """Load a single image and preprocess it.

        :param img_path: Path to image file
        :return: Preprocessed image array (C, H, W)
        """
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Resize
        img = img.resize((self.size, self.size), Image.BILINEAR)
        
        # Convert to tensor [0, 255]
        img_array = np.array(img, dtype=np.float32)
        
        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Standardize using mean/std
        mean = np.array(self.mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(self.std, dtype=np.float32).reshape(3, 1, 1)
        img_array = (img_array - mean) / std
        
        return img_array
    
    def _compute_dataset_stats(self, image_paths: npt.NDArray[np.str_]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Compute mean and std across the dataset.
        
        Uses online algorithm for memory efficiency.

        :param image_paths: Array of image file paths
        :return: (mean, std) tuples
        """
        # Use a sample for efficiency if dataset is large
        sample_size = min(len(image_paths), 1000)
        sample_paths = np.random.choice(image_paths, size=sample_size, replace=False)
        
        # Accumulate pixel values per channel
        pixel_sum = np.zeros(3, dtype=np.float64)
        pixel_sum_sq = np.zeros(3, dtype=np.float64)
        pixel_count = 0
        
        for img_path in tqdm(sample_paths, desc="Computing stats"):
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.size, self.size), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
            
            # Accumulate per channel
            for c in range(3):
                channel_data = img_array[:, :, c].flatten()
                pixel_sum[c] += channel_data.sum()
                pixel_sum_sq[c] += (channel_data ** 2).sum()
            
            pixel_count += img_array.shape[0] * img_array.shape[1]
        
        # Compute mean and std
        mean = pixel_sum / pixel_count
        std = np.sqrt(pixel_sum_sq / pixel_count - mean ** 2)
        
        return tuple(mean.tolist()), tuple(std.tolist())

