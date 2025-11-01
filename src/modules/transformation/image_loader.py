"""Image loading and preprocessing transformation."""
from typing import Optional

import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


class ImageLoader(VerboseTransformationBlock):
    """Load and preprocess images for training.
    
    Converts image paths to tensors in [0, 1] range with resizing.
    Normalization is handled in the dataset/augmentation pipeline.
    """
    
    def __init__(
        self,
        size: int = 224,
    ) -> None:
        """Initialize the image loader.
        
        :param size: Target size for images
        """
        super().__init__()
        self.size = size
    
    def custom_fit(self, x: npt.NDArray[np.str_], y: Optional[npt.NDArray] = None) -> None:
        """Fit the transformer.

        :param x: Array of image file paths
        :param y: Ignored
        """
        self.log_to_terminal(f"ImageLoader ready to process {len(x)} images (output range: [0, 1])")
    
    def custom_transform(self, x: npt.NDArray[np.str_]) -> npt.NDArray[np.float32]:
        """Transform image paths to tensors in [0, 1] range.

        :param x: Array of image file paths
        :return: Array of image tensors (N, C, H, W) in [0, 1] range
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
        :return: Preprocessed image array (C, H, W) in [0, 1] range
        """
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Resize
        img = img.resize((self.size, self.size), Image.BILINEAR)
        
        # Convert to tensor [0, 255]
        img_array = np.array(img, dtype=np.float32)
        
        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)
        
        # Scale to [0, 1] range (mean/std normalization happens in dataset)
        img_array = img_array / 255.0
        
        return img_array
    

