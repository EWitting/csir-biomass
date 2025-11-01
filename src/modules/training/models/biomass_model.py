"""Biomass prediction model using timm backbone."""
import timm
import torch.nn as nn

from src.utils.logger import logger


class BiomassModel(nn.Module):
    """Multi-output regression model for biomass prediction.
    
    Uses a timm backbone with a custom head for 5 biomass outputs.
    """
    
    def __init__(
        self,
        backbone: dict,
        num_outputs: int = 5,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the model.
        
        :param backbone: Dictionary of kwargs for timm.create_model()
        :param num_outputs: Number of output values
        :param dropout: Dropout rate for the head
        """
        super().__init__()
        
        self.num_outputs = num_outputs
        self.dropout = dropout
        
        # Create backbone with timm, handling internet connectivity issues
        try:
            self.backbone = timm.create_model(**backbone)
        except Exception as e:
            # Catch errors from huggingface/timm when internet is unavailable (e.g., on Kaggle)
            if "pretrained" in backbone and backbone["pretrained"]:
                logger.warning(
                    f"Failed to load pretrained weights (likely no internet access): {e}. "
                    "Retrying with pretrained=False..."
                )
                backbone_no_pretrain = backbone.copy()
                backbone_no_pretrain["pretrained"] = False
                self.backbone = timm.create_model(**backbone_no_pretrain)
            else:
                raise
        
        # Get feature dimension from backbone
        self.num_features = self.backbone.num_features
        
        # Create custom regression head
        self.head = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.num_features, self.num_outputs)
        )
    
    def forward(self, x):
        """Forward pass.

        :param x: Input images (B, C, H, W)
        :return: Predictions (B, num_outputs)
        """
        features = self.backbone(x)
        output = self.head(features)
        return output

