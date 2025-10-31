"""Biomass prediction model using timm backbone."""
import torch.nn as nn


class BiomassModel(nn.Module):
    """Multi-output regression model for biomass prediction.
    
    Uses a timm backbone with a custom head for 5 biomass outputs.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_outputs: int = 5,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the model.
        
        :param backbone: Instantiated timm model (with num_classes=0)
        :param num_outputs: Number of output values
        :param dropout: Dropout rate for the head
        """
        super().__init__()
        
        self.backbone = backbone
        self.num_outputs = num_outputs
        self.dropout = dropout
        
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

