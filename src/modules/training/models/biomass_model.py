"""Biomass prediction model using timm backbone."""
import timm
import torch
import torch.nn as nn

from src.utils.logger import logger


class BiomassModel(nn.Module):
    """Multi-output regression model for biomass prediction.
    
    Uses a timm backbone with a custom head for 5 biomass outputs.
    Can optionally learn only base components (clover, dead, green) and compute
    aggregates (GDM, Total) arithmetically.
    """
    
    def __init__(
        self,
        backbone: dict,
        num_outputs: int = 5,
        dropout: float = 0.2,
        compute_aggregates: bool = False,
        activation: nn.Module | None = None,
    ) -> None:
        """Initialize the model.
        
        :param backbone: Dictionary of kwargs for timm.create_model()
        :param num_outputs: Number of output values (5 by default)
        :param dropout: Dropout rate for the head
        :param compute_aggregates: If True, model learns only 3 base components 
            (clover, dead, green) and computes GDM and Total arithmetically.
            Target order: [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
            GDM = Green + Clover, Total = Green + Clover + Dead
        :param activation: Optional activation function to apply after head
        """
        super().__init__()
        
        self.num_outputs = num_outputs
        self.dropout = dropout
        self.compute_aggregates = compute_aggregates
        self.activation = activation
        
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
        
        # Determine number of outputs for the head
        # If compute_aggregates is True, learn only base components (clover, dead, green)
        head_outputs = 3 if self.compute_aggregates else self.num_outputs
        
        # Create custom regression head
        self.head = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.num_features, head_outputs)
        )
    
    def forward(self, x):
        """Forward pass.

        :param x: Input images (B, C, H, W)
        :return: Predictions (B, num_outputs)
        """
        features = self.backbone(x)
        output = self.head(features)

        # Apply optional activation function
        if self.activation is not None:
            output = self.activation(output)

        # If compute_aggregates is True, compute GDM and Total from base components
        if self.compute_aggregates:
            # output shape: (B, 3) with [clover, dead, green]
            clover = output[:, 0:1]  # (B, 1)
            dead = output[:, 1:2]    # (B, 1)
            green = output[:, 2:3]   # (B, 1)

            # Compute aggregates
            gdm = green + clover     # GDM = Green + Clover
            total = green + clover + dead  # Total = Green + Clover + Dead

            # Concatenate to match expected output order
            # [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
            output = torch.cat([clover, dead, green, gdm, total], dim=1)

        return output

    def get_embeddings(self, x):
        """Extract feature embeddings from the backbone without going through the head.

        This is useful for using the model as a feature extractor for downstream
        tabular models like AutoGluon.

        :param x: Input images (B, C, H, W)
        :return: Feature embeddings (B, num_features)
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features

