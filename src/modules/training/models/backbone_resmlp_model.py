"""Model combining a backbone encoder (timm) with ResMLPHead."""

import timm
import torch
import torch.nn as nn

from src.modules.training.models.resmlp_head import ResMLPBlock


class BackboneResMLPModel(nn.Module):
    """Model combining a timm backbone with ResMLPHead.

    Structure: Backbone (EfficientNet/ResNet/ViT) -> ResMLPHead

    This model is designed for two-stage training:
    - Stage 1: Freeze backbone, train ResMLPHead
    - Stage 2: Unfreeze backbone, finetune entire model
    """

    def __init__(
        self,
        backbone: dict,
        num_outputs: int = 5,
        num_blocks: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        compute_aggregates: bool = False,
        activation: nn.Module | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            backbone: Dict with timm model config:
                - model_name: timm model name (e.g., 'efficientnet_b3')
                - pretrained: bool, whether to use pretrained weights
                - num_classes: int, should be 0 to remove classifier
                - global_pool: str, pooling type (e.g., 'avg')
            num_outputs: Number of output values (5 by default)
            num_blocks: Number of ResMLPBlocks
            hidden_dim: Hidden dimension for ResMLPBlocks (defaults to backbone feature dim)
            dropout: Dropout rate
            compute_aggregates: If True, model learns only 3 base components
                (clover, dead, green) and computes GDM and Total arithmetically
            activation: Optional activation function after output layer

        Example:
            model = BackboneResMLPModel(
                backbone={
                    'model_name': 'efficientnet_b3',
                    'pretrained': True,
                    'num_classes': 0,
                    'global_pool': 'avg'
                },
                num_outputs=5,
                num_blocks=3,
                dropout=0.1
            )
        """
        super().__init__()

        self.num_outputs = num_outputs
        self.compute_aggregates = compute_aggregates
        self.activation = activation

        # Create backbone using timm
        self.backbone = timm.create_model(**backbone)

        # Get backbone output dimension
        # For models with num_classes=0 and global_pool, the output is the feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_output = self.backbone(dummy_input)
            backbone_dim = backbone_output.shape[-1]

        # Create ResMLPHead
        if hidden_dim is None:
            hidden_dim = backbone_dim

        # Stack of ResMLPBlocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResMLPBlock(backbone_dim, hidden_dim, dropout))
        self.blocks = nn.Sequential(*blocks)

        # Output layer
        head_outputs = 3 if self.compute_aggregates else self.num_outputs
        self.output = nn.Linear(backbone_dim, head_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Predictions (B, num_outputs)
        """
        # Extract features from backbone
        x = self.backbone(x)

        # Pass through ResMLPBlocks
        x = self.blocks(x)

        # Output layer with optional activation
        output = self.output(x)
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

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from the backbone (before ResMLPHead).

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Embeddings (B, backbone_dim)
        """
        return self.backbone(x)
