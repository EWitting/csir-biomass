"""ResMLP Head for foundation model embeddings."""
import torch
import torch.nn as nn


class ResMLPBlock(nn.Module):
    """Residual MLP block with LayerNorm and GELU activation.

    Architecture: Linear -> LayerNorm -> GELU -> Linear with skip connection
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        """Initialize the ResMLPBlock.

        :param dim: Input/output dimension
        :param hidden_dim: Hidden dimension (defaults to dim)
        :param dropout: Dropout rate
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        :param x: Input tensor (B, dim)
        :return: Output tensor (B, dim)
        """
        residual = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class ResMLPHead(nn.Module):
    """ResMLP Head for multi-output regression from foundation model embeddings.

    Uses a stack of ResMLPBlocks followed by a linear output layer with optional activation.
    Can optionally compute aggregates (GDM, Total) from base components (clover, dead, green).
    """

    def __init__(
        self,
        input_dim: int,
        num_outputs: int = 5,
        num_blocks: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        compute_aggregates: bool = False,
        activation: nn.Module | None = None,
    ) -> None:
        """Initialize the ResMLP Head.

        :param input_dim: Input embedding dimension (e.g., 3072 for 3x 1024-dim models)
        :param num_outputs: Number of output values (5 by default)
        :param num_blocks: Number of ResMLPBlocks (2-3 recommended)
        :param hidden_dim: Hidden dimension for blocks (defaults to input_dim)
        :param dropout: Dropout rate
        :param compute_aggregates: If True, model learns only 3 base components
            (clover, dead, green) and computes GDM and Total arithmetically.
            Target order: [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
            GDM = Green + Clover, Total = Green + Clover + Dead
        :param activation: Optional activation function to apply after output layer
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_outputs = num_outputs
        self.compute_aggregates = compute_aggregates
        self.activation = activation

        if hidden_dim is None:
            hidden_dim = input_dim

        # Stack of ResMLPBlocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResMLPBlock(input_dim, hidden_dim, dropout))
        self.blocks = nn.Sequential(*blocks)

        # Output layer
        # If compute_aggregates is True, learn only base components (clover, dead, green)
        head_outputs = 3 if self.compute_aggregates else self.num_outputs
        self.output = nn.Linear(input_dim, head_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input embeddings (B, input_dim)
        :return: Predictions (B, num_outputs)
        """
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
