"""DINOv3 with Masked LoRA Experts (MLAE) for biomass prediction.

This module implements MLAE fine-tuning on DINOv3 vision transformers.
MLAE decomposes LoRA matrices into rank-1 experts with adaptive coefficients
and applies stochastic masking during training for diversity.

Based on: "Masked LoRA Experts for Efficient Vision Transformer Fine-tuning"
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from typing import Optional, Literal
import math

from src.modules.training.models.mlae import MLAE_Linear


class DINOv3_MLAE_Model(nn.Module):
    """DINOv3 with Masked LoRA Experts and regression head.

    Applies MLAE to Q, K, V projections in all transformer blocks.
    Supports various masking strategies: fixed, stochastic, and mixed.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        drop_rate: float = 0.1,
        num_outputs: int = 5,
        head_dropout: float = 0.2,
        compute_aggregates: bool = False,
        activation: nn.Module | None = None,
        freeze_backbone: bool = True,
        apply_to_qkv: bool = True,
        apply_to_proj: bool = False,
        apply_to_mlp: bool = False,
        masking_strategy: Literal["random", "incremental", "decremental", "hourglass", "protruding"] = "random",
        masking_type: Literal["fixed", "stochastic", "mixed"] = "stochastic",
    ) -> None:
        """Initialize DINOv3 with MLAE.

        :param model_name: HuggingFace DINOv3 model name
        :param rank: LoRA rank (number of experts per layer)
        :param lora_alpha: Initial value for adaptive coefficients
        :param lora_dropout: Dropout for LoRA inputs
        :param drop_rate: Expert dropout rate for stochastic masking
        :param num_outputs: Number of regression outputs (5 for biomass)
        :param head_dropout: Dropout rate for regression head
        :param compute_aggregates: Compute GDM and Total from base components
        :param activation: Optional activation for head output
        :param freeze_backbone: Freeze non-LoRA backbone parameters
        :param apply_to_qkv: Apply MLAE to Q, K, V projections
        :param apply_to_proj: Apply MLAE to attention output projection
        :param apply_to_mlp: Apply MLAE to MLP layers
        :param masking_strategy: Layer-wise masking pattern
        :param masking_type: Type of masking (fixed, stochastic, mixed)
        """
        super().__init__()

        self.model_name = model_name
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.drop_rate = drop_rate
        self.num_outputs = num_outputs
        self.head_dropout = head_dropout
        self.compute_aggregates = compute_aggregates
        self.activation = activation
        self.freeze_backbone = freeze_backbone
        self.apply_to_qkv = apply_to_qkv
        self.apply_to_proj = apply_to_proj
        self.apply_to_mlp = apply_to_mlp
        self.masking_strategy = masking_strategy
        self.masking_type = masking_type

        # Load DINOv3 model
        self.backbone = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # Get embedding dimension
        self.hidden_dim = self.backbone.config.hidden_size

        # Count transformer blocks
        self.num_layers = len(self.backbone.layer)

        # Apply MLAE to specified layers
        self._inject_mlae_layers()

        # Create mask matrix based on strategy
        self.mask_matrix = self._create_mask_matrix()

        # Freeze backbone if requested
        if self.freeze_backbone:
            for name, param in self.backbone.named_parameters():
                # Only freeze non-LoRA parameters
                if 'lora_' not in name:
                    param.requires_grad = False

        # Determine head outputs
        head_outputs = 3 if self.compute_aggregates else self.num_outputs

        # Create regression head
        self.head = nn.Sequential(
            nn.Dropout(p=self.head_dropout),
            nn.Linear(self.hidden_dim, head_outputs)
        )

    def _inject_mlae_layers(self) -> None:
        """Replace linear layers with MLAE layers in the transformer."""
        for layer_idx, layer in enumerate(self.backbone.layer):
            attention = layer.attention

            # Apply to Q, K, V projections
            if self.apply_to_qkv:
                # Get dropout rate for this layer based on masking strategy
                layer_drop_rate = self._get_layer_drop_rate(layer_idx)

                # Replace query, key, value projections
                attention.q_proj = self._replace_linear_with_mlae(
                    attention.q_proj, layer_drop_rate
                )
                attention.k_proj = self._replace_linear_with_mlae(
                    attention.k_proj, layer_drop_rate
                )
                attention.v_proj = self._replace_linear_with_mlae(
                    attention.v_proj, layer_drop_rate
                )

            # Apply to output projection
            if self.apply_to_proj:
                layer_drop_rate = self._get_layer_drop_rate(layer_idx)
                attention.o_proj = self._replace_linear_with_mlae(
                    attention.o_proj, layer_drop_rate
                )

            # Apply to MLP layers
            if self.apply_to_mlp:
                layer_drop_rate = self._get_layer_drop_rate(layer_idx)
                layer.mlp.up_proj = self._replace_linear_with_mlae(
                    layer.mlp.up_proj, layer_drop_rate
                )
                layer.mlp.down_proj = self._replace_linear_with_mlae(
                    layer.mlp.down_proj, layer_drop_rate
                )

    def _replace_linear_with_mlae(
        self,
        linear_layer: nn.Linear,
        layer_drop_rate: float
    ) -> MLAE_Linear:
        """Replace a linear layer with MLAE_Linear.

        :param linear_layer: Original linear layer
        :param layer_drop_rate: Dropout rate for this layer's experts
        :return: MLAE_Linear layer with copied weights
        """
        mlae_layer = MLAE_Linear(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            r=self.rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            drop_rate=layer_drop_rate,
            bias=linear_layer.bias is not None,
        )

        # Copy original weights and bias
        mlae_layer.weight.data = linear_layer.weight.data.clone()
        if linear_layer.bias is not None:
            mlae_layer.bias.data = linear_layer.bias.data.clone()

        return mlae_layer

    def _get_layer_drop_rate(self, layer_idx: int) -> float:
        """Get dropout rate for a specific layer based on masking strategy.

        :param layer_idx: Index of the transformer layer
        :return: Dropout rate for this layer
        """
        if self.masking_type == "fixed":
            # Fixed masking: permanently mask experts (dropout in init only)
            return 0.0  # No dropout during training

        # Get normalized position (0 to 1)
        pos = layer_idx / max(1, self.num_layers - 1)

        # Apply masking strategy
        if self.masking_strategy == "random":
            # Uniform dropout across all layers
            return self.drop_rate
        elif self.masking_strategy == "incremental":
            # Lower layers: low dropout, upper layers: high dropout
            return self.drop_rate * pos
        elif self.masking_strategy == "decremental":
            # Lower layers: high dropout, upper layers: low dropout
            return self.drop_rate * (1 - pos)
        elif self.masking_strategy == "hourglass":
            # Middle layers: high dropout, endpoints: low dropout
            return self.drop_rate * (1 - abs(2 * pos - 1))
        elif self.masking_strategy == "protruding":
            # Middle layers: low dropout, endpoints: high dropout
            return self.drop_rate * abs(2 * pos - 1)
        else:
            return self.drop_rate

    def _create_mask_matrix(self) -> Optional[torch.Tensor]:
        """Create fixed mask matrix for permanent expert masking.

        Only used when masking_type is "fixed" or "mixed".
        Returns binary mask of shape (num_layers, rank).
        """
        if self.masking_type == "stochastic":
            return None  # No fixed mask needed

        # Initialize mask (1 = keep, 0 = permanently discard)
        mask = torch.ones(self.num_layers, self.rank)

        # Apply masking pattern based on strategy
        for layer_idx in range(self.num_layers):
            pos = layer_idx / max(1, self.num_layers - 1)

            # Determine how many experts to keep (inverse of dropout)
            if self.masking_strategy == "random":
                keep_ratio = 1 - self.drop_rate
            elif self.masking_strategy == "incremental":
                keep_ratio = 1 - self.drop_rate * pos
            elif self.masking_strategy == "decremental":
                keep_ratio = 1 - self.drop_rate * (1 - pos)
            elif self.masking_strategy == "hourglass":
                keep_ratio = 1 - self.drop_rate * (1 - abs(2 * pos - 1))
            elif self.masking_strategy == "protruding":
                keep_ratio = 1 - self.drop_rate * abs(2 * pos - 1)
            else:
                keep_ratio = 1 - self.drop_rate

            # Number of experts to keep
            num_keep = int(self.rank * keep_ratio)

            # Randomly select which experts to keep
            if num_keep < self.rank:
                perm = torch.randperm(self.rank)
                mask[layer_idx, perm[num_keep:]] = 0

        return mask

    def apply_fixed_mask(self) -> None:
        """Apply fixed mask by zeroing out permanently masked experts.

        Only used when masking_type is "fixed" or "mixed".
        """
        if self.mask_matrix is None:
            return

        layer_count = 0
        for layer in self.backbone.layer:
            attention = layer.attention

            # Apply mask to Q, K, V if they're MLAE layers
            if self.apply_to_qkv:
                if isinstance(attention.q_proj, MLAE_Linear):
                    self._mask_experts(attention.q_proj, layer_count)
                if isinstance(attention.k_proj, MLAE_Linear):
                    self._mask_experts(attention.k_proj, layer_count)
                if isinstance(attention.v_proj, MLAE_Linear):
                    self._mask_experts(attention.v_proj, layer_count)

            layer_count += 1

    def _mask_experts(self, mlae_layer: MLAE_Linear, layer_idx: int) -> None:
        """Zero out permanently masked experts in an MLAE layer.

        :param mlae_layer: MLAE_Linear layer
        :param layer_idx: Layer index for mask lookup
        """
        if self.mask_matrix is None or layer_idx >= len(self.mask_matrix):
            return

        mask = self.mask_matrix[layer_idx].to(mlae_layer.lora_lamda.device)

        # Zero out masked experts in lambda (adaptive coefficients)
        with torch.no_grad():
            mlae_layer.lora_lamda.data *= mask
            # Also zero out corresponding A and B rows
            mlae_layer.lora_A.data *= mask.unsqueeze(1)
            mlae_layer.lora_B.data *= mask.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input images (B, C, H, W) in [0, 1] range
        :return: Predictions (B, num_outputs)
        """
        # Extract features from backbone
        # DINOv3 expects pixel_values input
        outputs = self.backbone(pixel_values=x)

        # Get CLS token embedding (or use pooler_output if available)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output  # (B, hidden_dim)
        else:
            features = outputs.last_hidden_state[:, 0]  # (B, hidden_dim)

        # Regression head
        output = self.head(features)

        # Apply activation if specified
        if self.activation is not None:
            output = self.activation(output)

        # Compute aggregates if requested
        if self.compute_aggregates:
            clover = output[:, 0:1]  # (B, 1)
            dead = output[:, 1:2]    # (B, 1)
            green = output[:, 2:3]   # (B, 1)

            gdm = green + clover
            total = green + clover + dead

            output = torch.cat([clover, dead, green, gdm, total], dim=1)

        return output

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings without going through the head.

        :param x: Input images (B, C, H, W) in [0, 1] range
        :return: Feature embeddings (B, hidden_dim)
        """
        with torch.no_grad():
            outputs = self.backbone(pixel_values=x)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0]
        return features

    def get_lora_parameters(self):
        """Get only the LoRA parameters for training.

        :return: Iterator of LoRA parameters
        """
        for name, param in self.named_parameters():
            if 'lora_' in name or 'head' in name:
                yield param

    def print_trainable_parameters(self) -> None:
        """Print statistics about trainable parameters."""
        trainable_params = 0
        all_params = 0

        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {100 * trainable_params / all_params:.2f}%"
        )
