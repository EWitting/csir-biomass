"""RealMLP-TD Head inspired by the paper 'Revisiting Deep Learning Models for Tabular Data'.

Key features from the paper:
1. Scaling layer for soft feature selection (diagonal weight matrix)
2. Neural Tangent Parametrization (NTP) for linear layers
3. Parametric activation functions (PReLU-style with learnable alpha per neuron)
4. SELU activation for classification, Mish for regression
5. Dropout after each activation
6. Output clipping for regression (to observed range with margin)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalingLayer(nn.Module):
    """Diagonal scaling layer for soft feature selection.

    Computes: x_out = s * x_in, where s is a learnable scaling factor per feature.
    """

    def __init__(self, num_features: int) -> None:
        """Initialize scaling layer.

        Args:
            num_features: Number of input features
        """
        super().__init__()
        # Initialize scaling factors to 1.0
        self.scales = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature-wise scaling.

        Args:
            x: Input tensor (B, num_features)

        Returns:
            Scaled tensor (B, num_features)
        """
        return x * self.scales


class NTPLinear(nn.Module):
    """Linear layer with Neural Tangent Parametrization.

    Computes: z = (d_in)^(-1/2) * W @ x + b

    This effectively modifies the learning rate for the weight matrix depending
    on the input dimension, preventing too large steps when the number of
    columns is large.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Initialize NTP linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias term
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard linear layer
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # NTP scaling factor (constant, not learnable)
        self.register_buffer('ntp_scale', torch.tensor(in_features ** -0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with NTP.

        Args:
            x: Input tensor (B, in_features)

        Returns:
            Output tensor (B, out_features)
        """
        # Apply NTP scaling to weight matrix
        return F.linear(x, self.weight * self.ntp_scale, self.bias)


class ParametricActivation(nn.Module):
    """Parametric activation function with learnable alpha per neuron.

    Computes: σ_α(x) = (1 - α) * x + α * σ(x)

    When α = 1: recovers σ
    When α = 0: linear activation
    """

    def __init__(self, num_features: int, activation_fn: nn.Module) -> None:
        """Initialize parametric activation.

        Args:
            num_features: Number of neurons
            activation_fn: Base activation function (SELU or Mish)
        """
        super().__init__()
        self.activation_fn = activation_fn
        # Initialize alpha to 1.0 (fully uses activation function)
        self.alpha = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parametric activation.

        Args:
            x: Input tensor (B, num_features)

        Returns:
            Activated tensor (B, num_features)
        """
        # Clamp alpha to [0, 1] for stability
        alpha = self.alpha.clamp(0, 1)
        return (1 - alpha) * x + alpha * self.activation_fn(x)


class RealMLPHead(nn.Module):
    """RealMLP-TD Head for tabular data regression.

    Architecture:
    - Scaling layer for feature selection
    - 3 hidden layers with 256 neurons each
    - NTP linear layers
    - Parametric Mish activation for regression
    - Dropout after each activation
    - Output clipping for regression (positive values with 50% margin)
    """

    def __init__(
        self,
        input_dim: int,
        num_outputs: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        compute_aggregates: bool = False,
        activation: nn.Module | None = None,
        use_output_clipping: bool = True,
        max_output_margin: float = 0.5,
    ) -> None:
        """Initialize RealMLP Head.

        Args:
            input_dim: Input feature dimension
            num_outputs: Number of output values
            hidden_dim: Hidden layer dimension (256 by default as per paper)
            dropout: Dropout rate
            compute_aggregates: If True, model learns only 3 base components
                (clover, dead, green) and computes GDM and Total arithmetically
            activation: Optional activation function after output layer
            use_output_clipping: If True, clip outputs at test time
            max_output_margin: Margin for upper bound (50% = 1.5x max observed)
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_outputs = num_outputs
        self.compute_aggregates = compute_aggregates
        self.activation = activation
        self.use_output_clipping = use_output_clipping
        self.max_output_margin = max_output_margin

        # Register buffers for output clipping (set during training)
        self.register_buffer('output_min', torch.zeros(num_outputs))
        self.register_buffer('output_max', torch.ones(num_outputs))
        self.register_buffer('clipping_initialized', torch.tensor(False))

        # 1. Scaling layer for soft feature selection
        self.scaling_layer = ScalingLayer(input_dim)

        # 2. Three hidden layers with 256 neurons each (using NTP)
        self.layer1 = NTPLinear(input_dim, hidden_dim)
        self.activation1 = ParametricActivation(hidden_dim, nn.Mish())
        self.dropout1 = nn.Dropout(dropout)

        self.layer2 = NTPLinear(hidden_dim, hidden_dim)
        self.activation2 = ParametricActivation(hidden_dim, nn.Mish())
        self.dropout2 = nn.Dropout(dropout)

        self.layer3 = NTPLinear(hidden_dim, hidden_dim)
        self.activation3 = ParametricActivation(hidden_dim, nn.Mish())
        self.dropout3 = nn.Dropout(dropout)

        # 3. Output layer
        head_outputs = 3 if self.compute_aggregates else self.num_outputs
        self.output_layer = NTPLinear(hidden_dim, head_outputs)

        # Track if data-dependent initialization has been performed
        self._data_dependent_init_done = False

    def set_output_bounds(self, targets: torch.Tensor) -> None:
        """Set output bounds based on observed training data.

        Args:
            targets: Training targets (N, num_outputs)
        """
        # Minimum is 0 (targets are always positive)
        self.output_min = torch.zeros(targets.shape[1], device=targets.device)

        # Maximum with margin (50% higher than observed max)
        max_vals = targets.max(dim=0)[0]
        self.output_max = max_vals * (1 + self.max_output_margin)

        self.clipping_initialized = torch.tensor(True)

    def initialize_weights_data_dependent(
        self, x_sample: torch.Tensor, target_std: float = 1.0
    ) -> None:
        """Data-dependent initialization from RealMLP-TD paper.

        Rescales weight matrix rows to make output pre-activation variance equal to
        target_std^2 over the dataset. Uses hull+5 initialization for biases.

        Args:
            x_sample: Sample of training data (N, input_dim) to compute statistics
            target_std: Target standard deviation for pre-activations (default 1.0)
        """
        if self._data_dependent_init_done:
            return

        self.eval()  # Use eval mode for initialization
        with torch.no_grad():
            # Forward through layers and rescale weights to achieve unit variance
            x = x_sample

            # Layer 1
            x = self.scaling_layer(x)
            pre_act1 = self.layer1(x)
            std1 = pre_act1.std()
            if std1 > 1e-8:
                self.layer1.weight.data *= target_std / std1
            # hull+5 bias init: set bias to 5th percentile of pre-activations
            self.layer1.bias.data = torch.quantile(pre_act1, 0.05, dim=0)

            x = self.activation1(self.layer1(x))
            x = self.dropout1(x)

            # Layer 2
            pre_act2 = self.layer2(x)
            std2 = pre_act2.std()
            if std2 > 1e-8:
                self.layer2.weight.data *= target_std / std2
            self.layer2.bias.data = torch.quantile(pre_act2, 0.05, dim=0)

            x = self.activation2(self.layer2(x))
            x = self.dropout2(x)

            # Layer 3
            pre_act3 = self.layer3(x)
            std3 = pre_act3.std()
            if std3 > 1e-8:
                self.layer3.weight.data *= target_std / std3
            self.layer3.bias.data = torch.quantile(pre_act3, 0.05, dim=0)

            x = self.activation3(self.layer3(x))
            x = self.dropout3(x)

            # Output layer
            pre_act_out = self.output_layer(x)
            std_out = pre_act_out.std()
            if std_out > 1e-8:
                self.output_layer.weight.data *= target_std / std_out
            self.output_layer.bias.data = torch.quantile(pre_act_out, 0.05, dim=0)

        self._data_dependent_init_done = True
        self.train()  # Return to train mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, input_dim)

        Returns:
            Predictions (B, num_outputs)
        """
        # Feature scaling layer
        x = self.scaling_layer(x)

        # Hidden layer 1
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.dropout1(x)

        # Hidden layer 2
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.dropout2(x)

        # Hidden layer 3
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.dropout3(x)

        # Output layer
        output = self.output_layer(x)

        # Optional activation
        if self.activation is not None:
            output = self.activation(output)

        # Compute aggregates if needed
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

        # Output clipping at test time (if bounds are initialized)
        if not self.training and self.use_output_clipping and self.clipping_initialized:
            # Clip to [0, max_with_margin]
            # Using 0 as minimum since targets are always positive
            if self.compute_aggregates:
                # Only clip base components, aggregates are derived
                base_outputs = output[:, :3]
                base_outputs = torch.clamp(base_outputs,
                                          min=self.output_min[:3],
                                          max=self.output_max[:3])

                # Recompute aggregates with clipped values
                clover = base_outputs[:, 0:1]
                dead = base_outputs[:, 1:2]
                green = base_outputs[:, 2:3]
                gdm = green + clover
                total = green + clover + dead
                output = torch.cat([clover, dead, green, gdm, total], dim=1)
            else:
                output = torch.clamp(output, min=self.output_min, max=self.output_max)

        return output

    def get_scaling_factors(self) -> torch.Tensor:
        """Get learned feature scaling factors.

        Returns:
            Scaling factors (input_dim,)
        """
        return self.scaling_layer.scales.data

    def get_activation_alphas(self) -> dict[str, torch.Tensor]:
        """Get learned alpha values for parametric activations.

        Returns:
            Dictionary mapping layer names to alpha values
        """
        return {
            'layer1': self.activation1.alpha.data,
            'layer2': self.activation2.alpha.data,
            'layer3': self.activation3.alpha.data,
        }
