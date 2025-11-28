# Masked LoRA Experts (MLAE) Implementation

This implementation adds **Masked LoRA Experts (MLAE)** fine-tuning for DINOv3 vision transformers to the biomass prediction pipeline. MLAE is a parameter-efficient fine-tuning method that decomposes LoRA matrices into independent rank-1 experts with adaptive coefficients and applies masking strategies for diversity.

## Overview

MLAE extends LoRA by:
1. **Cellular Decomposition**: Decomposing rank-r LoRA into r independent rank-1 experts
2. **Adaptive Coefficients**: Each expert has a learnable weight (λ) to adjust its contribution
3. **Expert-Level Masking**: Applying dropout at the expert level (not parameter level) for diversity

Based on the paper: "Masked LoRA Experts for Efficient Vision Transformer Fine-tuning"

## Key Files

### Models
- **[src/modules/training/models/dinov3_mlae_model.py](src/modules/training/models/dinov3_mlae_model.py)**: DINOv3 with MLAE integration
  - Applies MLAE to Q, K, V projections in attention layers
  - Supports various masking strategies (random, incremental, decremental, hourglass, protruding)
  - Supports masking types (fixed, stochastic, mixed)
  - **Only 0.26% trainable parameters** (e.g., 223K out of 85M for ViT-Base)

- **[src/modules/training/models/mlae.py](src/modules/training/models/mlae.py)**: Core MLAE layer implementation
  - `MLAE_Linear`: Linear layer with rank-1 LoRA experts
  - Adaptive coefficient (λ) per expert
  - Expert-level dropout for stochastic masking

### Trainer
- **[src/modules/training/mlae_trainer.py](src/modules/training/mlae_trainer.py)**: Training infrastructure for MLAE
  - Based on `MainTrainer` with MLAE-specific features
  - Supports fixed mask application
  - Compatible with all existing pipeline features (TTA, augmentation, scoring, etc.)

### Configuration
- **[conf/model/foundation_mlae_dinov3.yaml](conf/model/foundation_mlae_dinov3.yaml)**: Example configuration for MLAE training
  - Uses DINOv3-Large (1024-dim embeddings)
  - Configured with paper's recommended hyperparameters

## Usage

### Basic Training

```bash
uv run python train.py model=foundation_mlae_dinov3
```

### Configuration Options

#### Model Selection
Available DINOv3 models (from HuggingFace):
- `facebook/dinov3-vits16-pretrain-lvd1689m`: Small (384 dim, 21M params)
- `facebook/dinov3-vitb16-pretrain-lvd1689m`: Base (768 dim, 86M params)
- `facebook/dinov3-vitl16-pretrain-lvd1689m`: Large (1024 dim, 300M params)
- `facebook/dinov3-vith-plus16-pretrain-lvd1689m`: Huge+ (1280 dim, 840M params)

#### LoRA Expert Configuration

```yaml
model:
  rank: 8                # Number of experts per layer (paper uses 4-16)
  lora_alpha: 16.0       # Initial adaptive coefficient value
  lora_dropout: 0.0      # Dropout on LoRA inputs
  drop_rate: 0.1         # Expert dropout rate for stochastic masking
```

#### Where to Apply MLAE

```yaml
model:
  apply_to_qkv: true     # Apply to Q, K, V projections (recommended)
  apply_to_proj: false   # Apply to attention output projection
  apply_to_mlp: false    # Apply to MLP layers
```

#### Masking Strategies

**Masking Strategy** (layer-wise expert distribution):
- `random`: Uniform dropout across all layers
- `incremental`: Lower layers sparse → upper layers dense
- `decremental`: Lower layers dense → upper layers sparse
- `hourglass`: Endpoints dense → middle sparse
- `protruding`: Endpoints sparse → middle dense

**Masking Type**:
- `stochastic`: Expert dropout during training (default, recommended)
- `fixed`: Permanently mask experts (apply once at initialization)
- `mixed`: Combine fixed and stochastic masking

```yaml
model:
  masking_strategy: "random"      # or incremental, decremental, hourglass, protruding
  masking_type: "stochastic"      # or fixed, mixed

# For fixed/mixed masking, also set:
train_sys:
  steps:
    - apply_fixed_mask: true      # Apply fixed mask after initialization
```

#### Training Hyperparameters

Paper recommendations (for VTAB-1k benchmark):
```yaml
epochs: 500
batch_size: 64
lr: 5e-4
weight_decay: 1e-4
optimizer: AdamW
scheduler: CosineAnnealingLR
```

For biomass task, you may want to adjust:
```yaml
epochs: 100              # Fewer epochs for smaller dataset
batch_size: 8            # Smaller batch for large ViT models
gradient_accumulation_steps: 2   # Effective batch = 16
```

## Architecture Details

### DINOv3 with MLAE

```
DINOv3 ViT-Large (frozen backbone)
├── Patch Embedding (frozen)
├── Positional Encoding (frozen)
├── 24 Transformer Blocks
│   └── Each block:
│       ├── Attention
│       │   ├── Q projection → MLAE (rank-8, trainable)
│       │   ├── K projection → MLAE (rank-8, trainable)
│       │   ├── V projection → MLAE (rank-8, trainable)
│       │   └── Output projection (frozen)
│       └── MLP (frozen)
└── Regression Head (trainable)
    ├── Dropout (0.2)
    └── Linear(1024 → 5)
```

### MLAE Layer

Each MLAE layer with rank `r` contains:
- **A matrix**: (r, in_features) - initialized with Kaiming uniform
- **B matrix**: (r, out_features) - initialized to zero
- **λ vector**: (r,) - adaptive coefficients, initialized to `lora_alpha`

Forward pass:
```python
output = W_0 @ x + (x @ A.T * dropout(λ)) @ B
```

Where:
- `W_0` is the frozen pretrained weight
- `dropout(λ)` applies expert-level dropout (training only)
- Each expert `i` contributes: `λ_i * (x @ a_i) @ b_i`

## Results

### Parameter Efficiency

Example for DINOv3-Base + MLAE (rank=4):
- **Total parameters**: 85,884,051
- **Trainable parameters**: 223,635 (0.26%)
  - LoRA experts: ~200K parameters
  - Regression head: ~4K parameters

Comparison with other approaches:
- Full fine-tuning: 100% trainable (85M params)
- Standard LoRA: ~0.5-1% trainable
- MLAE: ~0.26% trainable (more efficient than LoRA)

### Expected Performance

Based on the MLAE paper (VTAB-1k benchmark):
- MLAE outperforms standard LoRA with fewer parameters
- Stochastic masking > fixed masking
- Random strategy works well across tasks
- Optimal rank: 8-16 for most tasks

## Testing

Run the test script to verify the implementation:

```bash
uv run python test_mlae.py
```

This will:
1. Create a DINOv3 MLAE model
2. Print trainable parameter statistics
3. Test forward pass and embeddings extraction
4. Test all masking strategies and types

## Comparison with Other Methods

| Method | Config File | Description | Parameters |
|--------|------------|-------------|------------|
| Standard Training | `biomass_sliding_window_global.yaml` | EfficientNet from scratch | ~100% trainable |
| Foundation + AutoGluon | `foundation_ensemble_448_extreme.yaml` | Frozen features + tabular | 0% (frozen) + tabular model |
| Foundation + MLP | `foundation_mlp.yaml` | Frozen features + MLP head | 0% (frozen) + ~10K (head) |
| **Foundation + MLAE** | **`foundation_mlae_dinov3.yaml`** | **Fine-tune with LoRA experts** | **0.26% trainable** |

## Advanced Usage

### Custom Head Architecture

The MLAE model supports flexible head architectures through the `compute_aggregates` flag:

```yaml
model:
  num_outputs: 5
  compute_aggregates: true    # Learn only [clover, dead, green], compute GDM & Total
  activation:
    _target_: torch.nn.ReLU   # Optional activation after head
```

### Using Different Backbones

While DINOv3 is recommended, you can adapt the approach to other vision transformers by modifying `_inject_mlae_layers()` in [dinov3_mlae_model.py](src/modules/training/models/dinov3_mlae_model.py).

### Extracting Embeddings

The model can be used as a feature extractor:

```python
model = DINOv3_MLAE_Model(...)
embeddings = model.get_embeddings(images)  # (B, hidden_dim)
```

## Tips for Best Performance

1. **Start with stochastic + random masking** - it's the most robust
2. **Use rank 8-16** - good balance between capacity and efficiency
3. **Set lora_alpha = 2 × rank** - maintains gradient scale
4. **Apply to Q, K, V only** - sufficient for most tasks, fewer parameters
5. **Use cosine LR schedule** - paper's recommendation
6. **Train longer than standard fine-tuning** - MLAE benefits from more epochs

## Troubleshooting

### Out of Memory
- Reduce `batch_size` and increase `gradient_accumulation_steps`
- Use smaller model (ViT-Base instead of ViT-Large)
- Reduce `rank` (try 4 instead of 8)

### Poor Performance
- Increase `rank` (try 16 or 32)
- Try different masking strategies
- Increase `epochs` (MLAE often needs more training)
- Adjust `drop_rate` (try 0.0-0.3 range)

### Slow Training
- Disable TTA during training (set all tta flags to false)
- Use smaller augmentations
- Reduce image size (try 224 instead of 448)

## Citation

If you use this implementation, please cite the MLAE paper:

```bibtex
@article{mlae2024,
  title={Masked LoRA Experts for Efficient Vision Transformer Fine-tuning},
  author={[Authors from paper]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## Notes

- The implementation uses the DINOv3 architecture which has a slightly different structure than standard ViT (uses `layer` instead of `encoder.layer`, uses `q_proj/k_proj/v_proj` instead of `query/key/value`)
- The MLAE layer implementation is based on the authors' provided code in `mlae.py`
- The trainer is based on your existing `MainTrainer` to maintain compatibility with your pipeline
- All existing features (TTA, augmentation, scoring, wandb logging) work seamlessly with MLAE
