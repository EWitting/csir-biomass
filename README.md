# CSIR Biomass Prediction

Multi-output regression model predicting 5 biomass components from images.

## Competition Overview

**Targets** (alphabetical order):
- `Dry_Clover_g`, `Dry_Dead_g`, `Dry_Green_g`, `GDM_g`, `Dry_Total_g`

**Metric**: Weighted R² with weights `[0.1, 0.1, 0.1, 0.2, 0.5]`

## Project Structure

```
├── conf/                    # Hydra configuration files
│   ├── model/              # Model pipeline definitions
│   │   └── biomass_base.yaml  # Current model: EfficientNet-B0
│   ├── train.yaml          # Training config (single split)
│   ├── cv.yaml             # Cross-validation config (5-fold)
│   └── submit.yaml         # Inference config for Kaggle
├── data/
│   ├── raw/                # train.csv, test.csv, images/
│   └── processed/          # Cached preprocessing outputs
├── src/
│   ├── modules/
│   │   ├── training/       # Trainers, models, losses, augmentations
│   │   └── transformation/ # Image loading and preprocessing
│   ├── scoring/            # WeightedR2 scorer
│   └── setup/              # Data loading and W&B setup
├── notebooks/              # EDA and experimentation
├── train.py                # Single train-val split
├── cv.py                   # K-fold cross-validation
└── submit.py               # Generate Kaggle submissions
```

## Setup

### Installation

This project uses [UV](https://github.com/astral-sh/uv) for dependency management:

```bash
uv sync
```

### Weights & Biases

Project name: `csir-biomass`  
Entity: `team-epoch-iv`

Enable logging by modifying `conf/wandb/train.yaml` after initial setup is working.

## Current Model

**Architecture**: EfficientNet-B0 (timm) → Global Avg Pool → Dropout(0.3) → Linear(5 outputs)

**Key features**:
- Image size: 224×224
- ImageNet pre-trained backbone
- Kornia augmentations (horizontal/vertical flips, TrivialAugment)
- Huber loss with weighted R² scaling
- Cosine LR scheduler with warmup
- Mixed precision training
- Test-time augmentation (TTA) with horizontal flip

See `conf/model/biomass_base.yaml` for full configuration.

## Usage

### Training

```bash
# Single train-val split (80-20)
uv run train

# 5-fold cross-validation
uv run cv

# Override model config
uv run train model=biomass_base
```

### Generate Submission
Test with `uv run submit` to ensure it works. (change the path in submit.yaml for local testing)
Creates `submission.csv` in format required by Kaggle.
Use `uv run submission/manage_datasets.py` to upload to Kaggle. Usually no dependency update needed when prompted, but to say yes to uploading source code.
In Kaggle, open the notebook, refresh the dataset to get the latest version, and submit.

### Model Selection

Models are defined in `conf/model/`. To use a different model:

1. Create new config file: `conf/model/my_model.yaml`
2. Run: `uv run train model=my_model`

Or use Hydra overrides:
```bash
uv run train model.train_sys.steps[0].batch_size=32
```

## Development Workflow

### Configuration Philosophy

This codebase uses **Hydra instantiate** to minimize Python boilerplate. Define objects directly in YAML configs using `_target_`:

```yaml
# Good: Direct instantiation
criterion:
  _target_: torch.nn.HuberLoss
  delta: 1.0

# Bad: String parsing in Python
criterion: "huber"  # Then parse in __init__
```

### Pipeline Structure

Models follow a 5-stage pipeline (see `conf/model/pipeline/default.yaml`):

1. **x_sys**: Image loading and preprocessing
2. **y_sys**: Target transformations (currently empty)
3. **train_sys**: Model training (PyTorch trainer)
4. **pred_sys**: Post-process predictions (currently empty)
5. **label_sys**: Transform labels for final scoring (currently empty)


### Logging

When inheriting from `VerboseTrainingBlock` or `VerboseTransformationBlock`:

```python
self.log_to_terminal("Message")
self.log_to_debug("Debug info")
self.log_to_external({"metric": value})  # W&B logging
```

### Caching

Preprocessing results (`x_sys`, `y_sys`) are automatically cached in `data/processed/`. Delete cache to force recomputation.
loader.py**: Loads and resizes images from paths
---

Built with [epochlib](https://github.com/TeamEpochGithub/epochlib)
