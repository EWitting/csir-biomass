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
│   │   ├── biomass_*.yaml         # EfficientNet-based models
│   │   ├── foundation_*.yaml      # DINOv2/v3 foundation models
│   │   └── foundation_mlae_dinov3.yaml  # MLAE fine-tuning
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
├── submission/             # Kaggle submission utilities
│   ├── manage_datasets.py  # Upload code/models to Kaggle
│   └── config/             # Dataset configurations (source.json, dependencies.json)
├── notebooks/              # EDA and experimentation
├── train.py                # Single train-val split
├── cv.py                   # K-fold cross-validation
└── submit.py               # Generate Kaggle submissions
```

## Quick Start

### 1. Installation

This project uses [UV](https://github.com/astral-sh/uv) for dependency management:

```bash
uv sync
```

**Important**: Always run commands with `uv run`, not `python`:
```bash
uv run train      # NOT: python train.py
uv run cv         # NOT: python cv.py
```

### 2. Weights & Biases Setup

The codebase uses W&B for experiment tracking. You may need to configure this for your account:

1. Edit [src/setup/setup_wandb.py](src/setup/setup_wandb.py):
   - Line 30: Change `project_name` if needed
   - Line 39: Change `entity` to your W&B username or team name

2. Enable W&B logging in [conf/wandb/train.yaml](conf/wandb/train.yaml) after initial setup works

Current defaults:
- Project: `csir-biomass`
- Entity: `team-epoch-iv`

## Available Models

The codebase includes several model approaches:

### 1. Standard Training from Scratch
- **Config**: [conf/model/biomass_sliding_window_global.yaml](conf/model/biomass_sliding_window_global.yaml)
- **Description**: EfficientNet trained from scratch with sliding window aggregation
- **Best for**: Baseline experiments

### 2. Foundation Models (Frozen Features)
- **Configs**:
  - [conf/model/foundation_ensemble_224_fast.yaml](conf/model/foundation_ensemble_224_fast.yaml)
  - [conf/model/foundation_ensemble_384_best.yaml](conf/model/foundation_ensemble_384_best.yaml)
  - [conf/model/foundation_ensemble_448_extreme.yaml](conf/model/foundation_ensemble_448_extreme.yaml)
- **Description**: Frozen DINOv2/v3 features + AutoGluon tabular models
- **Best for**: Quick experiments with strong baselines

### 3. Foundation Models with Fine-Tuning (MLAE)
- **Config**: [conf/model/foundation_mlae_dinov3.yaml](conf/model/foundation_mlae_dinov3.yaml)
- **Description**: DINOv3 fine-tuned using Masked LoRA Experts (parameter-efficient)
- **Best for**: State-of-the-art performance with efficient fine-tuning
- **See**: [MLAE_README.md](MLAE_README.md) for detailed documentation

### 4. Simple MLP Head
- **Configs**: [conf/model/foundation_mlp.yaml](conf/model/foundation_mlp.yaml), [conf/model/foundation_realmlp.yaml](conf/model/foundation_realmlp.yaml)
- **Description**: Frozen features + MLP regression head
- **Best for**: Fast prototyping

## Basic Usage

### Running Training

```bash
# Single train-val split (80-20)
uv run train

# 5-fold cross-validation
uv run cv

# Use a specific model
uv run train model=foundation_mlae_dinov3
uv run cv model=foundation_ensemble_384_best
```

### Changing Model Parameters

Use Hydra command-line overrides:

```bash
# Change batch size
uv run train model.train_sys.steps[0].batch_size=16

# Change learning rate
uv run train model.train_sys.steps[0].optimizer.lr=1e-4

# Change image size
uv run train model.train_sys.steps[0].image_size=384
```

### Kaggle Submission

#### First-Time Setup

1. **Configure Kaggle Datasets**: Run the interactive setup:
   ```bash
   uv run python submission/manage_datasets.py --help
   ```

   You'll be prompted to configure:
   - Source code dataset name and ID
   - Dependencies dataset name and ID
   - Your Kaggle API credentials (if not already configured)

   **Note**: If you don't have access to the existing datasets, you'll need to create your own:
   - Create a new Kaggle dataset for source code
   - Create a new Kaggle dataset for dependencies
   - Update `submission/config/source.json` and `submission/config/dependencies.json` with your dataset IDs

2. **Test locally** (optional):
   ```bash
   # Edit conf/submit.yaml to point to local test data
   uv run submit
   ```
   This creates `submission.csv` in the required format.

#### Submitting to Kaggle

**Finding Your Model Hash**:
- During training, the model hash is printed in the logs
- When testing `submit.py` locally (with local paths in `conf/submit.yaml`), the hash is shown during model loading
- Model files are saved as `tm/{hash}.pt` (or `tm/{hash}_fold_{n}.pt` for cross-validation)
- **Hash filtering uses `startswith()`**: Providing a hash prefix will match ALL files starting with that hash (e.g., all folds of a model)

**Upload Dependencies** (first time or when packages change):
```bash
uv run python submission/manage_datasets.py --dependencies
```
This compiles `requirements.txt` from `pyproject.toml` and uploads Python packages.

**Managing Package Versions**:
- Edit [submission/config/package_config.json](submission/config/package_config.json) to control package uploads:
  - `excluded_packages`: Packages to exclude from upload (already on Kaggle)
  - `forced_packages`: Packages to force-include with specific versions (overrides Kaggle defaults)
  - Example: `"forced_packages": ["transformers>=4.57.0"]` ensures newer transformers version
  - `auto_exclude_kaggle_packages`: Auto-exclude all packages from `kaggle_container_packages.txt`

**Upload Source Code and Models**:

```bash
# Upload ALL trained models in tm/ folder
uv run python submission/manage_datasets.py --source

# Upload SPECIFIC model(s) by hash (recommended)
uv run python submission/manage_datasets.py --source --model-hash abc123def456

# Upload multiple specific models
uv run python submission/manage_datasets.py --source --model-hash abc123 def456 xyz789
```

**What gets uploaded with `--source`**:
- `src/`, `conf/`, `submit.py` (code)
- `tm/hf_models/` (HuggingFace model configs for offline use - always included)
- `tm/{hash}.pt` files (trained model weights):
  - Without `--model-hash`: **ALL models** in `tm/` folder
  - With `--model-hash`: **ONLY specified models**

**Upload Both** (dependencies + source):
```bash
uv run python submission/manage_datasets.py --dependencies --source --model-hash abc123
```

**Submit on Kaggle**:
- Open your Kaggle notebook
- Refresh the dataset to get the latest version
- Run the notebook to generate predictions
- Submit to the competition

**HuggingFace Models (DINOv3 MLAE)**:
- HuggingFace model configs are automatically saved to `tm/hf_models/` on first run with internet
- These configs are always included in uploads (small files ~few KB)
- Allows models to work offline on Kaggle without internet access

## Development Guide

### Adding New Functionality

#### Creating a New Model

1. **Create a model configuration** in `conf/model/my_new_model.yaml`:

```yaml
# Most of the pipeline structure is inherited from conf/model/pipeline/default.yaml
# You only need to specify what changes

train_sys:
  steps:
    - _target_: src.modules.training.main_trainer.MainTrainer
      model_name: "MyModel"
      epochs: 100
      batch_size: 16
      model:
        _target_: timm.create_model
        model_name: efficientnet_b0
        pretrained: true
        num_classes: 5
      criterion:
        _target_: torch.nn.MSELoss
      optimizer:
        _target_: functools.partial
        _args_:
          - _target_: hydra.utils.get_class
            path: torch.optim.Adam
        lr: 1e-3
```

2. **Run your model**:
```bash
uv run train model=my_new_model
```

#### Creating Custom Training Logic

1. **Create a new trainer class** in `src/modules/training/`:

```python
from dataclasses import dataclass
from src.modules.training.main_trainer import MainTrainer

@dataclass
class MyCustomTrainer(MainTrainer):
    """Custom trainer with special logic."""

    def train_epoch(self, ...):
        # Your custom training logic
        pass
```

2. **Reference it in your config**:
```yaml
train_sys:
  steps:
    - _target_: src.modules.training.my_custom_trainer.MyCustomTrainer
      # ... parameters
```

#### Adding Preprocessing Steps

Add transformation blocks to `x_sys` (for input images) or `y_sys` (for labels):

```yaml
x_sys:
  steps:
    - _target_: src.modules.transformation.image_loader.ImageLoader
      # ... parameters
    - _target_: src.modules.transformation.my_preprocessor.MyPreprocessor
      # ... parameters
```

### Configuration Philosophy

This codebase follows a **config-driven** approach using Hydra:

**DO**: Define classes that can be instantiated directly from config
```python
@dataclass
class MyModel:
    loss: nn.Module  # Instantiated from config
    learning_rate: float
```

**DON'T**: Use string-based configuration with manual parsing
```python
class MyModel:
    def __init__(self, loss: str):  # ❌ Manual parsing needed
        if loss == "mse":
            self.loss = nn.MSELoss()
        # ...
```

### Pipeline Structure

All models follow a 5-stage pipeline (inherited from `conf/model/pipeline/default.yaml`):

1. **x_sys**: Image loading and preprocessing (e.g., resize, normalize)
2. **y_sys**: Target transformations (usually empty)
3. **train_sys**: Model training (your PyTorch trainer goes here)
4. **pred_sys**: Post-process predictions (e.g., clipping, scaling)
5. **label_sys**: Transform labels for final scoring (usually empty)

### Logging

When extending `VerboseTrainingBlock` or `VerboseTransformationBlock`:

```python
self.log_to_terminal("Message for console")
self.log_to_debug("Debug information")
self.log_to_warning("Warning message")
self.log_to_external({"metric_name": value})  # W&B logging
```

### Caching

- Preprocessing results (`x_sys`, `y_sys`) are **automatically cached** in `data/processed/`
- Cache is based on configuration hash
- Delete cache folder to force recomputation
- Speeds up experimentation when only changing model hyperparameters

### Data Folder Structure

```
data/
├── raw/              # Original competition data (DO NOT MODIFY)
│   ├── train.csv
│   ├── test.csv
│   └── images/
└── processed/        # Auto-generated cache (safe to delete)
```

**Rule**: Keep `data/raw/` exactly as downloaded from Kaggle (after unzipping).

### Dependencies

**Adding new packages**:
```bash
uv add package-name          # Adds to pyproject.toml and installs
uv add --dev package-name    # Development dependency
```

**Updating dependencies**:
```bash
uv sync                      # Sync environment with pyproject.toml
```

**DO NOT**: Manually edit `pyproject.toml` to add packages (except for PyTorch GPU sources).

### Common Gotchas

1. **Always use `uv run`**: Don't use bare `python` commands
2. **No unicode characters**: Python files must use ASCII encoding only (no emojis)
3. **Use dataclasses**: Minimize boilerplate with `@dataclass` decorator
4. **Trust the config**: Let Hydra instantiate objects instead of manual factory patterns

## Additional Documentation

- **[epochlib docs](https://github.com/TeamEpochGithub/epochlib)**: Framework documentation

---
