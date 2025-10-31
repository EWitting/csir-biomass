# CSIR Biomass Competition Setup - Summary

This repository has been configured for the CSIR Biomass prediction competition on Kaggle.

## Competition Overview
- **Task**: Multi-output regression predicting 5 biomass components from pasture images
- **Targets**: Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g
- **Metric**: Weighted R² (weights: 0.1, 0.1, 0.1, 0.2, 0.5 respectively)
- **Data**: 357 training images, 800+ test images

## What Was Implemented

### 1. Data Loading (`src/setup/setup_data.py`)
- Loads train.csv and groups by image (5 targets per image)
- Returns image file paths (not loaded in memory for efficiency)
- Proper handling of multi-output format

### 2. Image Preprocessing (`src/modules/transformation/image_loader.py`)
- Loads and resizes images to 224x224
- Optional dynamic mean/std computation from dataset
- Currently uses ImageNet defaults for transfer learning

### 3. Augmentation System
- **`src/modules/training/augmentation.py`**: Wrapper for Kornia GPU augmentations
- **`src/modules/training/dataset.py`**: Custom dataset with batch augmentation via `__getitems__`
- Supports separate train and validation augmentations

### 4. Model Architecture (`src/modules/training/models/biomass_model.py`)
- Uses timm backbone (efficientnet_b0) properly instantiated via Hydra
- Custom regression head with dropout
- Outputs 5 continuous values

### 5. Weighted Loss (`src/modules/training/losses.py`)
- Wrapper that accepts any base loss (HuberLoss by default)
- Applies competition weights per-target: [0.1, 0.1, 0.1, 0.2, 0.5]
- Aligns loss optimization with evaluation metric

### 6. Training Pipeline (`src/modules/training/main_trainer.py`)
- Gradient accumulation support
- Mixed precision training
- Test-time augmentation (TTA) for inference
- Proper handling of multi-output regression
- Integration with augmented dataset

### 7. Evaluation (`src/scoring/weighted_r2.py`)
- Implements competition's weighted R² metric
- Computes R² per target, then weighted average
- Matches Kaggle leaderboard calculation

### 8. Configuration (`conf/model/biomass_base.yaml`)
- Complete pipeline with:
  - EfficientNet-B0 backbone via timm
  - Weighted Huber loss
  - AdamW optimizer with cosine LR schedule
  - Kornia augmentations (flips, rotation, color jitter, blur, noise)
  - TTA enabled
  - 100 epochs, early stopping patience=15

### 9. Dependencies
- Added: `timm`, `albumentations`, `kornia`, `Pillow`
- Configured PyTorch GPU sources for CUDA 11.8
- All via `uv add` with proper version resolution

### 10. WandB Integration
- Project name: `csir-biomass`
- Entity: `team-epoch-iv`
- Logging enabled for config and code

### 11. EDA Notebook (`notebooks/eda.ipynb`)
- Data loading and structure analysis
- Target distribution visualization
- Correlation analysis
- Sample image visualization

## How to Use

### Install Dependencies
```bash
uv sync
```

### Run Training
```bash
uv run train
```

### Run Cross-Validation
```bash
uv run cv
```

### Switch Models
```bash
uv run train model=biomass_base
```

## Key Design Decisions

1. **Multi-output approach**: Single model predicting all 5 targets simultaneously
2. **Image-only**: No metadata features (NDVI, height, species) for now - can be added later
3. **Weighted loss**: Aligns training objective with competition metric
4. **Dynamic instantiation**: Models and losses instantiated via Hydra (no hardcoded strings)
5. **GPU augmentation**: Using Kornia for fast on-GPU augmentations
6. **TTA**: 8x test-time augmentation (2 flips × 4 rotations) for better predictions

## Next Steps

1. Run EDA notebook to understand data characteristics
2. Train baseline model: `uv run train`
3. Monitor training on WandB
4. Iterate on:
   - Hyperparameters (learning rate, batch size, augmentation strength)
   - Model architecture (try different backbones)
   - Loss function (consider additional losses or weights)
   - Add metadata features if needed
5. Submit predictions to Kaggle leaderboard

## Files Created/Modified

**Created:**
- `src/modules/transformation/image_loader.py`
- `src/modules/training/augmentation.py`
- `src/modules/training/dataset.py`
- `src/modules/training/models/biomass_model.py`
- `src/modules/training/losses.py`
- `src/scoring/weighted_r2.py`
- `conf/model/biomass_base.yaml`
- `notebooks/eda.ipynb`

**Modified:**
- `src/setup/setup_data.py` - Multi-output data loading
- `src/modules/training/main_trainer.py` - Extended with augmentation, TTA, gradient accumulation
- `src/setup/setup_wandb.py` - Set project name
- `conf/train.yaml` - Updated paths, model, scorer
- `pyproject.toml` - Added CV dependencies and PyTorch GPU sources

## Notes

- The template follows Hydra instantiate patterns (see README_FOR_AI_AGENT.md)
- All configurations are defined in YAML, minimal Python code
- Caching is enabled for x_sys and y_sys pipelines
- GPU training is configured for CUDA 11.8

