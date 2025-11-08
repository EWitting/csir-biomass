This codebase is set up specifically for the workflow of (Kaggle) ML competitions. It is cloned from `Apoch`, which is a template that imports a lot of functionality from `epochlib` at https://github.com/TeamEpochGithub/epochlib the latest version is downloaded at /lib, you can explore the codebase there if necessary.

Pipelines are structured using Hydra instantiate and follow a specific pattern. Some parts are disabled during inference on Kaggle.
Wandb is used for logging and hyperparameter tuning.

# Design philosophy (IMPORTANT):
- Minimize python code, and don't use python code to parse configurations. Define classes and objects that can be instantiated through the configuration, and aim to directly support libraries. E.g.:
- Use @dataclass to minimize boilerplate code.

WRONG:
```python
class Trainer:
    def __init__(self, loss: str, ...):
        if self.loss == "mse":
            self.loss_fn = torch.nn.MSELoss()
        elif self.loss == "bce":
            self.loss_fn = torch.nn.BCELoss()
        elif self.loss == "cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid loss: {self.loss}")
```
```yaml
model: Trainer
  loss: "mse"
  ...
```


CORRECT:
```python
@dataclass
class Trainer:
    loss: nn.Module

    def __post_init__(self):
        super().__init__()
```

```yaml
  model:
    _target_: src.modules.training.models.trainer.Trainer
    loss:
      _target_: torch.nn.MSELoss
```

# For a new competion:
- update pyproject.toml, preferrably use UV sync. See notes below.
- set entity and project name in src\setup\setup_wandb.py
- modify setup_data to read data. can be in any format necessary, as file paths or loaded data objects depending on competition memory constraints.
- define a scoring function in scoring/. And select in train.yaml (and cv.yaml).
- choose a trainer. usually a torch model, in which case the TorchTrainer from epochlib supplies all necessary functionality, by default extended as MainTrainer in src/modules/training/main_trainer.py.
- create a model class, or directly instantiate the object through the configuration from a library (e.g. segmentation_models_pytorch)
- depending on the competition. you might want some operations to happen to data during training, and not preprocessed, e.g. random data augmentation. A common pattern is to create a custom dataset class that applies a list of augmentations of the appropriate data type. Pass this list to the trainer in the config, and extend/override the trainer class to use this dataset format.
- create a configuration file that describes the pipeline. see below.
- modify submit.py to create the submission format required by the competition. There is often a sample submission file in the data/raw folder.
- select the model in train.yaml (or cv.yaml), and run with commands as below.
- when framework succesfully created. enable wandb in conf/wandb/train.yaml to start logging runs
- for kaggle submissions, see below

# Hydra configuration:
A model pipeline is defined in a single yaml file. Which includes training parameters. Instantiate uses the _target_ pattern. Note that segmentation_models_pytorch is used directly. No python file is created for this model. Note also the partial. This is necessary for objects that need to be instantiated at runtime, e.g. the optimizer based on model parameters. Example:
```yaml	
train_sys:
  steps:
    # Keep the original MainTrainer
    - _target_: src.modules.training.main_trainer.MainTrainer
      model_name: "SMP"
      n_folds: 0 # 0 for train full,
      epochs: 100
      patience: -1
      batch_size: 8
      gradient_accumulation_steps: 2
      use_mixed_precision: true
      model:
        _target_: segmentation_models_pytorch.UnetPlusPlus
        encoder_name: tu-maxvit_small_tf_224
        encoder_weights: imagenet
        in_channels: 3
        classes: 1
        activation:
        decoder_attention_type: scse
      criterion:
        _target_: segmentation_models_pytorch.losses.SoftBCEWithLogitsLoss
        smooth_factor: 0.1
      optimizer:
        _target_: functools.partial
        _args_:
          - _target_: hydra.utils.get_class
            path: torch.optim.AdamW
        lr: 5e-4
```

Note that the type of train_sys isnot defined, that is because the default pattern is already specified in conf/model/pipeline/default.yaml, which should not be modified. Contents:
```yaml
_target_: epochlib.model.ModelPipeline
_convert_: partial

x_sys:
  _target_: src.modules.transformation.verbose_transformation_pipeline.VerboseTransformationPipeline
  title: "Preprocessing pipeline"

y_sys:
  _target_: src.modules.transformation.verbose_transformation_pipeline.VerboseTransformationPipeline
  title: "Label processing pipeline"

train_sys:
  _target_: src.modules.training.verbose_training_pipeline.VerboseTrainingPipeline

pred_sys:
  _target_: src.modules.transformation.verbose_transformation_pipeline.VerboseTransformationPipeline
  title: "Postprocessing pipeline"

label_sys:
  _target_: src.modules.transformation.verbose_transformation_pipeline.VerboseTransformationPipeline
```

# Pipeline structure:
x_sys: (fit) transforms only x
y_sys: (fit) transforms only y
train_sys: (fit) transforms anything that needs access to both x and y. Usually an instance of TorchTrainer. Depending on settings, returns predictions and/or transformed labels.
pred_sys: (transform) transforms the predictions made by the model.
label_sys: (transform) transforms the labels after being returned by the model. (mainly relevant if the final scoring function is significantly different than the one used during training)

There are examples for the different parts of the pipeline, for example example_transformation_block.py and example_training_block.py.

# UV Notes
Update the project description, python version etc. accordingly. DO NOT ADD PACKAGES DIRECTLY TO THE FILE AND GUESS VERSION NUMBERS. Run `uv add` instead.


Only exception for manual dependency adding, is to use the following pattern in pyproject.toml to specify PyTorch GPU sources. 
```toml	
[tool.uv.sources]
torch = [
  { index = "pytorch-cu118", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-cu118", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true
```

ALLWAYS run with `uv run` and not `python`.
Use `uv add` to add dependencies. Use `uv sync` to update the dependencies.
First time might require `uv init`. Be aware of created template file that are not needed. like main.py


# Running the code
There are three main files: train.py, cv.py and submit.py.
Train uses a single train-val split.
CV uses K-fold cross-validation. The Kaggle public score is treated as the test score.
Submit should be used during inference on Kaggle for code competitions.

Each file has a corresponding .yaml file in the conf/ directory. This specifies which model pipeline to use, which scorer to use, and splitter configuration. To allow Train and CV to use the same splitter, the train-test split is defined in train.yaml by setting a number of splits, after which only the first fold is used. E.g. 5 splits gives a 80-20 train-test split. It is somewhat limited in possible ratios. The default should be to use 80-20.

Use `uv run train` to train the model etc. Hydra allows command line arguments to switch models, e.g. `uv run train model=mlp`. LET THE USER RUN THE COMMAND THEMSELVES for long training runs, so they can do it in their own large terminal window instead of the integrated agent terminal.

# Kaggle submissions
Code and models are uploaded to Kaggle through the API with the submission/manage_datasets.py utility scripts. The user needs to fill this in their own terminal, as they are prompted for the API key and other information. This will then be used for submission.

The datasets names and paths are defined in the submission/config/source.json and submission/config/dependencies.json files. On first use, they need to be created, can be with a dummy file.

# Logging
When inheriting from VerboseTransformationBlock or VerboseTrainingBlock, the logger will be available as:
- `self.log_to_terminal(message: str)`
- `self.log_to_debug(message: str)`
- `self.log_to_warning(message: str)`
- `self.log_to_external(message: dict[str, Any], **kwargs: Any)` -> for wandb logging or custom plots
- `self.external_define_metric(metric: str, metric_type: str)` -> for defining metrics in wandb

# Caching
By default, the result of x_sys and y_sys is cached. This uses epochlib functionality. The data format is specified in train.py and cv.py.

# Data folder
The data folder contains /raw and /processed. The rule is that /raw is exactly in the format as you would get it when downloading the data from the competition page (and unzipping). /processed will automatically be used for caching.

# Output folder
Each run gets an output folder, assigned by Hydra. It is possible to save plots or other outputs to this folder in e.g. as post-processing steps. The main approach however should be to use wandb for logging. The folder can be accessed using `output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)`