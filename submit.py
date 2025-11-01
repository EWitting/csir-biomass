"""Submit.py is the main script for running inference on the test set and creating a submission."""
import os
import warnings
from pathlib import Path

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from src.config.submit_config import SubmitConfig
from src.setup.setup_data import setup_inference_data
from src.setup.setup_pipeline import setup_pipeline
from src.setup.setup_runtime_args import setup_pred_args
from src.utils.logger import logger, print_section_separator

warnings.filterwarnings("ignore", category=UserWarning)

# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_submit", node=SubmitConfig)


@hydra.main(version_base=None, config_path="conf", config_name="submit")
# TODO(Epoch): Use SubmitConfig instead of DictConfig
def run_submit(cfg: DictConfig) -> None:
    """Run the main script for submitting the predictions."""
    print_section_separator("Q? - 'competition' - Submit")

    # Set up logging
    import coloredlogs

    coloredlogs.install()

    # Preload the pipeline
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg, is_train=False)

    # Load the test data
    X = setup_inference_data(Path(cfg.data_path))

    # Predict on the test data
    logger.info("Making predictions...")
    pred_args = setup_pred_args(pipeline=model_pipeline)
    predictions = model_pipeline.predict(X, **pred_args)

    # Make submission
    if predictions is not None:
        # Load test.csv to get image IDs
        test_df = pd.read_csv(Path(cfg.data_path))
        
        # Get unique image IDs from test set (predictions are ordered by unique images)
        unique_images = test_df[['image_path']].drop_duplicates().reset_index(drop=True)
        
        # Extract image IDs from paths (e.g., "test/ID1001187975.jpg" -> "ID1001187975")
        image_ids = unique_images['image_path'].apply(lambda x: Path(x).stem).tolist()
        
        # Target names in the order they were predicted (must match setup_train_y_data)
        target_names = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'GDM_g', 'Dry_Total_g']
        
        # Create submission dataframe
        submission_data = []
        for i, image_id in enumerate(image_ids):
            for j, target_name in enumerate(target_names):
                sample_id = f"{image_id}__{target_name}"
                target_value = predictions[i, j]
                submission_data.append({'sample_id': sample_id, 'target': target_value})
        
        submission = pd.DataFrame(submission_data)
        
        logger.info(f"Created submission with {len(submission)} rows for {len(image_ids)} images")
        
        # Save submissions to path (Kaggle format)
        result_path = Path(cfg.result_path)
        os.makedirs(result_path, exist_ok=True)
        submission_path = result_path / "submission.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
    else:
        raise ValueError("Predictions are None")


if __name__ == "__main__":
    run_submit()
