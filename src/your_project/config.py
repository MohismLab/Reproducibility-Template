import datetime
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs" / "models"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"


class WandbSettings(BaseSettings):
    """Configuration for logging on Wandb"""

    model_config = {"env_prefix": "WANDB_"}  # to ensure auto-update from .env file

    # Basic settings
    project: str = Field(default="your-project", description="WandB project name")
    name: Optional[str] = Field(default=None, description="Experiment name, auto-generated if not set")
    entity: Optional[str] = Field(default=None, description="WandB team or username")


class TrainSettings(BaseSettings):
    """Configuration for training"""

    batch_size: int = Field(default=32, description="Batch size for training")
    epochs: int = Field(default=10, description="Number of epochs for training")
    learning_rate: float = Field(default=0.001, description="Learning rate for the optimizer")


class EvalSettings(BaseSettings):
    """Configuration for evaluation"""

    eval_batch_size: int = Field(default=64, description="Batch size for evaluation")


# An experiment should consist of all of the above modules
class ExperimentSettings(BaseSettings):
    """Configuration for the whole experiment"""

    # to ensure auto-update from .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # path to model configuration .yaml file, default is an xs model
    model_config_file: Path = Field(
        MODEL_CONFIG_DIR / "xs.yaml", description="Configuration file of model architecture"
    )
    git_commit_hash: str = Field("", description="Git commit hash of the snapshot of running codes")

    data_dir: Path = Field(DATA_DIR / "MNIST", description="Path to the directory of training and evaluation data")
    output_dir: Path = Field(OUTPUT_DIR, description="Output directory for experiments")
    log_file: Path = Field(OUTPUT_DIR / "train.log", description="Log file name for the experiment")

    wandb: WandbSettings = WandbSettings()
    train: TrainSettings = TrainSettings()
    eval: EvalSettings = EvalSettings()

    def model_post_init(self, _: Optional[dict]) -> None:
        # Generate a name for the experiment if not provided
        if self.wandb.name is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.wandb.name = f"{self.model_config_file.stem}_{timestamp}"

        # Update output_dir to be a subdirectory of the base output_dir
        self.output_dir = self.output_dir / self.wandb.name
        self.log_file = self.output_dir / "train.log"

        # Create the experiment-specific output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
