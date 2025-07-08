from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

MODEL_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs" / "models"


class WandbSettings(BaseSettings):
    """Configuration for logging on Wandb"""

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
    # path to model configuration .yaml file, default is an xs model
    model_config_file: Path = Field(
        MODEL_CONFIG_DIR / "xs.yaml", description="Configuration file of model architecture"
    )
    git_commit_hash: str = Field("", description="Git commit hash of the snapshot of running codes")
    wandb: WandbSettings = WandbSettings()
    train: TrainSettings = TrainSettings()
    eval: EvalSettings = EvalSettings()
