from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Configuration template for training and evaluation"""

    batch_size: int = Field(default=32, description="Batch size for training and evaluation")


class TrainSettings(Settings):
    """Configuration for training"""

    epochs: int = Field(default=10, description="Number of epochs for training")
    learning_rate: float = Field(default=0.001, description="Learning rate for the optimizer")


class EvalSettings(Settings):
    """Configuration for evaluation"""

    temperature: float = Field(default=1.0, description="Temperature for sampling during evaluation")
