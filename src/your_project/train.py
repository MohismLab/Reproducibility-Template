import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import ExperimentSettings
from .models import create_model_from_config
from .utils import (
    WandbLogHandler,
    convert_config_for_wandb,
    create_experiment_snapshot,
    parse_args_and_update_config,
    training_ascii_art,
)

# set the logger of the whole experiment
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def setup_wandb_logging(config: ExperimentSettings) -> None:
    """
    Initialize wandb and set up logging to sync with wandb

    Args:
        config: Experiment configuration
    """
    # Initialize wandb
    wandb_config = convert_config_for_wandb(config)

    # Initialize wandb with full config
    wandb.init(project=config.wandb.project, name=config.wandb.name, entity=config.wandb.entity, config=wandb_config)

    # Create and add wandb log handler
    wandb_handler = WandbLogHandler()
    wandb_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    wandb_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(wandb_handler)
    logger.info("WandB logging initialised")


def load_mnist_data(data_dir: Path, batch_size: int, eval_batch_size: int) -> tuple:
    """
    Load and prepare MNIST dataset

    Args:
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation

    Returns:
        tuple: (train_loader, test_loader)
    """
    logger.info("Loading MNIST dataset...")

    # Define transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]  # MNIST mean and std
    )

    # Load datasets
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=eval_batch_size, shuffle=False)

    logger.info(f"Dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")

    return train_loader, test_loader


def main() -> None:
    logger.info(training_ascii_art)
    config = parse_args_and_update_config()

    logger.info("Creating snapshot of the running codes to `exp_snapshot`...")
    commit_hash = create_experiment_snapshot()
    config.git_commit_hash = commit_hash
    logger.info(f"Running codes are snapshoted to commit with hash `{commit_hash}`")

    logger.info("Setting up wandb handler for logger....")
    setup_wandb_logging(config)

    logger.info(f"Building model from {config.model_config_file}...")
    model = create_model_from_config(config)
    logger.info(f"Built the following model:\n{model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)

    # Load data
    train_loader, test_loader = load_mnist_data(
        data_dir=config.data_dir, batch_size=config.train.batch_size, eval_batch_size=config.eval.eval_batch_size
    )

    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)

    # Train model
    logger.info("Starting training...")
    best_accuracy = 0

    # Calculate total steps for progress tracking
    total_steps = config.train.epochs * len(train_loader)
    global_step = 0

    # Create directory for saving models
    save_dir = Path(config.output_dir) / "ckpts"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(1, config.train.epochs + 1):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch_idx, (data, target) in enumerate(train_loader):
            global_step += 1
            progress = global_step / total_steps * 100

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Accumulate statistics
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

            # Log progress
            if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
                logger.info(
                    f"Epoch: {epoch}/{config.train.epochs} | "
                    f"Batch: {batch_idx+1}/{len(train_loader)} | "
                    f"Progress: {progress:.1f}% | "
                    f"Loss: {train_loss/(batch_idx+1):.4f} | "
                    f"Accuracy: {100.*train_correct/train_total:.2f}%"
                )

            # Log to wandb
            wandb.log({"train_batch_loss": loss.item(), "train_batch_acc": 100.0 * train_correct / train_total})

        train_avg_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total

        logger.info(f"Epoch {epoch} Training | Loss: {train_avg_loss:.4f} | Accuracy: {train_accuracy:.2f}%")

        # Evaluation phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()

        test_avg_loss = test_loss / len(test_loader)
        test_accuracy = 100.0 * test_correct / test_total

        logger.info(f"Epoch {epoch} Test | Loss: {test_avg_loss:.4f} | Accuracy: {test_accuracy:.2f}%")
        wandb.log(
            {
                "epoch": epoch,
                "train_epoch_loss": train_avg_loss,
                "train_epoch_acc": train_accuracy,
                "test_loss": test_avg_loss,
                "test_accuracy": test_accuracy,
            }
        )

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # Create filename with configuration details
            model_name = config.model_config_file.stem
            save_path = save_dir / f"{model_name}_epoch{epoch}_acc{test_accuracy:.2f}.pt"

            # Save model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "accuracy": test_accuracy,
                    "git_commit_hash": config.git_commit_hash,
                },
                save_path,
            )

            logger.info(f"New best model saved to {save_path} with accuracy: {test_accuracy:.2f}%")
            wandb.save(str(save_path))

    logger.info(f"Training finished! Best test accuracy: {best_accuracy:.2f}%")

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
