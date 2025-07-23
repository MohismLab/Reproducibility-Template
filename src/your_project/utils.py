import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Union

from pydantic_settings import BaseSettings

import wandb

from .config import ExperimentSettings

logger = logging.getLogger(__name__)
training_ascii_art = """
//  ____    ____   ______    __    __  .______              ___      .______      .___________.  //
//  \   \  /   /  /  __  \  |  |  |  | |   _  \            /   \     |   _  \     |           |  //
//   \   \/   /  |  |  |  | |  |  |  | |  |_)  |          /  ^  \    |  |_)  |    `---|  |----`  //
//    \_    _/   |  |  |  | |  |  |  | |      /          /  /_\  \   |      /         |  |       //
//      |  |     |  `--'  | |  `--'  | |  |\  \----.    /  _____  \  |  |\  \----.    |  |       //
//      |__|      \______/   \______/  | _| `._____|   /__/     \__\ | _| `._____|    |__|       //
//                                                                                               //
"""


def parse_args_and_update_config() -> ExperimentSettings:
    """
    Parse command line arguments and update configuration

    Command line format example:
    --model_config_file configs/models/cnn_mnist.yaml --batch_size 64 --learning_rate 0.01 --project my_project

    Returns:
        Updated ExperimentSettings object
    """

    # Create configuration object
    config = ExperimentSettings()

    # Create argument parser
    parser = argparse.ArgumentParser(description="Experiment hyper-parameters")

    # Add all fields from ExperimentSettings
    for field_name, field_info in ExperimentSettings.model_fields.items():
        # Skip git commit hash
        if field_name == "git_commit_hash":
            continue
        # Get field type from annotations
        field_type = ExperimentSettings.__annotations__.get(field_name)
        description = field_info.description

        # Handle Path type
        if field_type == Path:
            parser.add_argument(f"--{field_name}", type=str, help=description)
        # Handle nested settings
        elif hasattr(getattr(config, field_name), "model_fields"):
            nested_settings = getattr(config, field_name)
            nested_class = nested_settings.__class__

            for nested_name, nested_info in nested_class.model_fields.items():
                nested_type = nested_class.__annotations__.get(nested_name)
                nested_desc = nested_info.description

                if nested_type == Path:
                    parser.add_argument(f"--{nested_name}", type=str, help=nested_desc)
                else:
                    parser.add_argument(f"--{nested_name}", type=nested_type, help=nested_desc)
        else:
            parser.add_argument(f"--{field_name}", type=field_type, help=description)

    # Parse command line arguments
    args = parser.parse_args()
    args_dict = vars(args)

    # Update all fields in config
    for field_name in ExperimentSettings.model_fields:
        if field_name in args_dict and args_dict[field_name] is not None:
            value = args_dict[field_name]

            # Convert str to Path if needed
            if isinstance(getattr(config, field_name), Path) and isinstance(value, str):
                value = Path(value)

            setattr(config, field_name, value)

    # Update nested settings
    for field_name in ExperimentSettings.model_fields:
        nested_settings = getattr(config, field_name)

        # If it's a settings object with model_fields
        if hasattr(nested_settings, "model_fields"):
            for nested_name in nested_settings.__class__.model_fields:
                if nested_name in args_dict and args_dict[nested_name] is not None:
                    setattr(nested_settings, nested_name, args_dict[nested_name])

    return config


# Custom WandbLogHandler for syncing logs to wandb
class WandbLogHandler(logging.Handler):
    def emit(self, record):
        try:
            # Format the log message
            msg = self.format(record)
            # Log to wandb
            wandb.log({"log": msg, "level": record.levelname})
        except Exception:
            self.handleError(record)


def convert_config_for_wandb(cfg: Union[BaseSettings, Dict]) -> Dict[str, Any]:
    if hasattr(cfg, "model_dump"):
        # Get dict representation
        cfg_dict = cfg.model_dump()
        # Process each value
        for key, value in cfg_dict.items():
            # Convert nested BaseSettings
            if hasattr(value, "model_dump"):
                cfg_dict[key] = convert_config_for_wandb(value)
            # Convert Path to string
            elif isinstance(value, Path):
                cfg_dict[key] = str(value)
        return cfg_dict
    else:
        # If it's already a primitive type
        return cfg
