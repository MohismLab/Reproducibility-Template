import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import git
import wandb
from pydantic_settings import BaseSettings

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


def create_experiment_snapshot() -> str:
    """
    Create a snapshot of the current repository state in a separate branch.
    Bypasses pre-commit hooks to ensure the snapshot can be created.

    Returns:
        str: The commit hash of the snapshot
    """
    try:
        # Get the repository object (assuming it already exists)
        repo_path = Path(__file__).parent.parent.parent  # Assuming code is in src/your_project/
        repo = git.Repo(repo_path)

        # Check current working directory status
        if not repo.is_dirty() and not repo.untracked_files:
            logger.info("No changes detected in the repository. Using current commit.")
            return repo.head.commit.hexsha

        branch_name = "exp_snapshots"
        current_branch = repo.active_branch.name

        logger.info(f"Current branch: {current_branch}")
        logger.info(f"Repository has changes: dirty={repo.is_dirty()}, untracked={len(repo.untracked_files)}")

        try:
            # First, stash current changes if any
            stash_created = False
            if repo.is_dirty() or repo.untracked_files:
                logger.info("Stashing current changes before branch switch...")
                # Include untracked files in stash
                repo.git.stash("push", "-u", "-m", "Temporary stash for experiment snapshot")
                stash_created = True
                logger.info("Changes stashed successfully")

            # Now switch to experiment snapshot branch (create if doesn't exist)
            if branch_name not in repo.heads:
                repo.git.checkout("-b", branch_name)
                logger.info(f"Created new branch '{branch_name}' for experiment snapshot.")
            else:
                repo.git.checkout(branch_name)
                logger.info(f"Switched to existing branch '{branch_name}' for experiment snapshot.")

            # Apply the stashed changes to the snapshot branch
            if stash_created:
                try:
                    repo.git.stash("pop")
                    logger.info("Applied stashed changes to snapshot branch")
                except Exception as stash_e:
                    logger.warning(f"Failed to apply stash: {str(stash_e)}")
                    # If stash pop fails, try to get the changes from the original branch
                    try:
                        repo.git.checkout(current_branch, "--", ".")
                        logger.info("Copied changes from original branch")
                    except Exception as copy_e:
                        logger.error(f"Failed to copy changes: {str(copy_e)}")

            # Add all files (including untracked ones)
            repo.git.add("--all")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            commit_message = f"Experiment snapshot at {timestamp}"
            commit_hash = ""

            try:
                # Directly use subprocess to execute git commit command and skip hooks
                result = subprocess.run(
                    ["git", "commit", "-m", commit_message, "--no-verify"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    commit_hash = repo.head.commit.hexsha
                    logger.info(f"Successfully committed changes with hash: {commit_hash[:8]}")
                else:
                    # Check if it's because there are no changes to commit
                    if "nothing to commit" in result.stdout.lower():
                        logger.info("No changes to commit, using current HEAD")
                        commit_hash = repo.head.commit.hexsha
                    else:
                        logger.warning(f"Git commit failed: {result.stderr}")
                        # Try fallback method
                        try:
                            commit = repo.index.commit(commit_message)
                            commit_hash = commit.hexsha
                            logger.info(f"Fallback commit successful with hash: {commit_hash[:8]}")
                        except Exception as fallback_e:
                            logger.error(f"Fallback commit failed: {str(fallback_e)}")
                            return ""

            except Exception as e:
                logger.warning(f"Error during commit: {str(e)}")
                try:
                    commit = repo.index.commit(commit_message)
                    commit_hash = commit.hexsha
                    logger.info(f"Alternative commit method successful with hash: {commit_hash[:8]}")
                except Exception as alt_e:
                    logger.error(f"All commit methods failed: {str(alt_e)}")
                    return ""

            if commit_hash:
                logger.info(f"Created experiment snapshot in branch '{branch_name}' with commit {commit_hash[:8]}")
                return commit_hash
            else:
                logger.error("No commit hash was generated")
                return ""

        finally:
            # Switch back to original branch
            try:
                repo.git.checkout(current_branch)
                logger.info(f"Switched back to original branch '{current_branch}'")

                # If we had stashed changes and they're still in stash, restore them
                try:
                    stash_list = repo.git.stash("list")
                    if stash_list and "Temporary stash for experiment snapshot" in stash_list:
                        repo.git.stash("pop")
                        logger.info("Restored original changes from stash")
                except Exception as restore_e:
                    logger.warning(f"Could not restore stashed changes: {str(restore_e)}")

            except Exception as e:
                logger.error(f"Failed to switch back to original branch: {str(e)}")

    except Exception as e:
        logger.error(f"Failed to create experiment snapshot: {str(e)}")
        return ""


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
