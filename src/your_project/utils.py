import argparse
import logging
import os
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


def create_experiment_snapshot(repo_path: str = ".", snapshot_branch: str = "exp_snapshots") -> str:
    """
    Creates a snapshot of the current Git repository state without disturbing the working directory.

    This function performs the following steps:
    1.  Initializes a Repo object from the given path.
    2.  Records the current branch or commit (if in a detached HEAD state).
    3.  Stashes all local changes, including untracked files.
    4.  Switches to the specified snapshot branch (creates it if it doesn't exist).
    5.  Creates a new commit on the snapshot branch.
    6.  Switches back to the original branch/commit.
    7.  Applies the stashed changes back to the working directory.

    A `try...finally` block ensures that the repository state is always restored,
    even if an error occurs during the snapshot process.

    Args:
        commit_message (str): The commit message for the snapshot.
        repo_path (str, optional): The path to the Git repository. Defaults to the current directory.
        snapshot_branch (str, optional): The name of the branch to store snapshots. Defaults to 'exp_snapshots'.
    """
    try:
        # 1. Initialize the Repo object. This also checks if it's a valid Git repo.
        repo = git.Repo(repo_path, search_parent_directories=True)
        logging.info(f"Successfully connected to repository at: {repo.working_tree_dir}")
    except git.exc.InvalidGitRepositoryError:
        logging.error(f"The directory '{os.path.abspath(repo_path)}' is not a valid Git repository.")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred while initializing the repository: {e}")
        return

    original_ref = None
    stashed_something = False

    try:
        # --- Preparation Phase ---
        logging.info("--- Starting Git Snapshot Process ---")

        # 2. Record the current branch or commit hash (for detached HEAD)
        if repo.head.is_detached:
            original_ref = repo.head.commit.hexsha
            logging.warning(
                f"Currently in a 'detached HEAD' state at commit {original_ref[:7]}. Will return to this commit."
            )
        else:
            original_ref = repo.active_branch.name
            logging.info(f"Current branch is '{original_ref}'.")

        # 3. Stash local changes if the working directory is dirty
        if repo.is_dirty(untracked_files=True):
            logging.info("Local changes detected. Stashing them...")
            repo.git.stash("push", "-u", "-m", "snapshot_auto_stash")
            stashed_something = True
            logging.info("Local changes have been stashed.")
        else:
            logging.info("Workspace is clean. No need to stash.")

        # 4. Switch to the snapshot branch (create if it doesn't exist)
        if snapshot_branch in repo.heads:
            logging.info(f"Switching to existing branch '{snapshot_branch}'...")
            repo.heads[snapshot_branch].checkout()
        else:
            logging.info(f"Branch '{snapshot_branch}' not found. Creating and switching...")
            repo.create_head(snapshot_branch).checkout()
        logging.info(f"Successfully switched to branch '{snapshot_branch}'.")

        # 5. Create the snapshot commit
        logging.info(f"Creating snapshot commit on branch '{snapshot_branch}'...")
        repo.git.commit(
            "--allow-empty", "-m", f'"Snapshot commit for experiment at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"'
        )
        snapshot_hash = repo.head.commit.hexsha
        logging.info(f"Snapshot created successfully! Commit: {snapshot_hash[:8]}")

    except git.exc.GitCommandError as e:
        logging.error(f"A Git command failed during the snapshot process.")
        logging.error(f"Command: {' '.join(e.command)}")
        logging.error(f"Stderr: {e.stderr.strip()}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

    finally:
        # --- Cleanup Phase ---
        # This block will run regardless of whether an error occurred.
        logging.info("--- Cleaning up and restoring original state ---")

        if not original_ref:
            logging.warning("Could not determine the original branch/commit. Cleanup might be incomplete.")
            return

        # 6. Switch back to the original branch or commit
        logging.info(f"Switching back to '{original_ref}'...")
        repo.git.switch("--force", original_ref)
        logging.info("Successfully switched back.")

        # 7. Restore the stashed changes if something was stashed
        if stashed_something:
            logging.info("Restoring stashed changes...")
            try:
                repo.git.stash("pop")
                logging.info("Stashed changes have been successfully restored.")
            except git.exc.GitCommandError as e:
                logging.error("Failed to pop stash. This may be due to a conflict.")
                logging.error(f"Stderr: {e.stderr.strip()}")
                logging.warning("Your changes are still saved in the stash. Please run 'git stash list' to see them.")
                logging.warning("You may need to manually run 'git stash apply' and resolve conflicts.")

        logging.info("🎉 Snapshot process complete. Your workspace is back to its original state.")


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
