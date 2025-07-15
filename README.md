# A Demo Project for Reproducible Deep Learning Experiments

This project serves as a demonstration of a practice for ensuring reproducibility in deep learning research, using the classic `MNIST` dataset as a case study.
The primary goal is to showcase a robust workflow that combines configuration management, code versioning, and standardised development practices.

## 1. Project Purpose

The core objective is to provide a clear, practical template that addresses common reproducibility challenges in machine learning projects. This includes:
- **Transparent Configurations**: Managing all hyperparameters and settings in a clear and overridable way.
- **Precise Code Versioning**: Tracking not just major code milestones with Git, but also the exact code state for every single experiment run.
- **Standardised Workflow**: Establishing clear procedures for installation, experimentation, and contributing code changes.

## 2. Installation and Setup

To get started and run experiments locally, follow these steps.

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/Reproducibility-Template.git
cd Reproducibility-Template
```

### Step 2: Install Dependencies with `uv`
This project uses `uv` for fast and reliable dependency management. Install the project and its development dependencies with the following command:

```bash
# This installs the project in editable mode (-e) along with dev tools
uv sync --extra dev
source .venv/bin/activate
```

### Step 3: Install Pre-commit Hooks
To ensure code quality and consistency, this project uses pre-commit hooks. Install them with:
```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
```

### Step 4: Set Up Environment Variables
The project uses Weights & Biases (WandB) for experiment tracking. You need to provide your credentials. Create a file named `.env` in the root directory by copying the example:

```bash
# Create your own .env file
cp .env.example .env
```

Now, edit `.env` and add your WandB project and entity (your username or team name). The file is already listed in `.gitignore` to prevent you from accidentally committing your credentials.

```bash
# .env
WANDB_PROJECT="Your-WandB-Project"
WANDB_ENTITY="your-wandb-username"
```

## 3. Configuration with Pydantic

This project uses Pydantic's `BaseSettings` to manage all configurations. This approach offers several advantages:

- **Clarity and Type Safety**: All settings are defined in Python classes (see [`src/your_project/config.py`](src/your_project/config.py)), providing a single, type-annotated source of truth. This makes the configuration structure easy to understand and prevents common errors.
- **Hierarchical Structure**: Settings are neatly organized into logical groups like `TrainSettings`, `EvalSettings`, and `WandbSettings`.
- **Command-Line Overrides**: The default settings can be easily overridden from the command line when you run an experiment. The script [`src/your_project/utils.py`](src/your_project/utils.py) automatically parses arguments and updates the configuration object.

For example, to run an experiment with a different model architecture and learning rate, you can simply do:
```bash
train --model_arch_file configs/models/s.yaml --learning_rate 0.01
```

## 4. Tracking Ad-Hoc Code Changes with WandB `log_code()`

While `git` is essential for versioning major features and bug fixes, it is not always suitable for tracking the small, iterative changes made during active experimentation. Committing every minor tweak (e.g., changing a layer's activation function just to see what happens) would pollute the Git history.

To solve this, we use `wandb.log_code()` to save a snapshot of the entire codebase for *every single run*. This ensures that you always have a perfect record of the exact code that produced a given result, capturing any ad-hoc modifications.

This is implemented in the `setup_wandb_logging` function in [`src/your_project/train.py`](src/your_project/train.py).

### How to View the Code Snapshot in WandB
1. Go to your project page on the WandB website.
2. Click on a specific run from your run table.
3. In the run's dashboard, navigate to the **"Artifacts"** tab in the left-hand panel.
4. Under the "Files" directory, you will find a **`code`** artifact.
5. Click on it to browse the complete snapshot of the source code (`.py`, `.yaml` files) that was saved for that run.

## 5. Documenting Reproduction Steps

A key part of reproducibility is clear documentation. This project advocates for creating detailed guides on how to reproduce key results. These guides should be placed in the `/docs` directory.

A good reproduction guide should include:
- The exact `git commit` hash of the code version used.
- The precise command-line arguments required to launch the training script.
- A direct link to the resulting WandB run page.
- The expected final metrics (e.g., "Test Accuracy: 99.25%").

## 6. Contribution Workflow: Why Use Pull Requests?

Directly pushing changes to the `main` branch is strongly discouraged, especially in collaborative projects or for maintaining a stable codebase. Instead, all changes should be introduced through **Pull Requests (PRs)**.

**The standard workflow is:**
1. **Create a New Branch**: Create a descriptive branch for your new feature or fix (e.g., `feature/add-dropout-layer` or `fix/data-loader-bug`).
2. **Commit Your Changes**: Make your changes on this new branch.
3. **Open a Pull Request**: Once your changes are ready, open a pull request against the `main` branch.
4. **Review and Merge**: The PR should be reviewed by at least one other team member. Once approved, it can be merged into `main`.

Why is this important?
- **Code Review**: PRs allow for peer review, which improves code quality and knowledge sharing.
- **History Clarity**: Each PR represents a logical unit of work, making it easier to understand the project's evolution.
