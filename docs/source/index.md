# Welcome to Your Project

```{note}
This project is under active development.
```

## Installation

To develop this project, you will need `uv`.

```bash
git clone git@github.com:MohismLab/Reproducibility-Template.git
cd Reproducibility-Template
# Install development dependencies
uv sync --extra "dev"
# Activate the virtual environment
source .venv/bin/activate
# Install pre-commit hooks
pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Experiment 1

To reproduce the results of Experiment 1, run the following command:

```bash
train
```

The entry is defined in the `pyproject.toml` file under the `[tool.poetry.scripts]` section.
This command will execute the training script with the default configuration settings.

### Results

- Accuracy: 99.03%
- Loss: 0.02968575582429129
- Wandb Run URL: [link](https://wandb.ai/mc45197-university-of-macau/Reproducibility%20Demo/runs/xqxsp7gt/overview)
- Git Commit: `37a218d562a8928beb5808f1fc824ec182537746` of branch `sguo/dev`
- Code snapshot: [link](https://wandb.ai/mc45197-university-of-macau/Reproducibility%20Demo/artifacts/code/source-Reproducibility_Demo-.venv_bin_train/v5/files)
