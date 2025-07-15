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
