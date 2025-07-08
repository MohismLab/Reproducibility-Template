from typing import Any, Dict

import torch.nn as nn
import yaml


class CNNModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(CNNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.config = config

        # parse the layer configs
        for layer_config in config["layers"]:
            layer_type = layer_config["type"]

            if layer_type == "conv":
                self.layers.append(
                    nn.Conv2d(
                        in_channels=layer_config["in_channels"],
                        out_channels=layer_config["out_channels"],
                        kernel_size=layer_config["kernel_size"],
                        stride=layer_config["stride"],
                        padding=layer_config["padding"],
                    )
                )
            elif layer_type == "relu":
                self.layers.append(nn.ReLU())
            elif layer_type == "maxpool":
                self.layers.append(nn.MaxPool2d(kernel_size=layer_config["kernel_size"], stride=layer_config["stride"]))
            elif layer_type == "flatten":
                self.layers.append(nn.Flatten())
            elif layer_type == "linear":
                self.layers.append(
                    nn.Linear(in_features=layer_config["in_features"], out_features=layer_config["out_features"])
                )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def create_model_from_config(config_path: str) -> CNNModel:
    """Create a model from a config .yaml file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return CNNModel(config)
