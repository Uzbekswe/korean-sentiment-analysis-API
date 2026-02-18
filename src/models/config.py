"""
Configuration loader for model settings.

Reads from configs/model_config.yaml so no values are hardcoded in source code.
"""

import os
from pathlib import Path

import yaml


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml"


def get_model_config() -> dict:
    """Load model configuration from YAML file."""
    config_path = os.getenv("MODEL_CONFIG_PATH", str(_CONFIG_PATH))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
