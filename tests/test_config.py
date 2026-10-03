import os
import yaml
import pytest

def test_config_loading():
    assert os.path.exists("config.yaml"), "config.yaml does not exist"
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    assert "system" in config
    assert "dataset" in config
    assert "enhancement" in config
    assert "detection" in config
    assert "evaluation" in config
    assert "robustness" in config
    assert len(config["dataset"]["categories"]) == 12
