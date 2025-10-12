from __future__ import annotations

import yaml
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file and return its content as a Python dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed YAML content as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the file content cannot be parsed as valid YAML.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"❌ Config file not found: {path.resolve()}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"❌ Failed to parse YAML config: {path}\n{e}")

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML to parse into a dict, got {type(data)}")

    logger.debug(f"Loaded config from {path.resolve()}")
    return data


def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    """
    Recursively convert a nested dictionary into a SimpleNamespace object.

    This enables dot-notation access to configuration fields, e.g.:
    >>> cfg = dict_to_namespace({'model': {'name': 'ResNet'}})
    >>> print(cfg.model.name)
    'ResNet'

    Parameters
    ----------
    d : Dict[str, Any]
        Dictionary to convert.

    Returns
    -------
    SimpleNamespace
        Namespace object with dot-notation access.
    """
    namespace = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(namespace, k, dict_to_namespace(v))
        else:
            setattr(namespace, k, v)
    return namespace


def load_config(config_path: Union[str, Path]) -> SimpleNamespace:
    """
    Load a YAML configuration file and return it as a SimpleNamespace.

    This function combines `load_yaml` and `dict_to_namespace` to provide
    a structured configuration object that's easy to use in large ML pipelines.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    SimpleNamespace
        Structured configuration object.

    Examples
    --------
    >>> cfg = load_config("configs/decision_transformer.yaml")
    >>> print(cfg.model.name)
    DecisionTransformer
    """
    cfg_dict = load_yaml(config_path)
    cfg_namespace = dict_to_namespace(cfg_dict)
    return cfg_namespace


if __name__ == "__main__":
    cfg = load_config("src/configs/decision_transformer.yaml")

    print(cfg.model.name)
    print(cfg.optimizer.lr)