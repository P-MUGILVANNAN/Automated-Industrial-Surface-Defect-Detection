"""
System and experimentation utilities for Industrial Defect Detection.
"""

import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import numpy as np
import torch


def setup_logger(name: str = "industrial_defect_detection", level: str = "INFO") -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def load_config(config_path: Union[str, Path] = "config/config.yaml") -> Dict[str, Any]:
    """Loads configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "auto") -> torch.device:
    """
    Determines and returns torch.device based on availability and preference.
    Supports 'auto', 'cuda', 'cpu', or 'mps'.
    """
    pref = preferred.lower() if preferred else "auto"
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs(*dirs: Union[str, Path]) -> None:
    """Creates directories if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
