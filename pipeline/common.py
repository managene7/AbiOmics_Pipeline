from __future__ import annotations

import ast
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "config/config.yaml") -> dict:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)

    # YAML sometimes parses list literals as strings — coerce if needed
    for key in ("stress_types",):
        val = cfg["data"][key]
        if isinstance(val, str):
            cfg["data"][key] = ast.literal_eval(val)

    # Build label dictionary from stress_types
    cfg["label_dic"] = {0: "Control"}
    for i, stress in enumerate(cfg["data"]["stress_types"]):
        cfg["label_dic"][i + 1] = stress

    cfg["n_classes"] = len(cfg["label_dic"])   # 1 (control) + N stresses

    logger.info("Config loaded from %s", path)
    logger.info("  Stress types: %s", cfg["data"]["stress_types"])
    logger.info("  Classes: %s", cfg["label_dic"])
    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    logger.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    fmt     = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dirs(cfg: dict) -> None:
    dirs = [
        cfg["output"]["export_path"],
        cfg["training"]["model_save_path"],
        "Loss_graph",   # train.py saves loss plots here unconditionally
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Output directories ready: %s", dirs)
