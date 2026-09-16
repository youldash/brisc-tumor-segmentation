"""Shared helpers: configuration, reproducibility, device selection, I/O."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS: dict[str, Any] = {
    "run_name": "experiment",
    "seed": 42,
    "device": "auto",  # "auto" | "mps" | "cpu" | "cuda"
    "data": {
        "root": "data/raw",          # folder the Kaggle archive was unzipped into
        "splits_dir": "data/splits",  # CSVs written by src/prepare_data.py
        "img_size": 256,
        "num_workers": 2,
        "include_healthy": False,     # add healthy slices (empty masks) to train/val
        "healthy_val_fraction": 0.15,
        "limit_train": None,          # optional subsampling (debugging)
        "limit_val": None,
    },
    "model": {
        "name": "unet_scratch",       # unet_scratch | smp_unet | smp_unetplusplus | smp_fpn
        "base_channels": 32,          # only for unet_scratch
        "encoder_name": "resnet34",   # only for smp_* models
        "encoder_weights": "imagenet",
    },
    "train": {
        "epochs": 40,
        "batch_size": 8,
        "lr": 3.0e-4,
        "encoder_lr_mult": 1.0,       # <1 fine-tunes a pretrained encoder more gently
        "weight_decay": 1.0e-4,
        "min_lr": 1.0e-6,
        "warmup_epochs": 1,
        "patience": 10,               # early stopping on validation Dice
        "grad_clip": 1.0,
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "threshold": 0.5,
        "cooldown_seconds": 0,        # optional pause between epochs (fanless laptop)
    },
    "output": {"runs_dir": "runs"},
}


def deep_update(base: dict, updates: dict) -> dict:
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deep_update(copy.deepcopy(DEFAULTS), user_cfg)
    cfg["config_path"] = str(path)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Relative paths are interpreted relative to the project root."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preference: str = "auto"):
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_yaml(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
