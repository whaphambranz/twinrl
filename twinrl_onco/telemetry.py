from __future__ import annotations

import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LOGGER_NAME = "twinrl_onco"


def get_logger(name: str | None = None) -> logging.Logger:
    base = logging.getLogger(_LOGGER_NAME)
    if not base.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        base.addHandler(handler)
        base.setLevel(logging.INFO)
        base.propagate = False
    if name is None:
        return base
    return base.getChild(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generator_for(seed: int, device: torch.device | None = None) -> torch.Generator:
    gen = torch.Generator(device=device or torch.device("cpu"))
    gen.manual_seed(seed)
    return gen


def save_checkpoint(payload: dict[str, Any], path: str | os.PathLike[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")
