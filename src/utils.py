from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(diff * diff)))


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rl2(a: np.ndarray, b: np.ndarray) -> float:
    # Relative L2 = ||a-b||2 / ||a||2
    num = np.sqrt(np.sum((a - b) ** 2))
    den = np.sqrt(np.sum(a ** 2))
    return float(num / den) if den != 0 else float("nan")
