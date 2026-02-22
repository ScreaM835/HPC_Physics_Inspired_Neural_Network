from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
