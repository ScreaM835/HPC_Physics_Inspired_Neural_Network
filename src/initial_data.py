from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


VelocityProfile = Literal["paper", "outgoing"]


def gaussian_phi(x: np.ndarray, A: float, x0: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return A * np.exp(-((x - x0) ** 2) / (sigma**2))


def gaussian_phi_t(
    x: np.ndarray, A: float, x0: float, sigma: float, profile: VelocityProfile = "paper"
) -> np.ndarray:
    """
    Initial time derivative Φ_t(x,0).

    - profile="paper": follows the printed Eq. (23) in the target paper
        Φ_t = 2 * ((x-x0)^2 / sigma^2) * Φ

      (Note: in standard 1+1 wave mechanics, an outgoing pulse would instead use Φ_t = -Φ_x.)

    - profile="outgoing": enforces outgoing right-moving pulse at t=0:
        (∂t + ∂x)Φ = 0  =>  Φ_t = -Φ_x = 2 (x-x0)/sigma^2 * Φ.
    """
    x = np.asarray(x, dtype=float)
    phi = gaussian_phi(x, A, x0, sigma)

    if profile == "paper":
        return 2.0 * ((x - x0) ** 2) / (sigma**2) * phi
    elif profile == "outgoing":
        return 2.0 * (x - x0) / (sigma**2) * phi
    else:
        raise ValueError(f"Unknown velocity profile: {profile}")
