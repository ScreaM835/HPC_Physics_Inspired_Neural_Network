from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from scipy.special import lambertw
import torch


PotentialName = Literal["zerilli", "regge-wheeler"]


def f_schw(r: np.ndarray, M: float) -> np.ndarray:
    return 1.0 - 2.0 * M / r


def r_of_x(x: np.ndarray, M: float) -> np.ndarray:
    """
    Invert the tortoise coordinate

        x = r + 2M ln(r/(2M) - 1)

    for r(x), using the Lambert W closed form.

    Let r = 2M(1 + y). Then
        x/(2M) = 1 + y + ln y  =>  y e^y = exp(x/(2M) - 1)
        y = W(exp(x/(2M) - 1))
        r = 2M (1 + W(exp(x/(2M) - 1))).

    This is numerically robust on the real branch for r>2M.
    """
    x = np.asarray(x, dtype=float)
    z = np.exp(x / (2.0 * M) - 1.0)
    y = lambertw(z)  # principal branch
    y = np.real(y)
    return 2.0 * M * (1.0 + y)


def V_regge_wheeler(r: np.ndarray, M: float, l: int, s: int = 2) -> np.ndarray:
    """
    Regge–Wheeler family potential as written in the target paper:

        V(r) = f(r) [ 2(n+1)/r^2 + 2(1-s^2) M / r^3 ],
        2n = (l-1)(l+2).

    For gravitational perturbations: s=2.
    """
    r = np.asarray(r, dtype=float)
    n = 0.5 * (l - 1) * (l + 2)
    f = f_schw(r, M)
    return f * (2.0 * (n + 1.0) / r**2 + 2.0 * (1.0 - s**2) * M / r**3)


def V_zerilli(r: np.ndarray, M: float, l: int) -> np.ndarray:
    """
    Zerilli (even parity) potential as written in the target paper:

        V(r) = f(r) * 1/(r^3 (n r + 3M)^2) *
               [ 2 n^2 (n+1) r^3 + 6 n^2 M r^2 + 18 n M^2 r + 18 M^3 ],
        n = (l-1)(l+2)/2.

    """
    r = np.asarray(r, dtype=float)
    n = 0.5 * (l - 1) * (l + 2)
    f = f_schw(r, M)
    num = (
        2.0 * n**2 * (n + 1.0) * r**3
        + 6.0 * n**2 * M * r**2
        + 18.0 * n * (M**2) * r
        + 18.0 * (M**3)
    )
    den = r**3 * (n * r + 3.0 * M) ** 2
    return f * (num / den)


def V_of_x(x: np.ndarray, M: float, l: int, potential: PotentialName) -> np.ndarray:
    r = r_of_x(x, M)
    if potential == "zerilli":
        return V_zerilli(r, M, l)
    elif potential == "regge-wheeler":
        return V_regge_wheeler(r, M, l, s=2)
    else:
        raise ValueError(f"Unknown potential: {potential}")


def V_and_dVdx_of_x(
    x: np.ndarray, M: float, l: int, potential: PotentialName, h: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns V(x) and dV/dx using a centered finite difference in x.
    """
    x = np.asarray(x, dtype=float)
    V0 = V_of_x(x, M, l, potential)
    Vp = V_of_x(x + h, M, l, potential)
    Vm = V_of_x(x - h, M, l, potential)
    dVdx = (Vp - Vm) / (2.0 * h)
    return V0, dVdx


# ---------------------------------------------------------------------------
# Pure-torch implementations (fully inside the autograd graph)
# ---------------------------------------------------------------------------
# These use the Lambert-W identity:  r = 2M(1 + W(exp(x/(2M) - 1)))
# PyTorch does not provide Lambert-W, so we use the fixed-point iteration
#   y_{k+1} = (z * exp(-y_k) + y_k) / (1 + y_k)        (Halley's method)
# which converges in ~5 iterations to machine precision for z > 0.


def _lambert_w_torch(z: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
    """Lambert W₀ via Halley's method, fully differentiable."""
    # Initial guess: for large z, W ≈ ln(z); for small z, W ≈ z.
    w = torch.where(z > 1.0, torch.log(z + 1e-30), z)
    for _ in range(n_iter):
        ew = torch.exp(w)
        wew = w * ew
        f = wew - z
        fp = ew * (w + 1.0)
        fpp = ew * (w + 2.0)
        # Halley update: w -= f / (fp - f*fpp/(2*fp))
        w = w - f / (fp - 0.5 * f * fpp / (fp + 1e-30))
    return w


def r_of_x_torch(x: torch.Tensor, M: float) -> torch.Tensor:
    """Invert tortoise coordinate x → r using Lambert W, fully in torch."""
    z = torch.exp(x / (2.0 * M) - 1.0)
    y = _lambert_w_torch(z)
    return 2.0 * M * (1.0 + y)


def _f_schw_torch(r: torch.Tensor, M: float) -> torch.Tensor:
    return 1.0 - 2.0 * M / r


def V_zerilli_torch(x: torch.Tensor, M: float, l: int) -> torch.Tensor:
    """Zerilli potential V(x*) — pure torch, inside the autograd graph."""
    r = r_of_x_torch(x, M)
    n = 0.5 * (l - 1) * (l + 2)
    f = _f_schw_torch(r, M)
    num = (
        2.0 * n**2 * (n + 1.0) * r**3
        + 6.0 * n**2 * M * r**2
        + 18.0 * n * (M**2) * r
        + 18.0 * (M**3)
    )
    den = r**3 * (n * r + 3.0 * M) ** 2
    return f * (num / den)


def V_regge_wheeler_torch(x: torch.Tensor, M: float, l: int, s: int = 2) -> torch.Tensor:
    """Regge-Wheeler potential V(x*) — pure torch, inside the autograd graph."""
    r = r_of_x_torch(x, M)
    n = 0.5 * (l - 1) * (l + 2)
    f = _f_schw_torch(r, M)
    return f * (2.0 * (n + 1.0) / r**2 + 2.0 * (1.0 - s**2) * M / r**3)


def V_of_x_torch(x: torch.Tensor, M: float, l: int, potential: PotentialName) -> torch.Tensor:
    """Fully differentiable V(x*) for use inside the PINN autograd graph."""
    if potential == "zerilli":
        return V_zerilli_torch(x, M, l)
    elif potential == "regge-wheeler":
        return V_regge_wheeler_torch(x, M, l, s=2)
    else:
        raise ValueError(f"Unknown potential: {potential}")
