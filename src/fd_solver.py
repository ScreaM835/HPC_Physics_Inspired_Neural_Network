from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .potentials import V_of_x
from .initial_data import gaussian_phi, gaussian_phi_t


def _one_sided_dx_left(u: np.ndarray, dx: float) -> float:
    # 2nd-order one-sided derivative at i=0
    return (-3.0 * u[0] + 4.0 * u[1] - 1.0 * u[2]) / (2.0 * dx)


def _one_sided_dx_right(u: np.ndarray, dx: float) -> float:
    # 2nd-order one-sided derivative at i=-1
    return (3.0 * u[-1] - 4.0 * u[-2] + 1.0 * u[-3]) / (2.0 * dx)


def _second_derivative(u: np.ndarray, dx: float) -> np.ndarray:
    """
    2nd-order finite-difference approximation to u_xx on a uniform grid.
    Uses one-sided 2nd-order formulas on the boundaries.
    """
    N = u.size
    uxx = np.empty_like(u)
    # interior
    uxx[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx**2)
    # boundaries (2nd-order one-sided second derivative)
    uxx[0] = (2.0 * u[0] - 5.0 * u[1] + 4.0 * u[2] - 1.0 * u[3]) / (dx**2)
    uxx[-1] = (2.0 * u[-1] - 5.0 * u[-2] + 4.0 * u[-3] - 1.0 * u[-4]) / (dx**2)
    return uxx


def _apply_radiative_bc(u: np.ndarray, v: np.ndarray, dx: float) -> None:
    """
    Radiative (Sommerfeld) boundary conditions, consistent with the target paper:

      left (x -> -∞):  (∂t - ∂x)u = 0  =>  v = u_x
      right (x -> +∞): (∂t + ∂x)u = 0  =>  v = -u_x
    """
    ux_l = _one_sided_dx_left(u, dx)
    ux_r = _one_sided_dx_right(u, dx)
    v[0] = ux_l
    v[-1] = -ux_r


def solve_fd(config: Dict) -> Dict[str, np.ndarray]:
    """
    Solve the 1+1 master equation with a method-of-lines FD scheme and RK4 time integration.

    PDE convention used here is the *standard* stable form:
        u_tt - u_xx + V(x) u = 0  =>  u_tt = u_xx - V u.

    Returns a dict containing x, t, phi[t_index, x_index], and V(x).
    """
    M = float(config["physics"]["M"])
    l = int(config["physics"]["l"])
    potential = config["physics"]["potential"]

    xmin = float(config["domain"]["xmin"])
    xmax = float(config["domain"]["xmax"])
    tmin = float(config["domain"]["tmin"])
    tmax = float(config["domain"]["tmax"])

    dx = float(config["fd"]["dx"])
    dt = float(config["fd"]["dt"])

    # derive grid sizes so dx and dt are honored exactly over the closed intervals
    Nx = int(round((xmax - xmin) / dx)) + 1
    Nt = int(round((tmax - tmin) / dt))

    x = xmin + dx * np.arange(Nx)
    t = tmin + dt * np.arange(Nt + 1)

    Vx = V_of_x(x, M=M, l=l, potential=potential)

    A = float(config["initial_data"]["A"])
    x0 = float(config["initial_data"]["x0"])
    sigma = float(config["initial_data"]["sigma"])
    profile = config["initial_data"]["velocity_profile"]

    u = gaussian_phi(x, A=A, x0=x0, sigma=sigma)
    v = gaussian_phi_t(x, A=A, x0=x0, sigma=sigma, profile=profile)

    # enforce BC at t=0
    _apply_radiative_bc(u, v, dx)

    phi = np.zeros((Nt + 1, Nx), dtype=float)
    phi[0] = u.copy()

    def rhs(u_: np.ndarray, v_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # u_t = v
        du = v_.copy()
        # v_t = u_xx - V u
        uxx = _second_derivative(u_, dx)
        dv = uxx - Vx * u_
        return du, dv

    for n in range(Nt):
        # RK4
        k1u, k1v = rhs(u, v)

        u2 = u + 0.5 * dt * k1u
        v2 = v + 0.5 * dt * k1v
        _apply_radiative_bc(u2, v2, dx)
        k2u, k2v = rhs(u2, v2)

        u3 = u + 0.5 * dt * k2u
        v3 = v + 0.5 * dt * k2v
        _apply_radiative_bc(u3, v3, dx)
        k3u, k3v = rhs(u3, v3)

        u4 = u + dt * k3u
        v4 = v + dt * k3v
        _apply_radiative_bc(u4, v4, dx)
        k4u, k4v = rhs(u4, v4)

        u = u + (dt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        v = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

        _apply_radiative_bc(u, v, dx)
        phi[n + 1] = u.copy()

    return {"x": x, "t": t, "phi": phi, "V": Vx, "dx": dx, "dt": dt}
