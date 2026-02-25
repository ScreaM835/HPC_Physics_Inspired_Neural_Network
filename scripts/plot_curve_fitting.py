#!/usr/bin/env python3
"""
Generate the paper-style curve-fitting plot for a given experiment run.

Produces ``curve_fitting.png`` with five reconstructed damped-cosine curves
plotted as log|Φ| vs t/M:

    TRUE  – Leaver theoretical QNM values (ω, τ) with A, φ fitted to FD data
    FD₁   – FD waveform reconstructed using Method 1 (FFT + envelope) QNM values
    FD₂   – FD waveform reconstructed using Method 2 (curve_fit) QNM values
    PINN₁ – PINN waveform reconstructed using Method 1 QNM values
    PINN₂ – PINN waveform reconstructed using Method 2 QNM values

Usage:
    # Single run  (saves to outputs/pinn/<name>/curve_fitting.png)
    python scripts/plot_curve_fitting.py --config configs/zerilli_l2_rad_k2.yaml

    # All runs that have both FD and PINN data
    python scripts/plot_curve_fitting.py

Dependencies:
    Requires QNM JSONs to already exist in outputs/qnm/<name>/.
    Run  ``python scripts/extract_all_qnms.py``  first if they are missing.
"""
from __future__ import annotations

import os
import sys
import glob
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from scipy.optimize import curve_fit

from src.config import load_config
from src.qnm import _damped_cos, THEORY
from src.utils import ensure_dir


# ── helpers ──────────────────────────────────────────────────────


def _fit_amplitude_phase(
    t: np.ndarray,
    y: np.ndarray,
    omega: float,
    tau: float,
) -> tuple[float, float]:
    """
    Given fixed ω and τ, fit A and φ to the raw waveform by minimising
    ‖y - A exp(-t/τ) cos(ωt + φ)‖².

    This is needed for Method 1 (which returns only ω, τ) and for the
    TRUE curve (which uses theoretical ω, τ).
    """

    def model(t, A, phi):
        return _damped_cos(t, A, tau, omega, phi)

    A0 = float(np.max(np.abs(y)))
    popt, _ = curve_fit(model, t, y, p0=[A0, 0.0], maxfev=50000)
    return float(popt[0]), float(popt[1])


def _load_qnm_json(path: str) -> dict | None:
    """Load a QNM JSON file, returning None if it doesn't exist."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── main plotting function ───────────────────────────────────────


def plot_curve_fitting(
    cfg_path: str,
    save: bool = True,
) -> None:
    """
    Generate the curve-fitting plot for one experiment config.

    Parameters
    ----------
    cfg_path : str
        Path to the experiment YAML config.
    save : bool
        If True, save the figure to outputs/pinn/<name>/curve_fitting.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(cfg_path)
    name = cfg["experiment"]["name"]
    xq = float(cfg["evaluation"]["xq"])
    potential = cfg["physics"]["potential"]
    ell = int(cfg["physics"]["l"])
    t_start = float(cfg["qnm"]["t_start"])
    t_end = float(cfg["qnm"]["t_end"])

    # ── Load data ────────────────────────────────────────────────
    pinn_dir = os.path.join("outputs", "pinn", name)
    fd_npz = os.path.join(pinn_dir, f"{name}_fd.npz")
    pinn_npz = os.path.join(pinn_dir, f"{name}_pinn.npz")

    # Fallback for FD
    if not os.path.isfile(fd_npz):
        fd_npz = os.path.join("outputs", "fd", f"{name}_fd.npz")

    if not os.path.isfile(fd_npz) or not os.path.isfile(pinn_npz):
        print(f"  [{name}] SKIP — missing data (FD: {os.path.isfile(fd_npz)}, "
              f"PINN: {os.path.isfile(pinn_npz)})")
        return

    fd = np.load(fd_npz)
    pn = np.load(pinn_npz)

    x_fd, t_fd, phi_fd = fd["x"], fd["t"], fd["phi"]
    x_pn, t_pn, phi_pn = pn["x"], pn["t"], pn["phi"]

    # Extract ringdown at x = xq
    ix_fd = int(np.argmin(np.abs(x_fd - xq)))
    ix_pn = int(np.argmin(np.abs(x_pn - xq)))
    y_fd = phi_fd[:, ix_fd]
    y_pn = phi_pn[:, ix_pn]

    # Restrict to fit window [t_start, t_end]
    mask_fd = (t_fd >= t_start) & (t_fd <= t_end)
    mask_pn = (t_pn >= t_start) & (t_pn <= t_end)
    tt_fd = t_fd[mask_fd]
    yy_fd = y_fd[mask_fd]
    tt_pn = t_pn[mask_pn]
    yy_pn = y_pn[mask_pn]

    # ── Load QNM JSONs ───────────────────────────────────────────
    qnm_dir = os.path.join("outputs", "qnm", name)
    fd_m1 = _load_qnm_json(os.path.join(qnm_dir, "fd_method1.json"))
    fd_m2 = _load_qnm_json(os.path.join(qnm_dir, "fd_method2.json"))
    pn_m1 = _load_qnm_json(os.path.join(qnm_dir, "pinn_method1.json"))
    pn_m2 = _load_qnm_json(os.path.join(qnm_dir, "pinn_method2.json"))

    if fd_m1 is None or fd_m2 is None or pn_m1 is None or pn_m2 is None:
        print(f"  [{name}] SKIP — missing QNM JSONs in {qnm_dir}")
        print(f"    Run: python scripts/extract_all_qnms.py --config {cfg_path}")
        return

    # ── Theoretical reference ────────────────────────────────────
    ref = THEORY[potential][ell]
    omega_true, tau_true = ref["omega"], ref["tau"]

    # ── Reconstruct curves ───────────────────────────────────────
    # TRUE: fix ω, τ to Leaver values, fit A, φ to FD data
    A_true, phi_true = _fit_amplitude_phase(tt_fd, yy_fd, omega_true, tau_true)
    y_true = _damped_cos(tt_fd, A_true, tau_true, omega_true, phi_true)

    # FD Method 1: fix ω, τ from M1, fit A, φ to FD data
    A_fd1, phi_fd1 = _fit_amplitude_phase(tt_fd, yy_fd, fd_m1["omega"], fd_m1["tau"])
    y_fd1 = _damped_cos(tt_fd, A_fd1, fd_m1["tau"], fd_m1["omega"], phi_fd1)

    # FD Method 2: use A, φ, ω, τ directly from curve_fit
    y_fd2 = _damped_cos(tt_fd, fd_m2["A"], fd_m2["tau"], fd_m2["omega"], fd_m2["phi"])

    # PINN Method 1: fix ω, τ from M1, fit A, φ to PINN data
    A_pn1, phi_pn1 = _fit_amplitude_phase(tt_pn, yy_pn, pn_m1["omega"], pn_m1["tau"])
    y_pn1 = _damped_cos(tt_pn, A_pn1, pn_m1["tau"], pn_m1["omega"], phi_pn1)

    # PINN Method 2: use A, φ, ω, τ directly from curve_fit
    y_pn2 = _damped_cos(tt_pn, pn_m2["A"], pn_m2["tau"], pn_m2["omega"], pn_m2["phi"])

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.semilogy(tt_fd, np.abs(y_true) + 1e-30,  label="TRUE",     linewidth=2.0, color="black")
    ax.semilogy(tt_fd, np.abs(y_fd1) + 1e-30,   label=r"FD$_1$",  linewidth=1.3, linestyle="--", color="tab:blue")
    ax.semilogy(tt_fd, np.abs(y_fd2) + 1e-30,   label=r"FD$_2$",  linewidth=1.3, linestyle="-",  color="tab:blue")
    ax.semilogy(tt_pn, np.abs(y_pn1) + 1e-30,   label=r"PINN$_1$", linewidth=1.3, linestyle="--", color="tab:orange")
    ax.semilogy(tt_pn, np.abs(y_pn2) + 1e-30,   label=r"PINN$_2$", linewidth=1.3, linestyle="-",  color="tab:orange")

    ax.set_xlabel(r"$t / M$", fontsize=12)
    ax.set_ylabel(r"$|\Phi(x_q, t)|$", fontsize=12)
    ax.set_title(f"Curve Fitting — {name}  ($x_q = {xq}$)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        outpath = os.path.join(pinn_dir, "curve_fitting.png")
        ensure_dir(os.path.dirname(outpath))
        fig.savefig(outpath, dpi=200)
        print(f"  [{name}] Saved → {outpath}")

    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Generate paper-style curve-fitting plots.",
    )
    ap.add_argument(
        "--config", default=None,
        help="Path to a single config YAML.  If omitted, processes all "
             "configs with available data.",
    )
    args = ap.parse_args()

    if args.config:
        plot_curve_fitting(args.config)
    else:
        config_files = sorted(glob.glob("configs/zerilli_l2*.yaml"))
        if not config_files:
            print("No config files found in configs/")
            return

        for cfg_path in config_files:
            cfg = load_config(cfg_path)
            name = cfg["experiment"]["name"]
            pinn_dir = os.path.join("outputs", "pinn", name)
            if os.path.isdir(pinn_dir):
                plot_curve_fitting(cfg_path)
            else:
                print(f"  [{name}] SKIP — no output directory")

    print("\nAll done.")


if __name__ == "__main__":
    main()
