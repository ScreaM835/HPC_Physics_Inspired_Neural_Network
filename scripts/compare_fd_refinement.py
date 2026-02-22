from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.utils import ensure_dir, rmsd, mad, rl2


def _parse_times(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _xlim_tuple(xlim: List[float] | None) -> Tuple[float, float] | None:
    if xlim is None:
        return None
    if len(xlim) != 2:
        raise ValueError("--xlim must have exactly two values: XMIN XMAX")
    return float(xlim[0]), float(xlim[1])


def _find_nested_restriction(
    x1: np.ndarray,
    t1: np.ndarray,
    x2: np.ndarray,
    t2: np.ndarray,
    dx1: float,
    dt1: float,
    dx2: float,
    dt2: float,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Return indices (it2, ix2) such that x2[ix2]==x1 and t2[it2]==t1.

    If grids are nested by constant integer ratios (the intended refinement test),
    we can compare without interpolation, avoiding extra numerical artifacts.
    """
    # ratios should be integers: dx1 = sx * dx2 and dt1 = st * dt2
    sx = int(round(dx1 / dx2))
    st = int(round(dt1 / dt2))
    if not np.isclose(dx1, sx * dx2):
        return None
    if not np.isclose(dt1, st * dt2):
        return None
    if sx <= 0 or st <= 0:
        return None

    # check boundary alignment and nesting
    if x2.size < 1 or t2.size < 1:
        return None
    if not (np.isclose(x1[0], x2[0]) and np.isclose(x1[-1], x2[-1])):
        return None
    if not (np.isclose(t1[0], t2[0]) and np.isclose(t1[-1], t2[-1])):
        return None
    if x2.size < (x1.size - 1) * sx + 1:
        return None
    if t2.size < (t1.size - 1) * st + 1:
        return None

    ix2 = np.arange(0, x2.size, sx)
    it2 = np.arange(0, t2.size, st)
    if ix2.size != x1.size or it2.size != t1.size:
        return None
    if not np.allclose(x1, x2[ix2], rtol=0, atol=1e-12):
        return None
    if not np.allclose(t1, t2[it2], rtol=0, atol=1e-12):
        return None
    return it2, ix2


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare two FD solutions (coarse vs refined) and plot snapshots + differences. "
            "Intended for the 'FD refinement test' diagnostic."
        )
    )
    ap.add_argument("--fd_coarse", required=True, help="Path to coarse FD .npz")
    ap.add_argument("--fd_refined", required=True, help="Path to refined FD .npz")
    ap.add_argument(
        "--times",
        default="10,20,30,40",
        help="Comma-separated snapshot times (in M units). Default: 10,20,30,40",
    )
    ap.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Optional x-limits for plots (e.g. --xlim -20 20)",
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join("outputs", "diagnostics"),
        help="Output directory for plots and summary.",
    )
    args = ap.parse_args()

    times = _parse_times(args.times)
    xlim = _xlim_tuple(args.xlim)
    outdir = args.outdir
    ensure_dir(outdir)

    fd1 = np.load(args.fd_coarse)
    fd2 = np.load(args.fd_refined)

    x1, t1, phi1 = fd1["x"], fd1["t"], fd1["phi"]
    x2, t2, phi2 = fd2["x"], fd2["t"], fd2["phi"]

    dx1 = float(fd1.get("dx", x1[1] - x1[0]))
    dt1 = float(fd1.get("dt", t1[1] - t1[0]))
    dx2 = float(fd2.get("dx", x2[1] - x2[0]))
    dt2 = float(fd2.get("dt", t2[1] - t2[0]))

    nested = _find_nested_restriction(x1, t1, x2, t2, dx1, dt1, dx2, dt2)
    if nested is None:
        raise RuntimeError(
            "Refined grid is not a clean nesting of the coarse grid. "
            "For this diagnostic, keep the same xmin/xmax/tmin/tmax and choose dx,dt ratios as integers."
        )
    it2, ix2 = nested
    phi2_on_1 = phi2[np.ix_(it2, ix2)]

    # Global metrics over the full shared spacetime grid (coarse grid)
    metrics = {
        "RMSD": float(rmsd(phi1, phi2_on_1)),
        "MAD": float(mad(phi1, phi2_on_1)),
        "RL2": float(rl2(phi1, phi2_on_1)),
    }

    # Write a small text summary
    summary_path = os.path.join(outdir, "fd_refinement_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FD refinement comparison (coarse vs refined)\n")
        f.write(f"coarse:  dx={dx1:g}, dt={dt1:g}, Nx={x1.size}, Nt={t1.size - 1}\n")
        f.write(f"refined: dx={dx2:g}, dt={dt2:g}, Nx={x2.size}, Nt={t2.size - 1}\n")
        f.write("\nMetrics on coarse grid (refined restricted to coarse points):\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.8g}\n")

    print(f"[FD refine] Wrote summary: {summary_path}")
    print("[FD refine] Metrics:", metrics)

    # --- Plot overlays (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, tt in zip(axes, times):
        idx = int(np.argmin(np.abs(t1 - tt)))
        ax.plot(x1, phi1[idx], label="FD coarse")
        ax.plot(x1, phi2_on_1[idx], label="FD refined")
        ax.set_title(f"t/M = {t1[idx]:.0f}")
        ax.grid(True, alpha=0.3)
        if xlim is not None:
            ax.set_xlim(*xlim)

    axes[0].legend(fontsize=8)
    fig.suptitle("FD refinement test — Zerilli ℓ=2 (coarse vs refined)")
    fig.tight_layout()
    out_overlay = os.path.join(outdir, "fd_refinement_snapshots.png")
    fig.savefig(out_overlay, dpi=200)
    plt.close(fig)
    print(f"[FD refine] Wrote: {out_overlay}")

    # --- Plot absolute differences (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, tt in zip(axes, times):
        idx = int(np.argmin(np.abs(t1 - tt)))
        ax.plot(x1, np.abs(phi1[idx] - phi2_on_1[idx]))
        ax.set_title(f"t/M = {t1[idx]:.0f}")
        ax.grid(True, alpha=0.3)
        if xlim is not None:
            ax.set_xlim(*xlim)

    fig.suptitle("|FD coarse − FD refined| (restricted to coarse grid)")
    fig.tight_layout()
    out_diff = os.path.join(outdir, "fd_refinement_absdiff.png")
    fig.savefig(out_diff, dpi=200)
    plt.close(fig)
    print(f"[FD refine] Wrote: {out_diff}")


if __name__ == "__main__":
    main()
