#!/usr/bin/env python3
"""
Extract QNMs (Method 1 & 2) for both FD and PINN across all experiment runs.

Usage:
    python scripts/extract_all_qnms.py                     # all runs
    python scripts/extract_all_qnms.py --config configs/zerilli_l2_rad_k2.yaml  # single run

This script is the standardised replacement for running extract_qnm.py
separately with --source fd and --source pinn.  It saves:

    outputs/qnm/<name>/fd_method1.json
    outputs/qnm/<name>/fd_method2.json
    outputs/qnm/<name>/pinn_method1.json
    outputs/qnm/<name>/pinn_method2.json
    outputs/qnm/<name>/fd_ringdown.png
    outputs/qnm/<name>/pinn_ringdown.png

Notes:
    - FD data is loaded from outputs/pinn/<name>/<name>_fd.npz (saved by
      run_pinn.py alongside the PINN data).
    - PINN data is loaded from outputs/pinn/<name>/<name>_pinn.npz.
    - If the FD .npz is missing, falls back to outputs/fd/<name>_fd.npz.
"""
from __future__ import annotations

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from src.config import load_config
from src.utils import ensure_dir, save_json
from src.qnm import qnm_method_1, qnm_method_2, percentage_errors
from src.plotting import plot_ringdown


# ── helpers ──────────────────────────────────────────────────────


def _fmt_pct(val: float) -> str:
    """Format percentage error, handling NaN."""
    if val != val:  # NaN check
        return "  N/A"
    return f"{val:5.2f}%"


def _extract_one(
    tag: str,
    t: np.ndarray,
    y: np.ndarray,
    t_start: float,
    t_end: float,
    potential: str,
    ell: int,
    outdir: str,
) -> None:
    """Run Method 1 & 2 on a single waveform and save results."""
    m1 = qnm_method_1(t, y, t_start=t_start, t_end=t_end)
    m2 = qnm_method_2(t, y, t_start=t_start, t_end=t_end)

    e1 = percentage_errors(m1, potential=potential, ell=ell)
    e2 = percentage_errors(m2, potential=potential, ell=ell)

    m1_full = {**m1, **e1}
    m2_full = {**m2, **e2}

    save_json(os.path.join(outdir, f"{tag}_method1.json"), m1_full)
    save_json(os.path.join(outdir, f"{tag}_method2.json"), m2_full)

    plot_ringdown(
        t, y,
        os.path.join(outdir, f"{tag}_ringdown.png"),
        title=f"Ringdown at xq ({tag})",
    )

    return m1, m2, e1, e2


def _print_table(tag: str, m1, m2, e1, e2, name: str) -> None:
    """Print a formatted comparison table for one source."""
    print(f"  {tag.upper():<6} M1  tau={m1['tau']:8.4f} ({_fmt_pct(e1['tau_pct_err'])})  "
          f"omega={m1['omega']:7.4f} ({_fmt_pct(e1['omega_pct_err'])})")
    print(f"  {'':<6} M2  tau={m2['tau']:8.4f} ({_fmt_pct(e2['tau_pct_err'])})  "
          f"omega={m2['omega']:7.4f} ({_fmt_pct(e2['omega_pct_err'])})")


# ── main ─────────────────────────────────────────────────────────


def process_config(config_path: str) -> None:
    """Extract QNMs for a single config file."""
    cfg = load_config(config_path)
    name = cfg["experiment"]["name"]
    xq = float(cfg["evaluation"]["xq"])
    potential = cfg["physics"]["potential"]
    ell = int(cfg["physics"]["l"])
    t_start = float(cfg["qnm"]["t_start"])
    t_end = float(cfg["qnm"]["t_end"])

    outdir = os.path.join("outputs", "qnm", name)
    ensure_dir(outdir)

    # ── Locate data files ────────────────────────────────────────
    pinn_dir = os.path.join("outputs", "pinn", name)
    fd_npz_path = os.path.join(pinn_dir, f"{name}_fd.npz")
    pinn_npz_path = os.path.join(pinn_dir, f"{name}_pinn.npz")

    # Fallback: FD might be in outputs/fd/
    if not os.path.isfile(fd_npz_path):
        fd_npz_path = os.path.join("outputs", "fd", f"{name}_fd.npz")

    has_fd = os.path.isfile(fd_npz_path)
    has_pinn = os.path.isfile(pinn_npz_path)

    if not has_fd and not has_pinn:
        print(f"  [{name}] SKIP — no data files found")
        return

    # ── Extract ringdown at xq ───────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {name}  |  {potential} l={ell}  |  xq={xq}")
    print(f"  Theory:  omega*M = {0.3737},  tau/M = {11.241}")
    print(f"{'='*65}")

    if has_fd:
        fd = np.load(fd_npz_path)
        x_fd, t_fd, phi_fd = fd["x"], fd["t"], fd["phi"]
        ix = int(np.argmin(np.abs(x_fd - xq)))
        y_fd = phi_fd[:, ix]
        m1_fd, m2_fd, e1_fd, e2_fd = _extract_one(
            "fd", t_fd, y_fd, t_start, t_end, potential, ell, outdir,
        )
        _print_table("fd", m1_fd, m2_fd, e1_fd, e2_fd, name)
    else:
        print(f"  [{name}] No FD data — skipping FD extraction")

    if has_pinn:
        pn = np.load(pinn_npz_path)
        x_pn, t_pn, phi_pn = pn["x"], pn["t"], pn["phi"]
        ix = int(np.argmin(np.abs(x_pn - xq)))
        y_pn = phi_pn[:, ix]
        m1_pn, m2_pn, e1_pn, e2_pn = _extract_one(
            "pinn", t_pn, y_pn, t_start, t_end, potential, ell, outdir,
        )
        _print_table("pinn", m1_pn, m2_pn, e1_pn, e2_pn, name)
    else:
        print(f"  [{name}] No PINN data — skipping PINN extraction")

    print(f"  Outputs → {outdir}")


def main():
    ap = argparse.ArgumentParser(
        description="Extract QNMs for all (or one) experiment config(s).",
    )
    ap.add_argument(
        "--config", default=None,
        help="Path to a single config YAML.  If omitted, processes all "
             "configs in configs/ that have matching PINN output data.",
    )
    args = ap.parse_args()

    if args.config:
        process_config(args.config)
    else:
        # Discover all configs that have corresponding PINN output directories
        config_files = sorted(glob.glob("configs/zerilli_l2*.yaml"))
        if not config_files:
            print("No config files found in configs/")
            return

        for cfg_path in config_files:
            cfg = load_config(cfg_path)
            name = cfg["experiment"]["name"]
            pinn_dir = os.path.join("outputs", "pinn", name)
            if os.path.isdir(pinn_dir):
                process_config(cfg_path)
            else:
                print(f"  [{name}] SKIP — no output directory")

    print(f"\n{'='*65}")
    print("  All done.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
