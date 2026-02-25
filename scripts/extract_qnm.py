from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import os
import numpy as np

from src.config import load_config
from src.utils import ensure_dir, save_json
from src.qnm import qnm_method_1, qnm_method_2, percentage_errors
from src.plotting import plot_ringdown, plot_ringdown_overlay


def _fmt_pct(val: float) -> str:
    """Format percentage error, handling NaN."""
    if val != val:  # NaN check
        return "N/A"
    return f"{val:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", choices=["fd", "pinn"], required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]
    xq = float(cfg["evaluation"]["xq"])
    potential = cfg["physics"]["potential"]       # "zerilli" or "regge_wheeler"
    ell = int(cfg["physics"]["l"])                # angular mode number

    if args.source == "fd":
        npz = np.load(os.path.join("outputs", "fd", f"{name}_fd.npz"))
        x, t, phi = npz["x"], npz["t"], npz["phi"]
        tag = "fd"
    else:
        # expects you ran scripts/run_pinn.py already
        npz = np.load(os.path.join("outputs", "pinn", name, f"{name}_pinn.npz"))
        x, t, phi = npz["x"], npz["t"], npz["phi"]
        tag = "pinn"

    # pick nearest x index
    ix = int(np.argmin(np.abs(x - xq)))
    y = phi[:, ix]

    t_start = float(cfg["qnm"]["t_start"])
    t_end = float(cfg["qnm"]["t_end"])

    m1 = qnm_method_1(t, y, t_start=t_start, t_end=t_end)
    m2 = qnm_method_2(t, y, t_start=t_start, t_end=t_end)

    # Compute percentage errors vs theoretical values (Patel et al. Table 3 style)
    e1 = percentage_errors(m1, potential=potential, ell=ell)
    e2 = percentage_errors(m2, potential=potential, ell=ell)

    # Merge percentage errors into result dicts for saving
    m1_full = {**m1, **e1}
    m2_full = {**m2, **e2}

    outdir = os.path.join("outputs", "qnm", name)
    ensure_dir(outdir)

    save_json(os.path.join(outdir, f"{tag}_method1.json"), m1_full)
    save_json(os.path.join(outdir, f"{tag}_method2.json"), m2_full)

    plot_ringdown(t, y, os.path.join(outdir, f"{tag}_ringdown.png"), title=f"Ringdown at xq={xq} ({tag})")

    # Print formatted comparison table (matching Patel et al. Table 3 format)
    print(f"\n{'='*65}")
    print(f"  QNM Extraction: {tag.upper()} | {potential} l={ell} | xq={xq}")
    print(f"  Theory:  omega*M = {e1['omega_theory']},  tau/M = {e1['tau_theory']}")
    print(f"{'='*65}")
    print(f"  {'Method':<10} {'tau/M':>10} {'(% err)':>10} {'omega*M':>10} {'(% err)':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Method 1':<10} {m1['tau']:>10.4f} {_fmt_pct(e1['tau_pct_err']):>10} {m1['omega']:>10.4f} {_fmt_pct(e1['omega_pct_err']):>10}")
    print(f"  {'Method 2':<10} {m2['tau']:>10.4f} {_fmt_pct(e2['tau_pct_err']):>10} {m2['omega']:>10.4f} {_fmt_pct(e2['omega_pct_err']):>10}")
    print(f"{'='*65}\n")

    print(f"[QNM:{tag}] Outputs in: {outdir}")


if __name__ == "__main__":
    main()
