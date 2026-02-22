from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import os
import numpy as np

from src.config import load_config
from src.utils import ensure_dir, save_json
from src.qnm import qnm_method_1, qnm_method_2
from src.plotting import plot_ringdown, plot_ringdown_overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", choices=["fd", "pinn"], required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]
    xq = float(cfg["evaluation"]["xq"])

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

    outdir = os.path.join("outputs", "qnm", name)
    ensure_dir(outdir)

    save_json(os.path.join(outdir, f"{tag}_method1.json"), m1)
    save_json(os.path.join(outdir, f"{tag}_method2.json"), m2)

    plot_ringdown(t, y, os.path.join(outdir, f"{tag}_ringdown.png"), title=f"Ringdown at xq={xq} ({tag})")

    print(f"[QNM:{tag}] Method1:", m1)
    print(f"[QNM:{tag}] Method2:", m2)
    print(f"[QNM:{tag}] Outputs in: {outdir}")


if __name__ == "__main__":
    main()
