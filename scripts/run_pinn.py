from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import os
import numpy as np

from src.config import load_config
from src.fd_solver import solve_fd
from src.pinn import train_pinn, eval_on_grid
from src.utils import ensure_dir, save_json, rmsd, mad, rl2
from src.plotting import (
    plot_snapshots, plot_abs_diff_snapshots, plot_loss,
    plot_snapshots_zoomed, plot_error_heatmap, plot_ringdown_overlay,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="Resume training from the latest checkpoint")
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="Save a checkpoint every N iterations (default: 500)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]

    # --- FD baseline ---
    fd = solve_fd(cfg)
    x, t, phi_fd = fd["x"], fd["t"], fd["phi"]

    # --- Train PINN (with checkpointing) ---
    ckpt_dir = os.path.join("outputs", "pinn", name, "checkpoints")
    model, hist = train_pinn(
        cfg,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )

    # --- Evaluate ---
    # If decay factoring is enabled, convert g -> Psi = exp(-t/tau)*g
    df_cfg = cfg["pinn"].get("decay_factor", {})
    df_tau = float(df_cfg.get("tau", 0.0)) if df_cfg.get("enabled", False) else 0.0
    phi_pinn = eval_on_grid(model, x=x, t=t, dtype=cfg["pinn"]["dtype"],
                            decay_factor_tau=df_tau)

    metrics = {
        "RMSD": rmsd(phi_fd, phi_pinn),
        "MAD": mad(phi_fd, phi_pinn),
        "RL2": rl2(phi_fd, phi_pinn),
    }

    outdir = os.path.join("outputs", "pinn", name)
    ensure_dir(outdir)

    np.savez_compressed(os.path.join(outdir, f"{name}_fd.npz"), **fd)
    np.savez_compressed(os.path.join(outdir, f"{name}_pinn.npz"), x=x, t=t, phi=phi_pinn)
    save_json(os.path.join(outdir, "metrics.json"), metrics)
    save_json(os.path.join(outdir, "loss_history.json"), hist)

    # plots similar to the target paper
    times = [10.0, 20.0, 30.0, 40.0]
    plot_snapshots(
        x, t, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "snapshots.png"),
        title=f"Snapshots — {cfg['physics']['potential'].title()} potential (l={cfg['physics']['l']})"
    )
    plot_abs_diff_snapshots(
        x, t, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "abs_diff_snapshots.png"),
        title=f"Absolute difference — {cfg['physics']['potential'].title()} potential (l={cfg['physics']['l']})"
    )
    plot_loss(hist, os.path.join(outdir, "loss.png"), title="Loss evolution (Adam + LBFGS)")

    # Zoomed snapshots (matching paper presentation)
    plot_snapshots_zoomed(
        x, t, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "snapshots_zoomed.png"),
        title=f"Snapshots (zoomed) — {cfg['physics']['potential'].title()} potential (l={cfg['physics']['l']})",
    )

    # Pointwise error heatmap
    plot_error_heatmap(
        x, t, phi_fd, phi_pinn,
        outpath=os.path.join(outdir, "error_heatmap.png"),
        title=f"Pointwise error — {cfg['physics']['potential'].title()} (l={cfg['physics']['l']})",
    )
    plot_error_heatmap(
        x, t, phi_fd, phi_pinn,
        outpath=os.path.join(outdir, "error_heatmap_zoomed.png"),
        title=f"Pointwise error (zoomed) — {cfg['physics']['potential'].title()} (l={cfg['physics']['l']})",
        xlim=(-20.0, 60.0),
    )

    # Ringdown overlay at xq
    xq = float(cfg["evaluation"]["xq"])
    ix = int(np.argmin(np.abs(x - xq)))
    plot_ringdown_overlay(
        t, phi_fd[:, ix], phi_pinn[:, ix],
        outpath=os.path.join(outdir, "ringdown_overlay.png"),
        title=f"Ringdown — {cfg['physics']['potential'].title()} (l={cfg['physics']['l']})",
        xq=xq,
    )

    print("[PINN] Metrics:", metrics)
    print(f"[PINN] Outputs in: {outdir}")


if __name__ == "__main__":
    main()
