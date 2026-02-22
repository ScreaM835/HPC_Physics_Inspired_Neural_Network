from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import os
import numpy as np

from src.config import load_config
from src.fd_solver import solve_fd
from src.utils import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = os.path.join("outputs", "fd")
    ensure_dir(outdir)

    sol = solve_fd(cfg)
    name = cfg["experiment"]["name"]
    outpath = os.path.join(outdir, f"{name}_fd.npz")
    np.savez_compressed(outpath, **sol)
    print(f"[FD] Wrote {outpath}")


if __name__ == "__main__":
    main()
