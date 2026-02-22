from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import os
import numpy as np

from src.utils import rmsd, mad, rl2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", required=True, help="path to fd .npz")
    ap.add_argument("--pinn", required=True, help="path to pinn .npz")
    args = ap.parse_args()

    fd = np.load(args.fd)
    pn = np.load(args.pinn)

    phi_fd = fd["phi"]
    phi_pn = pn["phi"]

    print("RMSD:", rmsd(phi_fd, phi_pn))
    print("MAD:", mad(phi_fd, phi_pn))
    print("RL2:", rl2(phi_fd, phi_pn))


if __name__ == "__main__":
    main()
