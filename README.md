# Project 32 — Schwarzschild QNMs with a PINN (Even parity Zerilli, ℓ=2)

This repository is a **reproducibility-first** implementation of the time-domain
Schwarzschild perturbation problem used in Project 32, focusing on the **even-parity
(Zerilli) potential** with **ℓ = 2**, and extracting the corresponding quasi-normal mode (QNM)
frequency and damping time from the ringdown.

The implementation follows the methodology and parameter choices described in:

- Nirmal Patel, Aycin Aykutalp, Pablo Laguna, *Calculating Quasi-Normal Modes of Schwarzschild Black Holes with Physics Informed Neural Networks* (time-domain PINN + FD reference)

## What you get

- A **finite-difference (FD)** time-domain solver for the 1+1 master equation.
- A **PyTorch PINN** solver with:
  - PDE residual loss,
  - **gradient-enhanced** residual losses (∂x residual and ∂t residual),
  - initial condition + initial velocity losses,
  - radiative (Sommerfeld) boundary losses.
- QNM extraction tools:
  - Method 1: FFT for ω + log-peak linear fit for τ,
  - Method 2: direct nonlinear least-squares fit of `A exp(-t/τ) cos(ω t + φ)`.

## Quickstart

**Important:** the config defaults to an *outgoing* Gaussian pulse initial velocity (\(\Phi_t = -\Phi_x\)) because this reproduces the expected \(\ell=2\) QNM parameters. The paper’s printed Eq. (23) appears to contain a squared term; if you want to follow it literally, set `initial_data.velocity_profile: paper`.


### 1) Create an environment

You can use either `venv` or `conda`. Example with `venv`:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2) Run a quick sanity check (recommended first)

```bash
python scripts/run_pinn.py --config configs/quick_test.yaml
python scripts/extract_qnm.py --config configs/quick_test.yaml --source fd
```

### 3) Run the FD baseline (ground truth)


```bash
python scripts/run_fd.py --config configs/zerilli_l2.yaml
```

This writes an `.npz` file to `outputs/fd/` containing `x`, `t`, `phi`, and the potential data.

### FD refinement diagnostic (isolates FD dispersion / phase error)

If you want to check whether a visible FD–PINN mismatch (commonly near the leading edge around the
potential barrier, e.g. near **x≈0** at **t/M=30**) could be partly due to **FD dispersion**, run the
FD solver on a refined grid.

This repo includes a refinement config that halves the mesh spacings while keeping the CFL factor
fixed (dt/dx = 0.5):

```bash
# coarse (dx=0.2, dt=0.1)
python scripts/run_fd.py --config configs/zerilli_l2.yaml

# refined (dx=0.1, dt=0.05)
python scripts/run_fd.py --config configs/zerilli_l2_fd_refined.yaml
```

Then compare **FD(coarse)** vs **FD(refined)**:

```bash
python scripts/compare_fd_refinement.py \
  --fd_coarse  outputs/fd/zerilli_l2_fd.npz \
  --fd_refined outputs/fd/zerilli_l2_fd_refined_fd.npz \
  --times 10,20,30,40 \
  --xlim -20 20
```

Plots and a small metrics summary are written to `outputs/diagnostics/`.

### 4) Train the PINN and evaluate against FD

```bash
python scripts/run_pinn.py --config configs/zerilli_l2.yaml
```

By default this:
- samples training points,
- trains with Adam then LBFGS,
- evaluates the trained PINN on the FD grid,
- writes metrics + plots to `outputs/pinn/`.

### 5) Extract QNMs from the ringdown

```bash
python scripts/extract_qnm.py --config configs/zerilli_l2.yaml --source fd
python scripts/extract_qnm.py --config configs/zerilli_l2.yaml --source pinn
```

Outputs are saved under `outputs/qnm/`.

## Notes on reproducibility

- PINNs are sensitive to random seeds and optimizer settings. The config exposes:
  - number of residual/IC/BC points,
  - network widths,
  - Adam/LBFGS iteration counts,
  - loss weights (λ vector),
  - residual resampling period.

## Repository layout

- `src/` — library code
- `scripts/` — runnable entry points
- `configs/` — experiment configs
- `report/` — dissertation-style draft text you can extend
- `outputs/` — generated output (created when you run scripts)

## Citation

If you use this repo in a dissertation or report, cite the target paper and the Project 32 brief.
