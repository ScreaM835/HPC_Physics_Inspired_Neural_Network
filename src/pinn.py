"""
PINN for the 1+1D Zerilli/Regge-Wheeler equation using DeepXDE.

PDE:  phi_tt - phi_xx + V(x*) phi = 0

Improvements over the base model:
  1. Residual-Adaptive Distribution (RAD) — Wu et al. 2023
     Concentrates collocation points where the PDE residual is largest.
  2. Exponential reweighting of PDE residuals
     Multiplies residuals by exp(t / tau_est) to compensate for the
     amplitude decay of the QNM solution, ensuring late-time accuracy.

Architecture: standard FNN [2, 80, 40, 20, 10, 1] with tanh activation
and A*tanh(y) output transform, matching Patel, Laguna & Shoemaker (2024).
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import deepxde as dde
import numpy as np
import torch

from .potentials import V_of_x_torch
from .initial_data import gaussian_phi, gaussian_phi_t


# ---------------------------------------------------------------------------
# PDE residual
# ---------------------------------------------------------------------------

def _make_pde_func(cfg: Dict):
    """Build the PDE residual function.

    Supports two modes via config:

    1. **Decay factoring** (``decay_factor.enabled: true``):
       Change of dependent variable  g = exp(t/tau) * Psi.  The network
       learns g, which has ~constant amplitude.  The transformed PDE is
           g_tt - g_xx - (2/tau)*g_t + (V + 1/tau^2)*g = 0
       This eliminates the 86x dynamic range that causes MSE to ignore
       late-time phase errors.

    2. **Exponential reweighting** (``tau_est > 0`` without decay_factor):
       Multiplies residuals by exp(t/tau_est).  (Legacy — kept for
       comparison but generally inferior.)

    3. **Standard** (default): plain Zerilli PDE.
    """
    M = float(cfg["physics"]["M"])
    l = int(cfg["physics"]["l"])
    potential = cfg["physics"]["potential"]

    # Decay factoring: change of variable g = exp(t/tau)*Psi
    df_cfg = cfg["pinn"].get("decay_factor", {})
    use_decay_factor = df_cfg.get("enabled", False)
    df_tau = float(df_cfg.get("tau", 0.0)) if use_decay_factor else 0.0

    # Legacy exponential reweighting (only if decay_factor is off)
    tau_est = float(cfg["pinn"].get("tau_est", 0.0))
    use_exp_weight = (not use_decay_factor) and (tau_est > 0.0)

    def pde(x, y):
        """
        PDE residual.  Returns [r, r_x, r_t] for gradient-enhanced training.

        If decay_factor is enabled, the PDE solved is the g-equation:
            g_tt - g_xx - (2/tau)*g_t + (V + 1/tau^2)*g = 0
        Otherwise the standard Zerilli equation:
            phi_tt - phi_xx + V*phi = 0
        """
        y_xx = dde.grad.hessian(y, x, i=0, j=0)
        y_tt = dde.grad.hessian(y, x, i=1, j=1)

        V = V_of_x_torch(x[:, 0:1], M, l, potential)

        if use_decay_factor:
            # g-equation: g_tt - g_xx - (2/tau)*g_t + (V + 1/tau^2)*g = 0
            y_t = dde.grad.jacobian(y, x, i=0, j=1)
            r = y_tt - y_xx - (2.0 / df_tau) * y_t + (V + 1.0 / df_tau**2) * y
        else:
            # Standard Zerilli: phi_tt - phi_xx + V*phi = 0
            r = y_tt - y_xx + V * y

        dr = torch.autograd.grad(
            r, x,
            grad_outputs=torch.ones_like(r),
            create_graph=True,
            retain_graph=True,
        )[0]
        r_x = dr[:, 0:1]
        r_t = dr[:, 1:2]

        # Legacy exponential reweighting (only when decay_factor is off)
        if use_exp_weight:
            t_col = x[:, 1:2]
            w = torch.exp(t_col / tau_est)
            r = r * w
            r_x = r_x * w
            r_t = r_t * w

        return [r, r_x, r_t]

    return pde


# ---------------------------------------------------------------------------
# Standalone PDE residual for RAD (no gradient-enhancement)
# ---------------------------------------------------------------------------

def _make_pde_residual_only(cfg: Dict):
    """Build a lightweight PDE function returning only the primary residual r.

    Used by ResidualAdaptiveResampler (RAD) for evaluating candidate points.
    Skips the gradient-enhanced terms (r_x, r_t) to avoid expensive
    third-order autograd.

    Returns a function (x, y) -> r compatible with DeepXDE's
    model.predict(operator=...) interface.
    """
    M = float(cfg["physics"]["M"])
    l = int(cfg["physics"]["l"])
    potential = cfg["physics"]["potential"]

    # Decay factoring: g-equation for RAD too
    df_cfg = cfg["pinn"].get("decay_factor", {})
    use_decay_factor = df_cfg.get("enabled", False)
    df_tau = float(df_cfg.get("tau", 0.0)) if use_decay_factor else 0.0

    def pde_residual(x, y):
        y_xx = dde.grad.hessian(y, x, i=0, j=0)
        y_tt = dde.grad.hessian(y, x, i=1, j=1)
        V = V_of_x_torch(x[:, 0:1], M, l, potential)
        if use_decay_factor:
            y_t = dde.grad.jacobian(y, x, i=0, j=1)
            return y_tt - y_xx - (2.0 / df_tau) * y_t + (V + 1.0 / df_tau**2) * y
        return y_tt - y_xx + V * y

    return pde_residual


# ---------------------------------------------------------------------------
# Residual-Adaptive Distribution (RAD) — Wu et al. 2023
# ---------------------------------------------------------------------------

class ResidualAdaptiveResampler(dde.callbacks.Callback):
    """Adaptive collocation-point resampling based on PDE residual magnitude.

    Implements both RAD and RAR-D from:
      Wu, Zhu, Tan, Kartha & Lu (2023). "A comprehensive study of
      non-adaptive and residual-based adaptive sampling for PINNs."
      Computer Methods in Applied Mechanics and Engineering, 403, 115671.

    Every ``period`` training steps:
      1. Generate ``num_candidates`` random points in the domain.
      2. Evaluate PDE residual r at each candidate (lightweight, no grad
         enhancement).
      3. Compute sampling probability  p ∝ |r|^k / mean(|r|^k) + c.
      4. RAD:   sample ``num_domain`` points → replace all domain points.
         RAR-D: sample ``num_add``    points → append to existing points.

    Parameters
    ----------
    pde_residual : callable
        Lightweight PDE function  (x, y) -> r  (single tensor, not list).
        Should be ``_make_pde_residual_only(cfg)`` — avoids computing r_x, r_t.
    period : int
        Resample every this many training steps.
    num_candidates : int
        Number of random candidate points to evaluate.
    k : float
        Power for residual weighting (0 = uniform, 1 = linear, 2 = quadratic).
        RAD default: k=1.  RAR-D default: k=0.
    c : float
        Additive constant ensuring minimum probability everywhere.
        RAD default: c=1.  RAR-D default: c=2.
    method : str
        'RAD' (replace all) or 'RAR_D' (append new).
    num_add : int
        Points to add per round (RAR-D only).
    eval_batch_size : int
        Batch size for residual evaluation (controls peak memory).
    """

    def __init__(self, pde_residual, period: int = 1000,
                 num_candidates: int = 50000, k: float = 1.0,
                 c: float = 1.0, method: str = 'RAD',
                 num_add: int = 160, eval_batch_size: int = 5000,
                 anchor_fraction: float = 0.0):
        super().__init__()
        self.pde_residual = pde_residual
        self.period = period
        self.num_candidates = num_candidates
        self.k = k
        self.c = c
        self.method = method
        self.num_add = num_add
        self.eval_batch_size = eval_batch_size
        self.anchor_fraction = anchor_fraction  # fraction of old points to retain

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step == 0 or step % self.period != 0:
            return

        data = self.model.data
        geom = data.geom  # GeometryXTime

        # 1. Generate candidate points
        X_cand = geom.random_points(self.num_candidates)

        # 2. Evaluate PDE residual in batches (avoids large autograd graph)
        all_residuals = []
        for i in range(0, len(X_cand), self.eval_batch_size):
            batch = X_cand[i:i + self.eval_batch_size]
            # pde_residual returns a single tensor, so predict returns
            # a single numpy array (N, 1)
            res = self.model.predict(batch, operator=self.pde_residual)
            all_residuals.append(np.abs(res))
        Y = np.concatenate(all_residuals, axis=0).astype(np.float64)

        # 3. Compute sampling probability
        #    err_eq = |r|^k / mean(|r|^k) + c  (Wu et al. Eq. 2)
        if self.k == 0:
            Y_pow = np.ones_like(Y)
        else:
            Y_pow = np.power(Y, self.k)
        mean_Y = Y_pow.mean()
        if mean_Y < 1e-30:
            # All residuals near zero — fall back to uniform
            prob = np.ones(len(X_cand)) / len(X_cand)
        else:
            prob = (Y_pow / mean_Y + self.c).ravel()
            prob = prob / prob.sum()

        # 4. Sample and update training points
        if self.method == 'RAD':
            num_domain = data.num_domain

            if self.anchor_fraction > 0.0:
                # --- Anchor retention: keep top-residual old points ---
                # Evaluate residual on current domain points
                # data.train_x_all[:num_domain] are domain (collocation) points
                X_old = data.train_x_all[:num_domain]
                old_residuals = []
                for i in range(0, len(X_old), self.eval_batch_size):
                    batch = X_old[i:i + self.eval_batch_size]
                    res = self.model.predict(batch, operator=self.pde_residual)
                    old_residuals.append(np.abs(res))
                Y_old = np.concatenate(old_residuals, axis=0).ravel().astype(np.float64)

                # Keep top anchor_fraction of old points by residual magnitude
                n_keep = int(num_domain * self.anchor_fraction)
                n_new = num_domain - n_keep
                top_ids = np.argsort(Y_old)[-n_keep:]  # indices of highest residual
                X_anchored = X_old[top_ids]

                # Sample n_new from candidates
                do_replace = len(X_cand) >= n_new
                ids = np.random.choice(len(X_cand), size=min(n_new, len(X_cand)),
                                       replace=not do_replace, p=prob)
                X_new = np.concatenate([X_anchored, X_cand[ids]], axis=0)
                data.replace_with_anchors(X_new)
                print(f"  [RAD+anchor] step {step}: kept {n_keep} old + "
                      f"{len(ids)} new pts "
                      f"(max|r|={Y.max():.4e}, mean|r|={Y.mean():.4e}, "
                      f"anchor max|r|={Y_old[top_ids].max():.4e})")
            else:
                # Standard RAD: replace all domain points
                do_replace = len(X_cand) >= num_domain
                ids = np.random.choice(len(X_cand), size=min(num_domain, len(X_cand)),
                                       replace=not do_replace, p=prob)
                data.replace_with_anchors(X_cand[ids])
                print(f"  [RAD] step {step}: replaced {len(ids)} pts "
                      f"(max|r|={Y.max():.4e}, mean|r|={Y.mean():.4e})")
        elif self.method == 'RAR_D':
            ids = np.random.choice(len(X_cand), size=self.num_add,
                                   replace=False, p=prob)
            data.add_anchors(X_cand[ids])
            n_total = len(data.train_x_all)
            print(f"  [RAR-D] step {step}: added {self.num_add} pts "
                  f"(total domain={n_total}, max|r|={Y.max():.4e})")
        else:
            raise ValueError(f"Unknown adaptive sampling method: {self.method}")

        # Sync test data: DeepXDE defaults test_x = train_x at compile time.
        # After modifying training points, test_x is stale.  Update it and
        # the model's train_state so _test() uses consistent shapes.
        data.test_x = data.train_x
        data.test_y = data.train_y
        self.model.train_state.set_data_test(
            data.test_x, data.test_y,
        )


# ---------------------------------------------------------------------------
# Initial / boundary conditions
# ---------------------------------------------------------------------------

def _make_ic_bcs(cfg: Dict, geomtime):
    """Create the four IC/BC objects for the Zerilli equation."""
    A0 = float(cfg["initial_data"]["A"])
    x0_ic = float(cfg["initial_data"]["x0"])
    sigma = float(cfg["initial_data"]["sigma"])
    profile = cfg["initial_data"]["velocity_profile"]
    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])
    tmin = float(cfg["domain"]["tmin"])

    # ---- IC: phi(x, 0) = Gaussian ----
    def phi0_func(x):
        return gaussian_phi(x[:, 0:1], A=A0, x0=x0_ic, sigma=sigma)

    ic_disp = dde.icbc.IC(
        geomtime, phi0_func, lambda _, on_initial: on_initial
    )

    # Decay factoring: velocity IC becomes g_t(x,0) = Psi_t(x,0) + Psi(x,0)/tau
    df_cfg = cfg.get("pinn", {}).get("decay_factor", {})
    use_decay_factor = df_cfg.get("enabled", False)
    df_tau = float(df_cfg.get("tau", 0.0)) if use_decay_factor else 0.0

    # ---- IC: d_phi/dt(x, 0) = v0(x) ----
    #      (or d_g/dt(x, 0) = v0(x) + phi0(x)/tau for decay-factored PDE)
    def vel_func(inputs, outputs, X):
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        # Use `inputs` (tensor) rather than `X` (numpy self.train_x) to
        # guarantee size consistency.  After PDEPointResampler fires,
        # self.train_x is regenerated but X_test may still hold old data,
        # causing a shape mismatch if we use X.
        x_np = inputs[:, 0:1].detach().cpu().numpy()
        v0 = gaussian_phi_t(
            x_np, A=A0, x0=x0_ic, sigma=sigma, profile=profile
        )
        if use_decay_factor:
            # g_t(x,0) = Psi_t(x,0) + Psi_0(x)/tau  (since e^0 = 1)
            phi0 = gaussian_phi(x_np, A=A0, x0=x0_ic, sigma=sigma)
            v0 = v0 + phi0 / df_tau
        v0_t = torch.as_tensor(v0, dtype=outputs.dtype, device=outputs.device)
        return phi_t - v0_t

    ic_vel = dde.icbc.OperatorBC(
        geomtime, vel_func,
        lambda x, on_boundary: np.isclose(x[1], tmin),
    )

    # ---- BC left: Sommerfeld ingoing at x = xmin ----
    #   Standard:       (d_t - d_x) Psi = 0
    #   Decay-factored: (d_t - d_x) g - g/tau = 0
    def bc_left_func(inputs, outputs, X):
        phi_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        bc = phi_t - phi_x
        if use_decay_factor:
            bc = bc - outputs / df_tau
        return bc

    bc_left = dde.icbc.OperatorBC(
        geomtime, bc_left_func,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], xmin),
    )

    # ---- BC right: Sommerfeld outgoing at x = xmax ----
    #   Standard:       (d_t + d_x) Psi = 0
    #   Decay-factored: (d_t + d_x) g - g/tau = 0
    def bc_right_func(inputs, outputs, X):
        phi_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        bc = phi_t + phi_x
        if use_decay_factor:
            bc = bc - outputs / df_tau
        return bc

    bc_right = dde.icbc.OperatorBC(
        geomtime, bc_right_func,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], xmax),
    )

    return [ic_disp, ic_vel, bc_left, bc_right]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(cfg: Dict) -> Tuple[dde.Model, dde.data.TimePDE]:
    """Construct the DeepXDE Model from the experiment config.

    Uses a standard FNN with the paper's architecture [2, 80, 40, 20, 10, 1],
    tanh activation, Glorot uniform initialisation, and A*tanh(y) output
    transform to bound the solution.
    """
    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])
    tmin = float(cfg["domain"]["tmin"])
    tmax = float(cfg["domain"]["tmax"])

    geom = dde.geometry.Interval(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(tmin, tmax)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    pde_func = _make_pde_func(cfg)
    ic_bcs = _make_ic_bcs(cfg, geomtime)

    Nr = int(cfg["pinn"]["Nr"])
    Ni = int(cfg["pinn"]["Ni"])
    Nb = int(cfg["pinn"]["Nb"])

    data = dde.data.TimePDE(
        geomtime,
        pde_func,
        ic_bcs,
        num_domain=Nr,
        num_boundary=Nb,
        num_initial=Ni,
    )

    # Standard FNN — paper architecture
    layers = [2] + [int(w) for w in cfg["pinn"]["hidden_layers"]] + [1]
    net = dde.nn.FNN(layers, "tanh", "Glorot uniform")

    # Output transform: A * tanh(y)
    A_bound = float(cfg["initial_data"]["A"])
    net.apply_output_transform(lambda x, y: A_bound * torch.tanh(y))

    model = dde.Model(data, net)
    return model, data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pinn(
    cfg: Dict,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 500,
    resume: bool = False,
) -> Tuple[dde.Model, Dict[str, List[float]]]:
    """
    Train the PINN with DeepXDE: Adam -> L-BFGS.

    Supports:
      - RAD adaptive sampling during Adam phase
      - Checkpoint/resume for long HPC runs
      - Exponential reweighting (configured via tau_est in config)
    """
    seed = int(cfg["pinn"]["seed"])
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(cfg["pinn"]["dtype"])

    model, data = build_model(cfg)
    loss_weights = [float(w) for w in cfg["pinn"]["lambda"]]

    print(f"[PINN] Loss weights: {loss_weights}")

    tau_est = float(cfg["pinn"].get("tau_est", 0.0))
    if tau_est > 0:
        print(f"[PINN] Exponential reweighting: tau_est = {tau_est}")

    df_cfg = cfg["pinn"].get("decay_factor", {})
    if df_cfg.get("enabled", False):
        df_tau = float(df_cfg["tau"])
        print(f"[PINN] Decay factoring: solving g-equation with tau = {df_tau}")
        print(f"[PINN]   g_tt - g_xx - (2/tau)*g_t + (V + 1/tau^2)*g = 0")
        print(f"[PINN]   Network learns g = exp(t/tau)*Psi (constant amplitude)")

    model_save_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_save_path = os.path.join(checkpoint_dir, "model")

    # ---- Check if Adam already completed (for resume) ----
    adam_done_ckpt = _find_adam_done_checkpoint(checkpoint_dir) if checkpoint_dir else None
    adam_already_done = resume and adam_done_ckpt is not None

    losshistory_adam = None
    lbfgs_resume_ckpt = None

    adam_iters_total = int(cfg["pinn"]["adam"]["iters"])
    adaptive_cfg = cfg["pinn"].get("adaptive_sampling", {})

    if adam_already_done:
        print("[CKPT] Adam already completed — skipping to L-BFGS")
        model.compile("adam", lr=1e-3, loss_weights=loss_weights)
        model.train(iterations=0, display_every=1)
        model.restore(adam_done_ckpt, verbose=1)

        lbfgs_resume_ckpt = _find_latest_checkpoint(
            checkpoint_dir, exclude_prefix="model-adam_done"
        )
        if lbfgs_resume_ckpt is not None:
            print(f"[CKPT] Will restore L-BFGS checkpoint: {lbfgs_resume_ckpt}")

    else:
        # ---- Adam phase ----
        adam_cfg = cfg["pinn"]["adam"]
        adam_iters = int(adam_cfg["iters"])
        lr = float(adam_cfg["lr"])
        resample_period = int(adam_cfg["resample_period"])

        callbacks_adam: List = []

        # Adaptive sampling (RAD) or uniform resampling
        if adaptive_cfg.get("enabled", False):
            method = adaptive_cfg.get("method", "RAD")
            rad_period = int(adaptive_cfg.get("period", 1000))
            num_cand = int(adaptive_cfg.get("num_candidates", 50000))
            k = float(adaptive_cfg.get("k", 1.0))
            c = float(adaptive_cfg.get("c", 1.0))
            num_add = int(adaptive_cfg.get("num_add", 160))
            eval_bs = int(adaptive_cfg.get("eval_batch_size", 5000))
            anchor_frac = float(adaptive_cfg.get("anchor_fraction", 0.0))

            pde_residual = _make_pde_residual_only(cfg)
            callbacks_adam.append(
                ResidualAdaptiveResampler(
                    pde_residual=pde_residual,
                    period=rad_period,
                    num_candidates=num_cand,
                    k=k, c=c,
                    method=method,
                    num_add=num_add,
                    eval_batch_size=eval_bs,
                    anchor_fraction=anchor_frac,
                )
            )
            anchor_str = f", anchor={anchor_frac:.0%}" if anchor_frac > 0 else ""
            print(f"[PINN] Adaptive sampling: {method} "
                  f"(period={rad_period}, k={k}, c={c}, "
                  f"candidates={num_cand}{anchor_str})")
        else:
            callbacks_adam.append(
                dde.callbacks.PDEPointResampler(period=resample_period)
            )

        if model_save_path:
            callbacks_adam.append(
                dde.callbacks.ModelCheckpoint(
                    model_save_path,
                    save_better_only=False,
                    period=checkpoint_every,
                )
            )

        # Resume from mid-Adam checkpoint if available
        model_restore_path = None
        if resume and checkpoint_dir:
            ckpt = _find_latest_checkpoint(checkpoint_dir)
            if ckpt is not None:
                model_restore_path = ckpt
                print(f"[CKPT] Restoring from {ckpt}")

        model.compile("adam", lr=lr, loss_weights=loss_weights)

        print(f"[PINN] Adam: {adam_iters} iters, lr={lr}, "
              f"resample every {resample_period}")
        losshistory_adam, _ = model.train(
            iterations=adam_iters,
            callbacks=callbacks_adam,
            display_every=100,
            model_save_path=model_save_path,
            model_restore_path=model_restore_path,
        )

        if model_save_path:
            model.save(model_save_path + "-adam_done")
            print("[CKPT] Adam complete -- checkpoint saved")

    # ---- L-BFGS phase ----
    lbfgs_cfg = cfg["pinn"]["lbfgs"]
    lbfgs_iters = int(lbfgs_cfg["iters"])

    lbfgs_loss_weights = list(loss_weights)

    # Determine how many L-BFGS iterations have already been completed
    lbfgs_iters_done = 0
    if lbfgs_resume_ckpt is not None:
        m = re.search(r"model-(\d+)\.pt$", lbfgs_resume_ckpt)
        if m:
            ckpt_step = int(m.group(1))
            lbfgs_iters_done = max(0, ckpt_step - adam_iters_total)
            if lbfgs_iters_done > 0:
                print(f"[CKPT] L-BFGS iterations already done: {lbfgs_iters_done}")
            else:
                lbfgs_resume_ckpt = None

    iters_remaining = lbfgs_iters - lbfgs_iters_done
    if iters_remaining <= 0:
        print("[PINN] L-BFGS already completed.")
        return model, _convert_loss_history(losshistory_adam) if losshistory_adam else {}

    print(f"[PINN] L-BFGS: {iters_remaining} iterations remaining")

    dde.optimizers.set_LBFGS_options(
        maxcor=100,
        maxiter=iters_remaining,
        ftol=0,
        gtol=1e-8,
        maxls=50,
    )

    lbfgs_resample_period = int(lbfgs_cfg.get("resample_period", 0))
    if lbfgs_resample_period > 0:
        step_size = min(checkpoint_every, lbfgs_resample_period, iters_remaining)
    else:
        step_size = min(checkpoint_every, iters_remaining)

    from deepxde.optimizers.config import LBFGS_options as _lbfgs_opts
    _lbfgs_opts["iter_per_step"] = step_size
    _lbfgs_opts["fun_per_step"] = int(step_size * 1.25)

    model.compile("L-BFGS", loss_weights=lbfgs_loss_weights)

    if lbfgs_resume_ckpt is not None:
        print(f"[CKPT] Restoring model weights from {lbfgs_resume_ckpt}")
        checkpoint = torch.load(lbfgs_resume_ckpt, weights_only=True)
        model.net.load_state_dict(checkpoint["model_state_dict"])

    callbacks_lbfgs = []
    if model_save_path:
        callbacks_lbfgs.append(
            dde.callbacks.ModelCheckpoint(
                model_save_path,
                save_better_only=False,
                period=checkpoint_every,
            )
        )

    if lbfgs_resample_period > 0:
        callbacks_lbfgs.append(
            dde.callbacks.PDEPointResampler(period=lbfgs_resample_period)
        )

    losshistory_lbfgs, _ = model.train(
        iterations=iters_remaining,
        callbacks=callbacks_lbfgs,
        display_every=100,
    )

    if model_save_path:
        model.save(model_save_path + "-final")
        print("[CKPT] Training complete -- final checkpoint saved")

    # ---- Combine loss histories ----
    if losshistory_adam is not None:
        history = _combine_loss_histories(
            losshistory_adam, losshistory_lbfgs, adam_iters=adam_iters_total
        )
    else:
        history = _convert_loss_history(
            losshistory_lbfgs, step_offset=adam_iters_total, phase="lbfgs"
        )

    return model, history


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(
    checkpoint_dir: str, exclude_prefix: Optional[str] = None
) -> Optional[str]:
    """Find the latest DeepXDE checkpoint in the directory."""
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for f in os.listdir(checkpoint_dir):
        if not f.endswith(".pt"):
            continue
        if exclude_prefix and f.startswith(exclude_prefix):
            continue
        candidates.append(os.path.join(checkpoint_dir, f))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _find_adam_done_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the model-adam_done-*.pt checkpoint if it exists."""
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None
    for f in os.listdir(checkpoint_dir):
        if f.startswith("model-adam_done") and f.endswith(".pt"):
            return os.path.join(checkpoint_dir, f)
    return None


# ---------------------------------------------------------------------------
# Loss-history conversion
# ---------------------------------------------------------------------------

def _convert_loss_history(
    losshistory,
    step_offset: int = 0,
    phase: str = "adam",
) -> Dict[str, List]:
    """Convert a single DeepXDE LossHistory to our dict format.

    DeepXDE records per-component MSEs (unweighted).
    Order: [PDE outputs ..., IC/BCs ...] = [r, r_x, r_t, ic, iv, bl, br].
    """
    loss_names = ["Lr", "Lrx", "Lrt", "Lic", "Liv", "Lbl", "Lbr"]
    history: Dict[str, List] = {name: [] for name in loss_names}
    history["L_total"] = []
    history["w_min"] = []
    history["steps"] = []
    history["phase"] = []

    steps = losshistory.steps
    losses = np.array(losshistory.loss_train)

    for i in range(len(steps)):
        total = 0.0
        for j, name in enumerate(loss_names):
            val = float(losses[i, j]) if j < losses.shape[1] else 0.0
            history[name].append(val)
            total += val
        history["L_total"].append(total)
        history["w_min"].append(1.0)
        history["steps"].append(int(steps[i]) + step_offset)
        history["phase"].append(phase)

    return history


def _combine_loss_histories(lh_adam, lh_lbfgs, adam_iters: int = 0) -> Dict[str, List]:
    """Concatenate Adam and L-BFGS loss histories.

    DeepXDE's L-BFGS LossHistory already continues step numbering from where
    Adam left off, so no step_offset is needed for the L-BFGS portion.
    If the first L-BFGS step duplicates the last Adam step, skip it.
    """
    h1 = _convert_loss_history(lh_adam, step_offset=0, phase="adam")
    h2 = _convert_loss_history(lh_lbfgs, step_offset=0, phase="lbfgs")

    # Drop duplicate overlap step (L-BFGS often re-logs the initial state)
    if h1["steps"] and h2["steps"] and h2["steps"][0] <= h1["steps"][-1]:
        for key in h2:
            h2[key] = h2[key][1:]

    combined: Dict[str, List] = {}
    for key in h1:
        combined[key] = h1[key] + h2[key]
    return combined


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_on_grid(
    model, x: np.ndarray, t: np.ndarray, dtype: str = "float64",
    decay_factor_tau: float = 0.0,
) -> np.ndarray:
    """Evaluate the model on a full space-time grid.

    If ``decay_factor_tau > 0``, the network output is treated as
    g = exp(t/tau)*Psi and converted back to Psi = exp(-t/tau)*g.

    Returns phi[t_index, x_index].
    """
    X_list = []
    for ti in t:
        X_list.append(np.stack([x, np.full_like(x, ti)], axis=1))
    X = np.concatenate(X_list, axis=0)

    if isinstance(model, dde.Model):
        y = model.predict(X)
        y = y.reshape(len(t), len(x))
    else:
        device = next(model.parameters()).device
        tdtype = torch.float64 if dtype == "float64" else torch.float32
        X_t = torch.tensor(X, dtype=tdtype, device=device)
        with torch.no_grad():
            y = model(X_t).cpu().numpy()
        y = y.reshape(len(t), len(x))

    # Convert g -> Psi = exp(-t/tau) * g
    if decay_factor_tau > 0.0:
        decay = np.exp(-t / decay_factor_tau)  # shape (Nt,)
        y = y * decay[:, None]                 # broadcast over x

    return y
