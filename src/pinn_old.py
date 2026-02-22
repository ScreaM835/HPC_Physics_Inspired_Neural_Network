from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import trange

from .potentials import V_and_dVdx_of_x
from .initial_data import gaussian_phi, gaussian_phi_t


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    phase: str,
    iteration: int,
    history: Dict[str, List[float]],
    rng_state: dict,
) -> None:
    """Persist training state so that it can be resumed later."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "phase": phase,          # "adam" or "lbfgs"
            "iteration": iteration,  # completed iterations in current phase
            "history": history,
            "rng_state": rng_state,
            "torch_rng_state": torch.get_rng_state(),
        },
        path,
    )


def _load_checkpoint(path: str, model: nn.Module, device: torch.device):
    """Load a checkpoint. Returns the dict; caller wires up optimizers."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float64":
        return torch.float64
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


class FCN(nn.Module):
    def __init__(self, layers: List[int], activation: str = "tanh", A_bound: float = 1.0, output_transform: str = "tanh_bound"):
        super().__init__()
        act = nn.Tanh() if activation == "tanh" else None
        if act is None:
            raise ValueError("Only tanh activation is implemented (to match the target setup).")

        mods: List[nn.Module] = []
        for i in range(len(layers) - 2):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            mods.append(act)
        mods.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*mods)
        self.A_bound = float(A_bound)
        self.output_transform = output_transform

        # Xavier/Glorot uniform init as in the target paper
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y = self.net(X)
        if self.output_transform == "tanh_bound":
            return self.A_bound * torch.tanh(y)
        return y


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def _sample_uniform(n: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(lo, hi, size=(n,))


def _make_training_points(cfg: Dict, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    xmin, xmax = float(cfg["domain"]["xmin"]), float(cfg["domain"]["xmax"])
    tmin, tmax = float(cfg["domain"]["tmin"]), float(cfg["domain"]["tmax"])

    Nr = int(cfg["pinn"]["Nr"])
    Ni = int(cfg["pinn"]["Ni"])
    Nb = int(cfg["pinn"]["Nb"])

    # interior residual points
    xr = _sample_uniform(Nr, xmin, xmax, rng)
    tr = _sample_uniform(Nr, tmin, tmax, rng)

    # initial points (t=tmin)
    xi = _sample_uniform(Ni, xmin, xmax, rng)
    ti = np.full_like(xi, tmin)

    # boundary points
    tb = _sample_uniform(Nb, tmin, tmax, rng)
    xl = np.full_like(tb, xmin)
    xr_b = np.full_like(tb, xmax)

    return {
        "Xr": np.stack([xr, tr], axis=1),
        "Xi": np.stack([xi, ti], axis=1),
        "Xbl": np.stack([xl, tb], axis=1),
        "Xbr": np.stack([xr_b, tb], axis=1),
    }


def _with_requires_grad(X: torch.Tensor) -> torch.Tensor:
    X = X.clone().detach()
    X.requires_grad_(True)
    return X


def compute_losses(model: nn.Module, cfg: Dict, pts: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    M = float(cfg["physics"]["M"])
    l = int(cfg["physics"]["l"])
    potential = cfg["physics"]["potential"]

    # unpack points (base tensors live on the correct device)
    Xr_base = pts["Xr"]
    Xi_base = pts["Xi"]
    Xbl_base = pts["Xbl"]
    Xbr_base = pts["Xbr"]

    # ----- PDE residual + gradient-enhanced terms (optionally chunked) -----
    chunk_size = int(cfg["pinn"].get("chunk_size", 0) or 0)
    if chunk_size <= 0:
        chunk_size = Xr_base.shape[0]

    Nr = int(Xr_base.shape[0])

    # Collect per-point squared residuals for causal weighting
    all_r2: List[torch.Tensor] = []
    all_rx2: List[torch.Tensor] = []
    all_rt2: List[torch.Tensor] = []

    for s in range(0, Nr, chunk_size):
        e = min(Nr, s + chunk_size)
        Xr = _with_requires_grad(Xr_base[s:e])

        xr_np = Xr[:, 0].detach().cpu().numpy()
        Vx_np, dVdx_np = V_and_dVdx_of_x(xr_np, M=M, l=l, potential=potential)
        Vx = torch.tensor(Vx_np, dtype=Xr.dtype, device=Xr.device).view(-1, 1)
        dVdx = torch.tensor(dVdx_np, dtype=Xr.dtype, device=Xr.device).view(-1, 1)

        phi = model(Xr)
        g_phi = _grad(phi, Xr)
        phi_x = g_phi[:, 0:1]
        phi_t = g_phi[:, 1:2]

        phi_xx = _grad(phi_x, Xr)[:, 0:1]
        phi_tt = _grad(phi_t, Xr)[:, 1:2]

        # standard stable convention: phi_tt - phi_xx + V phi = 0
        r = phi_tt - phi_xx + Vx * phi

        dr = _grad(r, Xr)
        r_x = dr[:, 0:1] + dVdx * phi  # correct for V(x) dependence
        r_t = dr[:, 1:2]               # V independent of t

        all_r2.append((r**2).squeeze(1))
        all_rx2.append((r_x**2).squeeze(1))
        all_rt2.append((r_t**2).squeeze(1))

    all_r2_cat = torch.cat(all_r2, dim=0)    # shape (Nr,)
    all_rx2_cat = torch.cat(all_rx2, dim=0)
    all_rt2_cat = torch.cat(all_rt2, dim=0)

    # ----- Causal weighting (Wang et al., 2022) -----
    causal_cfg = cfg["pinn"].get("causal", None)
    causal_enabled = causal_cfg is not None and causal_cfg.get("enabled", False)
    w_min_val = 1.0  # diagnostic: minimum causal weight

    if causal_enabled:
        epsilon = float(causal_cfg["epsilon"])
        n_slices = int(causal_cfg.get("n_slices", 20))
        tmin_val = float(cfg["domain"]["tmin"])
        tmax_val = float(cfg["domain"]["tmax"])

        time_vals = Xr_base[:, 1]  # shape (Nr,)
        slice_bounds = torch.linspace(
            tmin_val, tmax_val, n_slices + 1,
            dtype=time_vals.dtype, device=time_vals.device,
        )

        cumulative_loss = torch.tensor(0.0, dtype=all_r2_cat.dtype, device=all_r2_cat.device)
        Lr = torch.tensor(0.0, dtype=all_r2_cat.dtype, device=all_r2_cat.device)
        Lrx = torch.tensor(0.0, dtype=all_r2_cat.dtype, device=all_r2_cat.device)
        Lrt = torch.tensor(0.0, dtype=all_r2_cat.dtype, device=all_r2_cat.device)

        for k in range(n_slices):
            if k < n_slices - 1:
                mask = (time_vals >= slice_bounds[k]) & (time_vals < slice_bounds[k + 1])
            else:
                # last slice includes right endpoint
                mask = (time_vals >= slice_bounds[k]) & (time_vals <= slice_bounds[k + 1])

            n_in_slice = mask.sum().item()
            if n_in_slice == 0:
                continue

            w_k = torch.exp(-epsilon * cumulative_loss)

            slice_r = torch.mean(all_r2_cat[mask])
            slice_rx = torch.mean(all_rx2_cat[mask])
            slice_rt = torch.mean(all_rt2_cat[mask])

            Lr = Lr + w_k * slice_r
            Lrx = Lrx + w_k * slice_rx
            Lrt = Lrt + w_k * slice_rt

            # Accumulate for next slice (detach so gradients don't flow through weights)
            cumulative_loss = cumulative_loss + slice_r.detach()

            w_min_val = min(w_min_val, float(w_k.detach().cpu().item()))

        # Normalize by n_slices so magnitude is comparable to non-causal loss
        Lr = Lr / n_slices
        Lrx = Lrx / n_slices
        Lrt = Lrt / n_slices
    else:
        Lr = torch.mean(all_r2_cat)
        Lrx = torch.mean(all_rx2_cat)
        Lrt = torch.mean(all_rt2_cat)

    # ----- Initial conditions -----
    Xi = _with_requires_grad(Xi_base)

    A0 = float(cfg["initial_data"]["A"])
    x0 = float(cfg["initial_data"]["x0"])
    sigma = float(cfg["initial_data"]["sigma"])
    profile = cfg["initial_data"]["velocity_profile"]

    xi_np = Xi[:, 0].detach().cpu().numpy()
    phi0_np = gaussian_phi(xi_np, A=A0, x0=x0, sigma=sigma)
    v0_np = gaussian_phi_t(xi_np, A=A0, x0=x0, sigma=sigma, profile=profile)
    phi0 = torch.tensor(phi0_np, dtype=Xi.dtype, device=Xi.device).view(-1, 1)
    v0 = torch.tensor(v0_np, dtype=Xi.dtype, device=Xi.device).view(-1, 1)

    phi_i = model(Xi)
    phi_i_t = _grad(phi_i, Xi)[:, 1:2]

    Lic = torch.mean((phi_i - phi0) ** 2)
    Liv = torch.mean((phi_i_t - v0) ** 2)

    # ----- Radiative boundary conditions -----
    Xbl = _with_requires_grad(Xbl_base)
    Xbr = _with_requires_grad(Xbr_base)

    # left: (∂t - ∂x)phi = 0
    phi_l = model(Xbl)
    grad_l = _grad(phi_l, Xbl)
    bc_l = grad_l[:, 1:2] - grad_l[:, 0:1]
    Lbl = torch.mean(bc_l**2)

    # right: (∂t + ∂x)phi = 0
    phi_r = model(Xbr)
    grad_r = _grad(phi_r, Xbr)
    bc_r = grad_r[:, 1:2] + grad_r[:, 0:1]
    Lbr = torch.mean(bc_r**2)

    loss_vec = torch.stack([Lr, Lrx, Lrt, Lic, Liv, Lbl, Lbr], dim=0)

    lam = torch.tensor(cfg["pinn"]["lambda"], dtype=loss_vec.dtype, device=loss_vec.device)
    total = torch.sum(lam * loss_vec)

    history = {
        "L_total": float(total.detach().cpu().item()),
        "Lr": float(Lr.detach().cpu().item()),
        "Lrx": float(Lrx.detach().cpu().item()),
        "Lrt": float(Lrt.detach().cpu().item()),
        "Lic": float(Lic.detach().cpu().item()),
        "Liv": float(Liv.detach().cpu().item()),
        "Lbl": float(Lbl.detach().cpu().item()),
        "Lbr": float(Lbr.detach().cpu().item()),
        "w_min": w_min_val,
    }
    return total, history


def train_pinn(
    cfg: Dict,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 500,
    resume: bool = False,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the PINN with optional checkpointing and resume support.

    Parameters
    ----------
    cfg : dict
        Full experiment configuration.
    checkpoint_dir : str or None
        Directory to save periodic checkpoints.  ``None`` disables checkpointing.
    checkpoint_every : int
        Save a checkpoint every this many iterations (applies to each phase).
    resume : bool
        If True and a checkpoint exists in *checkpoint_dir*, resume training
        from that checkpoint.
    """
    seed = int(cfg["pinn"]["seed"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    dtype = _torch_dtype(cfg["pinn"]["dtype"])
    torch.set_default_dtype(dtype)

    A_bound = float(cfg["initial_data"]["A"])

    layers = [2] + [int(w) for w in cfg["pinn"]["hidden_layers"]] + [1]
    model = FCN(
        layers=layers,
        activation=str(cfg["pinn"]["activation"]),
        A_bound=A_bound,
        output_transform=str(cfg["pinn"]["output_transform"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=dtype)

    # Checkpoint paths
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt") if checkpoint_dir else None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Determine resume state
    resume_phase = None   # "adam" or "lbfgs"
    resume_iter = 0
    ckpt = None
    if resume and ckpt_path and os.path.isfile(ckpt_path):
        print(f"[CKPT] Resuming from {ckpt_path}")
        ckpt = _load_checkpoint(ckpt_path, model, device)
        resume_phase = ckpt["phase"]
        resume_iter = ckpt["iteration"]
        # Restore RNG states for reproducibility
        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            rng = np.random.default_rng()
            rng.__setstate__(rng_state)
        torch_rng = ckpt.get("torch_rng_state")
        if torch_rng is not None:
            torch.set_rng_state(torch_rng)

    # initial training points
    pts_np = _make_training_points(cfg, rng)
    pts = {k: torch.tensor(v, dtype=dtype, device=device) for k, v in pts_np.items()}

    # Restore or initialize history
    if ckpt is not None and "history" in ckpt:
        history: Dict[str, List[float]] = ckpt["history"]
    else:
        history = {k: [] for k in ["L_total", "Lr", "Lrx", "Lrt", "Lic", "Liv", "Lbl", "Lbr", "w_min"]}

    skip_adam = (resume_phase == "lbfgs")  # Adam already completed

    # ----- Adam -----
    adam_cfg = cfg["pinn"]["adam"]
    adam_iters = int(adam_cfg["iters"])
    resample_period = int(adam_cfg["resample_period"])

    if not skip_adam:
        adam = torch.optim.Adam(model.parameters(), lr=float(adam_cfg["lr"]))

        # Restore optimizer state if resuming within Adam phase
        adam_start = 0
        if ckpt is not None and resume_phase == "adam":
            adam.load_state_dict(ckpt["optimizer_state_dict"])
            adam_start = resume_iter
            print(f"[CKPT] Resuming Adam from iteration {adam_start}/{adam_iters}")

        for it in trange(adam_start, adam_iters, desc="Adam", initial=adam_start, total=adam_iters):
            if it > 0 and (it % resample_period == 0):
                pts_np = _make_training_points(cfg, rng)
                pts = {k: torch.tensor(v, dtype=dtype, device=device) for k, v in pts_np.items()}

            adam.zero_grad(set_to_none=True)
            loss, h = compute_losses(model, cfg, pts)
            loss.backward()
            adam.step()

            for k in history:
                history[k].append(h[k])

            # Periodic checkpoint
            if ckpt_path and (it + 1) % checkpoint_every == 0:
                _save_checkpoint(ckpt_path, model, adam, "adam", it + 1, history, rng.__getstate__())

        # Save end-of-Adam checkpoint
        if ckpt_path:
            _save_checkpoint(ckpt_path, model, adam, "adam", adam_iters, history, rng.__getstate__())
            print(f"[CKPT] Adam complete — checkpoint saved")
    else:
        print(f"[CKPT] Skipping Adam (already completed in previous run)")

    # ----- LBFGS -----
    lbfgs_cfg = cfg["pinn"]["lbfgs"]
    lbfgs_iters = int(lbfgs_cfg["iters"])
    lbfgs_resample_period = int(lbfgs_cfg["resample_period"])

    # We run LBFGS with max_iter=1 in a loop, enabling periodic resampling between steps.
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    # Restore LBFGS optimizer state if resuming within LBFGS phase
    lbfgs_start = 0
    if ckpt is not None and resume_phase == "lbfgs":
        lbfgs.load_state_dict(ckpt["optimizer_state_dict"])
        lbfgs_start = resume_iter
        print(f"[CKPT] Resuming LBFGS from iteration {lbfgs_start}/{lbfgs_iters}")

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        loss, _ = compute_losses(model, cfg, pts)
        loss.backward()
        return loss

    for it in trange(lbfgs_start, lbfgs_iters, desc="LBFGS", initial=lbfgs_start, total=lbfgs_iters):
        if it > 0 and (it % lbfgs_resample_period == 0):
            pts_np = _make_training_points(cfg, rng)
            pts = {k: torch.tensor(v, dtype=dtype, device=device) for k, v in pts_np.items()}

        loss = lbfgs.step(closure)

        # record current losses
        _, h = compute_losses(model, cfg, pts)
        for k in history:
            history[k].append(h[k])

        # Periodic checkpoint
        if ckpt_path and (it + 1) % checkpoint_every == 0:
            _save_checkpoint(ckpt_path, model, lbfgs, "lbfgs", it + 1, history, rng.__getstate__())

    # Final checkpoint
    if ckpt_path:
        _save_checkpoint(ckpt_path, model, lbfgs, "lbfgs", lbfgs_iters, history, rng.__getstate__())
        print(f"[CKPT] Training complete — final checkpoint saved")

    return model, history


@torch.no_grad()
def eval_on_grid(model: nn.Module, x: np.ndarray, t: np.ndarray, dtype: str = "float64") -> np.ndarray:
    """
    Evaluate the trained model on a full space-time grid (t,x), returning phi[t_index, x_index].
    """
    model.eval()
    device = next(model.parameters()).device
    tdtype = _torch_dtype(dtype)

    X_list = []
    for ti in t:
        X_list.append(np.stack([x, np.full_like(x, ti)], axis=1))
    X = np.concatenate(X_list, axis=0)
    X_t = torch.tensor(X, dtype=tdtype, device=device)
    y = model(X_t).detach().cpu().numpy().reshape(len(t), len(x))
    return y
