import re

with open("src/pinn.py", "r") as f:
    content = f.read()

# 1. Modify build_model signature and implementation
old_build_model = """def build_model(cfg: Dict) -> Tuple[dde.Model, dde.data.TimePDE]:
    \"\"\"Construct the DeepXDE Model from the experiment config.

    Uses the Modified MLP with Trainable Random Fourier Features (RFF)
    instead of a standard FNN, to overcome spectral bias.
    \"\"\"
    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])
    tmin = float(cfg["domain"]["tmin"])
    tmax = float(cfg["domain"]["tmax"])"""

new_build_model = """def build_model(cfg: Dict, tmax_override: Optional[float] = None, net_override: Optional[torch.nn.Module] = None) -> Tuple[dde.Model, dde.data.TimePDE]:
    \"\"\"Construct the DeepXDE Model from the experiment config.

    Uses the Modified MLP with Trainable Random Fourier Features (RFF)
    instead of a standard FNN, to overcome spectral bias.
    \"\"\"
    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])
    tmin = float(cfg["domain"]["tmin"])
    tmax = tmax_override if tmax_override is not None else float(cfg["domain"]["tmax"])"""

content = content.replace(old_build_model, new_build_model)

# 2. Modify data scaling in build_model
old_data = """    Nr = int(cfg["pinn"]["Nr"])
    Ni = int(cfg["pinn"]["Ni"])
    Nb = int(cfg["pinn"]["Nb"])

    data = dde.data.TimePDE(
        geomtime,
        pde_func,
        ic_bcs,
        num_domain=Nr,
        num_boundary=Nb,
        num_initial=Ni,
        train_distribution="uniform",
    )"""

new_data = """    # Scale the number of points by the time domain fraction
    t_fraction = (tmax - tmin) / (float(cfg["domain"]["tmax"]) - tmin)
    Nr = int(int(cfg["pinn"]["Nr"]) * t_fraction)
    Ni = int(cfg["pinn"]["Ni"])
    Nb = int(int(cfg["pinn"]["Nb"]) * t_fraction)

    data = dde.data.TimePDE(
        geomtime,
        pde_func,
        ic_bcs,
        num_domain=Nr,
        num_boundary=Nb,
        num_initial=Ni,
        train_distribution="uniform",
    )"""

content = content.replace(old_data, new_data)

# 3. Modify net creation in build_model
old_net = """    # --- Modified MLP with Trainable RFF ---
    hidden = [int(w) for w in cfg["pinn"]["hidden_layers"]]
    rff_cfg = cfg["pinn"].get("rff", {})
    num_rff = int(rff_cfg.get("num_features", 64))
    rff_sigma = float(rff_cfg.get("sigma", 1.0))
    rff_trainable = bool(rff_cfg.get("trainable", True))
    activation = cfg["pinn"].get("activation", "tanh")

    net = ModifiedMLP(
        hidden_layers=hidden,
        num_rff=num_rff,
        rff_sigma=rff_sigma,
        rff_trainable=rff_trainable,
        activation=activation,
    )

    print(f"[PINN] ModifiedMLP: hidden={hidden}, "
          f"RFF(num={num_rff}, σ={rff_sigma}, trainable={rff_trainable})")
    print(f"[PINN] Trainable parameters: {net.num_trainable_parameters()}")

    # Output transform: A * tanh(y)
    # Bounding the output enforces the physical constraint of energy conservation
    # and prevents the network from adapting a blowing-up solution (Patel et al. 2024).
    A_bound = float(cfg["initial_data"]["A"])
    net.apply_output_transform(lambda x, y: A_bound * torch.tanh(y))"""

new_net = """    if net_override is not None:
        net = net_override
    else:
        # --- Modified MLP with Trainable RFF ---
        hidden = [int(w) for w in cfg["pinn"]["hidden_layers"]]
        rff_cfg = cfg["pinn"].get("rff", {})
        num_rff = int(rff_cfg.get("num_features", 64))
        rff_sigma = float(rff_cfg.get("sigma", 1.0))
        rff_trainable = bool(rff_cfg.get("trainable", True))
        activation = cfg["pinn"].get("activation", "tanh")

        net = ModifiedMLP(
            hidden_layers=hidden,
            num_rff=num_rff,
            rff_sigma=rff_sigma,
            rff_trainable=rff_trainable,
            activation=activation,
        )

        print(f"[PINN] ModifiedMLP: hidden={hidden}, "
              f"RFF(num={num_rff}, σ={rff_sigma}, trainable={rff_trainable})")
        print(f"[PINN] Trainable parameters: {net.num_trainable_parameters()}")

        # Output transform: A * tanh(y)
        # Bounding the output enforces the physical constraint of energy conservation
        # and prevents the network from adapting a blowing-up solution (Patel et al. 2024).
        A_bound = float(cfg["initial_data"]["A"])
        net.apply_output_transform(lambda x, y: A_bound * torch.tanh(y))"""

content = content.replace(old_net, new_net)

# 4. Add _train_pinn_curriculum and modify train_pinn
old_train_pinn = """def train_pinn(
    cfg: Dict,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 500,
    resume: bool = False,
) -> Tuple[dde.Model, Dict[str, List[float]]]:
    \"\"\"
    Train the PINN with DeepXDE: Adam -> L-BFGS.

    Parameters
    ----------
    cfg : dict
        Full experiment config.
    checkpoint_dir : str or None
        Directory for checkpoints (None disables).
    checkpoint_every : int
        Checkpoint interval during Adam phase.
    resume : bool
        If True, restore from the latest checkpoint.

    Returns
    -------
    model : dde.Model
    history : dict  -- per-step loss components
        Keys: L_total, Lr, Lrx, Lrt, Lic, Liv, Lbl, Lbr, w_min
    \"\"\"
    seed = int(cfg["pinn"]["seed"])
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(cfg["pinn"]["dtype"])

    model, data = build_model(cfg)"""

new_train_pinn = """def _train_pinn_curriculum(
    cfg: Dict,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 500,
    resume: bool = False,
) -> Tuple[dde.Model, Dict[str, List[float]]]:
    \"\"\"
    Train the PINN using Curriculum Learning (Expanding Time Windows).
    This is mathematically equivalent to Time-Marching but avoids error accumulation
    at window boundaries and bypasses PyTorch derivative extraction bugs.
    \"\"\"
    seed = int(cfg["pinn"]["seed"])
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(cfg["pinn"]["dtype"])

    loss_weights = [float(w) for w in cfg["pinn"]["lambda"]]
    
    curriculum_cfg = cfg["pinn"]["curriculum"]
    windows = curriculum_cfg["windows"]  # e.g., [10.0, 20.0, 30.0, 40.0, 50.0]
    
    net = None
    history_all = None
    
    for i, tmax in enumerate(windows):
        print(f"\\n{'='*50}")
        print(f"[PINN] Curriculum Window {i+1}/{len(windows)}: t in [0, {tmax}]")
        print(f"{'='*50}\\n")
        
        # Build model for this window
        model, data = build_model(cfg, tmax_override=tmax, net_override=net)
        net = model.net  # Keep the network for the next window
        
        # Set up checkpointing for this window
        window_ckpt_dir = None
        model_save_path = None
        if checkpoint_dir:
            window_ckpt_dir = os.path.join(checkpoint_dir, f"window_{i+1}")
            os.makedirs(window_ckpt_dir, exist_ok=True)
            model_save_path = os.path.join(window_ckpt_dir, "model")
            
        # ---- Adam phase ----
        adam_cfg = cfg["pinn"]["adam"]
        adam_iters = int(adam_cfg["iters"])
        lr = float(adam_cfg["lr"])
        resample_period = int(adam_cfg["resample_period"])

        callbacks_adam: List = []
        callbacks_adam.append(
            dde.callbacks.PDEPointResampler(period=resample_period)
        )

        # Gradient balancing (Wang et al. 2021)
        grad_bal_cfg = cfg["pinn"].get("gradient_balancing", {})
        if grad_bal_cfg.get("enabled", False):
            gb_period = int(grad_bal_cfg.get("period", 100))
            gb_alpha = float(grad_bal_cfg.get("alpha", 0.9))
            callbacks_adam.append(
                GradientBalancing(period=gb_period, alpha=gb_alpha)
            )

        if model_save_path:
            callbacks_adam.append(
                dde.callbacks.ModelCheckpoint(
                    model_save_path,
                    save_better_only=False,
                    period=checkpoint_every,
                )
            )

        model.compile("adam", lr=lr, loss_weights=loss_weights)

        print(f"[PINN] Adam: {adam_iters} iters, lr={lr}, resample every {resample_period}")
        losshistory_adam, _ = model.train(
            iterations=adam_iters,
            callbacks=callbacks_adam,
            display_every=100,
            model_save_path=model_save_path,
        )

        if model_save_path:
            model.save(model_save_path + "-adam_done")
            import json
            weights_file = os.path.join(window_ckpt_dir, "loss_weights_adam.json")
            with open(weights_file, "w") as f:
                json.dump(list(model.loss_weights), f)

        # ---- L-BFGS phase ----
        lbfgs_cfg = cfg["pinn"]["lbfgs"]
        lbfgs_iters = int(lbfgs_cfg["iters"])

        lbfgs_loss_weights = list(model.loss_weights)
        
        dde.optimizers.set_LBFGS_options(
            maxcor=100,
            maxiter=lbfgs_iters,
            ftol=0,
            gtol=1e-8,
            maxls=50,
        )
        
        lbfgs_resample_period = int(lbfgs_cfg.get("resample_period", 0))
        if lbfgs_resample_period > 0:
            step_size = min(checkpoint_every, lbfgs_resample_period, lbfgs_iters)
        else:
            step_size = min(checkpoint_every, lbfgs_iters)

        from deepxde.optimizers.config import LBFGS_options as _lbfgs_opts
        _lbfgs_opts["iter_per_step"] = step_size
        _lbfgs_opts["fun_per_step"] = int(_lbfgs_opts["iter_per_step"] * 1.25)

        model.compile("L-BFGS", loss_weights=lbfgs_loss_weights)

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
            iterations=lbfgs_iters,
            callbacks=callbacks_lbfgs,
            display_every=100
        )

        if model_save_path:
            model.save(model_save_path + "-final")

        # Combine history
        history = _combine_loss_histories(losshistory_adam, losshistory_lbfgs)
        if history_all is None:
            history_all = history
        else:
            for key in history_all:
                history_all[key].extend(history[key])

    return model, history_all


def train_pinn(
    cfg: Dict,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 500,
    resume: bool = False,
) -> Tuple[dde.Model, Dict[str, List[float]]]:
    \"\"\"
    Train the PINN with DeepXDE: Adam -> L-BFGS.

    Parameters
    ----------
    cfg : dict
        Full experiment config.
    checkpoint_dir : str or None
        Directory for checkpoints (None disables).
    checkpoint_every : int
        Checkpoint interval during Adam phase.
    resume : bool
        If True, restore from the latest checkpoint.

    Returns
    -------
    model : dde.Model
    history : dict  -- per-step loss components
        Keys: L_total, Lr, Lrx, Lrt, Lic, Liv, Lbl, Lbr, w_min
    \"\"\"
    # Check if curriculum learning is enabled
    curriculum_cfg = cfg["pinn"].get("curriculum", {})
    if curriculum_cfg.get("enabled", False):
        return _train_pinn_curriculum(cfg, checkpoint_dir, checkpoint_every, resume)

    seed = int(cfg["pinn"]["seed"])
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(cfg["pinn"]["dtype"])

    model, data = build_model(cfg)"""

content = content.replace(old_train_pinn, new_train_pinn)

with open("src/pinn.py", "w") as f:
    f.write(content)

print("Patched src/pinn.py successfully.")
