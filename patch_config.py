import yaml

with open("configs/zerilli_l2_fd_refined.yaml", "r") as f:
    cfg = yaml.safe_load(f)

cfg["pinn"]["curriculum"] = {
    "enabled": True,
    "windows": [10.0, 20.0, 30.0, 40.0, 50.0]
}

# Reduce iterations per window since we are doing 5 windows
cfg["pinn"]["adam"]["iters"] = 2000
cfg["pinn"]["lbfgs"]["iters"] = 3000

with open("configs/zerilli_l2_fd_refined.yaml", "w") as f:
    yaml.dump(cfg, f, sort_keys=False)

print("Patched configs/zerilli_l2_fd_refined.yaml successfully.")
