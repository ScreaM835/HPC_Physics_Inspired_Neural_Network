import re

with open("src/pinn.py", "r") as f:
    content = f.read()

old_code = """        # Set up checkpointing for this window
        window_ckpt_dir = None
        model_save_path = None
        if checkpoint_dir:
            window_ckpt_dir = os.path.join(checkpoint_dir, f"window_{i+1}")
            os.makedirs(window_ckpt_dir, exist_ok=True)
            model_save_path = os.path.join(window_ckpt_dir, "model")
            
        # ---- Adam phase ----"""

new_code = """        # Set up checkpointing for this window
        window_ckpt_dir = None
        model_save_path = None
        if checkpoint_dir:
            window_ckpt_dir = os.path.join(checkpoint_dir, f"window_{i+1}")
            os.makedirs(window_ckpt_dir, exist_ok=True)
            model_save_path = os.path.join(window_ckpt_dir, "model")
            
        # Check if this window is already fully trained
        if resume and window_ckpt_dir:
            final_ckpt = os.path.join(window_ckpt_dir, "model-final.pt")
            if os.path.exists(final_ckpt):
                print(f"[CKPT] Window {i+1} already completed. Restoring and skipping.")
                model.compile("adam", lr=1e-3) # dummy compile
                model.restore(final_ckpt, verbose=1)
                continue
            
        # ---- Adam phase ----"""

content = content.replace(old_code, new_code)

with open("src/pinn.py", "w") as f:
    f.write(content)

print("Patched resume logic better.")
