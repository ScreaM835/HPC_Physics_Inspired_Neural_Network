import re

with open("src/pinn.py", "r") as f:
    content = f.read()

old_code = """        # Check if this window is already fully trained
        if resume and window_ckpt_dir:
            final_ckpt = os.path.join(window_ckpt_dir, "model-final.pt")
            if os.path.exists(final_ckpt):
                print(f"[CKPT] Window {i+1} already completed. Restoring and skipping.")
                model.compile("adam", lr=1e-3) # dummy compile
                model.restore(final_ckpt, verbose=1)
                continue"""

new_code = """        # Check if this window is already fully trained
        if resume and window_ckpt_dir:
            final_ckpt = os.path.join(window_ckpt_dir, "model-final.pt")
            if os.path.exists(final_ckpt):
                print(f"[CKPT] Window {i+1} already completed. Restoring and skipping.")
                model.compile("adam", lr=1e-3) # dummy compile
                model.train(iterations=0, display_every=1) # init train state
                model.restore(final_ckpt, verbose=1)
                continue"""

content = content.replace(old_code, new_code)

with open("src/pinn.py", "w") as f:
    f.write(content)

print("Patched resume logic fix.")
