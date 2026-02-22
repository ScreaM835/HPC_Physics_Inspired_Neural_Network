import re

with open("src/pinn.py", "r") as f:
    content = f.read()

old_code = """        if model_save_path:
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
        )"""

new_code = """        model_restore_path = None
        if resume and window_ckpt_dir:
            ckpt = _find_latest_checkpoint(window_ckpt_dir)
            if ckpt is not None:
                model_restore_path = ckpt
                print(f"[CKPT] Restoring from {ckpt}")

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
            model_restore_path=model_restore_path,
        )"""

content = content.replace(old_code, new_code)

with open("src/pinn.py", "w") as f:
    f.write(content)

print("Patched resume logic.")
