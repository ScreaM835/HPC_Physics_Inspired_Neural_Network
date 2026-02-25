# Session Summary — 25 Feb 2026

## GitHub Repo
- **URL**: https://github.com/ScreaM835/HPC_Physics_Inspired_Neural_Network
- **`main` branch**: improved repo (`project32_qnm_pinn_improved`) — all code, configs, outputs, presentation
- **`base-repo` branch**: reproduction repo (`project32_qnm_pinn`) — original paper reproduction with outputs

---

## What Was Done This Session

### 1. Created Analysis Scripts
- **`scripts/extract_all_qnms.py`** — extracts QNM values (ω, τ) from all PINN and FD runs
- **`scripts/plot_curve_fitting.py`** — generates paper-style curve fitting plots for all runs

### 2. Fixed Loss History Bug (`src/pinn.py`)
- **Bug**: `_convert_loss_history` was dividing loss components by their weights, deflating values by ~100x
- **Fix**: removed the division — DeepXDE already stores unweighted MSEs
- **Also fixed**: `_combine_loss_histories` step offsetting for L-BFGS phase

### 3. Corrected Existing `loss_history.json` Files
- All 6 runs in `outputs/pinn/*/loss_history.json` were rewritten with corrected values
- Original files backed up as `.bak`
- L-BFGS steps reset to start at 10000 (matching old repo's visual overlap trick)

### 4. Presentation Slides
- **`presentation/slides.tex`** — full Beamer presentation with all results
- Added 2 appendix slides with loss plots from the base reproduction repo
- Loss plot images: `presentation/loss_reproduction.png`, `presentation/loss_rad.png`
- Updated `RAD k=2, P=500` final loss from 9.84e-7 to 1.06e-6 in slides

### 5. Pushed to GitHub
- Improved repo → `main` branch (with all outputs)
- Base repo → `base-repo` branch (with all outputs)

---

## Workspace Layout
```
project32_qnm_pinn/          ← base reproduction repo (pushed to base-repo branch)
project32_qnm_pinn_improved/ ← improved repo (pushed to main branch)
```

## Key Files Modified
| File | Change |
|------|--------|
| `src/pinn.py` | Fixed loss weight division bug |
| `src/plotting.py` | Loss plot cleanup |
| `presentation/slides.tex` | Added appendix loss plots, fixed loss value |
| `.gitignore` | Removed `outputs/` exclusion, added LaTeX artifacts |
| `outputs/pinn/*/loss_history.json` | Corrected loss values |

## Runs & Configs
| Run | Config |
|-----|--------|
| Uniform (reproduction) | `zerilli_l2_paper.yaml` (base repo) |
| RAD k=1 | `zerilli_l2_rad.yaml` |
| RAD k=2 | `zerilli_l2_rad_k2.yaml` |
| RAD k=2, P=500 | `zerilli_l2_rad_k2_p500.yaml` |
| RAD + Anchor | `zerilli_l2_rad_anchor.yaml` |
| Exp-weight | `zerilli_l2_expweight.yaml` |
| RAD + Exp-weight | `zerilli_l2_rad_expweight.yaml` |

## Known Issues / TODO
- Loss plots in `outputs/pinn/*/loss.png` for the improved runs may still look garbled (the L-BFGS transition spike issue was addressed by rewriting step arrays, but visual quality wasn't fully verified)
- The `loss_history.json.bak` files are the originals before correction — can be deleted once verified
