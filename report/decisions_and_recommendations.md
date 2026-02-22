# Decisions and Recommendations: PINN Implementation Changes

This document records all decisions made during the systematic comparison of our PINN implementation against Patel et al. (2024) [arXiv:2401.01440].

---

## Background: Full Issue Audit

A thorough comparison of our pure-PyTorch implementation against the paper's DeepXDE-based implementation identified **31 potential issues/differences**. These were categorised into:

- **14 issues** that would be eliminated by switching to DeepXDE (framework, autograd graph, optimizer defaults, sampling, etc.)
- **17 issues** that remain regardless of framework choice (physics choices, unknown paper details, our enhancements, post-processing)

The 14 framework-dependent issues are: #1 (framework mismatch), #2 (V(x) outside autograd), #3 (dV/dx via finite differences), #5 (history_size 100 vs 50), #6 (L-BFGS history not reset after resampling), #7 (strong_wolfe vs L-BFGS-B line search), #8 (double forward pass in L-BFGS), #9 (float64 vs float32), #14 (chunked residual), #16 (framework numerical differences), #19 (dV/dx correction logic), #25 (no input normalization), #26 (uniform random sampling only), #27 (no residual-adaptive refinement).

The 17 framework-independent issues are: #4 (velocity profile), #10 (unknown loss weights), #11 (causal training not in paper), #12 (causal normalization), #13 (BCs not causally weighted), #15 (resampling during L-BFGS), #17 (output transform saturation), #18 (no Adam LR scheduler), #20 (QNM extraction window), #21 (FD solver resolution), #22 (FD boundary implementation), #23 (soft BC enforcement with low weights), #24 (seed/reproducibility), #28 (evaluation grid), #29 (7 loss terms — combined or separate?), #30 (detached history — confirmed fine), #31 (checkpoint format).

---

## Decisions Made

### Decision 1: Loss Weights (#10 & #29)

**Issue:** Our implementation uses fixed weights λ = [100, 100, 100, 1, 100, 1, 1] for [L_r, L_rx, L_rt, L_ic, L_iv, L_bl, L_br]. The paper does not publish their weight values. It is unknown whether the paper treats L_r, L_rx, L_rt as independent terms with separate weights or combines them into a single PDE residual loss.

**Decision: Two-phase approach.**

**Phase 1 — Faithful baseline:** Set all λ_i = 1 (equal weights). Justification:
- Equal weights are the default in Raissi, Perdikaris & Karniadakis (2019), J. Comp. Phys. 378, 686–707 [arXiv:1711.10561], ~9,000 citations — the foundational PINN paper.
- Equal weights are DeepXDE's default. Since Patel et al. use DeepXDE and do not mention custom weights, they almost certainly used λ = 1 for all terms.
- Equal weights are the scientifically neutral starting point when the paper's weights are unknown.
- Our current weights [100, 100, 100, 1, 100, 1, 1] are hand-tuned with no published justification and create a specific (possibly wrong) bias: residual terms are 100× more influential than BCs, and displacement IC (λ_ic = 1) is 100× less than velocity IC (λ_iv = 100).

**Phase 2 — Principled improvement:** Implement Wang, Teng & Perdikaris (2021), SIAM Review 63(1), 208–228 [arXiv:2001.04536], ~2,000 citations — Algorithm 1 (learning rate annealing / gradient balancing):

$$\hat{\lambda}_i = \frac{\max_\theta |\nabla_\theta \mathcal{L}_{\text{PDE}}|}{\overline{|\nabla_\theta \mathcal{L}_i|}}, \qquad \lambda_i^{(k)} = (1-\alpha)\lambda_i^{(k-1)} + \alpha\hat{\lambda}_i$$

This dynamically balances gradient magnitudes using an exponential moving average (α = 0.1 typical). Justification:
- Eliminates the need to guess weights entirely.
- Supported by rigorous NTK theory (companion paper arXiv:2007.14527).
- Has reproducible reference code on GitHub.
- Most cited (by far) adaptive weighting scheme in the PINN literature.
- A hyperparameter sweep over 7 weights (even 3 values each) gives 3^7 = 2,187 combinations at ~12h per run — completely impractical. Gradient balancing achieves the same goal adaptively in a single run.

**Status:** Agreed. User confirmed: "i think we should go for this."

---

### Decision 2: Output Transform Saturation (#17)

**Issue:** The network applies φ = A_bound · tanh(y) with A_bound = A = 1.0 (read from the initial data amplitude in the config). The Gaussian peak Φ(x₀, 0) = 1.0 requires tanh(y) = 1, i.e. y → ∞. At this point, the gradient d(tanh)/dy = 1 − tanh²(y) → 0. The network cannot represent the exact peak value, and gradients vanish near it.

**Analysis of alternatives discussed:**
- **Option A (increase A_bound to 2.0 or 3.0):** Fixes saturation; preserves the bounding prior. But the specific value (2.0, 3.0, 5.0) is arbitrary — no principled way to choose it.
- **Option B (remove output transform):** Matches dominant practice in the field (Raissi et al. 2019, Wang et al. 2021, Wang et al. 2022 — none use output bounding). The solution's boundedness is already guaranteed by the PDE's energy conservation. Scientifically the cleanest approach.
- **Option C (hard IC enforcement):** Most sophisticated but changes the problem structure significantly.

**Why the paper uses tanh bounding:** The Zerilli equation conserves energy, so |φ| is physically bounded. The tanh encodes this prior. It also guards against catastrophic L-BFGS steps producing extreme outputs. DeepXDE documentation encourages output transforms.

**Decision: Follow the paper for now.** Keep the tanh output transform with A_bound = A = 1.0 to match the paper's architecture exactly. This means accepting the saturation issue at the Gaussian peak as a known limitation of the paper's approach.

**Status:** Decided. User said: "lets try to follow the paper for now."

---

### Decision 3: Causal Training (#11, #12, #13)

**Issue #11:** Causal training (Wang et al. 2022, arXiv:2203.07404) with ε = 50 and 20 temporal slices was added to our implementation. The paper does not use causal training. This changes the entire training dynamics compared to the paper.

**Issue #12:** The causal residual losses are divided by n_slices (line 225–227 of pinn.py). Concern was that this distorts the effective loss magnitude, causing IC/BC terms to dominate the residual.

**Issue #13:** Causal weighting applies only to L_r, L_rx, L_rt, not to L_bl, L_br. Concern was that this creates an imbalance.

**Reassessment:**
- **#11:** Wang et al. 2022 recommend ε ∈ [10, 100] for wave equations. ε = 50 is within their recommended range. The Zerilli equation is a hyperbolic PDE — exactly the class of problems causal training was designed for. The "suppression" of late-time residuals is the intended curriculum: the network learns early times first, then progressively extends to later times as early-time residuals decrease and causal weights grow toward 1.
- **#12:** The 1/n_slices normalization makes the causal and non-causal losses exactly equal when all w_k = 1 (i.e., when training has converged). The reduced magnitude during early training is the intended behavior, not a bug.
- **#13:** Wang et al. 2022's original implementation also does not causally weight the BCs — only the PDE residual. Our implementation matches their specification.

**Decision: Keep causal training.** All three issues are working-as-designed. Causal training is a well-motivated, literature-backed improvement (Wang et al. 2022, ~800 citations) over the paper's standard PINN approach, specifically designed for hyperbolic PDEs like the Zerilli equation.

**Status:** Decided. Reassessment presented; user did not object.

---

### Decision 4: Initial Velocity Profile (#4)

**Issue:** Our config uses `velocity_profile: outgoing`, giving:

$$\Phi_t(x, 0) = \frac{2(x - x_0)}{\sigma^2}\,\Phi(x, 0)$$

The paper's Eq. 23 specifies:

$$\Phi_t(x, 0) = \frac{2(x - x_0)^2}{\sigma^2}\,\Phi(x, 0)$$

The difference is (x − x₀) (linear, antisymmetric) vs (x − x₀)² (quadratic, symmetric). Our "outgoing" profile is physically correct for a right-moving pulse (Φ_t = −Φ_x). The paper's profile has a quadratic factor that is non-standard — no other paper in the black hole perturbation literature uses it (see §6 of literature_review.md for full survey). It is likely a typographical error in the paper.

**Key points:**
- QNM frequencies are eigenvalues of the system and are independent of initial data.
- Our PINN vs FD comparison is internally self-consistent (both use the same velocity profile from the config).
- The transient solution and signal shape differ between profiles, which affects the QNM extraction window and signal quality.

**Recommendation discussed:** Run both profiles:
1. `velocity_profile: paper` as the baseline reproduction (matches paper's Eq. 23 exactly).
2. `velocity_profile: outgoing` as a controlled variant (physically motivated, demonstrates robustness of QNM extraction to initial data choice).

This would validate against the paper and also produce a useful result about initial-data independence of QNM extraction.

**Status: Recommended, not yet explicitly confirmed by user.**

---

## Issues Not Yet Discussed

The following issues from the original 31-point audit have not yet been individually reviewed and decided:

### Framework-dependent issues (would be fixed by switching to DeepXDE)

| # | Issue | Brief description |
|---|-------|-------------------|
| 1 | Framework mismatch | Pure PyTorch vs DeepXDE |
| 2 | V(x) outside autograd | Potential computed in NumPy, not in the computational graph |
| 3 | dV/dx via finite differences | Centered FD (h=1e-4) instead of exact autodiff |
| 5 | L-BFGS history_size 100 vs 50 | PyTorch default vs SciPy/DeepXDE default |
| 6 | L-BFGS history not reset after resampling | Stale Hessian info persists for ~100 iterations |
| 7 | strong_wolfe vs L-BFGS-B | Different line search algorithms |
| 8 | Double forward pass in L-BFGS | Extra compute_losses call for history recording |
| 9 | float64 vs float32 | Different precision from paper's likely default |
| 14 | Chunked residual computation | chunk_size=4096 vs full-batch |
| 16 | Framework numerical differences | Different RNG, different floating-point paths |
| 19 | dV/dx correction logic | Manual correction needed because V(x) outside graph |
| 25 | No input normalization | Raw coordinates fed to network |
| 26 | Uniform random sampling only | No LHS, Halton, Sobol, etc. |
| 27 | No RAR | No residual-adaptive refinement |

### Framework-independent issues (remain regardless)

| # | Issue | Brief description |
|---|-------|-------------------|
| 15 | Resampling during L-BFGS | May disrupt quasi-Newton convergence |
| 18 | No Adam LR scheduler | Fixed lr=1e-3 throughout Adam phase |
| 20 | QNM extraction window [10, 50] | May start too early (prompt response not decayed) |
| 21 | FD solver resolution | dx=0.2, dt=0.1 — relatively coarse ground truth |
| 22 | FD boundary condition implementation | One-sided FD stencils may cause reflections |
| 23 | Soft BC enforcement with low weights | λ_bl = λ_br = 1 (lowest weights) — partially addressed by Decision 1 |
| 24 | Seed / reproducibility | Single seed, different from paper's unknown seed |
| 28 | Evaluation grid | PINN evaluated on FD grid — may miss fine structure |

### Confirmed non-issues

| # | Issue | Status |
|---|-------|--------|
| 30 | Detached history values | Correct as implemented |
| 31 | Checkpoint format | Functional; would be replaced if switching frameworks |

---

## Summary of Current Action Items

1. **Change loss weights to λ = 1 for all terms** (baseline matching paper's likely defaults).
2. **Implement Wang et al. 2021 Algorithm 1** (adaptive gradient balancing) as Phase 2 improvement.
3. **Keep tanh output transform with A_bound = 1.0** (follow paper, accept saturation as known limitation).
4. **Keep causal training** (ε = 50, 20 slices) as an improvement over the paper.
5. **[Pending]** Decide whether to run both velocity profiles (paper + outgoing) or commit to one.
6. **[Pending]** Decide on remaining 22 undiscussed issues.
