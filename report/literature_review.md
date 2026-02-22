# Literature Review: PINN Improvement Techniques for Black Hole QNM Extraction

This document surveys five research directions relevant to improving a Physics-Informed Neural Network (PINN) solving the 1+1 Zerilli/Regge-Wheeler equation for Schwarzschild black hole quasi-normal modes (QNMs).

---

## Table of Contents

1. [PINNs for QNMs in Modified Gravity (Luna et al.)](#1-pinns-for-qnms-in-modified-gravity)
2. [Gradient Pathologies and Learning Rate Annealing (Wang, Teng & Perdikaris)](#2-gradient-pathologies-and-learning-rate-annealing)
3. [Self-Adaptive PINNs with Soft Attention (McClenny & Braga-Neto)](#3-self-adaptive-pinns-with-soft-attention)
4. [Causal Training and Time-Marching for Wave PINNs](#4-causal-training-and-time-marching-for-wave-pinns)
5. [Prony Method for QNM Frequency Extraction](#5-prony-method-for-qnm-frequency-extraction)
6. [Choice of Initial Velocity Profile for Time-Domain Evolution](#6-choice-of-initial-velocity-profile-for-time-domain-evolution)

---

## 1. PINNs for QNMs in Modified Gravity

**Paper:** Luna, R., Maselli, A., & Pani, P. (2024). *"Probing new physics with quasi-normal modes: a neural-network approach."* [arXiv:2404.11583](https://arxiv.org/abs/2404.11583)

### Main Idea

This is the most directly relevant paper: a PINN that computes QNM frequencies of black holes in modified gravity theories (Einstein-scalar-Gauss-Bonnet, or EsGB). Instead of solving a time-domain wave equation, the PINN solves the *frequency-domain* ODE eigenvalue problem, with the complex QNM frequency Ω treated as a **trainable parameter** alongside the network weights.

### Methodology

| Aspect | Detail |
|--------|--------|
| **Formulation** | Frequency-domain ODE: factor out oscillatory/asymptotic behavior, solve for regular function g(x) |
| **Domain compactification** | x = r_H / r maps [r_H, ∞) → [0, 1] |
| **Factorization** | ψ(r) = (oscillatory factor) · g(x); network only learns the regular part g |
| **Architecture** | 3 hidden layers × 200 neurons, GELU activation, PyTorch |
| **Network I/O** | Single input x ∈ [0,1], two outputs: Re(g), Im(g) |
| **Trainable frequency** | Ω = ω_R + iω_I is a learnable `nn.Parameter` optimized alongside weights |
| **Boundary handling** | Hard-enforced normalization g(1) = 1 via: g(x) = (e^(x-1) − 1)·G(x) + 1 |
| **Background** | Coefficients A_{n,m} precomputed via cubic spline interpolation on numerical backgrounds |
| **Optimizer** | Adam, lr = 10⁻³, 1000 epochs per mode |
| **Chain training** | Once a QNM is found for coupling α₀, use its weights to warm-start training at α₀ + δα |
| **Quality criterion** | Final loss < 0.1 indicates reliable QNM frequency |

### Key Results

- **< 0.1% error** for most of the parameter range compared to known reference values
- **1873 QNMs** computed in ~2 hours on a single RTX 4090 GPU
- Accuracy degrades near the hyperbolicity-loss threshold of the modified gravity theory
- Successfully mapped out QNM spectra for the entire EsGB parameter space

### Applicability to Zerilli/RW PINN

| Technique | How to apply |
|-----------|-------------|
| **Frequency as trainable parameter** | Could reformulate the Zerilli equation in frequency domain, making ω a learnable parameter. This directly outputs the QNM frequency without post-processing. |
| **Domain compactification** | The mapping x = r_H/r could be applied to the tortoise coordinate to handle the semi-infinite domain. |
| **Factorization** | Factor out the known asymptotic behavior e^{±iωr*} from the wave solution, letting the network learn only the regular deviation. |
| **Hard boundary enforcement** | Replace soft BC penalty loss terms with hard-coded boundary transforms. This eliminates two loss terms from the multi-objective optimization. |
| **Chain training / warm start** | When sweeping over l values or exploring parameter variations, initialize from previously converged solutions. |

---

## 2. Gradient Pathologies and Learning Rate Annealing

**Paper:** Wang, S., Teng, Y., & Perdikaris, P. (2021). *"Understanding and mitigating gradient flow pathologies in physics-informed neural networks."* SIAM Review, 63(1), 208–228. [arXiv:2001.04536](https://arxiv.org/abs/2001.04536)

**Related:** Wang, S., Yu, X., & Perdikaris, P. (2022). *"When and why PINNs fail to train: A neural tangent kernel perspective."* [arXiv:2007.14527](https://arxiv.org/abs/2007.14527)

### Main Idea

Identifies **numerical stiffness leading to unbalanced back-propagated gradients** as a fundamental failure mode of PINNs. When a PINN loss is L = λ₁L_PDE + λ₂L_BC + λ₃L_IC, the gradients ∇_θ L_PDE and ∇_θ L_BC/IC can differ by orders of magnitude, causing the optimizer to neglect some terms entirely.

### Methodology

#### Learning Rate Annealing Algorithm

The key algorithm dynamically adjusts the loss weights using gradient statistics:

```
For each epoch:
  1. Compute individual loss terms: L_PDE, L_BC, L_IC
  2. Compute gradients: ∇_θ L_PDE, ∇_θ L_BC, ∇_θ L_IC
  3. Set adaptive weight:
     λ̂_i = max_θ |∇_θ L_PDE| / mean_θ |∇_θ L_i|
  4. Exponential moving average:
     λ_i^(k) = (1 − α)·λ_i^(k-1) + α·λ̂_i
  5. Update with total loss: L = L_PDE + Σ λ_i · L_i
```

This balances the gradient magnitudes of different loss terms so that no single objective dominates.

#### Neural Tangent Kernel (NTK) Theory for PINNs

The companion paper develops NTK theory for PINNs:
- The NTK matrix **K** governs training dynamics: du/dτ = −K·(u − v)
- Eigenvalues of K determine convergence rates of different loss components
- Large eigenvalue disparity → some components converge while others stagnate
- The adaptive weighting rescales NTK eigenvalues to equalize convergence rates

### Key Results

- **50–100× improvement** in predictive accuracy across benchmark problems
- Explains *why* PINNs fail: gradient imbalance between PDE residual and boundary terms
- Provides a principled, gradient-statistics-based approach rather than hand-tuned weights
- Code: [github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs](https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs)

### Applicability to Zerilli/RW PINN

| Technique | How to apply |
|-----------|-------------|
| **Learning rate annealing** | Replace fixed `lambda` weights `[1,1,1,1,1,1,1]` in the config with dynamically computed weights at each Adam step. Use the max/mean gradient ratio to auto-balance Lr, Lrx, Lrt, Lic, Liv, Lbl, Lbr. |
| **NTK diagnostics** | Compute the NTK matrix periodically during training to diagnose which loss components are converging vs. stagnating. |
| **Architecture improvements** | The paper proposes modified architectures (multiplicative filter networks) more resilient to gradient pathologies, relevant for wave-type equations. |

**Implementation sketch for `pinn.py`:**
```python
# After computing loss components in compute_losses():
grads_pde = torch.autograd.grad(Lr, model.parameters(), retain_graph=True)
grads_ic  = torch.autograd.grad(Lic, model.parameters(), retain_graph=True)
max_grad_pde = max(g.abs().max() for g in grads_pde)
mean_grad_ic = torch.mean(torch.stack([g.abs().mean() for g in grads_ic]))
lambda_ic = max_grad_pde / mean_grad_ic
# Apply with EMA smoothing
```

---

## 3. Self-Adaptive PINNs with Soft Attention

**Paper:** McClenny, L. D. & Braga-Neto, U. (2023). *"Self-Adaptive Physics-Informed Neural Networks."* Journal of Computational Physics, 474, 111722. [arXiv:2009.04544](https://arxiv.org/abs/2009.04544)

### Main Idea

Instead of a single weight per loss component, SA-PINNs assign a **trainable weight to each individual collocation point**. These weights form a "soft attention mask" that forces the network to focus on difficult regions of the solution. The weights are trained by **gradient ascent** (maximizing the loss w.r.t. weights) while the network weights are trained by gradient descent (minimizing the loss), creating a minimax optimization.

### Methodology

#### SA-PINN Loss Function

```
L(w, λ_r, λ_b, λ_0) = L_s(w) + L_r(w, λ_r) + L_b(w, λ_b) + L_0(w, λ_0)
```

where each weighted loss term is:
```
L_r(w, λ_r) = (1/N_r) Σ_i m(λ_r^i) · |N[u(x_r^i; w)] - f(x_r^i)|²
```

**Key principle:** L is *minimized* w.r.t. network weights **w**, but *maximized* w.r.t. self-adaptive weights **λ**:
```
w^{k+1}   = w^k   − η_k · ∇_w L      (gradient descent)
λ_r^{k+1} = λ_r^k + ρ_k · ∇_λ_r L    (gradient ascent)
```

The mask function m(λ) is a nonneg, strictly increasing, differentiable function (e.g., polynomial m(λ) = cλ^q, or sigmoidal).

#### Properties

- Weights are **monotonically increasing** during training (gradient of L w.r.t. λ is always ≥ 0)
- Points with larger residual errors get larger weights → network focuses on difficult regions
- Separate learning rates for network (η ~ 10⁻⁵) and self-adaptive weights (ρ ~ 10⁻³ to 10⁻¹)
- Compatible with Adam + L-BFGS two-stage training (weights only updated during Adam phase)
- For SGD/mini-batching: use Gaussian Process regression to build a continuous map of weights

#### NTK Analysis

The NTK matrix for SA-PINNs becomes:
```
K(τ) = [[K_rr·Γ_r, K_rb·Γ_b],
         [K_br·Γ_r, K_bb·Γ_b]]
```
where Γ_p is a diagonal matrix of mask values. SA-PINNs **equalize eigenvalues of the NTK** across loss components, which is more flexible than whole-component weighting because it also changes the *shape* of the eigenvalue distribution.

### Key Results

| Benchmark | Baseline PINN L₂ | SA-PINN L₂ | Notes |
|-----------|------------------|------------|-------|
| Burgers | 6.7×10⁻⁴ | **4.8×10⁻⁴** | 20% fewer training iterations |
| Helmholtz | 1.4×10⁻¹ | **3.2×10⁻³** | ~44× improvement |
| Allen-Cahn | 96.2×10⁻² (fails) | **2.1×10⁻²** | ~46× improvement |
| 1D Wave + SGD | fails | **2.95×10⁻²** | Only method that solves it |
| Advection | fails | **~5%** | Order of magnitude improvement |

- The self-adaptive weight maps are remarkably consistent across random restarts
- SA-PINN automatically discovers that early-time regions need heavier weighting in time-evolution problems
- SA-PINN stabilizes L-BFGS convergence where baseline PINN diverges

### Applicability to Zerilli/RW PINN

| Technique | How to apply |
|-----------|-------------|
| **Per-point weights** | Assign trainable weights λ_r^i to each residual collocation point (x_i, t_i). Use gradient ascent with learning rate ρ ~ 0.01–0.1 to update them. |
| **Separate learning rates** | Use a smaller lr (10⁻⁵) for network weights and larger lr (10⁻² to 10⁻¹) for attention weights. |
| **Focus on wavefront** | The soft attention mask should automatically discover that the outgoing wavefront region and early-time ringdown need more attention. |
| **Minimax formulation** | In `train_pinn()`, add gradient ascent step for λ weights after each Adam gradient descent step. Implementation: flip sign of λ gradients using `optimizer.param_groups`. |
| **SGD with GP weight maps** | If using periodic resampling (`resample_period`), build a GP map of learned weights and interpolate to new points. |

**Implementation sketch:**
```python
# Create per-point self-adaptive weights
lambda_r = torch.ones(Nr, requires_grad=True)
lambda_ic = torch.ones(Ni, requires_grad=True)

# Separate optimizers
opt_net = torch.optim.Adam(model.parameters(), lr=1e-5)
opt_lambda = torch.optim.Adam([lambda_r, lambda_ic], lr=0.01, maximize=True)

# Training loop:
loss = sum(m(lambda_r[i]) * r[i]**2) / Nr + ...
opt_net.zero_grad(); opt_lambda.zero_grad()
loss.backward()
opt_net.step()   # minimize w.r.t. network weights
opt_lambda.step() # maximize w.r.t. lambda (using maximize=True)
```

---

## 4. Causal Training and Time-Marching for Wave PINNs

### 4a. Causal Training

**Paper:** Wang, S., Sankaran, S., & Perdikaris, P. (2022). *"Respecting causality is all you need for training physics-informed neural networks."* [arXiv:2203.07404](https://arxiv.org/abs/2203.07404). 376+ citations.

#### Main Idea

Standard PINNs treat all spatio-temporal collocation points equally, violating the causal structure of hyperbolic PDEs (where information propagates forward in time). This causes the optimizer to fit later-time regions before earlier-time regions are accurate, producing cascading errors. The fix: add **causal weights** that force the PINN to satisfy the PDE at earlier times before later times.

#### Methodology

The causal loss reformulation:

1. **Partition time domain** into slices: t₁ < t₂ < ... < t_N
2. **Compute per-slice residuals:** L_i = mean PDE residual at time slice t_i  
3. **Apply causal weights:**
   ```
   w_i = exp(−ε · Σ_{j=1}^{i-1} L_j)
   ```
   where ε is a causality parameter (large ε → strict causality)
4. **Weighted loss:** L_causal = Σ_i w_i · L_i

The key insight: w_i is large (≈1) only when *all previous slices* have small residual. If any earlier slice has large error, w_i → 0, effectively masking out later-time points until earlier times converge.

#### Key Results

- First PINNs to successfully simulate: chaotic Lorenz system, Kuramoto-Sivashinsky in chaotic regime, turbulent Navier-Stokes
- The parameter ε controls strictness: ε → ∞ recovers sequential time-stepping, ε = 0 recovers standard PINN
- Provides a built-in convergence diagnostic: if all w_i ≈ 1, the solution has propagated causally across the full domain

### 4b. Time-Marching PINNs for Wave Equations

**Paper:** Bulut, I. (2025). *"Physics-Informed Neural Networks for Electromagnetic Wave Propagation with Rigorous Energy Conservation."* [arXiv:2512.23396](https://arxiv.org/abs/2512.23396)

#### Main Idea

A hybrid methodology combining **temporal domain decomposition** (time-marching) with **causality-aware loss weighting** and a **local energy conservation regularizer** to achieve finite-difference-level accuracy for wave equations using PINNs.

#### Methodology

| Component | Detail |
|-----------|--------|
| **Time windowing** | Divide [0, T_max] into N_w windows. Train each window independently and sequentially. |
| **Sequential transfer** | Previous window's terminal state → next window's initial condition. Transfer network weights (warm start). |
| **Causality weighting** | Within each window: w_c(τ) = exp(−γτ), γ = 2. Weights residuals more heavily at beginning of window. |
| **Interface continuity** | Two-stage: (i) IC matching loss at window boundary, (ii) dedicated interface sampling points. |
| **Local energy constraint** | Pointwise Poynting regularizer: ∂_t u + ∇·S = 0 enforced at collocation points. **Local** beats **global** energy constraint (which causes cancellation-of-errors). |
| **Architecture** | 8 hidden layers × 128 neurons, skip connections every 2 layers (ResNet-inspired) |
| **Training** | Two-stage per window: 1500 epochs Adam + 300 epochs L-BFGS. Dynamic loss weighting. |

#### The "Parenthesis Effect"

A surprising finding: algebraically equivalent code expressions (e.g., `a*(b+c)` vs `a*b + a*c`) produce **different optimization dynamics** in PINNs due to different computational graph structures and gradient flow paths.

#### Key Results

| Metric | Value |
|--------|-------|
| NRMSE | 0.09% |
| L² error | 1.01% |
| Energy mismatch vs FDTD | 0.024% |
| Comparison | Matches FDTD accuracy |

Ablation shows the **local** Poynting constraint is essential. The **global** Poynting integral constraint is actually *worse* than no constraint due to error cancellation hiding local violations.

### Applicability to Zerilli/RW PINN

| Technique | How to apply |
|-----------|-------------|
| **Causal loss weighting** | Sort collocation points by time, partition into time slices, apply exponential causal weights w_i = exp(−ε Σ L_j). In `compute_losses()`, weight each residual point by its causal factor. Start with ε ≈ 10–100. |
| **Time windowing** | Split the time domain [t_min, t_max] into N_w overlapping windows. Train each window sequentially, warm-starting from the previous window's weights. This is especially useful for long-time QNM ringdown evolution. |
| **Interface continuity** | At each window boundary, add extra collocation points and enforce continuity of φ and ∂φ/∂t. |
| **Local energy conservation** | For the Zerilli equation, enforce the local energy flux conservation law pointwise rather than as a global integral. Add L_energy = mean(|∂_t E + ∂_x F|²) to the loss. |
| **Skip connections** | Add ResNet-style skip connections every 2 layers in `FCN`. This improves gradient flow for deeper networks. |
| **Two-stage training** | The current Adam + L-BFGS pipeline matches this approach. Increase L-BFGS iterations per window. |

**Implementation sketch for causal weighting:**
```python
# In compute_losses(), add causal weighting to residual:
time_vals = Xr[:, 1]  # time coordinate
t_sorted, sort_idx = torch.sort(time_vals.squeeze())
n_slices = 20
slice_bounds = torch.linspace(tmin, tmax, n_slices + 1)

epsilon = 50.0  # causality strictness
cumulative_loss = 0.0
causal_loss = 0.0
for k in range(n_slices):
    mask = (time_vals >= slice_bounds[k]) & (time_vals < slice_bounds[k+1])
    slice_residual = torch.mean(r[mask]**2)
    w_k = torch.exp(-epsilon * cumulative_loss)
    causal_loss += w_k * slice_residual
    cumulative_loss += slice_residual.detach()
```

---

## 5. Prony Method for QNM Frequency Extraction

### Main Idea

The Prony method (also called the matrix pencil method) is a classical signal processing technique for decomposing a time-domain signal into a sum of **damped complex exponentials**:

$$\Psi(t) \approx \sum_{n=1}^{N} C_n \, e^{-i\omega_n t}$$

where ω_n = ω_R,n + iω_I,n are the complex QNM frequencies (real part = oscillation frequency, imaginary part = damping rate) and C_n are complex amplitudes.

### Methodology

#### Standard Prony Algorithm

Given uniformly sampled time-series data {ψ(t_k)} at times t_k = t₀ + kΔt:

1. **Form the data matrix (Hankel matrix):**
   ```
   H = [[ψ₀, ψ₁, ..., ψ_{N-1}],
        [ψ₁, ψ₂, ..., ψ_N],
        ...
        [ψ_{M-1}, ψ_M, ..., ψ_{M+N-2}]]
   ```

2. **Solve the linear prediction problem:**
   Find coefficients aₖ such that ψ_{n} = Σ aₖ ψ_{n-k} (autoregressive model)

3. **Find roots of the characteristic polynomial:**
   z^N + a₁z^{N-1} + ... + a_N = 0.
   The roots zₖ give the QNM frequencies via ωₖ = i·ln(zₖ)/Δt

4. **Recover amplitudes** by solving the linear system C_n

#### Practical Considerations for QNMs

- **Fitting window:** Start *after* the prompt response has decayed, during the pure ringdown phase
- **Number of modes:** Use SVD of the Hankel matrix; singular values that plateau above the noise floor indicate the number of physical modes
- **Overtones:** Multiple QNM overtones (n = 0, 1, 2, ...) can be extracted simultaneously if the signal quality is sufficient
- **Validation:** Compare extracted frequencies against WKB approximations or Leaver's continued-fraction method

### Key References

- **Konoplya & Stashko (PRD, 2025):** Use Prony method to fit time-domain integration data for QNMs of hairy black holes; results match WKB to high precision
- **Yang, Wu, Li & Zhang (PRD, 2023):** Prony method extracts QNMs from time-domain profiles, matching 6th-order WKB results
- **Xia, Lan & Miao (CQG, 2024):** Prony method for QNMs in loop quantum gravity with high-precision extraction
- **Aneesh, Bose & Kar (PRD, 2018):** Direct integration + Prony fit for wormhole QNMs, demonstrating robustness for exotic compact objects
- **Bolokhov & Skvortsova (2025):** Comprehensive comparison of WKB + Padé approximants + time-domain integration with Prony analysis

### Applicability to Zerilli/RW PINN

| Technique | How to apply |
|-----------|-------------|
| **Post-processing PINN output** | After training the PINN, evaluate φ(x₀, t) at a fixed observation point x₀. Apply Prony method to the resulting time series to extract QNM frequencies ω_n and damping times τ_n = 1/ω_I,n. |
| **Fitting window selection** | Identify the ringdown onset (after prompt response) from the PINN output. Use the signal from ~50M to ~200M for fundamental mode extraction. |
| **Multi-mode extraction** | The Prony method naturally extracts *all* QNM overtones present in the signal, not just the fundamental mode. |
| **Comparison with existing approach** | The current `extract_qnm.py` likely uses peak-fitting or FFT. Prony provides direct complex frequency extraction, which is more physical. |
| **Noise handling** | PINN outputs are smooth (no numerical noise from discretization), making them excellent inputs for Prony analysis. |
| **Hybrid validation** | Extract QNMs via Prony from both the PINN output and the finite-difference output (`zerilli_l2_fd.npz`), then compare. |

**Implementation sketch:**
```python
import numpy as np
from scipy.linalg import hankel, svd, lstsq

def prony_qnm(signal, dt, n_modes=3):
    """Extract QNM frequencies via Prony/matrix-pencil method."""
    N = len(signal)
    M = N // 2  # Hankel matrix size
    
    # Form Hankel matrix
    H0 = hankel(signal[:M], signal[M-1:N-1])
    H1 = hankel(signal[1:M+1], signal[M:N])
    
    # SVD for rank estimation
    U, s, Vh = svd(H0)
    
    # Truncate to n_modes
    U_r = U[:, :n_modes]
    S_r = np.diag(s[:n_modes])
    Vh_r = Vh[:n_modes, :]
    
    # Generalized eigenvalue problem
    H0_pinv = Vh_r.conj().T @ np.linalg.inv(S_r) @ U_r.conj().T
    Z = H0_pinv @ H1
    eigenvalues = np.linalg.eigvals(Z)
    
    # Convert to frequencies
    omega = 1j * np.log(eigenvalues) / dt
    
    # Sort by damping rate (imaginary part)
    idx = np.argsort(np.abs(omega.imag))
    return omega[idx]
```

---

## Summary: Prioritized Recommendations

Based on this review, the following improvements are prioritized by **expected impact** on the current Zerilli/RW PINN:

### High Priority (Directly addresses known PINN failure modes for wave equations)

1. **Causal loss weighting** (§4a): Minimal implementation effort, large expected accuracy gain for the wave equation. Add exponential causal weights to the residual loss in `compute_losses()`.

2. **Learning rate annealing / gradient balancing** (§2): Replace fixed `lambda` weights with dynamic gradient-statistics-based weights. The current 7-component loss (`Lr, Lrx, Lrt, Lic, Liv, Lbl, Lbr`) is highly susceptible to gradient imbalance.

3. **Prony method for QNM extraction** (§5): Pure post-processing improvement, independent of training. Add a Prony-based extraction script alongside existing QNM extraction methods.

### Medium Priority (Significant improvements but more implementation effort)

4. **Self-adaptive per-point weights** (§3): More powerful than whole-component weighting but requires minimax optimization (dual optimizer). Start with per-point weights on residual points only.

5. **Time-marching / domain decomposition** (§4b): Essential for long-time simulations. Split the time domain into windows and train sequentially with warm-starting.

### Lower Priority (Architectural changes, frequency-domain reformulation)

6. **Frequency-domain PINN** (§1): Complete reformulation of the problem. Very powerful (QNM frequency is a direct output) but requires a separate codebase. Consider as a complementary approach.

7. **Skip connections** (§4b): Simple architectural improvement. Add ResNet-style skip connections to `FCN` every 2 layers.

---

## 6. Choice of Initial Velocity Profile for Time-Domain Evolution

### Context and Motivation

When solving the 1+1 wave equation for black hole perturbations (Zerilli or Regge-Wheeler) in tortoise coordinates,

$$\frac{\partial^2 \Phi}{\partial t^2} = \frac{\partial^2 \Phi}{\partial x^2} - V(x)\,\Phi,$$

one must specify both the initial displacement $\Phi(x, 0)$ and the initial velocity $\partial_t \Phi(x, 0)$. The displacement is universally chosen as a Gaussian:

$$\Phi(x, 0) = A \exp\!\bigl[-(x - x_0)^2 / \sigma^2\bigr].$$

However, the choice of initial velocity profile $\Phi_t(x,0)$ varies across the literature, and different choices affect the physical content of the initial pulse (whether it is purely outgoing, purely ingoing, momentarily static, or a mixture). Because QNM frequencies are intrinsic properties of the black hole and do not depend on initial data, any reasonable initial data will eventually excite the same QNMs. The choice does however affect:

1. **The relative amplitude of QNM overtones** — initial data closer to the potential peak excites the fundamental mode more strongly relative to overtones (Silva et al. 2026, Appendix C).
2. **Whether the pulse is initially right-moving, left-moving, or static** — affecting how much energy reaches the black hole vs. spatial infinity.
3. **The duration and structure of the "prompt" (direct) signal** before the ringdown.

This section surveys what choices appear in the literature and evaluates the profile used in our reference paper (Patel et al. 2024).

### 6.1 Three Standard Approaches

#### (a) Momentarily static initial data: $\Phi_t = 0$

The simplest and most common choice in Cauchy evolution is **momentarily static** (also called "instantaneously static" or "time-symmetric") initial data, where $\partial_t \Phi\big|_{t=0} = 0$. The Gaussian sits at rest and then splits into equal left-moving and right-moving halves under the wave equation. Papers using this approach include:

- **Silva, Tambalo, Glampedakis & Yagi (2026)** [arXiv:2601.13411] — Explicitly state (Eqs. 18–20): "$\partial_t X_\ell^{(\pm)}\big|_{t=0} = 0$" with $X\big|_{t=0} = A\exp[-(r_* - r_*^{\text{med}})^2/(2\sigma^2)]$, $A=1$, $2\sigma = 1.5M$, $r_*^{\text{med}} = 100M$. They use 4th-order finite differences with 3rd-order Runge-Kutta (method of lines) and extract fundamental QNM frequencies to within 2% of frequency-domain values.

- **Ikeda, Bianchi, Consoli, Grillo, Morales et al. (2021)** [PRD 104, 066021] — "For the initial data, we took an **instantaneously static Gaussian**... [the] static Gaussian pulse initially splits into an outgoing pulse and an ingoing pulse."

- **Papadopoulos, Seidel & Wild (1998)** [PRD 58, 084002] — Use a "quadrupole Gaussian pulse in the Regge-Wheeler function" with "**vanishing initial time derivative**." This paper develops adaptive mesh refinement for black hole perturbation equations.

- **Berti, Cardoso, Cheung, Di Filippo et al. (2022)** [PRD 106, 084011] — Study stability of the fundamental QNM against potential perturbations using "initial data corresponding to a localized Gaussian pulse" in tortoise coordinates, with time-domain integration.

**Rationale:** Momentarily static data is the safest default because it introduces no bias toward ingoing or outgoing propagation. The Gaussian simply splits symmetrically. This is especially important when studying QNM excitation factors, which depend on effective source structure.

#### (b) Outgoing pulse: $\Phi_t = -\Phi_x$

In standard 1+1 wave mechanics (without a potential), a purely right-moving (outgoing) solution satisfies $\Phi(t, x) = f(x - t)$, which implies $\Phi_t = -\Phi_x$ at $t = 0$. For the Gaussian $\Phi = A\exp[-(x-x_0)^2/\sigma^2]$, this gives:

$$\Phi_t(x, 0) = -\Phi_x = \frac{2(x - x_0)}{\sigma^2}\,\Phi(x, 0).$$

Note the **linear** factor $(x - x_0)$: the velocity is antisymmetric about the pulse center, positive ahead and negative behind, consistent with a rightward-moving wavepacket.

Papers using outgoing initial data include:

- **Pazos-Avalos & Lousto (2005)** [PRD 72, 084022] — Numerical integration of the Teukolsky equation in the time domain: "As initial data we consider an **outgoing Gaussian pulse**" in tortoise coordinates.

**Rationale:** Outgoing data sends all the pulse energy toward the black hole (assuming $x_0$ is to the right of the potential peak), maximizing the excitation of QNMs and eliminating the rightward-propagating half that would otherwise carry energy to spatial infinity without interacting with the potential.

#### (c) Characteristic (null coordinate) methods — GPP approach

An entirely different approach, pioneered by **Gundlach, Price & Pullin (1994)** [PRD 49, 883 (Part I, gr-qc/9307009) and PRD 49, 890 (Part II, gr-qc/9307010)], reformulates the wave equation in double-null coordinates $(u, v)$ where $u = t - x$ and $v = t + x$. In this framework, initial data is specified on two null surfaces: typically $\Phi = 0$ on $u = u_0$ and a Gaussian in $v$ on $v = v_0$:

$$\Phi(u_0, v) = A\exp\!\bigl[-(v - v_c)^2/\sigma_v^2\bigr].$$

This is the most widely used method in the QNM community for time-domain calculations, employed in hundreds of papers following GPP. It avoids specifying $\Phi_t$ altogether — the characteristic initial-value problem requires data on null surfaces, not Cauchy data $(Φ, Φ_t)$.

Papers using this approach include:

- **Gundlach, Price & Pullin (1994)** [PRD 49, 883; 890] — Original papers using "null data... a Gaussian of width 3 centered at $v = 10$."
- **Konoplya (2003)** [PRD 68, 024018; gr-qc/0303052] — Uses the GPP finite-difference scheme for time-domain integration in higher-dimensional Schwarzschild spacetimes.
- **Zhidenko (2009)** [PhD thesis, arXiv:0903.3555] — Reviews and applies the GPP method extensively.
- **Konoplya & Zhidenko (2011)** [Rev. Mod. Phys. 83, 793; arXiv:1102.4014] — Comprehensive review discussing time-domain integration via the GPP method.
- **Bolokhov & Skvortsova (2025)** [arXiv:2508.19989] — "Initial data are specified as a Gaussian pulse on an ingoing null surface."
- **Dubinsky (2025)** [EPJC] — "Initial data are typically specified on the two null surfaces $u = u_0$ and $v = v_0$, for instance using a Gaussian pulse."

**Rationale:** Null coordinates naturally separate ingoing and outgoing propagation. The finite-difference stencil of GPP (integrating on the light cone) is simple, stable, and efficient. Most papers studying QNMs via time-domain evolution (outside the PINN context) use this method.

### 6.2 Analysis of Patel et al. (2024) — Equation 23

In **Patel, Cho & Purrer (2024)** [Gen. Relativ. Gravit., arXiv:2401.01440], the initial velocity is specified as (their Eq. 23):

$$\Phi_t(x, 0) = \frac{2(x - x_0)^2}{\sigma^2}\,\Phi(x, 0).$$

They describe this as a "Gaussian **outgoing** pulse." However, this expression has a **quadratic** factor $(x - x_0)^2$ rather than the standard **linear** factor $(x - x_0)$ that characterizes a true outgoing pulse.

#### Mathematical comparison

| Profile | Formula | Sign structure | Physical meaning |
|---------|---------|---------------|-----------------|
| **Momentarily static** | $\Phi_t = 0$ | Zero everywhere | Pulse at rest; splits symmetrically |
| **Outgoing** ($-\Phi_x$) | $\Phi_t = \frac{2(x-x_0)}{\sigma^2}\Phi$ | Antisymmetric; changes sign at $x_0$ | Purely right-moving pulse |
| **Patel et al. Eq. 23** | $\Phi_t = \frac{2(x-x_0)^2}{\sigma^2}\Phi$ | **Always non-negative**; zero only at $x_0$ | "Breathing" mode; expands outward |

The Patel et al. profile is everywhere non-negative (since $(x-x_0)^2 \geq 0$ and $\Phi > 0$ for the Gaussian). This means the pulse simultaneously expands in both directions — it has no net directionality. Physically, this is neither an outgoing pulse nor a static pulse, but rather a symmetric expansion (the wings of the Gaussian move outward from the center while the center remains at rest).

#### Is this profile used elsewhere?

After an extensive search of the literature, **no other paper was found using this specific quadratic profile** $(x-x_0)^2 \Phi / \sigma^2$ for the initial velocity in black hole perturbation theory. The profile appears to be unique to Patel et al. (2024). It may be a typographical error (perhaps $(x-x_0)$ was intended instead of $(x-x_0)^2$), or it may be a deliberate but unconventional choice.

#### Impact on QNM extraction

Despite being nonstandard, the choice of initial velocity profile does **not** affect the extracted QNM frequencies, because:

1. **QNM frequencies are eigenvalues of the system**, determined entirely by the potential $V(x)$ and the boundary conditions (ingoing at the horizon, outgoing at infinity). They are independent of initial data.
2. **Any initial data with support overlapping the potential barrier** will excite QNMs. The initial velocity only changes the **amplitudes** (excitation coefficients) of the various modes.
3. **The late-time ringdown** — the portion of the signal from which QNMs are extracted — is dominated by the fundamental mode regardless of the initial data, provided the data is "generic" (i.e., does not have fine-tuned symmetries that suppress specific modes).

However, the initial data does affect **which overtones contribute at early times**, and it affects the transient "prompt response" that precedes the ringdown. For a PINN that must learn the entire solution including early times, the choice of initial velocity is more consequential than for a traditional time-domain simulation where one simply discards early data.

### 6.3 Recommendation

For the finite-difference solver in this project, either of the following is appropriate:

1. **Momentarily static** ($\Phi_t = 0$): Simplest, well-established, requires no justification. This is the safest choice for benchmarking.
2. **Outgoing** ($\Phi_t = -\Phi_x$): Uses all pulse energy to excite QNMs (sends everything toward the BH), giving a cleaner ringdown signal.

The Patel et al. quadratic profile ($\Phi_t \propto (x-x_0)^2 \Phi$) can be retained for faithful reproduction of their paper's results, but should be clearly labelled as nonstandard. When comparing with literature values of QNM frequencies, the extracted frequencies should be identical regardless of the velocity profile.

### 6.4 Summary Table of Literature Initial Data Choices

| Reference | Year | Method | Coordinates | Initial velocity | Notes |
|-----------|------|--------|-------------|-----------------|-------|
| Vishveshwara | 1970 | Cauchy FD | tortoise | Gaussian scattering | Original BH scattering experiment |
| Gundlach, Price & Pullin | 1994 | Null FD (GPP) | $(u, v)$ | N/A (null data) | Gaussian on $v$-surface |
| Papadopoulos, Seidel & Wild | 1998 | Cauchy FD (AMR) | tortoise | $\Phi_t = 0$ | "Vanishing initial time derivative" |
| Konoplya | 2003 | Null FD (GPP) | $(u, v)$ | N/A (null data) | Standard GPP method |
| Pazos-Avalos & Lousto | 2005 | Cauchy FD | tortoise | $\Phi_t = -\Phi_x$ | "Outgoing Gaussian pulse" (Teukolsky eq.) |
| Konoplya & Zhidenko | 2011 | Review | both | both | Rev. Mod. Phys. review |
| Ikeda et al. | 2021 | Cauchy FD | tortoise | $\Phi_t = 0$ | "Instantaneously static Gaussian" |
| Berti et al. | 2022 | Cauchy FD | tortoise | $\Phi_t = 0$ | Localized Gaussian pulse |
| Patel, Cho & Purrer | 2024 | Cauchy FD + PINN | tortoise | $\Phi_t = \frac{2(x-x_0)^2}{\sigma^2}\Phi$ | **Nonstandard**; Eq. 23 |
| Zhu, Ripley et al. | 2024 | Cauchy FD | tortoise | Gaussian pulse | QNM extraction challenges |
| Silva et al. | 2026 | Cauchy FD (MOL) | tortoise | $\Phi_t = 0$ | "Momentarily static"; 4th-order FD |

---

## References

1. Luna, R., Maselli, A., & Pani, P. (2024). arXiv:2404.11583
2. Wang, S., Teng, Y., & Perdikaris, P. (2021). SIAM Review 63(1). arXiv:2001.04536
3. Wang, S., Yu, X., & Perdikaris, P. (2022). J. Comp. Phys. 449. arXiv:2007.14527
4. McClenny, L. D. & Braga-Neto, U. (2023). J. Comp. Phys. 474. arXiv:2009.04544
5. Wang, S., Sankaran, S., & Perdikaris, P. (2022). arXiv:2203.07404
6. Bulut, I. (2025). arXiv:2512.23396
7. Xiang, Z., Peng, W., Liu, X., & Yao, W. (2022). Neurocomputing 496
8. Penwarden, M. et al. (2023). "A unified scalable framework for causal sweeping strategies for PINNs"
9. Konoplya, R.A. & Stashko, O.S. (2025). PRD — Prony method for QNMs
10. Yang, Z., Wu, Y., Li, J. & Zhang, H. (2023). PRD — Prony-extracted QNMs
11. Patel, H., Cho, G. & Purrer, M. (2024). Gen. Relativ. Gravit. arXiv:2401.01440
12. Silva, H. O., Tambalo, G., Glampedakis, K. & Yagi, K. (2026). arXiv:2601.13411
13. Gundlach, C., Price, R. H. & Pullin, J. (1994). PRD 49, 883. gr-qc/9307009
14. Gundlach, C., Price, R. H. & Pullin, J. (1994). PRD 49, 890. gr-qc/9307010
15. Papadopoulos, P., Seidel, E. & Wild, L. (1998). PRD 58, 084002
16. Pazos-Avalos, E. & Lousto, C. O. (2005). PRD 72, 084022
17. Konoplya, R. A. (2003). PRD 68, 024018. gr-qc/0303052
18. Konoplya, R. A. & Zhidenko, A. (2011). Rev. Mod. Phys. 83, 793. arXiv:1102.4014
19. Zhidenko, A. (2009). PhD thesis. arXiv:0903.3555
20. Ikeda, T., Bianchi, M., Consoli, D. et al. (2021). PRD 104, 066021
21. Berti, E., Cardoso, V., Cheung, M. H. Y. et al. (2022). PRD 106, 084011
22. Zhu, H., Ripley, J. L., Cárdenas-Avendaño, A. & Pretorius, F. (2024). PRD 109, 044010
23. Nollert, H.-P. & Price, R. H. (1999). J. Math. Phys. 40, 980. gr-qc/9810074
