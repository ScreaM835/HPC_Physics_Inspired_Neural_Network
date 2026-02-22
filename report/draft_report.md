# Draft report (first pass): Project 32 — PINN for Schwarzschild Zerilli (ℓ=2)

This draft is intended as a **starting point** for your dissertation write-up. It is structured
so you can progressively replace sections with more rigorous derivations, stronger numerical
experiments, and improved discussion.

---

## 1. Scientific motivation

Project 32 aims to apply **scientific machine learning** to compute the **ringdown spectrum**
(quasi-normal modes, QNMs) of a perturbed Schwarzschild black hole by solving the **Regge–Wheeler**
(odd/axial) or **Zerilli** (even/polar) master equation in a way suitable for **Physics-Informed Neural Networks (PINNs)**.

The project brief emphasizes: (i) formulating the perturbation equation for PINNs,
(ii) implementing the PINN with the appropriate loss terms, and (iii) recovering QNM frequencies
and validating them against established results. (See the Project 32 description.) 

---

## 2. Background: Schwarzschild perturbations → 1+1 master equation

### 2.1 Master equation in tortoise coordinate

After linearizing Einstein’s equations around a Schwarzschild background metric and expanding
the perturbation in tensor spherical harmonics, one can reduce the dynamics to a single scalar
master function Φℓm(t,r), with an associated 1+1 wave-type equation in the tortoise coordinate x.
The target paper writes the master equation schematically as a 1+1 wave equation with an effective
potential Vℓ(r(x)). 

The tortoise coordinate x is defined by
x = r + 2M ln(r/(2M) - 1), mapping r∈(2M,∞) to x∈(-∞,∞). 

### 2.2 Even parity (Zerilli) potential, ℓ=2

For even-parity gravitational perturbations, the effective potential is the Zerilli potential Vℓ^Z(r),
parameterized by ℓ through n=(ℓ-1)(ℓ+2)/2. 

In this project draft, we focus on **ℓ=2**, which is the dominant quadrupolar gravitational mode and is the
main target of ringdown spectroscopy.

---

## 3. Boundary and initial conditions (time domain)

Because V(x)→0 as x→±∞, the asymptotic behavior reduces to the 1+1 flat wave equation.
The target paper enforces **radiative (Sommerfeld) boundary conditions**:
- at the horizon side (x→-∞): (∂t - ∂x)Φ = 0,
- at spatial infinity (x→+∞): (∂t + ∂x)Φ = 0.

The initial condition is a localized Gaussian pulse Φ(x,0)=A exp(-(x-x0)^2/σ^2) with a chosen initial velocity profile. 

---

## 4. Numerical methodology

### 4.1 Finite difference (FD) baseline

We compute a reference time-domain solution by discretizing x and evolving in time using a method-of-lines scheme with RK4 time updates and second-order spatial derivatives. The computational parameters follow the target paper (domain, dx, dt, etc.).

### 4.2 PINN methodology (forward problem)

We approximate Φ(x,t) with a neural network Φ_θ(x,t). The total loss is a weighted sum of:
- PDE residual loss in the interior,
- gradient-enhanced losses from derivatives of the residual,
- initial value and initial velocity losses,
- left/right boundary operator losses enforcing radiative conditions.

---

## 5. QNM extraction

At late times, after the initial burst has passed an observer at fixed x=x_q, the waveform is expected to approach a damped sinusoidal:
Φ(x_q,t) ≈ A exp(-t/τ) cos(ω t + φ).

We implement two QNM extraction methods consistent with the target paper:
1. FFT for ω plus log-peak linear fit for τ,
2. direct nonlinear fit to the damped cosine model.

---

## 6. Planned experiments and dissertation write-up

1. Reproduce the paper’s baseline case (Zerilli, ℓ=2): compare FD vs PINN solution snapshots and error metrics.
2. Sensitivity study:
   - architecture size,
   - training point counts,
   - loss weights (especially the “phase” weight on initial velocity),
   - optional ablations: removing gradient-enhancement terms.
3. QNM extraction accuracy and robustness:
   - effect of fitting window,
   - effect of sampling point x_q.
4. Optional extensions from Project 32:
   - compare against Fourier Neural Operators (FNO),
   - extend to Kerr/Teukolsky (as a future direction).

---

## Appendix A: How to reproduce results with this repository

See `README.md` for the exact commands.
