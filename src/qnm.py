from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit


def _damped_cos(t: np.ndarray, A: float, tau: float, omega: float, phi: float) -> np.ndarray:
    return A * np.exp(-t / tau) * np.cos(omega * t + phi)


def _fft_estimate_omega(tt: np.ndarray, yy: np.ndarray, pad_factor: int = 64) -> float:
    """
    Estimate omega (rad / unit time) from an FFT of the time series.

    Implementation details (chosen to make FFT-based estimates usable on short windows):
    - subtract mean (remove DC),
    - apply a Hann window (reduce spectral leakage),
    - zero-pad by `pad_factor` (increase frequency sampling density),
    - parabolic interpolation around the FFT magnitude peak.

    This is consistent with the spirit of the paper’s “Fourier transform” approach, but avoids
    the coarse frequency-bin artifact that occurs when the fit window is short.
    """
    tt = np.asarray(tt, dtype=float)
    yy = np.asarray(yy, dtype=float)
    dt = float(tt[1] - tt[0])

    N = tt.size
    w = np.hanning(N)
    y0 = (yy - np.mean(yy)) * w

    # choose FFT length as next power of 2 >= pad_factor*N
    Nfft = 1
    while Nfft < pad_factor * N:
        Nfft *= 2

    Y = np.abs(rfft(y0, n=Nfft))
    freqs = rfftfreq(Nfft, d=dt)  # cycles per unit time

    # ignore the zero bin
    k = int(np.argmax(Y[1:])) + 1

    # parabolic interpolation using k-1, k, k+1
    if 1 <= k < (Y.size - 1):
        alpha, beta, gamma = Y[k - 1], Y[k], Y[k + 1]
        denom = alpha - 2.0 * beta + gamma
        if denom != 0:
            p = 0.5 * (alpha - gamma) / denom
        else:
            p = 0.0
    else:
        p = 0.0

    k_interp = k + p
    freq_interp = k_interp * freqs[1]  # linear in k for uniform FFT bins
    omega = 2.0 * np.pi * freq_interp
    return float(omega)


def qnm_method_1(t: np.ndarray, y: np.ndarray, t_start: float, t_end: float) -> Dict[str, float]:
    """
    Method 1 (per the target paper):
      - FFT to estimate ω,
      - log-linear fit of the envelope maxima to estimate τ.

    Returns:
      { "omega": ω, "tau": τ }
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (t >= t_start) & (t <= t_end)
    tt = t[mask]
    yy = y[mask]

    omega = _fft_estimate_omega(tt, yy)

    peaks, _ = find_peaks(np.abs(yy))
    if peaks.size < 2:
        return {"omega": float(omega), "tau": float("nan")}

    tp = tt[peaks]
    ap = np.abs(yy[peaks])

    # log(ap) = c - tp/tau
    coeff = np.polyfit(tp, np.log(ap + 1e-30), deg=1)
    slope = float(coeff[0])
    tau = -1.0 / slope if slope < 0 else float("nan")

    return {"omega": float(omega), "tau": float(tau)}


def qnm_method_2(t: np.ndarray, y: np.ndarray, t_start: float, t_end: float) -> Dict[str, float]:
    """
    Method 2: direct nonlinear fit of a damped cosine:
        y(t) ≈ A exp(-t/τ) cos(ω t + φ).

    Returns:
      { "omega": ω, "tau": τ, "A": A, "phi": φ }
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (t >= t_start) & (t <= t_end)
    tt = t[mask]
    yy = y[mask]

    m1 = qnm_method_1(t, y, t_start, t_end)
    omega0 = m1["omega"]
    tau0 = m1["tau"] if np.isfinite(m1["tau"]) else 10.0

    A0 = float(np.max(np.abs(yy)))
    phi0 = 0.0

    popt, _ = curve_fit(
        _damped_cos,
        tt,
        yy,
        p0=[A0, tau0, omega0, phi0],
        maxfev=50000,
    )
    A, tau, omega, phi = popt
    return {"omega": float(omega), "tau": float(tau), "A": float(A), "phi": float(phi)}
