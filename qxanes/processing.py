"""Steps 1-2: build mu(E) from raw columns, dead-time correction, energy calibration, averaging."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import savgol_filter
from scipy.special import lambertw

from .readers import Scan


# ----------------------------------------------------------------------------- dead time
def deadtime_factor(icr_counts, dwell_s, tau_us, model="paralyzable"):
    """Correction factor (true / measured) from a channel's input count rate.

    paralyzable:      r_meas = r_true * exp(-r_true * tau)  ->  r_true * tau = -W0(-r_meas * tau)
    non-paralyzable:  r_true = r_meas / (1 - r_meas * tau)

    Returns (factor, saturated) where `saturated` marks points beyond the model's
    invertible range (the factor there is clipped).
    """
    r = np.asarray(icr_counts, float) / np.asarray(dwell_s, float)
    tau = float(tau_us or 0.0) * 1e-6
    if tau <= 0:
        return np.ones_like(r), np.zeros(r.shape, bool)
    x = np.clip(r * tau, 0.0, None)
    if model == "paralyzable":
        limit = 1.0 / np.e
        sat = x >= limit
        xc = np.minimum(x, limit * (1 - 1e-9))
        y = -np.real(lambertw(-xc, 0))
    else:
        sat = x >= 0.99
        xc = np.minimum(x, 0.99)
        y = xc / (1.0 - xc)
    factor = np.where(xc > 0, y / np.where(xc > 0, xc, 1.0), 1.0)
    return factor, sat


# ----------------------------------------------------------------------------- signal
def _col(scan: Scan, name: str, sig: dict) -> np.ndarray:
    v = scan.columns[name].astype(float)
    if sig.get("subtract_offsets") and name in scan.offsets:
        dwell = scan.columns.get(sig.get("dwell"), 1.0)
        v = v - scan.offsets[name] * dwell
    return v


def compute_signal(scan: Scan, sig: dict, dt: dict, channels: list[str] | None = None) -> dict:
    """mu(E) for one scan.

    sig: {mode: fluorescence|transmission|column, i0, channels, icr{ch: icr_col}, dwell, trans, column,
          subtract_offsets}
    dt:  {enabled, model, tau_us{ch: tau}}
    channels: per-file channel override (fluorescence only); None = sig["channels"].
    """
    E = scan.energy.astype(float)
    mode = sig.get("mode", "fluorescence")
    per: dict[str, np.ndarray] = {}
    dt_max = 0.0
    saturated = False
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "fluorescence":
            wanted = channels if channels is not None else sig.get("channels", [])
            chans = [c for c in wanted if c in scan.columns]
            if not chans or sig.get("i0") not in scan.columns:
                raise ValueError(f"{scan.name}: needs an I0 column and at least one fluorescence channel")
            i0 = _col(scan, sig["i0"], sig)
            dwell = scan.columns.get(sig.get("dwell"), np.ones_like(E))
            total = np.zeros_like(E)
            for ch in chans:
                counts = _col(scan, ch, sig)
                icr = sig.get("icr", {}).get(ch)
                tau = (dt.get("tau_us") or {}).get(ch, 0) or 0
                if dt.get("enabled") and icr in scan.columns and tau > 0:
                    f, s = deadtime_factor(_col(scan, icr, sig), dwell, tau, dt.get("model", "paralyzable"))
                    counts = counts * f
                    dt_max = max(dt_max, float((f.max() - 1.0) * 100.0))
                    saturated |= bool(s.any())
                per[ch] = counts / i0
                total += counts
            mu = total / i0
        elif mode == "transmission":
            mu = np.log(_col(scan, sig["i0"], sig) / _col(scan, sig["trans"], sig))
        else:
            mu = scan.columns[sig["column"]].astype(float)

    E_u, idx = np.unique(E, return_index=True)          # sorted, duplicate energies dropped
    mu = mu[idx]
    per = {k: v[idx] for k, v in per.items()}
    ok = np.isfinite(mu)
    return {
        "energy": E_u[ok],
        "mu": mu[ok],
        "index": idx[ok],          # rows of the original scan kept, in output order
        "channels": {k: v[ok] for k, v in per.items()},
        "dt_max_pct": dt_max,
        "saturated": saturated,
    }


# ----------------------------------------------------------------------------- calibration
def _parabolic_vertex(x, y, i):
    """Refine an extremum at index i with a parabola through (i-1, i, i+1)."""
    if 0 < i < len(y) - 1:
        c = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
        if c[0] != 0:
            v = -c[1] / (2 * c[0])
            if x[i - 1] <= v <= x[i + 1]:
                return float(v)
    return float(x[i])


def dense_region(E: np.ndarray) -> np.ndarray:
    """Mask of points in the finely sampled part of the scan (where the edge normally is)."""
    step = np.gradient(E)
    med = np.median(step[step > 0]) if np.any(step > 0) else 1.0
    return step <= 2.0 * med


def find_e0(E, mu, window=None) -> float:
    """Edge energy = maximum of the (lightly smoothed) first derivative within the dense region."""
    E = np.asarray(E, float)
    mu = np.asarray(mu, float)
    if len(E) < 5:
        return float(E[len(E) // 2]) if len(E) else float("nan")
    k = np.ones(3) / 3
    mu_s = np.convolve(mu, k, mode="same")
    mu_s[0], mu_s[-1] = mu[0], mu[-1]
    d = np.gradient(mu_s, E)
    mask = dense_region(E)
    if window is not None:
        mask &= (E >= window[0]) & (E <= window[1])
    if mask.sum() < 3:
        mask = np.ones_like(E, bool)
    d_masked = np.where(mask, d, -np.inf)
    i = int(np.argmax(d_masked))
    return _parabolic_vertex(E, d, i)


def find_feature(E, mu, kind="peak", window=None, smooth_pts=5) -> float:
    """Energy of a calibration feature: 'peak' (maximum of mu) or 'derivative' (max of dmu/dE).

    window: (lo, hi) in eV; None -> E0 - 10 ... E0 + 15 around the derivative maximum.
    """
    E = np.asarray(E, float)
    mu = np.asarray(mu, float)
    if not window or window[0] is None or window[1] is None:
        e0 = find_e0(E, mu)
        window = (e0 - 10.0, e0 + 15.0)
    sel = (E >= window[0]) & (E <= window[1])
    if sel.sum() < 5:
        raise ValueError(f"calibration window {window[0]:.1f}-{window[1]:.1f} eV has fewer than 5 points")
    step = float(np.median(np.diff(E[sel])))
    grid = np.arange(E[sel][0], E[sel][-1] + step / 2, step)
    y = np.interp(grid, E, mu)
    n = len(y)
    w = min(smooth_pts | 1, n if n % 2 else n - 1)  # odd Savitzky-Golay window, <= n points
    if kind == "derivative":
        trace = savgol_filter(y, w, 2, deriv=1, delta=step) if w >= 3 else np.gradient(y, step)
    else:
        trace = savgol_filter(y, w, 2) if w >= 3 else y
    return _parabolic_vertex(grid, trace, int(np.argmax(trace)))


# ----------------------------------------------------------------------------- averaging & QC
def average(spectra: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Average replicate spectra on the grid of the replicate with the most points."""
    if not spectra:
        raise ValueError("nothing to average")
    ref = max(range(len(spectra)), key=lambda k: len(spectra[k][0]))
    E = spectra[ref][0]
    stack = np.array([np.interp(E, e, m, left=np.nan, right=np.nan) for e, m in spectra])
    n = np.sum(np.isfinite(stack), axis=0)
    keep = n > 0
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack[:, keep], axis=0)
        std = np.nanstd(stack[:, keep], axis=0, ddof=1) if len(spectra) > 1 else np.zeros(keep.sum())
    std = np.where(n[keep] > 1, std, 0.0)
    return {"energy": E[keep], "mu": mean, "std": std, "n": n[keep]}


def replicate_deviation_pct(E, mu, avg: dict) -> float:
    """RMS deviation of one replicate from the group average, as % of the average's range."""
    y = np.interp(avg["energy"], E, mu, left=np.nan, right=np.nan)
    span = float(np.nanmax(avg["mu"]) - np.nanmin(avg["mu"])) or 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.sqrt(np.nanmean((y - avg["mu"]) ** 2)) / span * 100.0)
