"""Step 3: pre-edge background (Gaussian tail / linear / constant) and post-edge normalization.

Ranges are stored relative to E0 (eV) so one set of parameters transfers between spectra;
`None` means "start/end of the data".
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .processing import find_e0

PRE_MODELS = ("gaussian", "linear", "constant")

DEFAULT_NORM = {
    "e0": None,                 # None -> derivative maximum
    "pre_model": "gaussian",
    "pre_range": [None, -15.0],  # relative to E0
    "post_degree": 1,
    "post_range": [30.0, None],  # relative to E0
    "flatten": True,
}


def gaussian_tail(E, c, A, Ec, sigma):
    """Constant + Gaussian (e.g. a scatter peak passing through the fluorescence window)."""
    return c + A * np.exp(-0.5 * ((E - Ec) / sigma) ** 2)


def _abs_range(E, e0, rel):
    lo = E[0] if rel[0] is None else e0 + float(rel[0])
    hi = E[-1] if rel[1] is None else e0 + float(rel[1])
    return float(lo), float(hi)


def _spacing_sigma(x):
    """Per-point sigma so every eV counts equally (coarse pre-edge steps aren't drowned out by dense ones)."""
    dx = np.gradient(x) if len(x) > 1 else np.ones(1)
    return 1.0 / np.sqrt(np.clip(dx, 1e-6, None))


def fit_pre_edge(E, mu, lo, hi, model="gaussian") -> dict:
    sel = (E >= lo) & (E <= hi)
    x, y = E[sel], mu[sel]
    status = "ok"
    if len(x) < 2:
        return {"model": "constant", "c": float(np.mean(mu[:3])), "npts": int(len(x)),
                "status": "pre-edge window has <2 points; used the first points"}
    sig = _spacing_sigma(x)
    if model == "gaussian":
        if len(x) >= 5:
            w = max(hi - lo, 1e-6)
            lower = [-np.inf, 0.0, lo - 3 * w, w / 100]
            upper = [np.inf, np.inf, hi + 3 * w, 10 * w]
            p0 = [float(y.min()), float(max(y.max() - y.min(), 1e-12)), float(x[np.argmax(y)]), w / 4]
            p0 = list(np.clip(p0, np.array(lower) + 1e-12, np.array(upper) - 1e-12))
            try:
                p, _ = curve_fit(gaussian_tail, x, y, p0=p0, bounds=(lower, upper), sigma=sig, maxfev=20000)
                return {"model": "gaussian", "c": float(p[0]), "A": float(p[1]), "Ec": float(p[2]),
                        "sigma": float(p[3]), "npts": int(len(x)), "status": status}
            except (RuntimeError, ValueError) as exc:
                status = f"Gaussian fit failed ({exc}); fell back to linear"
        else:
            status = "fewer than 5 pre-edge points for a Gaussian; fell back to linear"
        model = "linear"
    if model == "linear":
        m, b = np.polyfit(x, y, 1, w=1.0 / sig)
        return {"model": "linear", "slope": float(m), "intercept": float(b), "npts": int(len(x)), "status": status}
    wts = 1.0 / sig**2
    return {"model": "constant", "c": float(np.sum(wts * y) / np.sum(wts)), "npts": int(len(x)), "status": status}


def eval_pre(E, p: dict) -> np.ndarray:
    E = np.asarray(E, float)
    if p["model"] == "gaussian":
        return gaussian_tail(E, p["c"], p["A"], p["Ec"], p["sigma"])
    if p["model"] == "linear":
        return p["slope"] * E + p["intercept"]
    return np.full_like(E, p["c"])


def normalize(E, mu, cfg: dict | None = None) -> dict:
    """Subtract the pre-edge model over the whole spectrum, fit the post-edge, scale the step to 1."""
    cfg = {**DEFAULT_NORM, **(cfg or {})}
    E = np.asarray(E, float)
    mu = np.asarray(mu, float)
    warn = []
    e0 = float(cfg["e0"]) if cfg.get("e0") not in (None, "") else find_e0(E, mu)
    pre_lo, pre_hi = _abs_range(E, e0, cfg["pre_range"])
    pre = fit_pre_edge(E, mu, pre_lo, pre_hi, cfg["pre_model"])
    if pre["status"] != "ok":
        warn.append(pre["status"])
    bkg = eval_pre(E, pre)
    sub = mu - bkg

    post_lo, post_hi = _abs_range(E, e0, cfg["post_range"])
    deg = int(cfg["post_degree"])
    sel = (E >= post_lo) & (E <= post_hi)
    if sel.sum() < deg + 2:
        raise ValueError(f"post-edge window {post_lo:.1f}-{post_hi:.1f} eV has too few points")
    coef = np.polyfit(E[sel] - e0, sub[sel], deg)   # centred on E0 for conditioning
    post = np.polyval(coef, E - e0)
    step = float(np.polyval(coef, 0.0))
    if step <= 0:
        warn.append(f"edge step is {step:.3g} (<= 0): check E0, windows or the pre-edge model")
    safe = step if step != 0 else np.nan
    norm = sub / safe
    flat = norm.copy()
    above = E >= e0
    flat[above] = norm[above] - (post[above] - step) / safe
    return {
        "energy": E, "mu": mu, "e0": e0, "edge_step": step,
        "pre": pre, "pre_bkg": bkg, "post_line": bkg + post, "post_coef": [float(c) for c in coef],
        "norm": norm, "flat": flat,
        "pre_window": (pre_lo, pre_hi), "post_window": (post_lo, post_hi),
        "config": cfg, "warnings": warn,
    }
