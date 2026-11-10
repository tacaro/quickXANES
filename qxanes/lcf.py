"""Step 4: combinatorial linear combination fitting (weights >= 0, sum not forced)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import nnls
from scipy.stats import f as f_dist


def _prepare(E, y, standards, erange):
    """Data on the fit range and each standard interpolated onto it (standards not covering it are skipped)."""
    E = np.asarray(E, float)
    y = np.asarray(y, float)
    sel = (E >= erange[0]) & (E <= erange[1]) & np.isfinite(y)
    x, yd = E[sel], y[sel]
    warn, basis = [], {}
    for name, (es, ys) in standards.items():
        b = np.interp(x, es, ys, left=np.nan, right=np.nan)
        if np.all(np.isfinite(b)):
            basis[name] = b
        else:
            warn.append(f"standard '{name}' does not cover {erange[0]:.1f}-{erange[1]:.1f} eV; skipped")
    return x, yd, basis, warn


def _fit(x, yd, basis, combo) -> dict:
    """NNLS fit of one combination of standards. R-factor = sum(resid^2) / sum(y^2) (Athena convention)."""
    k = len(combo)
    A = np.column_stack([basis[n] for n in combo])
    w, _ = nnls(A, yd)
    fit = A @ w
    chi2 = float(np.sum((yd - fit) ** 2))
    redchi = chi2 / max(len(yd) - k, 1)
    try:
        err = np.sqrt(np.clip(np.diag(np.linalg.inv(A.T @ A) * redchi), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(k, np.nan)
    total = float(w.sum())
    ss_y = float(np.sum(yd**2))
    return {
        "components": list(combo),
        "weights": [float(v) for v in w],
        "errors": [float(v) for v in err],
        "sum": total,
        "fractions": [float(v / total) if total else np.nan for v in w],
        "r_factor": chi2 / ss_y if ss_y else np.nan,
        "chi2": chi2,
        "redchi": redchi,
        "npts": int(len(yd)),
        "erange": (float(x[0]), float(x[-1])),
        "energy": x,
        "data": yd,
        "fit": fit,
        "parts": {n: A[:, i] * w[i] for i, n in enumerate(combo)},
        "p_vs_simpler": np.nan,
    }


def lcf_combinatorial(E, y, standards: dict[str, tuple[np.ndarray, np.ndarray]], erange: tuple[float, float],
                      max_components: int = 3) -> tuple[list[dict], list[str]]:
    """Fit every subset (1..max_components) of standards to y on erange.

    Returns (fits sorted by R-factor, warnings). Subsets where NNLS drives a weight to exactly 0
    are dropped: they duplicate a smaller subset's fit.
    """
    x, yd, basis, warn = _prepare(E, y, standards, erange)
    names = list(basis)
    if len(x) < 3 or not names:
        return [], warn + ["not enough points or standards in the fit range"]
    fits = []
    for k in range(1, min(max_components, len(names)) + 1):
        for combo in combinations(names, k):
            f = _fit(x, yd, basis, combo)
            if all(v > 0 for v in f["weights"]):
                fits.append(f)
    _hamilton(fits)
    fits.sort(key=lambda f: f["r_factor"])
    return fits, warn


def lcf_fit_combo(E, y, standards: dict[str, tuple[np.ndarray, np.ndarray]], erange: tuple[float, float],
                  combo: list[str]) -> tuple[dict | None, list[str]]:
    """Fit one user-chosen combination (e.g. a locked-in choice), whatever its rank or size."""
    missing = [n for n in combo if n not in standards]
    if missing:
        return None, [f"standard(s) {', '.join(missing)} not available (role changed or deselected)"]
    x, yd, basis, warn = _prepare(E, y, {n: standards[n] for n in combo}, erange)
    if len(x) < 3 or len(basis) != len(combo):
        return None, warn + ["the chosen standards do not cover the fit range"]
    f = _fit(x, yd, basis, combo)
    if any(v <= 0 for v in f["weights"]):
        warn.append("a chosen standard now has zero weight with the current settings")
    return f, warn


def _hamilton(fits: list[dict]) -> None:
    """p-value (F-test) that a fit's extra component is justified versus its best (k-1)-subset.

    Small p (< 0.05): the extra standard improves the fit beyond noise. XANES points are
    oversampled (correlated), so treat p as optimistic.
    """
    chi = {frozenset(f["components"]): f["chi2"] for f in fits}
    for f in fits:
        k = len(f["components"])
        f["p_vs_simpler"] = np.nan
        if k < 2:
            continue
        subs = [chi[s] for s in (frozenset(c) for c in combinations(f["components"], k - 1)) if s in chi]
        dof = f["npts"] - k
        if not subs or dof <= 0 or f["chi2"] <= 0:
            continue
        F = (min(subs) - f["chi2"]) / (f["chi2"] / dof)
        f["p_vs_simpler"] = float(f_dist.sf(max(F, 0.0), 1, dof))


def fit_to_row(sample: str, f: dict, all_standards: list[str], **extra) -> dict:
    """One CSV row: fit statistics plus a weight/err/fraction column per standard (0 / NaN when unused)."""
    row = {"sample": sample, **extra, "n_components": len(f["components"]),
           "components": " + ".join(f["components"]),
           "r_factor": f["r_factor"], "chi2": f["chi2"], "reduced_chi2": f["redchi"],
           "sum_weights": f["sum"], "p_vs_simpler": f.get("p_vs_simpler", np.nan),
           "fit_E_min": f["erange"][0], "fit_E_max": f["erange"][1], "n_points": f["npts"]}
    lookup = {n: i for i, n in enumerate(f["components"])}
    for s in all_standards:
        i = lookup.get(s)
        row[f"weight[{s}]"] = f["weights"][i] if i is not None else 0.0
        row[f"err[{s}]"] = f["errors"][i] if i is not None else np.nan
        row[f"fraction[{s}]"] = f["fractions"][i] if i is not None else 0.0
    return row


def fits_to_rows(sample: str, fits: list[dict], all_standards: list[str], top_n: int = 5,
                 selected: list[str] | None = None) -> list[dict]:
    """Candidate rows: one per (sample, rank); `selected` marks the user's locked-in combination."""
    chosen = set(selected or [])
    return [fit_to_row(sample, f, all_standards, rank=rank,
                       selected=bool(chosen) and set(f["components"]) == chosen)
            for rank, f in enumerate(fits[:top_n], start=1)]
