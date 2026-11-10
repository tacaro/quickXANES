"""Step 4: combinatorial linear combination fitting (weights >= 0, sum not forced)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import nnls
from scipy.stats import f as f_dist


def lcf_combinatorial(E, y, standards: dict[str, tuple[np.ndarray, np.ndarray]], erange: tuple[float, float],
                      max_components: int = 3) -> tuple[list[dict], list[str]]:
    """Fit every subset (1..max_components) of standards to y on erange.

    Returns (fits sorted by R-factor, warnings). Subsets where NNLS drives a weight to exactly 0
    are dropped: they duplicate a smaller subset's fit.
    R-factor = sum(resid^2) / sum(y^2)   (Athena convention)
    """
    E = np.asarray(E, float)
    y = np.asarray(y, float)
    sel = (E >= erange[0]) & (E <= erange[1]) & np.isfinite(y)
    x, yd = E[sel], y[sel]
    warn = []
    basis = {}
    for name, (es, ys) in standards.items():
        b = np.interp(x, es, ys, left=np.nan, right=np.nan)
        if np.all(np.isfinite(b)):
            basis[name] = b
        else:
            warn.append(f"standard '{name}' does not cover {erange[0]:.1f}-{erange[1]:.1f} eV; skipped")
    names = list(basis)
    if len(x) < 3 or not names:
        return [], warn + ["not enough points or standards in the fit range"]
    ss_y = float(np.sum(yd**2))
    fits = []
    for k in range(1, min(max_components, len(names)) + 1):
        for combo in combinations(names, k):
            A = np.column_stack([basis[n] for n in combo])
            w, _ = nnls(A, yd)
            if np.any(w <= 0):
                continue
            fit = A @ w
            resid = yd - fit
            chi2 = float(np.sum(resid**2))
            dof = max(len(yd) - k, 1)
            redchi = chi2 / dof
            try:
                cov = np.linalg.inv(A.T @ A) * redchi
                err = np.sqrt(np.clip(np.diag(cov), 0, None))
            except np.linalg.LinAlgError:
                err = np.full(k, np.nan)
            total = float(w.sum())
            fits.append({
                "components": list(combo),
                "weights": [float(v) for v in w],
                "errors": [float(v) for v in err],
                "sum": total,
                "fractions": [float(v / total) for v in w],
                "r_factor": chi2 / ss_y if ss_y else np.nan,
                "chi2": chi2,
                "redchi": redchi,
                "npts": int(len(yd)),
                "erange": (float(x[0]), float(x[-1])),
                "energy": x,
                "data": yd,
                "fit": fit,
                "parts": {n: A[:, i] * w[i] for i, n in enumerate(combo)},
            })
    _hamilton(fits)
    fits.sort(key=lambda f: f["r_factor"])
    return fits, warn


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


def fits_to_rows(sample: str, fits: list[dict], all_standards: list[str], top_n: int = 5) -> list[dict]:
    """Flatten fits into CSV rows: one row per (sample, rank), one weight/err/fraction column per standard."""
    rows = []
    for rank, f in enumerate(fits[:top_n], start=1):
        row = {
            "sample": sample, "rank": rank, "n_components": len(f["components"]),
            "components": " + ".join(f["components"]),
            "r_factor": f["r_factor"], "chi2": f["chi2"], "reduced_chi2": f["redchi"],
            "sum_weights": f["sum"], "p_vs_simpler": f.get("p_vs_simpler", np.nan), "fit_E_min": f["erange"][0], "fit_E_max": f["erange"][1],
            "n_points": f["npts"],
        }
        lookup = {n: i for i, n in enumerate(f["components"])}
        for s in all_standards:
            i = lookup.get(s)
            row[f"weight[{s}]"] = f["weights"][i] if i is not None else 0.0
            row[f"err[{s}]"] = f["errors"][i] if i is not None else np.nan
            row[f"fraction[{s}]"] = f["fractions"][i] if i is not None else 0.0
        rows.append(row)
    return rows
