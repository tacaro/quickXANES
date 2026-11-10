"""Session = all parameters (one JSON document) + the pipeline that turns them into outputs.

The Dash app and the headless CLI both drive this class, so a saved params JSON reproduces
every output exactly.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re

import numpy as np
import pandas as pd

from .lcf import fit_to_row, fits_to_rows, lcf_combinatorial, lcf_fit_combo
from .normalize import DEFAULT_NORM, normalize
from .processing import average, compute_signal, find_feature, replicate_deviation_pct
from .readers import Scan, detect_columns, discover, read_scan, suggest_group

ROLES = ("sample", "standard", "calibration")
PARAMS_VERSION = 1


def default_params() -> dict:
    return {
        "version": PARAMS_VERSION,
        "data_dir": "",
        "output_dir": "",
        "group_pattern": r"_\d+$",
        # stem -> {path, group, queued, use_in_avg, channels (None = global)}
        "files": {},
        # group -> role
        "roles": {},
        "signal": {"mode": "fluorescence", "i0": None, "channels": [], "icr": {}, "dwell": None,
                   "trans": None, "column": None, "subtract_offsets": False},
        "deadtime": {"enabled": False, "model": "paralyzable", "tau_us": {}},
        "calibration": {"group": None, "feature": "peak", "reference_eV": None, "window": [None, None],
                        "measured_eV": None, "shift_eV": 0.0},
        "normalization": {"defaults": copy.deepcopy(DEFAULT_NORM), "groups": {}},
        # selected: sample group -> standards of the fit the user locked in (what gets exported)
        "lcf": {"standards": None, "range": [-20.0, 50.0], "max_components": 3, "top_n": 5, "selected": {}},
    }


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("files", "roles", "groups", "icr", "tau_us"):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def safe_name(name: str) -> str:
    return re.sub(r"[^\w.+-]", "_", name.strip())


class Session:
    def __init__(self, params: dict | None = None):
        self.params = _merge(default_params(), params or {})
        self._scans: dict[str, tuple[float, Scan]] = {}
        self.current: str | None = None   # UI cursor (not saved)
        self.view: str = "all"            # UI queue-table view: "all" | "queue" (not saved)
        self.lcf_warnings: list[str] = []   # from the last LCF export
        for f in self.params["files"].values():   # params saved before v0.1.1 kept an include-list
            legacy = f.pop("channels", None)
            if legacy is not None and "exclude_channels" not in f:
                f["exclude_channels"] = [c for c in self.params["signal"]["channels"] if c not in legacy]
            f.setdefault("exclude_channels", [])

    # ------------------------------------------------------------------ params I/O
    @classmethod
    def from_json(cls, path: str) -> "Session":
        with open(path) as fh:
            return cls(json.load(fh))

    def save_params(self, path: str | None = None) -> str:
        path = path or os.path.join(self.out("params"), "quickxanes_params.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(_jsonable(self.params), fh, indent=2)
        return path

    # ------------------------------------------------------------------ files & queue
    def load_dir(self, data_dir: str, output_dir: str | None = None) -> list[str]:
        p = self.params
        p["data_dir"] = os.path.abspath(os.path.expanduser(data_dir))
        if output_dir:
            p["output_dir"] = os.path.abspath(os.path.expanduser(output_dir))
        elif not p["output_dir"]:
            p["output_dir"] = os.path.join(os.path.dirname(p["data_dir"]), "processed")
        stems = []
        for path in discover(p["data_dir"]):
            stem = os.path.splitext(os.path.basename(path))[0]
            stems.append(stem)
            f = p["files"].setdefault(stem, {"group": suggest_group(stem, p["group_pattern"]), "queued": False,
                                             "use_in_avg": True, "exclude_channels": []})
            f["path"] = path
            p["roles"].setdefault(f["group"], "sample")
        if stems and not p["signal"]["i0"] and not p["signal"]["channels"]:
            p["signal"].update(detect_columns(self.scan(stems[0])))
        return stems

    def scan(self, stem: str) -> Scan:
        path = self.params["files"][stem]["path"]
        mtime = os.path.getmtime(path)
        hit = self._scans.get(path)
        if hit is None or hit[0] != mtime:
            hit = (mtime, read_scan(path))
            self._scans[path] = hit
        return hit[1]

    def stems(self) -> list[str]:
        return sorted(self.params["files"], key=str.lower)

    def queue(self) -> list[str]:
        return [s for s in self.stems() if self.params["files"][s].get("queued")]

    def groups(self, queued_only: bool = True) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in (self.queue() if queued_only else self.stems()):
            out.setdefault(self.params["files"][s]["group"], []).append(s)
        return out

    def role(self, group: str) -> str:
        return self.params["roles"].get(group, "sample")

    def regroup(self, stems: list[str] | None = None) -> None:
        for s in stems or self.stems():
            g = suggest_group(s, self.params["group_pattern"])
            self.params["files"][s]["group"] = g
            self.params["roles"].setdefault(g, "sample")

    def calibration_group(self) -> str | None:
        cal = self.params["calibration"].get("group")
        if cal:
            return cal
        return next((g for g in self.groups(queued_only=False) if self.role(g) == "calibration"), None)

    # ------------------------------------------------------------------ step 1-2 data
    def used_channels(self, stem: str) -> list[str]:
        """Default fluorescence channels minus the ones excluded for this file."""
        excl = set(self.params["files"][stem].get("exclude_channels") or [])
        return [c for c in self.params["signal"]["channels"] if c not in excl]

    def signal(self, stem: str, calibrated: bool = True) -> dict:
        p = self.params
        res = compute_signal(self.scan(stem), p["signal"], p["deadtime"], self.used_channels(stem))
        if calibrated:
            res["energy"] = res["energy"] + float(p["calibration"].get("shift_eV") or 0.0)
        return res

    def group_members(self, group: str) -> list[str]:
        """Queued files of a group; all of its files if none is queued (e.g. an un-queued calibration scan)."""
        members = [s for s in self.stems() if self.params["files"][s]["group"] == group]
        return [s for s in members if self.params["files"][s].get("queued")] or members

    def group_data(self, group: str, calibrated: bool = True) -> dict:
        members = self.group_members(group)
        sig = {s: self.signal(s, calibrated) for s in members}
        used = [s for s in members if self.params["files"][s].get("use_in_avg", True)] or members
        avg = average([(sig[s]["energy"], sig[s]["mu"]) for s in used])
        return {"members": members, "used": used, "signals": sig, "avg": avg}

    def calibrate(self) -> dict:
        """Measure the calibration feature on the (uncalibrated) calibration group, set the global shift."""
        cal = self.params["calibration"]
        group = self.calibration_group()
        if not group:
            raise ValueError("no calibration group: set one file's role to 'calibration'")
        if cal.get("reference_eV") in (None, ""):
            raise ValueError("enter the reference energy of the calibration feature")
        avg = self.group_data(group, calibrated=False)["avg"]
        win = cal.get("window") or [None, None]
        measured = find_feature(avg["energy"], avg["mu"], cal.get("feature", "peak"), win)
        cal.update(group=group, measured_eV=measured, shift_eV=float(cal["reference_eV"]) - measured)
        return cal

    # ------------------------------------------------------------------ step 3
    def norm_config(self, group: str) -> dict:
        n = self.params["normalization"]
        return {**DEFAULT_NORM, **n["defaults"], **n["groups"].get(group, {})}

    def normalized(self, group: str) -> dict:
        avg = self.group_data(group)["avg"]
        return normalize(avg["energy"], avg["mu"], self.norm_config(group))

    @staticmethod
    def final_norm(res: dict) -> np.ndarray:
        return res["flat"] if res["config"].get("flatten", True) else res["norm"]

    # ------------------------------------------------------------------ step 4
    def standard_groups(self) -> list[str]:
        return [g for g in self.groups() if self.role(g) == "standard"]

    def sample_groups(self) -> list[str]:
        return [g for g in self.groups() if self.role(g) == "sample"]

    def lcf_standards(self) -> list[str]:
        chosen = self.params["lcf"].get("standards")
        avail = self.standard_groups()
        return avail if chosen is None else [g for g in chosen if g in avail]

    def _lcf_inputs(self, sample: str, cache: dict):
        for g in [sample, *self.lcf_standards()]:
            if g not in cache:
                cache[g] = self.normalized(g)
        s = cache[sample]
        stds = {g: (cache[g]["energy"], self.final_norm(cache[g])) for g in self.lcf_standards()}
        lo, hi = self.params["lcf"]["range"]
        return s, stds, (s["e0"] + float(lo), s["e0"] + float(hi))

    def lcf(self, sample: str, normalized_cache: dict | None = None) -> tuple[dict, list[dict], list[str]]:
        s, stds, erange = self._lcf_inputs(sample, normalized_cache if normalized_cache is not None else {})
        fits, warn = lcf_combinatorial(s["energy"], self.final_norm(s), stds, erange,
                                       int(self.params["lcf"]["max_components"]))
        return s, fits, warn

    # locked-in choice per sample: the standards of the fit the user decided on
    def lcf_selection(self, sample: str) -> list[str] | None:
        return self.params["lcf"].setdefault("selected", {}).get(sample) or None

    def lcf_select(self, sample: str, components: list[str] | None) -> None:
        sel = self.params["lcf"].setdefault("selected", {})
        if components:
            sel[sample] = list(components)
        else:
            sel.pop(sample, None)

    def lcf_selected_fit(self, sample: str, fits: list[dict], cache: dict) -> tuple[dict | None, int | None, list[str]]:
        """(fit, rank among candidates or None, warnings) for the locked-in combination.

        The choice is stored as a set of standards; weights are refit with the current settings, so
        a locked fit outside the candidate list (e.g. more components than max) still resolves."""
        combo = self.lcf_selection(sample)
        if not combo:
            return None, None, []
        for i, f in enumerate(fits):
            if set(f["components"]) == set(combo):
                return f, i + 1, []
        s, stds, erange = self._lcf_inputs(sample, cache)
        f, warn = lcf_fit_combo(s["energy"], self.final_norm(s), stds, erange, combo)
        return f, None, [f"locked fit {' + '.join(combo)}: {w}" for w in warn]

    def lcf_undecided(self) -> list[str]:
        return [g for g in self.sample_groups() if not self.lcf_selection(g)]

    # ------------------------------------------------------------------ outputs
    def out(self, sub: str) -> str:
        root = self.params["output_dir"] or os.path.join(os.getcwd(), "processed")
        path = os.path.join(root, sub)
        os.makedirs(path, exist_ok=True)
        return path

    def write_binned(self) -> list[str]:
        """Step 1 output: energy (as measured) + raw mu and the raw columns used."""
        sig, written = self.params["signal"], []
        for s in self.queue():
            scan = self.scan(s)
            raw = compute_signal(scan, sig, {"enabled": False}, self.used_channels(s))
            df = pd.DataFrame({"energy_eV": raw["energy"], "mu_raw": raw["mu"]})
            used = [sig.get("i0"), sig.get("trans") if sig["mode"] == "transmission" else None,
                    *(raw["channels"] if sig["mode"] == "fluorescence" else []),
                    *[sig["icr"].get(c) for c in raw["channels"]]]
            for c in [u for u in used if u and u in scan.columns]:
                df[c] = scan.columns[c][raw["index"]]
            path = os.path.join(self.out("binned"), safe_name(s) + ".csv")
            df.to_csv(path, index=False)
            written.append(path)
        return written

    def qc_report(self) -> pd.DataFrame:
        rows = []
        shift = float(self.params["calibration"].get("shift_eV") or 0.0)
        for g, stems in self.groups().items():
            gd = self.group_data(g)
            for s in stems:
                sg = gd["signals"][s]
                rows.append({
                    "file": s, "group": g, "role": self.role(g), "npts": len(sg["energy"]),
                    "E_min": float(sg["energy"].min()), "E_max": float(sg["energy"].max()),
                    "used_in_average": s in gd["used"], "energy_shift_eV": shift,
                    "deadtime_max_correction_pct": sg["dt_max_pct"], "deadtime_saturated": sg["saturated"],
                    "rms_dev_from_avg_pct": replicate_deviation_pct(sg["energy"], sg["mu"], gd["avg"]),
                })
        return pd.DataFrame(rows)

    def write_qc(self) -> list[str]:
        """Step 2 output: one averaged, dead-time-corrected, calibrated spectrum per replicate group."""
        written = []
        for g in self.groups():
            avg = self.group_data(g)["avg"]
            df = pd.DataFrame({"energy_eV": avg["energy"], "mu": avg["mu"], "mu_std": avg["std"], "n_scans": avg["n"]})
            path = os.path.join(self.out("qc"), safe_name(g) + ".csv")
            df.to_csv(path, index=False)
            written.append(path)
        rpt = os.path.join(self.out("qc"), "qc_report.csv")
        self.qc_report().to_csv(rpt, index=False)
        return written + [rpt]

    def write_normalized(self) -> list[str]:
        """Step 3 output: normalized/<group>.csv + normalized/normalization_params.json."""
        written, record = [], {}
        for g in self.groups():
            r = self.normalized(g)
            df = pd.DataFrame({"energy_eV": r["energy"], "mu": r["mu"], "pre_edge": r["pre_bkg"],
                               "post_edge": r["post_line"], "norm": r["norm"], "flat": r["flat"]})
            path = os.path.join(self.out("normalized"), safe_name(g) + ".csv")
            df.to_csv(path, index=False)
            written.append(path)
            record[g] = {"config": r["config"], "e0": r["e0"], "edge_step": r["edge_step"], "pre_edge_fit": r["pre"],
                         "post_edge_coef_about_e0": r["post_coef"], "pre_window_eV": r["pre_window"],
                         "post_window_eV": r["post_window"], "warnings": r["warnings"]}
        meta = {"energy_shift_eV": self.params["calibration"].get("shift_eV"),
                "defaults": self.params["normalization"]["defaults"], "groups": record}
        jpath = os.path.join(self.out("normalized"), "normalization_params.json")
        with open(jpath, "w") as fh:
            json.dump(_jsonable(meta), fh, indent=2)
        return written + [jpath]

    def write_lcf(self) -> list[str]:
        """Step 4 output.

        lcf/lcf_results.csv     one row per sample: the fit the user locked in (samples without one are left out)
        lcf/lcf_candidates.csv  top N fits per sample for the record, `selected` marks the locked-in one
        lcf/fits/<sample>_selected.csv  data, fit, residual and weighted components of the locked-in fit
        """
        stds = self.lcf_standards()
        if not stds:
            raise ValueError("no standards: set role 'standard' on at least one queued group")
        cache: dict = {}
        results, candidates, written = [], [], []
        self.lcf_warnings: list[str] = []
        top_n = int(self.params["lcf"]["top_n"])
        for g in self.sample_groups():
            _, fits, _ = self.lcf(g, cache)
            chosen, rank, warn = self.lcf_selected_fit(g, fits, cache)
            self.lcf_warnings += [f"{g}: {w}" for w in warn]
            candidates += fits_to_rows(g, fits, stds, top_n, self.lcf_selection(g))
            if chosen is None:
                continue
            if rank is None or rank > top_n:   # keep the chosen fit in the candidate record too
                candidates.append(fit_to_row(g, chosen, stds, rank=rank, selected=True))
            results.append(fit_to_row(g, chosen, stds, rank_by_r_factor=rank))
            df = pd.DataFrame({"energy_eV": chosen["energy"], "data": chosen["data"], "fit": chosen["fit"],
                               "residual": chosen["data"] - chosen["fit"],
                               **{f"part[{n}]": v for n, v in chosen["parts"].items()}})
            path = os.path.join(self.out(os.path.join("lcf", "fits")), safe_name(g) + "_selected.csv")
            df.to_csv(path, index=False)
            written.append(path)
        rpath = os.path.join(self.out("lcf"), "lcf_results.csv")
        cols = ["sample", "rank_by_r_factor", "n_components", "components", "r_factor", "chi2", "reduced_chi2",
                "sum_weights", "p_vs_simpler", "fit_E_min", "fit_E_max", "n_points"]
        pd.DataFrame(results, columns=None if results else cols).to_csv(rpath, index=False)
        cpath = os.path.join(self.out("lcf"), "lcf_candidates.csv")
        pd.DataFrame(candidates).to_csv(cpath, index=False)
        return [rpath, cpath] + written

    def run_all(self) -> dict[str, list[str]]:
        out = {"binned": self.write_binned(), "qc": self.write_qc(), "normalized": self.write_normalized()}
        if self.standard_groups() and self.sample_groups():
            out["lcf"] = self.write_lcf()
        out["params"] = [self.save_params()]
        return out
