"""Readers for XANES scan files (.xdi, SSRL-style .dat, and plain column .dat/.txt/.csv).

Nothing here is element- or beamline-specific beyond recognising the SSRL
"Data:" block layout; any whitespace-delimited column file also works.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

SCAN_EXTENSIONS = (".xdi", ".dat")
_META_RE = re.compile(r"^#\s*([A-Za-z][\w.\-]*)\s*:\s*(.*)$")


@dataclass
class Scan:
    path: str
    name: str                       # file stem, used as the unique key
    columns: dict[str, np.ndarray]  # column name -> values (energy always present)
    meta: dict[str, str] = field(default_factory=dict)
    offsets: dict[str, float] = field(default_factory=dict)  # dark offsets from header, if any

    @property
    def energy(self) -> np.ndarray:
        return self.columns["energy"]

    @property
    def npts(self) -> int:
        return len(self.energy)

    @property
    def colnames(self) -> list[str]:
        return list(self.columns)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _numeric(line: str) -> list[float] | None:
    parts = line.split()
    if not parts:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def _finalize(path, names, data, meta=None, offsets=None) -> Scan:
    data = np.atleast_2d(np.asarray(data, dtype=float))
    if data.shape[1] != len(names):
        names = [f"col{i + 1}" for i in range(data.shape[1])]
    names = [n.strip() for n in names]
    # make names unique
    seen: dict[str, int] = {}
    uniq = []
    for n in names:
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 0
        uniq.append(n)
    cols = {n: data[:, i] for i, n in enumerate(uniq)}
    if "energy" not in cols:
        # pick the first column whose name mentions energy, else the first column
        key = next((n for n in uniq if "energy" in n.lower()), uniq[0])
        cols = {("energy" if n == key else n): v for n, v in cols.items()}
    return Scan(path=path, name=_stem(path), columns=cols, meta=meta or {}, offsets=offsets or {})


def read_xdi(path: str) -> Scan:
    meta: dict[str, str] = {}
    colmap: dict[int, str] = {}
    last_comment = ""
    rows = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                m = _META_RE.match(line.rstrip("\n"))
                if m:
                    key, val = m.group(1), m.group(2).strip()
                    meta[key] = val
                    if key.lower().startswith("column."):
                        try:
                            colmap[int(key.split(".")[1])] = val.split()[0]
                        except (ValueError, IndexError):
                            pass
                stripped = line.lstrip("#").strip()
                if stripped and not stripped.startswith("-") and not stripped.startswith("/"):
                    last_comment = stripped
                continue
            vals = _numeric(line)
            if vals is not None:
                rows.append(vals)
    width = max(len(r) for r in rows) if rows else 0
    rows = [r for r in rows if len(r) == width]
    if colmap and len(colmap) == width:
        names = [colmap[i] for i in sorted(colmap)]
    else:
        names = last_comment.split()
    offsets = {}
    for k, v in meta.items():
        m = re.match(r"^XASSCAN\.(.+)OFFSET$", k)
        if m:
            try:
                offsets[m.group(1)] = float(v)
            except ValueError:
                pass
    return _finalize(path, names, rows, meta, offsets)


_SSRL_RENAME = {"real time clock": "rtc", "requested energy": "energy"}


def read_ssrl_dat(path: str) -> Scan:
    with open(path, errors="replace") as fh:
        lines = fh.read().splitlines()
    meta: dict[str, str] = {}
    offsets_vals: list[float] = []
    names: list[str] = []
    rows = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "Offsets:" and i + 1 < len(lines):
            offsets_vals = _numeric(lines[i + 1]) or []
            i += 2
            continue
        if s == "Data:":
            i += 1
            while i < len(lines) and _numeric(lines[i]) is None:
                if lines[i].strip():
                    names.append(lines[i].strip())
                i += 1
            while i < len(lines):
                vals = _numeric(lines[i])
                if vals is not None:
                    rows.append(vals)
                i += 1
            break
        if ":" in s and not names:
            k, _, v = s.partition(":")
            meta[k.strip()] = v.strip()
        i += 1
    renamed = []
    for n in names:
        new = _SSRL_RENAME.get(n.lower(), n)
        renamed.append(n if new in renamed else new)  # _finalize picks an energy column if none
    width = max(len(r) for r in rows) if rows else 0
    rows = [r for r in rows if len(r) == width]
    offsets = {}
    if offsets_vals and len(offsets_vals) <= len(renamed):
        for n, v in zip(renamed[len(renamed) - len(offsets_vals):], offsets_vals):
            offsets[n] = v
    return _finalize(path, renamed, rows, meta, offsets)


def read_columns(path: str) -> Scan:
    """Generic whitespace/comma-delimited columns; header names from the last comment line."""
    last_comment = ""
    header_line = ""
    rows = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                last_comment = line.lstrip("#").strip()
                continue
            vals = _numeric(line.replace(",", " "))
            if vals is None:
                if not rows and line.strip():
                    header_line = line.strip()
                continue
            rows.append(vals)
    width = max(len(r) for r in rows) if rows else 0
    rows = [r for r in rows if len(r) == width]
    names = (header_line or last_comment).replace(",", " ").split()
    return _finalize(path, names, rows)


def read_scan(path: str) -> Scan:
    with open(path, errors="replace") as fh:
        text = fh.read()
    if text.lstrip().startswith("# XDI") or path.lower().endswith(".xdi"):
        return read_xdi(path)
    if re.search(r"^\s*Data:\s*$", text, re.M):
        return read_ssrl_dat(path)
    return read_columns(path)


def discover(data_dir: str, extensions=SCAN_EXTENSIONS) -> list[str]:
    """List scan files; when the same stem exists as .xdi and .dat, keep only the .xdi."""
    by_stem: dict[str, str] = {}
    for fn in sorted(os.listdir(data_dir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in extensions or fn.startswith("."):
            continue
        prev = by_stem.get(stem)
        if prev is None or ext.lower() == ".xdi":
            by_stem[stem] = os.path.join(data_dir, fn)
    return [by_stem[k] for k in sorted(by_stem, key=str.lower)]


def suggest_group(stem: str, pattern: str = r"_\d+$") -> str:
    """Replicate group = stem with the trailing scan counter removed (e.g. name_001_002 -> name_001)."""
    g = re.sub(pattern, "", stem)
    return g or stem


def match_icr(channels: list[str], names: list[str]) -> dict[str, str]:
    """Pair each fluorescence channel with an input-count-rate column by its trailing number (FF3 -> ICR3)."""
    low = {n.lower(): n for n in names}
    out = {}
    for ch in channels:
        num = re.findall(r"\d+", ch)
        if num and f"icr{num[-1]}" in low:
            out[ch] = low[f"icr{num[-1]}"]
    return out


def detect_columns(scan: Scan) -> dict:
    """Guess column roles generically from names. Everything can be overridden in the app."""
    names = scan.colnames
    low = {n.lower(): n for n in names}
    i0 = low.get("i0") or next((n for n in names if "i0" in n.lower()), None)
    trans = low.get("i1") or low.get("it")
    dwell = low.get("rtc") or low.get("time") or low.get("count_time")
    chans: list[str] = []
    for pat in (r"^ff\d+$", r"^roi\d+$", r"^(ch|chan)\d+$", r"^(if|fl|fluo)\d*$", r"^sca\d+(\.\d+)?$"):
        chans = [n for n in names if re.match(pat, n.lower())]
        if chans:
            break
    icr = match_icr(chans, names)
    mu_col = next((n for n in names if n.lower() in ("mu", "xmu", "norm", "flat", "intensity")), None)
    if chans and i0:
        mode = "fluorescence"
    elif trans and i0:
        mode = "transmission"
    else:
        mode = "column"
    if mode == "column" and mu_col is None:
        mu_col = next((n for n in names if n != "energy"), None)
    return {"mode": mode, "i0": i0, "channels": chans, "icr": icr, "dwell": dwell,
            "trans": trans, "column": mu_col}
