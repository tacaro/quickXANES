"""Dash app: processing queue + four steps (Bin, QC, Normalize, LCF), each with a live Plotly view
of the current file in the queue. Single-user, local: state lives in one Session object."""
from __future__ import annotations

import os
import threading
import time
import traceback

import numpy as np
from dash import ALL, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

from . import plots
from .normalize import PRE_MODELS
from .processing import compute_signal, replicate_deviation_pct
from .readers import match_icr
from .session import ROLES, Session

LOCK = threading.RLock()
NORM_IDS = ["norm-e0", "pre-model", "pre-lo", "pre-hi", "post-deg", "post-lo", "post-hi", "flatten"]
SIG_IDS = ["sig-mode", "sig-i0", "sig-channels", "sig-trans", "sig-column", "sig-dwell", "sig-offsets"]
CAL_IDS = ["cal-group", "cal-feature", "cal-ref", "cal-lo", "cal-hi"]
LCF_IDS = ["lcf-standards", "lcf-lo", "lcf-hi", "lcf-max", "lcf-top"]


# ----------------------------------------------------------------------------- helpers
def _num(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rev() -> float:
    return time.time()


def _status(msg: str, kind: str = "ok"):
    icon = {"ok": "✓", "warn": "⚠", "error": "✕", "info": "•"}[kind]
    return html.Span([html.B(icon + " "), msg], className=f"status-{kind}")


def _field(label, comp, cls="field"):
    return html.Label([html.Span(label, className="label"), comp], className=cls)


def _opts(values):
    return [{"label": v, "value": v} for v in values]


def _numin(id_, value=None, placeholder="", width="6.5em"):
    return dcc.Input(id=id_, type="text", inputMode="decimal", value=value, placeholder=placeholder, debounce=True,
                     style={"width": width})


def _group_of(S: Session, stem):
    return S.params["files"][stem]["group"] if stem in S.params["files"] else None


# group shading (alternating per group) + muted text for files that are not in the queue
GROUP_BAND_STYLE = [{"if": {"filter_query": "{band} = 1"}, "backgroundColor": "#eef4fc"},
                    {"if": {"filter_query": "{q} = 0"}, "color": "#898781"}]


def _file_rows(S: Session):
    """Table rows for the current view. Hidden keys: `band` alternates 0/1 each time the group
    changes (group shading); `q` is 1 for queued files."""
    rows, prev, band = [], object(), 1
    for s in (S.queue() if S.view == "queue" else S.stems()):
        f = S.params["files"][s]
        try:
            n = S.scan(s).npts
        except Exception:  # unreadable file still listed
            n = None
        if f["group"] != prev:
            band, prev = 1 - band, f["group"]
        q = int(bool(f.get("queued")))
        rows.append({"id": s, "file": s, "group": f["group"], "role": S.role(f["group"]), "pts": n,
                     "queued": "✓" if q else "", "q": q, "band": band})
    return rows


def _queue_count(S: Session):
    return f"{len(S.queue())} of {len(S.stems())} files in the queue"


def _set_role(S: Session, group: str, role: str):
    S.params["roles"][group] = role
    lcf = S.params["lcf"]
    if role == "standard" and lcf.get("standards") is not None and group not in lcf["standards"]:
        lcf["standards"].append(group)
    if role == "calibration":
        S.params["calibration"]["group"] = group
    elif S.params["calibration"].get("group") == group:
        S.params["calibration"]["group"] = None


def _columns(S: Session):
    stem = S.current if S.current in S.params["files"] else (S.stems()[0] if S.stems() else None)
    try:
        return S.scan(stem).colnames if stem else []
    except Exception:
        return []


# ----------------------------------------------------------------------------- layout
def _queue_panel(S: Session):
    rows = _file_rows(S)
    return html.Aside(className="queue card", children=[
        html.H2("Processing queue"),
        html.P("Tick rows to select them, then add them to or remove them from the queue (✓). Only queued "
               "files go into QC, normalization and LCF. Click a row to plot it. Replicates share a group; "
               "role marks samples, standards and the calibration scan.", className="hint"),
        html.Div(className="row", children=[
            html.Button("Add selected to queue", id="queue-add", className="primary"),
            html.Button("Remove selected from queue", id="queue-remove"),
            html.Button("Add all shown", id="queue-shown", title="Queue every row matching the table filter"),
            html.Button("Clear queue", id="queue-clear"),
        ]),
        html.Div(className="row", children=[
            dcc.RadioItems(id="queue-view", value=S.view, inline=True,
                           options=[{"label": "all files", "value": "all"}, {"label": "queue only", "value": "queue"}]),
            html.Span(_queue_count(S), id="queue-count", className="hint"),
        ]),
        html.Div(className="row", children=[
            _field("Replicate suffix (regex)", dcc.Input(id="group-pattern", value=S.params["group_pattern"],
                                                          debounce=True, style={"width": "7em"}), "field inline"),
            html.Button("Auto-group", id="regroup", title="Re-derive groups for queued files from the regex"),
        ]),
        dash_table.DataTable(
            id="files", data=rows, selected_rows=[], selected_row_ids=[], row_selectable="multi",
            columns=[{"name": "file", "id": "file", "editable": False},
                     {"name": "group", "id": "group", "editable": True},
                     {"name": "role", "id": "role", "editable": True, "presentation": "dropdown"},
                     {"name": "✓", "id": "queued", "editable": False},
                     {"name": "pts", "id": "pts", "editable": False, "type": "numeric"}],
            dropdown={"role": {"options": _opts(ROLES), "clearable": False}},
            filter_action="native", page_action="none", sort_action="none",
            style_table={"height": "52vh", "overflowY": "auto", "overflowX": "auto"},
            style_cell_conditional=[{"if": {"column_id": "file"}, "width": "37%"},
                                    {"if": {"column_id": "group"}, "width": "28%"},
                                    {"if": {"column_id": "role"}, "width": "20%", "minWidth": "100px"},
                                    {"if": {"column_id": "queued"}, "width": "5%", "minWidth": "22px",
                                     "textAlign": "center"},
                                    {"if": {"column_id": "pts"}, "width": "9%", "minWidth": "38px",
                                     "wordBreak": "normal", "whiteSpace": "nowrap"}],
            # wrap long names: the replicate suffix at the end is what tells rows apart
            style_cell={"fontFamily": "inherit", "fontSize": "12px", "padding": "3px 6px", "textAlign": "left",
                        "maxWidth": 0, "whiteSpace": "normal", "wordBreak": "break-all", "height": "auto",
                        "backgroundColor": "#fcfcfb", "color": "#0b0b0b", "border": "1px solid #e1e0d9"},
            style_header={"fontWeight": 600, "backgroundColor": "#f3f2ee", "color": "#52514e"},
            style_data_conditional=GROUP_BAND_STYLE,
            css=[{"selector": ".Select-menu-outer", "rule": "display: block !important"}],
        ),
        html.Div(className="nav", children=[
            html.Button("◀ Prev", id="prev"),
            html.Div(id="current-label", className="current"),
            html.Button("Next ▶", id="next"),
        ]),
        _field("Role of current group",
               dcc.RadioItems(id="role-radio", options=_opts(ROLES), value=None, inline=True), "field"),
    ])


def _bin_panel():
    return html.Div(id="panel-bin", className="panel", children=[
        html.Div(className="controls card", children=[
            _field("Signal", dcc.RadioItems(id="sig-mode", options=_opts(["fluorescence", "transmission", "column"]),
                                            inline=True)),
            _field("I0", dcc.Dropdown(id="sig-i0", clearable=False, style={"width": "9em"})),
            _field("Fluorescence channels (default for all files)",
                   dcc.Dropdown(id="sig-channels", multi=True, style={"minWidth": "16em"})),
            _field("Transmission (I1)", dcc.Dropdown(id="sig-trans", style={"width": "9em"})),
            _field("μ column (column mode)", dcc.Dropdown(id="sig-column", style={"width": "9em"})),
            _field("Dwell column", dcc.Dropdown(id="sig-dwell", style={"width": "9em"})),
            dcc.Checklist(id="sig-offsets", options=[{"label": " subtract header dark offsets", "value": "on"}]),
        ]),
        html.Div(id="icr-map", className="hint"),
        html.Div(className="row card", children=[
            _field("Channels used for this file (untick a channel with spurious data)",
                   dcc.Checklist(id="file-channels", inline=True)),
            html.Span(id="file-channels-note", className="hint"),
        ]),
        dcc.Loading(dcc.Graph(id="bin-graph", config={"displaylogo": False}), type="dot", delay_show=400),
        html.Div(className="row", children=[
            html.Button("Write binned files (queue)", id="write-binned", className="primary"),
            html.Span("→ processed/binned/<file>.csv: energy, μ (raw), and the raw columns used", className="hint"),
        ]),
    ])


def _qc_panel():
    return html.Div(id="panel-qc", className="panel", hidden=True, children=[
        html.Div(className="grid2", children=[
            html.Div(className="card", children=[
                html.H3("Dead-time (ICR + τ per channel)"),
                html.Div(className="row", children=[
                    dcc.Checklist(id="dt-enabled", options=[{"label": " apply correction", "value": "on"}]),
                    dcc.RadioItems(id="dt-model", options=_opts(["paralyzable", "non-paralyzable"]), inline=True),
                ]),
                html.Div(id="tau-inputs", className="row"),
                html.P("τ in μs. Channel counts are scaled by ICR_true / ICR_measured. τ = 0 leaves a channel "
                       "uncorrected.", className="hint"),
                html.H3("Replicates averaged in this group"),
                dcc.Checklist(id="rep-used", className="vlist"),
            ]),
            html.Div(className="card", children=[
                html.H3("Energy calibration (one global shift)"),
                html.Div(className="row", children=[
                    _field("Calibration group", dcc.Dropdown(id="cal-group", style={"minWidth": "14em"})),
                    _field("Feature", dcc.RadioItems(id="cal-feature",
                                                     options=[{"label": "peak max", "value": "peak"},
                                                              {"label": "1st-derivative max", "value": "derivative"}],
                                                     inline=True)),
                ]),
                html.Div(className="row", children=[
                    _field("Reference energy (eV)", _numin("cal-ref", width="8em")),
                    _field("Search window (eV)", html.Div([_numin("cal-lo", placeholder="auto"), " – ",
                                                           _numin("cal-hi", placeholder="auto")])),
                    html.Button("Compute shift", id="cal-compute", className="primary"),
                    _field("ΔE applied (eV)", _numin("cal-shift", width="7em")),
                ]),
                html.Div(id="cal-readout", className="readout"),
                dcc.Graph(id="cal-graph", config={"displaylogo": False}),
            ]),
        ]),
        dcc.Loading(dcc.Graph(id="qc-graph", config={"displaylogo": False}), type="dot", delay_show=400),
        dash_table.DataTable(id="qc-table", columns=[{"name": c, "id": c} for c in
                                                     ["file", "pts", "E range (eV)", "channels", "averaged", "max DT corr. %",
                                                      "RMS dev. from avg %", "flags"]],
                             style_cell={"fontFamily": "inherit", "fontSize": "12px", "padding": "3px 6px",
                                         "textAlign": "left", "border": "1px solid #e1e0d9"},
                             style_header={"fontWeight": 600, "backgroundColor": "#f3f2ee", "color": "#52514e"}),
        html.Div(className="row", children=[
            html.Button("Write averaged spectra + QC report", id="write-qc", className="primary"),
            html.Span("→ processed/qc/<group>.csv and qc_report.csv", className="hint"),
        ]),
    ])


def _norm_panel():
    return html.Div(id="panel-norm", className="panel", hidden=True, children=[
        html.Div(className="controls card", children=[
            _field("E0 (eV)", _numin("norm-e0", placeholder="auto", width="8em")),
            _field("Pre-edge model", dcc.Dropdown(id="pre-model", options=_opts(PRE_MODELS), clearable=False,
                                                  style={"width": "9em"})),
            _field("Pre-edge window (eV rel. E0)", html.Div([_numin("pre-lo", placeholder="start"), " – ",
                                                             _numin("pre-hi")])),
            _field("Post-edge polynomial", dcc.RadioItems(id="post-deg", options=[0, 1, 2], inline=True)),
            _field("Post-edge window (eV rel. E0)", html.Div([_numin("post-lo"), " – ",
                                                              _numin("post-hi", placeholder="end")])),
            dcc.Checklist(id="flatten", options=[{"label": " flatten post-edge", "value": "on"}]),
        ]),
        html.Div(className="row", children=[
            html.Button("Apply these settings to all groups", id="norm-apply-all",
                        title="Make these the defaults and clear per-group overrides (manual E0s are kept)"),
            html.Button("Reset this group to defaults", id="norm-reset"),
            html.Div(id="norm-readout", className="readout"),
        ]),
        dcc.Loading(dcc.Graph(id="norm-graph", config={"displaylogo": False}), type="dot", delay_show=400),
        html.Div(className="row", children=[
            html.Button("Write normalized spectra + parameters JSON", id="write-norm", className="primary"),
            html.Span("→ processed/normalized/<group>.csv and normalization_params.json", className="hint"),
        ]),
    ])


def _lcf_panel():
    return html.Div(id="panel-lcf", className="panel", hidden=True, children=[
        html.Div(className="controls card", children=[
            _field("Standards (groups with role 'standard')", dcc.Checklist(id="lcf-standards", className="vlist")),
            _field("Fit range (eV rel. sample E0)", html.Div([_numin("lcf-lo"), " – ", _numin("lcf-hi")])),
            _field("Max components", _numin("lcf-max", width="4.5em")),
            _field("Fits kept per sample in CSV", _numin("lcf-top", width="4.5em")),
        ]),
        html.P("Every subset of the selected standards is fit with weights ≥ 0; the sum is not forced to 1 "
               "(a sum far from 1 flags a missing standard or a normalization problem). Ranked by R-factor, which "
               "always improves with more standards; 'p vs simpler' tests whether the extra one is justified.",
               className="hint"),
        html.Div(id="lcf-readout", className="readout"),
        dash_table.DataTable(id="lcf-table", row_selectable="single", selected_rows=[0],
                             columns=[{"name": c, "id": c} for c in
                                      ["rank", "components", "fractions", "weights ± err", "sum", "R-factor",
                                       "reduced χ²", "p vs simpler"]],
                             tooltip_header={"p vs simpler": "F-test (Hamilton) p-value that the last-added standard "
                                                             "is justified vs. the best fit with one fewer. Small = "
                                                             "justified. Optimistic: XANES points are correlated."},
                             style_cell={"fontFamily": "inherit", "fontSize": "12px", "padding": "3px 6px",
                                         "textAlign": "left", "border": "1px solid #e1e0d9",
                                         "whiteSpace": "normal", "height": "auto"},
                             style_header={"fontWeight": 600, "backgroundColor": "#f3f2ee", "color": "#52514e"},
                             page_size=10),
        dcc.Loading(dcc.Graph(id="lcf-graph", config={"displaylogo": False}), type="dot", delay_show=400),
        html.Div(className="row", children=[
            html.Button("Fit all samples + write CSV", id="write-lcf", className="primary"),
            html.Span("→ processed/lcf/lcf_results.csv and lcf/fits/<sample>_best.csv", className="hint"),
        ]),
    ])


def build_layout(S: Session):
    p = S.params
    params_path = os.path.join(p["output_dir"], "params", "quickxanes_params.json") if p["output_dir"] else ""
    return html.Div(className="app", children=[
        dcc.Store(id="rev", data=_rev()),
        dcc.Store(id="loaded", data=_rev()),
        dcc.Store(id="current", data=S.current),
        html.Header(className="topbar", children=[
            html.Div("quickXANES", className="brand"),
            _field("Data folder", dcc.Input(id="data-dir", value=p["data_dir"], className="path")),
            html.Button("Load folder", id="load-dir", className="primary"),
            _field("Output folder", dcc.Input(id="out-dir", value=p["output_dir"], className="path", debounce=True)),
            _field("Params JSON", dcc.Input(id="params-path", value=params_path, className="path")),
            html.Button("Load params", id="load-params"),
            html.Button("Save params", id="save-params"),
        ]),
        html.Div(id="status", className="status"),
        html.Div(className="body", children=[
            _queue_panel(S),
            html.Main(className="steps", children=[
                dcc.Tabs(id="tabs", value="bin", className="tabs", children=[
                    dcc.Tab(label="1 · Bin", value="bin"), dcc.Tab(label="2 · QC", value="qc"),
                    dcc.Tab(label="3 · Normalize", value="norm"), dcc.Tab(label="4 · LCF", value="lcf")]),
                _bin_panel(), _qc_panel(), _norm_panel(), _lcf_panel(),
            ]),
        ]),
    ])


# ----------------------------------------------------------------------------- callbacks
def register(app: Dash, S: Session):

    def fail(where: str, exc: Exception):
        traceback.print_exc()
        return _status(f"{where}: {exc}", "error")

    # --- tab visibility
    @app.callback([Output(f"panel-{k}", "hidden") for k in ("bin", "qc", "norm", "lcf")], Input("tabs", "value"))
    def show_tab(tab):
        return [tab != k for k in ("bin", "qc", "norm", "lcf")]

    # --- load folder / load params / save params
    @app.callback(Output("files", "data", allow_duplicate=True), Output("files", "selected_rows", allow_duplicate=True),
                  Output("out-dir", "value"), Output("params-path", "value"), Output("data-dir", "value"),
                  Output("loaded", "data"), Output("status", "children", allow_duplicate=True),
                  Output("files", "selected_row_ids", allow_duplicate=True),
                  Output("queue-count", "children", allow_duplicate=True),
                  Input("load-dir", "n_clicks"), Input("load-params", "n_clicks"),
                  State("data-dir", "value"), State("out-dir", "value"), State("params-path", "value"),
                  prevent_initial_call=True)
    def load(_d, _p, data_dir, out_dir, params_path):
        with LOCK:
            try:
                if ctx.triggered_id == "load-params":
                    fresh = Session.from_json(params_path)
                    S.params, S._scans = fresh.params, {}
                    if S.params["data_dir"] and os.path.isdir(S.params["data_dir"]):
                        S.load_dir(S.params["data_dir"])
                    msg = f"Loaded parameters from {params_path}"
                else:
                    if not data_dir or not os.path.isdir(os.path.expanduser(data_dir)):
                        return [no_update] * 6 + [_status(f"not a folder: {data_dir}", "error"), no_update, no_update]
                    new_dir = os.path.abspath(os.path.expanduser(data_dir))
                    if new_dir != S.params["data_dir"]:
                        S.params.update(files={}, roles={}, output_dir="")
                        S.params["calibration"].update(group=None, measured_eV=None)
                        S.current = None
                    n = len(S.load_dir(new_dir))
                    msg = f"Found {n} scans (.xdi preferred over duplicate .dat) in {S.params['data_dir']}"
                q = S.queue() or S.stems()
                S.current = S.current if S.current in S.params["files"] else (q[0] if q else None)
                pp = os.path.join(S.params["output_dir"], "params", "quickxanes_params.json")
                return (_file_rows(S), [], S.params["output_dir"], pp, S.params["data_dir"], _rev(), _status(msg),
                        [], _queue_count(S))
            except Exception as exc:
                return [no_update] * 6 + [fail("Load", exc), no_update, no_update]

    @app.callback(Output("status", "children", allow_duplicate=True), Input("save-params", "n_clicks"),
                  State("params-path", "value"), prevent_initial_call=True)
    def save(_n, path):
        with LOCK:
            try:
                return _status(f"Saved parameters → {S.save_params(path or None)}")
            except Exception as exc:
                return fail("Save params", exc)

    @app.callback(Output("rev", "data", allow_duplicate=True), Input("out-dir", "value"), prevent_initial_call=True)
    def set_out(out_dir):
        if out_dir:
            S.params["output_dir"] = os.path.abspath(os.path.expanduser(out_dir))
        return no_update

    # --- queue table: group/role edits, queue buttons
    @app.callback(Output("files", "data", allow_duplicate=True), Output("rev", "data", allow_duplicate=True),
                  Input("files", "data_timestamp"), State("files", "data"),
                  State("files", "data_previous"), prevent_initial_call=True)
    def table_edit(_ts, rows, previous):
        with LOCK:
            if not rows:
                return no_update, no_update
            # a cell was edited: apply only the cells that changed, then refresh sibling rows
            prev = {r["id"]: r for r in (previous or [])}
            for r in rows:
                f, old = S.params["files"].get(r["id"]), prev.get(r["id"])
                if f is None or old is None:
                    continue
                new_group = (r.get("group") or "").strip()
                if new_group and new_group != old.get("group"):
                    S.params["roles"].setdefault(new_group, S.role(f["group"]))
                    f["group"] = new_group
                if r.get("role") in ROLES and r.get("role") != old.get("role"):
                    _set_role(S, f["group"], r["role"])
            return _file_rows(S), _rev()

    @app.callback(Output("files", "data", allow_duplicate=True), Output("files", "selected_rows", allow_duplicate=True),
                  Output("files", "selected_row_ids", allow_duplicate=True), Output("queue-count", "children"),
                  Output("rev", "data", allow_duplicate=True), Output("status", "children", allow_duplicate=True),
                  Input("queue-add", "n_clicks"), Input("queue-remove", "n_clicks"), Input("queue-shown", "n_clicks"),
                  Input("queue-clear", "n_clicks"), Input("regroup", "n_clicks"), Input("queue-view", "value"),
                  State("files", "selected_row_ids"), State("files", "derived_virtual_row_ids"),
                  State("group-pattern", "value"), prevent_initial_call=True)
    def queue_buttons(_a, _r, _s, _c, _g, view, selected, shown, pattern):
        with LOCK:
            trig = ctx.triggered_id
            files = S.params["files"]
            selected = [s for s in (selected or []) if s in files]
            msg = no_update
            try:
                if trig in ("queue-add", "queue-remove"):
                    if not selected:
                        return [no_update] * 5 + [_status("Select rows first (tick their boxes).", "warn")]
                    for s in selected:
                        files[s]["queued"] = trig == "queue-add"
                    verb = "Added" if trig == "queue-add" else "Removed"
                    msg = _status(f"{verb} {len(selected)} file(s) {'to' if trig == 'queue-add' else 'from'} the queue")
                elif trig == "queue-shown":
                    for s in (shown or []):
                        if s in files:
                            files[s]["queued"] = True
                    msg = _status(f"Added {len(shown or [])} shown file(s) to the queue")
                elif trig == "queue-clear":
                    for f in files.values():
                        f["queued"] = False
                    msg = _status("Queue cleared", "info")
                elif trig == "regroup":
                    S.params["group_pattern"] = pattern or r"_\d+$"
                    S.regroup(S.queue() or None)
                    msg = _status(f"Re-grouped {'queued' if S.queue() else 'all'} files "
                                  f"with suffix pattern {S.params['group_pattern']!r}")
                elif trig == "queue-view":
                    S.view = view or "all"
                return _file_rows(S), [], [], _queue_count(S), _rev(), msg
            except Exception as exc:
                return [no_update] * 5 + [fail("Queue", exc)]

    # --- current file cursor
    @app.callback(Output("current", "data"), Output("current-label", "children"),
                  Output("files", "style_data_conditional"), Output("role-radio", "value"),
                  Output("rev", "data", allow_duplicate=True),
                  Input("files", "active_cell"), Input("prev", "n_clicks"), Input("next", "n_clicks"),
                  Input("loaded", "data"), State("current", "data"), prevent_initial_call="initial_duplicate")
    def set_current(cell, _p, _n, _l, cur):
        with LOCK:
            trig = ctx.triggered_id
            q = S.queue() or S.stems()
            if trig == "files" and cell and cell.get("row_id"):
                cur = cell["row_id"]
            elif trig in ("prev", "next") and q:
                cur = q[(q.index(cur) + (1 if trig == "next" else -1)) % len(q)] if cur in q else q[0]
            if cur not in S.params["files"]:
                cur = q[0] if q else None
            S.current = cur
            style = GROUP_BAND_STYLE + [{"if": {"state": "active"}, "backgroundColor": "#cde2fb",
                                             "border": "1px solid #2a78d6"}]
            if cur:
                style.append({"if": {"filter_query": f'{{id}} = "{cur}"'}, "backgroundColor": "#cde2fb",
                              "fontWeight": 600})
            pos = f" ({q.index(cur) + 1}/{len(q)} in queue)" if cur in q else " (not queued)"
            label = [html.B(cur or "—"), html.Span(pos if cur else "", className="hint")]
            role = S.role(_group_of(S, cur)) if cur else None
            return cur, label, style, role, _rev()

    @app.callback(Output("files", "data", allow_duplicate=True), Output("rev", "data", allow_duplicate=True),
                  Input("role-radio", "value"), State("current", "data"), prevent_initial_call=True)
    def role_radio(role, cur):
        with LOCK:
            g = _group_of(S, cur)
            if not g or role not in ROLES or role == S.role(g):
                return no_update, no_update
            _set_role(S, g, role)
            return _file_rows(S), _rev()

    # --- step 1: signal settings
    @app.callback([Output(i, "options") for i in ("sig-i0", "sig-channels", "sig-trans", "sig-column", "sig-dwell")]
                  + [Output(i, "value") for i in SIG_IDS]
                  + [Output("dt-enabled", "value"), Output("dt-model", "value")]
                  + [Output(i, "value") for i in CAL_IDS] + [Output("cal-shift", "value")]
                  + [Output(i, "value") for i in ["lcf-lo", "lcf-hi", "lcf-max", "lcf-top"]],
                  Input("loaded", "data"))
    def sync_controls(_l):
        with LOCK:
            cols = _columns(S)
            o = _opts(cols)
            sig, dt, cal, lcf = (S.params[k] for k in ("signal", "deadtime", "calibration", "lcf"))
            win = cal.get("window") or [None, None]
            return ([o, o, o, o, o]
                    + [sig["mode"], sig["i0"], sig["channels"], sig["trans"], sig["column"], sig["dwell"],
                       ["on"] if sig.get("subtract_offsets") else []]
                    + [["on"] if dt["enabled"] else [], dt["model"]]
                    + [S.calibration_group(), cal["feature"], cal["reference_eV"], win[0], win[1], round(float(cal["shift_eV"] or 0), 4)]
                    + [lcf["range"][0], lcf["range"][1], lcf["max_components"], lcf["top_n"]])

    @app.callback(Output("rev", "data", allow_duplicate=True), Output("icr-map", "children"),
                  [Input(i, "value") for i in SIG_IDS], prevent_initial_call=True)
    def set_signal(mode, i0, chans, trans, column, dwell, offsets):
        with LOCK:
            sig = S.params["signal"]
            chans = list(chans or [])
            sig.update(mode=mode or "fluorescence", i0=i0, channels=chans, trans=trans, column=column, dwell=dwell,
                       subtract_offsets=bool(offsets), icr=match_icr(chans, _columns(S)))
            m = ", ".join(f"{k}→{v}" for k, v in sig["icr"].items()) or "none found"
            return _rev(), f"Dead-time ICR pairing: {m}"

    @app.callback(Output("bin-graph", "figure"), Input("rev", "data"), Input("tabs", "value"),
                  State("current", "data"))
    def draw_bin(_r, tab, cur):
        if tab != "bin":
            return no_update
        with LOCK:
            if not cur:
                return plots.message("Load a data folder, queue files and click one to plot it.")
            try:
                sig = S.params["signal"]
                scan = S.scan(cur)
                allc = [c for c in sig["channels"] if c in scan.columns]
                used = [c for c in S.used_channels(cur) if c in scan.columns]
                raw_all = compute_signal(scan, sig, {"enabled": False}, allc)
                raw = compute_signal(scan, sig, {"enabled": False}, used)
                raw_all["mu"] = raw["mu"]
                return plots.fig_bin(cur, raw_all, sig["mode"], allc, used)
            except Exception as exc:
                traceback.print_exc()
                return plots.message(f"Can't build μ(E) for {cur}: {exc}")

    # --- step 2: dead-time
    @app.callback(Output("tau-inputs", "children"), Input("sig-channels", "value"), Input("loaded", "data"))
    def tau_inputs(chans, _l):
        tau = S.params["deadtime"].get("tau_us", {})
        return [_field(f"τ {ch}", dcc.Input(id={"type": "tau", "ch": ch}, type="text", inputMode="decimal",
                                            value=tau.get(ch, 0), debounce=True, style={"width": "5.5em"}),
                       "field inline") for ch in (chans or [])]

    @app.callback(Output("rev", "data", allow_duplicate=True), Input("dt-enabled", "value"),
                  Input("dt-model", "value"), Input({"type": "tau", "ch": ALL}, "value"), prevent_initial_call=True)
    def set_deadtime(enabled, model, taus):
        with LOCK:
            dt = S.params["deadtime"]
            dt["enabled"] = bool(enabled)
            dt["model"] = model or "paralyzable"
            for spec, v in zip(ctx.inputs_list[2], taus):
                dt.setdefault("tau_us", {})[spec["id"]["ch"]] = _num(v) or 0.0
            return _rev()

    # --- per-file channel exclusion (step 1) and replicate inclusion (step 2):
    #     load on cursor/group change, save on edit
    def _channel_note(f):
        excl = f.get("exclude_channels") or []
        return f"excluded for this file: {', '.join(excl)}" if excl else "all default channels used"

    @app.callback(Output("file-channels", "options"), Output("file-channels", "value"),
                  Output("file-channels-note", "children"),
                  Output("rep-used", "options"), Output("rep-used", "value"), Output("rev", "data", allow_duplicate=True),
                  Output("status", "children", allow_duplicate=True),
                  Input("current", "data"), Input("files", "data"), Input("sig-channels", "value"),
                  Input("file-channels", "value"), Input("rep-used", "value"), prevent_initial_call=True)
    def file_qc_controls(cur, _rows, chans, file_ch, rep_used):
        with LOCK:
            if not cur or cur not in S.params["files"]:
                return [], [], "", [], [], no_update, no_update
            f = S.params["files"][cur]
            members = S.group_members(f["group"])
            chans = list(chans or [])
            trig = ctx.triggered_id
            keep = [no_update] * 7
            if trig == "file-channels":
                chosen = [c for c in chans if c in (file_ch or [])]
                if chans and not chosen:   # refuse to drop the last channel
                    return (no_update, S.used_channels(cur), no_update, no_update, no_update, no_update,
                            _status("Keep at least one channel for this file.", "warn"))
                f["exclude_channels"] = [c for c in chans if c not in chosen]
                return no_update, no_update, _channel_note(f), no_update, no_update, _rev(), no_update
            if trig == "rep-used":
                for s in members:
                    S.params["files"][s]["use_in_avg"] = s in (rep_used or [])
                return keep[:5] + [_rev(), no_update]
            reps = [s for s in members if S.params["files"][s].get("use_in_avg", True)]
            return (_opts(chans), S.used_channels(cur), _channel_note(f), _opts(members), reps, no_update,
                    no_update)

    # --- step 2: calibration
    @app.callback(Output("cal-group", "options"), Output("cal-group", "value", allow_duplicate=True),
                  Output("lcf-standards", "options"), Output("lcf-standards", "value"),
                  Input("files", "data"), Input("loaded", "data"), prevent_initial_call="initial_duplicate")
    def refresh_group_lists(_rows, _l):
        with LOCK:
            groups = list(S.groups(queued_only=False))
            stds = [g for g in groups if S.role(g) == "standard"]
            chosen = S.params["lcf"].get("standards")
            value = stds if chosen is None else [g for g in chosen if g in stds]
            return _opts(groups), S.calibration_group(), _opts(stds), value

    @app.callback(Output("cal-shift", "value", allow_duplicate=True), Output("cal-readout", "children"),
                  Output("rev", "data", allow_duplicate=True), Output("status", "children", allow_duplicate=True),
                  [Input(i, "value") for i in CAL_IDS] + [Input("cal-shift", "value"), Input("cal-compute", "n_clicks")],
                  prevent_initial_call=True)
    def calibration(group, feature, ref, lo, hi, shift, _c):
        with LOCK:
            cal = S.params["calibration"]
            trig = ctx.triggered_id
            if trig == "cal-shift":
                if abs((_num(shift) or 0.0) - float(cal.get("shift_eV") or 0.0)) < 5e-5:
                    return no_update, _cal_text(cal), no_update, no_update  # echo of a computed value
                cal["shift_eV"] = _num(shift) or 0.0
                return no_update, _cal_text(cal, manual=True), _rev(), no_update
            cal.update(group=group, feature=feature or "peak", reference_eV=_num(ref), window=[_num(lo), _num(hi)])
            if trig != "cal-compute":
                return no_update, _cal_text(cal), _rev(), no_update
            try:
                S.calibrate()
                return round(cal["shift_eV"], 4), _cal_text(cal), _rev(), _status(
                    f"Calibration: {cal['group']} {cal['feature']} at {cal['measured_eV']:.3f} eV → "
                    f"ΔE = {cal['shift_eV']:+.3f} eV applied to every scan")
            except Exception as exc:
                return no_update, no_update, no_update, fail("Calibration", exc)

    def _cal_text(cal, manual=False):
        meas = cal.get("measured_eV")
        parts = [f"ΔE = {float(cal.get('shift_eV') or 0):+.3f} eV"]
        if meas is not None:
            parts.append(f"measured {meas:.3f} eV")
        if cal.get("reference_eV") is not None:
            parts.append(f"reference {cal['reference_eV']:.3f} eV")
        if manual:
            parts.append("(manual)")
        return " · ".join(parts)

    @app.callback(Output("qc-graph", "figure"), Output("cal-graph", "figure"), Output("qc-table", "data"),
                  Input("rev", "data"), Input("tabs", "value"), State("current", "data"))
    def draw_qc(_r, tab, cur):
        if tab != "qc":
            return no_update, no_update, no_update
        with LOCK:
            cal_fig = plots.message("Set a group's role to 'calibration' (or pick one above).", 300)
            cg = S.calibration_group()
            if cg:
                try:
                    a = S.group_data(cg, calibrated=False)["avg"]
                    cal_fig = plots.fig_calibration(cg, a["energy"], a["mu"], S.params["calibration"])
                except Exception as exc:
                    cal_fig = plots.message(f"calibration: {exc}", 300)
            if not cur:
                return plots.message("No file selected."), cal_fig, []
            g = _group_of(S, cur)
            try:
                gd = S.group_data(g)
                q_pts = [S.scan(s).npts for s in (S.queue() or [cur])]
                typical = float(np.median(q_pts))
                rows = []
                for s in gd["members"]:
                    sg = gd["signals"][s]
                    flags = []
                    if len(sg["energy"]) < 0.8 * typical:
                        flags.append("⚠ short scan")
                    if sg["saturated"]:
                        flags.append("⚠ dead-time beyond model range")
                    rows.append({"file": s, "pts": len(sg["energy"]),
                                 "E range (eV)": f"{sg['energy'][0]:.1f} – {sg['energy'][-1]:.1f}",
                                 "channels": ("all but " + ", ".join(S.params["files"][s]["exclude_channels"])
                                              if S.params["files"][s].get("exclude_channels") else "all"),
                                 "averaged": "yes" if s in gd["used"] else "no",
                                 "max DT corr. %": round(sg["dt_max_pct"], 2),
                                 "RMS dev. from avg %": round(replicate_deviation_pct(sg["energy"], sg["mu"],
                                                                                      gd["avg"]), 2),
                                 "flags": " ".join(flags)})
                return plots.fig_group(g, gd, cur), cal_fig, rows
            except Exception as exc:
                traceback.print_exc()
                return plots.message(f"QC for {g}: {exc}"), cal_fig, []

    # --- step 3: normalization controls (load on cursor change, save on edit)
    def _cfg_to_controls(cfg):
        return [cfg.get("e0"), cfg["pre_model"], cfg["pre_range"][0], cfg["pre_range"][1], cfg["post_degree"],
                cfg["post_range"][0], cfg["post_range"][1], ["on"] if cfg.get("flatten", True) else []]

    def _controls_to_cfg(vals):
        e0, model, plo, phi, deg, qlo, qhi, flat = vals
        return {"e0": _num(e0), "pre_model": model or "gaussian", "pre_range": [_num(plo), _num(phi)],
                "post_degree": int(deg if deg is not None else 1), "post_range": [_num(qlo), _num(qhi)],
                "flatten": bool(flat)}

    @app.callback([Output(i, "value") for i in NORM_IDS]
                  + [Output("rev", "data", allow_duplicate=True), Output("status", "children", allow_duplicate=True)],
                  [Input("current", "data"), Input("loaded", "data"), Input("norm-apply-all", "n_clicks"),
                   Input("norm-reset", "n_clicks")] + [Input(i, "value") for i in NORM_IDS],
                  prevent_initial_call="initial_duplicate")
    def norm_controls(cur, _l, _a, _r, *vals):
        with LOCK:
            g = _group_of(S, cur)
            trig = ctx.triggered_id
            nrm = S.params["normalization"]
            keep = [no_update] * len(NORM_IDS)
            if not g:
                return keep + [no_update, no_update]
            if trig in NORM_IDS:
                nrm["groups"][g] = _controls_to_cfg(vals)
                return keep + [_rev(), no_update]
            msg = no_update
            if trig == "norm-apply-all":
                cfg = _controls_to_cfg(vals)
                cfg.pop("e0")
                nrm["defaults"] = cfg
                nrm["groups"] = {k: {"e0": v["e0"]} for k, v in nrm["groups"].items() if v.get("e0") is not None}
                msg = _status("Applied these settings as defaults for every group (manual E0 values kept)")
            elif trig == "norm-reset":
                nrm["groups"].pop(g, None)
                msg = _status(f"{g}: back to default normalization settings", "info")
            return _cfg_to_controls(S.norm_config(g)) + [_rev(), msg]

    @app.callback(Output("norm-graph", "figure"), Output("norm-readout", "children"),
                  Output("norm-e0", "placeholder"), Input("rev", "data"), Input("tabs", "value"),
                  State("current", "data"))
    def draw_norm(_r, tab, cur):
        if tab != "norm":
            return no_update, no_update, no_update
        with LOCK:
            g = _group_of(S, cur)
            if not g:
                return plots.message("No file selected."), "", "auto"
            try:
                r = S.normalized(g)
            except Exception as exc:
                traceback.print_exc()
                return plots.message(f"Normalization of {g} failed: {exc}"), _status(str(exc), "error"), "auto"
            p = r["pre"]
            if p["model"] == "gaussian":
                pre = f"Gaussian tail: c={p['c']:.4g}, A={p['A']:.4g}, Ec={p['Ec']:.1f} eV, σ={p['sigma']:.1f} eV"
            elif p["model"] == "linear":
                pre = f"linear: slope={p['slope']:.4g}, intercept={p['intercept']:.4g}"
            else:
                pre = f"constant: {p['c']:.4g}"
            source = "group-specific settings" if g in S.params["normalization"]["groups"] else "default settings"
            items = [html.Span(f"E0 {r['e0']:.2f} eV ({'manual' if r['config'].get('e0') else 'auto'})"),
                     html.Span(f"edge step {r['edge_step']:.4g}"), html.Span(pre), html.Span(source, className="hint")]
            items += [_status(w, "warn") for w in r["warnings"]]
            return plots.fig_norm(g, r), items, f"auto {r['e0']:.2f}"

    # --- step 4: LCF
    @app.callback(Output("rev", "data", allow_duplicate=True), [Input(i, "value") for i in LCF_IDS],
                  prevent_initial_call=True)
    def set_lcf(stds, lo, hi, mx, top):
        with LOCK:
            lcf = S.params["lcf"]
            if ctx.triggered_id == "lcf-standards":
                lcf["standards"] = list(stds or [])
            lcf["range"] = [_num(lo) if _num(lo) is not None else -20.0, _num(hi) if _num(hi) is not None else 50.0]
            lcf["max_components"] = max(1, int(_num(mx) or 3))
            lcf["top_n"] = max(1, int(_num(top) or 5))
            return _rev()

    @app.callback(Output("lcf-graph", "figure"), Output("lcf-table", "data"), Output("lcf-readout", "children"),
                  Output("lcf-table", "selected_rows"),
                  Input("rev", "data"), Input("tabs", "value"), Input("lcf-table", "selected_rows"),
                  State("current", "data"))
    def draw_lcf(_r, tab, sel, cur):
        if tab != "lcf":
            return no_update, no_update, no_update, no_update
        with LOCK:
            g = _group_of(S, cur)
            if not g:
                return plots.message("No file selected."), [], "", [0]
            if S.role(g) != "sample":
                return (plots.message(f"'{g}' has role '{S.role(g)}'. Select a sample to fit it."), [],
                        _status("LCF runs on groups with role 'sample'.", "info"), [0])
            if not S.lcf_standards():
                return (plots.message("Mark at least one queued group as 'standard'."), [],
                        _status("No standards selected.", "warn"), [0])
            try:
                s, fits, warn = S.lcf(g)
            except Exception as exc:
                traceback.print_exc()
                return plots.message(f"LCF of {g} failed: {exc}"), [], _status(str(exc), "error"), [0]
            new_sample = ctx.triggered_id != "lcf-table"
            i = 0 if new_sample or not sel else min(sel[0], max(len(fits) - 1, 0))
            rows = [{"rank": k + 1, "components": " + ".join(f["components"]),
                     "fractions": ", ".join(f"{100 * v:.1f}%" for v in f["fractions"]),
                     "weights ± err": ", ".join(f"{w:.3f}±{e:.3f}" for w, e in zip(f["weights"], f["errors"])),
                     "sum": round(f["sum"], 3), "R-factor": f"{f['r_factor']:.3e}", "reduced χ²": f"{f['redchi']:.3e}",
                     "p vs simpler": "—" if np.isnan(f["p_vs_simpler"]) else
                     ("<0.001" if f["p_vs_simpler"] < 1e-3 else f"{f['p_vs_simpler']:.3f}")}
                    for k, f in enumerate(fits[:50])]
            fit = fits[i] if fits else None
            lo, hi = S.params["lcf"]["range"]
            msg = [html.Span(f"{g}: {len(fits)} acceptable fits, range E0{lo:+g} to E0{hi:+g} eV "
                             f"({s['e0'] + lo:.1f} – {s['e0'] + hi:.1f} eV)")]
            msg += [_status(w, "warn") for w in warn]
            if fit and abs(fit["sum"] - 1) > 0.1:
                msg.append(_status(f"weights sum to {fit['sum']:.2f}: check normalization or missing standards",
                                   "warn"))
            return plots.fig_lcf(g, fit, S.lcf_standards(), s), rows, msg, [i]

    # --- write outputs
    @app.callback(Output("status", "children", allow_duplicate=True),
                  Input("write-binned", "n_clicks"), Input("write-qc", "n_clicks"), Input("write-norm", "n_clicks"),
                  Input("write-lcf", "n_clicks"), prevent_initial_call=True)
    def write(_b, _q, _n, _l):
        with LOCK:
            trig = ctx.triggered_id
            if not S.queue():
                return _status("The queue is empty: tick files in the table first.", "warn")
            try:
                fn = {"write-binned": S.write_binned, "write-qc": S.write_qc, "write-norm": S.write_normalized,
                      "write-lcf": S.write_lcf}[trig]
                paths = fn()
                S.save_params()
                folder = os.path.dirname(paths[0]) if paths else S.params["output_dir"]
                return _status(f"Wrote {len(paths)} files → {folder} (parameters saved to params/)")
            except Exception as exc:
                return fail("Write", exc)


def create_app(S: Session) -> Dash:
    if S.current is None:
        q = S.queue() or S.stems()
        S.current = q[0] if q else None
    app = Dash(__name__, title="quickXANES", suppress_callback_exceptions=True,
               assets_folder=os.path.join(os.path.dirname(__file__), "assets"))
    app.layout = build_layout(S)
    register(app, S)
    return app
