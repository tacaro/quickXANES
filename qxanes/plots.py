"""Plotly figure builders. Colors follow the entity (channel, replicate, standard), never its rank."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# validated categorical order (dataviz reference palette, light surface)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
WINDOW_FILL = "rgba(137,135,129,0.10)"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def series_color(i: int, n: int) -> str:
    """Fixed-order categorical slot; beyond 8 series everything folds to muted ink (no generated hues)."""
    return SERIES[i] if n <= len(SERIES) and i < len(SERIES) else MUTED


def _style(fig: go.Figure, height: int = 520, uirev: str = "keep") -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK2),
        margin=dict(l=64, r=16, t=92, b=44),
        title=dict(x=0, xanchor="left", y=0.99, yanchor="top", pad=dict(l=8), font=dict(color=INK, size=14)),
        hovermode="x unified", hoverlabel=dict(bgcolor="white", font_color=INK, bordercolor=AXIS),
        legend=dict(orientation="h", x=0, y=1.01, yanchor="bottom", bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
        uirevision=uirev,  # zoom survives parameter tweaks, resets when the file changes
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, linecolor=AXIS, zeroline=False,
                     ticks="outside", tickcolor=AXIS, title_font=dict(color=INK2), tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, linecolor=AXIS, zeroline=False,
                     ticks="outside", tickcolor=AXIS, title_font=dict(color=INK2), tickfont=dict(color=MUTED),
                     exponentformat="power")
    return fig


def message(text: str, height: int = 520) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=INK2))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height)


def _line(x, y, name, color, width=2.0, dash=None, showlegend=True, hover=".4g", opacity=1.0):
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=width, dash=dash),
                      showlegend=showlegend, opacity=opacity,
                      hovertemplate=f"%{{y:{hover}}}<extra>{name}</extra>")


# step one
def fig_bin(stem: str, sig: dict, mode: str, all_channels: list[str], used: list[str]) -> go.Figure:
    """Top: mu(E). Bottom (fluorescence): each channel / I0; excluded channels muted & dotted."""
    if mode != "fluorescence" or not sig["channels"]:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(_line(sig["energy"], sig["mu"], "μ(E)", INK))
        fig.update_yaxes(title_text="μ(E)")
        fig.update_xaxes(title_text="Energy (eV)")
        fig.update_layout(title=dict(text=stem), showlegend=False)
        return _style(fig, 520, stem)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45])
    fig.add_trace(_line(sig["energy"], sig["mu"], "Σ/I0", INK, showlegend=False), row=1, col=1)
    n = len(all_channels)
    for i, ch in enumerate(all_channels):
        if ch not in sig["channels"]:
            continue
        on = ch in used
        fig.add_trace(_line(sig["energy"], sig["channels"][ch], ch + ("" if on else " (excluded)"),
                            series_color(i, n) if on else MUTED, width=2 if on else 1.5,
                            dash=None if on else "dot"), row=2, col=1)
    fig.update_yaxes(title_text="Σ channels / I0", row=1, col=1)
    fig.update_yaxes(title_text="each channel / I0", row=2, col=1)
    fig.update_xaxes(title_text="Energy (eV)", row=2, col=1)
    fig.update_layout(title=dict(text=stem))
    return _style(fig, 600, stem)


#  step 2
def fig_group(group: str, gd: dict, current: str | None) -> go.Figure:
    """Top: replicates + average. Bottom: replicate − average (QC residual)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.68, 0.32])
    avg = gd["avg"]
    n = len(gd["members"])
    for i, s in enumerate(gd["members"]):
        sg = gd["signals"][s]
        used = s in gd["used"]
        color = series_color(i, n) if used else MUTED
        width = 2.5 if s == current else 1.5
        label = s + ("" if used else " (excluded)")
        fig.add_trace(_line(sg["energy"], sg["mu"], label, color, width, None if used else "dot",
                            opacity=1.0 if used else 0.8), row=1, col=1)
        diff = np.interp(avg["energy"], sg["energy"], sg["mu"], left=np.nan, right=np.nan) - avg["mu"]
        fig.add_trace(_line(avg["energy"], diff, label, color, width, None if used else "dot",
                            showlegend=False), row=2, col=1)
    if n > 1:
        fig.add_trace(_line(avg["energy"], avg["mu"], f"average (n={len(gd['used'])})", INK, 2.5), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=2, col=1)
    fig.update_yaxes(title_text="μ(E)", row=1, col=1)
    fig.update_yaxes(title_text="replicate − avg", row=2, col=1)
    fig.update_xaxes(title_text="Energy (eV)", row=2, col=1)
    fig.update_layout(title=dict(text=group))
    return _style(fig, 600, group)


def fig_calibration(group: str, E, mu, cal: dict) -> go.Figure:
    fig = go.Figure(_line(E, mu, group, SERIES[0]))
    meas, ref = cal.get("measured_eV"), cal.get("reference_eV")
    if meas is not None:
        fig.add_vline(x=meas, line=dict(color=MUTED, width=1.5, dash="dash"))
    if ref not in (None, ""):
        fig.add_vline(x=float(ref), line=dict(color=INK, width=1.5, dash="dot"))
    label = []
    if meas is not None:
        label.append(f"measured {meas:.3f} eV (dashed)")
    if ref not in (None, ""):
        label.append(f"reference {float(ref):.3f} eV (dotted)")
    if label:
        fig.add_annotation(text="<br>".join(label), x=0.99, y=0.04, xref="paper", yref="paper", showarrow=False,
                           xanchor="right", yanchor="bottom", align="right", font=dict(size=11, color=INK2))
    center = meas if meas is not None else (float(ref) if ref not in (None, "") else None)
    if center is not None:
        fig.update_xaxes(range=[center - 25, center + 35])
    fig.update_xaxes(title_text="Energy (eV, uncalibrated)")
    fig.update_yaxes(title_text="μ(E)")
    fig.update_layout(showlegend=False)
    _style(fig, 300, group)
    fig.update_layout(title=dict(text=f"calibration: {group}", font=dict(size=13)), margin=dict(t=40))
    return fig


# step 3
def fig_norm(group: str, r: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45])
    E = r["energy"]
    for lo, hi in (r["pre_window"], r["post_window"]):
        fig.add_vrect(x0=lo, x1=hi, fillcolor=WINDOW_FILL, line_width=0, row=1, col=1)
    fig.add_trace(_line(E, r["mu"], "μ(E)", SERIES[0]), row=1, col=1)
    fig.add_trace(_line(E, r["pre_bkg"], f"pre-edge ({r['pre']['model']})", SERIES[1], dash="dash"), row=1, col=1)
    above = E >= min(r["e0"], r["post_window"][0])
    fig.add_trace(_line(E[above], r["post_line"][above], "post-edge", SERIES[2], dash="dash"), row=1, col=1)
    y = r["flat"] if r["config"].get("flatten", True) else r["norm"]
    fig.add_trace(_line(E, y, "normalized", INK, showlegend=False), row=2, col=1)
    for v in (0.0, 1.0):
        fig.add_hline(y=v, line=dict(color=AXIS, width=1, dash="dot"), row=2, col=1)
    for row in (1, 2):
        fig.add_vline(x=r["e0"], line=dict(color=MUTED, width=1), row=row, col=1)
    fig.add_annotation(x=r["e0"], y=1, yref="paper", text=f"E0 {r['e0']:.2f}", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(size=11, color=INK2))
    fig.update_yaxes(title_text="μ(E) + models", row=1, col=1)
    fig.update_yaxes(title_text="normalized" + (" (flat)" if r["config"].get("flatten", True) else ""), row=2, col=1)
    fig.update_xaxes(title_text="Energy (eV)", row=2, col=1)
    fig.update_layout(title=dict(text=group))
    return _style(fig, 620, group)


# Step 4
FIT_COLOR = SERIES[1]
_COMPONENT_SLOTS = [SERIES[i] for i in (0, 2, 3, 4, 5, 6, 7)]  # orange is reserved for the total fit


def component_color(name: str, all_standards: list[str]) -> str:
    i = all_standards.index(name) if name in all_standards else len(_COMPONENT_SLOTS)
    return _COMPONENT_SLOTS[i] if i < len(_COMPONENT_SLOTS) and len(all_standards) <= len(_COMPONENT_SLOTS) else MUTED


def fig_lcf(sample: str, fit: dict | None, all_standards: list[str], norm_res: dict | None = None) -> go.Figure:
    if fit is None:
        return message("No acceptable fit: check standards and the fit range.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.72, 0.28])
    if norm_res is not None:  # full normalized spectrum, muted, for context outside the fit range
        y = norm_res["flat"] if norm_res["config"].get("flatten", True) else norm_res["norm"]
        fig.add_trace(_line(norm_res["energy"], y, "outside fit range", AXIS, 1.5, showlegend=False), row=1, col=1)
    x = fit["energy"]
    fig.add_trace(_line(x, fit["data"], sample, INK), row=1, col=1)
    fig.add_trace(_line(x, fit["fit"], f"fit (R = {fit['r_factor']:.2e})", FIT_COLOR, dash="dash"), row=1, col=1)
    for name, part in fit["parts"].items():
        i = fit["components"].index(name)
        fig.add_trace(_line(x, part, f"{name} ×{fit['weights'][i]:.3f}", component_color(name, all_standards),
                            1.5, dash="dot"), row=1, col=1)
    fig.add_trace(_line(x, fit["data"] - fit["fit"], "residual", INK2, 1.5, showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=2, col=1)
    fig.update_xaxes(range=[x[0] - 5, x[-1] + 5])
    fig.update_yaxes(title_text="normalized μ", row=1, col=1)
    fig.update_yaxes(title_text="data − fit", row=2, col=1)
    fig.update_xaxes(title_text="Energy (eV)", row=2, col=1)
    fig.update_layout(title=dict(text=sample))
    return _style(fig, 600, sample)
