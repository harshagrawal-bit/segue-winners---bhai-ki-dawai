"""
TrialSense — shared visual helpers for the Streamlit app.

Kept out of app.py so the UI file stays readable: this module owns molecule
rendering, the risk gauge, and the small HTML card components.
"""

from __future__ import annotations

import math
import re

import plotly.graph_objects as go
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

# --- Palette ----------------------------------------------------------------
INK = "#E8EDF4"
MUTED = "#8FA3BF"
PANEL = "#141B26"
PANEL_EDGE = "#243247"
ACCENT = "#4F9CF9"

RISK_COLORS = {
    "Low": "#2E9E5B",
    "None": "#2E9E5B",
    "Info": "#5B7290",
    "Mild": "#E0A526",
    "Moderate": "#E0A526",
    "Medium": "#E0A526",
    "High": "#E8722C",
    "Severe": "#D6453D",
    "Critical": "#D6453D",
}


def molecule_svg(smiles: str, width: int = 330, height: int = 210) -> str:
    """
    Render a molecule as a dark-theme SVG.

    Showing the actual structure matters for the pitch: it makes visible that
    the tool reads chemistry, not just a drug name in a lookup table.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"<div style='color:{MUTED};padding:1rem'>Structure unavailable</div>"

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    # Light atom palette so heteroatoms stay legible on a dark panel.
    opts.updateAtomPalette(
        {
            0: (0.91, 0.93, 0.96),  # default for anything unlisted
            6: (0.91, 0.93, 0.96),  # C — must be set explicitly, or the carbon
            # skeleton (most of the drawing) renders in RDKit's dark default
            7: (0.42, 0.68, 1.00),  # N — blue
            8: (1.00, 0.45, 0.42),  # O — red
            9: (0.45, 0.90, 0.60),  # F
            15: (1.00, 0.65, 0.30),  # P
            16: (1.00, 0.85, 0.35),  # S — yellow
            17: (0.45, 0.90, 0.60),  # Cl — green
            35: (0.85, 0.55, 0.35),  # Br
            53: (0.75, 0.50, 0.95),  # I — purple
        }
    )
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def risk_gauge(score: float, band: str, color: str) -> go.Figure:
    """Composite 0-100 risk dial for the top of the unified report."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "<span style='font-size:0.5em'>/100</span>",
                    "font": {"size": 46, "color": INK}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": MUTED,
                    "tickfont": {"color": MUTED, "size": 11},
                },
                "bar": {"color": color, "thickness": 0.72},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "rgba(46,158,91,0.16)"},
                    {"range": [25, 50], "color": "rgba(224,165,38,0.16)"},
                    {"range": [50, 75], "color": "rgba(232,114,44,0.16)"},
                    {"range": [75, 100], "color": "rgba(214,69,61,0.18)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=18, r=18, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": INK},
    )
    return fig


def exposure_chart(exposures) -> go.Figure:
    """
    Exposure change vs a healthy adult, as diverging bars centred on 1.0.

    The bars plot log2(ratio), so they grow rightwards from the 1.0 baseline for
    accumulation and leftwards for faster clearance, and a 2x rise is the same
    visual distance as a 2x fall. Tick labels are relabelled back to plain
    multipliers so nobody has to read a logarithm.

    An earlier version put the raw ratio on a log-scaled axis. That was
    misleading: Plotly grows bars from the axis floor, not from 1.0, so a
    harmless 1.04x change rendered as a bar spanning most of the panel.
    """
    names = [e.drug_name for e in exposures]
    ratios = [e.exposure_ratio for e in exposures]
    logs = [math.log2(max(1e-6, r)) for r in ratios]
    colors = [
        RISK_COLORS["Severe"] if r >= 2.0
        else RISK_COLORS["High"] if r >= 1.5
        else RISK_COLORS["Moderate"] if r >= 1.2
        else RISK_COLORS["Low"]
        for r in ratios
    ]
    fig = go.Figure(
        go.Bar(
            x=logs,
            y=names,
            orientation="h",
            marker_color=colors,
            width=0.5,
            text=[f"  {r:.2f}×  " for r in ratios],
            textposition="outside",
            textfont={"color": INK, "size": 14},
            customdata=ratios,
            hovertemplate="%{y}: %{customdata:.2f}× normal exposure<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color=MUTED)

    # Symmetric range so the 1.0 baseline sits visually in a sensible place.
    span = max(1.0, max(abs(v) for v in logs) * 1.5) if logs else 1.0
    ticks = [t for t in (-2, -1, 0, 1, 2, 3) if abs(t) <= span + 0.5]
    fig.update_layout(
        height=130 + 54 * len(names),
        margin=dict(l=8, r=34, t=34, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": INK},
        xaxis=dict(
            title=dict(
                text="Exposure relative to a healthy adult (1× = unchanged)",
                font={"color": MUTED, "size": 12},
            ),
            range=[-span, span],
            tickmode="array",
            tickvals=ticks,
            ticktext=[f"{2.0**t:g}×" for t in ticks],
            gridcolor=PANEL_EDGE,
            zeroline=False,
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


def resistance_chart(probabilities: dict[str, float], last_resort: set[str]) -> go.Figure:
    """Per-antibiotic-class resistance probability, ordered most-resistant first."""
    items = sorted(probabilities.items(), key=lambda kv: kv[1])
    classes = [k for k, _ in items]
    values = [v * 100 for _, v in items]
    colors = [
        (RISK_COLORS["Critical"] if c in last_resort else RISK_COLORS["High"])
        if v >= 50
        else (RISK_COLORS["Moderate"] if v >= 25 else RISK_COLORS["Low"])
        for c, v in zip(classes, values)
    ]
    labels = [f"{c} ★" if c in last_resort else c for c in classes]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}%" for v in values],
            textposition="outside",
            textfont={"color": INK, "size": 12},
            hovertemplate="%{y}: %{x:.0f}% predicted resistance<extra></extra>",
        )
    )
    fig.add_vline(x=50, line_dash="dash", line_color=MUTED)
    fig.update_layout(
        height=460,
        margin=dict(l=8, r=44, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": INK},
        xaxis=dict(
            title=dict(
                text="Predicted resistance probability (★ = last-resort class)",
                font={"color": MUTED, "size": 12},
            ),
            range=[0, 112],
            gridcolor=PANEL_EDGE,
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


def screening_probability_chart(prob: float, threshold: float) -> go.Figure:
    """
    The structure-only filter's single probability, against its decision
    threshold. A one-bar chart rather than a gauge, so the threshold line —
    the part that actually determines the flag — is unmissable.
    """
    color = RISK_COLORS["Severe"] if prob >= threshold else RISK_COLORS["Low"]
    fig = go.Figure(
        go.Bar(
            x=[prob * 100],
            y=["structure-only<br>screening filter"],
            orientation="h",
            marker_color=color,
            width=0.5,
            text=[f"{prob*100:.0f}%"],
            textposition="outside",
            textfont={"color": INK, "size": 15},
            hovertemplate="%{x:.0f}% probability of a clinically significant interaction<extra></extra>",
        )
    )
    fig.add_vline(
        x=threshold * 100,
        line_dash="dash",
        line_color=INK,
        annotation_text=f"flag threshold {threshold:.0%}",
        annotation_font_color=MUTED,
        annotation_font_size=11,
        annotation_position="top",
    )
    fig.update_layout(
        height=150,
        margin=dict(l=8, r=44, t=34, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": INK},
        xaxis=dict(
            title=dict(
                text="Probability this pairing is clinically significant",
                font={"color": MUTED, "size": 12},
            ),
            range=[0, 112],
            gridcolor=PANEL_EDGE,
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont={"size": 11}),
        showlegend=False,
    )
    return fig


# --- Small HTML components ---------------------------------------------------


def md_bold(text: str) -> str:
    """
    Convert `**bold**` to `<b>bold</b>`.

    The report's action strings are written in Markdown, but they are rendered
    inside raw HTML cards where Streamlit's Markdown parser never runs — without
    this the user sees literal asterisks.
    """
    return re.sub(r"\*\*(.+?)\*\*", r"<b style='color:" + INK + r"'>\1</b>", text)


def card(title: str, body: str, color: str = ACCENT, tag: str = "") -> str:
    """A panel with a coloured left edge — the app's main content unit."""
    tag_html = (
        f"<span style='background:{color}22;color:{color};padding:.16rem .55rem;"
        f"border-radius:5px;font-size:.68rem;font-weight:700;letter-spacing:.04em;"
        f"text-transform:uppercase'>{tag}</span>"
        if tag
        else ""
    )
    return f"""
    <div style="background:{PANEL};border:1px solid {PANEL_EDGE};
                border-left:4px solid {color};border-radius:9px;
                padding:1rem 1.15rem;margin-bottom:.7rem">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:.7rem">
        <div style="font-weight:650;color:{INK};font-size:1.0rem">{title}</div>
        {tag_html}
      </div>
      <div style="color:{MUTED};margin-top:.42rem;line-height:1.6;font-size:.9rem">{md_bold(body)}</div>
    </div>"""


def stat(label: str, value: str, sub: str = "", color: str = INK) -> str:
    """A compact metric tile."""
    return f"""
    <div style="background:{PANEL};border:1px solid {PANEL_EDGE};border-radius:9px;
                padding:.85rem 1rem;height:100%">
      <div style="color:{MUTED};font-size:.7rem;text-transform:uppercase;
                  letter-spacing:.07em;font-weight:600">{label}</div>
      <div style="color:{color};font-size:1.5rem;font-weight:700;margin-top:.28rem;
                  line-height:1.15">{value}</div>
      <div style="color:{MUTED};font-size:.78rem;margin-top:.18rem">{sub}</div>
    </div>"""


def banner(text: str, color: str) -> str:
    """Full-width headline strip for the top of the report."""
    return f"""
    <div style="background:linear-gradient(90deg,{color}26,{color}0A);
                border:1px solid {color}55;border-left:5px solid {color};
                border-radius:10px;padding:1.05rem 1.3rem;margin:.3rem 0 1rem 0">
      <div style="color:{INK};font-size:1.12rem;font-weight:650;line-height:1.5">{text}</div>
    </div>"""


def pill(text: str, color: str) -> str:
    return (
        f"<span style='background:{color}1F;color:{color};border:1px solid {color}55;"
        f"padding:.2rem .6rem;border-radius:20px;font-size:.76rem;font-weight:600;"
        f"margin-right:.35rem;display:inline-block;margin-bottom:.3rem'>{text}</span>"
    )
