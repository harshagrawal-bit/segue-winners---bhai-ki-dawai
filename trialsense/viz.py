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


# =============================================================================
# Module 3 feature charts
#
# Each of the six Module 3 features renders its own separate report, so each
# gets its own dedicated figure rather than sharing a generic one.
# =============================================================================


def runway_chart(report) -> go.Figure:
    """
    Feature 1 — observed resistance, the fitted curve, and the projection cone.

    Three visually distinct layers, because conflating them would be exactly the
    dishonesty this feature exists to avoid: solid markers are OBSERVED data,
    the solid line is the FIT to those points, and the dashed line plus shaded
    band is EXTRAPOLATION. A reader must be able to see at a glance where the
    evidence stops and the model starts.
    """
    from .surveillance import VIABILITY_THRESHOLD

    fig = go.Figure()
    fc = report.forecast

    # Projection cone first, so it sits behind everything else.
    if fc and fc.projected:
        pyears = sorted(fc.projected)
        fig.add_trace(go.Scatter(
            x=pyears + pyears[::-1],
            y=[fc.hi[y] for y in pyears] + [fc.lo[y] for y in pyears][::-1],
            fill="toself", fillcolor="rgba(232,114,44,0.13)",
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        ))

    all_years = [p.year for p in report.points] + list(fc.projected if fc else [])
    if all_years:
        fig.add_shape(
            type="line", x0=min(all_years), x1=max(all_years),
            y0=VIABILITY_THRESHOLD, y1=VIABILITY_THRESHOLD,
            line=dict(color=RISK_COLORS["Severe"], width=1.5, dash="dot"),
        )
        fig.add_annotation(
            x=min(all_years), y=VIABILITY_THRESHOLD, xanchor="left", yanchor="bottom",
            text=f"{VIABILITY_THRESHOLD:.0f}% — commercial viability line",
            showarrow=False, font=dict(color=RISK_COLORS["Severe"], size=10),
        )

    if fc and fc.fitted:
        fyears = sorted(fc.fitted)
        fig.add_trace(go.Scatter(
            x=fyears, y=[fc.fitted[y] for y in fyears],
            mode="lines", name="Fitted trend",
            line=dict(color=ACCENT, width=2.5),
        ))

    if fc and fc.projected:
        pyears = sorted(fc.projected)
        last_fit = max(fc.fitted) if fc.fitted else pyears[0]
        bridge = [last_fit] + pyears
        bvals = [fc.fitted.get(last_fit, fc.projected[pyears[0]])] + [
            fc.projected[y] for y in pyears
        ]
        fig.add_trace(go.Scatter(
            x=bridge, y=bvals, mode="lines", name="Projection",
            line=dict(color=RISK_COLORS["High"], width=2.5, dash="dash"),
        ))

    obs = [p for p in report.points if p.comparable]
    held = [p for p in report.points if not p.comparable]
    if obs:
        fig.add_trace(go.Scatter(
            x=[p.year for p in obs], y=[p.percent for p in obs],
            mode="markers", name="Observed (in fit)",
            marker=dict(color=INK, size=9, line=dict(color=ACCENT, width=1.5)),
        ))
    if held:
        fig.add_trace(go.Scatter(
            x=[p.year for p in held], y=[p.percent for p in held],
            mode="markers", name="Observed (excluded — see notes)",
            marker=dict(color=MUTED, size=9, symbol="circle-open",
                        line=dict(color=MUTED, width=2)),
        ))

    if report.launch_pct is not None:
        fig.add_trace(go.Scatter(
            x=[report.launch_year], y=[report.launch_pct],
            mode="markers+text", name="At launch",
            marker=dict(color=report.color, size=15, symbol="diamond",
                        line=dict(color=INK, width=1.5)),
            text=[f" {report.launch_pct:.0f}%"], textposition="middle right",
            textfont=dict(color=INK, size=13),
        ))

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=11),
        xaxis=dict(title="Year", gridcolor=PANEL_EDGE, zeroline=False),
        yaxis=dict(title="% resistant", range=[0, 100], gridcolor=PANEL_EDGE,
                   zeroline=False),
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=10)),
        hovermode="x unified",
    )
    return fig


def portfolio_heatmap(report) -> go.Figure:
    """Feature 3 — candidates x organisms, coloured by predicted resistance."""
    cands = [c.name for c in report.candidates]
    orgs = list(report.organisms)

    z, text = [], []
    for c in cands:
        zrow, trow = [], []
        for o in orgs:
            cell = report.cells.get((c, o))
            if cell is None or cell.resistance is None:
                zrow.append(None)
                trow.append("—")
            else:
                zrow.append(cell.resistance * 100)
                trow.append(f"{cell.symbol} {cell.resistance:.0%}")
        z.append(zrow)
        text.append(trow)

    fig = go.Figure(go.Heatmap(
        z=z, x=[o[:26] for o in orgs], y=cands,
        text=text, texttemplate="%{text}",
        textfont=dict(size=12, color=INK),
        colorscale=[
            [0.0, "#1B5E3A"], [0.35, "#2E9E5B"], [0.5, "#E0A526"],
            [0.75, "#E8722C"], [1.0, "#D6453D"],
        ],
        zmin=0, zmax=100,
        colorbar=dict(title=dict(text="% resistant", side="right"),
                      tickfont=dict(color=MUTED, size=10), outlinewidth=0),
        hovertemplate="%{y} vs %{x}<br>predicted resistance %{z:.0f}%<extra></extra>",
        xgap=3, ygap=3,
    ))
    fig.update_layout(
        height=max(240, 70 * len(cands) + 110),
        margin=dict(l=10, r=10, t=14, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=11),
        xaxis=dict(side="top", tickangle=-18, tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    )
    return fig


def value_waterfall(report) -> go.Figure:
    """Feature 5 — cumulative spend up to the phase where the problem surfaces."""
    reached = [ln for ln in report.lines if ln.reached]
    labels = [ln.phase for ln in reached] + ["Total exposed", "With TrialSense"]
    values = [ln.cost for ln in reached] + [0.0, report.cost_with - report.cost_without]
    measures = ["relative"] * len(reached) + ["total", "relative"]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        text=[f"{ln.cost:.0f}" for ln in reached]
        + [f"{report.cost_without:.0f}", f"{report.cost_with:.2f}"],
        textposition="outside",
        textfont=dict(color=INK, size=11),
        connector=dict(line=dict(color=PANEL_EDGE, width=1)),
        increasing=dict(marker=dict(color=RISK_COLORS["High"])),
        decreasing=dict(marker=dict(color=RISK_COLORS["Low"])),
        totals=dict(marker=dict(color=RISK_COLORS["Severe"])),
    ))
    fig.update_layout(
        height=330, margin=dict(l=10, r=10, t=34, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=11),
        yaxis=dict(title="Rs crore", gridcolor=PANEL_EDGE, zeroline=False),
        xaxis=dict(tickangle=-12),
        showlegend=False,
    )
    return fig


def mobility_chart(report) -> go.Figure:
    """Feature 6 — detected determinants ranked by how fast they spread."""
    if not report.entries:
        fig = go.Figure()
        fig.update_layout(
            height=180, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED),
            annotations=[dict(text="No determinants detected", showarrow=False,
                              font=dict(color=MUTED, size=13))],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    entries = sorted(report.entries, key=lambda e: e.gene.spread_multiplier)
    names = [e.gene.name for e in entries]
    mults = [e.gene.spread_multiplier for e in entries]
    colors = [
        RISK_COLORS["Severe"] if m >= 2.5
        else RISK_COLORS["High"] if m >= 1.8
        else RISK_COLORS["Moderate"] if m > 1.0
        else RISK_COLORS["Low"]
        for m in mults
    ]
    labels = [e.label for e in entries]

    fig = go.Figure(go.Bar(
        x=mults, y=names, orientation="h",
        marker=dict(color=colors),
        text=[f"  {l}" for l in labels],
        textposition="outside",
        textfont=dict(color=MUTED, size=10),
        hovertemplate="%{y}<br>spread factor %{x:.1f}x<extra></extra>",
    ))
    fig.add_shape(
        type="line", x0=1.0, x1=1.0, y0=-0.5, y1=len(names) - 0.5,
        line=dict(color=MUTED, width=1, dash="dot"),
    )
    fig.update_layout(
        height=max(200, 46 * len(names) + 80),
        margin=dict(l=10, r=170, t=26, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=11),
        xaxis=dict(title="Spread factor vs a chromosomal mutation (1.0x)",
                   gridcolor=PANEL_EDGE, zeroline=False, range=[0, 3.6]),
        yaxis=dict(tickfont=dict(size=11)),
        showlegend=False,
    )
    return fig


def novelty_map(report) -> go.Figure:
    """Feature 4 — where along the genome the unexplained segments sit."""
    L = max(1, report.sequence_length)
    fig = go.Figure()

    fig.add_shape(
        type="rect", x0=0, x1=L, y0=0.38, y1=0.62,
        fillcolor=PANEL, line=dict(color=PANEL_EDGE, width=1),
    )
    for seg in report.segments:
        fig.add_shape(
            type="rect", x0=seg.start, x1=seg.end, y0=0.30, y1=0.70,
            fillcolor=RISK_COLORS["Moderate"], opacity=0.85,
            line=dict(color=RISK_COLORS["High"], width=1.5),
        )
        fig.add_annotation(
            x=(seg.start + seg.end) / 2, y=0.80,
            text=f"{seg.length:,} bp unexplained", showarrow=False,
            font=dict(color=RISK_COLORS["Moderate"], size=10),
        )

    fig.update_layout(
        height=160, margin=dict(l=10, r=10, t=34, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=11),
        xaxis=dict(title="Position (bp)", range=[0, L], gridcolor=PANEL_EDGE,
                   zeroline=False),
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=False,
    )
    return fig


def source_comparison_chart(series_by_source: dict) -> go.Figure:
    """
    Feature 1 side panel — the same organism and class from different
    surveillance programmes, plotted together.

    Showing that official sources disagree is a deliberate honesty move. It
    mirrors the documented AMRFinder-vs-ResFinder divergence, and it is a far
    stronger position than quietly picking whichever series flatters us.
    """
    fig = go.Figure()
    palette = [ACCENT, RISK_COLORS["High"], RISK_COLORS["Low"], RISK_COLORS["Moderate"]]
    for i, (label, series) in enumerate(series_by_source.items()):
        years = sorted(series)
        fig.add_trace(go.Scatter(
            x=years, y=[series[y] for y in years],
            mode="lines+markers", name=label,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=7),
        ))
    fig.update_layout(
        height=290, margin=dict(l=10, r=10, t=24, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=11),
        xaxis=dict(title="Year", gridcolor=PANEL_EDGE, zeroline=False),
        yaxis=dict(title="% resistant", range=[0, 100], gridcolor=PANEL_EDGE,
                   zeroline=False),
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
        hovermode="x unified",
    )
    return fig
