#!/usr/bin/env python3
"""
TrialSense — computational pre-trial risk screening for drug candidates.

Run with:   streamlit run app.py
(Train the models first with `python train.py` so the app opens instantly.)
"""

from __future__ import annotations

import datetime as _dt
import json
import pickle
from pathlib import Path

import streamlit as st

from trialsense import amr as amr_mod
from trialsense import dnabert as bert_mod
from trialsense import economics as econ_mod
from trialsense import mechanisms as mech_mod
from trialsense import novelty as nov_mod
from trialsense import perdrug as pd_mod
from trialsense import portfolio as port_mod
from trialsense import surveillance as surv_mod
from trialsense import viz
from trialsense.cases import DEMO_CASES
from trialsense.ddi import (
    SCREENING_THRESHOLD,
    SEVERITY_LEVELS,
    DDIModel,
    analyze_pair,
    build_dataset,
)
from trialsense.drugs import DRUG_NAMES, DRUGS, FLAG_LABELS, drug_from_smiles, get_drug
from trialsense.pk import PatientProfile, personalize, renal_study_plan
from trialsense.pk_validation import validate as validate_pk
from trialsense.report import build_report

MODELS_DIR = Path(__file__).parent / "models"

st.set_page_config(
    page_title="TrialSense — Pre-Trial Risk Screening",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Styling
# =============================================================================
st.markdown(
    f"""
<style>
  .stApp {{ background:#0B1017; }}
  /* Generous top padding: Streamlit's floating toolbar overlaps the first
     element otherwise, which clipped the report title. */
  .block-container {{ padding-top:3.4rem; padding-bottom:3rem; max-width:1400px; }}
  h1,h2,h3,h4 {{ color:{viz.INK} !important; letter-spacing:-.015em; }}
  p, li, label, .stMarkdown {{ color:{viz.MUTED}; }}
  section[data-testid="stSidebar"] {{ background:#0E141D; border-right:1px solid {viz.PANEL_EDGE}; }}
  section[data-testid="stSidebar"] .stMarkdown p {{ font-size:.86rem; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:.35rem; border-bottom:1px solid {viz.PANEL_EDGE}; }}
  .stTabs [data-baseweb="tab"] {{
      background:transparent; color:{viz.MUTED}; font-weight:600; font-size:.92rem;
      padding:.6rem 1.05rem; border-radius:8px 8px 0 0; }}
  .stTabs [aria-selected="true"] {{ background:{viz.PANEL}; color:{viz.INK} !important;
      border-bottom:2px solid {viz.ACCENT}; }}
  div[data-testid="stMetricValue"] {{ color:{viz.INK}; font-size:1.5rem; }}
  div[data-testid="stMetricLabel"] {{ color:{viz.MUTED}; }}
  .stAlert {{ background:{viz.PANEL}; border:1px solid {viz.PANEL_EDGE}; }}
  hr {{ border-color:{viz.PANEL_EDGE}; }}
  .stButton button {{ width:100%; border-radius:8px; font-weight:600; }}
  code {{ color:#7FC4FF; background:#16202E; }}
  .molbox {{ background:{viz.PANEL}; border:1px solid {viz.PANEL_EDGE};
             border-radius:9px; padding:.5rem; text-align:center; }}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Model loading (cached across reruns)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    """Load cached models, or train on the fly if train.py was never run."""
    ddi_path, amr_path = MODELS_DIR / "ddi_model.pkl", MODELS_DIR / "amr_model.pkl"
    metrics_path = MODELS_DIR / "metrics.json"

    if ddi_path.exists() and amr_path.exists():
        with open(ddi_path, "rb") as fh:
            ddi_model = pickle.load(fh)
        with open(amr_path, "rb") as fh:
            amr_model = pickle.load(fh)
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        return ddi_model, amr_model, metrics, True

    with st.spinner("First run — training models (about a minute)…"):
        X, y, _ = build_dataset()
        ddi_model = DDIModel().fit(X, y)
        Xa, Ya, _ = amr_mod.build_amr_dataset()
        amr_model = amr_mod.AMRModel().fit(Xa, Ya)
    return ddi_model, amr_model, {}, False


@st.cache_data(show_spinner=False)
def strain_sequence(strain_name: str) -> str:
    return amr_mod.demo_strain_sequence(strain_name)


ddi_model, _amr_model, METRICS, from_cache = load_models()


@st.cache_data(show_spinner=False)
def pk_validation():
    """Module 2 predictions scored against real renal impairment data."""
    res = validate_pk()
    return res.summary(), res.design_calls()


PK_VAL, PK_DESIGN = pk_validation()


# =============================================================================
# Module 3 helpers — caching and per-report export
#
# Each of Module 3's six features produces its OWN report, and each is
# downloadable separately. That separation is the point: a portfolio team needs
# to see where the analyses disagree, which a single merged verdict would hide.
# =============================================================================


@st.cache_data(show_spinner=False)
def _isolate_for(strain_name: str):
    """
    AMRReport for one demo strain, memoised.

    The portfolio grid screens N candidates against M organisms, and without
    this every candidate would re-synthesise and re-scan the same genome.
    """
    seq = amr_mod.demo_strain_sequence(strain_name)
    return amr_mod.analyze_isolate(
        seq, _amr_model, strain_name, amr_mod.DEMO_STRAINS[strain_name].context
    )


@st.cache_data(show_spinner=False)
def _novelty_for(seq: str, strain_name: str, n_genes: int):
    """Open-world scan, memoised — it slides a fine window over the genome."""
    return nov_mod.build_novelty_report(seq, strain_name, _amr_model, n_genes)


def _report_markdown(kind: str, title: str, rpt, strain: str) -> str:
    """
    Render one feature's report as standalone Markdown.

    Written so each downloaded file makes sense on its own desk, without the
    app open and without the other five reports next to it.
    """
    lines = [
        f"# TrialSense — Module 3 — {title}",
        "",
        f"**Isolate:** {strain}",
        f"**Generated:** {_dt.datetime.now():%Y-%m-%d %H:%M}",
        "",
        "---",
        "",
        f"## Verdict: {getattr(rpt, 'verdict', 'n/a')}",
        "",
        f"**{rpt.headline()}**",
        "",
        rpt.plain_summary(),
        "",
    ]
    if hasattr(rpt, "recommendation"):
        lines += ["### Recommended action", "", rpt.recommendation(), ""]

    if kind == "runway" and getattr(rpt, "available", False):
        lines += ["### Evidence", ""]
        lines += [f"- {e}" for e in rpt.evidence_lines()]
        lines += ["", "### Series", "", "| Year | % resistant | In fit |", "|---|---|---|"]
        lines += [
            f"| {p.year} | {p.percent:.1f} | {'yes' if p.comparable else 'no'} |"
            for p in rpt.points
        ]
        if rpt.forecast and rpt.forecast.projected:
            lines += ["", "| Year | Projected | Low | High |", "|---|---|---|---|"]
            lines += [
                f"| {y} | {rpt.forecast.projected[y]:.1f} | {rpt.forecast.lo[y]:.1f} "
                f"| {rpt.forecast.hi[y]:.1f} |"
                for y in sorted(rpt.forecast.projected)
            ]

    elif kind == "rescue":
        if rpt.contrast_note():
            lines += ["### Why mechanism matters", "", rpt.contrast_note(), ""]
        if rpt.blocking:
            lines += ["### Determinants blocking this class", ""]
            for gv in rpt.blocking:
                lines += [
                    f"**{gv.name}** (detected {gv.confidence:.0%}) — "
                    f"{gv.mechanism_label}"
                    + (f", {gv.gene.enzyme_subtype}" if gv.gene.enzyme_subtype else ""),
                    "",
                    f"- Rescuable: **{gv.rescue_viable}**",
                    f"- Strategy: {gv.gene.rescue_strategy}",
                    f"- Note: {gv.gene.rescue_note}",
                    "",
                ]

    elif kind == "portfolio":
        lines += ["### Grid", "", "| Candidate | Class | Organism | Resistance | Status |",
                  "|---|---|---|---|---|"]
        for row in rpt.as_rows():
            pr = "—" if row["predicted_resistance"] is None else f"{row['predicted_resistance']:.0%}"
            lines.append(
                f"| {row['candidate']} | {row['class']} | {row['organism']} "
                f"| {pr} | {row['status']} |"
            )
        lines += ["", "### Ranking", ""]
        lines += [
            f"{i}. **{s.name}** ({s.antibiotic_class}) — viable against "
            f"{s.viable_against}/{s.total_assessed}"
            for i, s in enumerate(rpt.scores, 1)
        ]
        lines += ["", f"_{rpt.n_combinations} combinations screened in "
                      f"{rpt.elapsed_seconds:.2f} seconds._"]
        cc = getattr(rpt, "cross_check", None)
        if cc is not None and cc.compared:
            lines += [
                "", "### Validated against real ICMR national surveillance", "",
                cc.plain_summary(), "",
                f"- Tracks the national rate: **{cc.n_aligned}**",
                f"- Differs in the expected direction: **{cc.n_as_expected}**",
                f"- Flagged for review: **{len(cc.flagged)}**",
                f"- No ICMR reference available: **{cc.n_no_reference}**",
                "",
                "| Organism | Class | Model | ICMR | Gap | Status |",
                "|---|---|---|---|---|---|",
            ]
            for r in cc.rows:
                real = f"{r.real_pct:.0f}%" if r.real_pct is not None else "-"
                gap = f"{r.delta:+.0f} pp" if r.delta is not None else "-"
                lines.append(
                    f"| {r.organism} | {r.antibiotic_class} | {r.model_pct:.0f}% "
                    f"| {real} | {gap} | {r.status} |"
                )
            lines += ["", "> " + cc.caveat().replace("**", "")]

    elif kind == "novelty":
        lines += [
            "### Scan detail", "",
            f"- Sequence length: {rpt.sequence_length:,} bp",
            f"- Windows examined: {rpt.n_windows}",
            f"- Windows explained by known genes: {rpt.n_explained}",
            f"- Model distribution band: {rpt.ood_band}"
            + (f" ({rpt.ood_percentile:.0f}th percentile)" if rpt.ood_available else ""),
            "",
        ]
        if rpt.segments:
            lines += ["### Unexplained segments", ""]
            lines += [f"- {s.describe()}" for s in rpt.segments]

    elif kind == "value":
        lines += [
            "### Counterfactual", "",
            f"- Discovery phase without screening: **{rpt.discovery_phase}**",
            f"- Cost exposed by then: **Rs {rpt.cost_without:.0f} crore**",
            f"- Time exposed by then: **{rpt.years_without:.1f} years**",
            f"- Cost of the computational screen: **Rs {rpt.cost_with:.2f} crore**",
            f"- Probability of reaching that phase: **{rpt.probability_reaching_discovery:.0%}**",
            f"- Expected value: **Rs {rpt.expected_value_crore:.0f} crore**",
            "",
            "### Assumptions used", "",
            "| Phase | Cost (Rs cr) | P(success) | Years |", "|---|---|---|---|",
        ]
        lines += [
            f"| {ph} | {rpt.costs[ph]:.0f} | {rpt.pts[ph]:.2f} | {rpt.durations[ph]:.1f} |"
            for ph in rpt.costs
        ]
        lines += ["", "> " + rpt.disclaimer().replace("**", "")]

    elif kind == "spread":
        lines += ["### Determinants by mobility", ""]
        for e in rpt.entries:
            lines += [
                f"**{e.gene.name}** — {e.label} ({e.gene.spread_multiplier:.1f}x)",
                "",
                f"- {e.gene.mobility_note}",
                "",
            ]
        lines += [f"Runway multiplier applied: **{rpt.multiplier:.1f}x**"]

    lines += [
        "",
        "---",
        "",
        "_TrialSense Module 3 — research screening prototype. Not a clinical "
        "decision-support system and not for treating patients._",
    ]
    return "\n".join(lines)


def _download(kind: str, title: str, rpt, strain: str) -> None:
    """Per-feature download button. Each report stands alone as its own file."""
    safe = "".join(c if c.isalnum() else "_" for c in strain)[:40]
    st.download_button(
        f"⬇ Download the {title} report",
        data=_report_markdown(kind, title, rpt, strain),
        file_name=f"trialsense_{kind}_{safe}.md",
        mime="text/markdown",
        width="stretch",
        key=f"dl_{kind}",
    )



# =============================================================================
# Sidebar — case selection and inputs
# =============================================================================
with st.sidebar:
    st.markdown(
        f"<div style='font-size:1.45rem;font-weight:750;color:{viz.INK};"
        f"letter-spacing:-.02em'>🧬 TrialSense</div>"
        f"<div style='color:{viz.MUTED};font-size:.82rem;margin-bottom:1.1rem'>"
        "Pre-trial risk screening for drug candidates</div>",
        unsafe_allow_html=True,
    )

    st.markdown("##### Preloaded case")
    case_names = ["— Custom case —"] + [c.name for c in DEMO_CASES]
    chosen_case = st.selectbox(
        "Preloaded case", case_names, label_visibility="collapsed",
        help="Five scenarios, each demonstrating a different capability.",
    )
    active_case = next((c for c in DEMO_CASES if c.name == chosen_case), None)

    if active_case:
        st.caption(active_case.pitch)

    st.divider()

    # --- Compounds
    st.markdown("##### Candidate compounds")
    input_mode = st.radio(
        "Input mode",
        ["Known drugs", "Custom structure (SMILES)"],
        label_visibility="collapsed",
        horizontal=False,
        disabled=active_case is not None,
    )

    if active_case:
        drug_a = DRUGS[active_case.drug_a]
        drug_b = DRUGS[active_case.drug_b]
        st.info(f"**{drug_a.name}**  +  **{drug_b.name}**")
    elif input_mode == "Known drugs":
        col1, col2 = st.columns(2)
        with col1:
            name_a = st.selectbox("Compound A", DRUG_NAMES, index=DRUG_NAMES.index("Warfarin"))
        with col2:
            name_b = st.selectbox("Compound B", DRUG_NAMES, index=DRUG_NAMES.index("Fluconazole"))
        drug_a, drug_b = DRUGS[name_a], DRUGS[name_b]
    else:
        st.caption("Paste any valid SMILES — the model scores structures it has never seen.")
        smiles_a = st.text_input("Compound A — SMILES", "CC(=O)Oc1ccccc1C(=O)O")
        label_a = st.text_input("Compound A — label", "Candidate TS-001")
        smiles_b = st.text_input("Compound B — SMILES", "CC(C)Cc1ccc(C(C)C(=O)O)cc1")
        label_b = st.text_input("Compound B — label", "Candidate TS-002")
        drug_a = get_drug(label_a) or drug_from_smiles(smiles_a, label_a)
        drug_b = get_drug(label_b) or drug_from_smiles(smiles_b, label_b)

    st.divider()

    # --- Patient covariates
    st.markdown("##### Patient subgroup")
    default = active_case.patient if active_case else PatientProfile()

    age = st.slider("Age (years)", 18, 95, default.age)
    weight = st.slider("Weight (kg)", 35, 140, int(default.weight_kg))
    height = st.slider("Height (cm)", 130, 200, int(default.height_cm),
                       help="Used for body surface area, to turn eGFR into the "
                            "patient's actual kidney clearance.")
    sex = st.radio("Sex", ["Male", "Female"],
                   index=0 if default.sex == "Male" else 1, horizontal=True)
    scr = st.slider("Serum creatinine (mg/dL)", 0.4, 6.0, float(default.serum_creatinine), 0.1,
                    help="Kidney function proxy. Higher = worse kidney function.")
    alt_val = st.slider("ALT (U/L)", 5, 400, int(default.alt),
                        help="Liver enzyme. Normal is under about 40 U/L.")
    albumin = st.slider("Albumin (g/dL)", 1.5, 5.5, float(default.albumin), 0.1,
                        help="Liver synthetic function. Normal is 3.5–5.0 g/dL.")
    bilirubin = st.slider("Bilirubin (mg/dL)", 0.2, 8.0, float(default.bilirubin), 0.1)
    potassium = st.slider("Serum potassium (mmol/L)", 2.5, 6.5,
                          float(default.serum_potassium), 0.1)

    patient = PatientProfile(
        age=age, weight_kg=weight, height_cm=height, sex=sex,
        serum_creatinine=scr, alt=alt_val,
        albumin=albumin, bilirubin=bilirubin, serum_potassium=potassium,
        label=default.label if active_case else "Custom profile",
    )

    st.markdown(
        viz.card("", f"eGFR <b style='color:{viz.INK}'>{patient.egfr():.0f}"
                 f"</b> mL/min/1.73m² · {patient.ckd_stage()}<br>"
                 f"Hepatic function factor <b style='color:{viz.INK}'>"
                 f"{patient.hepatic_function_factor():.2f}</b> · {patient.liver_status()}",
                 viz.ACCENT),
        unsafe_allow_html=True,
    )

    st.divider()

    # --- Organism
    st.markdown("##### Target organism")
    strain_names = list(amr_mod.DEMO_STRAINS)
    default_strain = active_case.strain if active_case and active_case.strain else strain_names[1]
    strain_name = st.selectbox(
        "Isolate", strain_names, index=strain_names.index(default_strain),
        label_visibility="collapsed",
    )
    use_upload = st.checkbox("Use my own FASTA / sequence instead")
    uploaded_seq = None
    if use_upload:
        up = st.file_uploader("FASTA file", type=["fasta", "fa", "fna", "txt"])
        pasted = st.text_area("…or paste a sequence", height=90)
        if up is not None:
            uploaded_seq = amr_mod.parse_fasta(up.read().decode("utf-8", "ignore"))
        elif pasted.strip():
            uploaded_seq = amr_mod.parse_fasta(pasted)

# =============================================================================
# Run all three modules
# =============================================================================
ddi_result = analyze_pair(drug_a, drug_b, ddi_model)
personal = personalize(
    drug_a, drug_b, patient, ddi_result.headline_severity, ddi_result.headline_score
)

if uploaded_seq:
    sequence, display_strain, strain_ctx = uploaded_seq, "Uploaded isolate", ""
else:
    sequence = strain_sequence(strain_name)
    display_strain = strain_name
    strain_ctx = amr_mod.DEMO_STRAINS[strain_name].context

amr_result = amr_mod.analyze_isolate(sequence, _amr_model, display_strain, strain_ctx)

report = build_report(
    case_name=chosen_case, drug_a=drug_a, drug_b=drug_b, patient=patient,
    ddi=ddi_result, personalized=personal, amr=amr_result,
)

# =============================================================================
# Header
# =============================================================================
head_l, head_r = st.columns([3, 1.15])
with head_l:
    st.markdown(
        f"<div style='font-size:2.05rem;font-weight:750;color:{viz.INK};"
        f"letter-spacing:-.028em;line-height:1.15'>Candidate Risk Report</div>"
        f"<div style='color:{viz.MUTED};font-size:.98rem;margin-top:.2rem'>"
        f"{drug_a.name} + {drug_b.name} &nbsp;·&nbsp; {patient.label} "
        f"&nbsp;·&nbsp; {display_strain}</div>",
        unsafe_allow_html=True,
    )
with head_r:
    st.markdown(
        viz.stat("Composite pre-trial risk", f"{report.composite_score:.0f}/100",
                 report.band.upper(), report.color),
        unsafe_allow_html=True,
    )

st.markdown("")

tab_report, tab_ddi, tab_patient, tab_amr, tab_methods = st.tabs(
    ["  Risk Report  ", "  ① Interaction  ", "  ② Patient Response  ",
     "  ③ Resistance  ", "  Methods & Honesty  "]
)

# =============================================================================
# TAB 1 — Unified report
# =============================================================================
with tab_report:
    st.markdown(viz.banner(report.headline(), report.color), unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.9])
    with c1:
        st.plotly_chart(
            viz.risk_gauge(report.composite_score, report.band, report.color),
            width="stretch", config={"displayModeBar": False},
        )
        if report.amr_applicable:
            st.markdown(
                f"<div style='color:{viz.MUTED};font-size:.78rem;text-align:center;"
                f"margin-top:-1rem'>60% interaction &amp; toxicity "
                f"({report.ddi_component:.0f}/100) &nbsp;·&nbsp; 40% resistance "
                f"({report.amr_component:.0f}/100)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='color:{viz.MUTED};font-size:.78rem;text-align:center;"
                f"margin-top:-1rem'>Interaction &amp; toxicity only — no "
                "antibacterial in this pairing</div>",
                unsafe_allow_html=True,
            )

    with c2:
        m1, m2, m3 = st.columns(3)
        base_lbl = SEVERITY_LEVELS[report.ddi.headline_severity]
        adj_lbl = SEVERITY_LEVELS[personal.adjusted_severity]
        with m1:
            st.markdown(
                viz.stat("① Interaction (population)", base_lbl,
                         f"{len(ddi_result.mechanisms)} pathway(s) found",
                         viz.RISK_COLORS[base_lbl]),
                unsafe_allow_html=True)
        with m2:
            arrow = "▲ escalated" if personal.escalated else (
                "▼ reduced" if personal.de_escalated else "= unchanged")
            st.markdown(
                viz.stat("② This patient subgroup", adj_lbl, arrow,
                         viz.RISK_COLORS[adj_lbl]),
                unsafe_allow_html=True)
        with m3:
            lvl, col = amr_result.risk_level()
            st.markdown(
                viz.stat("③ Resistance burden", lvl,
                         f"{len(amr_result.resistant_classes())}/"
                         f"{len(amr_mod.ANTIBIOTIC_CLASSES)} classes resisted", col),
                unsafe_allow_html=True)

        st.markdown("")
        st.markdown("##### Key findings")
        for f in report.findings[:5]:
            st.markdown(
                viz.card(f.title, f.detail, f.color, f.module.split("·")[0].strip()),
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### Recommended protocol actions")
    st.caption(
        "Concrete changes an R&D team could make to the trial design before "
        "committing to human studies."
    )
    for i, action in enumerate(report.actions, 1):
        st.markdown(
            f"<div style='background:{viz.PANEL};border:1px solid {viz.PANEL_EDGE};"
            f"border-radius:9px;padding:.8rem 1.05rem;margin-bottom:.5rem;"
            f"color:{viz.MUTED};line-height:1.6;font-size:.9rem'>"
            f"<b style='color:{viz.ACCENT}'>{i}.</b> {viz.md_bold(action)}</div>",
            unsafe_allow_html=True,
        )

    if len(report.findings) > 5:
        with st.expander(f"All {len(report.findings)} findings"):
            for f in report.findings:
                st.markdown(viz.card(f.title, f.detail, f.color, f.module),
                            unsafe_allow_html=True)

# =============================================================================
# TAB 2 — Interaction
# =============================================================================
with tab_ddi:
    st.markdown("### Module 1 · Drug-drug interaction")
    st.caption(
        "Two independent layers: a pharmacology knowledge base that explains the "
        "mechanism, and a machine-learning model that sees only molecular structure."
    )

    mol_l, mol_m, mol_r = st.columns([1, 1, 1.55])
    with mol_l:
        st.markdown(f"**{drug_a.name}**")
        st.markdown(f"<div class='molbox'>{viz.molecule_svg(drug_a.smiles)}</div>",
                    unsafe_allow_html=True)
        st.caption(drug_a.drug_class)
    with mol_m:
        st.markdown(f"**{drug_b.name}**")
        st.markdown(f"<div class='molbox'>{viz.molecule_svg(drug_b.smiles)}</div>",
                    unsafe_allow_html=True)
        st.caption(drug_b.drug_class)
    with mol_r:
        sev_label = SEVERITY_LEVELS[ddi_result.headline_severity]
        st.markdown(
            viz.banner(
                f"<b>{sev_label.upper()}</b> — {ddi_result.plain_summary()}",
                viz.RISK_COLORS[sev_label],
            ),
            unsafe_allow_html=True,
        )
        if ddi_result.ml_probability is not None:
            st.plotly_chart(
                viz.screening_probability_chart(
                    ddi_result.ml_probability, SCREENING_THRESHOLD
                ),
                width="stretch", config={"displayModeBar": False},
            )
            st.caption(
                "Layer B reads molecular structure only — no pharmacology profile. "
                "It answers one question (is this worth expert review?) rather "
                "than predicting severity, because with 39 drugs a 4-class "
                "severity model is not honest to present."
            )

    if ddi_result.models_agree is not None:
        kb_significant = ddi_result.kb_severity >= 2
        if ddi_result.models_agree:
            verdict = "clinically significant" if kb_significant else "not significant"
            st.success(
                f"**Both layers agree** — the structure-only filter independently "
                f"called this pairing **{verdict}**, matching the mechanism "
                "knowledge base without ever seeing either drug's pharmacology profile."
            )
        else:
            ml_says = "significant" if ddi_result.ml_flags_significant else "not significant"
            kb_says = "significant" if kb_significant else "not significant"
            st.warning(
                f"**The two layers disagree.** The knowledge base says this pairing "
                f"is **{kb_says}** ({SEVERITY_LEVELS[ddi_result.kb_severity]}); the "
                f"structure-only filter says **{ml_says}**. The knowledge base is "
                "authoritative for drugs it covers — the disagreement is shown "
                "rather than hidden, because it marks exactly the kind of case "
                "where a novel compound's prediction would be least reliable."
            )
    elif not ddi_result.from_knowledge_base:
        st.info(
            "**Novel structure.** One or both compounds are not in the knowledge "
            "base, so there is no curated mechanism to report — the assessment "
            "above comes purely from the structure-only filter. This is the "
            "intended use case for a new candidate, and also its least certain "
            "one: a flagged novel pair is mapped to *Moderate* rather than "
            "*Severe*, because a binary filter cannot tell those apart."
        )

    st.divider()
    st.markdown("#### Interaction pathways identified")
    if ddi_result.mechanisms:
        for m in sorted(ddi_result.mechanisms, key=lambda x: -x.severity):
            kind_label = {
                "cyp_inhibition": "Enzyme inhibition", "cyp_induction": "Enzyme induction",
                "pgp": "Transporter", "prodrug": "Prodrug activation blocked",
                "renal": "Kidney competition", "additive_pd": "Additive effect",
                "opposing": "Opposing effect",
            }.get(m.kind, m.kind)
            col = (viz.RISK_COLORS["Severe"] if m.severity >= 2.4
                   else viz.RISK_COLORS["Moderate"] if m.severity >= 1.5
                   else viz.RISK_COLORS["Low"] if m.severity > 0
                   else viz.RISK_COLORS["Info"])
            st.markdown(
                viz.card(kind_label + (f" · {m.enzyme}" if m.enzyme else ""), m.text, col),
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            viz.card("No interaction pathway found", ddi_result.plain_summary(),
                     viz.RISK_COLORS["Low"]),
            unsafe_allow_html=True,
        )

    with st.expander("Pharmacology profiles behind these mechanisms"):
        pc1, pc2 = st.columns(2)
        for col, d in ((pc1, drug_a), (pc2, drug_b)):
            with col:
                st.markdown(f"**{d.name}** — {d.drug_class}")
                rows = []
                if d.cyp_substrate:
                    rows.append(f"Cleared by: {', '.join(d.cyp_substrate)}")
                if d.cyp_inhibitor:
                    rows.append(f"Inhibits: {', '.join(d.cyp_inhibitor)}")
                if d.cyp_inducer:
                    rows.append(f"Induces: {', '.join(d.cyp_inducer)}")
                if d.pgp_substrate:
                    rows.append("Transported by P-glycoprotein")
                if d.pgp_inhibitor:
                    rows.append("Inhibits P-glycoprotein")
                rows.append(f"Renally excreted unchanged: {d.fe*100:.0f}%")
                st.markdown("\n".join(f"- {r}" for r in rows))
                if d.flags():
                    st.markdown(
                        " ".join(viz.pill(FLAG_LABELS[f].capitalize(), viz.RISK_COLORS["High"])
                                 for f in d.flags()),
                        unsafe_allow_html=True,
                    )
                if d.note:
                    st.caption(d.note)

# =============================================================================
# TAB 3 — Patient response
# =============================================================================
with tab_patient:
    st.markdown("### Module 2 · Personalised toxicity simulation")
    st.caption(
        "The same drug pair behaves differently in different patients. This "
        "module recomputes risk from the subgroup's kidney and liver function."
    )

    st.warning(
        "**Simplified educational approximation.** Built on real pharmacokinetic "
        "equations (CKD-EPI 2021 eGFR; clearance-weighted organ scaling), but the "
        "hepatic impairment factor and the risk weights are our own calibration. "
        "This is an R&D prioritisation aid — **not a validated clinical dosing "
        "tool, and not for treating patients.**"
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        gfr = patient.egfr()
        col = (viz.RISK_COLORS["Severe"] if gfr < 30 else
               viz.RISK_COLORS["High"] if gfr < 60 else
               viz.RISK_COLORS["Moderate"] if gfr < 90 else viz.RISK_COLORS["Low"])
        st.markdown(viz.stat("Kidney function (eGFR)", f"{gfr:.0f}",
                             f"mL/min/1.73m² · {patient.renal_category()}", col),
                    unsafe_allow_html=True)
    with k2:
        hf = patient.hepatic_function_factor()
        col = (viz.RISK_COLORS["Severe"] if hf < 0.45 else
               viz.RISK_COLORS["High"] if hf < 0.65 else
               viz.RISK_COLORS["Moderate"] if hf < 0.85 else viz.RISK_COLORS["Low"])
        st.markdown(viz.stat("Hepatic function factor", f"{hf:.2f}",
                             patient.liver_status(), col), unsafe_allow_html=True)
    with k3:
        base_lbl = SEVERITY_LEVELS[personal.base_severity]
        st.markdown(viz.stat("Population baseline risk", base_lbl,
                             f"score {personal.base_score:.2f}",
                             viz.RISK_COLORS[base_lbl]), unsafe_allow_html=True)
    with k4:
        adj_lbl = SEVERITY_LEVELS[personal.adjusted_severity]
        st.markdown(viz.stat("Risk in this subgroup", adj_lbl,
                             f"score {personal.adjusted_score:.2f}",
                             viz.RISK_COLORS[adj_lbl]), unsafe_allow_html=True)

    st.markdown("")
    if personal.escalated:
        st.markdown(viz.banner(
            f"⚠️ {personal.direction_text} Screening this pair only against an "
            "average adult would have under-stated the danger to this subgroup.",
            viz.RISK_COLORS["High"]), unsafe_allow_html=True)
    elif personal.de_escalated:
        st.markdown(viz.banner(personal.direction_text, viz.RISK_COLORS["Low"]),
                    unsafe_allow_html=True)
    else:
        st.markdown(viz.banner(personal.direction_text, viz.RISK_COLORS["Info"]),
                    unsafe_allow_html=True)

    ec1, ec2 = st.columns([1.3, 1])
    with ec1:
        st.markdown("#### Predicted drug exposure")
        st.plotly_chart(viz.exposure_chart(personal.exposures),
                        width="stretch", config={"displayModeBar": False})
        for e in personal.exposures:
            st.markdown(
                f"<div style='color:{viz.MUTED};font-size:.87rem;margin-bottom:.35rem'>"
                f"• {e.plain_text()} Suggested maintenance dose: "
                f"<b style='color:{viz.INK}'>{e.recommended_dose_fraction*100:.0f}%"
                f"</b> of standard.</div>",
                unsafe_allow_html=True,
            )
    with ec2:
        st.markdown("#### Why the risk moved")
        if personal.reasons:
            for r in personal.reasons:
                st.markdown(viz.card("Risk factor", r, viz.RISK_COLORS["High"]),
                            unsafe_allow_html=True)
        for g in personal.protective:
            st.markdown(viz.card("Risk-reducing factor", g, viz.RISK_COLORS["Low"]),
                        unsafe_allow_html=True)
        if not personal.reasons and not personal.protective:
            st.markdown(
                viz.card("No subgroup-specific modifiers",
                         "This profile's organ function does not meaningfully change "
                         "either drug's clearance, so the population baseline applies.",
                         viz.RISK_COLORS["Info"]),
                unsafe_allow_html=True,
            )

    with st.expander("How the exposure numbers are calculated"):
        st.markdown(
            f"""
**Step 1 — kidney function.** CKD-EPI 2021 (the equation FDA's 2024 renal
guidance recommends), from creatinine, age and sex:

`eGFR = 142 × min(SCr/κ, 1)^α × max(SCr/κ, 1)^−1.2 × 0.9938^age` (× 1.012 if female)

For this profile: eGFR = **{patient.egfr():.1f} mL/min/1.73m²** →
FDA category **{patient.renal_category()}**.

A drug is cleared by the patient's real kidney, not a standard-sized one, so
eGFR is converted to absolute mL/min using body surface area
({patient.body_surface_area():.2f} m²): **{patient.absolute_egfr():.1f} mL/min**.

Renal function factor KF = absolute eGFR ÷ 90 (the lower edge of "normal"),
capped at 1 = **{patient.renal_function_factor():.3f}**

*For reference, Cockcroft-Gault CrCl (used by older drug labels) =
{patient.creatinine_clearance():.1f} mL/min.*

**Step 2 — liver function.** A simplified penalty from ALT, albumin, bilirubin
and age gives HF = **{patient.hepatic_function_factor():.3f}**. *(This step is our
approximation, not a validated score — real practice uses Child-Pugh.)*

**Step 3 — combine by clearance route.** Each drug is cleared partly by kidney
(fraction `fe`) and partly by liver (`1 − fe`):

`CL_patient / CL_normal = fe × KF + (1 − fe) × HF`

**Step 4 — exposure.** Since AUC = Dose ÷ Clearance, exposure scales inversely:

`exposure ratio = 1 ÷ clearance ratio`
"""
        )
        for d, e in zip((drug_a, drug_b), personal.exposures):
            st.markdown(
                f"- **{d.name}** (fe = {d.fe:.2f}): "
                f"CL ratio = {d.fe:.2f}×{patient.renal_function_factor():.3f} + "
                f"{d.fh:.2f}×{patient.hepatic_function_factor():.3f} = "
                f"**{e.clearance_ratio:.3f}** → exposure **{e.exposure_ratio:.2f}×** normal"
            )

    st.markdown("#### Trial design · renal impairment study")
    st.caption(
        "Which kidney study each drug needs, and whether patients with impaired "
        "kidneys can join Phase 2/3. Follows the structure of FDA's 2024 renal "
        "impairment guidance; the cut-offs are our own heuristics, and this is "
        "not regulatory-grade PBPK. Does not depend on the patient profile above."
    )
    if PK_VAL.get("design_total"):
        st.caption(
            f"Checked against real renal studies: design call correct for "
            f"{PK_VAL['design_correct']} of {PK_VAL['design_total']} drugs, "
            f"{PK_VAL['design_missed_full']} full studies missed. Exposure "
            f"numbers run about {1 - PK_VAL['bias']:.0%} low on average "
            "(see Methods)."
        )
    dc1, dc2 = st.columns(2)
    for col, d in ((dc1, drug_a), (dc2, drug_b)):
        plan = renal_study_plan(d)
        with col:
            color = (viz.RISK_COLORS["High"] if plan.design == "Full study"
                     else viz.RISK_COLORS["Low"])
            st.markdown(viz.card(f"{d.name} — {plan.design}", plan.design_detail, color),
                        unsafe_allow_html=True)
            st.markdown(
                "| Kidney function | Typical eGFR | Predicted exposure | Phase 2/3 |\n"
                "|---|---|---|---|\n"
                + "\n".join(
                    f"| {c.category} | {c.typical_egfr:.0f} | {c.exposure_ratio:.2f}× "
                    f"| {c.enrolment} |"
                    for c in plan.categories
                )
            )
            if plan.model_supported_candidate:
                st.caption(
                    "Exposure barely moves even in severe impairment: a candidate "
                    "to discuss a model-supported approach (PBPK plus population "
                    "PK from Phase 2/3) with the regulator instead of a "
                    "standalone study."
                )

# =============================================================================
# TAB 4 — Resistance (Module 3)
#
# Seven independent reports, one per sub-tab. Deliberately NOT merged into a
# single verdict: each feature answers a different question, carries its own
# evidence, and is downloadable on its own. A combined number would hide
# exactly the disagreements a portfolio team needs to see.
# =============================================================================
with tab_amr:
    st.markdown("### Module 3 · Antimicrobial resistance intelligence")
    st.caption(
        "Seven separate analyses of the same isolate. Each produces its own "
        "report, its own verdict and its own downloadable summary — nothing is "
        "collapsed into one score."
    )

    # Shared inputs for every sub-report below.
    candidate_ab = report.candidate_antibiotic
    cand_name = candidate_ab.name if candidate_ab else f"{drug_a.name}/{drug_b.name}"
    cand_class = candidate_ab.antibiotic_class if candidate_ab else None
    detected = amr_result.detected_genes

    mobility_rpt = mech_mod.build_mobility_report(detected, display_strain)
    rescue_rpt = mech_mod.build_rescue_report(
        detected, cand_class, cand_name, display_strain, amr_mod.ANTIBIOTIC_CLASSES
    )

    (
        sub_core, sub_runway, sub_rescue, sub_grid,
        sub_novel, sub_value, sub_spread,
    ) = st.tabs([
        "🧬 Core Scan", "⏳ Runway", "🔧 Rescue", "📊 Portfolio",
        "🕵️ Unknown", "💰 Value", "📡 Spread",
    ])

    # -------------------------------------------------------------- Core scan
    with sub_core:
        lvl, lvl_col = amr_result.risk_level()
        st.markdown(viz.banner(amr_result.plain_summary(), lvl_col), unsafe_allow_html=True)
        if amr_result.context:
            st.caption(f"**Why this organism matters:** {amr_result.context}")

        ac1, ac2 = st.columns([1.25, 1])
        with ac1:
            st.markdown("#### Predicted resistance by antibiotic class")
            st.plotly_chart(
                viz.resistance_chart(amr_result.probabilities, amr_mod.LAST_RESORT),
                width="stretch", config={"displayModeBar": False},
            )
        with ac2:
            st.markdown("#### Resistance genes detected")
            if detected:
                for note in amr_result.mechanism_notes():
                    gene_name = note.split("**")[1]
                    gene = amr_mod.RESISTANCE_GENES[gene_name]
                    is_last = any(c in amr_mod.LAST_RESORT for c in gene.confers_resistance_to)
                    st.markdown(
                        viz.card(
                            gene_name,
                            note.split("— ", 1)[1] if "— " in note else note,
                            viz.RISK_COLORS["Critical"] if is_last else viz.RISK_COLORS["High"],
                            mech_mod.MECHANISM_LABELS.get(gene.mechanism_family, ""),
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    viz.card(
                        "No resistance determinants found",
                        "No marker cassette matched this sequence. All screened "
                        "antibiotic classes are predicted to remain active.",
                        viz.RISK_COLORS["Low"],
                    ),
                    unsafe_allow_html=True,
                )

            if candidate_ab:
                st.markdown("#### Impact on this candidate")
                if not report.amr_class_screened:
                    st.markdown(
                        viz.card(
                            f"{candidate_ab.name} ({cand_class})",
                            f"The <b>{cand_class}</b> class is not among the "
                            f"{len(amr_mod.ANTIBIOTIC_CLASSES)} classes Module 3 "
                            "screens, so no prediction is available. This is "
                            "<b>not checked</b>, not <b>no resistance found</b>.",
                            viz.RISK_COLORS["Info"],
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    p = amr_result.probabilities.get(cand_class, 0.0)
                    st.markdown(
                        viz.card(
                            f"{candidate_ab.name} ({cand_class})",
                            f"Predicted resistance of this organism to the "
                            f"{cand_class} class: <b>{p:.0%}</b>. "
                            + ("This candidate would likely fail against this isolate."
                               if p >= amr_mod.RESISTANCE_CALL_THRESHOLD else
                               "This candidate remains viable against this isolate."),
                            viz.RISK_COLORS["Critical"] if p >= amr_mod.RESISTANCE_CALL_THRESHOLD else viz.RISK_COLORS["Low"],
                        ),
                        unsafe_allow_html=True,
                    )

        with st.expander("Sequence and pipeline detail"):
            st.markdown(
                f"""
- Sequence length analysed: **{len([c for c in sequence.upper() if c in 'ACGT']):,} bp**
- Feature representation: **{4**amr_mod.KMER_K} normalised {amr_mod.KMER_K}-mer frequencies**
- Classifier: multi-label Random Forest, one resistance call per antibiotic class
- Gene identification: sliding-window k-mer cosine similarity against reference profiles

The featurisation and classification steps are exactly what would run on real
sequencing data — upload a real FASTA in the sidebar and the same pipeline executes.
"""
            )
            st.code(sequence[:600] + ("…" if len(sequence) > 600 else ""), language="text")

    # ----------------------------------------------------------- 1. Runway
    with sub_runway:
        st.markdown("#### ⏳ Feature 1 — Resistance Runway")
        st.caption(
            "Every other tool reports resistance today. A candidate entering "
            "development now reaches market in roughly a decade — this projects "
            "the resistance it will actually meet at launch, from real ICMR "
            "national surveillance."
        )

        rc1, rc2, rc3 = st.columns([1, 1, 1.4])
        with rc1:
            launch_year = st.slider(
                "Projected launch year", 2026, 2045, 2035, 1,
                help="When would this candidate realistically reach market?",
            )
        with rc2:
            src_key = st.selectbox(
                "Surveillance source",
                list(surv_mod.SOURCES),
                format_func=lambda k: surv_mod.SOURCES[k].name.split("(")[0].strip(),
            )
        with rc3:
            organism = surv_mod.organism_for_strain(display_strain)
            runway_class = st.selectbox(
                "Antibiotic class to project",
                amr_mod.ANTIBIOTIC_CLASSES,
                index=(
                    amr_mod.ANTIBIOTIC_CLASSES.index(cand_class)
                    if cand_class in amr_mod.ANTIBIOTIC_CLASSES else 2
                ),
            )

        runway_rpt = surv_mod.build_runway_report(
            organism=organism,
            antibiotic_class=runway_class,
            candidate_name=cand_name,
            launch_year=launch_year,
            source_key=src_key,
            mobility_multiplier=mobility_rpt.multiplier,
            mobility_note=mobility_rpt.runway_note(),
        )

        st.markdown(viz.banner(runway_rpt.headline(), runway_rpt.color), unsafe_allow_html=True)

        if runway_rpt.available:
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(viz.stat("Today", f"{runway_rpt.current_pct:.0f}%",
                                     f"observed {runway_rpt.current_year}",
                                     viz.INK), unsafe_allow_html=True)
            with k2:
                st.markdown(viz.stat(f"At {launch_year}", f"{runway_rpt.launch_pct:.0f}%",
                                     f"range {runway_rpt.launch_lo:.0f}–{runway_rpt.launch_hi:.0f}%",
                                     runway_rpt.color), unsafe_allow_html=True)
            with k3:
                rw = runway_rpt.runway_years
                st.markdown(viz.stat("Runway",
                                     f"{rw:.0f} yr" if rw is not None else "beyond horizon",
                                     f"crosses {runway_rpt.crossing_year}" if runway_rpt.crossing_year
                                     else f"stays under {surv_mod.VIABILITY_THRESHOLD:.0f}%",
                                     runway_rpt.color), unsafe_allow_html=True)
            with k4:
                st.markdown(viz.stat("Verdict", runway_rpt.verdict,
                                     f"{runway_rpt.forecast.annual_change_pp:+.1f} pp/year",
                                     runway_rpt.color), unsafe_allow_html=True)

            st.plotly_chart(viz.runway_chart(runway_rpt), width="stretch",
                            config={"displayModeBar": False})
            st.markdown(viz.card("What this means", runway_rpt.plain_summary(),
                                 runway_rpt.color), unsafe_allow_html=True)

            with st.expander("Evidence and provenance"):
                for line in runway_rpt.evidence_lines():
                    st.markdown(f"- {line}")
                st.caption(surv_mod.PEER_REVIEWED_CITATION)

            # Cross-source comparison: official programmes disagree, and we show it.
            comparison = {}
            for key in surv_mod.SOURCES:
                drugs = surv_mod.drugs_for_class(organism or "", runway_class, key)
                if not drugs:
                    continue
                blk = surv_mod.SURVEILLANCE_DATA[key][organism]
                years = sorted({y for ab in drugs for y in blk[ab]})
                comparison[surv_mod.SOURCES[key].name.split("(")[0].strip()] = {
                    y: sum(blk[ab][y] for ab in drugs if y in blk[ab])
                       / len([ab for ab in drugs if y in blk[ab]])
                    for y in years
                }
            if len(comparison) > 1:
                with st.expander("Do the surveillance sources agree? (they do not)"):
                    st.plotly_chart(viz.source_comparison_chart(comparison),
                                    width="stretch",
                                    config={"displayModeBar": False})
                    st.caption(
                        "Independent national programmes report different figures "
                        "for the same organism, year and drug class — driven by "
                        "specimen mix and which laboratories participate. This is "
                        "a documented norm in surveillance, and we show every "
                        "series rather than choosing the one that suits us."
                    )
        else:
            st.markdown(viz.card("No projection available", runway_rpt.plain_summary(),
                                 viz.RISK_COLORS["Info"]), unsafe_allow_html=True)

        _download("runway", "Resistance Runway", runway_rpt, display_strain)

    # ---------------------------------------------------------- 2. Rescue
    with sub_rescue:
        st.markdown("#### 🔧 Feature 2 — Mechanism-Aware Rescue Strategy")
        st.caption(
            "A resistance verdict is not a decision. Two isolates can both read "
            "'carbapenem resistant' and imply opposite business outcomes — one "
            "fixed by a partner drug, the other requiring a new programme. This "
            "reports which one you are looking at."
        )

        st.markdown(viz.banner(rescue_rpt.headline(), rescue_rpt.color), unsafe_allow_html=True)

        rr1, rr2 = st.columns([1, 1])
        with rr1:
            st.markdown(viz.stat("Verdict", rescue_rpt.verdict,
                                 f"{len(rescue_rpt.blocking)} determinant(s) hit this class",
                                 rescue_rpt.color), unsafe_allow_html=True)
        with rr2:
            st.markdown(viz.stat("Recommended action",
                                 rescue_rpt.recommendation().split(".")[0],
                                 cand_name, rescue_rpt.color), unsafe_allow_html=True)

        st.markdown("")
        st.markdown(viz.card("What this means", rescue_rpt.plain_summary(),
                             rescue_rpt.color), unsafe_allow_html=True)

        if rescue_rpt.contrast_note():
            st.markdown(
                viz.card("Why the mechanism matters, not just the class",
                         rescue_rpt.contrast_note(), viz.ACCENT),
                unsafe_allow_html=True,
            )

        if rescue_rpt.blocking:
            st.markdown("##### Determinants blocking this candidate's class")
            for gv in rescue_rpt.blocking:
                icon = {"yes": "🟡", "partial": "🟠", "no": "🔴"}.get(gv.rescue_viable, "⚪")
                body = (
                    f"<b>Mechanism:</b> {gv.mechanism_label}"
                    + (f" — {gv.gene.enzyme_subtype}" if gv.gene.enzyme_subtype else "")
                    + f"<br><i>{gv.analogy}</i>"
                    + f"<br><br><b>Rescue:</b> {gv.gene.rescue_strategy}"
                    + f"<br><br>{gv.gene.rescue_note}"
                )
                st.markdown(
                    viz.card(f"{icon} {gv.name} · detected at {gv.confidence:.0%}",
                             body,
                             viz.RISK_COLORS["Severe"] if gv.rescue_viable == "no"
                             else viz.RISK_COLORS["High"] if gv.rescue_viable == "partial"
                             else viz.RISK_COLORS["Moderate"],
                             gv.rescue_viable.upper()),
                    unsafe_allow_html=True,
                )

        if rescue_rpt.other:
            with st.expander(f"{len(rescue_rpt.other)} determinant(s) affecting other classes"):
                for gv in rescue_rpt.other:
                    st.markdown(
                        viz.card(
                            f"{gv.name} · {gv.confidence:.0%}",
                            f"<b>{gv.mechanism_label}</b> — defeats "
                            f"{', '.join(gv.gene.confers_resistance_to)}. "
                            f"{gv.gene.rescue_note}",
                            viz.RISK_COLORS["Info"],
                        ),
                        unsafe_allow_html=True,
                    )

        # --- Per-drug resolution -----------------------------------------
        # The class label is the coarsest useful unit, and in at least one
        # place it is too coarse: "Aminoglycosides resistant" is a different
        # finding depending on which gene caused it. This is where the
        # aac(6')-Ib over-call the ICMR cross-check exposed gets answered.
        if cand_class and rescue_rpt.blocking:
            resolutions = pd_mod.resolve_detected(detected, cand_class)
            resolved = [r for r in resolutions if r.resolved]
            if resolved:
                st.markdown("##### Which individual drugs, not just which class")
                verdict = pd_mod.class_verdict(resolutions, cand_class)
                if verdict:
                    st.markdown(
                        viz.banner(verdict, viz.RISK_COLORS["Moderate"]),
                        unsafe_allow_html=True,
                    )
                for r in resolved:
                    body = r.summary()
                    if r.disagreement:
                        body += (
                            "<br><br><b>Databases disagree here:</b> "
                            + r.disagreement
                        )
                    body += (
                        "<br><br><i>Source: "
                        + "; ".join(r.sources[:2])
                        + "</i>"
                    )
                    st.markdown(
                        viz.card(
                            f"{r.gene} · {cand_class} in detail",
                            body,
                            viz.RISK_COLORS["Low"] if r.spared
                            else viz.RISK_COLORS["High"],
                            f"{len(r.defeated)} drug(s) defeated",
                        ),
                        unsafe_allow_html=True,
                    )
                st.caption(
                    "Per-drug relations come from the CARD ontology and NCBI "
                    "AMRFinderPlus, not from our model. Where a database simply "
                    "does not curate a drug we say nothing about it — absence "
                    "from a reference list means 'not recorded', never "
                    "'susceptible'."
                )

        _download("rescue", "Rescue Strategy", rescue_rpt, display_strain)

    # --------------------------------------------------------- 3. Portfolio
    with sub_grid:
        st.markdown("#### 📊 Feature 3 — Portfolio Screen")
        st.caption(
            "A committee does not hold one candidate against one organism — it "
            "holds a pipeline and a budget for two. This screens every candidate "
            "against every target organism at once."
        )

        antibacterials = sorted(
            n for n, d in DRUGS.items() if d.antibiotic_class
        )
        pc1, pc2 = st.columns(2)
        with pc1:
            picked_drugs = st.multiselect(
                "Candidate compounds",
                antibacterials,
                default=[n for n in ["Meropenem", "Levofloxacin", "Linezolid", "Amoxicillin"]
                         if n in antibacterials],
            )
        with pc2:
            picked_orgs = st.multiselect(
                "Target organisms",
                list(amr_mod.DEMO_STRAINS),
                default=[s for s in [
                    "K. pneumoniae (carbapenem-resistant)",
                    "E. coli ST131 (ESBL)",
                    "S. aureus MRSA (hospital-acquired)",
                    "S. Typhi H58 (XDR, South Asia)",
                ] if s in amr_mod.DEMO_STRAINS],
            )

        if not picked_drugs or not picked_orgs:
            st.info("Pick at least one candidate and one organism to build the grid.")
        else:
            cands = [
                port_mod.Candidate(name=n, antibiotic_class=DRUGS[n].antibiotic_class or "")
                for n in picked_drugs
            ]

            def _rescue_verdict(amr_rep, cand):
                r = mech_mod.build_rescue_report(
                    amr_rep.detected_genes, cand.antibiotic_class, cand.name,
                    amr_rep.strain_name, amr_mod.ANTIBIOTIC_CLASSES,
                )
                return r.verdict

            grid = port_mod.screen_portfolio(
                cands, picked_orgs, _isolate_for, _rescue_verdict
            )
            port_mod.attach_cross_check(
                grid,
                {s: amr_mod.DEMO_STRAINS[s].genes for s in picked_orgs
                 if s in amr_mod.DEMO_STRAINS},
                {g: gg.confers_resistance_to
                 for g, gg in amr_mod.RESISTANCE_GENES.items()},
            )

            st.markdown(viz.banner(grid.headline(), grid.color), unsafe_allow_html=True)
            st.plotly_chart(viz.portfolio_heatmap(grid), width="stretch",
                            config={"displayModeBar": False})

            gc1, gc2 = st.columns([1.3, 1])
            with gc1:
                st.markdown(viz.card("Portfolio recommendation",
                                     grid.plain_summary() + " " + grid.recommendation(),
                                     grid.color), unsafe_allow_html=True)
            with gc2:
                st.markdown("##### Ranking")
                for i, s in enumerate(grid.scores, 1):
                    st.markdown(
                        viz.stat(
                            f"{i}. {s.name}",
                            f"{s.viable_against}/{s.total_assessed} viable",
                            f"{s.antibiotic_class}"
                            + (f" · {s.salvageable} rescuable" if s.salvageable else ""),
                            viz.RISK_COLORS["Low"] if s.viability_rate >= 0.6
                            else viz.RISK_COLORS["Moderate"] if s.viability_rate > 0
                            else viz.RISK_COLORS["Severe"],
                        ),
                        unsafe_allow_html=True,
                    )


            # --- Validation against real Indian surveillance -----------------
            # Feature 3 generates no data of its own, so this is where it earns
            # a validation claim rather than inheriting one.
            cc = grid.cross_check
            if cc is not None and cc.compared:
                st.markdown("")
                st.markdown("##### ✓ Cross-checked against real ICMR national data")
                st.markdown(
                    viz.banner(cc.headline(), cc.color), unsafe_allow_html=True
                )

                x1, x2, x3, x4 = st.columns(4)
                with x1:
                    st.markdown(viz.stat(
                        "Tracks national rate", str(cc.n_aligned),
                        f"within {surv_mod.ALIGNMENT_BAND_PP:.0f} pp",
                        viz.RISK_COLORS["Low"]), unsafe_allow_html=True)
                with x2:
                    st.markdown(viz.stat(
                        "Differs as expected", str(cc.n_as_expected),
                        "resistant strain above / control below", viz.ACCENT),
                        unsafe_allow_html=True)
                with x3:
                    st.markdown(viz.stat(
                        "Flagged", str(len(cc.flagged)),
                        "unexpected direction or wide gap",
                        viz.RISK_COLORS["High"] if cc.flagged
                        else viz.RISK_COLORS["Low"]), unsafe_allow_html=True)
                with x4:
                    st.markdown(viz.stat(
                        "No reference", str(cc.n_no_reference),
                        "ICMR does not report these", viz.RISK_COLORS["Info"]),
                        unsafe_allow_html=True)

                st.markdown(viz.card("What the comparison shows",
                                     cc.plain_summary(), cc.color),
                            unsafe_allow_html=True)

                if cc.flagged:
                    st.markdown("###### Pairings worth a second look")
                    for r in cc.flagged:
                        st.markdown(
                            viz.card(
                                f"{r.organism} × {r.antibiotic_class}",
                                r.explain(),
                                viz.RISK_COLORS["High"],
                                r.status,
                            ),
                            unsafe_allow_html=True,
                        )

                with st.expander(
                    f"Full comparison — all {len(cc.rows)} organism × class pairings"
                ):
                    st.caption(cc.caveat())
                    _rows = [
                        "| Organism | Class | Model | ICMR | Gap | Status |",
                        "|---|---|---|---|---|---|",
                    ]
                    for r in cc.rows:
                        _real = f"{r.real_pct:.0f}%" if r.real_pct is not None else "—"
                        _gap = f"{r.delta:+.0f} pp" if r.delta is not None else "—"
                        _rows.append(
                            f"| {r.organism} | {r.antibiotic_class} "
                            f"| {r.model_pct:.0f}% | {_real} | {_gap} | {r.status} |"
                        )
                    st.markdown("\n".join(_rows))
                    st.caption(
                        f"Reference: {surv_mod.SOURCES[cc.source_key].name}, "
                        f"{cc.reference_year}. Mean gap across the aligned "
                        f"pairings: {cc.aligned_mae_pp:.1f} pp."
                    )

            st.caption(
                f"⏱ {grid.n_combinations} combinations screened in "
                f"{grid.elapsed_seconds:.2f} s. The equivalent grid by laboratory "
                "susceptibility testing is weeks of bench work."
            )
            _download("portfolio", "Portfolio Screen", grid, display_strain)

    # ----------------------------------------------------------- 4. Unknown
    with sub_novel:
        st.markdown("#### 🕵️ Feature 4 — Dark Genome Detector")
        st.caption(
            "Every gene-detection tool is closed-world: it reports what matches "
            "its database and stays silent about everything else. Silence looks "
            "identical to 'all clear'. This looks for what we cannot name."
        )

        novelty_rpt = _novelty_for(sequence, display_strain, len(detected))

        st.markdown(viz.banner(novelty_rpt.headline(), novelty_rpt.color), unsafe_allow_html=True)

        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            st.markdown(viz.stat("Verdict", novelty_rpt.verdict,
                                 f"{novelty_rpt.n_windows} windows scanned",
                                 novelty_rpt.color), unsafe_allow_html=True)
        with nc2:
            st.markdown(viz.stat("Unexplained segments", str(len(novelty_rpt.segments)),
                                 f"{sum(s.length for s in novelty_rpt.segments):,} bp total",
                                 viz.RISK_COLORS["Moderate"] if novelty_rpt.segments
                                 else viz.RISK_COLORS["Low"]), unsafe_allow_html=True)
        with nc3:
            st.markdown(viz.stat(
                "Model confidence",
                novelty_rpt.ood_band.title() if novelty_rpt.ood_available else "Unavailable",
                f"{novelty_rpt.ood_percentile:.0f}th pct of training distance"
                if novelty_rpt.ood_available else "run train.py to enable",
                viz.RISK_COLORS["Low"] if novelty_rpt.ood_band == "IN DISTRIBUTION"
                else viz.RISK_COLORS["High"] if novelty_rpt.ood_band == "BORDERLINE"
                else viz.RISK_COLORS["Severe"] if novelty_rpt.ood_band == "OUT OF DISTRIBUTION"
                else viz.RISK_COLORS["Info"],
            ), unsafe_allow_html=True)

        if novelty_rpt.segments:
            st.plotly_chart(viz.novelty_map(novelty_rpt), width="stretch",
                            config={"displayModeBar": False})
            for seg in novelty_rpt.segments:
                st.markdown(
                    viz.card(f"Unidentified element at {seg.start:,}–{seg.end:,}",
                             seg.describe(), viz.RISK_COLORS["Moderate"]),
                    unsafe_allow_html=True,
                )

        st.markdown(viz.card("What this means", novelty_rpt.plain_summary(),
                             novelty_rpt.color), unsafe_allow_html=True)
        st.markdown(viz.card("Recommended action", novelty_rpt.recommendation(),
                             viz.ACCENT), unsafe_allow_html=True)

        with st.expander("How this detector was validated"):
            st.markdown(
                """
Measured by planting synthetic elements that match no reference, then checking
the twelve unmodified demo strains do not fire:

- **False positives:** 0 of 12 clean strains
- **True positives:** 7 of 7 spiked sequences
- **Separation:** clean strains peak at 0.08–0.12 excess compositional
  divergence; spiked sequences at 0.25–0.69. The detection floor sits between
  them with better than 2x margin on each side.

Known genes are masked before scoring, because a resistance cassette is itself
compositionally foreign — without masking, the detector simply rediscovers the
genes we already found.
"""
            )
        _download("novelty", "Dark Genome Scan", novelty_rpt, display_strain)

    # ------------------------------------------------------------- 5. Value
    with sub_value:
        st.markdown("#### 💰 Feature 5 — Value of Early Detection")
        st.caption(
            "Translating a resistance finding into the only unit a budget "
            "committee approves in. Every input below is yours to change."
        )

        triggered = bool(
            report.amr_class_screened
            and cand_class
            and amr_result.probabilities.get(cand_class, 0.0)
            >= amr_mod.RESISTANCE_CALL_THRESHOLD
        )
        res_pct = (amr_result.probabilities.get(cand_class, 0.0) * 100) if cand_class else 0.0

        with st.expander("Your assumptions — change any of these", expanded=not triggered):
            vc1, vc2 = st.columns(2)
            with vc1:
                st.markdown("**Cost per phase (₹ crore)**")
                costs = {
                    ph: st.slider(ph, 1.0, 600.0, float(econ_mod.DEFAULT_COSTS[ph]), 1.0,
                                  key=f"cost_{ph}")
                    for ph in econ_mod.PHASES
                }
            with vc2:
                st.markdown("**Probability of clearing each phase**")
                ptss = {
                    ph: st.slider(ph, 0.05, 1.0, float(econ_mod.DEFAULT_PTS[ph]), 0.05,
                                  key=f"pts_{ph}")
                    for ph in econ_mod.PHASES
                }
            disc = st.selectbox(
                "Where would this surface without a computational screen?",
                econ_mod.PHASES, index=econ_mod.PHASES.index(econ_mod.DEFAULT_DISCOVERY_PHASE),
            )

        value_rpt = econ_mod.build_value_report(
            candidate_name=cand_name, organism=display_strain,
            resistance_pct=res_pct, triggered=triggered,
            discovery_phase=disc, costs=costs, pts=ptss,
        )

        st.markdown(viz.banner(value_rpt.headline(), value_rpt.color), unsafe_allow_html=True)

        if value_rpt.triggered:
            v1, v2, v3 = st.columns(3)
            with v1:
                st.markdown(viz.stat("Cost avoided", f"₹{value_rpt.cost_avoided:.0f} cr",
                                     f"if caught at {value_rpt.discovery_phase}",
                                     viz.RISK_COLORS["High"]), unsafe_allow_html=True)
            with v2:
                st.markdown(viz.stat("Time avoided", f"{value_rpt.years_avoided:.1f} yr",
                                     "development time", viz.RISK_COLORS["High"]),
                            unsafe_allow_html=True)
            with v3:
                st.markdown(viz.stat("Expected value",
                                     f"₹{value_rpt.expected_value_crore:.0f} cr",
                                     f"weighted by {value_rpt.probability_reaching_discovery:.0%} "
                                     "chance of reaching that phase",
                                     value_rpt.color), unsafe_allow_html=True)
            st.plotly_chart(viz.value_waterfall(value_rpt), width="stretch",
                            config={"displayModeBar": False})

        st.markdown(viz.card("What this means", value_rpt.plain_summary(),
                             value_rpt.color), unsafe_allow_html=True)
        st.warning(value_rpt.disclaimer())
        _download("value", "Value of Early Detection", value_rpt, display_strain)

    # ------------------------------------------------------------ 6. Spread
    with sub_spread:
        st.markdown("#### 📡 Feature 6 — Spread Velocity")
        st.caption(
            "Resistance is not a static fact — it has a speed. Determinants on "
            "mobile elements cross between species; chromosomal mutations only "
            "pass to offspring. That difference changes how long a runway really is."
        )

        st.markdown(viz.banner(mobility_rpt.headline(), mobility_rpt.color), unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(viz.stat("Spread class", mobility_rpt.verdict,
                                 f"{len(mobility_rpt.mobile)} of {len(mobility_rpt.entries)} transferable",
                                 mobility_rpt.color), unsafe_allow_html=True)
        with sc2:
            st.markdown(viz.stat("Runway multiplier", f"{mobility_rpt.multiplier:.1f}×",
                                 "applied to Feature 1", mobility_rpt.color),
                        unsafe_allow_html=True)
        with sc3:
            lr = mobility_rpt.last_resort_mobile()
            st.markdown(viz.stat("Mobile last-resort", str(len(lr)),
                                 ", ".join(e.gene.name for e in lr) if lr else "none detected",
                                 viz.RISK_COLORS["Severe"] if lr else viz.RISK_COLORS["Low"]),
                        unsafe_allow_html=True)

        if mobility_rpt.entries:
            st.plotly_chart(viz.mobility_chart(mobility_rpt), width="stretch",
                            config={"displayModeBar": False})

        st.markdown(viz.card("What this means", mobility_rpt.plain_summary(),
                             mobility_rpt.color), unsafe_allow_html=True)

        if mobility_rpt.entries:
            st.markdown("##### How each determinant travels")
            for e in mobility_rpt.entries:
                st.markdown(
                    viz.card(
                        f"{e.gene.name} · {e.label}",
                        f"<i>{e.analogy}</i><br><br>{e.gene.mobility_note}",
                        viz.RISK_COLORS["Severe"] if e.gene.spread_multiplier >= 2.5
                        else viz.RISK_COLORS["High"] if e.gene.spread_multiplier >= 1.8
                        else viz.RISK_COLORS["Low"],
                        f"{e.gene.spread_multiplier:.1f}x",
                    ),
                    unsafe_allow_html=True,
                )
        _download("spread", "Spread Velocity", mobility_rpt, display_strain)

# =============================================================================
# TAB 5 — Methods & honesty
# =============================================================================
with tab_methods:
    st.markdown("### Methods, validation, and what is simulated")
    st.caption(
        "Everything a judge should know to assess this prototype accurately. "
        "We would rather show the limitations than have them found."
    )

    st.markdown("#### Held-out model performance")
    if METRICS:
        d, a = METRICS.get("ddi", {}), METRICS.get("amr", {})
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**Module 1 — screening filter (structure only)**")
            if "cold_drug_split" in d:
                cs, rs2 = d["cold_drug_split"], d["random_split"]
                st.markdown(
                    f"- **Cold-drug split** (entire drugs removed from training): "
                    f"ROC-AUC **{cs['roc_auc']:.3f}** (±{cs['roc_auc_std']:.3f}), "
                    f"recall **{cs['recall']:.0%}**, precision **{cs['precision']:.0%}**\n"
                    f"- Random pair split: ROC-AUC **{rs2['roc_auc']:.3f}**\n"
                    f"- Pooled over **{cs['n_repeats']}** hold-outs "
                    f"(n = {cs['n_test']} test pairs)\n"
                    f"- Trained on **{d.get('n_pairs', 0)}** pairs from "
                    f"**{d.get('n_drugs', 0)}** drugs; "
                    f"**{d.get('significant_rate', 0):.0%}** are significant"
                )
                st.caption(
                    f"The cold-drug number is the honest one — those drugs were "
                    f"removed entirely, so it measures performance on a genuinely "
                    f"new compound. At the {d.get('screening_threshold', 0.35):.0%} "
                    f"flag threshold the filter catches {cs['recall']:.0%} of "
                    f"significant pairs while flagging {cs['flagged_rate']:.0%} of "
                    "all pairs. Precision is deliberately low: for pre-trial "
                    "screening, a missed interaction costs far more than an "
                    "unnecessary review."
                )
        with mc2:
            st.markdown("**Module 3 — resistance model (k-mer)**")
            if "cold_combination_split" in a:
                acs = a["cold_combination_split"]
                st.markdown(
                    f"- Unseen gene combinations: macro-F1 **{acs['macro_f1']:.3f}**, "
                    f"per-label accuracy **{acs['per_label_accuracy']:.1%}**\n"
                    f"- Random split: macro-F1 **{a['random_split']['macro_f1']:.3f}**\n"
                    f"- Trained on **{a.get('n_isolates', 0)}** synthetic isolates"
                )
                st.caption(
                    "The cold-combination split tests isolates whose exact gene "
                    "combination never appeared in training — the model must "
                    "detect genes compositionally rather than memorise fingerprints."
                )
    else:
        st.info("Run `python train.py` to generate held-out evaluation metrics.")

    if PK_VAL.get("n_observations"):
        v = PK_VAL
        st.markdown("**Module 2 — exposure predictions vs real renal impairment data**")
        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown(
                f"- **{v['n_observations']}** measured exposure changes across "
                f"**{v['n_drugs']}** drugs, from FDA labels and published studies\n"
                f"- Within 2× of the measured value: **{v['within_2x']:.0%}** "
                f"(1.5×: {v['within_1_5x']:.0%})\n"
                f"- Geometric mean fold error **{v['gmfe']:.2f}**; bias "
                f"**{v['bias']:.2f}** (predictions run low)\n"
                f"- **{v['no_change_statements_matched']} of "
                f"{v['n_no_change_statements']}** 'kidney disease does not change "
                "exposure' label statements correctly reproduced"
            )
        with vc2:
            st.markdown(
                f"- Study-design call correct for **{v['design_correct']} of "
                f"{v['design_total']}** drugs\n"
                f"- Full renal studies missed: **{v['design_missed_full']}**; "
                f"unnecessary: **{v['design_unnecessary_full']}**"
            )
            st.caption(
                "Exposure is under-predicted for kidney-cleared drugs, mostly in "
                "moderate impairment — kidney disease also slows the liver, which "
                "this simple model ignores, and the source studies disagree with "
                "each other. The study-design decision is the robust output. "
                "Reproduce with `python validate_pk.py`; every number traces to "
                "a quote and URL in `data/renal_validation.json`."
            )

    st.divider()
    st.markdown("#### Module 3 validation — how we know it helps")
    st.caption(
        "Accuracy alone does not answer 'does this help an R&D team'. These are "
        "five separate checks, each answering a different question. Regenerate "
        "them with `python validate_module3.py`."
    )

    _val_path = MODELS_DIR / "validation.json"
    if _val_path.exists():
        V = json.loads(_val_path.read_text())

        ec = V.get("errors_calibration", {})
        nv = V.get("novelty", {})
        ch = V.get("clade_holdout", {})
        fb = V.get("forecast_backtest", {})

        vt1, vt2, vt3, vt4 = st.columns(4)
        with vt1:
            st.markdown(viz.stat(
                "Very major errors", f"{ec.get('very_major_error_rate', 0):.1%}",
                "predicted S, actually R",
                viz.RISK_COLORS["High"] if ec.get("very_major_error_rate", 0) > 0.03
                else viz.RISK_COLORS["Low"]), unsafe_allow_html=True)
        with vt2:
            st.markdown(viz.stat(
                "Major errors", f"{ec.get('major_error_rate', 0):.2%}",
                "predicted R, actually S", viz.RISK_COLORS["Low"]),
                unsafe_allow_html=True)
        with vt3:
            st.markdown(viz.stat(
                "Clade-held-out F1", f"{ch.get('mean_macro_f1', 0):.3f}",
                f"± {ch.get('std_macro_f1', 0):.3f} over {ch.get('n_clades', 0)} clades",
                viz.RISK_COLORS["Low"]), unsafe_allow_html=True)
        with vt4:
            st.markdown(viz.stat(
                "Forecast error", f"{fb.get('mae_pp', 0):.1f} pp",
                f"{fb.get('within_10pp', 0):.0%} within 10 pp",
                viz.RISK_COLORS["Low"]), unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Real genomes with laboratory AST. This sits ABOVE the constructed
        # figures deliberately: everything below is measured on isolates we
        # built ourselves, and a reader who sees only those walks away with a
        # number that does not describe real bacteria.
        # ------------------------------------------------------------------
        _real_path = MODELS_DIR / "validation_real.json"
        if _real_path.exists():
            R = json.loads(_real_path.read_text())
            rs = R.get("screening", {})
            rd = R.get("data", {})
            st.markdown("---")
            st.markdown("#### The honest test: real genomes, laboratory AST")

            rr1, rr2, rr3 = st.columns(3)
            with rr1:
                st.markdown(viz.stat(
                    "Accuracy on real isolates", f"{rs.get('accuracy', 0):.3f}",
                    f"{rs.get('n_comparisons', 0)} bench comparisons",
                    viz.RISK_COLORS["High"]), unsafe_allow_html=True)
            with rr2:
                st.markdown(viz.stat(
                    "Very major errors", f"{rs.get('very_major_error_rate', 0):.1%}",
                    "predicted S, laboratory R", viz.RISK_COLORS["High"]),
                    unsafe_allow_html=True)
            with rr3:
                st.markdown(viz.stat(
                    "Real isolates tested", f"{rd.get('n_isolates', 0)}",
                    "7 organisms, BV-BRC", viz.RISK_COLORS["Low"]),
                    unsafe_allow_html=True)

            st.markdown(
                f"""
These are **{rd.get('n_isolates', 0)} real bacterial genomes** with
**bench-measured** susceptibility results — broth dilution and disk diffusion,
read against CLSI or EUCAST breakpoints. Rows whose evidence column said
*Computational Method* (another group's machine-learning predictions) were
filtered out, so we never score a prediction against a prediction.

With {rs.get('n_resistant', 0)} resistant and {rs.get('n_susceptible', 0)}
susceptible comparisons, **{rs.get('accuracy', 0):.3f} is approximately
chance**, and no threshold rescues it — the full sweep peaks at 0.503.

**Why.** The model trains on ~900 bp cassettes planted in ~2.6 kb of
background. A real genome is 2.5-7 Mb, where a determinant is about 0.02% of
the sequence, so a whole-genome 5-mer profile is dominated by ordinary
housekeeping DNA. This is architectural, not a threshold that needs nudging.

**What did transfer: gene detection.** On the same genomes it found real
determinants — mecA in 9 *S. aureus*, blaSHV in 5, plus blaNDM-1, blaVIM and
blaCTX-M-15 — at a 13.8% false-resistance rate. It is precise but narrow: 21
reference genes cannot cover real resistance, so it misses 73% of it.

So the claim we defend is that Module 3 **detects known resistance genes in
real genomes with high precision** and resolves two point-mutation
determinants to the specific codon. It does **not** predict laboratory
phenotype on real bacteria, and it is not clinically validated.

*{rd.get('caveat', '')}*
"""
            )
            st.caption(
                "Reproduce: `python -m trialsense.ingest` then "
                "`python validate_real.py`. Everything below this line is "
                "measured on CONSTRUCTED isolates."
            )
            st.markdown("---")

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**1 + 2 · Error types and calibration**")
            st.markdown(
                f"""
Regulators do not score susceptibility devices on accuracy — they count two
error types separately, because the costs are wildly asymmetric.

- **Very major error** (predicted susceptible, actually resistant):
  **{ec.get('very_major_error_rate', 0):.2%}** — {ec.get('very_major_errors', 0)}
  of {ec.get('n_actually_resistant', 0)}. Programme-ending.
- **Major error** (predicted resistant, actually susceptible):
  **{ec.get('major_error_rate', 0):.2%}** — one confirmatory assay recovers it.
- **Brier score:** {ec.get('brier_score', 0):.4f} ·
  **calibration error:** {ec.get('expected_calibration_error', 0):.3f}
"""
            )
            if ec.get("recommended_threshold") is not None:
                st.warning(
                    f"**A finding we are not hiding.** At the default 0.50 cut "
                    f"the very-major-error rate is "
                    f"**{ec['very_major_error_rate']:.2%}**, above the ~1.5% bar "
                    f"commercial susceptibility devices are held to. Lowering the "
                    f"decision threshold to **{ec['recommended_threshold']:.2f}** "
                    f"brings it to **{ec['recommended_vme']:.2%}** while raising "
                    f"major errors only to {ec['recommended_me']:.2%}. That is the "
                    "correct trade for pre-trial screening, and it is the same "
                    "asymmetry Module 1 already ships."
                )
            with st.expander("Reliability — when we say X%, what happens?"):
                st.caption(
                    "A model can be accurate and still dishonest about its own "
                    "confidence. Our mid-range probabilities are visibly "
                    "under-confident: when this model says 55%, the true rate is "
                    "nearer 97%. We report it rather than quoting only the "
                    "headline accuracy."
                )
                for r in ec.get("reliability", []):
                    gap = r["observed"] - r["predicted"]
                    st.markdown(
                        f"- predicted **{r['predicted']:.2f}** → observed "
                        f"**{r['observed']:.2f}** (n={r['n']})"
                        + ("  ⚠️ under-confident" if gap > 0.15 else "")
                    )

        with mc2:
            st.markdown("**3 · Clade-held-out validation**")
            st.markdown(
                f"""
An entire species background is removed from training and the model is tested
only on it. A random split lets near-identical isolates land on both sides and
flatters the score; this is the split that says whether a new lineage would
work.

- macro-F1 **{ch.get('mean_macro_f1', 0):.3f} ± {ch.get('std_macro_f1', 0):.3f}**
  across {ch.get('n_clades', 0)} held-out clades
- worst single clade: **{ch.get('worst_macro_f1', 0):.3f}**
"""
            )
            st.markdown("**4 · Dark Genome detector**")
            _nc_path = MODELS_DIR / "novelty_calibration.json"
            if _nc_path.exists():
                NC = json.loads(_nc_path.read_text())
                th = NC.get("test_half", {})
                shipped_m = th.get("chosen", {})
                old_m = th.get("shipped", {})
                st.markdown(
                    f"""
An earlier version of this panel reported **0 false positives and 100%
sensitivity**. That was measured on 19 cases from the same generator that
produced the training data, and it did not survive real genomes: on 42 real
assemblies the old thresholds flagged something in **every single one**, at
{old_m.get('segments_per_mb', 0):.1f} segments per megabase — roughly a
hundred flags on a 5 Mb genome. Real genomes carry prophages and genomic
islands; generated "clean" strains are uniform by construction.

Recalibrated by splitting the 42 genomes in half, sweeping on one and
reporting on the other. **Held-out half:**

| thresholds | false alarms | sensitivity |
|---|---|---|
| previous | {old_m.get('segments_per_mb', 0):.2f} /Mb | {old_m.get('sensitivity', 0):.0%} |
| **shipped** | **{shipped_m.get('segments_per_mb', 0):.2f} /Mb** | **{shipped_m.get('sensitivity', 0):.0%}** |

Sensitivity is measured with a real 1200 bp segment from the most GC-distant
genome inserted into a real recipient — both real sequence, so this does not
repeat the original mistake of testing a generator against itself.

**The trade-off is real and is not tuned away.** The frontier runs from 100%
sensitivity at ~22 flags/Mb to ~24% at 0.5 flags/Mb: tetranucleotide
composition cannot tell an acquired element from a native genomic island,
because both are foreign to the host core. We chose about eight flags per
genome — reviewable by a person — and accept missing roughly half of true
insertions.
"""
                )
            else:
                st.markdown(
                    f"""
- **False positives:** {nv.get('false_positives', 0)} of
  {nv.get('clean_strains_assessed', 0)} clean strains
  ({nv.get('false_positive_rate', 0):.0%}) — **constructed strains only**
- **Sensitivity:** {nv.get('detected', 0)}/{nv.get('spiked_cases', 0)} planted
  unknown elements found ({nv.get('sensitivity', 0):.0%})

⚠️ These are same-generator cases. Run `python calibrate_novelty.py` for the
real-genome figures.
"""
                )
            st.markdown("**5 · Resistance-trend forecaster**")
            if fb.get("available"):
                st.markdown(
                    f"""
Fitted on years ≤ {fb['cutoff_year']} only, then asked to predict the years it
never saw.

- **{fb['n_predictions']}** held-out predictions across **{fb['n_series']}** series
- mean absolute error **{fb['mae_pp']:.2f} percentage points**
- **{fb['within_5pp']:.0%}** within 5 pp · **{fb['within_10pp']:.0%}** within 10 pp
"""
                )

        gc = V.get("gene_coverage", {})
        if gc:
            with st.expander(
                f"6 · Coverage of India-relevant determinants "
                f"({gc.get('icmr_genes_covered', 0)}/{gc.get('icmr_genes_listed', 0)})"
            ):
                st.caption(gc.get("note", ""))
                for r in gc.get("rows", []):
                    mark = "✅" if r["in_our_reference_set"] else "❌"
                    st.markdown(
                        f"- {mark} **{r['gene']}** — ICMR prevalence "
                        f"{r['icmr_prevalence']:.0%}; carried by "
                        f"{r['demo_strains_carrying']} demo strain(s)"
                    )
                st.info(gc.get("caveat", ""))
    else:
        st.info(
            "Run `python validate_module3.py` to generate the validation figures "
            "(error types, calibration, clade-held-out, novelty detection, "
            "forecast backtest)."
        )

    st.divider()
    st.markdown("#### DNABERT-2 — benchmarked, not just imported")
    _bert_status = bert_mod.check_backend()
    _bench_path = MODELS_DIR / "dnabert2_benchmark.json"
    if _bench_path.exists():
        B = json.loads(_bench_path.read_text())
        st.markdown(
            f"- k-mer + Random Forest: macro-F1 **{B.get('kmer_macro_f1', 0):.4f}**\n"
            f"- DNABERT-2 fine-tuned: macro-F1 **{B.get('dnabert_macro_f1', 0):.4f}**\n"
            f"- **Winner: {B.get('winner', 'n/a')}**"
        )
    st.markdown(
        """
DNABERT-2 is a genomic language model pretrained on **real** bacterial genomes.
Our bundled cassettes are synthetic, assembled from random motifs — so there is
no real biological structure for it to recognise, while the k-mer counter is
near-perfectly matched to how those cassettes were built. We therefore expect it
to **lose** here, and we run the comparison anyway:

> *If the k-mer baseline wins, that is measured evidence our **data** is the
> binding constraint, not our method — which is precisely the argument for
> swapping in real NCBI Pathogen Detection sequences.*

Reporting a benchmark you might lose is a stronger position than reporting an
integration you never evaluated.
"""
    )
    st.caption(
        _bert_status.message()
        + "  ·  GPU fine-tuning: `notebooks/dnabert2_finetune_kaggle.ipynb` "
        "(free Kaggle GPU, ~30 min)."
    )

    st.divider()
    st.markdown("#### What is real vs. simulated")

    real_col, sim_col = st.columns(2)
    with real_col:
        st.markdown(
            f"<div style='color:{viz.RISK_COLORS['Low']};font-weight:700;"
            f"margin-bottom:.4rem'>✓ REAL</div>", unsafe_allow_html=True)
        st.markdown(
            """
- **Molecular structures.** All 39 SMILES are the real published structures.
  Every one is verified at build time by recomputing its molecular weight with
  RDKit and comparing to the literature value — a typo cannot reach the demo.
- **Fingerprints and descriptors.** Genuine RDKit Morgan/ECFP4 fingerprints,
  MACCS keys and physicochemical descriptors.
- **Pharmacology profiles.** CYP450 roles, P-glycoprotein handling, fraction
  renally excreted and risk flags are textbook clinical pharmacology.
- **Interaction mechanisms.** Every mechanism described is a real, documented
  pathway (CYP inhibition/induction, transporter effects, additive toxicity).
- **PK equations.** CKD-EPI 2021 eGFR, Cockcroft-Gault and clearance-weighted
  organ scaling are the standard equations used in real dose-adjustment guidance.
- **Gene → resistance mapping.** Real microbiology (blaNDM-1 → carbapenems,
  mecA → beta-lactams, vanA → glycopeptides, and so on).
- **The ML pipelines.** Featurisation, training, and held-out evaluation are
  genuine — including the deliberately harsh cold-split protocols.
"""
        )
    with sim_col:
        st.markdown(
            f"<div style='color:{viz.RISK_COLORS['High']};font-weight:700;"
            f"margin-bottom:.4rem'>⚠ SIMULATED OR SIMPLIFIED</div>",
            unsafe_allow_html=True)
        st.markdown(
            """
- **Interaction labels.** Derived from our curated mechanism knowledge base,
  **not** from DrugBank or TWOSIDES — both need licensing we could not obtain
  in the hackathon window. Severity thresholds are our own calibration,
  cross-checked against documented clinical severity for 24 well-known pairs.
- **Module 1's ML layer is a weak-but-honest filter.** With 39 drugs there is
  not enough data for a reliable severity predictor — we measured a 4-class
  model at 55% accuracy on held-out drugs and rejected it. The binary filter
  we ship reaches ROC-AUC ≈0.68–0.73 on unseen drugs (the exact figure is in
  the panel above, with its spread): a real signal, well short of production
  quality. For drugs in the knowledge base, severity comes from the mechanism
  rules, not from this model.
- **Genomic sequences.** Fully synthetic. Each resistance gene is a
  deterministic marker cassette with a stable identity — **not** the real
  published sequence of blaNDM-1, mecA, etc. We chose not to ship approximate
  biological sequences presented as authentic.
- **Hepatic impairment factor.** Our own heuristic from ALT, albumin and
  bilirubin. Real practice uses Child-Pugh, which needs clinical assessment
  that a form field cannot capture.
- **Risk weights.** The escalation weights in Module 2 and the 60/40 composite
  split are reasoned, not fitted to outcome data.
- **Linear kinetics assumed.** Phenytoin in particular is non-linear, so its
  true exposure change is under-estimated here.
- **Not modelled:** pharmacogenomics (CYP2D6/2C19 metaboliser status),
  transporters beyond P-gp, protein-binding displacement, disease-state effects.
"""
        )

    st.divider()
    st.markdown("#### The path to production")
    st.markdown(
        """
The architecture was built so the simulated parts are *swappable*, not
load-bearing:

1. **Real interaction labels** — replace `build_dataset()` in `trialsense/ddi.py`
   with a licensed DrugBank or TWOSIDES loader. Featurisation, model and
   evaluation code are unchanged.
2. **Real sequences** — replace `_gene_cassette()` in `trialsense/amr.py` with
   reference sequences from NCBI Pathogen Detection or CARD, then retrain.
   The k-mer pipeline already accepts real FASTA today.
3. **Validated hepatic scoring** — swap the ALT/albumin heuristic for Child-Pugh
   or MELD where the clinical inputs are available.
4. **Stronger models** — the current Random Forests are deliberately small so
   they train in a minute on a laptop. With real data at scale, a graph neural
   network over molecular graphs is the natural upgrade for Module 1.
"""
    )

    st.divider()
    st.markdown("#### Positioning")
    st.info(
        "TrialSense is an **R&D pre-screening tool for pharmaceutical teams** — "
        "it produces candidate risk reports to inform protocol design and "
        "portfolio decisions before expensive human trials begin. It is **not** a "
        "patient-facing application, not a diagnostic, and not a clinical "
        "decision-support system. No output here should be used to treat anyone."
    )

    if not from_cache:
        st.caption("Models were trained in-session. Run `python train.py` to cache them.")

# --- Footer ------------------------------------------------------------------
st.markdown(
    f"<div style='color:{viz.MUTED};font-size:.76rem;text-align:center;"
    f"margin-top:2.5rem;padding-top:1rem;border-top:1px solid {viz.PANEL_EDGE}'>"
    "TrialSense · Segue 3.0 prototype · Research demonstration only — "
    "not for clinical use</div>",
    unsafe_allow_html=True,
)
