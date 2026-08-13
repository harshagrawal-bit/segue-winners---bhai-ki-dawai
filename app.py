#!/usr/bin/env python3
"""
TrialSense — computational pre-trial risk screening for drug candidates.

Run with:   streamlit run app.py
(Train the models first with `python train.py` so the app opens instantly.)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import streamlit as st

from trialsense import amr as amr_mod
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
from trialsense.pk import PatientProfile, personalize
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
        age=age, weight_kg=weight, sex=sex, serum_creatinine=scr, alt=alt_val,
        albumin=albumin, bilirubin=bilirubin, serum_potassium=potassium,
        label=default.label if active_case else "Custom profile",
    )

    st.markdown(
        viz.card("", f"<b style='color:{viz.INK}'>{patient.creatinine_clearance():.0f}"
                 f"</b> mL/min CrCl · {patient.ckd_stage()}<br>"
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
            use_container_width=True, config={"displayModeBar": False},
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
                use_container_width=True, config={"displayModeBar": False},
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
        "equations (Cockcroft-Gault; clearance-weighted organ scaling), but the "
        "hepatic impairment factor and the risk weights are our own calibration. "
        "This is an R&D prioritisation aid — **not a validated clinical dosing "
        "tool, and not for treating patients.**"
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        crcl = patient.creatinine_clearance()
        col = (viz.RISK_COLORS["Severe"] if crcl < 30 else
               viz.RISK_COLORS["High"] if crcl < 60 else
               viz.RISK_COLORS["Moderate"] if crcl < 90 else viz.RISK_COLORS["Low"])
        st.markdown(viz.stat("Creatinine clearance", f"{crcl:.0f}",
                             f"mL/min · {patient.ckd_stage()}", col), unsafe_allow_html=True)
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
                        use_container_width=True, config={"displayModeBar": False})
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
**Step 1 — kidney function.** Cockcroft-Gault:

`CrCl = (140 − age) × weight ÷ (72 × serum creatinine)` , × 0.85 if female

For this profile: (140 − {patient.age}) × {patient.weight_kg:.0f} ÷ (72 × {patient.serum_creatinine:.1f})
{"× 0.85 " if patient.sex.lower().startswith("f") else ""}= **{patient.creatinine_clearance():.1f} mL/min**

Renal function factor KF = CrCl ÷ 120 = **{patient.renal_function_factor():.3f}**

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

# =============================================================================
# TAB 4 — Resistance
# =============================================================================
with tab_amr:
    st.markdown("### Module 3 · Antimicrobial resistance intelligence")
    st.caption(
        "Predicts resistance across antibiotic classes from genomic k-mer "
        "composition, to show whether a candidate's target organism is still treatable."
    )

    lvl, lvl_col = amr_result.risk_level()
    st.markdown(viz.banner(amr_result.plain_summary(), lvl_col), unsafe_allow_html=True)
    if amr_result.context:
        st.caption(f"**Why this organism matters:** {amr_result.context}")

    ac1, ac2 = st.columns([1.25, 1])
    with ac1:
        st.markdown("#### Predicted resistance by antibiotic class")
        st.plotly_chart(
            viz.resistance_chart(amr_result.probabilities, amr_mod.LAST_RESORT),
            use_container_width=True, config={"displayModeBar": False},
        )
    with ac2:
        st.markdown("#### Resistance genes detected")
        if amr_result.detected_genes:
            for note in amr_result.mechanism_notes():
                gene_name = note.split("**")[1]
                is_last = any(
                    c in amr_mod.LAST_RESORT
                    for c in amr_mod.RESISTANCE_GENES[gene_name].confers_resistance_to
                )
                st.markdown(
                    viz.card(gene_name, note.split("— ", 1)[1] if "— " in note else note,
                             viz.RISK_COLORS["Critical"] if is_last else viz.RISK_COLORS["High"]),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                viz.card("No resistance determinants found",
                         "No marker cassette matched this sequence. All screened "
                         "antibiotic classes are predicted to remain active.",
                         viz.RISK_COLORS["Low"]),
                unsafe_allow_html=True,
            )

        if report.candidate_antibiotic:
            ab = report.candidate_antibiotic
            p = amr_result.probabilities.get(ab.antibiotic_class or "", 0.0)
            st.markdown("#### Impact on this candidate")
            st.markdown(
                viz.card(
                    f"{ab.name} ({ab.antibiotic_class})",
                    f"Predicted resistance of this organism to the "
                    f"{ab.antibiotic_class} class: <b>{p:.0%}</b>. "
                    + ("This candidate would likely fail against this isolate."
                       if p >= 0.5 else
                       "This candidate remains viable against this isolate."),
                    viz.RISK_COLORS["Critical"] if p >= 0.5 else viz.RISK_COLORS["Low"],
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
- **PK equations.** Cockcroft-Gault and clearance-weighted organ scaling are
  the standard equations used in real dose-adjustment guidance.
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
