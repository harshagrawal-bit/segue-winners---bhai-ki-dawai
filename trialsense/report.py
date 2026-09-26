"""
TrialSense — Unified candidate risk report.

Fuses the three modules into the single artefact an R&D portfolio team would
actually act on: for a given candidate pairing, patient subgroup and target
organism, what is the pre-trial risk and what should change in the protocol?

The composite score is a TRANSPARENT WEIGHTED COMBINATION, not a learned model.
That is deliberate: a portfolio decision needs a number whose provenance can be
explained line by line to a regulator or an investment committee. Every
contribution below is individually reported in the UI so nothing is hidden
inside an opaque aggregate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .amr import (
    LAST_RESORT,
    RESISTANCE_CALL_THRESHOLD,
    AMRReport,
)
from .ddi import SEVERITY_LEVELS, DDIAssessment
from .drugs import Drug
from .pk import PatientProfile, PersonalizedRisk, dose_guidance, renal_study_plan

RISK_BANDS = [
    (0, 25, "Low", "#2E7D32"),
    (25, 50, "Moderate", "#F9A825"),
    (50, 75, "High", "#EF6C00"),
    (75, 101, "Critical", "#C62828"),
]


def band_for(score: float) -> tuple[str, str]:
    """(label, colour) for a 0-100 composite score."""
    for lo, hi, label, color in RISK_BANDS:
        if lo <= score < hi:
            return label, color
    return "Critical", "#C62828"


@dataclass
class Finding:
    """One ranked, plain-English risk statement for the report."""

    severity: str  # Critical | High | Moderate | Low | Info
    module: str  # which module produced it
    title: str
    detail: str

    @property
    def color(self) -> str:
        return {
            "Critical": "#C62828",
            "High": "#EF6C00",
            "Moderate": "#F9A825",
            "Low": "#2E7D32",
            "Info": "#546E7A",
        }[self.severity]

    @property
    def rank(self) -> int:
        return {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3, "Info": 4}[self.severity]


@dataclass
class CandidateRiskReport:
    """The combined output of all three modules for one case."""

    case_name: str
    drug_a: Drug
    drug_b: Drug
    patient: PatientProfile
    ddi: DDIAssessment
    personalized: PersonalizedRisk
    amr: AMRReport | None = None

    composite_score: float = 0.0
    ddi_component: float = 0.0
    amr_component: float = 0.0
    findings: list[Finding] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    amr_applicable: bool = False
    candidate_antibiotic: Drug | None = None
    # False when the candidate IS an antibacterial but its class is outside the
    # twelve Module 3 screens — distinct from "no antibacterial in this pairing".
    amr_class_screened: bool = True

    @property
    def band(self) -> str:
        return band_for(self.composite_score)[0]

    @property
    def color(self) -> str:
        return band_for(self.composite_score)[1]

    def headline(self) -> str:
        """One sentence for the top of the report."""
        return (
            f"Composite pre-trial risk for {self.drug_a.name} + {self.drug_b.name} "
            f"in the '{self.patient.label}' subgroup is {self.band.upper()} "
            f"({self.composite_score:.0f}/100)."
        )


def build_report(
    case_name: str,
    drug_a: Drug,
    drug_b: Drug,
    patient: PatientProfile,
    ddi: DDIAssessment,
    personalized: PersonalizedRisk,
    amr: AMRReport | None = None,
) -> CandidateRiskReport:
    """Assemble the unified report, its composite score and its action list."""

    report = CandidateRiskReport(
        case_name=case_name,
        drug_a=drug_a,
        drug_b=drug_b,
        patient=patient,
        ddi=ddi,
        personalized=personalized,
        amr=amr,
    )

    # --- Is the AMR module even relevant to this pairing? --------------------
    antibiotics = [d for d in (drug_a, drug_b) if d.antibiotic_class]
    report.candidate_antibiotic = antibiotics[0] if antibiotics else None
    report.amr_applicable = bool(antibiotics) and amr is not None

    # --- Component 1: interaction + patient-specific toxicity ----------------
    # Uses the PERSONALISED severity, because the whole point of Module 2 is that
    # population-average risk is the wrong number for protocol design.
    #
    # Mapped through a saturating exponential rather than a hard clip. The
    # underlying mechanism score is unbounded above — stacked pathways routinely
    # reach 4-5 — so any linear cap pinned every dangerous pairing at exactly
    # 100/100 and destroyed the resolution at the top of the range, which is the
    # region a portfolio team most needs to rank within. This curve approaches
    # 100 asymptotically, so the ordering of high-risk candidates is preserved:
    #   score 1.5 -> 41   2.42 (Severe threshold) -> 58   4.0 -> 76   5.0 -> 83
    ddi_norm = 1.0 - math.exp(-personalized.adjusted_score / 2.8)
    report.ddi_component = ddi_norm * 100

    # Is the candidate's own class one Module 3 actually screens? If not, we
    # cannot score resistance against it, and substituting 0.0 would read as
    # "no resistance" when the truth is "never checked".
    if report.amr_applicable and amr is not None and report.candidate_antibiotic:
        cls = report.candidate_antibiotic.antibiotic_class or ""
        if cls not in amr.probabilities:
            report.amr_class_screened = False
            report.amr_applicable = False

    # --- Component 2: resistance pressure on the candidate -------------------
    amr_norm = 0.0
    if report.amr_applicable and amr is not None:
        ab = report.candidate_antibiotic
        assert ab is not None
        target_class = ab.antibiotic_class
        prob = amr.probabilities.get(target_class or "", 0.0)
        # Direct hit: the organism resists the class our candidate belongs to.
        amr_norm = 0.75 * prob
        # Breadth of resistance: how constrained the fallback options are.
        breadth = len(amr.resistant_classes()) / max(1, len(amr.probabilities))
        amr_norm += 0.25 * breadth
        amr_norm = min(1.0, amr_norm)
    report.amr_component = amr_norm * 100

    # --- Composite -----------------------------------------------------------
    if report.amr_applicable:
        report.composite_score = round(0.6 * report.ddi_component + 0.4 * report.amr_component, 1)
    else:
        report.composite_score = round(report.ddi_component, 1)

    report.findings = _collect_findings(report)
    report.findings.sort(key=lambda f: f.rank)
    report.actions = _recommend_actions(report)
    return report


def _collect_findings(r: CandidateRiskReport) -> list[Finding]:
    """Turn every module's output into ranked, plain-English findings."""
    out: list[Finding] = []
    sev = r.personalized.adjusted_severity

    # --- Module 1: the interaction itself
    if r.ddi.mechanisms:
        level = {0: "Low", 1: "Low", 2: "Moderate", 3: "High"}[r.ddi.headline_severity]
        out.append(
            Finding(
                severity=level,
                module="Module 1 · Interaction",
                title=(
                    f"{SEVERITY_LEVELS[r.ddi.headline_severity]} interaction between "
                    f"{r.drug_a.name} and {r.drug_b.name}"
                ),
                detail=r.ddi.plain_summary(),
            )
        )
    else:
        out.append(
            Finding(
                severity="Low",
                module="Module 1 · Interaction",
                title="No interaction pathway identified",
                detail=r.ddi.plain_summary(),
            )
        )

    # --- Module 2: what this patient subgroup changes
    if r.personalized.escalated:
        level = "Critical" if sev >= 3 else "High"
        out.append(
            Finding(
                severity=level,
                module="Module 2 · Patient response",
                title=(
                    f"Risk escalates to {SEVERITY_LEVELS[sev]} in the "
                    f"'{r.patient.label}' subgroup"
                ),
                detail=(
                    r.personalized.reasons[0]
                    if r.personalized.reasons
                    else r.personalized.direction_text
                ),
            )
        )
    for reason in r.personalized.reasons[1:]:
        out.append(
            Finding(
                severity="Moderate" if sev < 3 else "High",
                module="Module 2 · Patient response",
                title="Patient-specific risk factor",
                detail=reason,
            )
        )
    for good in r.personalized.protective:
        out.append(
            Finding(
                severity="Info",
                module="Module 2 · Patient response",
                title="Risk-reducing factor",
                detail=good,
            )
        )

    # --- Module 3: resistance context
    #
    # Three genuinely different situations, which earlier collapsed into two:
    #   a) no antibacterial in the pairing      -> nothing to screen
    #   b) antibacterial, class NOT screened    -> we did not check
    #   c) antibacterial, class screened        -> a real result
    # Conflating (b) with (c) produced a confident "remains susceptible,
    # 0% predicted resistance" for drugs such as metronidazole, whose class
    # (Nitroimidazoles) is not one of the twelve. An unrun check must never be
    # rendered as a passed one.
    if r.amr is not None and r.candidate_antibiotic is not None and not r.amr_class_screened:
        ab = r.candidate_antibiotic
        out.append(
            Finding(
                severity="Info",
                module="Module 3 · Resistance",
                title=f"{ab.antibiotic_class} is not screened by Module 3",
                detail=(
                    f"{ab.name} belongs to the {ab.antibiotic_class} class, which "
                    f"is not among the {len(r.amr.probabilities)} classes this "
                    "module screens. No resistance prediction is available for it "
                    "— this is 'not checked', not 'no resistance found'. The "
                    "resistance component is therefore excluded from the "
                    "composite score, which reflects interaction and toxicity only."
                ),
            )
        )
        last = r.amr.last_resort_hits()
        if last:
            out.append(
                Finding(
                    severity="Critical",
                    module="Module 3 · Resistance",
                    title=f"Last-resort resistance detected ({', '.join(last)})",
                    detail=(
                        f"{r.amr.strain_name} is predicted resistant to "
                        f"{', '.join(last)}. This does not bear on the candidate's "
                        "own class, which was not screened, but it is material to "
                        "the value of any successful agent against this organism."
                    ),
                )
            )
    elif r.amr_applicable and r.amr is not None:
        ab = r.candidate_antibiotic
        assert ab is not None
        target = ab.antibiotic_class or ""
        prob = r.amr.probabilities[target]
        if prob >= RESISTANCE_CALL_THRESHOLD:
            crit = target in LAST_RESORT
            out.append(
                Finding(
                    severity="Critical" if crit else "High",
                    module="Module 3 · Resistance",
                    title=f"Target organism is predicted resistant to {ab.name}",
                    detail=(
                        f"{r.amr.strain_name} carries determinants predicting "
                        f"resistance to the {target} class "
                        f"({prob:.0%} confidence). {ab.name} belongs to that "
                        "class, so this organism would likely not respond — a "
                        "strong signal to reconsider the indication before "
                        "trial entry."
                    ),
                )
            )
        else:
            out.append(
                Finding(
                    severity="Low",
                    module="Module 3 · Resistance",
                    title=f"Target organism remains susceptible to {ab.name}",
                    detail=(
                        f"No resistance determinant for the {target} class was "
                        f"detected in {r.amr.strain_name} "
                        f"({prob:.0%} predicted resistance). This supports "
                        f"{ab.name} as a viable candidate against this organism."
                    ),
                )
            )

        last = r.amr.last_resort_hits()
        if last:
            out.append(
                Finding(
                    severity="Critical",
                    module="Module 3 · Resistance",
                    title=f"Last-resort resistance detected ({', '.join(last)})",
                    detail=(
                        f"{r.amr.strain_name} is predicted resistant to "
                        f"{', '.join(last)} — classes reserved for when everything "
                        "else has failed. This organism represents genuine unmet "
                        "need, which raises both the clinical value and the "
                        "regulatory priority of a successful candidate."
                    ),
                )
            )
    elif not r.amr_applicable:
        out.append(
            Finding(
                severity="Info",
                module="Module 3 · Resistance",
                title="Resistance screening not applicable",
                detail=(
                    "Neither compound in this pairing is an antibacterial, so "
                    "resistance analysis does not contribute to the composite "
                    "score. The score reflects interaction and toxicity risk only."
                ),
            )
        )

    return out


def _recommend_actions(r: CandidateRiskReport) -> list[str]:
    """Concrete protocol-design recommendations, written for an R&D reader."""
    actions: list[str] = []
    sev = r.personalized.adjusted_severity

    if sev >= 3:
        actions.append(
            f"**Do not co-administer {r.drug_a.name} and {r.drug_b.name} in this "
            "trial without a protocol amendment.** Either exclude the combination "
            "or add it as a formal contraindication in the study design."
        )
    elif sev == 2:
        actions.append(
            f"**Add dose adjustment and monitoring** for the "
            f"{r.drug_a.name} + {r.drug_b.name} combination rather than excluding it."
        )

    if r.personalized.escalated:
        actions.append(
            f"**Stratify enrolment by organ function.** Risk in the "
            f"'{r.patient.label}' subgroup is a full severity band above the "
            "population average, so pooling these patients into a single arm will "
            "dilute the safety signal and delay its detection."
        )

    # Concrete dosing guidance per drug where clearance is meaningfully reduced.
    for drug, exp in zip((r.drug_a, r.drug_b), r.personalized.exposures):
        if exp.recommended_dose_fraction < 0.9:
            actions.append(dose_guidance(drug, r.patient))

    if r.patient.egfr() < 60 and any(d.nephrotoxic for d in (r.drug_a, r.drug_b)):
        actions.append(
            "**Add renal function monitoring** at baseline and through the dosing "
            "period; consider a minimum eGFR threshold as an inclusion criterion."
        )

    # Renal impairment study design (FDA 2024 guidance logic) — only raised when
    # the subgroup under review actually has impaired kidneys, so a healthy
    # control profile is not cluttered with drug-level planning notes.
    for drug in (r.drug_a, r.drug_b):
        plan = renal_study_plan(drug)
        if r.patient.egfr() < 60 and plan.design == "Full study":
            actions.append(
                f"**Plan a full renal impairment study for {drug.name}** "
                f"(normal vs mild, moderate and severe). Predicted exposure in "
                f"severe impairment: {plan.severe.exposure_ratio:.1f}× normal."
            )
    if any(d.qt_prolonging for d in (r.drug_a, r.drug_b)):
        actions.append(
            "**Add ECG monitoring and electrolyte correction** to the protocol — "
            "QT-prolonging drugs are far more dangerous when potassium or "
            "magnesium is low, and that is a correctable trial variable."
        )
    if any(d.narrow_therapeutic_index for d in (r.drug_a, r.drug_b)):
        nti = [d.name for d in (r.drug_a, r.drug_b) if d.narrow_therapeutic_index]
        actions.append(
            f"**Include therapeutic drug monitoring for {', '.join(nti)}** — a "
            "narrow therapeutic window is the single strongest predictor that an "
            "interaction will become a reportable adverse event."
        )

    if r.amr_applicable and r.amr is not None and r.candidate_antibiotic is not None:
        ab = r.candidate_antibiotic
        target = ab.antibiotic_class or ""
        if r.amr.probabilities.get(target, 0.0) >= RESISTANCE_CALL_THRESHOLD:
            viable = [c for c in r.amr.susceptible_classes()]
            actions.append(
                f"**Reconsider the target indication for {ab.name}.** The modelled "
                f"organism already resists the {target} class."
                + (
                    f" Classes still predicted active: {', '.join(viable[:4])} — "
                    "these represent better-positioned starting points."
                    if viable
                    else " No screened class remains active against this organism."
                )
            )

    if not actions:
        actions.append(
            "**No protocol changes indicated.** This combination and subgroup "
            "show no risk signal above baseline; standard monitoring is sufficient."
        )
    return actions
