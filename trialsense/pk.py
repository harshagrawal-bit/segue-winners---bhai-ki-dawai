"""
TrialSense — Module 2: Personalised toxicity / patient-response simulator.

WHAT THIS DOES
--------------
Module 1 asks "is this drug pair dangerous in general?". Module 2 asks the
question that actually decides trial outcomes: "is it dangerous in THIS patient
subgroup?" — because the adverse events that halt trials usually concentrate in
a subgroup (elderly, renally impaired, hepatically impaired) rather than the
average enrolled patient.

WHAT IS REAL PHARMACOLOGY
-------------------------
  * Cockcroft-Gault (1976) creatinine clearance estimate. Real, and still the
    equation most drug labels use for renal dose adjustment.
  * The additive clearance model:
        CL_patient / CL_normal  =  fe * KF  +  fh * HF
    where fe/fh are the renal/hepatic fractions of clearance and KF/HF are the
    patient's residual function in each organ. This is the standard framework
    behind renal dose-adjustment guidance (the Rowland-Matin approach).
  * Steady-state exposure is inversely proportional to clearance:
        AUC_ratio = 1 / (CL_ratio)
    This follows directly from AUC = Dose / CL and is exact for linear kinetics.
  * Proportional dose adjustment (scale the dose by the clearance ratio) is the
    standard first-pass method for maintenance dosing.

WHAT IS SIMPLIFIED — SAY THIS OUT LOUD IN THE DEMO
--------------------------------------------------
  * The HEPATIC function factor (HF) is our own heuristic built from ALT,
    albumin and bilirubin. Real practice uses Child-Pugh, which additionally
    requires clinical assessment of ascites and encephalopathy that cannot be
    captured in a form field. Ours is a defensible ordering, not a validated score.
  * The risk-escalation weights that turn an exposure ratio into a severity
    bump are our own calibration, informed by known pharmacology but not fitted
    to outcome data.
  * Linear kinetics are assumed. Phenytoin in particular is famously non-linear,
    so its true exposure change is UNDER-estimated here.
  * Genetics (CYP2D6/2C19 poor-metaboliser status), drug transporters other than
    P-gp, protein-binding displacement and disease-state effects are not modelled.

THIS IS AN R&D PRE-SCREENING APPROXIMATION FOR PRIORITISING PROTOCOL DESIGN AND
TRIAL EXCLUSION CRITERIA. IT IS NOT A VALIDATED CLINICAL DOSING TOOL AND MUST
NOT BE USED TO TREAT A PATIENT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .drugs import Drug

# Reference healthy adult creatinine clearance (mL/min) used as the denominator
# for the renal function factor.
CRCL_NORMAL = 120.0


@dataclass
class PatientProfile:
    """The covariates a trial screening form would realistically collect."""

    age: int = 45
    weight_kg: float = 70.0
    sex: str = "Male"  # affects the Cockcroft-Gault constant only
    serum_creatinine: float = 1.0  # mg/dL — kidney function proxy
    alt: float = 30.0  # U/L — liver enzyme, hepatocellular injury proxy
    albumin: float = 4.2  # g/dL — liver synthetic function proxy
    bilirubin: float = 0.8  # mg/dL — liver excretory function proxy
    serum_potassium: float = 4.2  # mmol/L — modifies arrhythmia risk
    label: str = "Custom profile"

    # ---------------------------------------------------------------- Kidney
    def creatinine_clearance(self) -> float:
        """
        Cockcroft-Gault estimated creatinine clearance in mL/min.

            CrCl = (140 - age) * weight / (72 * SCr)   [* 0.85 if female]
        """
        crcl = ((140 - self.age) * self.weight_kg) / (72 * max(0.1, self.serum_creatinine))
        if self.sex.lower().startswith("f"):
            crcl *= 0.85
        return round(crcl, 1)

    def renal_function_factor(self) -> float:
        """Residual kidney function as a fraction of healthy (KF), capped at 1.0."""
        return round(min(1.0, self.creatinine_clearance() / CRCL_NORMAL), 3)

    def ckd_stage(self) -> str:
        """Plain-English kidney status from the estimated clearance."""
        crcl = self.creatinine_clearance()
        if crcl >= 90:
            return "Normal kidney function"
        if crcl >= 60:
            return "Mildly reduced kidney function"
        if crcl >= 30:
            return "Moderately reduced kidney function"
        if crcl >= 15:
            return "Severely reduced kidney function"
        return "Kidney failure"

    # ---------------------------------------------------------------- Liver
    def hepatic_function_factor(self) -> float:
        """
        Residual metabolic capacity (HF), 0.15-1.0.

        SIMPLIFIED: a weighted penalty from three routine labs. Chosen so the
        ordering of patients is sensible; the absolute values are not validated.
        """
        hf = 1.0
        if self.alt > 40:  # upper limit of normal
            hf -= min(0.35, (self.alt - 40) / 40 * 0.12)
        if self.albumin < 3.5:  # falling synthetic function
            hf -= min(0.30, (3.5 - self.albumin) * 0.18)
        if self.bilirubin > 1.2:  # failing excretory function
            hf -= min(0.25, (self.bilirubin - 1.2) * 0.10)
        # Hepatic blood flow and enzyme mass decline with age (~1%/yr after 40).
        hf *= 1.0 - min(0.30, 0.008 * max(0, self.age - 40))
        return round(max(0.15, min(1.0, hf)), 3)

    def liver_status(self) -> str:
        hf = self.hepatic_function_factor()
        if hf >= 0.85:
            return "Normal liver function"
        if hf >= 0.65:
            return "Mildly impaired liver function"
        if hf >= 0.45:
            return "Moderately impaired liver function"
        return "Severely impaired liver function"

    def summary_line(self) -> str:
        return (
            f"{self.age}y {self.sex.lower()}, {self.weight_kg:.0f} kg · "
            f"CrCl {self.creatinine_clearance():.0f} mL/min · "
            f"ALT {self.alt:.0f} U/L · K⁺ {self.serum_potassium:.1f} mmol/L"
        )


@dataclass
class DrugExposure:
    """How one drug behaves in one patient."""

    drug_name: str
    clearance_ratio: float  # CL_patient / CL_normal
    exposure_ratio: float  # AUC_patient / AUC_normal  (= 1 / clearance_ratio)
    renal_contribution: float  # fe — how much of clearance is renal
    recommended_dose_fraction: float  # proportional maintenance-dose scaling

    @property
    def percent_change(self) -> float:
        return (self.exposure_ratio - 1.0) * 100.0

    def plain_text(self) -> str:
        pct = self.percent_change
        if abs(pct) < 10:
            return f"{self.drug_name} exposure is essentially unchanged in this profile."
        direction = "higher" if pct > 0 else "lower"
        return (
            f"{self.drug_name} blood exposure is about {abs(pct):.0f}% {direction} "
            f"than in a healthy adult ({self.exposure_ratio:.2f}× normal)."
        )


@dataclass
class PersonalizedRisk:
    """Module 2 output for a drug pair in a specific patient profile."""

    base_severity: int
    base_score: float
    adjusted_score: float
    adjusted_severity: int
    exposures: list[DrugExposure] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # why risk moved
    protective: list[str] = field(default_factory=list)  # why risk is lower

    @property
    def escalated(self) -> bool:
        return self.adjusted_severity > self.base_severity

    @property
    def de_escalated(self) -> bool:
        return self.adjusted_severity < self.base_severity

    @property
    def direction_text(self) -> str:
        if self.escalated:
            return "Risk is HIGHER in this patient profile than in an average adult."
        if self.de_escalated:
            return "Risk is LOWER in this patient profile than in an average adult."
        return "Risk is unchanged from the average-adult baseline."


# =============================================================================
# Core calculation
# =============================================================================


def compute_exposure(drug: Drug, patient: PatientProfile) -> DrugExposure:
    """
    Exposure change for one drug in one patient.

        CL_ratio = fe * KF + fh * HF
        AUC_ratio = 1 / CL_ratio

    A drug cleared 90% by the kidney (fe=0.9) in a patient with 30% kidney
    function accumulates dramatically; a purely hepatic drug in the same patient
    is unaffected. That organ-specific behaviour is the whole point of the module.
    """
    kf = patient.renal_function_factor()
    hf = patient.hepatic_function_factor()

    cl_ratio = drug.fe * kf + drug.fh * hf
    cl_ratio = max(0.05, cl_ratio)  # floor: avoid an infinite exposure ratio
    exposure_ratio = 1.0 / cl_ratio

    return DrugExposure(
        drug_name=drug.name,
        clearance_ratio=round(cl_ratio, 3),
        exposure_ratio=round(exposure_ratio, 3),
        renal_contribution=drug.fe,
        recommended_dose_fraction=round(cl_ratio, 2),
    )


def personalize(
    drug_a: Drug,
    drug_b: Drug,
    patient: PatientProfile,
    base_severity: int,
    base_score: float,
) -> PersonalizedRisk:
    """
    Adjust Module 1's interaction severity for this patient's physiology.

    Two independent contributions:
      1. EXPOSURE — impaired clearance raises drug levels, scaling the whole
         interaction. Applied on a log2 scale because pharmacological effect
         grows with the ORDER of magnitude of exposure, not linearly with it.
      2. ORGAN-SPECIFIC HAZARDS — named clinical situations where a particular
         drug property meets a particular patient vulnerability. These are the
         findings a screening tool exists to surface.
    """
    kf = patient.renal_function_factor()
    hf = patient.hepatic_function_factor()
    crcl = patient.creatinine_clearance()

    exposures = [compute_exposure(d, patient) for d in (drug_a, drug_b)]
    reasons: list[str] = []
    protective: list[str] = []

    # --- 1. Exposure-driven scaling -----------------------------------------
    worst = max(exposures, key=lambda e: e.exposure_ratio)
    # log2: doubling exposure adds a fixed increment of risk.
    exposure_factor = 1.0 + 0.45 * math.log2(max(1.0, worst.exposure_ratio))
    adjusted = base_score * exposure_factor

    if worst.exposure_ratio >= 1.25:
        reasons.append(
            f"Reduced organ clearance raises {worst.drug_name} exposure to "
            f"{worst.exposure_ratio:.2f}× the level expected in a healthy adult, "
            "amplifying any interaction between the two drugs."
        )

    # Impaired clearance of a narrow-window drug is the classic trial-halting event.
    for drug, exp in zip((drug_a, drug_b), exposures):
        if drug.narrow_therapeutic_index and exp.exposure_ratio >= 1.3:
            adjusted += 0.4
            reasons.append(
                f"{drug.name} has a narrow safety margin AND accumulates in this "
                f"profile ({exp.exposure_ratio:.2f}× normal) — a small dosing "
                "error here becomes a toxic one."
            )

    # --- 2. Organ-specific hazards ------------------------------------------
    for drug in (drug_a, drug_b):
        # Kidney-toxic drug given to an already-impaired kidney.
        if drug.nephrotoxic and crcl < 60:
            bump = 0.75 if crcl < 30 else 0.45
            adjusted += bump
            reasons.append(
                f"{drug.name} can itself injure the kidney, and this profile "
                f"already has {patient.ckd_stage().lower()} "
                f"(CrCl {crcl:.0f} mL/min) — a self-reinforcing decline."
            )
        # Liver-toxic drug given to an already-impaired liver.
        if drug.hepatotoxic and hf < 0.65:
            adjusted += 0.45
            reasons.append(
                f"{drug.name} carries a risk of liver injury and this profile "
                f"shows {patient.liver_status().lower()}."
            )
        # Metformin + advanced renal impairment: a specific, real, labelled risk.
        if drug.name == "Metformin" and crcl < 30:
            adjusted += 1.0
            reasons.append(
                "Metformin is contraindicated below a CrCl of 30 mL/min: it "
                "accumulates and can cause life-threatening lactic acidosis. "
                "This profile falls below that threshold."
            )
        # Sedatives in the elderly — increased brain sensitivity, not just PK.
        if drug.cns_depressant and patient.age >= 75:
            adjusted += 0.35
            reasons.append(
                f"Patients over 75 are more sensitive to the sedative effect of "
                f"{drug.name} independently of blood levels (falls, confusion, "
                "respiratory depression)."
            )

    # Potassium: an interaction that only becomes dangerous in the right kidney.
    k_raisers = [d for d in (drug_a, drug_b) if d.potassium_raising]
    if k_raisers and crcl < 60:
        adjusted += 0.6
        names = " and ".join(d.name for d in k_raisers)
        reasons.append(
            f"{names} raise{'s' if len(k_raisers) == 1 else ''} serum potassium, "
            "and a weakened kidney is the organ responsible for excreting it. "
            f"Baseline potassium is already {patient.serum_potassium:.1f} mmol/L."
        )
    if k_raisers and patient.serum_potassium >= 5.0:
        adjusted += 0.5
        reasons.append(
            f"Baseline potassium of {patient.serum_potassium:.1f} mmol/L is "
            "already at the top of the normal range before adding a "
            "potassium-retaining drug."
        )

    # QT risk is driven by electrolytes as much as by the drugs themselves.
    qt_drugs = [d for d in (drug_a, drug_b) if d.qt_prolonging]
    if qt_drugs and patient.serum_potassium < 3.5:
        adjusted += 0.55
        reasons.append(
            f"Low potassium ({patient.serum_potassium:.1f} mmol/L) markedly "
            "increases the arrhythmia risk of QT-prolonging drugs — the "
            "electrolyte, not the drug, is often the trigger."
        )

    # --- 3. Genuinely lower-risk profiles ------------------------------------
    if kf >= 0.95 and hf >= 0.9 and patient.age < 60:
        protective.append(
            "Kidney and liver function are both normal for age, so neither drug "
            "accumulates. This profile sits at the low end of the risk range."
        )
    if base_score > 0 and worst.exposure_ratio < 1.05 and not reasons:
        protective.append(
            "Neither drug's clearance is meaningfully affected by this profile, "
            "so the interaction is no worse than the population baseline."
        )

    from .ddi import score_to_severity

    return PersonalizedRisk(
        base_severity=base_severity,
        base_score=round(base_score, 2),
        adjusted_score=round(adjusted, 2),
        adjusted_severity=score_to_severity(adjusted),
        exposures=exposures,
        reasons=reasons,
        protective=protective,
    )


def dose_guidance(drug: Drug, patient: PatientProfile) -> str:
    """Plain-English proportional dosing note for the report."""
    exp = compute_exposure(drug, patient)
    pct = exp.recommended_dose_fraction * 100
    if pct >= 90:
        return f"{drug.name}: standard dosing appropriate for this profile."
    if pct >= 60:
        return (
            f"{drug.name}: consider approximately {pct:.0f}% of the standard "
            "maintenance dose, or an extended dosing interval."
        )
    return (
        f"{drug.name}: markedly reduced clearance — approximately {pct:.0f}% of "
        "the standard maintenance dose would be needed to match normal exposure. "
        "Consider excluding this subgroup from early-phase cohorts."
    )
