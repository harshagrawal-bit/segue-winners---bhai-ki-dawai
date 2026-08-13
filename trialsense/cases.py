"""
TrialSense — Preloaded demo cases.

Each case is a complete, self-consistent scenario (drug pair + patient subgroup
+ target organism) chosen to demonstrate a DIFFERENT capability of the system,
so a live demo can move through them without repeating a point.

They are constructed scenarios built on real pharmacology, not records of real
patients or real trials.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pk import PatientProfile


@dataclass
class DemoCase:
    name: str
    drug_a: str
    drug_b: str
    strain: str | None
    patient: PatientProfile
    pitch: str  # what this case is meant to show a judging panel


DEMO_CASES: list[DemoCase] = [
    DemoCase(
        name="Case 1 — Elderly renal impairment turns a manageable pairing dangerous",
        drug_a="Lisinopril",
        drug_b="Trimethoprim",
        strain="E. coli ST131 (ESBL)",
        patient=PatientProfile(
            age=78,
            weight_kg=58,
            sex="Female",
            serum_creatinine=1.9,
            alt=32,
            albumin=3.6,
            bilirubin=0.9,
            serum_potassium=5.1,
            label="Elderly, reduced kidney function",
        ),
        pitch=(
            "The headline capability: population-average screening calls this "
            "combination MODERATE, but in the elderly renal subgroup it escalates "
            "to SEVERE. Both drugs raise potassium and the kidney that should "
            "excrete it is impaired. This is the kind of subgroup-specific signal "
            "that surfaces late in a trial as unexplained cardiac events."
        ),
    ),
    DemoCase(
        name="Case 2 — Structural prediction catches a rhabdomyolysis risk",
        drug_a="Simvastatin",
        drug_b="Clarithromycin",
        strain="S. aureus MRSA (hospital-acquired)",
        patient=PatientProfile(
            age=62,
            weight_kg=82,
            sex="Male",
            serum_creatinine=1.1,
            alt=48,
            albumin=4.0,
            bilirubin=1.0,
            serum_potassium=4.3,
            label="Middle-aged, mild liver enzyme elevation",
        ),
        pitch=(
            "A documented severe interaction reproduced from molecular structure "
            "alone. Clarithromycin blocks CYP3A4, simvastatin's only clearance "
            "route, and the result is muscle breakdown. Module 3 adds the twist: "
            "the MRSA target already resists the beta-lactams clarithromycin "
            "would otherwise replace, so this pairing is hard to avoid clinically."
        ),
    ),
    DemoCase(
        name="Case 3 — Antibiotic candidate versus an untreatable organism",
        drug_a="Meropenem",
        drug_b="Furosemide",
        strain="K. pneumoniae (carbapenem-resistant)",
        patient=PatientProfile(
            age=54,
            weight_kg=70,
            sex="Male",
            serum_creatinine=1.4,
            alt=38,
            albumin=3.8,
            bilirubin=1.1,
            serum_potassium=3.9,
            label="ICU patient, mild renal impairment",
        ),
        pitch=(
            "The R&D portfolio question. Meropenem is a last-line carbapenem, but "
            "this NDM-1-carrying Klebsiella already defeats the entire class. "
            "TrialSense flags that developing another carbapenem against this "
            "organism is a dead end — the exact decision that should be made "
            "before spending on trials, not after."
        ),
    ),
    DemoCase(
        name="Case 4 — Enzyme induction: the failure mode that looks like nothing",
        drug_a="Warfarin",
        drug_b="Rifampicin",
        strain="M. tuberculosis (MDR-TB)",
        patient=PatientProfile(
            age=45,
            weight_kg=64,
            sex="Male",
            serum_creatinine=1.0,
            alt=55,
            albumin=3.9,
            bilirubin=1.2,
            serum_potassium=4.1,
            label="TB treatment cohort, normal organ function",
        ),
        pitch=(
            "Most interaction tools only look for toxicity. Rifampicin does the "
            "opposite — it speeds up warfarin's clearance until warfarin stops "
            "working, producing clots rather than bleeds. In a trial this reads "
            "as 'the drug didn't work', not 'the drug was unsafe', and the cause "
            "is missed. Highly relevant given India's MDR-TB burden."
        ),
    ),
    DemoCase(
        name="Case 5 — A clean pairing, correctly cleared",
        drug_a="Amoxicillin",
        drug_b="Metformin",
        strain="E. coli ATCC 25922 (susceptible reference)",
        patient=PatientProfile(
            age=41,
            weight_kg=76,
            sex="Female",
            serum_creatinine=0.85,
            alt=24,
            albumin=4.4,
            bilirubin=0.7,
            serum_potassium=4.2,
            label="Healthy adult, normal organ function",
        ),
        pitch=(
            "The control case, and the one that proves the tool is usable. A "
            "screening system that flags everything is worthless — over-alerting "
            "is why clinicians ignore interaction warnings. Here every module "
            "returns green and the recommendation is simply to proceed."
        ),
    ),
]

CASES_BY_NAME = {c.name: c for c in DEMO_CASES}
