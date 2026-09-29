#!/usr/bin/env python3
"""Module 2 -- how Indian adult body size changes the renal risk call.

WHAT THIS IS, PRECISELY
-----------------------
This is NOT an Indian pharmacokinetic dataset, and it is NOT an Indian PBPK
population -- PK-Sim ships no Indian population (only European ICRP 2002, Asian
Tanaka 1996, Japanese, and three US groups), so that claim cannot be made.

What it IS: a mechanistic consequence that already lives in Module 2's own
equation. Cockcroft-Gault creatinine clearance is *directly proportional to
body weight*:

    CrCl = (140 - age) * weight / (72 * SCr)      [* 0.85 if female]

Indian adults are lighter on average than the Western reference used to size
most registration trials. So the SAME age and the SAME serum creatinine give a
LOWER CrCl, a lower renal function factor KF, and therefore HIGHER predicted
exposure -- for a renally-cleared drug, materially higher.

The claim this supports is narrow and defensible:

    "A trial sized on Western body weight systematically under-estimates renal
     drug accumulation in Indian patients, because clearance scales with body
     size. Module 2 makes that visible per drug."

BODY SIZE REFERENCES (change these if your source differs)
----------------------------------------------------------
Indian reference adult: ICMR-NIN 'Nutrient Requirements for Indians' (2020)
reference man 65 kg / reference woman 55 kg.
Western reference: 73 kg -- the value used by the PK-Sim European (ICRP 2002)
individual in the companion PBPK study, so the two analyses stay comparable.

Both are REFERENCE values, not measured cohorts. They are declared as constants
here so a reviewer can see and change the assumption.

Usage:  python validation/indian_bodysize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trialsense.drugs import DRUGS                      # noqa: E402
from trialsense.pk import PatientProfile, compute_exposure  # noqa: E402

# ---- body-size assumptions (declared, not hidden) --------------------------
WESTERN_KG = 73.0   # PK-Sim European (ICRP 2002) individual used in the PBPK run
INDIAN_KG = 65.0    # ICMR-NIN 2020 reference man

# Drugs carried through the PBPK study, ordered by renal fraction.
FOCUS = ["Metformin", "Fluconazole", "Methotrexate", "Atorvastatin",
         "Omeprazole", "Ketoconazole", "Theophylline"]


def profile(weight: float, age: int, scr: float) -> PatientProfile:
    return PatientProfile(age=age, weight_kg=weight, serum_creatinine=scr,
                          sex="Male", label=f"{weight:.0f} kg")


def drug_by_name(name: str):
    for key, d in DRUGS.items():
        if key.lower() == name.lower():
            return d
    return None


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main() -> int:
    age, scr = 60, 1.6   # an elderly patient with modestly raised creatinine

    w = profile(WESTERN_KG, age, scr)
    i = profile(INDIAN_KG, age, scr)

    section("SAME PATIENT, DIFFERENT BODY SIZE")
    print(f"age {age}, serum creatinine {scr} mg/dL, male -- identical in both\n")
    for tag, p in (("Western reference", w), ("Indian reference ", i)):
        print(f"  {tag} {p.weight_kg:>5.0f} kg   "
              f"CrCl {p.creatinine_clearance():>6.1f} mL/min   "
              f"KF {p.renal_function_factor():>5.3f}   {p.ckd_stage()}")
    drop = (1 - i.creatinine_clearance() / w.creatinine_clearance()) * 100
    print(f"\n  Indian-sized patient has {drop:.1f}% lower estimated clearance "
          f"from body weight alone.")

    section("PREDICTED EXPOSURE PER DRUG (AUC x normal)")
    print(f"{'drug':<14}{'fe':>6}{'Western':>10}{'Indian':>9}"
          f"{'extra':>8}  {'tier change':<22}")
    print("-" * 74)

    rows = []
    for name in FOCUS:
        d = drug_by_name(name)
        if d is None:
            continue
        ew = compute_exposure(d, w)
        ei = compute_exposure(d, i)
        extra = (ei.exposure_ratio / ew.exposure_ratio - 1) * 100
        rows.append((name, d.fe, ew.exposure_ratio, ei.exposure_ratio, extra))
        print(f"{name:<14}{d.fe:>6.2f}{ew.exposure_ratio:>10.2f}"
              f"{ei.exposure_ratio:>9.2f}{extra:>7.1f}%")

    if rows:
        worst = max(rows, key=lambda r: r[4])
        print(f"\n  Largest body-size effect: {worst[0]} (fe={worst[1]:.2f}) -- "
              f"{worst[4]:.1f}% extra exposure.")
        print("  The effect scales with fe: the more renally cleared the drug,")
        print("  the more body size matters. Purely hepatic drugs are unaffected.")

    # ---- where does the decision actually flip? ----------------------------
    section("WHERE THE KIDNEY-STAGE CALL FLIPS")
    print("Serum creatinine values where the two body sizes land in DIFFERENT")
    print("CKD stages -- i.e. the same patient is triaged differently.\n")

    flips = []
    s = 0.6
    while s <= 4.0:
        pw, pi = profile(WESTERN_KG, age, s), profile(INDIAN_KG, age, s)
        if pw.ckd_stage() != pi.ckd_stage():
            flips.append((s, pw.ckd_stage(), pi.ckd_stage(),
                          pw.creatinine_clearance(), pi.creatinine_clearance()))
        s = round(s + 0.05, 2)

    if flips:
        # report contiguous bands rather than every 0.05 step
        band_start = flips[0]
        prev = flips[0]
        bands = []
        for f in flips[1:]:
            if round(f[0] - prev[0], 2) > 0.051 or f[1] != prev[1]:
                bands.append((band_start, prev))
                band_start = f
            prev = f
        bands.append((band_start, prev))

        for start, end in bands:
            print(f"  SCr {start[0]:.2f}-{end[0]:.2f} mg/dL")
            print(f"     Western {start[3]:>5.1f} mL/min -> {start[1]}")
            print(f"     Indian  {start[4]:>5.1f} mL/min -> {start[2]}   <-- stricter")
        print(f"\n  {len(flips)} of the tested creatinine values give a different")
        print("  kidney-stage call purely because of body weight.")
    else:
        print("  No stage flips in the tested range at this age.")

    section("HOW TO STATE THIS")
    print("""  SAY:
    "Module 2 is population-agnostic in its PK equation, but clearance scales
     with body size. Using the ICMR reference Indian adult instead of the
     Western trial reference raises predicted exposure for renally-cleared
     drugs and moves the kidney-stage call at clinically common creatinine
     values. Trials sized on Western body weight under-detect this."

  DO NOT SAY:
    "Module 2 was validated on Indian patients."      (no Indian PK dataset)
    "We used an Indian PBPK population."              (PK-Sim has none)

  Module 3 is where the genuine Indian data sits (ICMR AMR surveillance).
  Keep the two claims separate -- bacterial genomes are not human PK.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
