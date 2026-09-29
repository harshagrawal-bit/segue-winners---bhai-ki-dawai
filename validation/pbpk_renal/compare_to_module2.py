#!/usr/bin/env python3
"""Compare Module 2's analytic AUC ratio against the PK-Sim PBPK ground truth.

Reads pksim_renal_predictions.csv (written by run_pbpk_renal.R) and, for every
drug/eGFR arm that actually simulated, computes what trialsense.pk would have
predicted for the same kidney function.

Module 2's equation:
    CL_ratio  = fe * KF + fh * HF
    AUC_ratio = 1 / CL_ratio
with a healthy liver (HF = 1.0), so the only thing varying here is KF.

KF is taken as eGFR / (the healthy reference arm's realised eGFR) rather than
pk.py's CRCL_NORMAL constant, so both sides are normalised to the *same*
reference individual and the comparison isolates the equation rather than a
difference in what "normal" means.

Usage:  python compare_to_module2.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from trialsense.drugs import DRUGS  # noqa: E402

CSV_PATH = HERE / "pksim_renal_predictions.csv"
REF_EGFR_LABEL = 90


def fe_for(name: str) -> float | None:
    # DRUGS is a dict keyed by drug name -> Drug
    for key, d in DRUGS.items():
        if key.lower() == name.lower():
            return d.fe
    return None


def main() -> int:
    if not CSV_PATH.exists():
        print(f"missing {CSV_PATH} - run run_pbpk_renal.R first")
        return 1

    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == "ok"]

    if not rows:
        print("No simulated arms in the CSV yet. Nothing to compare.")
        return 0

    by_drug: dict[str, list[dict]] = {}
    for r in rows:
        by_drug.setdefault(r["drug"], []).append(r)

    all_folds: list[float] = []

    for drug, arms in by_drug.items():
        fe = fe_for(drug)
        if fe is None:
            print(f"{drug}: not in trialsense.drugs - skipped\n")
            continue
        fh = 1.0 - fe

        ref = next((a for a in arms
                    if int(float(a["egfr"])) == REF_EGFR_LABEL), None)
        if ref is None or not ref["egfr_actual"]:
            print(f"{drug}: no reference arm simulated - skipped\n")
            continue
        ref_egfr = float(ref["egfr_actual"])

        print(f"{drug}  (fe={fe:.2f}, fh={fh:.2f})   "
              f"reference eGFR {ref_egfr:.1f} mL/min/1.73m2")
        print(f"{'eGFR':>6} {'KF':>6} {'Module2':>9} {'PBPK':>8} "
              f"{'M2/PBPK':>8} {'error':>8} {'tail':>7}")
        print("-" * 60)

        folds = []
        for a in sorted(arms, key=lambda x: -float(x["egfr"])):
            egfr_actual = float(a["egfr_actual"]) if a["egfr_actual"] else None
            pbpk = float(a["auc_ratio"]) if a["auc_ratio"] else None
            if egfr_actual is None or pbpk is None:
                continue
            kf = min(1.0, egfr_actual / ref_egfr)
            m2 = 1.0 / (fe * kf + fh * 1.0)
            fold = m2 / pbpk
            tail = a["frac_auc_extrapolated"]
            tail_s = f"{100*float(tail):.0f}%" if tail else "-"
            print(f"{float(a['egfr']):>6.0f} {kf:>6.3f} {m2:>9.2f} "
                  f"{pbpk:>8.2f} {fold:>8.2f} "
                  f"{(m2-pbpk)/pbpk*100:>7.1f}% {tail_s:>7}")
            if int(float(a["egfr"])) != REF_EGFR_LABEL:
                folds.append(fold)

        if folds:
            gmfe = statistics.geometric_mean(folds)
            all_folds.extend(folds)
            print(f"\n  geometric mean fold-error (impaired arms): {gmfe:.2f}")
            print(f"  {'UNDER' if gmfe < 1 else 'OVER'}-predicts accumulation "
                  f"by {abs(1-gmfe)*100:.0f}% on average\n")

    if len(by_drug) > 1 and all_folds:
        print("=" * 60)
        print(f"ALL DRUGS  geometric mean fold-error: "
              f"{statistics.geometric_mean(all_folds):.2f}  "
              f"(n={len(all_folds)} impaired arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
