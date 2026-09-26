#!/usr/bin/env python3
"""
TrialSense — validate Module 2 against real renal impairment data.

    python validate_pk.py

Reads data/renal_validation.json (sourced from FDA labels and published
studies, every number with a verbatim quote and URL), compares Module 2's
predicted exposure change with what was actually measured, prints a per-drug
table and writes models/pk_validation.json for the app's Methods tab.
"""

from __future__ import annotations

import json
from pathlib import Path

from trialsense.pk_validation import decision_band, validate

OUT = Path(__file__).parent / "models" / "pk_validation.json"


def main() -> int:
    res = validate()
    s = res.summary()
    if not s["n_observations"]:
        print("No scorable observations in data/renal_validation.json.")
        return 1

    print("=" * 96)
    print("Module 2 validation — predicted vs observed exposure (AUC ratio) in renal impairment")
    print("=" * 96)
    print(f"{'Drug':16s} {'Group':34s} {'GFR':>5s} {'Obs':>6s} {'Pred':>6s} {'Fold':>5s}  Bands (obs → pred)")
    for o in sorted(res.scored, key=lambda o: (o.drug, -o.gfr)):
        flag = "" if o.band_agrees else "  ✗"
        obs = "  same" if o.no_change_statement else f"{o.observed:6.2f}"
        fold = "    -" if o.no_change_statement else f"{o.fold_error:5.2f}"
        print(
            f"{o.drug:16s} {o.renal_group[:34]:34s} {o.gfr:5.0f} {obs} "
            f"{o.predicted:6.2f} {fold}  "
            f"{decision_band(o.observed)} → {decision_band(o.predicted)}{flag}"
        )

    print("-" * 96)
    print(f"Observations scored : {s['n_observations']} across {s['n_drugs']} drugs "
          f"({s['n_excluded']} excluded: half-life only, dialysis or qualitative)")
    print(f"Within 1.25× / 1.5× / 2× : {s['within_1_25x']:.0%} / {s['within_1_5x']:.0%} / "
          f"{s['within_2x']:.0%}")
    print(f"GMFE                : {s['gmfe']:.2f}   (1.00 = perfect)")
    print(f"Bias (pred/obs)     : {s['bias']:.2f}   (<1 = under-predicts)")
    print(f"Decision-band match : {s['band_agreement']:.0%}  (all rows, incl. "
          f"{s['n_no_change_statements']} 'no change' label statements — "
          f"{s['no_change_statements_matched']} matched)")
    print(f"Meaningful (≥2×) changes missed: {s['meaningful_missed']} of "
          f"{s['meaningful_observed']}")
    if s["gmfe_with_label_fe"] is not None:
        print(f"GMFE using the source's own fe ({s['n_with_label_fe']} obs): "
              f"{s['gmfe_with_label_fe']:.2f}")
    print("\nStudy-design call per drug (the decision that costs money):")
    for c in res.design_calls():
        mark = "" if c["outcome"] == "correct" else f"   ← {c['outcome']}"
        print(f"  {c['drug']:16s} data support: {c['data_support']:14s} "
              f"recommended: {c['recommended']}{mark}")
    print(f"Design correct      : {s['design_correct']} of {s['design_total']} drugs "
          f"({s['design_missed_full']} missed full studies, "
          f"{s['design_unnecessary_full']} unnecessary)")
    if s["drugs_without_data"]:
        print(f"No usable data      : {', '.join(s['drugs_without_data'])}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "summary": s,
        "design_calls": res.design_calls(),
        "observations": [
            {**o.__dict__, "fold_error": round(o.fold_error, 3), "band_agrees": o.band_agrees}
            for o in res.scored
        ],
    }, indent=2))
    print(f"\nSaved {OUT.relative_to(Path(__file__).parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
