#!/usr/bin/env python3
"""
TrialSense — train and cache both ML models.

Run this once before the app:

    python train.py

Writes to models/:
    ddi_model.pkl   Random Forest over paired molecular fingerprints
    amr_model.pkl   Multi-label Random Forest over genomic k-mer profiles
    metrics.json    Held-out evaluation numbers, displayed in the app

The app loads these caches at startup so the demo opens instantly. If a cache
is missing the app trains on first launch instead, which just takes longer.

Total runtime is roughly one to two minutes on a laptop CPU.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    started = time.time()

    # --- Step 0: verify every molecular structure before anything else -------
    print("=" * 70)
    print("TrialSense — model training")
    print("=" * 70)
    print("\n[0/4] Validating drug structures against reference molecular weights…")
    from trialsense.drugs import DRUGS, validate_drug_db

    problems = validate_drug_db()
    if problems:
        print(f"  FAILED — {len(problems)} structure(s) do not match:")
        for p in problems:
            print("    -", p)
        return 1
    print(f"  OK — all {len(DRUGS)} structures parse and match their reference MW.")

    # --- Step 1: DDI model ---------------------------------------------------
    print("\n[1/4] Module 1 — drug-drug interaction model")
    from trialsense.ddi import DDIModel, build_dataset, evaluate

    t0 = time.time()
    X, y, pairs = build_dataset()
    print(f"  Built {X.shape[0]} drug pairs x {X.shape[1]} structural features.")

    print("  Task: binary screening — is this pairing clinically significant?")
    ddi_metrics = evaluate()
    rs, cs = ddi_metrics["random_split"], ddi_metrics["cold_drug_split"]
    print(
        f"  Random pair split : ROC-AUC {rs['roc_auc']:.3f} | "
        f"recall {rs['recall']:.3f} | precision {rs['precision']:.3f}"
    )
    print(
        f"  Cold-drug split   : ROC-AUC {cs['roc_auc']:.3f} (±{cs['roc_auc_std']:.3f}) | "
        f"recall {cs['recall']:.3f} | precision {cs['precision']:.3f}"
    )
    print(
        f"                      pooled over {cs['n_repeats']} hold-outs, "
        f"n={cs['n_test']}, flags {cs['flagged_rate']:.0%} of pairs"
    )
    print(f"    (one hold-out was: {', '.join(cs['held_out_drugs_example'])})")

    ddi_model = DDIModel().fit(X, y)  # final model uses all data
    ddi_model.metrics = ddi_metrics
    with open(MODELS_DIR / "ddi_model.pkl", "wb") as fh:
        pickle.dump(ddi_model, fh)
    print(f"  Saved models/ddi_model.pkl ({time.time() - t0:.1f}s)")

    # --- Step 2: AMR model ---------------------------------------------------
    print("\n[2/4] Module 3 — antimicrobial resistance model")
    from trialsense.amr import AMRModel, build_amr_dataset, evaluate_amr

    t0 = time.time()
    Xa, Ya, combos = build_amr_dataset()
    print(f"  Synthesised {Xa.shape[0]} isolates x {Xa.shape[1]} k-mer features.")

    amr_metrics = evaluate_amr()
    ars = amr_metrics["random_split"]
    print(
        f"  Random split      : exact-match {ars['exact_match']:.3f} | "
        f"per-label {ars['per_label_accuracy']:.3f} | macro-F1 {ars['macro_f1']:.3f}"
    )
    if "cold_combination_split" in amr_metrics:
        acs = amr_metrics["cold_combination_split"]
        print(
            f"  Cold-combination  : exact-match {acs['exact_match']:.3f} | "
            f"per-label {acs['per_label_accuracy']:.3f} | macro-F1 {acs['macro_f1']:.3f}"
        )

    amr_model = AMRModel().fit(Xa, Ya)
    amr_model.metrics = amr_metrics
    stats = amr_model.training_stats
    print(
        f"  Recorded training distribution for out-of-distribution scoring "
        f"(n={stats.get('n_train', 0)})."
    )
    with open(MODELS_DIR / "amr_model.pkl", "wb") as fh:
        pickle.dump(amr_model, fh)
    print(f"  Saved models/amr_model.pkl ({time.time() - t0:.1f}s)")

    # --- Step 3: resistance-trend forecaster --------------------------------
    # Module 3 Feature 1 projects national resistance to a candidate's launch
    # year. A projection nobody has tested is decoration, so we time-split it
    # here: fit on early years only, predict the years held out, measure error.
    print("\n[3/4] Module 3 Feature 1 — resistance-trend forecaster")
    from trialsense.surveillance import SURVEILLANCE_DATA, backtest_forecast

    n_series = sum(
        len(abx) for org in SURVEILLANCE_DATA["icmr"].values() for abx in [org]
    )
    print(
        f"  Loaded real surveillance series for "
        f"{len(SURVEILLANCE_DATA['icmr'])} organisms (ICMR AMRSN 2016-2024)."
    )
    forecast_metrics = backtest_forecast(cutoff_year=2020)
    if forecast_metrics.get("available"):
        fm = forecast_metrics
        print(
            f"  Time-split backtest (fit <=2020, predict 2021-2024): "
            f"MAE {fm['mae_pp']:.2f} pp over {fm['n_predictions']} predictions "
            f"across {fm['n_series']} series"
        )
        print(
            f"    within 5 pp: {fm['within_5pp']:.0%} | "
            f"within 10 pp: {fm['within_10pp']:.0%}"
        )
    else:
        print("  Not enough multi-year data to backtest.")

    # --- Step 4: metrics for the app's Methods tab ---------------------------
    print("\n[4/4] Writing metrics")
    with open(MODELS_DIR / "metrics.json", "w") as fh:
        json.dump(
            {
                "ddi": ddi_metrics,
                "amr": amr_metrics,
                "forecast": forecast_metrics,
            },
            fh,
            indent=2,
        )
    print("  Saved models/metrics.json")

    print("\n" + "=" * 70)
    print(f"Done in {time.time() - started:.1f}s.  Now run:  streamlit run app.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
