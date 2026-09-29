"""
Derive the resistance-call threshold from a stated cost, on both datasets.

THE PROBLEM
-----------
The shipped threshold was 0.35, justified in a code comment by a table reading
"very major 1.45%, major 8.38%". Those numbers predated a retrain and were
never re-measured; the live figures were 7.41% and 3.65%, so the justification
was false. The sweep's own recommendation was 0.05. The comment also argued
that 0.35 matches Module 1's screening threshold, which is symmetry, not
evidence.

WHAT A THRESHOLD ACTUALLY DEPENDS ON
------------------------------------
There is no threshold that is correct in the abstract. Choosing one requires
saying how much worse a missed resistance is than a false alarm:

    very major error  we predicted susceptible, the laboratory says resistant.
                      The drug fails in patients. In pre-clinical screening
                      this means a candidate advances that should not have.
    major error       we predicted resistant, the laboratory says susceptible.
                      One confirmatory assay recovers it.

COST_RATIO below is that judgement, stated as a number so it can be argued
with. At 10, one missed resistance is treated as costing the same as ten
unnecessary confirmatory assays. We pick the threshold minimising

    expected cost = COST_RATIO * P(very major) + P(major)

BOTH DATASETS, BECAUSE THEY DISAGREE
------------------------------------
The sweep is run twice:

  CONSTRUCTED   700 isolates built by synthesize_isolate. This is what the
                model was trained on, and the threshold that looks best here
                is the one that will look best in the demo.

  REAL          350 laboratory susceptibility results from 42 real genomes.
                This is the one that matters, and on it the model scores near
                chance, so no threshold rescues it.

Reporting only the first would repeat the mistake this file exists to correct.

Run:  python derive_threshold.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from trialsense.amr import ANTIBIOTIC_CLASSES

MODELS = Path("models")
OUT = MODELS / "threshold_derivation.json"

# How many unnecessary confirmatory assays one missed resistance is worth.
COST_RATIO = 10.0
GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def sweep(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    rows = []
    n_res = int(y_true.sum())
    n_sus = int((1 - y_true).sum())
    for t in GRID:
        pred = (y_prob >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        vme = fn / n_res if n_res else 0.0
        me = fp / n_sus if n_sus else 0.0
        rows.append({
            "threshold": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "very_major": vme, "major": me,
            "accuracy": (tp + tn) / len(y_true),
            "expected_cost": COST_RATIO * vme + me,
        })
    return rows


def constructed() -> tuple[np.ndarray, np.ndarray] | None:
    """Held-out constructed isolates, scored by the shipped model."""
    from trialsense.amr import build_amr_dataset

    path = MODELS / "amr_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        model = pickle.load(fh)

    # seed 99 is not the training seed, so these are isolates the model has
    # not seen, built by the same generator.
    X, Y, _ = build_amr_dataset(n_isolates=300, seed=99)
    per_class = model.model.predict_proba(X)
    P = np.column_stack([
        col[:, 1] if col.shape[1] > 1 else np.zeros(len(X)) for col in per_class
    ])
    return Y.ravel(), P.ravel()


def real() -> tuple[np.ndarray, np.ndarray] | None:
    """Real genomes with laboratory AST."""
    path = MODELS / "validation_real.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    y, p = [], []
    for iso in data["per_isolate"]:
        for _cls, d in iso["classes"].items():
            y.append(d["lab"])
            p.append(d["p"])
    return np.array(y), np.array(p)


def report(name: str, pair, out: dict) -> None:
    if pair is None:
        print(f"\n{name}: unavailable")
        return
    y, p = pair
    rows = sweep(y, p)
    best = min(rows, key=lambda r: r["expected_cost"])
    out[name] = {"n": int(len(y)), "rows": rows, "best": best}

    print(f"\n{name.upper()}  ({len(y)} comparisons, "
          f"{int(y.sum())} resistant / {int((1-y).sum())} susceptible)")
    print("  %-10s %10s %10s %10s %12s" %
          ("threshold", "very major", "major", "accuracy", "cost"))
    for r in rows:
        mark = "  <- lowest cost" if r is best else ""
        if r["threshold"] in (0.05, 0.35, 0.50) or r is best:
            print("  %-10.2f %9.1f%% %9.1f%% %9.3f %12.3f%s"
                  % (r["threshold"], 100 * r["very_major"], 100 * r["major"],
                     r["accuracy"], r["expected_cost"], mark))


def main() -> dict:
    out: dict = {
        "cost_ratio": COST_RATIO,
        "cost_model": "expected_cost = COST_RATIO * P(very major) + P(major)",
        "note": "COST_RATIO is a stated judgement, not a measurement. "
                "Change it and the recommended threshold changes.",
    }
    print("=" * 78)
    print(f"Threshold derivation  (one missed resistance = {COST_RATIO:.0f} "
          "unnecessary confirmatory assays)")
    print("=" * 78)

    report("constructed", constructed(), out)
    report("real", real(), out)

    if "constructed" in out and "real" in out:
        c = out["constructed"]["best"]["threshold"]
        r = out["real"]["best"]["threshold"]
        out["recommendation"] = {
            "constructed_optimum": c,
            "real_optimum": r,
            "agree": bool(abs(c - r) < 1e-9),
        }
        print("\n" + "=" * 78)
        print(f"constructed data recommends {c:.2f}; real data recommends {r:.2f}")
        if abs(c - r) > 1e-9:
            print("They disagree. The real-data figure is the one that describes")
            print("behaviour on bacteria nobody constructed, and it is the one to")
            print("quote — together with the fact that accuracy there is near chance,")
            print("which no threshold fixes.")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwritten -> {OUT}")
    return out


if __name__ == "__main__":
    main()
