#!/usr/bin/env python3
"""
TrialSense — Module 3 validation suite.

    python validate_module3.py

THE QUESTION THIS FILE EXISTS TO ANSWER
---------------------------------------
"How do you know any of this actually helps?"

Accuracy alone does not answer it. A model can be accurate and still useless
for decisions, and it can be accurate and still dishonest about its own
confidence. So this runs seven distinct checks, each answering a different
question, and writes the results to models/validation.json for the app's
Methods tab to display.

    1. VERY MAJOR / MAJOR ERRORS   Is it wrong in the expensive direction?
    2. CALIBRATION                 When it says 80%, is it right 80% of the time?
    3. CLADE-HELD-OUT              Does it work on a lineage it never trained on?
    4. NOVELTY DETECTOR            Does the open-world scan actually detect, and
                                   does it stay quiet when it should?
    5. FORECAST BACKTEST           Does the resistance projection predict years
                                   it never saw?
    6. GENE COVERAGE               Do we model the determinants that actually
                                   matter in India?
    7. ICMR CROSS-CHECK            Do the portfolio grid's predictions line up
                                   with real national surveillance?

WHY ERROR TYPES ARE REPORTED SEPARATELY
---------------------------------------
Regulators assessing a susceptibility-testing device do not quote accuracy.
They count two error types separately, because the costs are wildly asymmetric:

    VERY MAJOR ERROR  predicted susceptible, actually resistant
                      -> the drug fails in patients. Programme-ending.

    MAJOR ERROR       predicted resistant, actually susceptible
                      -> a good candidate looks worse than it is. One
                         confirmatory assay recovers it.

Commercial devices are held to a very-major-error rate on the order of a few
percent while tolerating far more major errors. We report both and tune the
same way, which is also how Module 1's screening threshold is set.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent / "models"
OUT_PATH = MODELS_DIR / "validation.json"


def _hr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# =============================================================================
# 1 + 2. Error types and calibration
# =============================================================================


def _class_probabilities(model, X_te) -> np.ndarray:
    """P(resistant) per antibiotic class, as a (n_samples, n_classes) array."""
    return np.column_stack(
        [
            p[:, 1] if p.shape[1] > 1 else np.zeros(len(X_te))
            for p in model.model.predict_proba(X_te)
        ]
    )


def _calibration_stats(probs: np.ndarray, Y_te: np.ndarray) -> tuple[float, float, list]:
    """Brier score, expected calibration error, and the reliability table."""
    flat_p = probs.ravel()
    flat_y = Y_te.ravel().astype(float)
    brier = float(np.mean((flat_p - flat_y) ** 2))

    bins = np.linspace(0, 1, 11)
    reliability = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (flat_p >= lo) & (flat_p < hi if hi < 1.0 else flat_p <= 1.0)
        if m.sum() >= 5:
            reliability.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "predicted": float(flat_p[m].mean()),
                    "observed": float(flat_y[m].mean()),
                    "n": int(m.sum()),
                }
            )
    total = sum(r["n"] for r in reliability) or 1
    ece = sum(abs(r["predicted"] - r["observed"]) * r["n"] for r in reliability) / total
    return brier, float(ece), reliability


def evaluate_errors_and_calibration(seed: int = 42) -> dict:
    from sklearn.model_selection import train_test_split

    from trialsense.amr import RESISTANCE_CALL_THRESHOLD, AMRModel, build_amr_dataset

    X, Y, combos = build_amr_dataset(seed=seed)
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.25, random_state=seed
    )
    model = AMRModel(random_state=seed).fit(X_tr, Y_tr)
    probs = _class_probabilities(model, X_te)

    # Head-to-head against the uncalibrated forest. Shipping a calibration step
    # without measuring whether it helped would be the same unexamined-claim
    # problem this whole file exists to avoid.
    raw_model = AMRModel(random_state=seed, calibrate=False).fit(X_tr, Y_tr)
    raw_probs = _class_probabilities(raw_model, X_te)
    raw_brier, raw_ece, raw_reliability = _calibration_stats(raw_probs, Y_te)

    ar = Y_te == 1
    asus = Y_te == 0
    raw_vme_at = {}
    for t in (0.35, 0.50):
        rp = (raw_probs >= t).astype(int)
        raw_vme_at[t] = (
            float(((rp == 0) & ar).sum() / max(ar.sum(), 1)),
            float(((rp == 1) & asus).sum() / max(asus.sum(), 1)),
        )

    pred = (probs >= RESISTANCE_CALL_THRESHOLD).astype(int)

    # --- error types
    actually_resistant = Y_te == 1
    actually_susceptible = Y_te == 0
    vme = int(((pred == 0) & actually_resistant).sum())
    me = int(((pred == 1) & actually_susceptible).sum())
    n_r = int(actually_resistant.sum())
    n_s = int(actually_susceptible.sum())

    # --- calibration: Brier score plus a reliability table
    brier, ece, reliability = _calibration_stats(probs, Y_te)

    # --- threshold sweep
    # The default 0.5 cut is the wrong operating point for pre-trial screening.
    # A very major error (calling a resistant organism susceptible) kills a
    # programme; a major error costs one confirmatory assay. Regulators hold
    # susceptibility devices to a very low very-major rate while tolerating far
    # more major errors, and Module 1 already ships a recall-favouring
    # threshold for the same reason. This sweep finds the matching point here.
    sweep = []
    for t in np.arange(0.05, 0.65, 0.05):
        p = (probs >= t).astype(int)
        v = int(((p == 0) & actually_resistant).sum())
        m = int(((p == 1) & actually_susceptible).sum())
        sweep.append(
            {
                "threshold": round(float(t), 2),
                "very_major_error_rate": v / n_r if n_r else 0.0,
                "major_error_rate": m / n_s if n_s else 0.0,
                "flagged_rate": float(p.mean()),
            }
        )
    # HIGHEST threshold that still keeps very major errors under 1.5% — the
    # strict end of the range commercial devices are held to. Taking the
    # highest, not the lowest, matters: any threshold low enough satisfies the
    # very-major bound, but the lowest ones do so by flagging almost
    # everything, which makes the tool useless in a different way.
    target = next(
        (s for s in reversed(sweep) if s["very_major_error_rate"] <= 0.015), None
    )

    return {
        "n_test_isolates": int(len(Y_te)),
        "n_label_decisions": int(Y_te.size),
        "shipped_threshold": float(RESISTANCE_CALL_THRESHOLD),
        "calibration_comparison": {
            "raw_brier": raw_brier,
            "calibrated_brier": brier,
            "raw_ece": raw_ece,
            "calibrated_ece": ece,
            "raw_vme_at_035": raw_vme_at[0.35][0],
            "raw_me_at_035": raw_vme_at[0.35][1],
            "raw_vme_at_050": raw_vme_at[0.50][0],
            "raw_me_at_050": raw_vme_at[0.50][1],
            "raw_reliability": raw_reliability,
        },
        "very_major_errors": vme,
        "very_major_error_rate": vme / n_r if n_r else 0.0,
        "major_errors": me,
        "major_error_rate": me / n_s if n_s else 0.0,
        "n_actually_resistant": n_r,
        "n_actually_susceptible": n_s,
        "brier_score": brier,
        "expected_calibration_error": float(ece),
        "reliability": reliability,
        "threshold_sweep": sweep,
        "recommended_threshold": target["threshold"] if target else None,
        "recommended_vme": target["very_major_error_rate"] if target else None,
        "recommended_me": target["major_error_rate"] if target else None,
    }


# =============================================================================
# 3. Clade-held-out validation
# =============================================================================


def evaluate_clade_holdout(seed: int = 42) -> dict:
    """
    Hold out an entire species background, not random rows.

    A random split lets near-identical isolates land on both sides, which
    flatters the score. Published work calls this population-structure control
    and finds genomic AMR models generalise poorly across clades — so this is
    the split that tells you whether the model would survive a new lineage.
    """
    from sklearn.metrics import f1_score

    from trialsense.amr import (
        AMRModel,
        _realistic_gene_combinations,
        kmer_features,
        labels_for_genes,
        synthesize_isolate,
    )

    rng = np.random.RandomState(seed)
    gc_values = [0.327, 0.380, 0.390, 0.507, 0.521, 0.572, 0.656, 0.664]
    combos = _realistic_gene_combinations(rng, 700)

    X, Y, clade = [], [], []
    for genes in combos:
        gc = gc_values[rng.randint(len(gc_values))]
        seq = synthesize_isolate(genes, gc_content=gc, seed=int(rng.randint(1 << 30)))
        X.append(kmer_features(seq))
        Y.append(labels_for_genes(genes))
        clade.append(gc)
    X, Y, clade = np.vstack(X), np.vstack(Y), np.array(clade)

    results = []
    for held_gc in gc_values:
        mask = clade == held_gc
        if mask.sum() < 10 or (~mask).sum() < 50:
            continue
        m = AMRModel(random_state=seed).fit(X[~mask], Y[~mask])
        pred = np.asarray(m.model.predict(X[mask]))
        results.append(
            {
                "held_out_gc": float(held_gc),
                "n_test": int(mask.sum()),
                "macro_f1": float(
                    f1_score(Y[mask], pred, average="macro", zero_division=0)
                ),
                "per_label_accuracy": float((pred == Y[mask]).mean()),
            }
        )

    f1s = [r["macro_f1"] for r in results]
    return {
        "n_clades": len(results),
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "std_macro_f1": float(np.std(f1s)) if f1s else 0.0,
        "worst_macro_f1": float(np.min(f1s)) if f1s else 0.0,
        "per_clade": results,
    }


# =============================================================================
# 4. Novelty detector
# =============================================================================


def evaluate_novelty(seed: int = 7) -> dict:
    """
    Two questions, measured separately.

    SPECIFICITY  Do the unmodified demo strains stay quiet? Every determinant
                 in them is known, so any flag is a false positive.

    SENSITIVITY  When an element matching no reference is planted, is it found?
    """
    from trialsense.amr import DEMO_STRAINS, demo_strain_sequence, detect_genes, synthesize_isolate
    from trialsense.novelty import build_novelty_report

    def foreign(gc: float, n: int, s: int) -> str:
        r = np.random.RandomState(s)
        p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
        motifs = ["".join(r.choice(list("ACGT"), size=8, p=p)) for _ in range(14)]
        out: list[str] = []
        while sum(len(x) for x in out) < n:
            out.append(motifs[r.randint(len(motifs))])
        return "".join(out)[:n]

    from trialsense.novelty import novelty_scan

    false_positives, assessed, peaks_clean = 0, 0, []
    for name in DEMO_STRAINS:
        seq = demo_strain_sequence(name)
        rpt = build_novelty_report(seq, name, None, len(detect_genes(seq)))
        if not rpt.scan_available:
            continue
        assessed += 1
        # Record the peak divergence whether or not it cleared the floor —
        # "nothing fired" tells you less than "the closest it came was X".
        _, diag = novelty_scan(seq)
        peaks_clean.append(float(diag.get("peak_excess", 0.0)))
        if rpt.segments:
            false_positives += 1

    cases = [
        (["blaCTX-M-15", "sul1"], 0.507, 3, 0.80, 1100),
        (["blaNDM-1", "sul1"], 0.572, 7, 0.25, 1100),
        (["mecA", "ermB"], 0.327, 5, 0.75, 900),
        (["blaNDM-1", "blaCTX-M-15", "sul1", "aac(6')-Ib", "qnrS1"], 0.572, 7, 0.85, 1100),
        (["blaCTX-M-15"], 0.507, 3, 0.65, 1000),
        ([], 0.507, 11, 0.80, 1100),
        (["blaNDM-1", "blaCTX-M-15", "sul1", "aac(6')-Ib", "qnrS1"], 0.572, 7, 0.20, 1200),
    ]
    detected, total, peaks_spiked = 0, 0, []
    for genes, gc, s, fgc, flen in cases:
        host = synthesize_isolate(genes, gc_content=gc, seed=s)
        cut = len(host) // 2
        spiked = host[:cut] + foreign(fgc, flen, 99) + host[cut:]
        rpt = build_novelty_report(spiked, "spiked", None, 0)
        if not rpt.scan_available:
            continue
        total += 1
        if rpt.segments:
            detected += 1
            peaks_spiked.append(max(sg.excess_divergence for sg in rpt.segments))

    return {
        "clean_strains_assessed": assessed,
        "false_positives": false_positives,
        "false_positive_rate": false_positives / assessed if assessed else 0.0,
        "spiked_cases": total,
        "detected": detected,
        "sensitivity": detected / total if total else 0.0,
        "peak_divergence_clean": float(max(peaks_clean)) if peaks_clean else 0.0,
        "peak_divergence_spiked_min": float(min(peaks_spiked)) if peaks_spiked else 0.0,
    }


# =============================================================================
# 5. Gene detection vs real Indian gene prevalence
# =============================================================================


def evaluate_gene_detection() -> dict:
    """
    Sanity-check the gene-detection layer against real ICMR molecular
    surveillance, and be explicit about what this does and does not show.

    ICMR reports how often each gene occurs across Indian isolates. Our demo
    strains are hand-built, so their gene frequencies are OUR choice, not a
    sample of India. Comparing the two cannot validate accuracy — what it can
    show is whether the determinants we chose to model are the ones that
    actually matter in the Indian context.
    """
    from trialsense.amr import DEMO_STRAINS, demo_strain_sequence, detect_genes
    from trialsense.surveillance import ICMR_GENE_PREVALENCE, ICMR_GENE_PREVALENCE_NOTE

    found_counts: dict[str, int] = {}
    n = 0
    for name in DEMO_STRAINS:
        n += 1
        for gene, _ in detect_genes(demo_strain_sequence(name)):
            found_counts[gene] = found_counts.get(gene, 0) + 1

    rows = []
    for gene, real in sorted(ICMR_GENE_PREVALENCE.items(), key=lambda kv: -kv[1]):
        rows.append(
            {
                "gene": gene,
                "icmr_prevalence": real,
                "in_our_reference_set": gene in __import__(
                    "trialsense.amr", fromlist=["RESISTANCE_GENES"]
                ).RESISTANCE_GENES,
                "demo_strains_carrying": found_counts.get(gene, 0),
                "demo_strain_frequency": found_counts.get(gene, 0) / n if n else 0.0,
            }
        )
    covered = [r for r in rows if r["in_our_reference_set"]]
    return {
        "note": ICMR_GENE_PREVALENCE_NOTE,
        "caveat": (
            "Demo-strain frequencies are a design choice, not a sample of Indian "
            "isolates. This table shows COVERAGE of the determinants that matter "
            "in India, not predictive accuracy."
        ),
        "n_demo_strains": n,
        "icmr_genes_covered": len(covered),
        "icmr_genes_listed": len(rows),
        "rows": rows,
    }


# =============================================================================
# 8. The deployed pipeline, end to end
# =============================================================================


def evaluate_deployed_pipeline() -> dict:
    """
    Audit what the app ACTUALLY ships, not just the classifier inside it.

    Check 1 measures the classifier alone on synthetic test isolates. That is
    the right way to measure a classifier, but it is not what a user sees: the
    deployed path is classifier + gene detection + mechanism floor, run over
    the bundled strains whose true resistance we know exactly from the genes
    we put in them.

    Measuring the component and calling it the product is a common way to
    overstate a system. This measures the product.
    """
    import pickle

    from trialsense.amr import (
        DEMO_STRAINS,
        RESISTANCE_CONFIRMED_THRESHOLD,
        RESISTANCE_GENES,
        analyze_isolate,
        demo_strain_sequence,
    )

    path = MODELS_DIR / "amr_model.pkl"
    if not path.exists():
        return {"available": False, "reason": "run train.py first"}
    with open(path, "rb") as fh:
        model = pickle.load(fh)

    def audit(use_floor: bool) -> dict:
        vme, me, total = 0, 0, 0
        errors = []
        for name, strain in DEMO_STRAINS.items():
            truth = set()
            for g in strain.genes:
                truth |= set(RESISTANCE_GENES[g].confers_resistance_to)
            rpt = analyze_isolate(
                demo_strain_sequence(name), model, name,
                use_mechanism_floor=use_floor,
            )
            for cls, p in rpt.probabilities.items():
                total += 1
                says_resistant = p >= RESISTANCE_CONFIRMED_THRESHOLD
                is_resistant = cls in truth
                if is_resistant and not says_resistant:
                    vme += 1
                    errors.append({
                        "type": "very_major", "strain": name,
                        "antibiotic_class": cls, "probability": float(p),
                    })
                elif (not is_resistant) and says_resistant:
                    me += 1
                    errors.append({
                        "type": "major", "strain": name,
                        "antibiotic_class": cls, "probability": float(p),
                    })
        return {
            "very_major": vme, "major": me, "total_calls": total,
            "accuracy": (total - vme - me) / total if total else 0.0,
            "errors": errors,
        }

    without = audit(False)
    with_floor = audit(True)
    return {
        "available": True,
        "n_strains": len(DEMO_STRAINS),
        "without_mechanism_floor": without,
        "with_mechanism_floor": with_floor,
        "very_major_fixed": without["very_major"] - with_floor["very_major"],
        "major_introduced": with_floor["major"] - without["major"],
    }


# =============================================================================
# 7. Feature 3 grid vs real national surveillance
# =============================================================================


def evaluate_cross_check() -> dict:
    """
    Lay every Feature 3 prediction against real ICMR national data.

    Feature 3 produces no data of its own — it orchestrates the core
    classifier — so without this it would have no validation claim beyond
    "the grid has the right number of cells". This is how it earns one.

    Read the result carefully. Our figure describes ONE isolate; ICMR
    describes a NATIONAL POPULATION. A strain built to carry a determinant
    should read above the national average, so each comparison is judged
    against its expected direction and only the unexpected ones are flagged.
    """
    import pickle

    from trialsense.amr import (
        ANTIBIOTIC_CLASSES,
        DEMO_STRAINS,
        RESISTANCE_GENES,
        analyze_isolate,
        demo_strain_sequence,
    )
    from trialsense.surveillance import STRAIN_TO_ORGANISM, cross_check_predictions

    path = MODELS_DIR / "amr_model.pkl"
    if not path.exists():
        return {"available": False, "reason": "run train.py first"}
    with open(path, "rb") as fh:
        model = pickle.load(fh)

    strains = [s for s in DEMO_STRAINS if STRAIN_TO_ORGANISM.get(s)]
    reports = {
        s: analyze_isolate(demo_strain_sequence(s), model, s) for s in strains
    }
    cc = cross_check_predictions(
        reports,
        ANTIBIOTIC_CLASSES,
        {s: DEMO_STRAINS[s].genes for s in strains},
        {g: gg.confers_resistance_to for g, gg in RESISTANCE_GENES.items()},
    )
    return {
        "available": True,
        "verdict": cc.verdict,
        "reference_year": cc.reference_year,
        "n_strains": len(strains),
        "n_rows": len(cc.rows),
        "n_compared": len(cc.compared),
        "n_aligned": cc.n_aligned,
        "n_as_expected": cc.n_as_expected,
        "n_concern": cc.n_concern,
        "n_flagged": len(cc.flagged),
        "n_no_reference": cc.n_no_reference,
        "mae_pp": cc.mae_pp,
        "aligned_mae_pp": cc.aligned_mae_pp,
        "flagged": [
            {
                "organism": r.organism,
                "strain": r.strain_name,
                "antibiotic_class": r.antibiotic_class,
                "model_pct": r.model_pct,
                "real_pct": r.real_pct,
                "delta": r.delta,
                "status": r.status,
                "carries_determinant": r.carries_determinant,
                "determinants": r.determinants,
                "explanation": r.explain(),
            }
            for r in cc.flagged
        ],
    }


# =============================================================================
# Driver
# =============================================================================


def main() -> int:
    started = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    out: dict = {}

    print("=" * 74)
    print("TrialSense — Module 3 validation suite")
    print("=" * 74)

    _hr("1 + 2. Error types and calibration")
    ec = evaluate_errors_and_calibration()
    out["errors_calibration"] = ec
    print(
        f"  Very major errors (predicted S, actually R): {ec['very_major_errors']} "
        f"of {ec['n_actually_resistant']}  = {ec['very_major_error_rate']:.2%}"
    )
    print(
        f"  Major errors      (predicted R, actually S): {ec['major_errors']} "
        f"of {ec['n_actually_susceptible']}  = {ec['major_error_rate']:.2%}"
    )
    print(f"  Brier score (lower is better):              {ec['brier_score']:.4f}")
    print(f"  Expected calibration error:                 {ec['expected_calibration_error']:.4f}")
    cmp_ = ec.get("calibration_comparison")
    if cmp_:
        print()
        print("  Did isotonic calibration actually help? Raw forest vs shipped:")
        print(f"    Brier            {cmp_['raw_brier']:.4f} -> {cmp_['calibrated_brier']:.4f}")
        print(f"    Calibration err  {cmp_['raw_ece']:.4f} -> {cmp_['calibrated_ece']:.4f}")
        print(f"    very-major @0.35 {cmp_['raw_vme_at_035']:.2%} -> "
              f"{ec['very_major_error_rate']:.2%}")
        print(f"    major      @0.35 {cmp_['raw_me_at_035']:.2%} -> "
              f"{ec['major_error_rate']:.2%}")
    print("\n  Reliability — when we say X%, what actually happens:")
    for r in ec["reliability"]:
        gap = r["observed"] - r["predicted"]
        flag = "  <-- under-confident" if gap > 0.15 else ""
        print(
            f"    predicted {r['predicted']:.2f} -> observed {r['observed']:.2f} "
            f"(n={r['n']}){flag}"
        )

    print("\n  Threshold sweep — trading the cheap error for the expensive one:")
    print("    thresh |  very major |    major | flagged")
    for s in ec["threshold_sweep"]:
        print(
            f"      {s['threshold']:.2f} |     {s['very_major_error_rate']:6.2%} "
            f"|  {s['major_error_rate']:6.2%} | {s['flagged_rate']:6.1%}"
        )
    if ec["recommended_threshold"] is not None:
        print(
            f"\n  Shipped threshold {ec['shipped_threshold']:.2f}: very-major "
            f"{ec['very_major_error_rate']:.2%}, major "
            f"{ec['major_error_rate']:.2%}."
            f"\n  The sweep's own pick for a <=1.5% very-major rate is "
            f"{ec['recommended_threshold']:.2f} "
            f"({ec['recommended_vme']:.2%} / {ec['recommended_me']:.2%})."
        )
        # Compute the trade rather than asserting it. An earlier version of
        # this file hardcoded "the gap is 0.16 pp", which was true of the
        # numbers at the time and became false the moment the reference
        # sequences changed. A justification that does not recompute is a
        # justification that will eventually lie.
        vme_gap = ec["very_major_error_rate"] - ec["recommended_vme"]
        me_gap = ec["recommended_me"] - ec["major_error_rate"]
        print(
            f"\n  Moving to {ec['recommended_threshold']:.2f} would cut very-major "
            f"errors by {vme_gap:.2%} and raise major errors by {me_gap:.2%}."
        )
        if me_gap > 5 * max(vme_gap, 1e-9):
            print(
                "  That is a poor trade — it buys a small safety gain by "
                "flagging far more\n  of everything, which makes the tool "
                "useless in a different way."
            )
        print(
            f"  We ship {ec['shipped_threshold']:.2f} because it is exactly "
            "Module 1's screening threshold,\n  so both modules answer the same "
            "question the same way."
        )
        print(
            "\n  NOTE: these figures are for the CLASSIFIER ALONE. Check 8 "
            "measures the\n  deployed pipeline, where gene detection overrides "
            "it — that is the number\n  that describes the product."
        )
    else:
        print(
            "\n  No threshold in the swept range reaches a 1.5% very-major-error "
            "rate. Report that plainly."
        )

    _hr("3. Clade-held-out validation")
    ch = evaluate_clade_holdout()
    out["clade_holdout"] = ch
    print(
        f"  Held out {ch['n_clades']} species backgrounds in turn.\n"
        f"  macro-F1 {ch['mean_macro_f1']:.3f} +/- {ch['std_macro_f1']:.3f} "
        f"(worst clade {ch['worst_macro_f1']:.3f})"
    )

    _hr("4. Dark Genome detector")
    nv = evaluate_novelty()
    out["novelty"] = nv
    print(
        f"  Specificity: {nv['false_positives']} false positive(s) across "
        f"{nv['clean_strains_assessed']} clean strains "
        f"({nv['false_positive_rate']:.0%})"
    )
    print(
        f"  Sensitivity: {nv['detected']}/{nv['spiked_cases']} planted unknown "
        f"elements detected ({nv['sensitivity']:.0%})"
    )
    print(
        f"  Separation : clean peak {nv['peak_divergence_clean']:.3f} vs "
        f"spiked minimum {nv['peak_divergence_spiked_min']:.3f}"
    )

    _hr("5. Resistance-trend forecaster (time-split backtest)")
    from trialsense.surveillance import backtest_forecast

    fc = backtest_forecast(cutoff_year=2020)
    out["forecast_backtest"] = fc
    if fc.get("available"):
        print(
            f"  Fit on years <= {fc['cutoff_year']}, predicted "
            f"{fc['n_predictions']} held-out observations across "
            f"{fc['n_series']} series."
        )
        print(f"  Mean absolute error: {fc['mae_pp']:.2f} percentage points")
        print(f"  Median absolute error: {fc['median_ae_pp']:.2f} pp")
        print(
            f"  Within 5 pp: {fc['within_5pp']:.0%} | "
            f"within 10 pp: {fc['within_10pp']:.0%}"
        )
        print("  Worst series:")
        for w in fc["worst"]:
            print(f"    {w['organism']} / {w['antibiotic']}: {w['mae_pp']:.1f} pp")
    else:
        print("  Not enough multi-year data.")

    _hr("6. Coverage of India-relevant determinants")
    gd = evaluate_gene_detection()
    out["gene_coverage"] = gd
    print(f"  {gd['icmr_genes_covered']} of {gd['icmr_genes_listed']} genes "
          "reported by ICMR molecular surveillance are in our reference set.")
    for r in gd["rows"]:
        mark = "yes" if r["in_our_reference_set"] else "NO "
        print(
            f"    [{mark}] {r['gene']:<14} ICMR prevalence {r['icmr_prevalence']:.0%}"
            f"  demo strains carrying: {r['demo_strains_carrying']}"
        )
    print(f"\n  {gd['caveat']}")

    _hr("7. Feature 3 grid vs real ICMR national data")
    xc = evaluate_cross_check()
    out["cross_check"] = xc
    print(
        f"  {xc['n_compared']} organism x class pairings had an ICMR reference "
        f"({xc['reference_year']})."
    )
    print(f"    tracks national rate (within 15 pp) : {xc['n_aligned']}")
    print(f"    differs in the expected direction   : {xc['n_as_expected']}")
    print(f"    flagged for review                  : {xc['n_flagged']}")
    print(f"    no ICMR reference available         : {xc['n_no_reference']}")
    print(f"  Mean gap across aligned pairings: {xc['aligned_mae_pp']:.1f} pp")
    if xc["flagged"]:
        print()
        print("  Flagged:")
        for f in xc["flagged"]:
            print(
                f"    {f['organism']} x {f['antibiotic_class']}: "
                f"model {f['model_pct']:.0f}% vs ICMR {f['real_pct']:.0f}% "
                f"({f['delta']:+.0f} pp) — {f['status']}"
            )

    _hr("8. The deployed pipeline, end to end")
    dp = evaluate_deployed_pipeline()
    out["deployed_pipeline"] = dp
    if dp.get("available"):
        wo, wf = dp["without_mechanism_floor"], dp["with_mechanism_floor"]
        print(
            f"  Classifier alone across {dp['n_strains']} bundled strains "
            f"({wo['total_calls']} class calls):"
        )
        print(
            f"    very-major {wo['very_major']} | major {wo['major']} "
            f"| correct {wo['accuracy']:.1%}"
        )
        print("  Deployed pipeline (classifier + gene detection + mechanism floor):")
        print(
            f"    very-major {wf['very_major']} | major {wf['major']} "
            f"| correct {wf['accuracy']:.1%}"
        )
        print(
            f"\n  The gene-detection layer removes {dp['very_major_fixed']} "
            f"very-major error(s) and introduces\n  "
            f"{dp['major_introduced']} major error(s). It can only ever raise a "
            "probability, and it\n  fires only where a real reference sequence "
            "matches, so it cannot invent danger."
        )
        if wf["errors"]:
            print("\n  Remaining errors:")
            for e in wf["errors"]:
                print(
                    f"    [{e['type']}] {e['strain'][:40]} x "
                    f"{e['antibiotic_class']} @ {e['probability']:.0%}"
                )
        print(
            "\n  CAVEAT: the bundled strains are not a held-out test set — we "
            "chose their\n  genes. Treat this as an integration check, and the "
            "clade-held-out figure in\n  check 3 as the honest generalisation "
            "number."
        )
    else:
        print(f"  Unavailable: {dp.get('reason')}")

    out["generated_seconds"] = round(time.time() - started, 1)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)

    _hr("Done")
    print(f"Wrote {OUT_PATH} in {out['generated_seconds']}s.")
    print("These numbers surface in the app's Methods & Honesty tab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
