"""
TrialSense — Module 1 calibration against published interaction severities.

WHY THIS FILE EXISTS
--------------------
Module 1's severity scores come from our own mechanism knowledge base, not from
a licensed clinical database such as DrugBank or TWOSIDES. That raises an
obvious question: are our severities anywhere near the accepted clinical ones?

Without this check the answer is an assertion. With it, the answer is a number
that anyone can reproduce by running the file.

This is the ONLY external validation Module 1 has. Everything else measures the
model against labels the knowledge base itself produced, which tests internal
consistency rather than correctness.

HOW THE REFERENCE SEVERITIES WERE CHOSEN
----------------------------------------
Each pair below is a well-documented interaction between two drugs already in
our set, with a generally-agreed clinical classification mapped onto our scale:

    0 None    1 Mild    2 Moderate    3 Severe

Sources differ in vocabulary — "major", "contraindicated" and "avoid
combination" all map to 3 here. Where a pair sits genuinely on a boundary the
comment says so.

WHAT COUNTS AS AGREEMENT, AND WHY THREE MEASURES
------------------------------------------------
A single "match / no match" number would be misleading, because published
classifications themselves disagree at the boundaries. So we report three:

  EXACT        our severity equals the published one. The strictest measure.
  WITHIN ONE   off by at most one level. Reasonable given that published
               sources disagree with each other by about this much.
  DIRECTION    both agree on whether the pairing is clinically significant,
               i.e. Moderate or worse. This is what the tool is actually used
               for — deciding whether a pair needs expert review.

WHAT THIS CHECK FOUND
---------------------
On first run it scored Methotrexate + Trimethoprim as Mild against a published
Severe. That is an error in the dangerous direction, and the cause was that the
mechanism — additive folate blockade — was not modelled at all. A rule was added
to ddi.py and the pairing now scores correctly.

Finding that is the point of the file. A calibration check that agrees with
everything on its first run has not been tested.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ddi import SEVERITY_LEVELS, knowledge_base_assess
from .drugs import DRUGS

# A pairing is "clinically significant" at Moderate or above — the same
# threshold the ML layer in ddi.py is trained against.
SIGNIFICANT = 2


@dataclass(frozen=True)
class KnownPair:
    drug_a: str
    drug_b: str
    published: int  # 0-3 on our scale
    mechanism: str  # why it interacts, in one line
    note: str = ""  # boundary cases and disagreements between sources


# =============================================================================
# The reference set.
#
# Restricted to pairs where BOTH drugs are already in our knowledge base, and
# where the interaction is well enough established that a clinician would
# recognise it without looking it up.
# =============================================================================

KNOWN_PAIRS: list[KnownPair] = [
    # --- Severe: widely classified as avoid, contraindicate, or major ---------
    KnownPair("Simvastatin", "Clarithromycin", 3,
              "CYP3A4 inhibition of simvastatin's only clearance route",
              "A documented cause of rhabdomyolysis; co-prescription is advised against."),
    KnownPair("Simvastatin", "Ketoconazole", 3,
              "CYP3A4 inhibition, same route as above"),
    KnownPair("Warfarin", "Fluconazole", 3,
              "CYP2C9 inhibition raises INR sharply",
              "The textbook example of a narrow-window drug meeting a strong inhibitor."),
    KnownPair("Linezolid", "Fluoxetine", 3,
              "Reversible MAO inhibition plus SSRI, risking serotonin syndrome"),
    KnownPair("Linezolid", "Sertraline", 3,
              "Same mechanism as above"),
    KnownPair("Midazolam", "Ketoconazole", 3,
              "CYP3A4 is midazolam's sole route; exposure rises many-fold",
              "Midazolam is the regulatory probe substrate for CYP3A4 activity."),
    KnownPair("Theophylline", "Ciprofloxacin", 3,
              "CYP1A2 inhibition pushes a narrow-window drug into toxicity"),
    KnownPair("Methotrexate", "Trimethoprim", 3,
              "Additive folate-pathway blockade causing marrow suppression",
              "This pair exposed a missing mechanism on first run — see module docstring."),
    KnownPair("Warfarin", "Rifampicin", 3,
              "Enzyme induction clears warfarin faster, losing anticoagulation",
              "Failure of efficacy rather than toxicity, which is easy to miss clinically."),
    KnownPair("Digoxin", "Amiodarone", 3,
              "P-glycoprotein inhibition raises digoxin, which has a narrow window"),
    KnownPair("Warfarin", "Aspirin", 3,
              "Additive bleeding risk with a narrow-window anticoagulant"),
    KnownPair("Carbamazepine", "Clarithromycin", 3,
              "CYP3A4 inhibition of a narrow-window antiepileptic"),
    KnownPair("Phenytoin", "Fluconazole", 3,
              "CYP2C9 inhibition of a narrow-window antiepileptic"),
    KnownPair("Methotrexate", "Ibuprofen", 3,
              "NSAID reduces renal clearance of a narrow-window cytotoxic"),

    # --- Moderate: real, manageable with monitoring or dose change ------------
    KnownPair("Warfarin", "Metronidazole", 2,
              "CYP2C9 inhibition raising INR",
              "Sources split between moderate and major; weaker than fluconazole."),
    KnownPair("Lisinopril", "Spironolactone", 2,
              "Additive potassium retention"),
    KnownPair("Amiodarone", "Clarithromycin", 2,
              "Additive QT-interval prolongation",
              "Severity depends heavily on baseline electrolytes and ECG."),
    KnownPair("Digoxin", "Verapamil", 2,
              "P-glycoprotein inhibition raising digoxin levels"),
    KnownPair("Atorvastatin", "Clarithromycin", 2,
              "CYP3A4 inhibition, but atorvastatin depends on it less than simvastatin"),
    KnownPair("Clopidogrel", "Omeprazole", 2,
              "CYP2C19 inhibition prevents activation of a prodrug",
              "Reduced efficacy rather than toxicity."),
    KnownPair("Tramadol", "Fluoxetine", 2,
              "CYP2D6 inhibition blocks activation, plus additive serotonergic effect"),
    KnownPair("Simvastatin", "Diltiazem", 2,
              "Moderate CYP3A4 inhibition raising myopathy risk"),
    KnownPair("Lisinopril", "Trimethoprim", 2,
              "Additive potassium retention",
              "Trimethoprim blocks the renal sodium channel, a frequently missed cause."),

    # --- None: pairs that must NOT be flagged ---------------------------------
    KnownPair("Amoxicillin", "Metformin", 0,
              "Separate clearance routes, no shared toxicity target"),
    KnownPair("Azithromycin", "Simvastatin", 0,
              "Azithromycin barely inhibits CYP3A4, unlike clarithromycin",
              "A deliberate near-miss: two macrolides with different profiles."),
    KnownPair("Levofloxacin", "Theophylline", 0,
              "Levofloxacin does not meaningfully inhibit CYP1A2",
              "The counterpart to the ciprofloxacin pair above."),
    KnownPair("Trimethoprim", "Sulfamethoxazole", 0,
              "Co-formulated and given together deliberately",
              "Both block the folate pathway, so this guards against the "
              "antifolate rule firing where the combination is intended."),
]


def evaluate() -> dict:
    """Score the knowledge base against the published severities."""
    rows = []
    for kp in KNOWN_PAIRS:
        a, b = DRUGS.get(kp.drug_a), DRUGS.get(kp.drug_b)
        if a is None or b is None:
            continue
        ours, score, _ = knowledge_base_assess(a, b)
        rows.append(
            {
                "pair": f"{kp.drug_a} + {kp.drug_b}",
                "mechanism": kp.mechanism,
                "note": kp.note,
                "published": kp.published,
                "ours": ours,
                "raw_score": round(score, 2),
                "exact": ours == kp.published,
                "within_one": abs(ours - kp.published) <= 1,
                "direction": (ours >= SIGNIFICANT) == (kp.published >= SIGNIFICANT),
            }
        )

    n = len(rows) or 1
    return {
        "n_pairs": len(rows),
        "exact": sum(r["exact"] for r in rows),
        "within_one": sum(r["within_one"] for r in rows),
        "direction": sum(r["direction"] for r in rows),
        "exact_rate": sum(r["exact"] for r in rows) / n,
        "within_one_rate": sum(r["within_one"] for r in rows) / n,
        "direction_rate": sum(r["direction"] for r in rows) / n,
        "disagreements": [r for r in rows if not r["exact"]],
        "rows": rows,
    }


def report() -> None:
    """Print the calibration table. Run via `python -m trialsense.known_pairs`."""
    res = evaluate()
    print("=" * 78)
    print("Module 1 — calibration against published interaction severities")
    print("=" * 78)
    print()
    print("%-34s %-9s %-9s %s" % ("PAIR", "PUBLISHED", "OURS", ""))
    print("-" * 78)
    for r in res["rows"]:
        mark = "exact" if r["exact"] else ("within 1" if r["within_one"] else "MISS")
        print("%-34s %-9s %-9s %s"
              % (r["pair"][:34], SEVERITY_LEVELS[r["published"]],
                 SEVERITY_LEVELS[r["ours"]], mark))
    print("-" * 78)
    print("Exact severity match   : %2d/%d  (%.0f%%)"
          % (res["exact"], res["n_pairs"], 100 * res["exact_rate"]))
    print("Within one level       : %2d/%d  (%.0f%%)"
          % (res["within_one"], res["n_pairs"], 100 * res["within_one_rate"]))
    print("Significant / not      : %2d/%d  (%.0f%%)"
          % (res["direction"], res["n_pairs"], 100 * res["direction_rate"]))

    if res["disagreements"]:
        print()
        print("Disagreements:")
        for r in res["disagreements"]:
            direction = "over-call" if r["ours"] > r["published"] else "UNDER-CALL"
            print("   %-32s published %-9s ours %-9s  %s"
                  % (r["pair"], SEVERITY_LEVELS[r["published"]],
                     SEVERITY_LEVELS[r["ours"]], direction))
        print()
        print("   Over-calls are the safe direction for a screening tool — they")
        print("   cost an unnecessary review. Under-calls are the dangerous ones.")


if __name__ == "__main__":
    report()
