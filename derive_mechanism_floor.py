"""
Derive the mechanism floor from laboratory evidence instead of asserting it.

THE PROBLEM
-----------
`apply_mechanism_floor` raised a class to 0.90 whenever a gene conferring that
class was detected. 0.90 was never measured; it encoded the belief that gene
presence almost always means resistance. The ICMR cross-check showed what that
costs — an overall mean absolute error of 41.2 percentage points against 5.1
for the rows that lined up — and the audit flagged it as systematic over-calling.

WHAT THE LABORATORY DATA SAYS
-----------------------------
With 42 real genomes and 350 bench susceptibility results we can measure the
quantity the floor is guessing at:

    P(laboratory says resistant | we detected a gene conferring that class)

Measured overall, that is 0.66, not 0.90. Per class it ranges from 0.18 for
fluoroquinolones — where detection fires on the gyrA locus, which every E.
coli carries — to 1.00 for several classes with only three observations.

SHRINKAGE, AND WHY
------------------
Three-for-three is not evidence that the true rate is 1.00. Each class is
shrunk toward the overall rate with PSEUDO_COUNT pseudo-observations:

    floor = (resistant + PSEUDO_COUNT * overall) / (n + PSEUDO_COUNT)

so a class with plenty of data keeps its own estimate and a class with three
observations stays near the pooled one. This is a standard Bayesian posterior
mean under a Beta prior centred on the pooled rate.

LIMITATIONS OF THIS ESTIMATE, WHICH ARE REAL
--------------------------------------------
  * Between 3 and 16 observations per class. These are indicative, not precise.
  * The isolates were selected for breadth of bench testing, so they
    over-represent strains somebody had reason to test. The rate in an
    unselected population would differ.
  * Class-level aggregation: a class counts as resistant if any drug in it
    failed, so the floor inherits that coarseness.

It is still a measurement rather than an assertion, which is the improvement.

Run:  python derive_mechanism_floor.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from trialsense.amr import (ANTIBIOTIC_CLASSES, POINT_MUTATION_GENES,
                            RESISTANCE_GENES)

REAL = Path("models") / "validation_real.json"
OUT = Path("models") / "mechanism_floor.json"

# Strength of the prior, in observations. Five keeps a 3-observation class
# close to the pooled rate while letting a 16-observation class speak mostly
# for itself.
PSEUDO_COUNT = 5.0


def main() -> dict:
    if not REAL.exists():
        raise SystemExit("Run validate_real.py first — this needs laboratory labels.")

    data = json.loads(REAL.read_text())
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [resistant, n]

    for iso in data["per_isolate"]:
        # Point-mutation determinants are EXCLUDED from this estimate.
        # Detecting gyrA means the locus is present, which every E. coli
        # satisfies, so counting those detections drags the fluoroquinolone
        # estimate down to a number that describes locus presence rather than
        # resistance evidence. The floor for those is governed by the codon
        # read instead — see POINT_MUTATION_FLOOR in amr.py.
        conferred = set()
        for g in iso["genes"]:
            if g in POINT_MUTATION_GENES:
                continue
            gene = RESISTANCE_GENES.get(g)
            if gene:
                conferred.update(gene.confers_resistance_to)
        for cls, d in iso["classes"].items():
            if cls in conferred:
                counts[cls][1] += 1
                counts[cls][0] += d["lab"]

    total_r = sum(v[0] for v in counts.values())
    total_n = sum(v[1] for v in counts.values())
    overall = total_r / total_n if total_n else 0.0

    floors: dict[str, float] = {}
    rows = []
    for cls in ANTIBIOTIC_CLASSES:
        r, n = counts.get(cls, [0, 0])
        if n == 0:
            rows.append({"class": cls, "resistant": 0, "n": 0,
                         "raw": None, "floor": None, "basis": "no observations"})
            continue
        floor = (r + PSEUDO_COUNT * overall) / (n + PSEUDO_COUNT)
        floors[cls] = round(float(floor), 3)
        rows.append({"class": cls, "resistant": r, "n": n,
                     "raw": round(r / n, 3), "floor": round(float(floor), 3),
                     "basis": "laboratory AST on real genomes"})

    result = {
        "source": "42 real genomes with laboratory AST (models/validation_real.json)",
        "quantity": "P(laboratory resistant | a gene conferring that class detected)",
        "overall": round(overall, 3),
        "pseudo_count": PSEUDO_COUNT,
        "previous_constant": 0.90,
        "floors": floors,
        "rows": rows,
        "caveats": [
            "3-16 observations per class; indicative, not precise",
            "isolates selected for breadth of bench testing, so not a random sample",
            "class-level aggregation: resistant if any drug in the class failed",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2))

    print("P(laboratory resistant | gene conferring that class detected)")
    print(f"  overall {overall:.3f}   (the previous constant asserted 0.900)\n")
    print("  %-27s %4s %4s %7s %7s" % ("CLASS", "R", "n", "raw", "floor"))
    for row in rows:
        if row["n"] == 0:
            print("  %-27s %4s %4s %7s %7s" % (row["class"], "-", 0, "-", "-"))
        else:
            print("  %-27s %4d %4d %7.2f %7.2f"
                  % (row["class"], row["resistant"], row["n"],
                     row["raw"], row["floor"]))
    print(f"\nwritten -> {OUT}")
    return result


if __name__ == "__main__":
    main()
