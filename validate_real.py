"""
TrialSense Module 3 — validation against REAL genomes and LABORATORY AST.

WHAT MAKES THIS DIFFERENT FROM validate_module3.py
--------------------------------------------------
`validate_module3.py` measures the model on isolates that we constructed:
generated background sequence with real resistance genes planted into it. Its
numbers answer "can the pipeline recover gene content we inserted?"

This file answers a harder and more useful question: "does the pipeline predict
what the laboratory actually measured, on bacteria nobody constructed?"

    genomes   real assemblies from BV-BRC, 2.5-7 Mb each
    labels    bench susceptibility results — broth dilution and disk
              diffusion, read against CLSI or EUCAST breakpoints
    filter    rows whose `evidence` column said "Computational Method" were
              excluded at ingestion, so we are never scoring our predictions
              against somebody else's predictions

EXPECT THE NUMBERS TO BE WORSE, AND UNDERSTAND WHY
--------------------------------------------------
The model was trained on ~900 bp cassettes planted in ~2.6 kb of generated
background. A real genome is a thousand times longer and its k-mer profile is
dominated by ordinary housekeeping sequence rather than by the resistance
determinant. This is an out-of-distribution test, and the gap between these
numbers and the constructed-isolate numbers measures how much the constructed
set was flattering us.

Both sets of numbers are kept. The constructed ones are not deleted, because
the difference between them is the finding.

HOW TO READ THE ERROR TYPES
---------------------------
    very major error  predicted susceptible, laboratory says resistant.
                      The drug fails in patients. This is the costly one.
    major error       predicted resistant, laboratory says susceptible.
                      One confirmatory assay recovers it.

Run:  python validate_real.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from trialsense import amr as amr_mod

DATA = Path("data/real")
GENOMES = DATA / "genomes"
LABELS = DATA / "lab_ast_labels.json"
OUT = Path("models") / "validation_real.json"


def load_sequence(path: Path) -> str:
    return "".join(l.strip() for l in path.read_text().splitlines()
                   if not l.startswith(">"))


def main() -> dict:
    if not LABELS.exists():
        raise SystemExit(
            "No laboratory labels found. Run `python -m trialsense.ingest` first.\n"
            "This validation does not run without bench labels, and no labels "
            "are generated to substitute for them.")

    labels = json.loads(LABELS.read_text())
    import pickle
    model_path = Path("models") / "amr_model.pkl"
    if not model_path.exists():
        raise SystemExit("No trained model at models/amr_model.pkl — run train.py first.")
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    print("=" * 78)
    print("Module 3 — validation on REAL genomes with LABORATORY AST labels")
    print("=" * 78)
    print(f"\nisolates: {len(labels)}")

    screening = amr_mod.RESISTANCE_CALL_THRESHOLD
    confirmed = amr_mod.RESISTANCE_CONFIRMED_THRESHOLD

    # counters[threshold][class] = dict of tp/fp/tn/fn
    counters: dict[float, dict[str, dict[str, int]]] = {
        t: defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        for t in (screening, confirmed)
    }
    gene_hits: dict[str, int] = defaultdict(int)
    allele_states: dict[str, int] = defaultdict(int)
    per_isolate: list[dict] = []

    t0 = time.time()
    for n, (gid, rec) in enumerate(sorted(labels.items()), 1):
        path = GENOMES / f"{gid}.fna"
        if not path.exists():
            continue
        seq = load_sequence(path)
        report = amr_mod.analyze_isolate(seq, model, strain_name=rec["name"])

        for gene, _score in report.detected_genes:
            gene_hits[gene] += 1
            if gene in amr_mod.POINT_MUTATION_GENES:
                state, _ = amr_mod.read_allele(seq, gene)
                allele_states[f"{gene}:{state}"] += 1

        row = {"genome_id": gid, "organism": rec["organism"],
               "genes": [g for g, _ in report.detected_genes], "classes": {}}
        for cls, truth in rec["class_labels"].items():
            prob = report.probabilities.get(cls)
            if prob is None:
                continue
            row["classes"][cls] = {"lab": truth, "p": round(prob, 3)}
            for t in (screening, confirmed):
                pred = 1 if prob >= t else 0
                c = counters[t][cls]
                if pred and truth:
                    c["tp"] += 1
                elif pred and not truth:
                    c["fp"] += 1
                elif not pred and truth:
                    c["fn"] += 1
                else:
                    c["tn"] += 1
        per_isolate.append(row)
        print(f"  [{n:>2}/{len(labels)}] {rec['organism'][:24]:24s} {gid:14s} "
              f"{len(seq):>9,} bp  genes={len(report.detected_genes)}")

    elapsed = time.time() - t0

    def summarise(t: float) -> dict:
        tp = sum(c["tp"] for c in counters[t].values())
        fp = sum(c["fp"] for c in counters[t].values())
        tn = sum(c["tn"] for c in counters[t].values())
        fn = sum(c["fn"] for c in counters[t].values())
        total = tp + fp + tn + fn
        n_res, n_sus = tp + fn, tn + fp
        return {
            "threshold": t,
            "n_comparisons": total,
            "n_resistant": n_res,
            "n_susceptible": n_sus,
            "accuracy": (tp + tn) / total if total else 0.0,
            "very_major_error_rate": fn / n_res if n_res else None,
            "major_error_rate": fp / n_sus if n_sus else None,
            "sensitivity": tp / n_res if n_res else None,
            "specificity": tn / n_sus if n_sus else None,
            "by_class": {
                cls: {**c,
                      "vme": (c["fn"] / (c["tp"] + c["fn"]))
                             if (c["tp"] + c["fn"]) else None,
                      "me": (c["fp"] / (c["tn"] + c["fp"]))
                            if (c["tn"] + c["fp"]) else None}
                for cls, c in sorted(counters[t].items())
            },
        }

    result = {
        "data": {
            "classification": "real genomes, laboratory-confirmed AST labels",
            "source": "BV-BRC (evidence = 'Laboratory Method' only)",
            "n_isolates": len(per_isolate),
            "caveat": "isolates were selected for breadth of bench testing, so "
                      "they over-represent strains somebody had reason to test; "
                      "these rates describe this set, not a patient population",
        },
        "screening": summarise(screening),
        "confirmed": summarise(confirmed),
        "gene_detection": dict(sorted(gene_hits.items(), key=lambda kv: -kv[1])),
        "allele_states": dict(allele_states),
        "runtime_seconds": round(elapsed, 1),
        "per_isolate": per_isolate,
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 78)
    for key in ("screening", "confirmed"):
        s = result[key]
        print(f"\n{key.upper()} threshold {s['threshold']:.2f}  "
              f"({s['n_comparisons']} comparisons: "
              f"{s['n_resistant']} resistant, {s['n_susceptible']} susceptible)")
        print(f"  accuracy          {s['accuracy']:.3f}")
        if s["very_major_error_rate"] is not None:
            print(f"  very major errors {s['very_major_error_rate']:.1%}  "
                  "(predicted susceptible, laboratory resistant)")
        if s["major_error_rate"] is not None:
            print(f"  major errors      {s['major_error_rate']:.1%}  "
                  "(predicted resistant, laboratory susceptible)")

    print("\nGenes detected across real genomes:")
    for g, n in list(result["gene_detection"].items())[:15]:
        print(f"  {g:16s} {n:>3} isolates")
    if result["allele_states"]:
        print("\nPoint-mutation allele reads:")
        for k, v in sorted(result["allele_states"].items()):
            print(f"  {k:34s} {v}")

    print(f"\nwritten -> {OUT}   ({elapsed:.0f}s)")
    return result


if __name__ == "__main__":
    main()
