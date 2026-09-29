"""
Module 3 Feature 4 (novelty scan) — validation on REAL genomes.

WHY THIS EXISTS
---------------
The novelty detector's thresholds were tuned on 19 cases — 12 clean strains
and 7 spiked ones — all produced by the same generator that produced its
training data. A detector tuned and tested on one generator has been shown to
separate that generator's two modes, which is a weaker claim than it sounds.

This file re-measures it on real bacterial genomes.

THE TWO QUESTIONS

  FALSE ALARMS   Run on 42 real, unmodified genomes. Every segment reported is
                 a false alarm for our purposes. Note that real genomes truly
                 do contain compositionally atypical DNA — prophages, genomic
                 islands, rRNA operons — which the generated "clean" strains
                 never did. That is the point: the question is whether the
                 detector's idea of "foreign" survives contact with real
                 sequence, not whether real sequence is uniform.

  SENSITIVITY    Insert a real segment taken from a compositionally distant
                 organism into a real recipient genome, and ask whether the
                 scan finds it. Donor and recipient are both real, and the
                 pairing maximises GC divergence, which is what an acquired
                 element from a distant source actually looks like.

                 The insert is REAL sequence from another organism, not a
                 generated one, so this does not repeat the original mistake
                 of testing a generator against itself.

Run:  python validate_novelty_real.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from trialsense import novelty as nv

GENOMES = Path("data/real/genomes")
LABELS = Path("data/real/lab_ast_labels.json")
OUT = Path("models") / "validation_novelty_real.json"

INSERT_LEN = 1200
SEED = 0


def load(path: Path) -> str:
    return "".join(l.strip() for l in path.read_text().splitlines()
                   if not l.startswith(">"))


def gc(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def main() -> dict:
    if not GENOMES.exists():
        raise SystemExit("No real genomes. Run `python -m trialsense.ingest` first.")

    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    paths = sorted(GENOMES.glob("*.fna"))
    rng = random.Random(SEED)

    print("=" * 78)
    print("Feature 4 (novelty) — validation on REAL genomes")
    print("=" * 78)

    genomes: dict[str, str] = {}
    for p in paths:
        seq = "".join(c for c in load(p).upper() if c in "ACGT")
        if len(seq) > 20000:
            genomes[p.stem] = seq
    print(f"\nloaded {len(genomes)} real genomes")

    gcs = {g: gc(s) for g, s in genomes.items()}

    # ---- Question 1: false alarms on unmodified real genomes ---------------
    print("\n[1] False alarms on unmodified real genomes")
    clean_rows = []
    t0 = time.time()
    for gid, seq in genomes.items():
        segs, diag = nv.novelty_scan(seq)
        org = labels.get(gid, {}).get("organism", "")
        clean_rows.append({
            "genome_id": gid, "organism": org, "length": len(seq),
            "n_windows": diag.get("n_windows", 0),
            "n_segments": len(segs),
            "peak": round(max((s.excess_divergence for s in segs), default=0.0), 4),
        })
        print(f"  {org[:24]:24s} {gid:14s} {len(seq):>9,} bp  "
              f"segments={len(segs):>2}  peak={clean_rows[-1]['peak']:.3f}")

    n_flagged = sum(1 for r in clean_rows if r["n_segments"] > 0)
    total_segments = sum(r["n_segments"] for r in clean_rows)

    # ---- Question 2: sensitivity to a real foreign insert ------------------
    print("\n[2] Sensitivity to a real segment from a compositionally distant organism")
    ids = list(genomes)
    spiked_rows = []
    for gid in ids:
        # Donor = the genome with the most different GC content.
        donor = max((d for d in ids if d != gid),
                    key=lambda d: abs(gcs[d] - gcs[gid]))
        dseq = genomes[donor]
        dstart = rng.randrange(0, len(dseq) - INSERT_LEN)
        insert = dseq[dstart:dstart + INSERT_LEN]

        host = genomes[gid]
        at = len(host) // 2
        spiked = host[:at] + insert + host[at:]

        segs, _ = nv.novelty_scan(spiked)
        hit = any(s.start < at + INSERT_LEN and s.end > at for s in segs)
        spiked_rows.append({
            "genome_id": gid,
            "donor": donor,
            "gc_host": round(gcs[gid], 3),
            "gc_donor": round(gcs[donor], 3),
            "gc_gap": round(abs(gcs[donor] - gcs[gid]), 3),
            "detected": hit,
            "n_segments": len(segs),
        })
        print(f"  {gid:14s} <- {donor:14s} "
              f"GC {gcs[gid]:.2f}->{gcs[donor]:.2f} "
              f"(gap {abs(gcs[donor]-gcs[gid]):.3f})  "
              f"{'DETECTED' if hit else 'missed'}")

    detected = sum(1 for r in spiked_rows if r["detected"])
    elapsed = time.time() - t0

    result = {
        "data": "real genome assemblies (BV-BRC), unmodified and spiked",
        "insert": {
            "length_bp": INSERT_LEN,
            "source": "a real segment from the most GC-distant genome in the set",
            "note": "both donor and recipient are real sequence; nothing is generated",
        },
        "false_alarms": {
            "n_genomes": len(clean_rows),
            "n_genomes_flagged": n_flagged,
            "genome_flag_rate": n_flagged / len(clean_rows) if clean_rows else 0.0,
            "total_segments": total_segments,
            "segments_per_megabase": (
                total_segments / (sum(r["length"] for r in clean_rows) / 1e6)
                if clean_rows else 0.0),
            "rows": clean_rows,
        },
        "sensitivity": {
            "n_spiked": len(spiked_rows),
            "n_detected": detected,
            "rate": detected / len(spiked_rows) if spiked_rows else 0.0,
            "rows": spiked_rows,
        },
        "prior_claim": {
            "source": "models/validation.json -> novelty",
            "clean_strains_assessed": 12,
            "false_positive_rate": 0.0,
            "spiked_cases": 7,
            "sensitivity": 1.0,
            "caveat": "all 19 cases came from the same generator as the "
                      "training data; this file replaces that evidence",
        },
        "runtime_seconds": round(elapsed, 1),
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    fa = result["false_alarms"]
    se = result["sensitivity"]
    print("\n" + "=" * 78)
    print(f"FALSE ALARMS   {fa['n_genomes_flagged']}/{fa['n_genomes']} real genomes "
          f"flagged something ({fa['genome_flag_rate']:.0%})")
    print(f"               {fa['total_segments']} segments total, "
          f"{fa['segments_per_megabase']:.2f} per megabase")
    print(f"SENSITIVITY    {se['n_detected']}/{se['n_spiked']} real foreign "
          f"inserts detected ({se['rate']:.0%})")
    print(f"\nPrior claim was 0% false positives and 100% sensitivity on 19 "
          f"same-generator cases.")
    print(f"\nwritten -> {OUT}   ({elapsed:.0f}s)")
    return result


if __name__ == "__main__":
    main()
