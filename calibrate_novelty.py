"""
Recalibrate the novelty detector's thresholds against REAL genomes.

WHY
---
`validate_novelty_real.py` showed the shipped thresholds (Z_FLOOR 8.0,
ABSOLUTE_FLOOR 0.18) flag something in 42 of 42 real genomes, at 24 segments
per megabase. On a 5 Mb genome that is roughly 120 flagged regions, which is
not a screening signal.

The thresholds were tuned on generated "clean" strains whose composition is
uniform by construction. Real genomes are not uniform — they carry prophages,
genomic islands and rRNA operons — so a detector calibrated on generated
sequence fires constantly on real sequence.

METHOD
------
The 42 genomes are split in half by a fixed seed.

    CALIBRATION HALF   used to choose the thresholds
    TEST HALF          never seen during the sweep; all reported numbers
                       come from here

Reporting the sweep's own best score would be circular — it would describe
how well we fitted the thresholds, not how well they work. The split is what
makes the reported number mean something.

Sensitivity is measured the same way as in the validation: a real 1200 bp
segment from the most GC-distant genome in the set, inserted at the midpoint.
Both donor and recipient are real sequence.

TARGET
------
Keep sensitivity as high as possible while bringing false alarms down to a
level where a flagged region is worth a human looking at it — about two
segments per megabase, so roughly ten flags on a 5 Mb genome.

There is no setting that is good at both. The sweep traces a hard trade-off:
100% sensitivity costs ~22 flags per megabase, and ~1 flag per megabase costs
more than half the sensitivity. That trade-off is the finding, and it is
reported rather than hidden behind whichever end of it looks better.

Run:  python calibrate_novelty.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from trialsense import novelty as nv

GENOMES = Path("data/real/genomes")
OUT = Path("models") / "novelty_calibration.json"
INSERT_LEN = 1200
SEED = 0

# Flags per megabase we are willing to hand a reviewer.
TARGET_SEG_PER_MB = 2.0

Z_GRID = [8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0]
ABS_GRID = [0.18, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def load(p: Path) -> str:
    return "".join(c for c in "".join(
        l.strip() for l in p.read_text().splitlines() if not l.startswith(">")
    ).upper() if c in "ACGT")


def gc(s: str) -> float:
    return (s.count("G") + s.count("C")) / len(s) if s else 0.0


def window_stats(seq: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the scan once and return (starts, z, excess) for every window.

    Running the expensive part once per genome and sweeping thresholds over
    the cached statistics is what makes a grid search affordable; re-running
    novelty_scan for every threshold pair would take hours.
    """
    return nv.scan_statistics(seq)


def segments_from(starts, z, excess, z_floor, abs_floor, step):
    """Count merged segments at a given threshold pair, and their spans."""
    flagged = [i for i in range(len(starts))
               if z[i] >= z_floor and excess[i] >= abs_floor]
    if not flagged:
        return []
    spans, run = [], [flagged[0]]
    for i in flagged[1:]:
        if starts[i] - starts[run[-1]] <= step:
            run.append(i)
        else:
            spans.append((starts[run[0]], starts[run[-1]] + nv.SCAN_WINDOW))
            run = [i]
    spans.append((starts[run[0]], starts[run[-1]] + nv.SCAN_WINDOW))
    return spans


def main() -> dict:
    paths = sorted(GENOMES.glob("*.fna"))
    genomes = {p.stem: load(p) for p in paths}
    genomes = {k: v for k, v in genomes.items() if len(v) > 20000}

    rng = random.Random(SEED)
    ids = sorted(genomes)
    rng.shuffle(ids)
    calib, test = ids[:len(ids) // 2], ids[len(ids) // 2:]
    print(f"{len(genomes)} genomes -> {len(calib)} calibration, {len(test)} test")

    gcs = {g: gc(s) for g, s in genomes.items()}

    def spiked_of(gid: str) -> tuple[str, int]:
        donor = max((d for d in ids if d != gid),
                    key=lambda d: abs(gcs[d] - gcs[gid]))
        dseq = genomes[donor]
        r = random.Random(SEED + hash(gid) % 1000)
        st = r.randrange(0, len(dseq) - INSERT_LEN)
        host = genomes[gid]
        at = len(host) // 2
        return host[:at] + dseq[st:st + INSERT_LEN] + host[at:], at

    # Cache window statistics once per genome, clean and spiked.
    print("computing window statistics (once per genome)...")
    clean_stats, spiked_stats = {}, {}
    for n, gid in enumerate(ids, 1):
        clean_stats[gid] = window_stats(genomes[gid])
        sp, at = spiked_of(gid)
        spiked_stats[gid] = (*window_stats(sp), at)
        print(f"  [{n}/{len(ids)}] {gid}")

    def evaluate(group, z_floor, abs_floor):
        total_seg = total_mb = 0.0
        detected = 0
        for gid in group:
            st, z, ex = clean_stats[gid]
            segs = segments_from(st, z, ex, z_floor, abs_floor, nv.SCAN_STEP)
            total_seg += len(segs)
            total_mb += len(genomes[gid]) / 1e6

            sst, sz, sex, at = spiked_stats[gid]
            ssegs = segments_from(sst, sz, sex, z_floor, abs_floor, nv.SCAN_STEP)
            if any(s < at + INSERT_LEN and e > at for s, e in ssegs):
                detected += 1
        return {
            "segments_per_mb": total_seg / total_mb if total_mb else 0.0,
            "sensitivity": detected / len(group) if group else 0.0,
        }

    print("\nsweeping thresholds on the calibration half...")
    rows = []
    for zf in Z_GRID:
        for af in ABS_GRID:
            m = evaluate(calib, zf, af)
            rows.append({"z_floor": zf, "abs_floor": af, **m})

    # Choose: the most sensitive setting that still produces a reviewable
    # number of flags. TARGET_SEG_PER_MB is the budget — on a 5 Mb genome it
    # allows about ten flagged regions, which a person can actually look at.
    # Demanding one per megabase instead costs roughly twenty points of
    # sensitivity for a difference nobody reviewing the output would notice.
    eligible = [r for r in rows if r["segments_per_mb"] <= TARGET_SEG_PER_MB]
    if eligible:
        best = max(eligible, key=lambda r: (r["sensitivity"], -r["segments_per_mb"]))
    else:
        best = min(rows, key=lambda r: r["segments_per_mb"])

    shipped = evaluate(test, 8.0, 0.18)
    chosen = evaluate(test, best["z_floor"], best["abs_floor"])

    print(f"\nchosen on calibration half: Z_FLOOR={best['z_floor']}, "
          f"ABSOLUTE_FLOOR={best['abs_floor']}")
    print(f"  calibration: {best['segments_per_mb']:.2f} seg/Mb, "
          f"sensitivity {best['sensitivity']:.0%}")
    print("\nHELD-OUT TEST HALF")
    print(f"  shipped (8.0 / 0.18) : {shipped['segments_per_mb']:6.2f} seg/Mb, "
          f"sensitivity {shipped['sensitivity']:.0%}")
    print(f"  chosen  ({best['z_floor']} / {best['abs_floor']}) : "
          f"{chosen['segments_per_mb']:6.2f} seg/Mb, "
          f"sensitivity {chosen['sensitivity']:.0%}")

    result = {
        "method": "half-split on 42 real genomes; thresholds chosen on the "
                  "calibration half, reported on the held-out test half",
        "n_calibration": len(calib), "n_test": len(test),
        "grid": rows,
        "chosen": {"z_floor": best["z_floor"], "abs_floor": best["abs_floor"]},
        "test_half": {"shipped": shipped, "chosen": chosen},
        "shipped_before": {"z_floor": 8.0, "abs_floor": 0.18},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwritten -> {OUT}")
    return result


if __name__ == "__main__":
    main()
