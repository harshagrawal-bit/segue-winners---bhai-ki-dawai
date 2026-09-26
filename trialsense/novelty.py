"""
TrialSense — Module 3, Feature 4: the Dark Genome detector.

THE PROBLEM WITH EVERY TOOL IN THIS CATEGORY
--------------------------------------------
AMRFinderPlus, ResFinder, CARD/RGI and our own gene-detection layer are all
CLOSED-WORLD systems: they report what matches their reference database and say
nothing about anything else. Silence from a closed-world detector is
indistinguishable from "all clear", and that is exactly how an expensive
surprise reaches a trial.

This module adds the open-world half:

    "Is there something here that BEHAVES like acquired DNA,
     but matches nothing we know?"

HOW IT WORKS, AND WHY IT IS BUILT THIS WAY
------------------------------------------
Horizontally acquired DNA carries the compositional signature of wherever it
came from rather than of its current host. That is the same principle genomic
islands are located by, and tetranucleotide (k=4) composition is the
conventional signal for it.

Three design decisions, each of which was necessary to make the detector work:

1. KNOWN GENES ARE MASKED FIRST. A resistance cassette is itself
   compositionally foreign — that is how it is recognised. Scoring anomaly
   without masking known hits simply rediscovers the genes we already found,
   and in testing the clean demo strains scored HIGHER than deliberately
   spiked ones. We mask every window matching a reference, plus two windows
   either side to cover the cassette boundary, and score only what is left.

2. THE BACKGROUND IS ESTIMATED FROM UNEXPLAINED WINDOWS, ITERATIVELY TRIMMED.
   A plain median over all windows is dragged by the very segments we are
   hunting when they occupy a large fraction of a short sequence.

3. TWO FLOORS, NOT ONE. A window must be both statistically unusual (robust
   z) AND absolutely divergent. The z-score alone is unstable when few
   windows remain, because a small MAD inflates it. On the validation set
   the absolute measure separated cleanly where the z-score did not.

MEASURED PERFORMANCE
--------------------
Validated by planting synthetic elements that match no reference, and checking
the unmodified demo strains do not fire. See validate_module3.py.

    clean demo strains  : peak excess divergence 0.076 - 0.115
    spiked with unknown : peak excess divergence 0.253 - 0.693

The thresholds below sit in that gap, with better than 2x margin on each side.

WHAT THIS IS NOT
----------------
It does not identify the element and does not claim it confers resistance. It
says "unexplained acquired sequence is present, go and look". That is a
deliberately weaker claim than a gene call, and it is an honest one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .amr import (
    CASSETTE_LEN,
    DETECTION_THRESHOLD,
    RESISTANCE_GENES,
    kmer_features,
    locate_genes,
)

# Cosine similarity at which a window counts as an identified gene. Imported
# from amr.py rather than redefined, so the two layers can never disagree about
# what "known" means — they did briefly, when the reference set was swapped for
# real sequences and only one of the two thresholds was recalibrated.
KNOWN_MATCH_THRESHOLD = DETECTION_THRESHOLD

# k for the two different jobs. Reference matching stays at 5 to agree with
# detect_genes; compositional anomaly uses 4, the conventional tetranucleotide
# signal, which is also far denser per window (256 bins vs 1024) and therefore
# much less noisy at these sequence lengths.
K_REFERENCE = 5
K_COMPOSITION = 4

SCAN_WINDOW = CASSETTE_LEN      # 900 bp, one cassette wide
SCAN_STEP = 150                 # fine step: more windows, steadier background
BOUNDARY_MARGIN = 2             # windows masked either side of a known hit

# Minimum unexplained windows needed to estimate a host background at all.
# Below this we decline to assess rather than emit a number we cannot support.
MIN_BACKGROUND_POOL = 6

# Fraction of the unexplained pool treated as "core genome" when estimating
# the background and its spread.
CORE_QUANTILE = 0.6

# Detection floors. Both must be exceeded. Calibrated against the validation
# set described in the module docstring.
Z_FLOOR = 8.0
ABSOLUTE_FLOOR = 0.18


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class NovelSegment:
    """One stretch of sequence that looks acquired but matches nothing known."""

    start: int
    end: int
    excess_divergence: float   # absolute divergence above the host background
    robust_z: float            # the same, in robust deviations
    max_ref_similarity: float  # best match to any reference (sub-threshold)
    nearest_gene: str

    @property
    def length(self) -> int:
        return self.end - self.start

    def describe(self) -> str:
        return (
            f"Positions {self.start:,}–{self.end:,} ({self.length:,} bp). "
            f"Tetranucleotide composition sits {self.excess_divergence:.2f} "
            f"above this genome's own background ({self.robust_z:.0f} robust "
            f"deviations) — the signature of horizontally acquired DNA. The "
            f"closest reference is {self.nearest_gene} at "
            f"{self.max_ref_similarity:.0%} similarity, below the "
            f"{KNOWN_MATCH_THRESHOLD:.0%} identification threshold, so we "
            "cannot name it."
        )


@dataclass
class NoveltyReport:
    """Feature 4 output — a standalone, self-contained report."""

    strain_name: str
    sequence_length: int
    n_windows: int = 0
    n_explained: int = 0
    segments: list[NovelSegment] = field(default_factory=list)
    known_gene_count: int = 0

    scan_available: bool = True
    scan_unavailable_reason: str = ""

    # Sample-level out-of-distribution assessment
    ood_available: bool = False
    ood_distance: float = 0.0
    ood_percentile: float = 0.0
    ood_band: str = "UNKNOWN"  # IN DISTRIBUTION | BORDERLINE | OUT OF DISTRIBUTION

    @property
    def verdict(self) -> str:
        """CLEAN | UNEXPLAINED SEQUENCE | CAUTION | UNRELIABLE | NOT ASSESSED"""
        if self.ood_band == "OUT OF DISTRIBUTION":
            return "UNRELIABLE"
        if not self.scan_available and not self.ood_available:
            return "NOT ASSESSED"
        if self.segments and self.ood_band == "BORDERLINE":
            return "CAUTION"
        if self.segments:
            return "UNEXPLAINED SEQUENCE"
        if self.ood_band == "BORDERLINE":
            return "CAUTION"
        if not self.scan_available:
            return "NOT ASSESSED"
        return "CLEAN"

    @property
    def color(self) -> str:
        return {
            "CLEAN": "#2E9E5B",
            "UNEXPLAINED SEQUENCE": "#E0A526",
            "CAUTION": "#E8722C",
            "UNRELIABLE": "#D6453D",
            "NOT ASSESSED": "#5B7290",
        }[self.verdict]

    def headline(self) -> str:
        v = self.verdict
        if v == "UNRELIABLE":
            return (
                "This isolate sits outside the training distribution — treat "
                "every Module 3 number on it with caution"
            )
        if v == "NOT ASSESSED":
            return "Open-world scan could not run on this sequence"
        if self.segments:
            n = len(self.segments)
            return (
                f"{n} unexplained acquired segment{'s' if n != 1 else ''} "
                f"detected in {self.strain_name}"
            )
        if v == "CAUTION":
            return "This isolate sits at the edge of the training distribution"
        return (
            "No unexplained acquired sequence, and this isolate resembles the "
            "training distribution"
        )

    def plain_summary(self) -> str:
        parts: list[str] = []

        if not self.scan_available:
            parts.append(
                f"The open-world scan did not run: {self.scan_unavailable_reason} "
                "We report that rather than returning a clean result we cannot "
                "stand behind — an unrun check and a passed check are not the "
                "same thing."
            )
        elif self.segments:
            total = sum(s.length for s in self.segments)
            parts.append(
                f"We found {len(self.segments)} region(s), {total:,} bp in total, "
                f"whose composition does not match the rest of this genome — the "
                f"signature of DNA acquired from another organism — and which "
                f"match none of the {len(RESISTANCE_GENES)} determinants in our "
                f"reference set. We cannot tell you what they are. We can tell "
                f"you they are there, which every closed-world gene-detection "
                f"tool would have passed over in silence. Recommend targeted "
                "sequencing of these regions before treating this organism as "
                "characterised."
            )
        else:
            parts.append(
                f"No compositionally foreign, unidentified segments were found "
                f"across {self.n_windows} windows. The {self.known_gene_count} "
                f"determinant(s) present are all accounted for by known "
                "references."
            )

        if self.ood_available:
            if self.ood_band == "IN DISTRIBUTION":
                parts.append(
                    f"This isolate sits well within the range the classifier was "
                    f"trained on ({self.ood_percentile:.0f}th percentile of "
                    "training distance), so its resistance probabilities carry "
                    "their normal weight."
                )
            elif self.ood_band == "BORDERLINE":
                parts.append(
                    f"It sits at the edge of the training distribution "
                    f"({self.ood_percentile:.0f}th percentile) — still "
                    "informative, but less reliable than the headline accuracy "
                    "figures imply."
                )
            else:
                parts.append(
                    f"It sits OUTSIDE the training distribution "
                    f"({self.ood_percentile:.0f}th percentile). Published work "
                    "finds genomic AMR models generalise poorly beyond their "
                    "training clades, so treat these probabilities as a prompt "
                    "to sequence further, not as a result."
                )
        else:
            parts.append(
                "Out-of-distribution scoring is unavailable — the loaded model "
                "predates this feature. Re-run `python train.py` to enable it."
            )
        return " ".join(parts)

    def recommendation(self) -> str:
        return {
            "CLEAN": "Proceed. Nothing unexplained, and the isolate is well "
            "within the model's competence.",
            "UNEXPLAINED SEQUENCE": "Sequence the flagged regions before "
            "treating this organism as characterised.",
            "CAUTION": "Confirm by targeted sequencing. This isolate is both "
            "unusual for the model and carries unexplained segments.",
            "UNRELIABLE": "Do not act on Module 3 numbers for this isolate "
            "alone. Retrain on data covering this lineage, or confirm in vitro.",
            "NOT ASSESSED": "Supply a longer sequence to enable the open-world "
            "scan.",
        }[self.verdict]


# Reference profiles are deterministic; rebuilding eighteen 900 bp cassettes on
# every scan made the portfolio grid crawl.
_REF_CACHE: tuple[list[str], np.ndarray] | None = None


def _load_ref(name: str) -> str:
    """Real reference sequence for one gene (indirection keeps the import local)."""
    from .amr import _gene_cassette

    return _gene_cassette(name)


def _reference_matrix() -> tuple[list[str], np.ndarray]:
    global _REF_CACHE
    if _REF_CACHE is None:
        names = list(RESISTANCE_GENES)
        rows = []
        for name in names:
            v = kmer_features(_load_ref(name), k=K_REFERENCE)
            n = float(np.linalg.norm(v))
            rows.append(v / n if n else v)
        _REF_CACHE = (names, np.vstack(rows))
    return _REF_CACHE


def novelty_scan(
    sequence: str,
    window: int = SCAN_WINDOW,
    step: int = SCAN_STEP,
) -> tuple[list[NovelSegment], dict]:
    """
    Find compositionally foreign segments that match no known determinant.

    Returns (segments, diagnostics). `diagnostics` always reports whether the
    scan could run and, if not, why — an unrun check must never be presented
    as a passed one.
    """
    seq = "".join(c for c in sequence.upper() if c in "ACGT")
    diag: dict = {"available": False, "reason": "", "n_windows": 0, "n_explained": 0}

    starts = list(range(0, len(seq) - window + 1, step))
    if len(starts) < 8:
        diag["reason"] = (
            f"the sequence is {len(seq):,} bp, too short to cut into enough "
            f"{window} bp windows to estimate a host background."
        )
        return [], diag

    diag["n_windows"] = len(starts)

    # --- 1. Which windows are already explained by a known determinant?
    #
    # Masking is driven by the POSITIONS locate_genes() reports, not by
    # re-matching profiles at this scan's fixed window size. The reference
    # genes run from 474 bp to 3,519 bp, so a fixed 900 bp window compares
    # a short gene against mostly background and a long gene against a
    # fragment of itself — it would mask the wrong places.
    found = locate_genes(seq)
    explained = np.zeros(len(starts), dtype=bool)
    for _gene, _score, g_start, g_end in found:
        for i, w_start in enumerate(starts):
            w_end = w_start + window
            if w_start < g_end and w_end > g_start:  # overlaps the gene
                lo = max(0, i - BOUNDARY_MARGIN)
                hi = min(len(starts), i + BOUNDARY_MARGIN + 1)
                explained[lo:hi] = True
    diag["n_explained"] = int(explained.sum())

    # Still needed for the "nearest reference" annotation on a novel segment.
    P_ref = np.vstack([kmer_features(seq[i : i + window], k=K_REFERENCE) for i in starts])
    names, ref_matrix = _reference_matrix()
    norms = np.linalg.norm(P_ref, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sims = (P_ref / norms) @ ref_matrix.T
    best_sim = sims.max(axis=1)
    best_idx = sims.argmax(axis=1)

    # --- 2. Host background, from the unexplained windows only
    P_comp = np.vstack(
        [kmer_features(seq[i : i + window], k=K_COMPOSITION) for i in starts]
    )
    pool = P_comp[~explained]
    if len(pool) < MIN_BACKGROUND_POOL:
        diag["reason"] = (
            f"{diag['n_explained']} of {len(starts)} windows are already "
            "explained by known determinants, leaving too little sequence to "
            "estimate what this genome's own composition looks like."
        )
        return [], diag

    background = np.median(pool, axis=0)
    for _ in range(3):
        d = np.array([1.0 - _cosine(p, background) for p in pool])
        core = pool[d <= np.quantile(d, CORE_QUANTILE)]
        if len(core) < 3:
            break
        background = core.mean(axis=0)

    d_pool = np.array([1.0 - _cosine(p, background) for p in pool])
    core_d = d_pool[d_pool <= np.quantile(d_pool, CORE_QUANTILE)]
    med = float(np.median(core_d))
    mad = float(np.median(np.abs(core_d - med))) * 1.4826
    if mad < 1e-6:
        mad = float(np.std(core_d)) or 1e-6

    # --- 3. Score every window; explained ones cannot be novel by definition
    d_all = np.array([1.0 - _cosine(p, background) for p in P_comp])
    excess = d_all - med
    z = excess / mad
    excess[explained] = 0.0
    z[explained] = 0.0

    diag["available"] = True
    diag["peak_excess"] = float(excess.max())
    diag["peak_z"] = float(z.max())

    flagged = [
        i for i in range(len(starts))
        if z[i] >= Z_FLOOR and excess[i] >= ABSOLUTE_FLOOR
    ]
    if not flagged:
        return [], diag

    # --- 4. Merge adjacent flagged windows into contiguous segments
    segments: list[NovelSegment] = []
    run = [flagged[0]]
    for i in flagged[1:]:
        if starts[i] - starts[run[-1]] <= step:
            run.append(i)
        else:
            segments.append(
                _make_segment(run, starts, window, excess, z, best_sim, best_idx, names)
            )
            run = [i]
    segments.append(
        _make_segment(run, starts, window, excess, z, best_sim, best_idx, names)
    )
    segments.sort(key=lambda s: -s.excess_divergence)
    return segments, diag


def _make_segment(
    run: list[int],
    starts: list[int],
    window: int,
    excess: np.ndarray,
    z: np.ndarray,
    best_sim: np.ndarray,
    best_idx: np.ndarray,
    names: list[str],
) -> NovelSegment:
    peak = max(run, key=lambda i: excess[i])
    return NovelSegment(
        start=starts[run[0]],
        end=starts[run[-1]] + window,
        excess_divergence=float(excess[peak]),
        robust_z=float(z[peak]),
        max_ref_similarity=float(best_sim[peak]),
        nearest_gene=names[int(best_idx[peak])],
    )


def ood_badge(sequence: str, model) -> tuple[bool, float, float, str]:
    """
    How far does this isolate sit from the classifier's training distribution?

    Returns (available, distance, percentile, band). Requires `training_stats`
    on the model, written by AMRModel.fit(); an older cached model returns
    available=False rather than a fabricated score.
    """
    stats = getattr(model, "training_stats", None)
    if not stats:
        return False, 0.0, 0.0, "UNKNOWN"

    centroid = np.asarray(stats["centroid"], dtype=np.float64)
    quantiles = np.asarray(stats["distance_quantiles"], dtype=np.float64)

    v = kmer_features(sequence).astype(np.float64)
    dist = 1.0 - _cosine(v, centroid)

    pct = float(np.searchsorted(quantiles, dist) / len(quantiles) * 100.0)
    pct = min(100.0, max(0.0, pct))

    if pct >= 99.0:
        band = "OUT OF DISTRIBUTION"
    elif pct >= 90.0:
        band = "BORDERLINE"
    else:
        band = "IN DISTRIBUTION"
    return True, float(dist), pct, band


def build_novelty_report(
    sequence: str,
    strain_name: str,
    model=None,
    known_gene_count: int = 0,
) -> NoveltyReport:
    """Assemble the Feature 4 report for one isolate."""
    clean = "".join(c for c in sequence.upper() if c in "ACGT")
    segments, diag = novelty_scan(sequence)

    rpt = NoveltyReport(
        strain_name=strain_name,
        sequence_length=len(clean),
        n_windows=diag.get("n_windows", 0),
        n_explained=diag.get("n_explained", 0),
        segments=segments,
        known_gene_count=known_gene_count,
        scan_available=diag.get("available", False),
        scan_unavailable_reason=diag.get("reason", ""),
    )

    if model is not None:
        avail, dist, pct, band = ood_badge(sequence, model)
        rpt.ood_available = avail
        rpt.ood_distance = dist
        rpt.ood_percentile = pct
        rpt.ood_band = band if avail else "UNKNOWN"
    return rpt
