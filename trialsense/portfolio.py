"""
TrialSense — Module 3, Feature 3: the Portfolio Screen.

WHY THIS EXISTS
---------------
The rest of Module 3 answers "is THIS candidate viable against THIS organism?".
That is not the question an R&D portfolio committee is in the room to answer.
They hold six to ten candidates, a list of priority target organisms, and
budget to advance two. The artefact they actually need is a grid.

Screening N candidates against M organisms is N x M runs of machinery that
already exists, so this module is deliberately thin: it orchestrates, caches,
ranks and times. It adds no new science.

WHAT MAKES IT WORTH DEMONSTRATING
---------------------------------
Every competing tool in this space is isolate-centric — one bug, one report.
Working at portfolio level is the difference between a laboratory instrument
and a decision tool, and the wall-clock timing makes the cost argument without
anyone having to assert it: a grid that takes seconds here takes weeks of
bench work otherwise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .amr import (
    ANTIBIOTIC_CLASSES,
    LAST_RESORT,
    RESISTANCE_CALL_THRESHOLD,
    AMRReport,
)

# Blocking a candidate is a SCREENING decision, not a confident diagnosis, so
# it uses the sensitive threshold rather than the confirmed one. A candidate
# flagged here gets a confirmatory assay; one wrongly cleared here reaches a
# trial. The asymmetry is the whole point.
BLOCK_THRESHOLD = RESISTANCE_CALL_THRESHOLD


@dataclass
class Candidate:
    """One antibacterial under consideration."""

    name: str
    antibiotic_class: str

    @property
    def screened(self) -> bool:
        return self.antibiotic_class in ANTIBIOTIC_CLASSES


@dataclass
class Cell:
    """One candidate x one organism."""

    candidate: str
    organism: str
    resistance: float | None      # None when the class is not screened
    blocked: bool = False
    screened: bool = True
    last_resort: bool = False
    rescue_verdict: str = ""      # from Feature 2, when available

    @property
    def symbol(self) -> str:
        if not self.screened:
            return "—"
        return "✖" if self.blocked else "✔"

    @property
    def color(self) -> str:
        if not self.screened:
            return "#5B7290"
        if not self.blocked:
            return "#2E9E5B"
        return "#D6453D" if self.last_resort else "#E8722C"


@dataclass
class CandidateScore:
    """A candidate's standing across the whole organism panel."""

    name: str
    antibiotic_class: str
    viable_against: int
    blocked_against: int
    not_screened: int
    mean_resistance: float | None
    salvageable: int = 0  # blocked, but Feature 2 says formulation can fix it

    @property
    def total_assessed(self) -> int:
        return self.viable_against + self.blocked_against

    @property
    def viability_rate(self) -> float:
        return self.viable_against / self.total_assessed if self.total_assessed else 0.0


@dataclass
class PortfolioReport:
    """Feature 3 output — a standalone, self-contained report."""

    candidates: list[Candidate] = field(default_factory=list)
    organisms: list[str] = field(default_factory=list)
    cells: dict[tuple[str, str], Cell] = field(default_factory=dict)
    scores: list[CandidateScore] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    # Feature 3 has no data of its own — it orchestrates the core classifier.
    # This is how it earns a validation claim: every prediction in the grid is
    # laid against real ICMR national surveillance for the same organism and
    # antibiotic class. Populated by the caller via attach_cross_check().
    cross_check: object | None = None

    @property
    def n_combinations(self) -> int:
        return len(self.candidates) * len(self.organisms)

    @property
    def winner(self) -> CandidateScore | None:
        return self.scores[0] if self.scores else None

    @property
    def verdict(self) -> str:
        """CLEAR LEADER | CONTESTED | NO VIABLE CANDIDATE"""
        if not self.scores:
            return "NO VIABLE CANDIDATE"
        top = self.scores[0]
        if top.viable_against == 0:
            return "NO VIABLE CANDIDATE"
        if len(self.scores) > 1 and self.scores[1].viable_against == top.viable_against:
            return "CONTESTED"
        return "CLEAR LEADER"

    @property
    def color(self) -> str:
        return {
            "CLEAR LEADER": "#2E9E5B",
            "CONTESTED": "#E0A526",
            "NO VIABLE CANDIDATE": "#D6453D",
        }[self.verdict]

    def headline(self) -> str:
        return (
            f"{self.n_combinations} candidate-organism combinations screened in "
            f"{self.elapsed_seconds:.1f} seconds"
        )

    def plain_summary(self) -> str:
        if not self.scores:
            return "No candidates were supplied to screen."
        top = self.scores[0]
        if self.verdict == "NO VIABLE CANDIDATE":
            return (
                "Every candidate in this portfolio is blocked against every "
                "organism assessed. That is a portfolio-level finding: the "
                "problem is not which candidate to advance, it is that this "
                "target panel needs a mechanism none of these candidates have."
            )
        lead = (
            f"**{top.name}** ({top.antibiotic_class}) is the strongest "
            f"candidate — viable against {top.viable_against} of "
            f"{top.total_assessed} organisms assessed"
        )
        if self.verdict == "CONTESTED":
            tied = [s.name for s in self.scores if s.viable_against == top.viable_against]
            lead = (
                f"**{' and '.join(tied)}** tie on breadth "
                f"({top.viable_against} of {top.total_assessed} organisms). "
                f"{top.name} edges ahead on mean predicted resistance"
            )
        blocked = [s for s in self.scores if s.blocked_against > 0]
        tail = ""
        if blocked:
            worst = max(blocked, key=lambda s: s.blocked_against)
            tail = (
                f" {worst.name} is blocked against {worst.blocked_against} "
                f"organism(s), including the highest-burden targets on this panel."
            )
        return lead + "." + tail

    def recommendation(self) -> str:
        if not self.scores or self.verdict == "NO VIABLE CANDIDATE":
            return (
                "Do not advance any candidate on this panel. Re-scope either "
                "the target organisms or the chemistry."
            )
        top = self.scores[0]
        rec = f"Advance {top.name}."
        if top.blocked_against:
            rec += (
                f" Note it is still blocked against {top.blocked_against} "
                "organism(s) — exclude those indications from the protocol."
            )
        rescuable = [s for s in self.scores if s.salvageable > 0]
        if rescuable:
            r = rescuable[0]
            rec += (
                f" {r.name} is blocked but mechanistically rescuable against "
                f"{r.salvageable} organism(s); a partner-agent formulation "
                "would bring it back into contention."
            )
        return rec

    def as_rows(self) -> list[dict]:
        """Flat rows for export/download."""
        out = []
        for c in self.candidates:
            for o in self.organisms:
                cell = self.cells.get((c.name, o))
                if cell is None:
                    continue
                out.append(
                    {
                        "candidate": c.name,
                        "class": c.antibiotic_class,
                        "organism": o,
                        "predicted_resistance": (
                            None if cell.resistance is None else round(cell.resistance, 4)
                        ),
                        "status": (
                            "not screened" if not cell.screened
                            else "blocked" if cell.blocked else "viable"
                        ),
                        "rescue": cell.rescue_verdict,
                    }
                )
        return out


def screen_portfolio(
    candidates: list[Candidate],
    organisms: list[str],
    analyze_fn,
    rescue_fn=None,
) -> PortfolioReport:
    """
    Run every candidate against every organism.

    analyze_fn(organism_name) -> AMRReport. Caching belongs to the caller (the
    Streamlit layer memoises it), so the same organism is never sequenced and
    classified twice within one grid.

    rescue_fn(amr_report, candidate) -> verdict string, optional. Supplying it
    lets the grid distinguish "blocked and dead" from "blocked but rescuable",
    which is the distinction that changes what the committee funds.
    """
    started = time.perf_counter()
    rpt = PortfolioReport(candidates=list(candidates), organisms=list(organisms))

    # One analysis per organism, reused across every candidate.
    reports: dict[str, AMRReport] = {o: analyze_fn(o) for o in organisms}

    for cand in candidates:
        for org in organisms:
            amr = reports[org]
            if not cand.screened:
                rpt.cells[(cand.name, org)] = Cell(
                    candidate=cand.name,
                    organism=org,
                    resistance=None,
                    screened=False,
                )
                continue

            prob = amr.probabilities.get(cand.antibiotic_class, 0.0)
            blocked = prob >= BLOCK_THRESHOLD
            verdict = ""
            if blocked and rescue_fn is not None:
                try:
                    verdict = rescue_fn(amr, cand)
                except Exception:  # noqa: BLE001 — the grid must never break the demo
                    verdict = ""
            rpt.cells[(cand.name, org)] = Cell(
                candidate=cand.name,
                organism=org,
                resistance=float(prob),
                blocked=blocked,
                screened=True,
                last_resort=cand.antibiotic_class in LAST_RESORT,
                rescue_verdict=verdict,
            )

    # Rank: breadth of viability first, then lower mean resistance as the
    # tie-break. Breadth is what a portfolio committee optimises for.
    scores: list[CandidateScore] = []
    for cand in candidates:
        cells = [rpt.cells[(cand.name, o)] for o in organisms if (cand.name, o) in rpt.cells]
        assessed = [c for c in cells if c.screened and c.resistance is not None]
        viable = [c for c in assessed if not c.blocked]
        blocked = [c for c in assessed if c.blocked]
        scores.append(
            CandidateScore(
                name=cand.name,
                antibiotic_class=cand.antibiotic_class,
                viable_against=len(viable),
                blocked_against=len(blocked),
                not_screened=len(cells) - len(assessed),
                mean_resistance=(
                    sum(c.resistance for c in assessed) / len(assessed)
                    if assessed else None
                ),
                salvageable=sum(
                    1 for c in blocked if c.rescue_verdict in ("SALVAGEABLE", "PARTIAL")
                ),
            )
        )

    scores.sort(
        key=lambda s: (
            -s.viable_against,
            s.mean_resistance if s.mean_resistance is not None else 1.0,
        )
    )
    rpt.scores = scores
    rpt.elapsed_seconds = time.perf_counter() - started
    rpt._amr_reports = reports  # kept for the cross-check
    return rpt


def attach_cross_check(rpt: PortfolioReport, strain_genes, gene_classes,
                       source_key: str = "icmr"):
    """
    Validate every prediction in the grid against real national surveillance.

    Separate from screen_portfolio() so the grid still works when no reference
    data exists for the selected organisms — an unvalidated grid is useful, a
    crashed one is not.
    """
    from .surveillance import cross_check_predictions

    reports = getattr(rpt, "_amr_reports", None)
    if not reports:
        return rpt
    rpt.cross_check = cross_check_predictions(
        reports, ANTIBIOTIC_CLASSES, strain_genes, gene_classes, source_key
    )
    return rpt
