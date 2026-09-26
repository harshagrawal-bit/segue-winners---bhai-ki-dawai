"""
TrialSense — Module 3, Feature 5: the Cost Counterfactual.

WHAT THIS IS
------------
A transparent decision-analysis model that answers the only question a budget
committee actually asks: what is it worth to learn this now rather than later?

The framework is standard health economics — the expected value of sample
information. Informally: a test has value equal to the cost of the mistakes it
prevents, weighted by how likely those mistakes were.

WHAT THIS IS EMPHATICALLY NOT
-----------------------------
It is NOT a claim that TrialSense has saved anyone money. Nobody has run a drug
programme through this tool for ten years, and pretending otherwise would be
the fastest way to lose a technical audience.

Every input below is a DEFAULT ASSUMPTION exposed as a slider. The honest claim
is narrow and robust:

    Running this check costs effectively nothing.
    Not running it costs, at minimum, one failed development phase.

You do not need to accept any specific number for that asymmetry to hold, which
is exactly why we expose the inputs instead of burying them.

ON THE DEFAULTS
---------------
The phase costs and success probabilities below are order-of-magnitude figures
for anti-infective development, stated in crore rupees. They are STARTING
POINTS for a user who has their own numbers, not measurements, and the UI says
so on the same screen. Antibacterial development is widely reported as having
lower per-phase cost and higher technical success than oncology, and these
defaults reflect that shape rather than any single published study.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Development phases, in order. Cost in crore rupees (1 crore = 10 million INR).
# `pts` = probability of technical success: the chance of clearing this phase
# given you entered it.
PHASES = ["Preclinical", "Phase I", "Phase II", "Phase III"]

DEFAULT_COSTS: dict[str, float] = {
    "Preclinical": 15.0,
    "Phase I": 25.0,
    "Phase II": 60.0,
    "Phase III": 250.0,
}

DEFAULT_PTS: dict[str, float] = {
    "Preclinical": 0.70,
    "Phase I": 0.65,
    "Phase II": 0.45,
    "Phase III": 0.60,
}

DEFAULT_DURATION_YEARS: dict[str, float] = {
    "Preclinical": 1.5,
    "Phase I": 1.5,
    "Phase II": 2.5,
    "Phase III": 3.0,
}

# Where a resistance problem surfaces if nobody screens for it computationally.
# Efficacy against the target organism is not seriously tested until patients
# with that infection are enrolled, which is Phase II.
DEFAULT_DISCOVERY_PHASE = "Phase II"

# What the computational screen itself costs. Effectively a laptop and an
# afternoon; we keep it as an explicit non-zero line so the comparison is fair.
SCREEN_COST_CRORE = 0.02
SCREEN_DURATION_YEARS = 0.01


@dataclass
class PhaseLine:
    """One row of the waterfall."""

    phase: str
    cost: float
    duration_years: float
    reached: bool
    cumulative_cost: float
    cumulative_years: float


@dataclass
class ValueReport:
    """Feature 5 output — a standalone, self-contained report."""

    candidate_name: str
    organism: str
    resistance_pct: float
    triggered: bool  # did Module 3 actually flag a problem?

    discovery_phase: str = DEFAULT_DISCOVERY_PHASE
    costs: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COSTS))
    pts: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PTS))
    durations: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DURATION_YEARS)
    )

    lines: list[PhaseLine] = field(default_factory=list)
    cost_without: float = 0.0
    years_without: float = 0.0
    cost_with: float = 0.0
    years_with: float = 0.0
    probability_reaching_discovery: float = 1.0
    expected_value_crore: float = 0.0

    @property
    def cost_avoided(self) -> float:
        return max(0.0, self.cost_without - self.cost_with)

    @property
    def years_avoided(self) -> float:
        return max(0.0, self.years_without - self.years_with)

    @property
    def verdict(self) -> str:
        """HIGH VALUE | MODERATE VALUE | CONFIRMATORY"""
        if not self.triggered:
            return "CONFIRMATORY"
        if self.expected_value_crore >= 50:
            return "HIGH VALUE"
        return "MODERATE VALUE"

    @property
    def color(self) -> str:
        return {
            "HIGH VALUE": "#2E9E5B",
            "MODERATE VALUE": "#E0A526",
            "CONFIRMATORY": "#5B7290",
        }[self.verdict]

    def headline(self) -> str:
        if not self.triggered:
            return (
                "No resistance flag — this screen returns a clean result, not a "
                "saving"
            )
        return (
            f"Finding this at screening rather than {self.discovery_phase} avoids "
            f"an estimated ₹{self.cost_avoided:.0f} crore and "
            f"{self.years_avoided:.1f} years"
        )

    def plain_summary(self) -> str:
        if not self.triggered:
            return (
                f"Module 3 found no resistance obstacle for {self.candidate_name} "
                f"against {self.organism}. There is no avoided cost to report — "
                "the value of a negative screen is confirmation, not savings. "
                "Quoting a rupee figure here would be dishonest, so we do not."
            )
        return (
            f"Without a computational screen, resistance to "
            f"{self.candidate_name} in {self.organism} "
            f"({self.resistance_pct:.0f}% predicted) would most likely surface "
            f"during {self.discovery_phase} — after roughly "
            f"₹{self.cost_without:.0f} crore and {self.years_without:.1f} years "
            f"were already committed. Module 3 surfaces the same signal on day "
            f"one for ₹{self.cost_with:.2f} crore.\n\n"
            f"Weighting by the {self.probability_reaching_discovery:.0%} chance "
            f"the programme survives far enough to hit that wall, the expected "
            f"value of running this screen is **₹{self.expected_value_crore:.0f} "
            f"crore** per candidate assessed."
        )

    def disclaimer(self) -> str:
        return (
            "**These are your assumptions, not our measurements.** Every cost, "
            "probability and duration above is an adjustable input seeded with "
            "an order-of-magnitude default. Change any of them and this number "
            "changes. What does not depend on the numbers is the shape of the "
            "asymmetry: the screen costs approximately nothing, and the "
            "alternative costs at minimum one failed phase."
        )

    def recommendation(self) -> str:
        if not self.triggered:
            return (
                "No action. Record the negative screen as supporting evidence "
                "for this indication and proceed."
            )
        return (
            f"Re-scope before committing preclinical spend. The signal is "
            f"available now for ₹{self.cost_with:.2f} crore; the same finding "
            f"costs ₹{self.cost_without:.0f} crore and {self.years_without:.1f} "
            f"years if left to emerge in {self.discovery_phase}."
        )


def build_value_report(
    candidate_name: str,
    organism: str,
    resistance_pct: float,
    triggered: bool,
    discovery_phase: str = DEFAULT_DISCOVERY_PHASE,
    costs: dict[str, float] | None = None,
    pts: dict[str, float] | None = None,
    durations: dict[str, float] | None = None,
    screen_cost: float = SCREEN_COST_CRORE,
) -> ValueReport:
    """
    Build the Feature 5 counterfactual.

    The arithmetic is deliberately simple enough to check by hand on screen:
      cost_without = sum of every phase cost up to and including the phase
                     where the problem would have surfaced
      cost_with    = the screen itself
      expected     = (cost_without - cost_with) x P(programme survives to
                     that phase)

    The probability weighting is what stops this being a headline number that
    assumes every flagged programme would otherwise have marched all the way
    to Phase III. Most would have died earlier for unrelated reasons, and
    ignoring that would inflate the claim.
    """
    rpt = ValueReport(
        candidate_name=candidate_name,
        organism=organism,
        resistance_pct=resistance_pct,
        triggered=triggered,
        discovery_phase=discovery_phase,
        costs=dict(costs or DEFAULT_COSTS),
        pts=dict(pts or DEFAULT_PTS),
        durations=dict(durations or DEFAULT_DURATION_YEARS),
    )

    if discovery_phase not in PHASES:
        discovery_phase = DEFAULT_DISCOVERY_PHASE
        rpt.discovery_phase = discovery_phase

    stop = PHASES.index(discovery_phase)

    cum_cost = 0.0
    cum_years = 0.0
    survive = 1.0
    for i, phase in enumerate(PHASES):
        reached = i <= stop
        if reached:
            cum_cost += rpt.costs.get(phase, 0.0)
            cum_years += rpt.durations.get(phase, 0.0)
        rpt.lines.append(
            PhaseLine(
                phase=phase,
                cost=rpt.costs.get(phase, 0.0),
                duration_years=rpt.durations.get(phase, 0.0),
                reached=reached,
                cumulative_cost=cum_cost,
                cumulative_years=cum_years,
            )
        )
        # Probability of surviving the phases BEFORE the discovery phase —
        # you must clear those to reach the wall at all.
        if i < stop:
            survive *= rpt.pts.get(phase, 1.0)

    rpt.cost_without = cum_cost
    rpt.years_without = cum_years
    rpt.cost_with = screen_cost
    rpt.years_with = SCREEN_DURATION_YEARS
    rpt.probability_reaching_discovery = survive
    rpt.expected_value_crore = (
        (rpt.cost_without - rpt.cost_with) * survive if triggered else 0.0
    )
    return rpt
