"""
TrialSense — Module 3, Features 2 and 6.

FEATURE 2 — RESCUE STRATEGY
---------------------------
The core classifier answers "resistant: yes or no". That verdict is not a
decision, because two isolates can produce the identical call and imply
opposite business outcomes:

    blaKPC-2   carbapenem resistant, SERINE enzyme
               -> avibactam / vaborbactam / relebactam all neutralise it
               -> pair the candidate with an inhibitor; programme survives

    blaNDM-1   carbapenem resistant, METALLO enzyme
               -> none of those inhibitors touch a zinc-dependent enzyme
               -> standard partner chemistry fails; different route required

Same class-level verdict. One is a formulation change costing months. The other
is a programme decision. This module reports which one you are looking at.

FEATURE 6 — SPREAD VELOCITY
---------------------------
Resistance determinants travel two ways. Chromosomal mutations pass only to
offspring, so they spread by clonal expansion. Mobile elements — plasmids,
integrons, transposons — transfer between cells and across species. mcr-1 went
from first description in 2015 to reports on multiple continents within roughly
eighteen months; no chromosomal point mutation does that.

That difference matters to Feature 1, because a national trend line fitted to
historical averages understates how fast a plasmid-borne determinant will move.
We therefore hand Feature 1 a multiplier and say we have done so.

Everything in both features is encoded textbook microbiology, read off the
annotations in amr.py. Neither feature runs a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .amr import LAST_RESORT, RESISTANCE_GENES, ResistanceGene

# Plain-English names for the mechanism families, used in the UI.
MECHANISM_LABELS: dict[str, str] = {
    "enzymatic_hydrolysis": "Enzymatic destruction",
    "target_modification": "Target modification",
    "ribosomal_methylation": "Ribosomal methylation",
    "drug_modification": "Drug modification",
    "target_protection": "Target protection",
    "membrane_modification": "Membrane modification",
    "bypass": "Metabolic bypass",
    "unclassified": "Unclassified",
}

# The lock-and-key analogy, kept in one place so the UI and any exported
# report tell the same story.
MECHANISM_ANALOGY: dict[str, str] = {
    "enzymatic_hydrolysis": (
        "The organism secretes a molecular scissor that cuts the drug apart "
        "before it can reach its target — gum jammed into the keyhole."
    ),
    "target_modification": (
        "The target the drug binds to has been structurally altered. The whole "
        "lock has been swapped for a different model."
    ),
    "ribosomal_methylation": (
        "A chemical cap is added to the drug's landing site on the ribosome, "
        "blocking the entire class at once — a metal plate bolted over the lock."
    ),
    "drug_modification": (
        "The organism chemically defaces the drug so it no longer fits its "
        "target — the key is filed down before it reaches the door."
    ),
    "target_protection": (
        "A shield protein physically displaces the drug from its target. The "
        "lock is unchanged; something keeps knocking the key out."
    ),
    "membrane_modification": (
        "The outer membrane is remodelled so the drug can no longer attach — "
        "the door has been moved."
    ),
    "bypass": (
        "The organism runs an alternative, drug-insensitive version of the "
        "enzyme being blocked — a second door that ignores the lock entirely."
    ),
}

MOBILITY_LABELS: dict[str, str] = {
    "plasmid": "Plasmid-borne",
    "integron": "Integron cassette",
    "transposon": "Conjugative transposon",
    "sccmec": "SCCmec element (chromosomally integrated)",
    "chromosomal": "Chromosomal",
    "unknown": "Unknown",
}

MOBILITY_ANALOGY: dict[str, str] = {
    "plasmid": (
        "Transfers cell-to-cell and across species — the cheat sheet forwarded "
        "on WhatsApp, on every phone by evening."
    ),
    "integron": (
        "A cassette on a mobile element, usually carrying several resistance "
        "genes together — the whole answer key shared as one file."
    ),
    "transposon": (
        "Jumps between DNA molecules and transfers by conjugation — slower "
        "than a plasmid, far faster than inheritance."
    ),
    "sccmec": (
        "A mobile element that integrates into the chromosome. It moves, but "
        "much less readily than a free plasmid."
    ),
    "chromosomal": (
        "Passes only to offspring. Spreads by clonal expansion — the family "
        "recipe, never leaving the family."
    ),
}


# =============================================================================
# Feature 2 — Rescue strategy
# =============================================================================


@dataclass
class GeneVerdict:
    """One detected gene, assessed against the candidate's own class."""

    gene: ResistanceGene
    confidence: float
    hits_candidate_class: bool
    rescue_viable: str  # yes | partial | no | unknown

    @property
    def name(self) -> str:
        return self.gene.name

    @property
    def mechanism_label(self) -> str:
        return MECHANISM_LABELS.get(self.gene.mechanism_family, "Unclassified")

    @property
    def analogy(self) -> str:
        return MECHANISM_ANALOGY.get(self.gene.mechanism_family, "")


@dataclass
class RescueReport:
    """Feature 2 output — a standalone, self-contained report."""

    candidate_name: str
    candidate_class: str | None
    strain_name: str

    blocking: list[GeneVerdict] = field(default_factory=list)   # hit our class
    other: list[GeneVerdict] = field(default_factory=list)      # hit other classes
    class_screened: bool = True
    unavailable_reason: str = ""

    @property
    def verdict(self) -> str:
        """CLEAR | SALVAGEABLE | PARTIAL | DEAD END | NOT SCREENED"""
        if not self.class_screened:
            return "NOT SCREENED"
        if not self.blocking:
            return "CLEAR"
        levels = {gv.rescue_viable for gv in self.blocking}
        # The worst blocking determinant governs — rescuing one enzyme while
        # another still destroys the drug is not a rescue.
        if "no" in levels or "unknown" in levels:
            return "DEAD END"
        if "partial" in levels:
            return "PARTIAL"
        return "SALVAGEABLE"

    @property
    def color(self) -> str:
        return {
            "CLEAR": "#2E9E5B",
            "SALVAGEABLE": "#E0A526",
            "PARTIAL": "#E8722C",
            "DEAD END": "#D6453D",
            "NOT SCREENED": "#5B7290",
        }[self.verdict]

    def headline(self) -> str:
        v = self.verdict
        if v == "NOT SCREENED":
            return f"{self.candidate_class or 'This class'} is not screened by Module 3"
        if v == "CLEAR":
            return (
                f"No determinant against the {self.candidate_class} class in "
                f"{self.strain_name}"
            )
        names = ", ".join(gv.name for gv in self.blocking)
        return f"{self.candidate_class} blocked by {names} — verdict: {v}"

    def plain_summary(self) -> str:
        v = self.verdict
        if v == "NOT SCREENED":
            return (
                f"{self.candidate_name} belongs to the {self.candidate_class} "
                f"class, which is not among the twelve classes Module 3 screens. "
                f"We report that honestly rather than returning a confident "
                f"0% resistance for a class we never checked. "
                f"{self.unavailable_reason}"
            )
        if v == "CLEAR":
            return (
                f"No resistance determinant against the {self.candidate_class} "
                f"class was detected in {self.strain_name}. On the mechanism "
                f"evidence, {self.candidate_name} remains viable against this "
                "isolate — no rescue strategy is needed."
            )
        lead = self.blocking[0]
        if v == "SALVAGEABLE":
            return (
                f"Resistance here is {lead.mechanism_label.lower()} via "
                f"{lead.name}, and it is chemically rescuable. "
                f"{lead.gene.rescue_strategy} "
                f"The recommendation is to change the formulation, not the "
                f"programme — {self.candidate_name} itself is not the problem."
            )
        if v == "PARTIAL":
            return (
                f"Resistance here is {lead.mechanism_label.lower()} via "
                f"{lead.name}. A rescue route exists but is not complete. "
                f"{lead.gene.rescue_strategy} "
                "Treat this as a viable but narrowed path — verify against the "
                "specific isolate before committing."
            )
        blocker = next(
            (gv for gv in self.blocking if gv.rescue_viable in ("no", "unknown")),
            lead,
        )
        return (
            f"Resistance here is {blocker.mechanism_label.lower()} via "
            f"{blocker.name}, and it is NOT rescuable by partner chemistry. "
            f"{blocker.gene.rescue_strategy} "
            f"This is a programme-level finding, not a formulation problem: "
            f"{self.candidate_name} needs a different target organism, or the "
            "scaffold needs to change."
        )

    def recommendation(self) -> str:
        return {
            "CLEAR": "Proceed. No mechanism-based obstacle for this class.",
            "SALVAGEABLE": "Change the formulation. Pair with the named "
            "inhibitor and re-screen; the scaffold survives.",
            "PARTIAL": "Narrow path. Pursue the named alternative agent, but "
            "confirm activity against this specific isolate first.",
            "DEAD END": "Change the programme. Re-target to a less "
            "resistant organism, or move to a different mechanism class.",
            "NOT SCREENED": "Out of scope. Extend the class panel, or assess "
            "this candidate outside Module 3.",
        }[self.verdict]

    def contrast_note(self) -> str:
        """
        The teaching moment: show the reader that the same class-level verdict
        can carry the opposite decision, so the mechanism layer is not cosmetic.
        """
        if self.verdict not in ("DEAD END", "SALVAGEABLE", "PARTIAL"):
            return ""
        families = {gv.gene.mechanism_family for gv in self.blocking}
        if "enzymatic_hydrolysis" not in families:
            return ""
        subtypes = " / ".join(
            sorted({gv.gene.enzyme_subtype for gv in self.blocking if gv.gene.enzyme_subtype})
        )
        if "METALLO" in subtypes.upper():
            return (
                "Class-level reporting would call this simply 'carbapenem "
                "resistant' — identical to a blaKPC-2 isolate, which IS "
                "rescuable with an off-the-shelf inhibitor. The metallo/serine "
                "distinction is the whole decision, and it is invisible at "
                "class level."
            )
        if "serine" in subtypes.lower():
            return (
                "Class-level reporting would call this simply 'resistant' — "
                "identical to a blaNDM-1 isolate, which is NOT rescuable. Here "
                "the enzyme is a serine type, so existing inhibitors apply."
            )
        return ""


def build_rescue_report(
    detected_genes: list[tuple[str, float]],
    candidate_class: str | None,
    candidate_name: str,
    strain_name: str,
    screened_classes: list[str],
) -> RescueReport:
    """Assemble the Feature 2 report for one candidate against one isolate."""
    rpt = RescueReport(
        candidate_name=candidate_name,
        candidate_class=candidate_class,
        strain_name=strain_name,
    )

    if candidate_class is None:
        rpt.class_screened = False
        rpt.unavailable_reason = (
            "This compound is not an antibacterial, so there is no antibiotic "
            "class to assess."
        )
    elif candidate_class not in screened_classes:
        rpt.class_screened = False
        rpt.unavailable_reason = (
            f"Module 3 screens {len(screened_classes)} classes; "
            f"{candidate_class} is not one of them. Extending the panel means "
            "adding reference determinants for that class."
        )

    for gene_name, confidence in detected_genes:
        gene = RESISTANCE_GENES.get(gene_name)
        if gene is None:
            continue
        hits = bool(candidate_class and candidate_class in gene.confers_resistance_to)
        gv = GeneVerdict(
            gene=gene,
            confidence=confidence,
            hits_candidate_class=hits,
            rescue_viable=gene.rescue_viable,
        )
        (rpt.blocking if hits else rpt.other).append(gv)

    # Worst blocker first — that is the one driving the decision.
    order = {"no": 0, "unknown": 1, "partial": 2, "yes": 3}
    rpt.blocking.sort(key=lambda gv: (order.get(gv.rescue_viable, 1), -gv.confidence))
    rpt.other.sort(key=lambda gv: -gv.confidence)
    return rpt


# =============================================================================
# Feature 6 — Spread velocity
# =============================================================================


@dataclass
class MobilityEntry:
    gene: ResistanceGene
    confidence: float

    @property
    def label(self) -> str:
        return MOBILITY_LABELS.get(self.gene.mobility, "Unknown")

    @property
    def analogy(self) -> str:
        return MOBILITY_ANALOGY.get(self.gene.mobility, "")


@dataclass
class MobilityReport:
    """Feature 6 output — a standalone, self-contained report."""

    strain_name: str
    entries: list[MobilityEntry] = field(default_factory=list)

    @property
    def mobile(self) -> list[MobilityEntry]:
        return [e for e in self.entries if e.gene.is_mobile]

    @property
    def fixed(self) -> list[MobilityEntry]:
        return [e for e in self.entries if not e.gene.is_mobile]

    @property
    def multiplier(self) -> float:
        """
        The single factor handed to Feature 1. Driven by the FASTEST-moving
        determinant present, not the average — an isolate carrying one plasmid
        gene and four chromosomal ones still spreads that plasmid gene at
        plasmid speed.
        """
        if not self.entries:
            return 1.0
        return max(e.gene.spread_multiplier for e in self.entries)

    @property
    def verdict(self) -> str:
        """STATIC | SLOW | MOBILE | HIGHLY MOBILE | NONE DETECTED"""
        if not self.entries:
            return "NONE DETECTED"
        m = self.multiplier
        if m >= 2.5:
            return "HIGHLY MOBILE"
        if m >= 1.8:
            return "MOBILE"
        if m > 1.0:
            return "SLOW"
        return "STATIC"

    @property
    def color(self) -> str:
        return {
            "HIGHLY MOBILE": "#D6453D",
            "MOBILE": "#E8722C",
            "SLOW": "#E0A526",
            "STATIC": "#2E9E5B",
            "NONE DETECTED": "#5B7290",
        }[self.verdict]

    def headline(self) -> str:
        if not self.entries:
            return f"No resistance determinants detected in {self.strain_name}"
        n_mob = len(self.mobile)
        return (
            f"{n_mob} of {len(self.entries)} determinants are horizontally "
            f"transferable — spread velocity: {self.verdict}"
        )

    def plain_summary(self) -> str:
        if not self.entries:
            return (
                f"No resistance determinants were detected in {self.strain_name}, "
                "so there is nothing here to spread."
            )
        v = self.verdict
        if v == "STATIC":
            return (
                "Every determinant detected is chromosomal — it passes only to "
                "offspring. Resistance here spreads by clonal expansion of "
                "resistant strains, which is slow and geographically traceable. "
                "The national trend projection can be taken at face value."
            )
        fastest = max(self.entries, key=lambda e: e.gene.spread_multiplier)
        return (
            f"{fastest.gene.name} is {fastest.label.lower()} — it transfers "
            f"between cells and across species, not merely to offspring. "
            f"Historical spread of this kind has outpaced chromosomal "
            f"resistance by roughly an order of magnitude (mcr-1 reached "
            f"multiple continents within about eighteen months of first "
            f"description in 2015). Treat the resistance-runway projection as "
            f"OPTIMISTIC: we shorten it by a factor of {self.multiplier:.1f} to "
            "reflect that, and flag that this adjustment is a reasoned order of "
            "magnitude rather than a fitted constant."
        )

    def runway_note(self) -> str:
        """The one-line caption Feature 1 prints alongside its projection."""
        if self.multiplier <= 1.0:
            return (
                "**Mobility:** all determinants chromosomal — no acceleration "
                "applied to the runway projection."
            )
        fastest = max(self.entries, key=lambda e: e.gene.spread_multiplier)
        return (
            f"**Mobility adjustment:** {fastest.gene.name} is "
            f"{fastest.label.lower()}, so the runway is shortened by "
            f"{self.multiplier:.1f}x. This is a reasoned order of magnitude, "
            "not a fitted parameter."
        )

    def last_resort_mobile(self) -> list[MobilityEntry]:
        """Mobile determinants that defeat a last-resort class — the alarm set."""
        return [
            e for e in self.mobile
            if any(c in LAST_RESORT for c in e.gene.confers_resistance_to)
        ]


def build_mobility_report(
    detected_genes: list[tuple[str, float]], strain_name: str
) -> MobilityReport:
    """Assemble the Feature 6 report for one isolate."""
    entries = []
    for gene_name, confidence in detected_genes:
        gene = RESISTANCE_GENES.get(gene_name)
        if gene is not None:
            entries.append(MobilityEntry(gene=gene, confidence=confidence))
    entries.sort(key=lambda e: -e.gene.spread_multiplier)
    return MobilityReport(strain_name=strain_name, entries=entries)
