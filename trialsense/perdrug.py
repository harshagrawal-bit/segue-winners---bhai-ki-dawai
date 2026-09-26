"""
TrialSense — Module 3: per-drug resolution within an antibiotic class.

THE PROBLEM THIS FIXES
----------------------
Module 3 predicts resistance at CLASS level — twelve classes, one call each.
That is too coarse in at least one place we can name precisely, and it is a
limitation the project's own ICMR cross-check exposed:

    aac(6')-Ib detected -> we report "Aminoglycosides resistant"
    real ICMR amikacin resistance in Indian E. coli: far lower

Two independent reference databases confirm why. aac(6')-Ib is an
acetyltransferase that modifies amikacin, kanamycin and tobramycin — and
NOT gentamicin. Reporting the whole class as defeated overstates the damage
and, worse, hides the agent that still works.

The contrast with armA makes the point sharper. Both are "aminoglycoside
resistance" at class level. They are not the same finding:

    aac(6')-Ib  amikacin, kanamycin, tobramycin      gentamicin SPARED
    armA        essentially the entire class          gentamicin included

One leaves you a drug. The other does not.

WHAT THIS MODULE DOES, AND DELIBERATELY DOES NOT
------------------------------------------------
It does NOT change the classifier. The model still predicts twelve classes,
and retraining it on a per-drug label space would need per-drug ground truth
we do not have.

What it adds is a RESOLUTION layer over the gene-detection results: when a
determinant is identified, report which individual drugs it actually defeats
and — the useful half — which drugs in that class it leaves alone.

That is a smaller claim than a per-drug model, and it is one the data
genuinely supports, because it comes from curated databases rather than from
our own inference.

PROVENANCE
----------
trialsense/reference_sequences/gene_drug_mappings.csv, built from:
  * CARD ontology `confers_resistance_to_antibiotic` relations
    (card.mcmaster.ca/latest/ontology)
  * NCBI AMRFinderPlus ReferenceGeneCatalog.txt `subclass` field,
    database 2026-08-07.1

Every row carries its source relation and a resolvable URL. Where the two
databases disagree, both are kept and the disagreement is reported rather than
silently resolved — see DB_DISAGREEMENTS.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

_MAPPING_PATH = (
    Path(__file__).parent / "reference_sequences" / "gene_drug_mappings.csv"
)

# Rows carrying this literal mean "the database links this gene to the class
# but never enumerates individual drugs". That is NOT the same as "no drugs",
# and treating it as data would invent precision we do not have.
_CLASS_ONLY = "(class-level link only, no individual drugs enumerated)"

# Drugs we care about reporting on, per class. Anything outside this list is
# real but not clinically front-line in the Indian context this tool targets,
# so listing it would add noise rather than information.
CLINICAL_DRUGS: dict[str, list[str]] = {
    "Aminoglycosides": ["amikacin", "gentamicin", "tobramycin", "kanamycin", "plazomicin"],
    "Carbapenems": ["imipenem", "meropenem", "ertapenem"],
    "Cephalosporins": ["cefotaxime", "ceftazidime", "ceftriaxone", "cefepime"],
    "Fluoroquinolones": ["ciprofloxacin", "levofloxacin", "moxifloxacin"],
    "Tetracyclines": ["tetracycline", "doxycycline", "minocycline", "tigecycline"],
    "Glycopeptides": ["vancomycin", "teicoplanin"],
    "Oxazolidinones": ["linezolid", "tedizolid"],
    "Macrolides": ["erythromycin", "azithromycin", "clarithromycin"],
    "Penicillins": ["ampicillin", "amoxicillin", "piperacillin-tazobactam"],
    "Polymyxins": ["colistin", "polymyxin b"],
    "Trimethoprim-sulfonamides": ["trimethoprim", "sulfamethoxazole"],
    "Rifamycins": ["rifampicin", "rifabutin"],
}

# Documented disagreements between the two reference databases. Surfacing these
# is more useful than picking a winner silently — the same argument the project
# already makes about AMRFinder and ResFinder disagreeing on real data.
DB_DISAGREEMENTS: dict[str, str] = {
    "armA": (
        "CARD lists eleven aminoglycosides including gentamicin; NCBI's "
        "subclass field lists gentamicin alone. CARD is biologically right — "
        "armA methylates 16S rRNA at G1405 and blocks essentially the whole "
        "4,6-disubstituted deoxystreptamine group — so we follow CARD here and "
        "treat NCBI's single-drug entry as an under-call."
    ),
    "cfr": (
        "CARD links linezolid only at class level; NCBI names it explicitly. "
        "We follow NCBI, since linezolid resistance is the clinically defining "
        "property of cfr."
    ),
}

# Agents specifically engineered to evade a determinant. Sourced from the drug
# labels, not inferred from database absence — a gene missing from a curated
# list means "not curated", never "susceptible".
ENGINEERED_ESCAPES: dict[str, list[str]] = {
    "aac(6')-Ib": ["plazomicin"],
    "tetM": ["tigecycline"],
    "mecA": ["ceftaroline"],
    "cfr": ["tedizolid"],
}

# Drugs the reference databases barely curate. For these, absence from a gene's
# drug list carries NO information, so they must never be reported as spared on
# that basis — only named where we have positive evidence (an ENGINEERED_ESCAPES
# entry, which comes from the drug label rather than from a database gap).
#
# This is not hypothetical. CARD lists eleven aminoglycosides for armA and
# omits plazomicin, and only ONE gene in the whole of CARD is linked to
# plazomicin at all. Inferring from that absence would have produced the
# confident and flatly wrong claim that "armA spares plazomicin" — 16S rRNA
# methyltransferases defeat plazomicin, which is stated on the drug's own label.
#
# The general rule this encodes: a curated database tells you what IS known,
# never what is false.
POORLY_CURATED_DRUGS = {"plazomicin", "tedizolid", "ceftaroline"}


@dataclass
class DrugResolution:
    """Which individual drugs a detected determinant defeats, and which it spares."""

    gene: str
    antibiotic_class: str
    defeated: list[str] = field(default_factory=list)
    spared: list[str] = field(default_factory=list)
    engineered_escape: list[str] = field(default_factory=list)
    class_level_only: bool = False
    sources: list[str] = field(default_factory=list)
    disagreement: str = ""

    @property
    def resolved(self) -> bool:
        """Did we actually learn anything beyond the class label?"""
        return bool(self.defeated) and not self.class_level_only

    def summary(self) -> str:
        if not self.resolved:
            return (
                f"{self.gene} is linked to the {self.antibiotic_class} class, but "
                "the reference databases do not enumerate individual drugs for "
                "it. We report the class and stop there rather than inventing "
                "per-drug detail."
            )
        out = (
            f"**{self.gene}** defeats "
            f"**{', '.join(self.defeated)}** within {self.antibiotic_class}."
        )
        if self.spared:
            out += (
                f" It does **not** modify **{', '.join(self.spared)}**, which "
                "therefore remains a candidate agent against this isolate — the "
                "detail a class-level call throws away."
            )
        if self.engineered_escape:
            out += (
                f" {', '.join(self.engineered_escape).capitalize()} was "
                "specifically engineered to evade this determinant."
            )
        return out


_CACHE: dict[str, dict[str, list[str]]] | None = None


def _normalise(drug: str) -> str:
    """CARD and NCBI spell the same drug differently; fold them together."""
    d = drug.strip().lower()
    # CARD names variants (gentamicin B / gentamicin C, kanamycin A) that a
    # clinician reports as one agent.
    for base in ("gentamicin", "kanamycin", "colistin"):
        if d.startswith(base):
            return base
    return d


def _load() -> dict[str, dict[str, list[str]]]:
    """gene -> {"drugs": [...], "sources": [...], "class_only": bool}"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    data: dict[str, dict] = {}
    if not _MAPPING_PATH.exists():
        _CACHE = {}
        return _CACHE

    with open(_MAPPING_PATH, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gene = row["gene"].strip()
            drug = row["drug"].strip()
            src = row["source"].strip()
            entry = data.setdefault(
                gene, {"drugs": set(), "sources": set(), "class_only": False}
            )
            if drug == _CLASS_ONLY:
                entry["class_only"] = True
                continue
            entry["drugs"].add(_normalise(drug))
            # Keep the database name, not the whole URL-bearing row.
            entry["sources"].add(src.split("(")[0].strip() or src[:40])

    _CACHE = {
        g: {
            "drugs": sorted(v["drugs"]),
            "sources": sorted(v["sources"]),
            "class_only": v["class_only"],
        }
        for g, v in data.items()
    }
    return _CACHE


def resolve_gene(gene: str, antibiotic_class: str) -> DrugResolution:
    """Per-drug detail for one determinant within one class."""
    table = _load()
    res = DrugResolution(gene=gene, antibiotic_class=antibiotic_class)

    entry = table.get(gene)
    clinical = CLINICAL_DRUGS.get(antibiotic_class, [])
    if entry is None:
        res.class_level_only = True
        return res

    known = set(entry["drugs"])
    res.sources = entry["sources"]
    res.disagreement = DB_DISAGREEMENTS.get(gene, "")

    res.defeated = [d for d in clinical if d in known]
    if not res.defeated:
        # Databases have rows for this gene, but none for a drug in the class
        # we are asking about. Report that honestly.
        res.class_level_only = True
        return res

    escapes = set(ENGINEERED_ESCAPES.get(gene, []))
    # Reaching this line means the databases DO enumerate individual drugs for
    # this gene in this class (otherwise we returned above). So a clinical drug
    # absent from that enumeration is genuinely spared, and saying so is the
    # whole value of this layer — it is how gentamicin survives an aac(6')-Ib
    # call that a class-level report would have written off.
    #
    # A gene may carry BOTH a class-level row and enumerated drug rows; the
    # class-level row only records that the database also links it generically,
    # and must not suppress the detail we do have.
    res.spared = [
        d for d in clinical
        if d not in known
        and d not in escapes
        and d not in POORLY_CURATED_DRUGS  # absence here means "not curated"
    ]
    res.engineered_escape = [d for d in clinical if d in escapes]
    return res


def resolve_detected(
    detected_genes: list[tuple[str, float]],
    antibiotic_class: str,
) -> list[DrugResolution]:
    """Per-drug detail for every detected determinant hitting one class."""
    from .amr import RESISTANCE_GENES

    out = []
    for gene_name, _score in detected_genes:
        gene = RESISTANCE_GENES.get(gene_name)
        if gene is None or antibiotic_class not in gene.confers_resistance_to:
            continue
        out.append(resolve_gene(gene_name, antibiotic_class))
    return out


def class_verdict(resolutions: list[DrugResolution], antibiotic_class: str) -> str:
    """
    One sentence on what is actually left in this class.

    The union of what every detected determinant defeats, against the clinical
    drug list — because two genes can spare different agents, and only the
    intersection of what they both spare genuinely survives.
    """
    clinical = CLINICAL_DRUGS.get(antibiotic_class, [])
    if not clinical or not resolutions:
        return ""

    resolved = [r for r in resolutions if r.resolved]
    if not resolved:
        return (
            f"The databases do not enumerate individual {antibiotic_class} drugs "
            "for the determinants found here, so we report at class level only."
        )

    defeated: set[str] = set()
    escapes: set[str] = set()
    for r in resolved:
        defeated |= set(r.defeated)
        escapes |= set(r.engineered_escape)

    # Same rule as resolve_gene: a poorly-curated drug is only claimed as
    # remaining when we have positive evidence it evades something here, never
    # because a database simply does not mention it.
    remaining = [
        d for d in clinical
        if d not in defeated
        and (d not in POORLY_CURATED_DRUGS or d in escapes)
    ]

    if not remaining:
        return (
            f"Every {antibiotic_class} agent we track is defeated by the "
            f"determinants present ({', '.join(sorted(defeated))}). The class is "
            "genuinely exhausted for this isolate."
        )
    return (
        f"**{', '.join(sorted(defeated))}** are defeated, but "
        f"**{', '.join(remaining)}** {'is' if len(remaining) == 1 else 'are'} not "
        f"covered by any determinant detected — a class-level call would have "
        f"written off the whole of {antibiotic_class} and lost that."
    )


def coverage_report() -> dict:
    """How much of our gene set we can actually resolve, for the Methods tab."""
    from .amr import RESISTANCE_GENES

    table = _load()
    rows = []
    for name, gene in RESISTANCE_GENES.items():
        resolutions = [resolve_gene(name, c) for c in gene.confers_resistance_to]
        rows.append(
            {
                "gene": name,
                "classes": gene.confers_resistance_to,
                "resolved_classes": [r.antibiotic_class for r in resolutions if r.resolved],
                "in_database": name in table,
                "n_drugs": len(table.get(name, {}).get("drugs", [])),
            }
        )
    resolved = [r for r in rows if r["resolved_classes"]]
    return {
        "n_genes": len(rows),
        "n_in_database": sum(1 for r in rows if r["in_database"]),
        "n_resolved": len(resolved),
        "rows": rows,
        "disagreements": DB_DISAGREEMENTS,
    }
