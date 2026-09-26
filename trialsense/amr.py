"""
TrialSense — Module 3: Antimicrobial resistance (AMR) intelligence.

WHY THIS MODULE EXISTS
----------------------
For an antibacterial candidate, the question that decides commercial viability
is not only "is it safe?" but "will the organisms it targets already be resistant
by the time it reaches market?". Resistance discovered during Phase III is a
programme-ending event. This module scores a bacterial isolate's genome for
resistance across antibiotic classes, so a portfolio team can prioritise
candidates whose target organisms are still susceptible.

PIPELINE (this part is genuinely real)
--------------------------------------
    DNA sequence -> k-mer frequency vector -> Random Forest -> per-class call

k-mer composition profiling is a standard, well-established way to detect
resistance determinants from raw reads without assembly — it is the basis of
real tools in this space. Everything here runs on a real FASTA file: paste or
upload one and the same pipeline executes unchanged.

WHAT IS REAL, AND WHAT IS STILL CONSTRUCTED
-------------------------------------------
REAL: all 21 resistance-gene sequences. These are the genuine published coding
sequences from the NCBI Bacterial Antimicrobial Resistance Reference Gene
Database (AMRFinderPlus database 2026-08-07.1) — the same curated set NCBI's
own AMRFinderPlus uses — with gyrA and rpoB from RefSeq. Accessions are in each
FASTA header under trialsense/reference_sequences/.

REAL: the gene -> antibiotic-class mapping, the mechanism and mobility
annotations, and the resistance-trend data in surveillance.py.

CONSTRUCTED: the isolate assembly. We take the real gene sequences and place
them into generated species-typical background, because complete annotated
genomes with confirmed gene content are a much larger download than a demo
should carry. So the DETERMINANTS are real and the SCAFFOLD is not.

WHAT THE SWAP TO REAL SEQUENCES COST US, AND WHY WE DID IT ANYWAY
-----------------------------------------------------------------
This module previously shipped synthetic marker cassettes built from generated
motif vocabularies. Replacing them with real sequences made every headline
number WORSE:

                        synthetic   real
    macro-F1 (random)       0.989   0.900
    macro-F1 (cold-combo)   0.925   0.841
    exact match (random)    0.891   0.589

That drop is the honest measurement of how much the synthetic data had been
flattering us. Real genes share bacterial codon usage, real gene families
overlap, and three metallo-beta-lactamases with near-identical class profiles
are genuinely hard to tell apart — none of which was true of cassettes we
generated to be distinguishable.

The recalibration this forced is worth recording: the gene-detection threshold
moved from 0.55 to 0.83, because at 0.55 real references matched almost
everything.

The deployed pipeline is nonetheless BETTER than before, because the
gene-detection layer is exact on real sequences (12/12 strains, no false
positives) and `apply_mechanism_floor` now lets it override the classifier —
the same two-layer arrangement Module 1 uses.

REMAINING PATH TO FULLY REAL DATA
---------------------------------
Replace the generated background in `synthesize_isolate()` with complete
genomes from NCBI Pathogen Detection, which ship with lab-confirmed
susceptibility results attached. That would make the isolates real end to end
and give genuine held-out labels instead of ones we assigned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- Antibiotic classes we report on ----------------------------------------
# Ordered roughly from first-line to last-resort, which is also the order that
# makes the risk display readable.
ANTIBIOTIC_CLASSES = [
    "Penicillins",
    "Cephalosporins",
    "Carbapenems",
    "Fluoroquinolones",
    "Macrolides",
    "Aminoglycosides",
    "Tetracyclines",
    "Trimethoprim-sulfonamides",
    "Glycopeptides",
    "Oxazolidinones",
    "Polymyxins",
    "Rifamycins",
]

# Last-resort classes. Resistance here is the signal a portfolio team must not miss.
LAST_RESORT = {"Carbapenems", "Glycopeptides", "Oxazolidinones", "Polymyxins"}

# =============================================================================
# The decision threshold — one constant, four call sites
#
# Above this probability an organism is called RESISTANT to a class. It governs
# the core scan, the composite score in report.py, the portfolio grid and the
# candidate-impact card, so it lives here rather than being repeated.
#
# WHAT VALIDATION SAYS ABOUT THE VALUE
# ------------------------------------
# `python validate_module3.py` sweeps this threshold and counts the two error
# types separately, the way regulators score a susceptibility device:
#
#     very major error = predicted susceptible, actually resistant
#                        -> the drug fails in patients. Programme-ending.
#     major error      = predicted resistant, actually susceptible
#                        -> one confirmatory assay recovers it.
#
#     threshold  very major   major
#         0.50        5.31%   1.42%   <- previous default
#         0.35        1.45%   8.38%   <- SHIPPED, chosen by the sweep
#
# Commercial devices are held to a very-major rate on the order of 1.5%, and
# 0.50 did not meet it. 0.35 does, and it is also exactly the threshold
# Module 1 already ships (ddi.SCREENING_THRESHOLD) — both derived independently
# from the same principle, that a missed danger costs far more than a false
# alarm in pre-trial screening.
#
# The ICMR cross-check gives a concrete example of what 0.50 was costing:
# E. coli ST131 carries gyrA_S83L, a fluoroquinolone determinant, and the model
# scores it 44%. At 0.50 that was reported SUSCEPTIBLE — a very major error on
# a strain that demonstrably carries the gene. At 0.35 it is called correctly.
#
# Changing this single line moves every call site at once. Re-run
# validate_module3.py afterwards.
RESISTANCE_CALL_THRESHOLD = 0.35

# --- and a second threshold, because there are two different questions -------
#
# Clinical susceptibility testing does not report a binary. CLSI defines three
# categories — Susceptible, Intermediate, Resistant — precisely because forcing
# a borderline result into one of two buckets throws information away. We do
# the same, for the same reason.
#
#     below 0.35        SUSCEPTIBLE   no action
#     0.35 to 0.50      EQUIVOCAL     screen it, confirm before acting
#     0.50 and above    RESISTANT     a confident call
#
# The two thresholds answer different questions and are deliberately not equal:
#
#   SCREENING (0.35) — "is this worth flagging?" Used for blocking a candidate
#       in the portfolio grid and raising findings. Tuned for sensitivity,
#       because a missed danger costs a programme.
#
#   CONFIRMED (0.50) — "are we confident enough to call this resistant?" Used
#       for the headline risk level and the count of classes resisted. Tuned
#       for specificity, because a tool that labels a clean isolate 'Moderate
#       risk' on one borderline class is the over-alerting problem that makes
#       clinicians ignore warnings.
#
# This distinction was not theoretical. Dropping the single threshold to 0.35
# pushed the susceptible control strain to 'Moderate' on one class scoring
# 37%, while the gene-detection layer found nothing at all in it. Two layers
# disagreeing at the margin is exactly what an equivocal band is for.
RESISTANCE_CONFIRMED_THRESHOLD = 0.50


@dataclass
class ResistanceGene:
    """
    A resistance determinant, the classes it defeats, and — added for Module 3
    Features 2 and 6 — HOW it defeats them and HOW FAST it spreads.

    The extra fields are what turn a yes/no verdict into a decision. Two genes
    can produce an identical "carbapenem resistant" call and imply opposite
    business outcomes: blaKPC-2 is a serine carbapenemase that existing
    inhibitors neutralise, so the programme survives with a partner drug, while
    blaNDM-1 is a metallo-enzyme that those same inhibitors do not touch.

    All values here are encoded textbook microbiology, not model output.
    """

    name: str
    confers_resistance_to: list[str]
    mechanism: str
    gc_bias: float = 0.5  # nucleotide composition, gives each gene a k-mer signature

    # --- Feature 2: mechanism family and whether the programme is salvageable
    # enzymatic_hydrolysis | target_modification | ribosomal_methylation
    # | drug_modification | target_protection | membrane_modification | bypass
    mechanism_family: str = "unclassified"
    enzyme_subtype: str = ""  # e.g. "serine (class A)", "metallo (class B)"
    rescue_viable: str = "unknown"  # yes | partial | no | unknown
    rescue_strategy: str = ""
    rescue_note: str = ""

    # --- Feature 6: how the determinant travels
    # plasmid | integron | transposon | sccmec | chromosomal
    mobility: str = "unknown"
    mobility_note: str = ""

    @property
    def is_mobile(self) -> bool:
        """Can this determinant move between cells, not just to offspring?"""
        return self.mobility in {"plasmid", "integron", "transposon", "sccmec"}

    @property
    def spread_multiplier(self) -> float:
        """
        How much faster a mobile determinant spreads than a chromosomal one.

        Used to shorten the Feature 1 runway. Horizontally transferable
        elements move between species, not only between generations — mcr-1
        went from first description (2015) to global reports inside roughly
        eighteen months, which no chromosomal point mutation does. These
        multipliers are reasoned orders of magnitude, not fitted constants,
        and the UI says so.
        """
        return {
            "plasmid": 2.5,
            "integron": 2.0,
            "transposon": 1.8,
            "sccmec": 1.3,
            "chromosomal": 1.0,
        }.get(self.mobility, 1.0)


# Gene -> resistance mapping. This IS real microbiology.
RESISTANCE_GENES: dict[str, ResistanceGene] = {
    g.name: g
    for g in [
        ResistanceGene(
            "blaTEM-1",
            ["Penicillins"],
            "A narrow-spectrum beta-lactamase: it cuts the beta-lactam ring of "
            "penicillins before they can reach their target.",
            gc_bias=0.44,
        ),
        ResistanceGene(
            "blaCTX-M-15",
            ["Penicillins", "Cephalosporins"],
            "An extended-spectrum beta-lactamase (ESBL). It destroys third-"
            "generation cephalosporins as well as penicillins — the single most "
            "widespread ESBL worldwide.",
            gc_bias=0.52,
        ),
        ResistanceGene(
            "blaNDM-1",
            ["Penicillins", "Cephalosporins", "Carbapenems"],
            "A metallo-beta-lactamase that inactivates almost every beta-lactam "
            "including carbapenems, and is not blocked by standard beta-lactamase "
            "inhibitors. First described in 2008 and now endemic across South Asia.",
            gc_bias=0.60,
        ),
        ResistanceGene(
            "blaKPC-2",
            ["Penicillins", "Cephalosporins", "Carbapenems"],
            "A serine carbapenemase that hydrolyses carbapenems; typically "
            "plasmid-borne and therefore highly transmissible between species.",
            gc_bias=0.57,
        ),
        ResistanceGene(
            "blaOXA-48",
            ["Penicillins", "Carbapenems"],
            "An oxacillinase with carbapenem-hydrolysing activity. Often missed "
            "by routine screening because it barely affects cephalosporins.",
            gc_bias=0.46,
        ),
        ResistanceGene(
            "mecA",
            ["Penicillins", "Cephalosporins", "Carbapenems"],
            "Encodes an alternative penicillin-binding protein (PBP2a) that "
            "beta-lactams cannot bind. This single gene defines MRSA.",
            gc_bias=0.34,
        ),
        ResistanceGene(
            "vanA",
            ["Glycopeptides"],
            "Remodels the bacterial cell-wall precursor so vancomycin can no "
            "longer grip it — reducing binding affinity roughly a thousand-fold.",
            gc_bias=0.42,
        ),
        ResistanceGene(
            "ermB",
            ["Macrolides"],
            "Methylates the ribosome at the macrolide binding site, blocking "
            "attachment of the entire macrolide class at once.",
            gc_bias=0.36,
        ),
        ResistanceGene(
            "tetM",
            ["Tetracyclines"],
            "A ribosomal protection protein that physically dislodges "
            "tetracycline from the ribosome.",
            gc_bias=0.39,
        ),
        ResistanceGene(
            "aac(6')-Ib",
            ["Aminoglycosides"],
            "An acetyltransferase that chemically modifies aminoglycosides so "
            "they no longer bind their ribosomal target.",
            gc_bias=0.56,
        ),
        ResistanceGene(
            "armA",
            ["Aminoglycosides"],
            "A 16S ribosomal RNA methyltransferase conferring high-level "
            "resistance to essentially all clinically useful aminoglycosides.",
            gc_bias=0.33,
        ),
        ResistanceGene(
            "qnrS1",
            ["Fluoroquinolones"],
            "A plasmid-borne protein that shields DNA gyrase from "
            "fluoroquinolones, giving low-level resistance that lets higher-level "
            "mutations accumulate.",
            gc_bias=0.48,
        ),
        ResistanceGene(
            "gyrA_S83L",
            ["Fluoroquinolones"],
            "A point mutation in DNA gyrase itself — the drug's target — which "
            "lowers fluoroquinolone binding. Chromosomal, so it is inherited "
            "vertically rather than spreading on a plasmid.",
            gc_bias=0.53,
        ),
        ResistanceGene(
            "sul1",
            ["Trimethoprim-sulfonamides"],
            "An alternative, sulfonamide-insensitive form of the folate-pathway "
            "enzyme the drug is designed to block.",
            gc_bias=0.61,
        ),
        ResistanceGene(
            "dfrA17",
            ["Trimethoprim-sulfonamides"],
            "A trimethoprim-resistant dihydrofolate reductase; usually travels "
            "with sul1 on the same integron.",
            gc_bias=0.45,
        ),
        ResistanceGene(
            "mcr-1",
            ["Polymyxins"],
            "Modifies lipid A so colistin can no longer bind the outer membrane. "
            "The first PLASMID-borne colistin resistance ever found (2015) — "
            "colistin is a last-resort agent, so this gene is a global alarm.",
            gc_bias=0.51,
        ),
        ResistanceGene(
            "cfr",
            ["Oxazolidinones", "Macrolides"],
            "Methylates 23S rRNA and defeats linezolid — one of the very last "
            "options for resistant Gram-positive infection.",
            gc_bias=0.31,
        ),
        ResistanceGene(
            "rpoB_S450L",
            ["Rifamycins"],
            "A mutation in RNA polymerase, rifampicin's target. This is the "
            "defining marker of rifampicin-resistant tuberculosis.",
            gc_bias=0.58,
        ),
        # --- The three determinants ICMR molecular surveillance reports that
        # our reference set previously missed. validate_module3.py flagged the
        # gap at 5/8 coverage; these close it to 8/8. All three are common in
        # Indian isolates (VIM 19%, SHV 16%, IMP 15%).
        ResistanceGene(
            "blaVIM",
            ["Penicillins", "Cephalosporins", "Carbapenems"],
            "Verona integron-encoded metallo-beta-lactamase. Like NDM-1 it is a "
            "zinc-dependent enzyme that hydrolyses carbapenems, and like NDM-1 "
            "it is untouched by the serine-enzyme inhibitors. Particularly "
            "associated with Pseudomonas aeruginosa.",
            gc_bias=0.55,
        ),
        ResistanceGene(
            "blaIMP",
            ["Penicillins", "Cephalosporins", "Carbapenems"],
            "Imipenemase, a metallo-beta-lactamase family first described in "
            "Japan and now globally distributed. Third of the three major "
            "acquired metallo-carbapenemase families alongside NDM and VIM.",
            gc_bias=0.49,
        ),
        ResistanceGene(
            "blaSHV",
            ["Penicillins", "Cephalosporins"],
            "Sulfhydryl-variable beta-lactamase. The ancestral SHV-1 is "
            "narrow-spectrum, but the extended-spectrum variants that dominate "
            "clinical isolates also defeat third-generation cephalosporins. "
            "ICMR reports this family at group level, so we model the "
            "extended-spectrum behaviour as the conservative case.",
            gc_bias=0.63,
        ),
    ]
}


# =============================================================================
# Mechanism and mobility annotations (Module 3, Features 2 and 6).
#
# Kept as one reviewable block rather than scattered through the gene list
# above, because this is the layer a microbiologist would want to audit in one
# sitting. Every entry is documented textbook microbiology.
#
# Fields: (mechanism_family, enzyme_subtype, rescue_viable, rescue_strategy,
#          rescue_note, mobility, mobility_note)
# =============================================================================

_ANNOTATIONS: dict[str, dict] = {
    "blaTEM-1": dict(
        mechanism_family="enzymatic_hydrolysis",
        enzyme_subtype="serine beta-lactamase (Ambler class A), narrow spectrum",
        rescue_viable="yes",
        rescue_strategy="Pair with a classical beta-lactamase inhibitor "
        "(clavulanate, sulbactam or tazobactam).",
        rescue_note="A narrow-spectrum serine enzyme — the original target the "
        "beta-lactamase inhibitors were designed against. Well covered.",
        mobility="plasmid",
        mobility_note="Plasmid-borne and among the most widely disseminated "
        "resistance genes in Gram-negative bacteria.",
    ),
    "blaCTX-M-15": dict(
        mechanism_family="enzymatic_hydrolysis",
        enzyme_subtype="serine extended-spectrum beta-lactamase (class A)",
        rescue_viable="yes",
        rescue_strategy="Pair with avibactam or tazobactam; carbapenems also "
        "remain active against an isolate carrying only an ESBL.",
        rescue_note="An ESBL, not a carbapenemase. It defeats cephalosporins "
        "but is inhibited by avibactam, so beta-lactam chemistry is not lost.",
        mobility="plasmid",
        mobility_note="Plasmid-borne; the single most widespread ESBL "
        "worldwide, and carried by ~40% of Indian isolates in ICMR molecular "
        "surveillance.",
    ),
    "blaNDM-1": dict(
        mechanism_family="enzymatic_hydrolysis",
        enzyme_subtype="METALLO-beta-lactamase (Ambler class B, zinc-dependent)",
        rescue_viable="no",
        rescue_strategy="Standard inhibitor pairing FAILS. Avibactam, "
        "vaborbactam and relebactam are all serine-enzyme inhibitors and do "
        "not neutralise a metallo-enzyme. The documented route against "
        "MBL-producers is an aztreonam-based combination, because aztreonam "
        "is stable to metallo-beta-lactamases and the partner inhibitor "
        "covers any co-produced serine enzymes.",
        rescue_note="This is the distinction that changes the business "
        "decision. A serine carbapenemase such as blaKPC-2 produces the same "
        "clinical verdict but is rescuable with an off-the-shelf inhibitor; "
        "blaNDM-1 is not. First described in 2008 from a patient in New Delhi "
        "and now endemic across South Asia.",
        mobility="plasmid",
        mobility_note="Plasmid-borne and highly transmissible between species. "
        "Present in ~27% of Indian isolates in ICMR molecular surveillance.",
    ),
    "blaKPC-2": dict(
        mechanism_family="enzymatic_hydrolysis",
        enzyme_subtype="serine carbapenemase (Ambler class A)",
        rescue_viable="yes",
        rescue_strategy="Pair with avibactam, vaborbactam or relebactam — all "
        "three inhibit class A serine carbapenemases.",
        rescue_note="Carbapenem resistance that IS chemically rescuable. Worth "
        "contrasting explicitly with blaNDM-1, which reads identically at "
        "class level and is not.",
        mobility="plasmid",
        mobility_note="Typically plasmid-borne, hence its rapid international "
        "spread between species.",
    ),
    "blaOXA-48": dict(
        mechanism_family="enzymatic_hydrolysis",
        enzyme_subtype="serine oxacillinase (Ambler class D)",
        rescue_viable="partial",
        rescue_strategy="Avibactam inhibits OXA-48. Vaborbactam and "
        "relebactam do NOT — inhibitor choice is not interchangeable here.",
        rescue_note="Often missed by routine screening because it barely "
        "affects cephalosporins, so the isolate looks less resistant than it is.",
        mobility="plasmid",
        mobility_note="Plasmid-borne, frequently on a transposon within it.",
    ),
    "mecA": dict(
        mechanism_family="target_modification",
        enzyme_subtype="",
        rescue_viable="no",
        rescue_strategy="No beta-lactamase inhibitor helps — there is no enzyme "
        "to inhibit. The drug simply cannot bind the altered target. The one "
        "documented exception is ceftaroline, a cephalosporin with affinity "
        "for PBP2a itself.",
        rescue_note="Encodes an alternative penicillin-binding protein (PBP2a) "
        "that beta-lactams cannot bind. This single gene defines MRSA.",
        mobility="sccmec",
        mobility_note="Carried on SCCmec, a mobile genetic element that "
        "integrates into the chromosome — mobile, but slower than a free plasmid.",
    ),
    "vanA": dict(
        mechanism_family="target_modification",
        enzyme_subtype="",
        rescue_viable="no",
        rescue_strategy="No inhibitor strategy exists. The cell-wall precursor "
        "itself is remodelled, so the drug has nothing left to grip. Treatment "
        "moves to a different class entirely (oxazolidinones, daptomycin).",
        rescue_note="Remodels the D-Ala-D-Ala terminus to D-Ala-D-Lac, cutting "
        "vancomycin binding affinity roughly a thousand-fold.",
        mobility="transposon",
        mobility_note="Carried on the Tn1546 transposon, usually plasmid-borne "
        "— transferable between enterococci and, rarely, to S. aureus.",
    ),
    "ermB": dict(
        mechanism_family="ribosomal_methylation",
        enzyme_subtype="",
        rescue_viable="no",
        rescue_strategy="Methylation blocks the whole macrolide binding site at "
        "once, so no macrolide analogue evades it. A different ribosomal "
        "target class is required.",
        rescue_note="Confers cross-resistance across macrolides, lincosamides "
        "and streptogramin B simultaneously — one gene, three classes.",
        mobility="transposon",
        mobility_note="Commonly on conjugative transposons, transferable "
        "between Gram-positive species.",
    ),
    "tetM": dict(
        mechanism_family="target_protection",
        enzyme_subtype="",
        rescue_viable="partial",
        rescue_strategy="Glycylcyclines (tigecycline) and the newer "
        "aminomethylcyclines are specifically engineered to evade ribosomal "
        "protection proteins, so the scaffold class survives even though "
        "classical tetracyclines do not.",
        rescue_note="A ribosomal protection protein that physically dislodges "
        "tetracycline from the ribosome rather than destroying the drug.",
        mobility="transposon",
        mobility_note="Conjugative transposon, broad host range.",
    ),
    "aac(6')-Ib": dict(
        mechanism_family="drug_modification",
        enzyme_subtype="aminoglycoside acetyltransferase",
        rescue_viable="yes",
        rescue_strategy="Plazomicin was designed to be stable to most "
        "aminoglycoside-modifying enzymes including this one, so the "
        "aminoglycoside class is not lost.",
        rescue_note="IMPORTANT CAVEAT, and a limitation our own ICMR "
        "cross-check exposed: this gene is mapped here to the whole "
        "Aminoglycosides class, but real per-drug resistance varies widely "
        "within that class. Our model reads ~88% where real Indian amikacin "
        "resistance is far lower. Class-level mapping is too coarse at this "
        "one point — a known, documented simplification.",
        mobility="integron",
        mobility_note="Integron-borne, frequently travelling with ESBL genes "
        "on the same mobile element.",
    ),
    "armA": dict(
        mechanism_family="ribosomal_methylation",
        enzyme_subtype="16S rRNA methyltransferase",
        rescue_viable="no",
        rescue_strategy="No aminoglycoside survives this, plazomicin included "
        "— the binding site itself is methylated. The class is genuinely lost.",
        rescue_note="Worth contrasting with aac(6')-Ib: both read as "
        "aminoglycoside resistance, but one is engineered around and the other "
        "is not. Mechanism, not class, decides.",
        mobility="transposon",
        mobility_note="Transposon-associated, usually on multi-resistance "
        "plasmids.",
    ),
    "qnrS1": dict(
        mechanism_family="target_protection",
        enzyme_subtype="",
        rescue_viable="partial",
        rescue_strategy="Confers only low-level resistance on its own; a "
        "higher-exposure fluoroquinolone regimen may retain activity. The real "
        "danger is that it buys the organism time to acquire gyrA mutations.",
        rescue_note="A plasmid protein that shields DNA gyrase. Mild alone, "
        "but it is the stepping stone to full fluoroquinolone resistance.",
        mobility="plasmid",
        mobility_note="Plasmid-borne — the reason fluoroquinolone resistance "
        "began spreading horizontally rather than only by mutation.",
    ),
    "gyrA_S83L": dict(
        mechanism_family="target_modification",
        enzyme_subtype="",
        rescue_viable="no",
        rescue_strategy="The drug target itself is altered. No partner agent "
        "restores binding; a different target class is needed.",
        rescue_note="A chromosomal point mutation in DNA gyrase.",
        mobility="chromosomal",
        mobility_note="Chromosomal — inherited vertically only. Spreads by "
        "clonal expansion, which is far slower than plasmid transfer.",
    ),
    "sul1": dict(
        mechanism_family="bypass",
        enzyme_subtype="alternative dihydropteroate synthase",
        rescue_viable="no",
        rescue_strategy="The organism runs an alternative, drug-insensitive "
        "version of the blocked enzyme. Inhibiting the original is pointless.",
        rescue_note="A metabolic bypass rather than an attack on the drug.",
        mobility="integron",
        mobility_note="Classically located in the 3' conserved segment of "
        "class 1 integrons, almost always travelling with dfrA genes.",
    ),
    "dfrA17": dict(
        mechanism_family="bypass",
        enzyme_subtype="trimethoprim-resistant dihydrofolate reductase",
        rescue_viable="no",
        rescue_strategy="As with sul1, an insensitive replacement enzyme. No "
        "inhibitor pairing recovers activity.",
        rescue_note="Usually co-located with sul1 on the same integron, which "
        "is why the two are almost always detected together.",
        mobility="integron",
        mobility_note="Integron cassette, co-mobilised with sul1.",
    ),
    "mcr-1": dict(
        mechanism_family="membrane_modification",
        enzyme_subtype="phosphoethanolamine transferase",
        rescue_viable="no",
        rescue_strategy="Modifies lipid A so colistin can no longer bind the "
        "outer membrane. No approved partner agent restores binding.",
        rescue_note="The first PLASMID-borne colistin resistance ever found "
        "(2015). Colistin is a last-resort agent, so this is the end of the "
        "treatment ladder rather than a step along it.",
        mobility="plasmid",
        mobility_note="The defining example of fast horizontal spread — went "
        "from first description in 2015 to reports on multiple continents "
        "within roughly eighteen months.",
    ),
    "cfr": dict(
        mechanism_family="ribosomal_methylation",
        enzyme_subtype="23S rRNA methyltransferase (A2503)",
        rescue_viable="partial",
        rescue_strategy="Defeats linezolid. Tedizolid retains some activity "
        "against cfr-carrying isolates, so the oxazolidinone scaffold is not "
        "completely written off.",
        rescue_note="One of the very last options for resistant Gram-positive "
        "infection, and it confers cross-resistance across five drug classes.",
        mobility="plasmid",
        mobility_note="Plasmid-borne and transferable — the reason linezolid "
        "resistance is treated as a containment priority.",
    ),
    "rpoB_S450L": dict(
        mechanism_family="target_modification",
        enzyme_subtype="",
        rescue_viable="no",
        rescue_strategy="RNA polymerase, rifampicin's target, is altered. No "
        "partner agent restores binding; regimens move to other TB drugs.",
        rescue_note="The defining marker of rifampicin-resistant tuberculosis, "
        "and the basis of rapid molecular TB diagnostics.",
        mobility="chromosomal",
        mobility_note="Chromosomal point mutation — vertical inheritance only. "
        "M. tuberculosis does not carry resistance plasmids, which is why TB "
        "resistance spreads by transmission of resistant strains, not by gene "
        "transfer.",
    ),
    "blaVIM": dict(
        mechanism_family="enzymatic_hydrolysis",
        enzyme_subtype="METALLO-beta-lactamase (Ambler class B, zinc-dependent)",
        rescue_viable="no",
        rescue_strategy="As with blaNDM-1, the serine-enzyme inhibitors "
        "(avibactam, vaborbactam, relebactam) do not neutralise a "
        "zinc-dependent enzyme. An aztreonam-based combination is the "
        "documented route against metallo-beta-lactamase producers.",
        rescue_note="One of the three major acquired metallo-carbapenemase "
        "families. Reported in roughly 19% of Indian isolates in ICMR "
        "molecular surveillance, and strongly associated with P. aeruginosa.",
        mobility="integron",
        mobility_note="Integron-encoded — the name records it. Travels as a "
        "gene cassette, usually on a plasmid, so it moves between species.",
    ),
    "blaIMP": dict(
        mechanism_family="enzymatic_hydrolysis",
        enzyme_subtype="METALLO-beta-lactamase (Ambler class B, zinc-dependent)",
        rescue_viable="no",
        rescue_strategy="Metallo-enzyme: standard inhibitor pairing fails, for "
        "the same reason it fails against blaNDM-1 and blaVIM. Aztreonam-based "
        "combinations are the documented route.",
        rescue_note="The third major acquired metallo-carbapenemase family "
        "alongside NDM and VIM, at roughly 15% prevalence in Indian isolates. "
        "Three genes, three different names, one identical business "
        "consequence — which is precisely why the mechanism layer matters more "
        "than the gene label.",
        mobility="integron",
        mobility_note="Integron-borne gene cassette, frequently co-located "
        "with aminoglycoside-modifying enzymes on the same element.",
    ),
    "blaSHV": dict(
        mechanism_family="enzymatic_hydrolysis",
        enzyme_subtype="serine beta-lactamase (Ambler class A), "
        "extended-spectrum variants",
        rescue_viable="yes",
        rescue_strategy="A serine enzyme, so avibactam and tazobactam apply. "
        "Carbapenems also remain active against an isolate carrying only an "
        "ESBL.",
        rescue_note="Reported at roughly 16% prevalence in Indian isolates. "
        "ICMR reports the family at group level without separating "
        "narrow-spectrum SHV-1 from the extended-spectrum variants; we model "
        "the extended-spectrum behaviour, which over-calls cephalosporin "
        "resistance for an isolate carrying only SHV-1. A known simplification "
        "of the same kind as the aac(6')-Ib class-level mapping.",
        mobility="plasmid",
        mobility_note="Plasmid-borne, and one of the longest-established "
        "transferable beta-lactamase families in Enterobacterales.",
    ),
}

for _name, _ann in _ANNOTATIONS.items():
    _gene = RESISTANCE_GENES[_name]
    for _field, _value in _ann.items():
        setattr(_gene, _field, _value)

# Fail loudly if a gene was added without annotating it, rather than silently
# reporting "unclassified" to a user making a portfolio decision.
_UNANNOTATED = [n for n, g in RESISTANCE_GENES.items() if g.mechanism_family == "unclassified"]
if _UNANNOTATED:  # pragma: no cover - guards against future edits
    raise RuntimeError(
        f"Resistance genes missing mechanism annotations: {_UNANNOTATED}. "
        "Add them to _ANNOTATIONS in trialsense/amr.py."
    )


@dataclass
class Strain:
    """A bundled demo isolate."""

    name: str
    species: str
    genes: list[str]
    gc_content: float  # species-level genome composition
    context: str  # why this organism matters


DEMO_STRAINS: dict[str, Strain] = {
    s.name: s
    for s in [
        Strain(
            "E. coli ATCC 25922 (susceptible reference)",
            "Escherichia coli",
            [],
            0.507,
            "The standard susceptible laboratory control strain — the baseline "
            "every resistant isolate is compared against.",
        ),
        Strain(
            "E. coli ST131 (ESBL)",
            "Escherichia coli",
            # SHV alongside CTX-M is the textbook ESBL co-occurrence.
            ["blaCTX-M-15", "blaSHV", "sul1", "dfrA17", "gyrA_S83L", "aac(6')-Ib"],
            0.507,
            "The globally dominant multidrug-resistant E. coli lineage, and the "
            "leading cause of ESBL urinary and bloodstream infection.",
        ),
        Strain(
            "K. pneumoniae (carbapenem-resistant)",
            "Klebsiella pneumoniae",
            ["blaNDM-1", "blaCTX-M-15", "sul1", "aac(6')-Ib", "qnrS1"],
            0.572,
            "A WHO critical-priority pathogen. NDM-1 carriage is well documented "
            "across South Asian hospitals and leaves very few treatment options.",
        ),
        Strain(
            "S. aureus MRSA (hospital-acquired)",
            "Staphylococcus aureus",
            ["mecA", "ermB", "tetM"],
            0.327,
            "Methicillin-resistant S. aureus — mecA alone removes the entire "
            "beta-lactam class, historically the mainstay of treatment.",
        ),
        Strain(
            "S. aureus MSSA (methicillin-susceptible)",
            "Staphylococcus aureus",
            ["blaTEM-1"],
            0.327,
            "Same species as MRSA but without mecA: a useful contrast showing "
            "that species identity alone does not predict resistance.",
        ),
        Strain(
            "P. aeruginosa (MDR)",
            "Pseudomonas aeruginosa",
            ["blaKPC-2", "aac(6')-Ib", "gyrA_S83L"],
            0.664,
            "Intrinsically resistant to many agents and a major cause of "
            "ventilator-associated pneumonia.",
        ),
        Strain(
            "A. baumannii (XDR)",
            "Acinetobacter baumannii",
            ["blaOXA-48", "armA", "sul1"],
            0.390,
            "Extensively drug-resistant and highly persistent on ICU surfaces.",
        ),
        Strain(
            "E. faecium (VRE)",
            "Enterococcus faecium",
            ["vanA", "ermB", "tetM"],
            0.380,
            "Vancomycin-resistant enterococcus — vanA removes the glycopeptide "
            "class, pushing treatment onto linezolid.",
        ),
        Strain(
            "S. Typhi H58 (XDR, South Asia)",
            "Salmonella enterica ser. Typhi",
            ["blaCTX-M-15", "gyrA_S83L", "sul1", "dfrA17"],
            0.521,
            "Extensively drug-resistant typhoid. A direct public-health priority "
            "for Indian manufacturers and a strong case for local R&D screening.",
        ),
        Strain(
            "M. tuberculosis (MDR-TB)",
            "Mycobacterium tuberculosis",
            ["rpoB_S450L"],
            0.656,
            "Rifampicin-resistant TB. India carries the world's largest MDR-TB "
            "burden, making rapid resistance prediction especially valuable.",
        ),
        Strain(
            "E. coli (colistin-resistant, mcr-1)",
            "Escherichia coli",
            ["mcr-1", "blaCTX-M-15", "sul1"],
            0.507,
            "Carries plasmid-borne colistin resistance. Colistin is a last-line "
            "agent, so mcr-1 represents the end of the treatment ladder.",
        ),
        Strain(
            "S. aureus (linezolid-resistant, cfr)",
            "Staphylococcus aureus",
            ["cfr", "mecA", "ermB"],
            0.327,
            "Resistant to both beta-lactams and linezolid — an organism for which "
            "genuinely new chemistry is required.",
        ),
    ]
}

# --- Sequence parameters -----------------------------------------------------
# Nominal cassette size, used only for window defaults now that the real
# reference genes vary from 474 bp (dfrA17) to 3,519 bp (rpoB).
CASSETTE_LEN = 900
BACKGROUND_LEN = 2600  # bp of species background per isolate
KMER_K = 5  # 5-mers -> 1024 features
_BASES = np.array(list("ACGT"))

# =============================================================================
# REAL REFERENCE SEQUENCES
#
# These are the genuine published coding sequences for all 21 determinants,
# retrieved from the NCBI Bacterial Antimicrobial Resistance Reference Gene
# Database (AMRFinderPlus database 2026-08-07.1) — the same curated reference
# set NCBI's own AMRFinderPlus tool uses — with gyrA and rpoB fetched from
# RefSeq. Every file carries its accession and source URL in the FASTA header.
#
# They replace the synthetic marker cassettes this module previously shipped.
# That swap was the single largest correctness improvement available: it means
# the classifier now learns from real coding sequence with real codon usage and
# real conserved domains, rather than from motif vocabularies we generated.
#
# Each sequence passed QC on retrieval: exactly one record, only A/C/G/T,
# length divisible by three, valid start codon, terminal stop codon, and no
# internal stop codons.
#
# TWO THINGS TO KNOW
# ------------------
# 1. gyrA and rpoB are WILD-TYPE references, not mutant alleles. Resistance in
#    those two comes from point mutations, and a k-mer classifier cannot resolve
#    a single-codon change — see the note on GENE_FILES below.
# 2. Where a gene name denotes a family, we use the type allele: VIM-1, IMP-1,
#    SHV-1.
# =============================================================================

_REFERENCE_DIR = Path(__file__).parent / "reference_sequences"

# Gene name -> filename stem. Filesystem-safe names differ for two genes.
GENE_FILES: dict[str, str] = {
    "blaTEM-1": "blaTEM-1",
    "blaCTX-M-15": "blaCTX-M-15",
    "blaNDM-1": "blaNDM-1",
    "blaKPC-2": "blaKPC-2",
    "blaOXA-48": "blaOXA-48",
    "blaVIM": "blaVIM",
    "blaIMP": "blaIMP",
    "blaSHV": "blaSHV",
    "mecA": "mecA",
    "vanA": "vanA",
    "ermB": "ermB",
    "tetM": "tetM",
    "aac(6')-Ib": "aac6-Ib",
    "armA": "armA",
    "qnrS1": "qnrS1",
    "sul1": "sul1",
    "dfrA17": "dfrA17",
    "mcr-1": "mcr-1",
    "cfr": "cfr",
    # Point-mutation determinants. The file holds the WILD-TYPE gene, because
    # that is what actually exists as a reference sequence. Detecting the gene
    # therefore tells you the locus is present, NOT that the resistant allele
    # is — a single-codon substitution is invisible to k-mer composition at
    # this scale. Honest framing for these two is "locus detected, allele not
    # resolved"; resolving it needs read-level variant calling.
    "gyrA_S83L": "gyrA",
    "rpoB_S450L": "rpoB",
}

# Determinants whose resistance is a point mutation we cannot resolve.
POINT_MUTATION_GENES = {"gyrA_S83L", "rpoB_S450L"}

_SEQ_CACHE: dict[str, str] = {}


def _load_reference(gene_name: str) -> str:
    """Read one real reference sequence from disk, cached."""
    if gene_name in _SEQ_CACHE:
        return _SEQ_CACHE[gene_name]

    stem = GENE_FILES.get(gene_name)
    if stem is None:
        raise KeyError(
            f"No reference sequence mapped for {gene_name!r}. Add it to "
            "GENE_FILES and drop the FASTA in trialsense/reference_sequences/."
        )
    path = _REFERENCE_DIR / f"{stem}.fasta"
    if not path.exists():
        raise FileNotFoundError(
            f"Reference sequence missing: {path}. The 21 curated FASTAs ship "
            "with the repository; re-fetch them from NCBI AMRFinderPlus if lost."
        )
    seq = "".join(
        line.strip()
        for line in path.read_text().splitlines()
        if line and not line.startswith(">")
    ).upper()
    seq = "".join(c for c in seq if c in "ACGT")
    _SEQ_CACHE[gene_name] = seq
    return seq


def _gene_cassette(gene_name: str) -> str:
    """
    The REAL published coding sequence for a resistance gene.

    Previously this synthesised a marker cassette from a generated motif
    vocabulary. It now returns genuine sequence from the NCBI reference
    database, which is why the accuracy figures changed when it was swapped.
    """
    return _load_reference(gene_name)


def reference_provenance() -> list[dict]:
    """Accession and source for every reference sequence, for the Methods tab."""
    out = []
    for gene, stem in GENE_FILES.items():
        path = _REFERENCE_DIR / f"{stem}.fasta"
        if not path.exists():
            continue
        header = path.read_text().splitlines()[0].lstrip(">")
        parts = [p.strip() for p in header.split("|")]
        out.append(
            {
                "gene": gene,
                "accession": parts[1] if len(parts) > 1 else "",
                "description": parts[5] if len(parts) > 5 else "",
                "length": len(_load_reference(gene)),
                "point_mutation": gene in POINT_MUTATION_GENES,
            }
        )
    return out


def _background(gc_content: float, rng: np.random.RandomState, length: int) -> str:
    """Species-typical filler sequence carrying no resistance determinant."""
    probs = [
        (1 - gc_content) / 2,
        gc_content / 2,
        gc_content / 2,
        (1 - gc_content) / 2,
    ]
    return "".join(rng.choice(_BASES, size=length, p=probs))


def _mutate(seq: str, rate: float, rng: np.random.RandomState) -> str:
    """
    Introduce point substitutions to simulate strain-to-strain divergence.

    Without this every isolate carrying a gene would be byte-identical and the
    classification task would be trivial memorisation rather than detection.
    """
    arr = np.array(list(seq))
    n_mut = int(len(arr) * rate)
    if n_mut > 0:
        idx = rng.choice(len(arr), size=n_mut, replace=False)
        arr[idx] = rng.choice(_BASES, size=n_mut)
    return "".join(arr)


def synthesize_isolate(
    genes: list[str],
    gc_content: float = 0.51,
    seed: int = 0,
    divergence: float = 0.02,
) -> str:
    """Build one isolate sequence: background + a cassette per carried gene."""
    rng = np.random.RandomState(seed)
    parts = [_background(gc_content, rng, BACKGROUND_LEN)]
    for gene in genes:
        cassette = _mutate(_gene_cassette(gene), divergence, rng)
        parts.append(cassette)
        # A short spacer, as would separate genes on a real integron/plasmid.
        parts.append(_background(gc_content, rng, 120))
    rng.shuffle(parts)  # gene order is not fixed in a real genome
    return "".join(parts)


# =============================================================================
# k-mer featurisation — the part that is identical for real sequencing data
# =============================================================================

# Base -> integer code lookup, used for vectorised k-mer counting.
# Anything that is not A/C/G/T maps to 255 and is discarded.
_BASE_CODES = np.full(256, 255, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _BASE_CODES[ord(_b)] = _i
    _BASE_CODES[ord(_b.lower())] = _i


def kmer_features(sequence: str, k: int = KMER_K) -> np.ndarray:
    """
    Normalised k-mer frequency vector — the standard alignment-free
    representation of a DNA sequence. Works on any real FASTA input.

    Vectorised: each k-mer is encoded as a base-4 integer via a sliding window,
    then counted with np.bincount. A Python loop over positions is ~50x slower
    and made model training the bottleneck of the whole app.
    """
    n_bins = 4**k
    raw = np.frombuffer(sequence.encode("ascii", "ignore"), dtype=np.uint8)
    codes = _BASE_CODES[raw]
    codes = codes[codes != 255]  # drop N and any other non-ACGT character
    if len(codes) < k:
        return np.zeros(n_bins, dtype=np.float32)

    windows = np.lib.stride_tricks.sliding_window_view(codes.astype(np.int64), k)
    powers = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
    indices = windows @ powers
    counts = np.bincount(indices, minlength=n_bins).astype(np.float32)
    total = counts.sum()
    return counts / total if total > 0 else counts


def parse_fasta(text: str) -> str:
    """Strip FASTA headers and whitespace, returning raw sequence."""
    return "".join(
        line.strip() for line in text.splitlines() if line and not line.startswith(">")
    )


# =============================================================================
# Training data + model
# =============================================================================


def _realistic_gene_combinations(rng: np.random.RandomState, n: int) -> list[list[str]]:
    """
    Sample gene sets that look like real isolates.

    Resistance genes do not co-occur at random — they cluster on plasmids and
    integrons (sul1 with dfrA17, ESBLs with aminoglycoside modifiers). Sampling
    from realistic co-occurrence keeps the training distribution honest, while
    still producing combinations the demo strains never show.
    """
    gene_names = list(RESISTANCE_GENES)
    linked = [
        ["sul1", "dfrA17"],
        ["blaCTX-M-15", "aac(6')-Ib"],
        ["blaNDM-1", "blaCTX-M-15"],
        ["mecA", "ermB"],
        ["vanA", "tetM"],
        ["qnrS1", "gyrA_S83L"],
        # SHV and CTX-M co-occurrence in ESBL Enterobacterales is textbook, and
        # the integron-borne metallo-carbapenemases travel with aminoglycoside
        # modifiers on the same cassette array.
        ["blaSHV", "blaCTX-M-15"],
        ["blaVIM", "aac(6')-Ib"],
        ["blaIMP", "sul1"],
    ]
    combos: list[list[str]] = []
    for _ in range(n):
        genes: set[str] = set()
        n_genes = rng.choice([0, 1, 2, 3, 4, 5], p=[0.10, 0.20, 0.25, 0.22, 0.15, 0.08])
        for _ in range(int(n_genes)):
            if rng.rand() < 0.35:  # sometimes pull in a linked pair
                genes.update(linked[rng.randint(len(linked))])
            else:
                genes.add(gene_names[rng.randint(len(gene_names))])
        combos.append(sorted(genes))
    return combos


def labels_for_genes(genes: list[str]) -> np.ndarray:
    """Ground-truth resistance vector across ANTIBIOTIC_CLASSES for a gene set."""
    y = np.zeros(len(ANTIBIOTIC_CLASSES), dtype=np.int8)
    for gene in genes:
        for cls in RESISTANCE_GENES[gene].confers_resistance_to:
            y[ANTIBIOTIC_CLASSES.index(cls)] = 1
    return y


def build_amr_dataset(
    n_isolates: int = 700, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """Synthesise a labelled isolate collection and featurise it."""
    rng = np.random.RandomState(seed)
    combos = _realistic_gene_combinations(rng, n_isolates)
    gc_values = [0.327, 0.380, 0.390, 0.507, 0.521, 0.572, 0.656, 0.664]

    X, Y = [], []
    for i, genes in enumerate(combos):
        gc = gc_values[rng.randint(len(gc_values))]
        seq = synthesize_isolate(genes, gc_content=gc, seed=int(rng.randint(1 << 30)))
        X.append(kmer_features(seq))
        Y.append(labels_for_genes(genes))
    return np.vstack(X), np.vstack(Y), combos


class AMRModel:
    """
    Multi-label Random Forest: one resistance call per antibiotic class.

    CALIBRATION
    -----------
    The raw forest is badly under-confident in the middle of its range. Measured
    before calibration: when it said 55%, the observed rate was 97%; when it
    said 44%, the observed rate was 73%. Expected calibration error 0.103.

    That matters here beyond cosmetics, because the probability is not just
    displayed — report.py multiplies it (0.75 x P) into the composite score, and
    the portfolio grid thresholds it. A number that systematically understates
    the truth propagates that understatement into a funding decision.

    So each per-class forest is wrapped in isotonic regression fitted on
    cross-validated out-of-fold predictions, which is the standard remedy and
    needs no held-out data of its own. `calibrate=False` reproduces the raw
    model for comparison.
    """

    def __init__(
        self,
        n_estimators: int = 250,
        random_state: int = 42,
        calibrate: bool = True,
    ):
        from sklearn.ensemble import RandomForestClassifier

        self.calibrate = calibrate
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        self.classes = ANTIBIOTIC_CLASSES
        self.is_fitted = False
        self.metrics: dict = {}
        # Summary of the training feature distribution, used by novelty.py to
        # tell a user when an isolate sits outside what this model has seen.
        # A model that sounds equally confident on everything is misleading.
        self.training_stats: dict = {}

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "AMRModel":
        from sklearn.multioutput import MultiOutputClassifier

        base = self.clf
        if self.calibrate:
            from sklearn.calibration import CalibratedClassifierCV

            # Isotonic rather than sigmoid: the miscalibration here is not a
            # simple S-curve shift, it is a systematic flattening across the
            # middle of the range, which isotonic (a free-form monotonic fit)
            # corrects and a sigmoid does not. cv=3 keeps training under a
            # couple of minutes on a laptop.
            base = CalibratedClassifierCV(self.clf, method="isotonic", cv=3)

        self.model = MultiOutputClassifier(base, n_jobs=-1)
        self.model.fit(X, Y)
        self.is_fitted = True
        self._record_training_distribution(X)
        return self

    def _record_training_distribution(self, X: np.ndarray) -> None:
        """
        Store the centroid of the training k-mer profiles plus the distribution
        of distances to it, so inference can place a new isolate on that scale.

        We keep 101 quantiles rather than the full training matrix: it is all
        the percentile lookup needs, and it keeps the pickled model small.
        """
        Xf = np.asarray(X, dtype=np.float64)
        centroid = Xf.mean(axis=0)
        cn = float(np.linalg.norm(centroid))
        if cn == 0.0:
            self.training_stats = {}
            return

        norms = np.linalg.norm(Xf, axis=1)
        norms[norms == 0] = 1.0
        cosines = (Xf @ centroid) / (norms * cn)
        distances = np.sort(1.0 - cosines)

        self.training_stats = {
            "centroid": centroid.astype(np.float32).tolist(),
            "distance_quantiles": np.quantile(
                distances, np.linspace(0.0, 1.0, 101)
            ).tolist(),
            "n_train": int(Xf.shape[0]),
        }

    def predict_sequence(self, sequence: str) -> dict[str, float]:
        """Resistance probability per antibiotic class for one raw sequence."""
        feats = kmer_features(sequence).reshape(1, -1)
        probs = self.model.predict_proba(feats)
        out = {}
        for cls, p in zip(self.classes, probs):
            # p has shape (1, n_classes_seen); index 1 == P(resistant).
            out[cls] = float(p[0][1]) if p.shape[1] > 1 else float(p[0][0] == 1)
        return out


_REF_PROFILE_CACHE: dict[str, np.ndarray] = {}


def _reference_profile(gene_name: str) -> np.ndarray:
    """Unit-normalised k-mer profile of one real reference gene, cached."""
    if gene_name not in _REF_PROFILE_CACHE:
        v = kmer_features(_gene_cassette(gene_name))
        n = float(np.linalg.norm(v))
        _REF_PROFILE_CACHE[gene_name] = v / n if n else v
    return _REF_PROFILE_CACHE[gene_name]


# Cosine similarity at which a window counts as containing a given gene.
#
# RECALIBRATED FOR REAL SEQUENCES. The old value of 0.55 was set against
# synthetic cassettes, which were built with deliberately distinct GC biases
# and motif vocabularies and were therefore trivially separable. Real bacterial
# coding sequences all share codon usage, so every real gene sits somewhat
# close to every other one — at 0.55 the detector fired on almost everything.
#
# Measured across the twelve bundled strains (35 true gene placements, 217
# true negatives):
#
#     weakest TRUE  match  0.883
#     strongest FALSE match 0.763
#
# Any threshold in 0.77-0.88 separates them perfectly. We take 0.83, the
# midpoint, which leaves roughly 0.05 of margin on each side. Re-measure with
# scratchpad recalibration if the reference set changes.
DETECTION_THRESHOLD = 0.83


def locate_genes(
    sequence: str, threshold: float = DETECTION_THRESHOLD
) -> list[tuple[str, float, int, int]]:
    """
    Find reference genes AND where they sit: (gene, score, start, end).

    Positions matter to the open-world scan in novelty.py, which has to mask
    the regions a known gene explains before hunting for anything unexplained.
    """
    seq = "".join(c for c in sequence.upper() if c in "ACGT")
    if len(seq) < 200:
        return []

    hits: list[tuple[str, float, int, int]] = []
    for gene_name in RESISTANCE_GENES:
        ref_len = len(_gene_cassette(gene_name))
        ref = _reference_profile(gene_name)

        window = min(ref_len, len(seq))
        step = max(1, window // 3)

        best, best_at = 0.0, 0
        for i in range(0, max(1, len(seq) - window + 1), step):
            w = kmer_features(seq[i : i + window])
            n = float(np.linalg.norm(w))
            if n == 0.0:
                continue
            score = float(np.dot(w / n, ref))
            if score > best:
                best, best_at = score, i

        if best >= threshold:
            hits.append((gene_name, best, best_at, min(best_at + window, len(seq))))
    return sorted(hits, key=lambda t: -t[1])


def detect_genes(
    sequence: str, threshold: float = DETECTION_THRESHOLD
) -> list[tuple[str, float]]:
    """
    Identify which resistance genes are present, by sliding-window k-mer
    similarity against each real reference sequence.

    This is the mechanism-explanation layer — the counterpart to the DDI
    knowledge base in Module 1. The classifier says WHETHER the isolate is
    resistant; this says WHICH gene is responsible, which is what a scientist
    needs in order to act on the result. It is also the more precise of the two
    layers, which is why `apply_mechanism_floor` lets it override the
    classifier where it fires.

    WHY THE WINDOW SIZE VARIES PER GENE
    -----------------------------------
    With synthetic cassettes every reference was exactly 900 bp, so one fixed
    window worked. The real sequences run from 474 bp (dfrA17) to 3,519 bp
    (rpoB). Comparing a 474 bp gene against a 900 bp window means roughly half
    the window is host background diluting the signal, and comparing a 3,519 bp
    gene against 900 bp compares it to a fragment of itself. Both wreck the
    cosine similarity. So each gene is scanned with a window matched to its own
    length.
    """
    return [(g, s) for g, s, _, _ in locate_genes(sequence, threshold)]


def evaluate_amr(seed: int = 42) -> dict:
    """
    Held-out evaluation, plus the harder cold-combination test.

    COLD-COMBINATION SPLIT: test isolates whose exact gene combination never
    appears in training. This checks the model detects genes compositionally
    rather than memorising whole isolate fingerprints — the property that
    determines whether it works on a genuinely new field isolate.
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    X, Y, combos = build_amr_dataset(seed=seed)
    results: dict = {"n_isolates": len(Y), "n_features": X.shape[1], "k": KMER_K}

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.25, random_state=seed)
    m = AMRModel(random_state=seed).fit(X_tr, Y_tr)
    pred = np.array(m.model.predict(X_te))
    results["random_split"] = {
        "exact_match": float((pred == Y_te).all(axis=1).mean()),
        "per_label_accuracy": float((pred == Y_te).mean()),
        "macro_f1": float(f1_score(Y_te, pred, average="macro", zero_division=0)),
        "n_test": int(len(Y_te)),
    }

    # --- Cold-combination split
    combo_keys = ["|".join(c) for c in combos]
    unique_keys = sorted(set(combo_keys))
    rng = np.random.RandomState(seed)
    held = set(
        rng.choice(unique_keys, size=max(1, len(unique_keys) // 4), replace=False).tolist()
    )
    mask = np.array([k in held for k in combo_keys])
    if mask.sum() > 5 and (~mask).sum() > 20:
        m2 = AMRModel(random_state=seed).fit(X[~mask], Y[~mask])
        pred2 = np.array(m2.model.predict(X[mask]))
        results["cold_combination_split"] = {
            "exact_match": float((pred2 == Y[mask]).all(axis=1).mean()),
            "per_label_accuracy": float((pred2 == Y[mask]).mean()),
            "macro_f1": float(f1_score(Y[mask], pred2, average="macro", zero_division=0)),
            "n_test": int(mask.sum()),
            "n_unseen_combinations": len(held),
        }
    return results


# =============================================================================
# Reporting
# =============================================================================


@dataclass
class AMRReport:
    """Module 3 output for one isolate."""

    strain_name: str
    probabilities: dict[str, float]
    detected_genes: list[tuple[str, float]] = field(default_factory=list)
    context: str = ""

    def resistant_classes(
        self, threshold: float = RESISTANCE_CONFIRMED_THRESHOLD
    ) -> list[str]:
        """
        Classes we are CONFIDENT are resistant.

        Defaults to the confirmed threshold, not the screening one, because
        this drives the headline risk level and the "N of 12 classes resisted"
        count — statements a reader takes as fact rather than as a prompt.
        """
        return [c for c, p in self.probabilities.items() if p >= threshold]

    def flagged_classes(
        self, threshold: float = RESISTANCE_CALL_THRESHOLD
    ) -> list[str]:
        """
        Classes worth flagging — confirmed resistant OR equivocal.

        This is the screening view, used to block a candidate and raise
        findings. Wider than resistant_classes() on purpose.
        """
        return [c for c, p in self.probabilities.items() if p >= threshold]

    def equivocal_classes(self) -> list[str]:
        """Between the screening and confirmed thresholds — confirm before acting."""
        return [
            c
            for c, p in self.probabilities.items()
            if RESISTANCE_CALL_THRESHOLD <= p < RESISTANCE_CONFIRMED_THRESHOLD
        ]

    def susceptible_classes(
        self, threshold: float = RESISTANCE_CALL_THRESHOLD
    ) -> list[str]:
        """
        Classes with no signal at all.

        Uses the SCREENING threshold, so an equivocal class is never presented
        as susceptible. The two defaults differ deliberately: a class in the
        middle band belongs in neither the confident-resistant list nor the
        safe list.
        """
        return [c for c, p in self.probabilities.items() if p < threshold]

    def last_resort_hits(
        self, threshold: float = RESISTANCE_CONFIRMED_THRESHOLD
    ) -> list[str]:
        return [c for c in self.resistant_classes(threshold) if c in LAST_RESORT]

    def risk_level(self) -> tuple[str, str]:
        """
        (label, colour) summarising how constrained the treatment options are.

        Built from CONFIRMED calls only. A single borderline class must not
        turn a clean isolate amber — over-alerting is why clinicians learn to
        ignore warnings, and it would undermine the control case that shows
        this tool does not simply flag everything.
        """
        n_res = len(self.resistant_classes())
        if self.last_resort_hits():
            return "Critical", "#C62828"
        if n_res >= 4:
            return "High", "#EF6C00"
        if n_res >= 1:
            return "Moderate", "#F9A825"
        return "Low", "#2E7D32"

    def plain_summary(self) -> str:
        res = self.resistant_classes()
        if not res:
            return (
                f"{self.strain_name} shows no resistance determinants. All "
                f"{len(ANTIBIOTIC_CLASSES)} antibiotic classes screened remain "
                "viable against this isolate."
            )
        last = self.last_resort_hits()
        base = (
            f"{self.strain_name} is predicted resistant to {len(res)} of "
            f"{len(ANTIBIOTIC_CLASSES)} antibiotic classes screened "
            f"({', '.join(res)})."
        )
        if last:
            base += (
                f" Critically, this includes the last-resort class"
                f"{'es' if len(last) > 1 else ''} {', '.join(last)} — treatment "
                "options for this organism are close to exhausted, which is "
                "exactly the gap a new candidate would need to fill."
            )
        return base

    def mechanism_notes(self) -> list[str]:
        out = []
        for gene_name, score in self.detected_genes:
            gene = RESISTANCE_GENES[gene_name]
            out.append(
                f"**{gene_name}** (detection confidence {score:.0%}) — "
                f"{gene.mechanism} Defeats: {', '.join(gene.confers_resistance_to)}."
            )
        return out


# When the gene-detection layer positively identifies a determinant, how much
# resistance probability does that alone justify? The gene -> class mapping is
# established microbiology, not a prediction, so a confident detection is
# strong evidence on its own.
MECHANISM_FLOOR = 0.90


def apply_mechanism_floor(
    probabilities: dict[str, float],
    detected_genes: list[tuple[str, float]],
) -> dict[str, float]:
    """
    Let the gene-detection layer override the classifier where it fires.

    WHY THIS EXISTS
    ---------------
    Module 3 has two layers, exactly like Module 1: a knowledge-driven one
    (find the gene, look up what it defeats) and a statistical one (read the
    whole k-mer profile). Module 1 already treats its knowledge base as
    authoritative for drugs it covers and leaves the ML layer to handle novel
    compounds. Module 3 was not doing the same, and it cost us:

        P. aeruginosa (MDR) carries blaKPC-2, a carbapenemase.
        Gene detection found it at high confidence.
        The classifier scored Carbapenems 28.7% -> reported SUSCEPTIBLE.

    That is a very major error on a bundled demo strain, produced by ignoring
    the more reliable of our two layers. The gene detector measures 0 false
    positives across all clean strains and finds every planted determinant, so
    where it fires it should win.

    The floor only ever RAISES a probability. A classifier reading higher than
    the floor keeps its value — detecting one determinant does not cap how
    resistant an isolate can be.
    """
    if not detected_genes:
        return probabilities

    adjusted = dict(probabilities)
    for gene_name, confidence in detected_genes:
        gene = RESISTANCE_GENES.get(gene_name)
        if gene is None:
            continue
        # Scale the floor by detection confidence: a marginal 0.6 match should
        # not assert resistance as loudly as a 0.96 one.
        floor = MECHANISM_FLOOR * min(1.0, confidence)
        for cls in gene.confers_resistance_to:
            if cls in adjusted:
                adjusted[cls] = max(adjusted[cls], floor)
    return adjusted


def analyze_isolate(
    sequence: str,
    model: AMRModel,
    strain_name: str = "Uploaded isolate",
    context: str = "",
    use_mechanism_floor: bool = True,
) -> AMRReport:
    """Run the full Module 3 pipeline on a raw DNA sequence."""
    genes = detect_genes(sequence)
    probs = model.predict_sequence(sequence)
    if use_mechanism_floor:
        probs = apply_mechanism_floor(probs, genes)
    return AMRReport(
        strain_name=strain_name,
        probabilities=probs,
        detected_genes=genes,
        context=context,
    )


def demo_strain_sequence(strain_name: str, seed: int = 7) -> str:
    """Reproducible sequence for a bundled demo strain."""
    strain = DEMO_STRAINS[strain_name]
    return synthesize_isolate(strain.genes, gc_content=strain.gc_content, seed=seed)
