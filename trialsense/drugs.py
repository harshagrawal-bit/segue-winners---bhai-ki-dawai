"""
TrialSense — Drug knowledge base.

WHAT IS REAL HERE
-----------------
  * SMILES strings are the real, published structures for each compound. Every
    entry carries `mw_ref` (reference molecular weight from the literature);
    `validate_drug_db()` recomputes MW with RDKit and fails loudly on a mismatch,
    so a typo in a SMILES string cannot silently reach the demo.
  * The pharmacology profile (CYP450 substrate/inhibitor/inducer roles, P-gp
    handling, `fe` = fraction of an absorbed dose excreted unchanged in urine,
    and the risk flags) is drawn from standard clinical pharmacology references.
    These are approximate, rounded, textbook-level values chosen because they are
    stable and widely agreed on — not patient-specific or dose-specific.

WHAT IS SIMPLIFIED
------------------
  * `fe` is a single scalar per drug. Real fraction-excreted values vary with
    dose, formulation and genotype.
  * Risk flags are booleans. Reality is a continuum.
  * The set is 38 drugs chosen for demo coverage, not an exhaustive formulary.

This module holds NO interaction labels — those live in `ddi.py`. Keeping the
per-drug facts separate from the pairwise outcomes is what lets the ML model in
`ddi.py` be trained on structure and evaluated honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- CYP450 enzymes we track -------------------------------------------------
# These account for the large majority of clinically significant oxidative
# drug metabolism.
CYP_ENZYMES = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4", "CYP2C8"]

# Human-readable names for the risk flags, used to build plain-English output.
FLAG_LABELS = {
    "qt_prolonging": "prolongs the cardiac QT interval",
    "bleeding_risk": "increases bleeding risk",
    "nephrotoxic": "can injure the kidney",
    "hepatotoxic": "can injure the liver",
    "serotonergic": "raises serotonin activity",
    "cns_depressant": "causes sedation / CNS depression",
    "potassium_raising": "raises serum potassium",
    "potassium_lowering": "lowers serum potassium",
    "myopathy_risk": "can cause muscle injury (myopathy)",
    "narrow_therapeutic_index": "has a narrow therapeutic window",
    "seizure_risk": "lowers the seizure threshold",
    "antifolate": "blocks the folate pathway",
}


@dataclass
class Drug:
    """One compound plus the pharmacology needed by all three modules."""

    name: str
    smiles: str
    drug_class: str
    mw_ref: float  # literature molecular weight, used to validate the SMILES

    # --- Metabolism / elimination ---
    # fe = fraction of absorbed drug excreted UNCHANGED by the kidney (0-1).
    # (1 - fe) is treated as the hepatically-cleared fraction. This split drives
    # the organ-impairment maths in pk.py.
    fe: float = 0.0
    cyp_substrate: list[str] = field(default_factory=list)
    cyp_inhibitor: list[str] = field(default_factory=list)
    cyp_inducer: list[str] = field(default_factory=list)
    pgp_substrate: bool = False
    pgp_inhibitor: bool = False

    # --- Pharmacodynamic risk flags ---
    qt_prolonging: bool = False
    bleeding_risk: bool = False
    nephrotoxic: bool = False
    hepatotoxic: bool = False
    serotonergic: bool = False
    cns_depressant: bool = False
    potassium_raising: bool = False
    potassium_lowering: bool = False
    myopathy_risk: bool = False
    narrow_therapeutic_index: bool = False
    seizure_risk: bool = False
    # Blocks folate metabolism. Two antifolates together deepen the same
    # block; dangerous when one of them has a narrow safety margin.
    antifolate: bool = False

    # Is this an antibacterial? Links Module 1 to Module 3.
    antibiotic_class: str | None = None

    note: str = ""

    def flags(self) -> list[str]:
        """Names of every risk flag set True on this drug."""
        return [f for f in FLAG_LABELS if getattr(self, f)]

    @property
    def fh(self) -> float:
        """Fraction cleared hepatically (metabolised), as the complement of fe."""
        return round(1.0 - self.fe, 3)


# =============================================================================
# The knowledge base.
#
# Chosen to cover: anticoagulants, statins, antiarrhythmics, the big CYP
# perpetrators (inhibitors + inducers), renally-cleared drugs, psychotropics,
# and a broad antibiotic panel that ties into the AMR module.
# =============================================================================

_DRUG_LIST: list[Drug] = [
    # ------------------------------------------- Anticoagulant / antiplatelet
    Drug(
        name="Warfarin",
        smiles="CC(=O)CC(c1ccccc1)C1=C(O)c2ccccc2OC1=O",
        drug_class="Vitamin K antagonist (anticoagulant)",
        mw_ref=308.33,
        fe=0.01,
        cyp_substrate=["CYP2C9", "CYP3A4", "CYP1A2"],
        bleeding_risk=True,
        narrow_therapeutic_index=True,
        note="The classic narrow-therapeutic-index drug. S-warfarin is cleared "
        "almost entirely by CYP2C9, so 2C9 inhibitors raise INR sharply.",
    ),
    Drug(
        name="Clopidogrel",
        smiles="COC(=O)C(c1ccccc1Cl)N1CCc2sccc2C1",
        drug_class="P2Y12 antiplatelet (prodrug)",
        mw_ref=321.82,
        fe=0.0,
        cyp_substrate=["CYP2C19", "CYP3A4"],
        bleeding_risk=True,
        note="A prodrug: CYP2C19 is required to ACTIVATE it. 2C19 inhibitors "
        "therefore reduce its effect rather than increasing it.",
    ),
    Drug(
        name="Aspirin",
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        drug_class="NSAID / irreversible COX inhibitor",
        mw_ref=180.16,
        fe=0.05,
        bleeding_risk=True,
        nephrotoxic=True,
        note="Irreversible platelet COX-1 inhibition; additive bleeding with any "
        "other antithrombotic.",
    ),
    Drug(
        name="Ibuprofen",
        smiles="CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        drug_class="NSAID",
        mw_ref=206.28,
        fe=0.01,
        cyp_substrate=["CYP2C9"],
        bleeding_risk=True,
        nephrotoxic=True,
        potassium_raising=True,
        note="Constricts the afferent renal arteriole — a major contributor to "
        "the NSAID + ACE-inhibitor + diuretic 'triple whammy'.",
    ),
    # ------------------------------------------------------------ Statins
    Drug(
        name="Simvastatin",
        smiles="CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12",
        drug_class="HMG-CoA reductase inhibitor (statin)",
        mw_ref=418.57,
        fe=0.01,
        cyp_substrate=["CYP3A4"],
        pgp_substrate=True,
        myopathy_risk=True,
        hepatotoxic=True,
        note="Very high first-pass CYP3A4 metabolism, so 3A4 inhibition raises "
        "exposure many-fold and can precipitate rhabdomyolysis.",
    ),
    Drug(
        name="Atorvastatin",
        smiles="CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
        drug_class="HMG-CoA reductase inhibitor (statin)",
        mw_ref=558.64,
        fe=0.02,
        cyp_substrate=["CYP3A4"],
        pgp_substrate=True,
        myopathy_risk=True,
        hepatotoxic=True,
        note="Also 3A4-cleared, but less dependent on it than simvastatin.",
    ),
    # ------------------------------------------------------------ Cardiac
    Drug(
        name="Amiodarone",
        smiles="CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1",
        drug_class="Class III antiarrhythmic",
        mw_ref=645.31,
        fe=0.0,
        cyp_substrate=["CYP3A4", "CYP2C8"],
        cyp_inhibitor=["CYP2C9", "CYP2D6", "CYP3A4"],
        pgp_inhibitor=True,
        qt_prolonging=True,
        hepatotoxic=True,
        narrow_therapeutic_index=True,
        note="Inhibits several CYPs AND P-gp, and has a half-life measured in "
        "weeks — interactions persist long after stopping.",
    ),
    Drug(
        name="Digoxin",
        smiles="CC1OC(OC2C(O)CC(OC3C(O)CC(OC4CCC5(C)C(CCC6C5CC(O)C5(C)C(C7=CC(=O)OC7)CCC65O)C4)OC3C)OC2C)CC(O)C1O",
        drug_class="Cardiac glycoside",
        mw_ref=780.94,
        fe=0.70,
        pgp_substrate=True,
        narrow_therapeutic_index=True,
        note="Renally cleared AND a P-gp substrate — hit by both kidney "
        "impairment and P-gp inhibitors.",
    ),
    Drug(
        name="Verapamil",
        smiles="COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC",
        drug_class="Non-dihydropyridine calcium channel blocker",
        mw_ref=454.60,
        fe=0.03,
        cyp_substrate=["CYP3A4"],
        cyp_inhibitor=["CYP3A4"],
        pgp_inhibitor=True,
        note="Both a substrate and an inhibitor of CYP3A4, plus a P-gp inhibitor.",
    ),
    Drug(
        name="Diltiazem",
        smiles="COc1ccc(C2Sc3ccccc3N(CCN(C)C)C(=O)C2OC(C)=O)cc1",
        drug_class="Non-dihydropyridine calcium channel blocker",
        mw_ref=414.52,
        fe=0.04,
        cyp_substrate=["CYP3A4"],
        cyp_inhibitor=["CYP3A4"],
        pgp_inhibitor=True,
    ),
    Drug(
        name="Lisinopril",
        smiles="NCCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O",
        drug_class="ACE inhibitor",
        mw_ref=405.49,
        fe=1.0,
        potassium_raising=True,
        note="Not metabolised at all — cleared entirely unchanged by the kidney, "
        "so renal impairment translates directly into higher exposure.",
    ),
    Drug(
        name="Spironolactone",
        smiles="CC(=O)SC1CC2=CC(=O)CCC2(C)C2CCC3(C)C(CCC34CCC(=O)O4)C12",
        drug_class="Potassium-sparing diuretic / MR antagonist",
        mw_ref=416.57,
        fe=0.0,
        cyp_substrate=["CYP3A4"],
        potassium_raising=True,
    ),
    Drug(
        name="Furosemide",
        smiles="NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl",
        drug_class="Loop diuretic",
        mw_ref=330.74,
        fe=0.60,
        potassium_lowering=True,
        nephrotoxic=True,
        note="Potassium loss from loop diuretics amplifies the arrhythmia risk of "
        "QT-prolonging drugs.",
    ),
    # ------------------------------------------------- CYP perpetrators
    Drug(
        name="Fluconazole",
        smiles="OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
        drug_class="Triazole antifungal",
        mw_ref=306.27,
        fe=0.80,
        cyp_inhibitor=["CYP2C9", "CYP2C19", "CYP3A4"],
        qt_prolonging=True,
        hepatotoxic=True,
        note="A strong CYP2C9 inhibitor — the textbook cause of warfarin INR spikes.",
    ),
    Drug(
        name="Ketoconazole",
        smiles="CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1",
        drug_class="Imidazole antifungal",
        mw_ref=531.43,
        fe=0.03,
        cyp_inhibitor=["CYP3A4"],
        cyp_substrate=["CYP3A4"],
        qt_prolonging=True,
        hepatotoxic=True,
        note="The reference strong CYP3A4 inhibitor used in regulatory "
        "interaction studies.",
    ),
    Drug(
        name="Rifampicin",
        smiles="CO[C@H]1/C=C/O[C@@]2(C)Oc3c(C)c(O)c4c(O)c(/C=N/N5CCN(C)CC5)c(NC(=O)/C(C)=C\\C=C\\[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C)c(O)c4c3C2=O",
        drug_class="Rifamycin antibiotic (antitubercular)",
        mw_ref=822.94,
        fe=0.10,
        cyp_inducer=["CYP3A4", "CYP2C9", "CYP2C19", "CYP1A2"],
        hepatotoxic=True,
        antibiotic_class="Rifamycins",
        note="The most powerful enzyme INDUCER in common use — it lowers the "
        "exposure of co-administered drugs, causing therapeutic failure rather "
        "than toxicity. Central to TB regimens, hence highly relevant in India.",
    ),
    Drug(
        name="Carbamazepine",
        smiles="NC(=O)N1c2ccccc2C=Cc2ccccc21",
        drug_class="Antiepileptic / sodium channel blocker",
        mw_ref=236.27,
        fe=0.03,
        cyp_substrate=["CYP3A4"],
        cyp_inducer=["CYP3A4", "CYP2C9", "CYP1A2"],
        hepatotoxic=True,
        narrow_therapeutic_index=True,
        note="Induces the very enzyme that clears it (auto-induction).",
    ),
    Drug(
        name="Phenytoin",
        smiles="O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",
        drug_class="Antiepileptic / sodium channel blocker",
        mw_ref=252.27,
        fe=0.05,
        cyp_substrate=["CYP2C9", "CYP2C19"],
        cyp_inducer=["CYP3A4", "CYP2C9"],
        hepatotoxic=True,
        narrow_therapeutic_index=True,
        note="Saturable (non-linear) kinetics — small dose changes can cause "
        "large, disproportionate exposure changes.",
    ),
    Drug(
        name="Omeprazole",
        smiles="COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
        drug_class="Proton pump inhibitor",
        mw_ref=345.42,
        fe=0.0,
        cyp_substrate=["CYP2C19", "CYP3A4"],
        cyp_inhibitor=["CYP2C19"],
    ),
    # ------------------------------------------------- CNS / psychotropic
    Drug(
        name="Fluoxetine",
        smiles="CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
        drug_class="SSRI antidepressant",
        mw_ref=309.33,
        fe=0.03,
        cyp_substrate=["CYP2D6"],
        cyp_inhibitor=["CYP2D6", "CYP2C19"],
        serotonergic=True,
        bleeding_risk=True,
        note="A strong CYP2D6 inhibitor with an exceptionally long half-life; "
        "SSRIs also impair platelet serotonin uptake, adding bleeding risk.",
    ),
    Drug(
        name="Sertraline",
        smiles="CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
        drug_class="SSRI antidepressant",
        mw_ref=306.23,
        fe=0.0,
        cyp_substrate=["CYP2D6", "CYP3A4"],
        cyp_inhibitor=["CYP2D6"],
        serotonergic=True,
        bleeding_risk=True,
    ),
    Drug(
        name="Tramadol",
        smiles="COc1cccc(C2(O)CCCCC2CN(C)C)c1",
        drug_class="Opioid analgesic (prodrug)",
        mw_ref=263.38,
        fe=0.30,
        cyp_substrate=["CYP2D6", "CYP3A4"],
        serotonergic=True,
        cns_depressant=True,
        seizure_risk=True,
        note="Needs CYP2D6 to form its active metabolite; also serotonergic and "
        "lowers the seizure threshold.",
    ),
    Drug(
        name="Diazepam",
        smiles="CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O",
        drug_class="Benzodiazepine",
        mw_ref=284.74,
        fe=0.0,
        cyp_substrate=["CYP3A4", "CYP2C19"],
        cns_depressant=True,
    ),
    Drug(
        name="Midazolam",
        smiles="Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2",
        drug_class="Benzodiazepine",
        mw_ref=325.77,
        fe=0.0,
        cyp_substrate=["CYP3A4"],
        cns_depressant=True,
        note="The regulatory probe substrate for CYP3A4 activity.",
    ),
    Drug(
        name="Theophylline",
        smiles="Cn1c(=O)c2[nH]cnc2n(C)c1=O",
        drug_class="Methylxanthine bronchodilator",
        mw_ref=180.16,
        fe=0.10,
        cyp_substrate=["CYP1A2"],
        narrow_therapeutic_index=True,
        seizure_risk=True,
        note="Narrow window; CYP1A2 inhibitors push it into the toxic range "
        "(arrhythmia, seizures).",
    ),
    # ------------------------------------------------- Metabolic / other
    Drug(
        name="Metformin",
        smiles="CN(C)C(=N)NC(N)=N",
        drug_class="Biguanide antihyperglycaemic",
        mw_ref=129.16,
        fe=0.90,
        note="Not metabolised; cleared by renal tubular secretion. Accumulation "
        "in renal impairment is the mechanism of lactic acidosis.",
    ),
    Drug(
        name="Methotrexate",
        smiles="CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1",
        drug_class="Antifolate / DMARD",
        mw_ref=454.44,
        fe=0.90,
        antifolate=True,
        nephrotoxic=True,
        hepatotoxic=True,
        narrow_therapeutic_index=True,
        note="Renally cleared and narrow-window; NSAIDs reduce its clearance and "
        "can cause severe marrow suppression.",
    ),
    Drug(
        name="Allopurinol",
        smiles="O=c1[nH]cnc2[nH]ncc12",
        drug_class="Xanthine oxidase inhibitor",
        mw_ref=136.11,
        fe=0.10,
        note="Blocks xanthine oxidase — the enzyme that inactivates azathioprine "
        "and mercaptopurine.",
    ),
    # ------------------------------------- Antibiotics (link to Module 3)
    Drug(
        name="Clarithromycin",
        smiles="CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H](N(C)C)[C@H]2O)[C@](C)(OC)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
        drug_class="Macrolide antibiotic",
        mw_ref=747.95,
        fe=0.30,
        cyp_inhibitor=["CYP3A4"],
        cyp_substrate=["CYP3A4"],
        pgp_inhibitor=True,
        qt_prolonging=True,
        hepatotoxic=True,
        antibiotic_class="Macrolides",
        note="Strong CYP3A4 inhibitor. Co-prescribing with simvastatin is a "
        "well-documented cause of rhabdomyolysis.",
    ),
    Drug(
        name="Azithromycin",
        smiles="CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@@H](C)CN(C)[C@H](C)[C@@H](O)[C@]1(C)O",
        drug_class="Azalide antibiotic",
        mw_ref=748.98,
        fe=0.12,
        qt_prolonging=True,
        antibiotic_class="Macrolides",
        note="Unlike clarithromycin it barely touches CYP3A4 — a good example of "
        "structurally similar drugs with different interaction profiles.",
    ),
    Drug(
        name="Ciprofloxacin",
        smiles="O=C(O)C1=CN(C2CC2)c2cc(N3CCNCC3)c(F)cc2C1=O",
        drug_class="Fluoroquinolone antibiotic",
        mw_ref=331.34,
        fe=0.45,
        cyp_inhibitor=["CYP1A2"],
        qt_prolonging=True,
        seizure_risk=True,
        antibiotic_class="Fluoroquinolones",
        note="A potent CYP1A2 inhibitor — the classic cause of theophylline toxicity.",
    ),
    Drug(
        name="Levofloxacin",
        smiles="CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
        drug_class="Fluoroquinolone antibiotic",
        mw_ref=361.37,
        fe=0.85,
        qt_prolonging=True,
        seizure_risk=True,
        antibiotic_class="Fluoroquinolones",
    ),
    Drug(
        name="Amoxicillin",
        smiles="CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
        drug_class="Aminopenicillin antibiotic",
        mw_ref=365.40,
        fe=0.70,
        antibiotic_class="Penicillins",
    ),
    Drug(
        name="Meropenem",
        smiles="CC1C2C(C(=O)N2C(=C1SC1CNC(C1)C(=O)N(C)C)C(=O)O)C(C)O",
        drug_class="Carbapenem antibiotic",
        mw_ref=383.46,
        fe=0.70,
        seizure_risk=True,
        antibiotic_class="Carbapenems",
        note="A last-line agent — the reason carbapenem resistance is a "
        "priority signal in Module 3.",
    ),
    Drug(
        name="Linezolid",
        smiles="CC(=O)NCC1CN(c2ccc(N3CCOCC3)c(F)c2)C(=O)O1",
        drug_class="Oxazolidinone antibiotic",
        mw_ref=337.35,
        fe=0.30,
        serotonergic=True,
        antibiotic_class="Oxazolidinones",
        note="A reversible MAO inhibitor — serotonin syndrome risk with SSRIs.",
    ),
    Drug(
        name="Metronidazole",
        smiles="Cc1ncc([N+](=O)[O-])n1CCO",
        drug_class="Nitroimidazole antibiotic",
        mw_ref=171.15,
        fe=0.10,
        cyp_inhibitor=["CYP2C9"],
        antibiotic_class="Nitroimidazoles",
    ),
    Drug(
        name="Trimethoprim",
        smiles="COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC",
        drug_class="Dihydrofolate reductase inhibitor",
        mw_ref=290.32,
        fe=0.60,
        antifolate=True,
        potassium_raising=True,
        antibiotic_class="Trimethoprim-sulfonamides",
        note="Blocks the renal epithelial sodium channel like amiloride — a "
        "frequently missed cause of hyperkalaemia.",
    ),
    Drug(
        name="Sulfamethoxazole",
        smiles="Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",
        drug_class="Sulfonamide antibiotic",
        mw_ref=253.28,
        fe=0.20,
        antifolate=True,
        cyp_inhibitor=["CYP2C9"],
        nephrotoxic=True,
        antibiotic_class="Trimethoprim-sulfonamides",
    ),
    Drug(
        name="Tobramycin",
        smiles="NC[C@H]1O[C@H](O[C@@H]2[C@@H](N)C[C@@H](N)[C@H](O[C@H]3O[C@H](CO)[C@@H](O)[C@H](N)[C@H]3O)[C@H]2O)[C@H](N)C[C@@H]1O",
        drug_class="Aminoglycoside antibiotic",
        mw_ref=467.51,
        fe=0.90,
        nephrotoxic=True,
        narrow_therapeutic_index=True,
        antibiotic_class="Aminoglycosides",
        note="Almost entirely renally cleared AND directly nephrotoxic — a "
        "self-reinforcing spiral in kidney impairment.",
    ),
]

DRUGS: dict[str, Drug] = {d.name: d for d in _DRUG_LIST}
DRUG_NAMES: list[str] = sorted(DRUGS)


# =============================================================================
# Lookup + validation
# =============================================================================


def get_drug(identifier: str) -> Drug | None:
    """Resolve a drug by name (case-insensitive). Returns None if unknown."""
    if not identifier:
        return None
    key = identifier.strip().lower()
    for name, drug in DRUGS.items():
        if name.lower() == key:
            return drug
    return None


def drug_from_smiles(smiles: str, label: str = "Custom compound") -> Drug:
    """
    Wrap a raw SMILES string as a Drug with an empty pharmacology profile.

    This is the "novel candidate" path: a compound the knowledge base has never
    seen. The DDI model can still score it, because the model reads STRUCTURE.
    What we lose is the curated mechanism text — which is exactly the honest
    limitation to show a judging panel.
    """
    return Drug(
        name=label,
        smiles=smiles.strip(),
        drug_class="Uncharacterised candidate compound",
        mw_ref=0.0,
        note="Novel/unregistered structure — no curated pharmacology profile. "
        "Prediction is structure-derived only.",
    )


def validate_drug_db(tolerance: float = 1.0) -> list[str]:
    """
    Parse every SMILES with RDKit and compare against its reference MW.

    Returns a list of human-readable problems (empty list == all good). This is
    called by train.py and surfaced in the app's Methods tab, so the demo can
    truthfully claim its structures are machine-verified.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")

    problems: list[str] = []
    for name, drug in DRUGS.items():
        mol = Chem.MolFromSmiles(drug.smiles)
        if mol is None:
            problems.append(f"{name}: SMILES failed to parse")
            continue
        mw = Descriptors.MolWt(mol)
        if abs(mw - drug.mw_ref) > tolerance:
            problems.append(
                f"{name}: computed MW {mw:.2f} vs reference {drug.mw_ref:.2f} "
                f"(off by {abs(mw - drug.mw_ref):.2f})"
            )
    return problems


if __name__ == "__main__":
    issues = validate_drug_db()
    if issues:
        print(f"{len(issues)} problem(s) found out of {len(DRUGS)} structures:")
        for i in issues:
            print("  -", i)
    else:
        print(f"All {len(DRUGS)} structures validated against reference MW.")
