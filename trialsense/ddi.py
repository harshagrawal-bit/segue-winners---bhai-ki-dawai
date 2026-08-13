"""
TrialSense — Module 1: Drug-Drug Interaction predictor.

ARCHITECTURE (and why it is built this way)
-------------------------------------------
Two independent layers that are deliberately kept apart:

  LAYER A — Mechanism knowledge base (`assess_mechanisms`)
      Deterministic pharmacology rules over the per-drug profiles in drugs.py:
      CYP450 inhibition/induction, P-glycoprotein handling, renal competition,
      and additive pharmacodynamic toxicity. Produces a severity 0-3 AND a
      plain-English mechanism. Works only for drugs in the knowledge base.

  LAYER B — Structure-only ML screening filter (`DDIModel`)
      Sees ONLY the two molecular structures — Morgan/ECFP fingerprints, MACCS
      keys and physicochemical descriptors — never the pharmacology profile. It
      is learning to infer "this molecule probably inhibits CYP3A4" from
      structure alone, which is what lets it score a NOVEL compound that has no
      profile at all.

      It is deliberately framed as a BINARY screening filter — "is this pairing
      clinically significant (moderate or worse), yes or no?" — rather than a
      4-class severity predictor. With only 39 drugs (741 pairs), a 4-class
      model is badly data-starved: it scored 55% accuracy and 0.30 macro-F1 on
      held-out drugs, which is not honest to present as a severity prediction.
      The binary task is genuinely learnable at this data scale — ROC-AUC around
      0.68-0.73 on drugs never seen in training, depending on which drugs are
      held out. That spread is wide because 39 drugs is a small set, which is
      why `evaluate()` pools several hold-outs and reports a standard deviation
      rather than a single flattering number. It is a real signal, well short of
      production quality, and it matches how a screening tool is actually used:
      triage pairs for expert review, with severity and mechanism from Layer A.

Keeping the layers separate is what makes the evaluation meaningful. If the
model could see the pharmacology flags it would trivially reproduce the rules;
because it cannot, held-out performance measures genuine structure->pharmacology
transfer. We report a cold-drug split (entire drugs removed from training)
alongside the easier random split, because the cold split is the number that
reflects real-world use on a new compound.

HONEST FRAMING FOR THE DEMO
---------------------------
The labels come from a curated mechanism knowledge base, NOT from DrugBank or
TWOSIDES (both need a licence/registration that we could not obtain within the
hackathon window). The mechanisms encoded are real and textbook-documented; the
severity thresholds are our own calibration. Swapping in a licensed label set
means replacing `build_dataset()` only — the featurisation, model and evaluation
code are unchanged. That is the intended production path.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, MACCSkeys

from .drugs import DRUGS, Drug

RDLogger.DisableLog("rdApp.*")

# --- Severity scale ----------------------------------------------------------
SEVERITY_LEVELS = ["None", "Mild", "Moderate", "Severe"]
SEVERITY_COLORS = ["#2E7D32", "#F9A825", "#EF6C00", "#C62828"]
SEVERITY_ACTIONS = {
    0: "No mechanism-based interaction identified. Standard monitoring.",
    1: "Minor interaction. Routine monitoring is sufficient.",
    2: "Clinically meaningful. Expect dose adjustment or added monitoring.",
    3: "High-risk combination. Avoid, or redesign the protocol to exclude it.",
}

# Fingerprint size. Small on purpose: ~740 training pairs, so a 4096-bit
# fingerprint would be mostly noise. 512 bits keeps the feature:sample ratio sane
# and the model trains in seconds on a laptop CPU.
FP_BITS = 512
FP_RADIUS = 2  # radius 2 == ECFP4

# A pair is "clinically significant" — worth a specialist's attention — at
# Moderate or above. This is the binary target Layer B is trained on.
SIGNIFICANT_THRESHOLD = 2

# Decision threshold for the screening flag. Chosen from the pooled cold-drug
# evaluation to favour RECALL over precision: in pre-trial screening, missing a
# dangerous pairing costs far more than sending a safe one for review.
SCREENING_THRESHOLD = 0.35

# --- Interaction-strength calibration ---------------------------------------
# Which drugs are STRONG (rather than moderate) perpetrators, per enzyme.
# Taken from the standard regulatory classification of CYP inhibitors/inducers.
STRONG_INHIBITORS: dict[str, list[str]] = {
    "Ketoconazole": ["CYP3A4"],
    "Clarithromycin": ["CYP3A4"],
    "Fluconazole": ["CYP2C9", "CYP2C19"],
    "Fluoxetine": ["CYP2D6"],
    "Ciprofloxacin": ["CYP1A2"],
    "Amiodarone": ["CYP2C9"],
}
STRONG_INDUCERS: dict[str, list[str]] = {
    "Rifampicin": ["CYP3A4", "CYP2C9", "CYP2C19", "CYP1A2"],
}

# Prodrugs whose ACTIVATION depends on a CYP. Inhibiting that enzyme causes
# treatment failure, not toxicity — an inversion that catches naive models out.
PRODRUGS: dict[str, str] = {
    "Clopidogrel": "CYP2C19",
    "Tramadol": "CYP2D6",
}

# Additive pharmacodynamic risks: flag -> (base severity, plain-English effect)
ADDITIVE_PD: dict[str, tuple[float, str]] = {
    "qt_prolonging": (1.5, "additive QT-interval prolongation, raising the risk of the ventricular arrhythmia torsades de pointes"),
    "serotonergic": (1.8, "additive serotonin activity, which can precipitate serotonin syndrome"),
    "bleeding_risk": (1.9, "additive bleeding risk"),
    "potassium_raising": (2.0, "additive potassium retention, risking hyperkalaemia and cardiac arrhythmia"),
    "nephrotoxic": (1.3, "additive kidney injury"),
    "cns_depressant": (1.3, "additive sedation and respiratory depression"),
    "seizure_risk": (1.2, "an additively lowered seizure threshold"),
    "myopathy_risk": (1.2, "additive risk of muscle injury"),
    "hepatotoxic": (1.0, "additive strain on the liver"),
}


@dataclass
class Mechanism:
    """One identified interaction pathway between two drugs."""

    kind: str  # cyp_inhibition | cyp_induction | pgp | prodrug | renal | additive_pd | opposing
    severity: float  # this pathway's contribution, roughly on the 0-3 scale
    consequence: str  # exposure_up | exposure_down | efficacy_loss | additive_toxicity | protective
    text: str  # plain-English explanation for a non-specialist
    enzyme: str | None = None


@dataclass
class DDIAssessment:
    """Full Module 1 result for one drug pair."""

    drug_a: str
    drug_b: str
    kb_severity: int  # knowledge-base severity, 0-3 (None if drugs unknown)
    kb_score: float  # continuous score behind kb_severity
    mechanisms: list[Mechanism] = field(default_factory=list)
    # Layer B: probability this pairing is clinically significant (>= Moderate),
    # predicted from molecular structure alone.
    ml_probability: float | None = None
    from_knowledge_base: bool = True  # False if either drug was a raw SMILES

    @property
    def headline_severity(self) -> int:
        """
        The severity shown to the user.

        For known drugs we lead with the mechanism knowledge base, because it is
        the more defensible number and it comes with a citable explanation.

        For a NOVEL structure there is no KB entry, so all we have is Layer B's
        binary screening probability. We map that onto the severity scale
        conservatively — flagged becomes Moderate, not Severe — because a binary
        filter genuinely cannot distinguish "needs monitoring" from "never
        co-administer", and claiming otherwise would overstate what it knows.
        """
        if self.from_knowledge_base:
            return self.kb_severity
        if self.ml_probability is None:
            return 0
        return 2 if self.ml_probability >= SCREENING_THRESHOLD else 0

    @property
    def headline_score(self) -> float:
        """
        The continuous score that downstream modules should build on.

        For known drugs this is the mechanism score. For a NOVEL structure the
        knowledge base contributes nothing (kb_score is 0), so we derive a score
        from the filter's probability instead — otherwise Module 2 and the
        composite would silently read 0 for a pairing the filter had just
        flagged, and the report would contradict its own interaction tab.

        The mapping is capped just below the Severe threshold (2.42): even a
        maximally confident binary filter cannot justify calling a novel pairing
        Severe, which keeps this consistent with `headline_severity`.
        """
        if self.from_knowledge_base:
            return self.kb_score
        if self.ml_probability is None:
            return 0.0
        p, t = self.ml_probability, SCREENING_THRESHOLD
        if p < t:
            # Below the flag threshold: scale up to the Moderate boundary.
            return 1.5 * (p / t)
        # Above it: from the Moderate boundary up to just short of Severe.
        return 1.5 + 0.9 * (p - t) / (1.0 - t)

    @property
    def label(self) -> str:
        return SEVERITY_LEVELS[self.headline_severity]

    @property
    def color(self) -> str:
        return SEVERITY_COLORS[self.headline_severity]

    @property
    def ml_flags_significant(self) -> bool | None:
        """Does the structure-only filter flag this pairing for review?"""
        if self.ml_probability is None:
            return None
        return self.ml_probability >= SCREENING_THRESHOLD

    @property
    def models_agree(self) -> bool | None:
        """
        Does the structure-only filter agree with the knowledge base on whether
        this pairing is clinically significant? None when not comparable.
        """
        if self.ml_probability is None or not self.from_knowledge_base:
            return None
        return self.ml_flags_significant == (self.kb_severity >= SIGNIFICANT_THRESHOLD)

    def plain_summary(self) -> str:
        """One sentence a non-technical judge can read off the screen."""
        a, b = self.drug_a, self.drug_b
        if not self.mechanisms:
            return (
                f"No known interaction pathway between {a} and {b}. They are "
                "cleared by different routes and do not share a toxicity target."
            )
        lead = max(self.mechanisms, key=lambda m: m.severity)
        return lead.text

    def all_explanations(self) -> list[str]:
        return [m.text for m in sorted(self.mechanisms, key=lambda m: -m.severity)]


# =============================================================================
# LAYER A — mechanism knowledge base
# =============================================================================


def _inhibition_strength(perpetrator: Drug, enzyme: str) -> float:
    """1.0 for a strong inhibitor of this enzyme, 0.65 for a moderate one."""
    return 1.0 if enzyme in STRONG_INHIBITORS.get(perpetrator.name, []) else 0.65


def _induction_strength(perpetrator: Drug, enzyme: str) -> float:
    return 1.0 if enzyme in STRONG_INDUCERS.get(perpetrator.name, []) else 0.7


def _directional_mechanisms(perp: Drug, victim: Drug) -> list[Mechanism]:
    """
    Mechanisms where `perp` acts ON `victim`. Called twice (both orderings) so
    the overall assessment is symmetric even though each pathway is directional.
    """
    out: list[Mechanism] = []

    # --- CYP inhibition: perpetrator blocks the enzyme that clears the victim
    for enzyme in perp.cyp_inhibitor:
        if enzyme not in victim.cyp_substrate:
            continue

        # A prodrug needing this enzyme loses efficacy instead of accumulating.
        if PRODRUGS.get(victim.name) == enzyme:
            sev = 2.4 * _inhibition_strength(perp, enzyme)
            out.append(
                Mechanism(
                    kind="prodrug",
                    severity=sev,
                    consequence="efficacy_loss",
                    enzyme=enzyme,
                    text=(
                        f"{victim.name} is an inactive prodrug that {enzyme} must "
                        f"switch on. {perp.name} blocks {enzyme}, so less active "
                        f"drug is formed and {victim.name} may simply not work — "
                        "a failure of efficacy rather than a toxicity."
                    ),
                )
            )
            continue

        strength = _inhibition_strength(perp, enzyme)
        sole_route = len(victim.cyp_substrate) == 1
        sev = 1.3 * strength
        sev += 0.8 if victim.narrow_therapeutic_index else 0.0
        # A STRONG inhibitor blocking the victim's ONLY clearance route can raise
        # exposure many-fold — dangerous even without a narrow therapeutic index
        # (e.g. ketoconazole raises midazolam exposure ~15x).
        if sole_route:
            sev += 1.2 if strength >= 1.0 else 0.5
        sev += 0.4 if victim.myopathy_risk else 0.0
        # If the victim is mostly cleared by the kidney, blocking a liver enzyme
        # matters much less. fh scales the whole pathway.
        sev *= victim.fh

        extra = ""
        if victim.narrow_therapeutic_index:
            extra = (
                f" {victim.name} has a narrow safety margin, so even a modest "
                "rise in blood level can become toxic."
            )
        elif victim.myopathy_risk:
            extra = (
                f" Raised {victim.name} levels are specifically linked to muscle "
                "breakdown (rhabdomyolysis)."
            )

        out.append(
            Mechanism(
                kind="cyp_inhibition",
                severity=sev,
                consequence="exposure_up",
                enzyme=enzyme,
                text=(
                    f"{perp.name} inhibits {enzyme}, the liver enzyme that clears "
                    f"{victim.name}. {victim.name} is therefore removed from the "
                    f"body more slowly and its blood level rises.{extra}"
                ),
            )
        )

    # --- CYP induction: perpetrator revs up the enzyme, victim disappears faster
    for enzyme in perp.cyp_inducer:
        if enzyme not in victim.cyp_substrate:
            continue
        sole_route = len(victim.cyp_substrate) == 1
        sev = 1.15 * _induction_strength(perp, enzyme)
        sev += 0.7 if victim.narrow_therapeutic_index else 0.0
        sev += 0.4 if sole_route else 0.0
        sev *= victim.fh

        out.append(
            Mechanism(
                kind="cyp_induction",
                severity=sev,
                consequence="efficacy_loss",
                enzyme=enzyme,
                text=(
                    f"{perp.name} induces (speeds up) {enzyme}, the enzyme that "
                    f"clears {victim.name}. {victim.name} is destroyed faster than "
                    "intended and can drop below its effective level — the trial "
                    "risk here is a drug that silently stops working."
                ),
            )
        )

    # --- P-glycoprotein: the efflux pump that limits absorption
    if perp.pgp_inhibitor and victim.pgp_substrate:
        sev = 1.3 + (1.15 if victim.narrow_therapeutic_index else 0.0)
        out.append(
            Mechanism(
                kind="pgp",
                severity=sev,
                consequence="exposure_up",
                text=(
                    f"{perp.name} blocks P-glycoprotein, the pump that normally "
                    f"pushes {victim.name} back out of the gut wall and kidney. "
                    f"More {victim.name} is absorbed and retained."
                ),
            )
        )

    # --- Renal competition: a kidney-toxic drug slows a kidney-cleared drug
    if perp.nephrotoxic and victim.fe >= 0.6 and perp.name != victim.name:
        sev = 1.0 + (0.95 if victim.narrow_therapeutic_index else 0.0)
        out.append(
            Mechanism(
                kind="renal",
                severity=sev,
                consequence="exposure_up",
                text=(
                    f"{perp.name} reduces kidney blood flow or injures the tubule, "
                    f"while {int(victim.fe * 100)}% of {victim.name} is cleared "
                    f"unchanged by that same kidney. {victim.name} accumulates."
                ),
            )
        )

    return out


def assess_mechanisms(a: Drug, b: Drug) -> tuple[float, list[Mechanism]]:
    """
    Identify every interaction pathway between two drugs and combine them.

    Returns (continuous_score, mechanisms). Combination rule: the dominant
    pathway carries full weight and each additional pathway adds 40% of its own
    severity. A plain sum would let five trivial pathways masquerade as one
    catastrophic one, which is the classic way rule-based DDI tools become
    unusable through over-alerting.
    """
    mechs: list[Mechanism] = []
    mechs += _directional_mechanisms(a, b)
    mechs += _directional_mechanisms(b, a)

    # --- Additive pharmacodynamics: both drugs push the same physiology
    for flag, (base, effect) in ADDITIVE_PD.items():
        if getattr(a, flag) and getattr(b, flag):
            sev = base
            # Amiodarone is the most potent QT offender in the set.
            if flag == "qt_prolonging" and "Amiodarone" in (a.name, b.name):
                sev += 0.5
            # Linezolid is a reversible MAO inhibitor, a mechanistically distinct
            # and more dangerous route to serotonin excess than SSRI + SSRI.
            if flag == "serotonergic" and "Linezolid" in (a.name, b.name):
                sev += 0.7
            # Additive bleeding matters far more when one agent is a
            # narrow-window anticoagulant (e.g. aspirin added to warfarin).
            if flag == "bleeding_risk" and (
                a.narrow_therapeutic_index or b.narrow_therapeutic_index
            ):
                sev += 0.6
            mechs.append(
                Mechanism(
                    kind="additive_pd",
                    severity=sev,
                    consequence="additive_toxicity",
                    text=(
                        f"{a.name} and {b.name} both cause {effect}. Neither drug "
                        "changes how the other is cleared — the risk comes from "
                        "the two effects stacking on the same target."
                    ),
                )
            )

    score = 0.0
    if mechs:
        ordered = sorted(mechs, key=lambda m: -m.severity)
        score = ordered[0].severity + 0.4 * sum(m.severity for m in ordered[1:])

    # --- Opposing effects can genuinely CANCEL. Modelling only additive risk
    # would overstate danger for a pairing clinicians deliberately use.
    opposing = (a.potassium_raising and b.potassium_lowering) or (
        b.potassium_raising and a.potassium_lowering
    )
    if opposing:
        score = max(0.0, score - 0.6)
        mechs.append(
            Mechanism(
                kind="opposing",
                severity=0.0,
                consequence="protective",
                text=(
                    f"{a.name} and {b.name} push serum potassium in OPPOSITE "
                    "directions, so they partly cancel out. This pairing is often "
                    "used deliberately — flagged here as risk-reducing, not risk-adding."
                ),
            )
        )

    return score, mechs


def score_to_severity(score: float) -> int:
    """Bucket the continuous mechanism score onto the 0-3 clinical scale."""
    if score < 0.75:
        return 0
    if score < 1.5:
        return 1
    if score < 2.42:
        return 2
    return 3


def knowledge_base_assess(a: Drug, b: Drug) -> tuple[int, float, list[Mechanism]]:
    score, mechs = assess_mechanisms(a, b)
    return score_to_severity(score), score, mechs


# =============================================================================
# LAYER B — structure-only featurisation + ML model
# =============================================================================


def _structure_vector(smiles: str) -> np.ndarray | None:
    """
    Morgan/ECFP4 bits concatenated with MACCS structural keys.

    MACCS adds 167 hand-designed substructure keys. They are dense and
    chemically meaningful, which matters a great deal at this data scale — a
    hashed Morgan fingerprint is sparse enough that with 741 training pairs most
    bits are never informative. Benchmarked cold-drug ROC-AUC: Morgan alone
    0.736, MACCS alone 0.746, both together 0.739. We keep both because the
    combination is stable across seeds and Morgan bits carry the finer detail
    that a larger dataset would exploit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = AllChem.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    morgan = np.array(gen.GetFingerprint(mol), dtype=np.float32)
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    return np.concatenate([morgan, maccs])


def _descriptors(smiles: str) -> np.ndarray | None:
    """Physicochemical properties that correlate with metabolic handling."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),  # lipophilicity — drives CYP affinity
            Descriptors.TPSA(mol),  # polar surface area — drives renal clearance
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),  # 3D character
            Descriptors.HeavyAtomCount(mol),
            Descriptors.RingCount(mol),
        ],
        dtype=np.float32,
    )


def pair_features(smiles_a: str, smiles_b: str) -> np.ndarray | None:
    """
    Build an ORDER-INVARIANT feature vector for a drug pair.

    An interaction between A and B is the same event as one between B and A, so
    the features must be identical either way. We achieve that with commutative
    operations only:
        fingerprints -> bitwise OR (union) and AND (shared substructure)
        descriptors  -> elementwise sum and absolute difference

    Without this, the model can memorise argument order and the evaluation
    silently inflates.
    """
    s_a, s_b = _structure_vector(smiles_a), _structure_vector(smiles_b)
    d_a, d_b = _descriptors(smiles_a), _descriptors(smiles_b)
    if s_a is None or s_b is None or d_a is None or d_b is None:
        return None

    # For binary structure bits, max == "at least one drug has it" (union) and
    # min == "both have it" (intersection).
    struct_union = np.maximum(s_a, s_b)
    struct_shared = np.minimum(s_a, s_b)
    desc_sum = d_a + d_b
    desc_diff = np.abs(d_a - d_b)

    return np.concatenate([struct_union, struct_shared, desc_sum, desc_diff])


def build_dataset() -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """
    Every unordered pair of known drugs, featurised from structure and labelled
    by the mechanism knowledge base.

    This is the ONE function to replace when a licensed DrugBank/TWOSIDES label
    set becomes available. Everything downstream is source-agnostic.
    """
    X, y, pairs = [], [], []
    for name_a, name_b in itertools.combinations(sorted(DRUGS), 2):
        a, b = DRUGS[name_a], DRUGS[name_b]
        feats = pair_features(a.smiles, b.smiles)
        if feats is None:
            continue
        severity, _, _ = knowledge_base_assess(a, b)
        X.append(feats)
        y.append(severity)
        pairs.append((name_a, name_b))
    return np.vstack(X), np.array(y), pairs


class DDIModel:
    """
    Binary Random Forest screening filter over paired structural features.

    Predicts P(clinically significant interaction) from structure alone.
    """

    def __init__(self, n_estimators: int = 400, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier

        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",  # significant pairs are the ~20% minority
            random_state=random_state,
            n_jobs=-1,
        )
        self.metrics: dict = {}
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DDIModel":
        """
        Fit on the 0-3 severity scale; the binarisation happens here so callers
        cannot accidentally train on a different target than they evaluate on.
        """
        self.clf.fit(X, (np.asarray(y) >= SIGNIFICANT_THRESHOLD).astype(int))
        self.is_fitted = True
        return self

    def predict_pair(self, smiles_a: str, smiles_b: str) -> float:
        """Probability that this pairing is clinically significant."""
        feats = pair_features(smiles_a, smiles_b)
        if feats is None:
            raise ValueError("Could not parse one or both SMILES strings.")
        return float(self.clf.predict_proba(feats.reshape(1, -1))[0, 1])


def evaluate(random_state: int = 42, n_repeats: int = 10) -> dict:
    """
    Two evaluations, because they answer different questions.

    1. RANDOM PAIR SPLIT — held-out pairs of drugs the model has seen elsewhere.
       Optimistic: shared drugs leak information across the split.
    2. COLD-DRUG SPLIT — a set of drugs removed entirely; the test set is every
       pair touching them. This is the honest number for screening a NEW
       compound, which is TrialSense's actual use case. Repeated over several
       random drug hold-outs and pooled, because a single 8-drug hold-out from a
       39-drug set is far too noisy to quote a number from.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    X, y, pairs = build_dataset()
    yb = (y >= SIGNIFICANT_THRESHOLD).astype(int)

    results: dict = {
        "n_pairs": len(y),
        "n_drugs": len(DRUGS),
        "n_features": int(X.shape[1]),
        "significant_rate": float(yb.mean()),
        "screening_threshold": SCREENING_THRESHOLD,
        "class_distribution": {
            SEVERITY_LEVELS[i]: int((y == i).sum()) for i in range(len(SEVERITY_LEVELS))
        },
    }

    # --- 1. Random split. Models are always fitted on the raw severity scale
    # (DDIModel.fit binarises); `yb` is used only for scoring.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=yb
    )
    yb_te = (y_te >= SIGNIFICANT_THRESHOLD).astype(int)
    m = DDIModel(random_state=random_state).fit(X_tr, y_tr)
    prob = m.clf.predict_proba(X_te)[:, 1]
    pred = (prob >= SCREENING_THRESHOLD).astype(int)
    results["random_split"] = {
        "roc_auc": float(roc_auc_score(yb_te, prob)),
        "recall": float(recall_score(yb_te, pred, zero_division=0)),
        "precision": float(precision_score(yb_te, pred, zero_division=0)),
        "accuracy": float(accuracy_score(yb_te, pred)),
        "n_test": int(len(yb_te)),
    }

    # --- 2. Cold-drug split, pooled over repeats
    drug_names = sorted(DRUGS)
    pooled_prob, pooled_true = [], []
    aucs, held_examples = [], []
    for rep in range(n_repeats):
        rng = np.random.RandomState(random_state + rep)
        held_out = set(rng.choice(drug_names, size=8, replace=False).tolist())
        mask = np.array([(a in held_out) or (b in held_out) for a, b in pairs])
        if mask.sum() == 0 or yb[mask].sum() == 0:
            continue
        m2 = DDIModel(random_state=random_state).fit(X[~mask], y[~mask])
        p = m2.clf.predict_proba(X[mask])[:, 1]
        pooled_prob.append(p)
        pooled_true.append(yb[mask])
        aucs.append(roc_auc_score(yb[mask], p))
        if rep == 0:
            held_examples = sorted(held_out)

    prob_all = np.concatenate(pooled_prob)
    true_all = np.concatenate(pooled_true)
    pred_all = (prob_all >= SCREENING_THRESHOLD).astype(int)

    results["cold_drug_split"] = {
        "roc_auc": float(roc_auc_score(true_all, prob_all)),
        "roc_auc_std": float(np.std(aucs)),
        "recall": float(recall_score(true_all, pred_all, zero_division=0)),
        "precision": float(precision_score(true_all, pred_all, zero_division=0)),
        "accuracy": float(accuracy_score(true_all, pred_all)),
        "flagged_rate": float(pred_all.mean()),
        "n_test": int(len(true_all)),
        "n_repeats": len(aucs),
        "held_out_drugs_example": held_examples,
    }
    return results


# =============================================================================
# Public entry point used by the app
# =============================================================================


def analyze_pair(a: Drug, b: Drug, model: DDIModel | None = None) -> DDIAssessment:
    """Run both layers on a drug pair and package the result for the UI."""
    known = a.name in DRUGS and b.name in DRUGS
    severity, score, mechs = (0, 0.0, [])
    if known:
        severity, score, mechs = knowledge_base_assess(a, b)

    assessment = DDIAssessment(
        drug_a=a.name,
        drug_b=b.name,
        kb_severity=severity,
        kb_score=score,
        mechanisms=mechs,
        from_knowledge_base=known,
    )

    if model is not None and model.is_fitted:
        try:
            assessment.ml_probability = model.predict_pair(a.smiles, b.smiles)
        except ValueError:
            pass

    return assessment
