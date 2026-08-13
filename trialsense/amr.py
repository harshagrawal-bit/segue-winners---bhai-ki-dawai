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

WHAT IS SYNTHETIC — BE EXPLICIT ABOUT THIS
------------------------------------------
The bundled isolate sequences are SYNTHETIC. Each resistance gene is represented
by a deterministic, gene-specific "marker cassette": a stable pseudo-random
sequence with its own nucleotide composition, generated from a fixed seed so it
is byte-identical on every run and across machines. These cassettes are NOT the
real published sequences of blaNDM-1, mecA and so on — we did not want to ship
approximate biological sequences and present them as authentic.

What that means honestly:
  * The gene -> antibiotic-class resistance mapping IS real microbiology.
  * The strain gene contents ARE realistic (modelled on well-documented
    epidemiology, including India-relevant lineages such as XDR S. Typhi).
  * The classifier genuinely learns to detect cassettes it was trained on, and
    genuinely generalises to gene COMBINATIONS it has never seen — which is the
    property that matters and is what the cold-combination evaluation measures.
  * Absolute accuracy figures reflect a cleaner signal than real sequencing data,
    which carries noise, contamination and incomplete assembly.

To run this on real data you swap `_gene_cassette()` for real reference sequences
(from NCBI Pathogen Detection or the CARD database) and retrain. Nothing else
changes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

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


@dataclass
class ResistanceGene:
    """A resistance determinant and the classes it defeats."""

    name: str
    confers_resistance_to: list[str]
    mechanism: str
    gc_bias: float = 0.5  # nucleotide composition, gives each gene a k-mer signature


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
            "rpoB_S531L",
            ["Rifamycins"],
            "A mutation in RNA polymerase, rifampicin's target. This is the "
            "defining marker of rifampicin-resistant tuberculosis.",
            gc_bias=0.58,
        ),
    ]
}


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
            ["blaCTX-M-15", "sul1", "dfrA17", "gyrA_S83L", "aac(6')-Ib"],
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
            ["rpoB_S531L"],
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

# --- Sequence synthesis parameters ------------------------------------------
CASSETTE_LEN = 900  # bp per marker cassette
BACKGROUND_LEN = 2600  # bp of species background per isolate
KMER_K = 5  # 5-mers -> 1024 features
_BASES = np.array(list("ACGT"))

# Each cassette is built from a gene-specific vocabulary of signature motifs.
# This mirrors the real biology that makes k-mer detection work: coding sequences
# have characteristic oligonucleotide composition (codon usage, conserved
# domains), so a gene is recognisable by WHICH short words it is built from —
# not merely by its GC content.
N_MOTIFS = 18
MOTIF_LEN = 8


def _gene_motifs(gene_name: str) -> list[str]:
    """The deterministic signature motif vocabulary for one gene."""
    gene = RESISTANCE_GENES[gene_name]
    seed = int(hashlib.sha256(gene_name.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    gc = gene.gc_bias
    probs = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A, C, G, T
    return [
        "".join(rng.choice(_BASES, size=MOTIF_LEN, p=probs)) for _ in range(N_MOTIFS)
    ]


def _gene_cassette(gene_name: str) -> str:
    """
    Deterministic synthetic marker sequence for a resistance gene.

    Seeded from a hash of the gene name, so the same gene always produces the
    same sequence on every machine and every run. Built by sampling that gene's
    signature motifs with short filler between them, which gives each cassette a
    distinctive k-mer profile the classifier can detect.

    SYNTHETIC — this is not the published sequence of the real gene. Replace
    this function with real reference sequences to run on live data.
    """
    gene = RESISTANCE_GENES[gene_name]
    seed = int(hashlib.sha256(gene_name.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed + 1)
    motifs = _gene_motifs(gene_name)
    gc = gene.gc_bias
    probs = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]

    parts: list[str] = []
    length = 0
    while length < CASSETTE_LEN:
        parts.append(motifs[rng.randint(len(motifs))])
        length += MOTIF_LEN
        if rng.rand() < 0.45:  # occasional filler between signature motifs
            filler = "".join(rng.choice(_BASES, size=rng.randint(2, 6), p=probs))
            parts.append(filler)
            length += len(filler)
    return "".join(parts)[:CASSETTE_LEN]


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
    """Multi-label Random Forest: one resistance call per antibiotic class."""

    def __init__(self, n_estimators: int = 250, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier

        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        self.classes = ANTIBIOTIC_CLASSES
        self.is_fitted = False
        self.metrics: dict = {}

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "AMRModel":
        from sklearn.multioutput import MultiOutputClassifier

        self.model = MultiOutputClassifier(self.clf, n_jobs=-1)
        self.model.fit(X, Y)
        self.is_fitted = True
        return self

    def predict_sequence(self, sequence: str) -> dict[str, float]:
        """Resistance probability per antibiotic class for one raw sequence."""
        feats = kmer_features(sequence).reshape(1, -1)
        probs = self.model.predict_proba(feats)
        out = {}
        for cls, p in zip(self.classes, probs):
            # p has shape (1, n_classes_seen); index 1 == P(resistant).
            out[cls] = float(p[0][1]) if p.shape[1] > 1 else float(p[0][0] == 1)
        return out


def detect_genes(sequence: str, threshold: float = 0.55) -> list[tuple[str, float]]:
    """
    Identify which marker cassettes are present, by sliding-window k-mer
    similarity against each reference gene profile.

    This is the mechanism-explanation layer — the counterpart to the DDI
    knowledge base in Module 1. The classifier says WHETHER the isolate is
    resistant; this says WHICH gene is responsible, which is what a scientist
    needs in order to act on the result.
    """
    seq = "".join(c for c in sequence.upper() if c in "ACGT")
    if len(seq) < CASSETTE_LEN // 2:
        return []

    window = CASSETTE_LEN
    step = max(1, CASSETTE_LEN // 3)
    windows = [
        kmer_features(seq[i : i + window])
        for i in range(0, max(1, len(seq) - window + 1), step)
    ]
    if not windows:
        windows = [kmer_features(seq)]
    W = np.vstack(windows)
    W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)

    hits: list[tuple[str, float]] = []
    for gene_name in RESISTANCE_GENES:
        ref = kmer_features(_gene_cassette(gene_name))
        ref = ref / (np.linalg.norm(ref) + 1e-9)
        best = float(np.max(W_norm @ ref))
        if best >= threshold:
            hits.append((gene_name, best))
    return sorted(hits, key=lambda t: -t[1])


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

    def resistant_classes(self, threshold: float = 0.5) -> list[str]:
        return [c for c, p in self.probabilities.items() if p >= threshold]

    def susceptible_classes(self, threshold: float = 0.5) -> list[str]:
        return [c for c, p in self.probabilities.items() if p < threshold]

    def last_resort_hits(self, threshold: float = 0.5) -> list[str]:
        return [c for c in self.resistant_classes(threshold) if c in LAST_RESORT]

    def risk_level(self) -> tuple[str, str]:
        """(label, colour) summarising how constrained the treatment options are."""
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


def analyze_isolate(
    sequence: str, model: AMRModel, strain_name: str = "Uploaded isolate", context: str = ""
) -> AMRReport:
    """Run the full Module 3 pipeline on a raw DNA sequence."""
    return AMRReport(
        strain_name=strain_name,
        probabilities=model.predict_sequence(sequence),
        detected_genes=detect_genes(sequence),
        context=context,
    )


def demo_strain_sequence(strain_name: str, seed: int = 7) -> str:
    """Reproducible sequence for a bundled demo strain."""
    strain = DEMO_STRAINS[strain_name]
    return synthesize_isolate(strain.genes, gc_content=strain.gc_content, seed=seed)
