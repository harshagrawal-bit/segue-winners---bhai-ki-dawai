"""
TrialSense — Module 3: optional DNABERT-2 comparison path.

WHY THIS IS A BENCHMARK AND NOT A REPLACEMENT
---------------------------------------------
DNABERT-2 is a genomic language model pretrained on real bacterial genomes. Its
advantage comes from having learned the statistical structure of real DNA —
codon usage, conserved domains, the grammar of actual coding sequence.

Our bundled sequences are synthetic marker cassettes assembled from randomly
generated motifs. There is no real biology in them for a language model to
recognise, while the k-mer counter is near-perfectly matched to how those
cassettes were constructed.

We therefore expect DNABERT-2 to UNDERPERFORM the k-mer baseline on this data,
and we run the comparison anyway, because the result is informative either way:

    If k-mer wins  -> measured evidence that the synthetic data, not the
                      method, is the binding constraint. Swapping in real
                      NCBI/CARD sequences is the upgrade that matters, and we
                      can now say that with a number instead of a hunch.

    If DNABERT wins -> the transformer is extracting structure the k-mer
                       counter misses, and the upgrade path is immediate.

Reporting a benchmark you might lose is a stronger position than reporting an
integration you cannot evaluate.

RUNNING IT
----------
CPU-only inference. No GPU required, nothing to rent.

    pip install torch transformers einops

The first call downloads roughly 500 MB of checkpoint. Embeddings are cached to
models/dnabert_cache.npz, so the cost is paid once.

KNOWN FRICTION
--------------
DNABERT-2's published remote code imports Triton flash-attention, which is
CUDA-only. On CPU this raises at import. `load_model()` catches that and
reports it as an unavailable backend rather than letting it reach the UI —
nothing in this module is allowed to break the demo.

For real fine-tuning (as opposed to frozen-feature extraction), see
notebooks/dnabert2_finetune_kaggle.ipynb, which runs on Kaggle's free GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODEL_ID = "zhihan1996/DNABERT-2-117M"
MAX_TOKENS = 512          # model context; sequences are chunked to fit
CHUNK_BASES = 2000        # ~512 BPE tokens at DNABERT-2's average token length
CACHE_PATH = Path(__file__).parent.parent / "models" / "dnabert_cache.npz"


@dataclass
class BackendStatus:
    """Whether the optional DNABERT-2 path can run, and why not if it cannot."""

    available: bool
    reason: str = ""
    torch_version: str = ""
    device: str = "cpu"

    def message(self) -> str:
        if self.available:
            return (
                f"DNABERT-2 backend ready (torch {self.torch_version}, "
                f"{self.device})."
            )
        return f"DNABERT-2 backend unavailable — {self.reason}"


def check_backend() -> BackendStatus:
    """Probe for torch/transformers without importing the heavy model."""
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return BackendStatus(
            False,
            "PyTorch is not installed. Run `pip install torch transformers einops` "
            f"to enable the comparison. ({type(exc).__name__})",
        )
    try:
        import transformers  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return BackendStatus(
            False,
            "the transformers library is not installed. Run "
            f"`pip install transformers einops`. ({type(exc).__name__})",
        )
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BackendStatus(True, torch_version=torch.__version__, device=device)


def chunk_sequence(sequence: str, size: int = CHUNK_BASES) -> list[str]:
    """Split a sequence into model-sized pieces. Real genomes exceed the context."""
    seq = "".join(c for c in sequence.upper() if c in "ACGT")
    if not seq:
        return []
    return [seq[i : i + size] for i in range(0, len(seq), size)] or [seq]


class DNABertEmbedder:
    """
    Frozen-feature extractor: mean-pooled DNABERT-2 embeddings per sequence.

    Deliberately NOT a fine-tuned classifier. Fine-tuning a 117M-parameter model
    needs a CUDA GPU, which the target machine does not have, and doing it badly
    on CPU would produce a worse number than not doing it at all. Frozen
    embeddings plus a light classifier head is the standard, honest way to
    evaluate a pretrained encoder on a small dataset.
    """

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.status = check_backend()

    def load(self) -> BackendStatus:
        if self.model is not None:
            return self.status
        if not self.status.available:
            return self.status
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self.model.eval()
            torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
        except Exception as exc:  # noqa: BLE001 — must never reach the UI
            msg = str(exc)
            if "triton" in msg.lower() or "flash" in msg.lower():
                reason = (
                    "DNABERT-2's published code requires Triton flash-attention, "
                    "which is CUDA-only and cannot load on CPU. This is a known "
                    "upstream issue, not a fault in this project. Use the Kaggle "
                    "notebook for a GPU run."
                )
            else:
                reason = f"{type(exc).__name__}: {msg[:200]}"
            self.status = BackendStatus(False, reason)
        return self.status

    def embed(self, sequence: str) -> np.ndarray | None:
        """Mean-pooled embedding for one sequence, averaged over its chunks."""
        if self.load().available is False:
            return None
        import torch

        chunks = chunk_sequence(sequence)
        if not chunks:
            return None

        vectors = []
        with torch.no_grad():
            for chunk in chunks:
                enc = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_TOKENS,
                )
                out = self.model(**enc)
                hidden = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
                vectors.append(hidden.mean(dim=1).squeeze(0).cpu().numpy())
        return np.mean(np.vstack(vectors), axis=0)

    def embed_many(self, sequences: list[str], progress=None) -> np.ndarray | None:
        rows = []
        for i, seq in enumerate(sequences):
            v = self.embed(seq)
            if v is None:
                return None
            rows.append(v)
            if progress is not None:
                progress(i + 1, len(sequences))
        return np.vstack(rows)


@dataclass
class BenchmarkResult:
    """Head-to-head comparison of the two featurisations."""

    ran: bool = False
    reason: str = ""
    n_samples: int = 0
    kmer_macro_f1: float = 0.0
    dnabert_macro_f1: float = 0.0
    kmer_per_label_acc: float = 0.0
    dnabert_per_label_acc: float = 0.0
    kmer_dim: int = 0
    dnabert_dim: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def winner(self) -> str:
        if not self.ran:
            return "not run"
        if abs(self.kmer_macro_f1 - self.dnabert_macro_f1) < 0.01:
            return "tie"
        return "k-mer" if self.kmer_macro_f1 > self.dnabert_macro_f1 else "DNABERT-2"

    def interpretation(self) -> str:
        if not self.ran:
            return (
                f"The DNABERT-2 comparison did not run: {self.reason} The k-mer "
                "pipeline is unaffected — this is an optional benchmark, not a "
                "dependency."
            )
        if self.winner == "k-mer":
            return (
                f"The k-mer baseline wins on this data (macro-F1 "
                f"{self.kmer_macro_f1:.3f} vs {self.dnabert_macro_f1:.3f}), "
                "which is exactly what theory predicts. A genomic language "
                "model's advantage comes from real biological sequence "
                "structure, and our cassettes are synthetic — there is nothing "
                "for it to recognise, while the k-mer counter matches how they "
                "were built. This is measured evidence that our DATA is the "
                "binding constraint, not our method. On real NCBI sequences we "
                "would expect the ordering to reverse."
            )
        if self.winner == "tie":
            return (
                f"The two featurisations are within noise of each other "
                f"(macro-F1 {self.kmer_macro_f1:.3f} vs "
                f"{self.dnabert_macro_f1:.3f}). On synthetic cassettes neither "
                "has a real edge, which is itself a reason to prioritise real "
                "sequence data over a larger model."
            )
        return (
            f"DNABERT-2 embeddings beat the k-mer baseline (macro-F1 "
            f"{self.dnabert_macro_f1:.3f} vs {self.kmer_macro_f1:.3f}) even on "
            "synthetic sequences, which suggests it is extracting structure the "
            "k-mer counter misses. The transformer path is worth pursuing."
        )


def run_benchmark(
    n_samples: int = 120,
    seed: int = 42,
    progress=None,
) -> BenchmarkResult:
    """
    Train the same light classifier head on both featurisations and compare.

    Kept small by default (120 isolates): the point is a fair head-to-head on
    identical data, not a production training run, and CPU transformer
    inference is the bottleneck.
    """
    import time

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputClassifier

    from .amr import _realistic_gene_combinations, kmer_features, labels_for_genes, synthesize_isolate

    result = BenchmarkResult()
    embedder = DNABertEmbedder()
    status = embedder.load()
    if not status.available:
        result.reason = status.reason
        return result

    started = time.perf_counter()
    rng = np.random.RandomState(seed)
    combos = _realistic_gene_combinations(rng, n_samples)
    gc_values = [0.327, 0.380, 0.390, 0.507, 0.521, 0.572, 0.656, 0.664]

    sequences, Y = [], []
    for genes in combos:
        gc = gc_values[rng.randint(len(gc_values))]
        sequences.append(
            synthesize_isolate(genes, gc_content=gc, seed=int(rng.randint(1 << 30)))
        )
        Y.append(labels_for_genes(genes))
    Y = np.vstack(Y)

    X_kmer = np.vstack([kmer_features(s) for s in sequences])
    X_bert = embedder.embed_many(sequences, progress=progress)
    if X_bert is None:
        result.reason = "embedding failed partway through"
        return result

    # Drop labels with a single class present — f1 is undefined and it would
    # silently favour whichever featurisation happened to see more variety.
    keep = [j for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
    if not keep:
        result.reason = "no label had both classes present in this sample"
        return result
    Yk = Y[:, keep]

    idx_tr, idx_te = train_test_split(
        np.arange(len(sequences)), test_size=0.3, random_state=seed
    )

    def score(X):
        clf = MultiOutputClassifier(
            LogisticRegression(max_iter=2000, class_weight="balanced")
        )
        clf.fit(X[idx_tr], Yk[idx_tr])
        pred = np.asarray(clf.predict(X[idx_te]))
        return (
            float(f1_score(Yk[idx_te], pred, average="macro", zero_division=0)),
            float((pred == Yk[idx_te]).mean()),
        )

    result.kmer_macro_f1, result.kmer_per_label_acc = score(X_kmer)
    result.dnabert_macro_f1, result.dnabert_per_label_acc = score(X_bert)
    result.kmer_dim = int(X_kmer.shape[1])
    result.dnabert_dim = int(X_bert.shape[1])
    result.n_samples = len(sequences)
    result.seconds = time.perf_counter() - started
    result.ran = True
    result.notes.append(
        f"Both featurisations trained the identical classifier head "
        f"(multi-label logistic regression) on the identical {len(idx_tr)}/"
        f"{len(idx_te)} split of {len(sequences)} isolates, scored over "
        f"{len(keep)} labels with both classes present."
    )
    return result
