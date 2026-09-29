"""
TrialSense — ingestion of real bacterial genomes with laboratory AST labels.

WHY THIS FILE EXISTS
--------------------
Every accuracy number Module 3 previously reported was measured on isolates
that `amr.synthesize_isolate()` constructed: a generated background sequence
with real resistance genes planted into it. That design tests whether the
pipeline can recover gene content we ourselves inserted. It cannot tell us
whether the pipeline predicts the resistance of a bacterium taken from a
patient.

Closing that gap needs two things joined on the same isolate:

    1. a real genome assembly, and
    2. a laboratory susceptibility result for THAT SAME isolate, measured at
       the bench (an MIC, or an S/I/R call derived from one).

A genome alone is not enough. Without a bench label there is nothing to
compare a prediction against.

WHERE THE DATA COMES FROM
-------------------------
BV-BRC (the Bacterial and Viral Bioinformatics Resource Center, formerly
PATRIC) publishes both halves and keys them to the same `genome_id`:

    genome_amr       antibiotic, phenotype, MIC, testing standard, citation
    genome_sequence  the assembled contigs

Note that BV-BRC's FTP host does not resolve from every network, including
ours. The REST API does. Everything here goes through the API.

THE TRAP THIS MODULE IS BUILT TO AVOID
--------------------------------------
The `genome_amr` table mixes two kinds of row, distinguished by its `evidence`
column:

    "Laboratory Method"      a bench measurement — broth dilution, disk
                             diffusion — usually with an MIC, a CLSI or EUCAST
                             standard, and a PubMed citation.
    "Computational Method"   somebody else's MACHINE LEARNING PREDICTION, with
                             their model's accuracy recorded alongside it.

Validating our model against "Computational Method" rows would mean scoring
our predictions against another model's predictions and reporting the
agreement as accuracy. Every query in this module filters to
`evidence == "Laboratory Method"`. Nothing else is ever treated as a label.

WHAT THIS MODULE WILL NOT DO
----------------------------
It will not generate, infer, or impute a susceptibility label. If the bench
data cannot be obtained, it reports the gap and the dependent validation does
not run. A fabricated label would silently convert an untested system into an
apparently validated one.

STATUS AS MEASURED ON 2026-09-29
--------------------------------
Run `python -m trialsense.ingest --probe` to re-measure.

    genome assemblies          OBTAINABLE   BV-BRC API, ~5 MB each
    laboratory AST phenotypes  OBTAINABLE   1,285,111 rows carry
                                            evidence = "Laboratory Method"

An earlier revision of this file recorded AST labels as unobtainable, on the
basis that NCBI Pathogen Detection's `AST/` directories return 404 and its
metadata table carries no susceptibility column. Both of those observations
are still true — see AST_SOURCES, which keeps probing them — but the
conclusion drawn from them was wrong, because BV-BRC serves the data NCBI no
longer does.
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path

from .amr import ANTIBIOTIC_CLASSES

# -----------------------------------------------------------------------------
# Where things land. Kept outside the package so a large download never ends up
# inside an installable directory.
# -----------------------------------------------------------------------------
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "real"
GENOME_DIR = DATA_ROOT / "genomes"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
LABELS_PATH = DATA_ROOT / "lab_ast_labels.json"
MISSING_REPORT_PATH = DATA_ROOT / "MISSING_DATA.md"

BVBRC_API = "https://www.bv-brc.org/api"
USER_AGENT = "TrialSense/1.0 (ved@axolotl.ai)"
REQUEST_TIMEOUT = 180
PAGE_SIZE = 5000

# Only bench measurements. See the module docstring.
LAB_EVIDENCE = "Laboratory Method"

# The organisms Module 3's surveillance layer already covers, so anything we
# ingest can be set beside a trend we already model. Keys are NCBI taxon IDs.
ORGANISMS: dict[int, str] = {
    562: "Escherichia coli",
    573: "Klebsiella pneumoniae",
    287: "Pseudomonas aeruginosa",
    470: "Acinetobacter baumannii",
    1280: "Staphylococcus aureus",
    1352: "Enterococcus faecium",
    28901: "Salmonella enterica",
}


# =============================================================================
# Mapping BV-BRC drug names onto our 12 classes.
#
# Only drugs whose class membership is unambiguous are included. Anything whose
# resistance behaviour does not follow its structural class is deliberately
# left out and listed in EXCLUDED_DRUGS with the reason, because a drug that
# breaks the class pattern injects label noise that looks like model error.
# =============================================================================

ANTIBIOTIC_TO_CLASS: dict[str, str] = {
    # Penicillins
    "ampicillin": "Penicillins",
    "penicillin": "Penicillins",
    "amoxicillin": "Penicillins",
    "amoxicillin/clavulanic acid": "Penicillins",
    "ampicillin/sulbactam": "Penicillins",
    "piperacillin": "Penicillins",
    "piperacillin/tazobactam": "Penicillins",
    "oxacillin": "Penicillins",
    "ticarcillin": "Penicillins",
    "ticarcillin/clavulanic acid": "Penicillins",
    "methicillin": "Penicillins",
    "carbenicillin": "Penicillins",
    # Cephalosporins
    "ceftriaxone": "Cephalosporins",
    "cefotaxime": "Cephalosporins",
    "ceftazidime": "Cephalosporins",
    "cefepime": "Cephalosporins",
    "cefoxitin": "Cephalosporins",
    "cefuroxime": "Cephalosporins",
    "cefazolin": "Cephalosporins",
    "cephalothin": "Cephalosporins",
    "ceftiofur": "Cephalosporins",
    "cefotetan": "Cephalosporins",
    # Carbapenems
    "meropenem": "Carbapenems",
    "imipenem": "Carbapenems",
    "ertapenem": "Carbapenems",
    "doripenem": "Carbapenems",
    # Fluoroquinolones
    "ciprofloxacin": "Fluoroquinolones",
    "levofloxacin": "Fluoroquinolones",
    "moxifloxacin": "Fluoroquinolones",
    "ofloxacin": "Fluoroquinolones",
    "norfloxacin": "Fluoroquinolones",
    "nalidixic acid": "Fluoroquinolones",
    # Macrolides
    "azithromycin": "Macrolides",
    "erythromycin": "Macrolides",
    "clarithromycin": "Macrolides",
    # Aminoglycosides
    "gentamicin": "Aminoglycosides",
    "amikacin": "Aminoglycosides",
    "tobramycin": "Aminoglycosides",
    "kanamycin": "Aminoglycosides",
    "streptomycin": "Aminoglycosides",
    "netilmicin": "Aminoglycosides",
    "neomycin": "Aminoglycosides",
    # Tetracyclines
    "tetracycline": "Tetracyclines",
    "doxycycline": "Tetracyclines",
    "minocycline": "Tetracyclines",
    # Trimethoprim-sulfonamides
    "trimethoprim/sulfamethoxazole": "Trimethoprim-sulfonamides",
    "trimethoprim": "Trimethoprim-sulfonamides",
    "sulfamethoxazole": "Trimethoprim-sulfonamides",
    "sulfisoxazole": "Trimethoprim-sulfonamides",
    "sulfathiazole": "Trimethoprim-sulfonamides",
    "sulfonamides": "Trimethoprim-sulfonamides",
    # Glycopeptides
    "vancomycin": "Glycopeptides",
    "teicoplanin": "Glycopeptides",
    # Oxazolidinones
    "linezolid": "Oxazolidinones",
    "tedizolid": "Oxazolidinones",
    # Polymyxins
    "colistin": "Polymyxins",
    "polymyxin b": "Polymyxins",
    # Rifamycins
    "rifampin": "Rifamycins",
    "rifampicin": "Rifamycins",
    "rifabutin": "Rifamycins",
}

EXCLUDED_DRUGS: dict[str, str] = {
    "tigecycline": "a glycylcycline; tet(M) and tet(A) do not defeat it, so grouping "
                   "it under Tetracyclines would label resistant isolates susceptible",
    "cefiderocol": "a siderophore cephalosporin taken up by an iron transporter; its "
                   "resistance pattern does not follow the cephalosporin class",
    "aztreonam": "a monobactam; stable to metallo-beta-lactamases that defeat every "
                 "carbapenem, and we model no monobactam class",
    "chloramphenicol": "phenicol class, which Module 3 does not cover",
    "isoniazid": "M. tuberculosis-specific; no class of ours and no organism of ours",
    "ethambutol": "M. tuberculosis-specific",
    "pyrazinamide": "M. tuberculosis-specific",
    "ethionamide": "M. tuberculosis-specific",
    "nitrofurantoin": "nitrofuran class, which Module 3 does not cover",
    "fosfomycin": "fosfomycin class, which Module 3 does not cover",
    "phosphomycin": "spelling variant of fosfomycin; same exclusion",
    "clindamycin": "lincosamide class, which Module 3 does not cover",
    "fusidic acid": "fusidane class, which Module 3 does not cover",
    "quinupristin/dalfopristin": "streptogramin class, which Module 3 does not cover",
    "mupirocin": "topical only; no systemic class of ours",
    "daptomycin": "lipopeptide class, which Module 3 does not cover",
    "spectinomycin": "an aminocyclitol; aminoglycoside-modifying enzymes largely do "
                     "not touch it, so grouping it under Aminoglycosides misleads",
    "ceftazidime/avibactam": "avibactam restores activity against KPC-producing "
                             "strains, so this reads Susceptible where the plain "
                             "cephalosporin reads Resistant",
    "ceftolozane/tazobactam": "same inhibitor-combination problem as above",

}

# CLSI "Intermediate" means the drug may work at higher dose or at a site where
# it concentrates. It is neither a clean positive nor a clean negative, so these
# rows are dropped rather than forced into one. Dropping is recorded in the
# manifest; it is not silent.
PHENOTYPE_MAP: dict[str, int] = {
    "resistant": 1,
    "susceptible": 0,
    "non-susceptible": 1,
    "susceptible-dose dependent": 0,
}
DROPPED_PHENOTYPES = {"intermediate", "not defined", "inconclusive", ""}


# =============================================================================
# HTTP plumbing
# =============================================================================


def _opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    op = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    op.addheaders = [("User-Agent", USER_AGENT)]
    return op


def _get(url: str, accept: str = "application/json",
         limit_bytes: int | None = None, retries: int = 4) -> bytes:
    """
    Fetch a URL, retrying transient failures.

    A long ingest makes hundreds of requests, and DNS resolution for this host
    intermittently fails for a second or two — observed in practice, not
    hypothetical. Without a retry a single blip aborts the whole run, so
    network-level errors back off and try again. HTTP 404 and other status
    errors are NOT retried: those are answers, not blips.
    """
    req = urllib.request.Request(url, headers={"Accept": accept,
                                               "User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with _opener().open(req, timeout=REQUEST_TIMEOUT) as resp:
                return resp.read(limit_bytes) if limit_bytes else resp.read()
        except urllib.error.HTTPError:
            raise  # the server answered; that answer is the result
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last if last else RuntimeError(f"unreachable: {url}")


def _rql(endpoint: str, *clauses: str, accept: str = "application/json") -> bytes:
    return _get(f"{BVBRC_API}/{endpoint}/?" + "&".join(clauses), accept=accept)


# =============================================================================
# Source probing. Kept so that "NCBI no longer serves this" stays evidence
# rather than a remembered assertion.
# =============================================================================


@dataclass
class AstSource:
    name: str
    url: str
    expects: str


AST_SOURCES: list[AstSource] = [
    AstSource("NCBI Pathogen Detection AST directory (Salmonella)",
              "https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Salmonella/latest_snps/AST/",
              "per-isolate MIC tables from NARMS"),
    AstSource("NCBI Isolates Browser bulk download",
              "https://www.ncbi.nlm.nih.gov/pathogens/isolates/api/download/?format=tsv&limit=5",
              "TSV with an AST phenotype column"),
    AstSource("BV-BRC FTP genome AMR table",
              "https://ftp.bvbrc.org/RELEASE_NOTES/PATRIC_genomes_AMR.txt",
              "genome_id / antibiotic / phenotype triples"),
    AstSource("BV-BRC API genome_amr (laboratory evidence only)",
              f"{BVBRC_API}/genome_amr/?eq(evidence,%22Laboratory%20Method%22)&limit(2)",
              "bench-measured susceptibility rows"),
]


@dataclass
class ProbeResult:
    source: str
    url: str
    ok: bool
    detail: str
    looks_like_data: bool = False

    def line(self) -> str:
        return f"  {'REACHED ' if self.ok else 'FAILED  '} {self.source}\n           {self.detail}"


def probe_ast_sources() -> list[ProbeResult]:
    """Visit every candidate AST source and record what actually came back."""
    out: list[ProbeResult] = []
    for src in AST_SOURCES:
        try:
            head = _get(src.url, accept="text/tsv", limit_bytes=4000)
        except urllib.error.HTTPError as exc:
            out.append(ProbeResult(src.name, src.url, False,
                                   f"HTTP {exc.code} — {src.expects} not served here"))
            continue
        except Exception as exc:
            out.append(ProbeResult(src.name, src.url, False,
                                   f"{type(exc).__name__}: {str(exc)[:90]}"))
            continue

        text = head.decode("utf-8", "ignore")
        if text.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
            out.append(ProbeResult(src.name, src.url, True,
                                   "HTML page, not tabular data — no labels here", False))
            continue
        has_pheno = bool(re.search(r"\b(resistant|susceptible|intermediate|MIC)\b", text, re.I))
        out.append(ProbeResult(
            src.name, src.url, True,
            "tabular payload with susceptibility fields" if has_pheno
            else "tabular payload, but no susceptibility field detected",
            has_pheno))
    return out


# =============================================================================
# Laboratory AST labels
# =============================================================================


@dataclass
class LabelledIsolate:
    genome_id: str
    genome_name: str
    taxon_id: int
    organism: str
    # class -> 0/1, aggregated from the individual drugs tested
    class_labels: dict[str, int] = field(default_factory=dict)
    # the raw bench rows behind each class call, kept for provenance
    evidence: list[dict] = field(default_factory=list)
    local_path: str = ""
    length_bp: int = 0
    n_contigs: int = 0


def fetch_lab_ast(taxon_id: int, max_rows: int = 20000) -> list[dict]:
    """
    Pull bench-measured AST rows for one organism.

    Filtered to evidence == "Laboratory Method" at the server, so computational
    predictions never enter the result set.
    """
    fields = ("genome_id,genome_name,taxon_id,antibiotic,resistant_phenotype,"
              "measurement,measurement_unit,laboratory_typing_method,"
              "testing_standard,pmid,evidence")
    rows: list[dict] = []
    start = 0
    while start < max_rows:
        page = min(PAGE_SIZE, max_rows - start)
        raw = _rql(
            "genome_amr",
            f'eq(evidence,%22{urllib.parse.quote(LAB_EVIDENCE)}%22)',
            f"eq(taxon_id,{taxon_id})",
            f"select({fields})",
            f"limit({page},{start})",
        )
        batch = json.loads(raw)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page:
            break
        start += page
        time.sleep(0.2)  # be polite to a free public API
    return rows


def aggregate_to_classes(rows: list[dict]) -> tuple[dict[str, LabelledIsolate], dict]:
    """
    Turn per-drug bench results into per-class labels, one record per genome.

    AGGREGATION RULE, and why:
        a class is Resistant if ANY drug tested in that class came back
        Resistant, otherwise Susceptible.

    Module 3 predicts at class level, and a resistance gene typically defeats
    part of a class rather than all of it — aac(6')-Ib defeats amikacin and
    tobramycin while sparing gentamicin. Requiring every drug to fail would
    therefore label a genuinely aminoglycoside-resistant isolate susceptible.
    "Any" matches what the class-level prediction is claiming.

    This is a real loss of resolution and it is recorded in the manifest.
    Per-drug evaluation is what perdrug.py exists for.
    """
    isolates: dict[str, LabelledIsolate] = {}
    per_class_votes: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    stats = {
        "rows_seen": len(rows),
        "dropped_unmapped_drug": 0,
        "dropped_intermediate": 0,
        "dropped_non_lab_evidence": 0,
        "used": 0,
        "unmapped_drug_names": defaultdict(int),
    }

    for r in rows:
        if r.get("evidence") != LAB_EVIDENCE:
            stats["dropped_non_lab_evidence"] += 1
            continue

        drug = (r.get("antibiotic") or "").strip().lower()
        cls = ANTIBIOTIC_TO_CLASS.get(drug)
        if cls is None:
            stats["dropped_unmapped_drug"] += 1
            if drug not in EXCLUDED_DRUGS:
                stats["unmapped_drug_names"][drug] += 1
            continue

        pheno = (r.get("resistant_phenotype") or "").strip().lower()
        if pheno in DROPPED_PHENOTYPES or pheno not in PHENOTYPE_MAP:
            stats["dropped_intermediate"] += 1
            continue

        gid = str(r.get("genome_id"))
        if gid not in isolates:
            isolates[gid] = LabelledIsolate(
                genome_id=gid,
                genome_name=r.get("genome_name", ""),
                taxon_id=int(r.get("taxon_id", 0) or 0),
                organism=ORGANISMS.get(int(r.get("taxon_id", 0) or 0), ""),
            )
        per_class_votes[gid][cls].append(PHENOTYPE_MAP[pheno])
        isolates[gid].evidence.append({
            "antibiotic": drug,
            "class": cls,
            "phenotype": r.get("resistant_phenotype"),
            "measurement": r.get("measurement"),
            "unit": r.get("measurement_unit"),
            "method": r.get("laboratory_typing_method"),
            "standard": r.get("testing_standard"),
            "pmid": r.get("pmid"),
        })
        stats["used"] += 1

    for gid, votes in per_class_votes.items():
        isolates[gid].class_labels = {c: (1 if any(v) else 0) for c, v in votes.items()}

    stats["unmapped_drug_names"] = dict(
        sorted(stats["unmapped_drug_names"].items(), key=lambda kv: -kv[1])[:15])
    stats["n_isolates"] = len(isolates)
    return isolates, stats


def download_genome(iso: LabelledIsolate, dest_dir: Path = GENOME_DIR) -> LabelledIsolate:
    """Fetch one genome's contigs as FASTA and record its true size."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / f"{iso.genome_id}.fna"
    if not local.exists():
        raw = _rql("genome_sequence",
                   f"eq(genome_id,{iso.genome_id})",
                   "limit(10000)",
                   accept="application/dna+fasta")
        local.write_bytes(raw)

    text = local.read_text("utf-8", errors="ignore")
    iso.n_contigs = text.count(">")
    iso.length_bp = sum(len(l.strip()) for l in text.splitlines() if not l.startswith(">"))
    iso.local_path = str(local.relative_to(DATA_ROOT.parent.parent))
    return iso


def select_isolates(isolates: dict[str, LabelledIsolate],
                    per_organism: int,
                    min_classes: int = 4) -> list[LabelledIsolate]:
    """
    Choose which isolates to download.

    Preference goes to isolates tested against the most classes, because an
    isolate with one bench result contributes one comparison while an isolate
    tested against eight contributes eight, for the same download.

    A resistant/susceptible balance is not engineered here. Doing so would
    hand-pick the test set, and the resulting rates would describe our
    selection rather than the organism.
    """
    by_org: dict[int, list[LabelledIsolate]] = defaultdict(list)
    for iso in isolates.values():
        if len(iso.class_labels) >= min_classes:
            by_org[iso.taxon_id].append(iso)

    chosen: list[LabelledIsolate] = []
    for taxon, group in by_org.items():
        group.sort(key=lambda i: (-len(i.class_labels), i.genome_id))
        chosen.extend(group[:per_organism])
    return chosen


# =============================================================================
# Manifest and reporting
# =============================================================================


def build_manifest(isolates: list[LabelledIsolate], stats: dict,
                   probes: list[ProbeResult]) -> dict:
    downloaded = [i for i in isolates if i.local_path]
    class_counts: dict[str, dict[str, int]] = {
        c: {"resistant": 0, "susceptible": 0} for c in ANTIBIOTIC_CLASSES}
    for iso in downloaded:
        for c, v in iso.class_labels.items():
            class_counts[c]["resistant" if v else "susceptible"] += 1

    return {
        "generated": date.today().isoformat(),
        "source": {
            "labels": "BV-BRC genome_amr, evidence == 'Laboratory Method' only",
            "genomes": "BV-BRC genome_sequence API",
            "url": BVBRC_API,
        },
        "data_classification": {
            "genomes": "real observed data",
            "ast_labels": "laboratory-confirmed labels (bench MIC / disk diffusion)",
            "excluded": "computational predictions from the same table were filtered out",
        },
        "counts": {
            "isolates_downloaded": len(downloaded),
            "total_bp": sum(i.length_bp for i in downloaded),
            "label_comparisons": sum(len(i.class_labels) for i in downloaded),
            "by_organism": {
                ORGANISMS.get(t, str(t)): sum(1 for i in downloaded if i.taxon_id == t)
                for t in {i.taxon_id for i in downloaded}
            },
            "by_class": {c: v for c, v in class_counts.items()
                         if v["resistant"] or v["susceptible"]},
        },
        "aggregation": {
            "rule": "class is Resistant if ANY drug tested in that class was Resistant",
            "intermediate_rows": "dropped, not coerced",
            "resolution_lost": "per-drug distinctions within a class; see perdrug.py",
        },
        "ingestion_stats": stats,
        "probes": [asdict(p) for p in probes],
        "usable_for": {
            "gene_detection_on_real_sequence": bool(downloaded),
            "novelty_scanning_on_real_sequence": bool(downloaded),
            "resistance_prediction_accuracy": bool(downloaded),
        },
        "isolates": [
            {k: v for k, v in asdict(i).items() if k != "evidence"} for i in downloaded
        ],
    }


def write_missing_report(manifest: dict) -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    c = manifest["counts"]
    got = c["isolates_downloaded"] > 0

    lines = [
        "# Real-data ingestion — what was obtained and what is missing",
        "",
        f"Generated {manifest['generated']} by `python -m trialsense.ingest`.",
        "",
        "## Obtained",
        "",
        f"- **{c['isolates_downloaded']} real genome assemblies** "
        f"({c['total_bp']:,} bp total) from BV-BRC.",
        f"- **{c['label_comparisons']} laboratory susceptibility labels** across "
        f"{len(c['by_class'])} antibiotic classes.",
        "",
        "  Labels are bench measurements only. Rows whose `evidence` column read",
        "  `Computational Method` — another group's machine-learning predictions —",
        "  were filtered out at the server, so no prediction is ever scored against",
        "  another prediction.",
        "",
        "  Isolates per organism:",
        "",
    ]
    for org, n in sorted(c["by_organism"].items()):
        lines.append(f"  - {org}: {n}")

    lines += ["", "## Known limits of this set", ""]
    if got:
        lines += [
            "- **Not a random sample.** Isolates were chosen for breadth of bench",
            "  testing, so they skew towards strains somebody had reason to test —",
            "  typically clinically interesting, often resistant ones. Resistance",
            "  rates here describe this set, not any patient population.",
            "- **Class-level, not drug-level.** A class counts as resistant if any",
            "  drug in it failed at the bench. Within-class detail is lost.",
            f"- **Intermediate results dropped**, not coerced: "
            f"{manifest['ingestion_stats']['dropped_intermediate']:,} rows.",
            "- **No Indian isolates were specifically selected.** BV-BRC geography is",
            "  uneven, so this set does not speak to Indian resistance patterns; the",
            "  ICMR surveillance layer remains the only Indian-specific evidence.",
        ]
    else:
        lines += ["- No isolates were obtained on this run; the validation that depends",
                  "  on them does not run, and no labels were invented to substitute."]

    lines += [
        "",
        "## What this permits us to claim",
        "",
        "| Claim | Permitted? |",
        "|---|---|",
        "| Validated on constructed isolates | yes, with 'constructed' stated |",
        f"| Gene detection works on real genomes | {'yes, once the real-genome check runs' if got else 'no'} |",
        f"| Measured against laboratory AST | {'yes, on this set, with its skew stated' if got else 'no'} |",
        "| Clinically validated | **no** |",
        "| Accurate on Indian patient isolates | **no — never measured** |",
        "",
        "The third row is the new one. It is not the same as the fourth: a bench",
        "result on a curated public strain set is evidence, but it is not a",
        "prospective clinical trial and must never be described as one.",
    ]
    MISSING_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MISSING_REPORT_PATH


def run(per_organism: int = 6, skip_download: bool = False,
        max_rows: int = 20000) -> dict:
    print("=" * 78)
    print("TrialSense — real genome + laboratory AST ingestion")
    print("=" * 78)

    print("\nProbing AST label sources...")
    probes = probe_ast_sources()
    for p in probes:
        print(p.line())

    print("\nFetching bench-measured AST rows (evidence = 'Laboratory Method')...")
    all_rows: list[dict] = []
    for taxon, name in ORGANISMS.items():
        try:
            rows = fetch_lab_ast(taxon, max_rows=max_rows)
            all_rows.extend(rows)
            print(f"  {name:26s} {len(rows):>7,} bench rows")
        except Exception as exc:
            print(f"  {name:26s} FAILED {type(exc).__name__}: {str(exc)[:60]}")

    isolates, stats = aggregate_to_classes(all_rows)
    print(f"\n  {stats['rows_seen']:,} rows -> {stats['used']:,} usable "
          f"-> {stats['n_isolates']:,} isolates with class labels")
    print(f"  dropped: {stats['dropped_unmapped_drug']:,} drug not in our classes, "
          f"{stats['dropped_intermediate']:,} intermediate/undefined")

    selected = select_isolates(isolates, per_organism)
    print(f"\n  selected {len(selected)} isolates "
          f"(>=4 classes tested, richest first)")

    downloaded: list[LabelledIsolate] = []
    if not skip_download:
        print("\nDownloading genomes...")
        for iso in selected:
            try:
                downloaded.append(download_genome(iso))
                print(f"  {iso.organism:26s} {iso.genome_id:14s} "
                      f"{iso.length_bp:>10,} bp  {len(iso.class_labels)} classes")
            except Exception as exc:
                print(f"  {iso.genome_id}: FAILED {type(exc).__name__}: {str(exc)[:60]}")

    manifest = build_manifest(downloaded or selected, stats, probes)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LABELS_PATH.write_text(json.dumps(
        {i.genome_id: {"organism": i.organism, "name": i.genome_name,
                       "class_labels": i.class_labels, "evidence": i.evidence}
         for i in (downloaded or selected)}, indent=2), encoding="utf-8")
    report = write_missing_report(manifest)

    print(f"\nmanifest -> {MANIFEST_PATH}")
    print(f"labels   -> {LABELS_PATH}")
    print(f"missing  -> {report}")
    print(f"\nisolates with bench labels : {manifest['counts']['isolates_downloaded']}")
    print(f"label comparisons available: {manifest['counts']['label_comparisons']}")
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Ingest real genomes with laboratory AST labels from BV-BRC.")
    ap.add_argument("--probe", action="store_true",
                    help="probe sources and build labels, but download no genomes")
    ap.add_argument("--per-organism", type=int, default=6,
                    help="genomes to download per organism (default 6)")
    ap.add_argument("--max-rows", type=int, default=20000,
                    help="max AST rows to pull per organism (default 20000)")
    args = ap.parse_args()

    run(per_organism=args.per_organism, skip_download=args.probe,
        max_rows=args.max_rows)
