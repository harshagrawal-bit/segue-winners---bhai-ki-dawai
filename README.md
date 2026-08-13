# TrialSense

**Computational pre-trial risk screening for drug candidates.**
Segue 3.0 prototype — an R&D decision-support tool for pharmaceutical teams.

Drug development fails most expensively *after* the money is committed: during
clinical trials, from drug–drug interactions, subgroup-specific toxicity, and —
for antibacterials — resistance that was already widespread before the trial
began. TrialSense screens for all three computationally, before human studies
start, and produces a single **candidate risk report** a portfolio team can act on.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py        # ~2.5 min on a laptop CPU; caches models to models/
streamlit run app.py   # opens the demo
```

`train.py` is optional — the app trains on first launch if no cache exists, it
just takes longer to open. Run `python smoke_test.py` to verify the whole
pipeline end to end (18 checks, no browser needed).

**Five preloaded cases** are in the sidebar dropdown. Pick *Case 1* for the
strongest single demo.

---

## What it does

### Module 1 — Drug–drug interaction

Two deliberately independent layers:

| Layer | Input | Output | Works on novel compounds? |
|---|---|---|---|
| **A. Mechanism knowledge base** | Pharmacology profiles | Severity 0–3 **+ plain-English mechanism** | No |
| **B. ML screening filter** | Molecular structure only | P(clinically significant) | **Yes** |

Layer B never sees the pharmacology profile — only Morgan/ECFP4 fingerprints,
MACCS keys and physicochemical descriptors. It is therefore genuinely learning
to infer *"this molecule probably inhibits CYP3A4"* from structure, which is what
lets it score a compound that has no profile at all. Keeping the layers separate
is what makes the held-out evaluation meaningful.

Pair features are built with commutative operations only (bitwise union /
intersection, sum / absolute difference), so A+B and B+A are byte-identical —
otherwise the model can memorise argument order and the evaluation silently
inflates. This is asserted in the smoke test.

### Module 2 — Personalised toxicity simulation

Recomputes the Module 1 severity for a specific patient subgroup:

```
CrCl        = (140 − age) × weight ÷ (72 × serum creatinine)   [× 0.85 if female]
CL_ratio    = fe × KF + (1 − fe) × HF        (renal + hepatic clearance fractions)
AUC_ratio   = 1 ÷ CL_ratio                   (exact for linear kinetics)
```

Then applies named organ-specific hazards — nephrotoxic drug meeting an impaired
kidney, potassium-raising drugs meeting a kidney that cannot excrete potassium,
QT-prolonging drugs meeting low potassium, metformin below CrCl 30, sedatives in
the over-75s.

This is the module that carries the pitch: **the same pairing is Moderate in an
average adult and Severe in the elderly renal subgroup.** Pooling those patients
into one trial arm is how a safety signal gets diluted until Phase III.

### Module 3 — Antimicrobial resistance intelligence

```
DNA sequence → 5-mer frequency vector (1024 features) → multi-label Random Forest
             → resistance call per antibiotic class (12 classes)
```

Plus a sliding-window k-mer similarity search that names *which* resistance gene
is responsible — the mechanism layer, mirroring Module 1's structure. Accepts a
real FASTA upload; the pipeline is unchanged.

### Unified report

Composite 0–100 score = 60% interaction & toxicity + 40% resistance (resistance
is dropped and the interaction component used alone when neither compound is an
antibacterial). The blend is a transparent weighted combination, not a learned
model, and every contribution is shown separately in the UI — a portfolio
decision needs a number whose provenance can be explained line by line.

---

## What is real vs. simulated

Judges should be able to assess this accurately, so here it is plainly.

### Real

- **All 39 molecular structures.** Real published SMILES. Every one is verified
  at build time by recomputing its molecular weight with RDKit against the
  literature value — `train.py` refuses to proceed on a mismatch, so a typo
  cannot silently reach the demo. (This caught two genuine errors during
  development: digoxin missing an oxygen, azithromycin an extra CH₂.)
- **Fingerprints, descriptors, k-mer featurisation.** Genuine RDKit / NumPy.
- **Pharmacology profiles.** CYP450 roles, P-glycoprotein handling, fraction
  renally excreted, risk flags — standard clinical pharmacology.
- **Interaction mechanisms.** Every mechanism described is real and documented.
  Calibration was checked against **24 well-known pairs** with published
  severities (warfarin + fluconazole, simvastatin + clarithromycin, theophylline
  + ciprofloxacin, …): **24/24 match**.
- **PK equations.** Cockcroft-Gault and clearance-weighted organ scaling are the
  standard equations behind real renal dose-adjustment guidance.
- **Gene → resistance mapping.** Real microbiology (blaNDM-1 → carbapenems,
  mecA → β-lactams, vanA → glycopeptides, mcr-1 → colistin, …).
- **The ML pipelines and their evaluation**, including deliberately harsh
  cold-split protocols.

### Simulated or simplified

- **Interaction labels** come from our curated mechanism knowledge base, **not**
  DrugBank or TWOSIDES — both require licensing we could not obtain in the
  hackathon window. Severity thresholds are our own calibration.
- **Genomic sequences are fully synthetic.** Each resistance gene is a
  deterministic marker cassette built from gene-specific signature motifs —
  **not** the real published sequence of blaNDM-1, mecA, etc. We chose not to
  ship approximate biological sequences presented as authentic. Absolute AMR
  accuracy therefore reflects a cleaner signal than real sequencing data, which
  carries noise, contamination and incomplete assembly.
- **The hepatic impairment factor** is our own heuristic from ALT, albumin and
  bilirubin. Real practice uses Child-Pugh, which needs clinical assessment a
  form field cannot capture.
- **Risk weights** in Module 2 and the 60/40 composite split are reasoned, not
  fitted to outcome data.
- **Linear kinetics assumed.** Phenytoin is famously non-linear, so its true
  exposure change is *under*-estimated here.
- **Not modelled:** pharmacogenomics (CYP2D6/2C19 metaboliser status),
  transporters beyond P-gp, protein-binding displacement, disease-state effects.

---

## Measured performance

All numbers are held-out and reproduced by `python train.py`.

### Module 1 — interaction screening filter (structure only)

| Split | ROC-AUC | Recall | Precision |
|---|---|---|---|
| Random pair split | 0.814 | 0.76 | 0.38 |
| **Cold-drug split** (drugs removed entirely) | **0.679 ± 0.060** | 0.61 | 0.28 |

Pooled over 10 independent drug hold-outs (n = 2,760 test pairs). Trained on 741
pairs from 39 drugs, 19.7% of which are clinically significant.

**Read this honestly.** The cold-drug number is the one that matters — it is how
the filter would perform on a genuinely new compound — and 0.68 is a real signal
well short of production quality. Two deliberate choices follow from that:

1. **It is a binary filter, not a severity predictor.** We built and measured a
   4-class severity model first: 55% accuracy, 0.30 macro-F1 on held-out drugs.
   That is not honest to present as a severity prediction, so we did not ship it.
   With 39 drugs there simply is not enough data. Severity and mechanism come
   from the knowledge base; the ML layer answers only *"is this worth expert
   review?"*
2. **Precision is low on purpose.** At the 35% flag threshold the filter catches
   61% of significant pairs while flagging 41% of all pairs. For pre-trial
   screening a missed interaction costs far more than an unnecessary review.

The ±0.060 spread across hold-outs is wide because 39 drugs is a small set —
which is exactly why the evaluation pools several hold-outs and reports a
standard deviation rather than quoting one flattering number.

### Module 3 — resistance model

| Split | Macro-F1 | Per-label accuracy | Exact match |
|---|---|---|---|
| Random split | 0.921 | 97.9% | 85.1% |
| **Cold-combination split** (unseen gene combinations) | **0.924** | 96.7% | 66.7% |

Holding macro-F1 on gene combinations never seen in training shows the model
detects genes *compositionally* rather than memorising whole-isolate
fingerprints — the property that determines whether it works on a new field
isolate. Gene identification on the 12 bundled strains: **34/34 genes found,
zero false positives**, including a correctly clean susceptible control.

The caveat above still applies: these sequences are synthetic and cleaner than
real data.

---

## The five demo cases

| # | Scenario | What it demonstrates |
|---|---|---|
| 1 | Lisinopril + Trimethoprim, elderly renal, ESBL *E. coli* | **Subgroup escalation** — Moderate → Severe; composite 57 → 84 |
| 2 | Simvastatin + Clarithromycin, MRSA | A documented severe interaction recovered from structure |
| 3 | Meropenem + Furosemide, carbapenem-resistant *K. pneumoniae* | **Portfolio decision** — the target organism already defeats the class |
| 4 | Warfarin + Rifampicin, MDR-TB | **Enzyme induction** — failure looks like "the drug didn't work", not toxicity |
| 5 | Amoxicillin + Metformin, susceptible control | **Correctly cleared** — a tool that flags everything is worthless |

---

## Path to production

The architecture was built so the simulated parts are *swappable*, not
load-bearing:

1. **Real interaction labels** — replace `build_dataset()` in
   `trialsense/ddi.py` with a licensed DrugBank/TWOSIDES loader. Featurisation,
   model and evaluation are unchanged. This is also what would lift Module 1
   from "real signal" to "useful": the bottleneck is 39 drugs, not the method.
2. **Real sequences** — replace `_gene_cassette()` in `trialsense/amr.py` with
   references from NCBI Pathogen Detection or CARD, then retrain.
3. **Validated hepatic scoring** — swap the ALT/albumin heuristic for Child-Pugh
   or MELD where the clinical inputs exist.
4. **Stronger models** — the Random Forests are deliberately small so they train
   in minutes on a laptop. With real data at scale, a graph neural network over
   molecular graphs is the natural upgrade for Module 1.

---

## Project layout

```
app.py                    Streamlit UI (5 tabs)
train.py                  Trains + caches both models, writes metrics.json
smoke_test.py             18 end-to-end checks, no browser required
capture_screens.py        Headless-browser screenshots + live render check
trialsense/
  drugs.py                39 drugs: structures + pharmacology (self-validating)
  ddi.py                  Module 1 — mechanism KB + structure-only ML filter
  pk.py                   Module 2 — PK equations + risk escalation
  amr.py                  Module 3 — sequence synthesis, k-mer pipeline, model
  report.py               Unified composite report, findings, protocol actions
  cases.py                Five preloaded demo scenarios
  viz.py                  Molecule rendering, charts, UI components
assets/screenshots/       Captured UI screens
```

---

## Positioning

TrialSense is an **R&D pre-screening tool for pharmaceutical teams**. It produces
candidate risk reports to inform protocol design and portfolio decisions before
expensive human trials begin.

It is **not** a patient-facing application, not a diagnostic, and not a clinical
decision-support system. **No output here should be used to treat anyone.**

Built for Segue 3.0. Python · RDKit · scikit-learn · Streamlit · Plotly — all
free and open source, running entirely locally with no paid APIs.
