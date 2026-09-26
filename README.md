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

python train.py            # ~3 min on a laptop CPU; caches models to models/
python validate_module3.py # ~80 s; writes the validation figures
streamlit run app.py       # opens the demo
```

`train.py` is optional — the app trains on first launch if no cache exists, it
just takes longer to open. It is **required** for the out-of-distribution badge,
which needs the training-distribution stats it records.

Run `python smoke_test.py` to verify the whole pipeline end to end (**51
checks**, no browser needed).

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

**The 21 reference gene sequences are real.** They come from the NCBI Bacterial
Antimicrobial Resistance Reference Gene Database (AMRFinderPlus DB 2026-08-07.1)
— the same curated set NCBI's own tool uses — with gyrA and rpoB from RefSeq.
Accessions are in each FASTA header under `trialsense/reference_sequences/`.

**Two layers, and the precise one wins.** Exactly as in Module 1, a
knowledge-driven layer (find the gene, look up what it defeats) sits alongside a
statistical one (read the whole k-mer profile). Where gene detection fires, it
overrides the classifier — it is measurably the more reliable of the two, and
deferring to it removes every very-major error on the bundled strains.

**Six further analyses sit on top of that core scan. Each produces its own
independent report, with its own verdict and its own download** — deliberately
not merged, because a single number would hide exactly the disagreements a
portfolio team needs to see.

| # | Feature | The question it answers | Nobody else does this because… |
|---|---|---|---|
| 1 | **Resistance Runway** | Will this class still work at the **launch year**, not today? | Existing tools are built for a clinician treating a patient now, not a company committing a decade of spend |
| 2 | **Rescue Strategy** | Resistant — but is it *fixable*? | Class-level reporting collapses metallo- and serine-carbapenemases into one verdict with opposite business outcomes |
| 3 | **Portfolio Screen** | Which of our N candidates survives across M organisms? | Every competing tool is isolate-centric; committees decide at portfolio level |
| 4 | **Dark Genome** | Is something here we *cannot name*? | Gene detectors are closed-world — their silence is indistinguishable from "all clear" |
| 5 | **Value of Detection** | What is finding this early actually worth? | The output is a scientific fact; funding decisions need rupees |
| 6 | **Spread Velocity** | How fast is this resistance moving? | Resistance is reported as a static yes/no, never as a rate |

#### Feature 1 — Resistance Runway

Fits a logistic trajectory to **real ICMR AMRSN national surveillance
(2016–2024)** and projects it to a user-specified launch year.

Logistic, not linear, because resistance is bounded: *K. pneumoniae* carbapenem
resistance has risen ~4 percentage points a year since 2016, and a straight line
extended to 2035 gives over 110%. Fitted by least squares in **logit space**
rather than a non-linear optimiser — with nine annual points a `curve_fit` can
fail to converge or settle on a wild asymptote.

The 2015 ICMR figures are shown but **held out of every fit**: 2015 covered
blood and CSF isolates only, while 2016+ covers all specimens. Joining them
creates an artefact (MRSA reads 42.6% in 2015 and 28.4% in 2016 — both correct,
measuring different things).

Source disagreement is displayed rather than resolved. For *K. pneumoniae*
carbapenem resistance in 2023, ICMR reports ~62–64%, WHO GLASS 61.2%, NCDC
~49–51%. Picking the most alarming one silently would be the easy move.

#### Feature 2 — Rescue Strategy

Every one of the 21 determinants carries a `mechanism_family`, `enzyme_subtype`,
and `rescue_viable` annotation. The distinction that matters:

| Gene | Mechanism | Rescuable? |
|---|---|---|
| `blaKPC-2` | Serine carbapenemase (class A) | ✅ avibactam / vaborbactam / relebactam |
| `blaOXA-48` | Serine oxacillinase (class D) | ⚠️ avibactam yes, vaborbactam **no** |
| **`blaNDM-1`** | **Metallo (class B, zinc)** | ❌ **no** — serine-enzyme inhibitors do not touch it |
| `mecA` | Target modification (PBP2a) | ❌ no enzyme to inhibit; ceftaroline is the exception |
| `aac(6')-Ib` | Drug modification | ✅ plazomicin is stable to it |
| `armA` | 16S rRNA methylation | ❌ defeats all aminoglycosides, plazomicin included |
| `blaVIM` / `blaIMP` | **Metallo (class B, zinc)** | ❌ **no** — same reason as blaNDM-1 |
| `blaSHV` | Serine ESBL (class A) | ✅ avibactam / tazobactam |

`blaNDM-1` and `blaKPC-2` produce the *identical* class-level verdict. One is a
formulation change; the other is a programme decision. `blaVIM` and `blaIMP`
carry different gene names and the same business consequence as `blaNDM-1` —
which is the argument for reporting mechanism rather than gene label.

#### Feature 4 — Dark Genome detector

Finds compositionally foreign segments matching **none** of the references —
the open-world counterpart to gene detection.

Three things were necessary to make it work, each discovered by measurement:
known-gene windows are **masked first** (a resistance cassette is itself
compositionally foreign, so without masking the clean strains scored *higher*
than deliberately spiked ones); the background is estimated from the unexplained
windows with **iterative trimming**; and detection requires **two floors**, since
the robust z-score alone is unstable when few windows remain.

#### Feature 6 — Spread Velocity

Plasmid, integron, transposon, SCCmec or chromosomal per determinant, feeding a
multiplier back into Feature 1. The multipliers are reasoned orders of magnitude,
not fitted constants, and the UI says so.

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
- **Mechanism and mobility annotations.** The metallo-vs-serine β-lactamase
  distinction, which inhibitors cover which enzyme class, plazomicin's stability
  to `aac(6')-Ib` but not `armA`, plasmid vs chromosomal inheritance — all
  documented textbook microbiology, encoded not invented.
- **The resistance surveillance data.** Every figure in `surveillance.py` was
  read from a primary source and is attributed: ICMR AMRSN annual reports
  2015–2024 and the WHO Global Health Observatory API. Nothing is interpolated,
  smoothed or filled in — a missing year is absent, not estimated. ICMR prints
  % susceptible from 2016; we store `100 − S`, which is ICMR's own convention,
  verified against three verbatim trend statements in the 2024 report.
- **All 21 resistance-gene sequences.** Genuine published coding sequences from
  the NCBI Bacterial Antimicrobial Resistance Reference Gene Database
  (AMRFinderPlus DB 2026-08-07.1), the same curated set NCBI's own tool uses,
  with gyrA and rpoB from RefSeq. Each FASTA header carries its accession.
  Every sequence passed QC on retrieval: one record, only A/C/G/T, length
  divisible by three, valid start and stop codons, no internal stops.
- **The ML pipelines and their evaluation**, including deliberately harsh
  cold-split, clade-held-out and time-split protocols.

### Simulated or simplified

- **Interaction labels** come from our curated mechanism knowledge base, **not**
  DrugBank or TWOSIDES — both require licensing we could not obtain in the
  hackathon window. Severity thresholds are our own calibration.
- **Isolate scaffolds are constructed.** The 21 resistance-gene sequences are
  now real (see the Real list above), but they are placed into *generated*
  species-typical background rather than complete annotated genomes, which are
  a much larger download than a demo should carry. So the **determinants are
  real and the scaffold is not** — and because we assign the gene content, the
  labels are ours rather than lab-confirmed. Replacing the background with
  NCBI Pathogen Detection genomes, which ship with real susceptibility results
  attached, is the remaining step to fully real data.
- **The hepatic impairment factor** is our own heuristic from ALT, albumin and
  bilirubin. Real practice uses Child-Pugh, which needs clinical assessment a
  form field cannot capture.
- **Risk weights** in Module 2 and the 60/40 composite split are reasoned, not
  fitted to outcome data.
- **Linear kinetics assumed.** Phenytoin is famously non-linear, so its true
  exposure change is *under*-estimated here.
- **Not modelled:** pharmacogenomics (CYP2D6/2C19 metaboliser status),
  transporters beyond P-gp, protein-binding displacement, disease-state effects.
- **The resistance-runway projection** is an extrapolation, drawn as a dashed
  line with a widening cone precisely so it cannot be mistaken for observation.
  Its backtested error is 4.93 pp — real, but not a guarantee.
- **The cost model in Feature 5** is a transparent decision-analysis model over
  user-adjustable assumptions, **not** a measured saving. Nobody has run a drug
  programme through this tool for ten years. The claim we defend is narrow: the
  screen costs approximately nothing and the alternative costs at minimum one
  failed phase — which holds regardless of the specific numbers.
- **The spread multipliers in Feature 6** are reasoned orders of magnitude, not
  fitted constants.
- ~~**Class-level gene mapping is too coarse.**~~ **Fixed.** The classifier
  still predicts twelve classes, but a per-drug resolution layer
  (`trialsense/perdrug.py`) now reports which individual agents a detected
  determinant actually defeats — and which it leaves alone. Both CARD and NCBI
  confirm `aac(6')-Ib` covers **amikacin, kanamycin and tobramycin but NOT
  gentamicin**, while `armA` covers the whole class including gentamicin. Two
  genes, identical class label, opposite consequences. The app now says so.
- **A labelling error we found and corrected.** The rifampicin determinant was
  labelled `rpoB_S531L`. S531 is *E. coli* rpoB numbering; the *M. tuberculosis*
  hotspot is codon **450** (Ser450Leu). Renamed to `rpoB_S450L` throughout.
- **Point mutations are not resolved.** `gyrA_S83L` and `rpoB_S450L` are
  detected via their **wild-type** reference genes, because a single-codon
  substitution is invisible to k-mer composition. Detecting them means "the
  locus is present", not "the resistant allele is present". Resolving that needs
  read-level variant calling.

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
| Random split | 0.900 | 95.1% | 58.9% |
| **Cold-combination split** (unseen gene combinations) | **0.841** | 94.3% | 51.1% |

Holding macro-F1 on gene combinations never seen in training shows the model
detects genes *compositionally* rather than memorising whole-isolate
fingerprints — the property that determines whether it works on a new field
isolate. Gene identification on the 12 bundled strains: **34/34 genes found,
zero false positives**, including a correctly clean susceptible control.

### Swapping to real sequences made these numbers worse. We kept the swap.

|  | synthetic cassettes | real NCBI sequences |
|---|---|---|
| macro-F1 (random) | 0.989 | **0.900** |
| macro-F1 (cold-combination) | 0.925 | **0.841** |
| exact match (random) | 0.891 | **0.589** |

That drop is the honest measurement of how much synthetic data had been
flattering us. Real genes share bacterial codon usage, real families overlap,
and three metallo-beta-lactamases with near-identical class profiles are
genuinely hard to separate — none of which was true of cassettes generated to be
distinguishable.

It also forced a recalibration worth recording: the gene-detection threshold
moved from **0.55 to 0.83**, because at 0.55 real references matched nearly
everything. Measured across the twelve strains, the weakest true match scores
0.883 and the strongest false match 0.763, so any threshold in 0.77–0.88
separates them perfectly; 0.83 is the midpoint.

**The deployed pipeline nonetheless got better**, because gene detection on real
sequences is exact and now overrides the classifier — see the validation below.

### Module 3 — the six features, validated

Accuracy alone does not answer *"does this help an R&D team?"*. Seven separate
checks, all reproduced by `python validate_module3.py`.

#### 1. Error types, reported separately — and which layer they belong to

Regulators do not score susceptibility devices on accuracy. They count two
error types, because the costs are wildly asymmetric.

**The classifier alone**, on held-out synthetic-background isolates at the
shipped 0.35 screening threshold:

| | Rate |
|---|---|
| Very major (predicted S, actually R) | 7.41% |
| Major (predicted R, actually S) | 3.65% |

**The deployed pipeline**, which is classifier + gene detection + mechanism
floor, across all 12 bundled strains (144 class calls):

| | Very major | Major | Correct |
|---|---|---|---|
| Classifier alone | 5 | 0 | 96.5% |
| **+ gene detection override** | **0** | **0** | **100.0%** |

The gene layer removes **5 very-major errors and introduces
0**. It can only ever raise a probability, and it fires only where a
real reference sequence matches, so it cannot invent danger.

⚠️ **The bundled strains are not a held-out test set** — we chose their genes.
Treat the 100% as an integration check; the clade-held-out figure below is the
honest generalisation number.

#### 2. Calibration — measured, because we changed it

A model can be accurate and still dishonest about its own confidence. Ours was:
before calibration, when it said 55% the observed rate was 97%. Since
`report.py` multiplies this probability into the composite score, that
understatement propagated into a funding decision.

Each per-class forest is now wrapped in isotonic regression fitted on
cross-validated out-of-fold predictions. Shipping that without measuring it
would be the same unexamined-claim problem this suite exists to catch, so:

| | raw forest | calibrated |
|---|---|---|
| Brier score | 0.0699 | **0.0375** |
| Expected calibration error | 0.1002 | **0.0145** |
| Major errors @0.35 | 10.01% | **3.65%** |

#### 3. Clade-held-out validation

An entire species background removed from training, tested only on it —
population-structure control, rather than a random split that lets
near-identical isolates land on both sides.

**macro-F1 0.904 ± 0.021** across 8 held-out clades, worst clade
**0.883**.

#### 3b. Two thresholds, because there are two questions

Clinical susceptibility testing reports three categories, not two. We do the
same, for the same reason — forcing a borderline result into one of two buckets
throws information away.

| Band | Meaning | Used for |
|---|---|---|
| below 0.35 | Susceptible | — |
| 0.35 – 0.50 | **Equivocal** | screen and confirm, do not act |
| 0.50 and above | Resistant | headline risk level, class counts |

This was not theoretical. Dropping to a single 0.35 threshold pushed the
*susceptible control strain* to "Moderate" on one class scoring 37%, while gene
detection found nothing in it at all. Two layers disagreeing at the margin is
exactly what an equivocal band is for.

#### 4. Dark Genome detector

| | Result |
|---|---|
| False positives | **0 of 12** clean demo strains |
| Sensitivity | **7 of 7** planted unknown elements found |
| Separation | clean peak 0.118 vs spiked minimum 0.232 |

The detection floor sits between those with better than 2× margin either side.

#### 5. Forecast backtest — the number that makes Feature 1 real

Fitted on ICMR years **≤ 2020 only**, then asked to predict 2021–2024, which it
never saw.

| | Result |
|---|---|
| Predictions | 184 across 46 organism × antibiotic series |
| Mean absolute error | **4.93 percentage points** |
| Median absolute error | 3.55 pp |
| Within 5 pp / 10 pp | **64% / 85%** |

Worst series: *K. pneumoniae* × ciprofloxacin (17.2 pp), *P. aeruginosa* ×
imipenem (15.5 pp).

#### 6. Feature 3 measured against real ICMR national data

Feature 3 produces no data of its own — it orchestrates the core classifier —
so without this it would have no validation claim beyond *"the grid has the
right number of cells"*. Every prediction is laid against the real ICMR
national figure for the same organism and antibiotic class.

| | Result |
|---|---|
| Pairings with an ICMR reference | **49** (2024) |
| Track the national rate (within 15 pp) | **14** |
| Differ in the direction we expect | **34** |
| Flagged for review | **16** |
| No ICMR reference published | 27 — reported, not skipped |
| Mean gap across aligned pairings | **5.1 pp** |

**The two numbers measure different things, and the code knows it.** Our grid
says *"this isolate carries determinants predicting resistance"*; ICMR says
*"X% of all isolates nationally tested resistant"*. A strain built to be
resistant *should* read above a national average that mixes resistant and
susceptible isolates. Each comparison is therefore judged against its
**expected direction** — carrying a determinant means we expect to read high, a
susceptible control means we expect to read low — and only the unexpected ones
are flagged. Scoring every difference as failure would mislead as badly as
scoring every difference as success.

Diverging in an unexpected direction (1 pairings):

| Pairing | Model | ICMR | Gap |
|---|---|---|---|
| *Staphylococcus aureus* × Glycopeptides | 28% | 0% | +28 pp |

The remaining flags are wide-but-expected gaps: the XDR *S.* Typhi, VRE and
*cfr*-carrying strains are deliberately extreme, so reading far above a national
average that is mostly ordinary isolates is correct, not an error.

#### 7. Coverage of India-relevant determinants

ICMR molecular surveillance (369 *E. coli* + 374 *K. pneumoniae*, 7 centres)
reports 8 determinants. **We cover all 8**: blaTEM-1 (54% prevalence),
blaCTX-M-15 (40%), blaNDM-1 (27%), blaOXA-48 group (22%), blaVIM (19%), blaSHV
(16%), blaKPC-2 (15%), blaIMP (15%).

`blaVIM`, `blaSHV` and `blaIMP` were added *because* this check flagged them
missing — the reference set went from 18 determinants to 21. Adding two further
metallo-carbapenemases whose class profile overlaps blaNDM-1 makes the task
strictly harder, and macro-F1 held at 0.922 while exact-match fell from 0.851 to
0.760. That trade is the honest one to report: per-class discrimination is
unchanged, getting all twelve labels simultaneously right is harder.

Two rows remain approximations, both documented in `amr.py`: the OXA figure is
for the OXA-1 group rather than OXA-48 specifically, and ICMR reports SHV at
family level without separating narrow- from extended-spectrum variants, so we
model the extended-spectrum case and over-call cephalosporin resistance for an
isolate carrying only SHV-1.

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
2. ~~**Real sequences** — replace `_gene_cassette()` with references from NCBI
   or CARD.~~ **Done.** All 21 determinants now use real NCBI AMRFinderPlus
   reference sequences. The remaining half of this step is replacing the
   generated isolate *background* with complete NCBI Pathogen Detection genomes,
   which would also supply lab-confirmed labels instead of ones we assign.
3. **Validated hepatic scoring** — swap the ALT/albumin heuristic for Child-Pugh
   or MELD where the clinical inputs exist.
4. **Stronger models** — the Random Forests are deliberately small so they train
   in minutes on a laptop. With real data at scale, a graph neural network over
   molecular graphs is the natural upgrade for Module 1.

---

## Project layout

```
app.py                    Streamlit UI (5 tabs; Module 3 has 7 sub-tabs)
train.py                  Trains + caches both models, writes metrics.json
validate_module3.py       The proof layer — writes models/validation.json
smoke_test.py             51 end-to-end checks, no browser required
capture_screens.py        Headless-browser screenshots + live render check
trialsense/
  drugs.py                39 drugs: structures + pharmacology (self-validating)
  ddi.py                  Module 1 — mechanism KB + structure-only ML filter
  pk.py                   Module 2 — PK equations + risk escalation
  amr.py                  Module 3 core — k-mer pipeline, model, the two
                          decision thresholds, the mechanism floor, and the
                          mechanism/mobility annotations for all 21 determinants
  reference_sequences/    The 21 REAL gene FASTAs from NCBI AMRFinderPlus,
                          plus CARD/NCBI per-drug mappings
  surveillance.py         Feature 1 — real ICMR/WHO data + logistic forecaster
  mechanisms.py           Features 2 & 6 — rescue strategy, spread velocity
  portfolio.py            Feature 3 — N x M candidate screening
  novelty.py              Feature 4 — open-world detector + OOD badge
  economics.py            Feature 5 — value-of-information model
  perdrug.py              Per-drug resolution within a class (CARD + NCBI)
  dnabert.py              Optional DNABERT-2 benchmark (CPU, degrades safely)
  report.py               Unified composite report, findings, protocol actions
  cases.py                Five preloaded demo scenarios
  viz.py                  Molecule rendering, charts, UI components
notebooks/
  dnabert2_finetune_kaggle.ipynb   GPU fine-tuning on Kaggle's free tier
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
