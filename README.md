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

Run `python smoke_test.py` to verify the whole pipeline end to end (**57
checks**, no browser needed), and `python validate_pk.py` to score Module 2
against real renal impairment data.

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
eGFR        = CKD-EPI 2021 (creatinine, age, sex)          mL/min/1.73 m²
abs eGFR    = eGFR × BSA ÷ 1.73                            mL/min (Du Bois BSA)
KF          = min(1, abs eGFR ÷ 90)          (90 = lower edge of FDA "normal")
CL_ratio    = fe × KF + (1 − fe) × HF        (renal + hepatic clearance fractions)
AUC_ratio   = 1 ÷ CL_ratio                   (exact for linear kinetics)
```

CKD-EPI and absolute eGFR follow FDA's 2024 renal impairment guidance.
Cockcroft-Gault CrCl is still shown for reference, because older drug labels
state their renal cut-offs in CrCl.

Then applies named organ-specific hazards — nephrotoxic drug meeting an impaired
kidney, potassium-raising drugs meeting a kidney that cannot excrete potassium,
QT-prolonging drugs meeting low potassium, metformin below eGFR 30, sedatives in
the over-75s.

**Trial design output.** For each drug, `renal_study_plan()` predicts exposure
for a typical mild / moderate / severe patient and recommends a **full** or
**reduced** renal impairment study, plus Phase 2/3 enrolment advice per
category — the planning decision a clinical pharmacology team actually spends
money on. It follows the structure of the FDA guidance; the cut-offs (fe ≥ 0.3
= major renal route; 1.25× and 2× exposure bands) are our own heuristics.

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
  Calibration is checked against **27 well-known pairs** with published
  severities, reproducibly — run `python -m trialsense.known_pairs`:

  | Measure | Result |
  |---|---|
  | Exact severity match | 21/27 (78%) |
  | Within one severity level | **27/27** (100%) |
  | Agree on clinically significant or not | **27/27** (100%) |
  | **Under-calls** (we say safe, published says dangerous) | **0** |

  All 6 disagreements are over-calls — the safe direction for a
  screening tool, costing an unnecessary review rather than a missed hazard. The
  set includes negative controls (azithromycin + simvastatin, levofloxacin +
  theophylline) chosen as near-misses of pairs that *are* dangerous.

  **This check found a real defect.** On its first run it scored methotrexate +
  trimethoprim as Mild against a published Severe, because additive folate
  blockade was not modelled at all. A rule was added to `ddi.py`, guarded so it
  does not fire on co-trimoxazole, where the same combined action is the
  therapeutic intent.
- **PK equations.** CKD-EPI 2021 eGFR (de-indexed by body surface area),
  Cockcroft-Gault and clearance-weighted organ scaling are the standard
  equations behind real renal dose-adjustment guidance.
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
  fitted to outcome data. The same goes for the renal study-plan cut-offs.
- **Kidney disease is assumed not to affect liver clearance.** In reality
  severe renal impairment also suppresses some liver enzymes and transporters,
  so exposure for mostly non-renal drugs is *under*-estimated in severe
  impairment. (This is why the study plan still recommends a reduced study for
  them.)
- **Linear kinetics assumed.** Phenytoin is famously non-linear, so its true
  exposure change is *under*-estimated here.
- **Not modelled:** pharmacogenomics (CYP2D6/2C19 metaboliser status),
  transporters beyond P-gp, protein-binding displacement, disease-state effects.
- **The resistance-runway projection** is an extrapolation, drawn as a dashed
  line with a widening cone precisely so it cannot be mistaken for observation.
  Its backtested error is 4.93 pp overall — but that average hides its own
  structure. On series with six or more annual observations the error is
  **0.48 pp**; on series with fewer it is **5.13 pp**, and all three worst
  series have five points. Forecasts now carry an `n_points` count and a
  `reliability` flag so a thin projection is visible as thin. The 95%
  prediction intervals were measured to cover **82.6%** of held-out actuals,
  not 95% — treat them as roughly 80% bands. This is a mean absolute error in
  percentage points, NOT an accuracy percentage.
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

### Module 2 — validated against real renal impairment data

`python validate_pk.py` compares Module 2's predicted exposure change with what
was actually measured in patients with impaired kidneys. The data
(`data/renal_validation.json`) come from FDA labels and published studies, and
every number carries a verbatim quote and URL. Anything that could not be
scored fairly is kept in the file with a written reason: half-life-only data,
dialysis-only groups, acute kidney injury, metabolite-only data and simulated
values.

| Measure | Result |
|---|---|
| Measured exposure changes scored | 25, across 17 drugs (+ 9 "no change" label statements) |
| Within 2× / 1.5× of measured | 88% / 64% |
| Geometric mean fold error | 1.46 |
| Bias (predicted ÷ measured) | 0.74 — predictions run low |
| "No change" statements reproduced | 9 of 9 |
| **Study-design call correct** | **16 of 17 drugs — 0 full studies missed, 1 unnecessary** |

**Read this honestly.** The exposure numbers under-predict for kidney-cleared
drugs, mostly in moderate impairment. They miss 7 of 13 measured changes of
2× or more: six are called 1.25–2×, and one (ciprofloxacin, mild impairment,
measured 2.3×) is called "no change". Kidney disease also slows the liver,
which the additive model ignores, and the source studies disagree with each
other (three ciprofloxacin studies give 1.75× to 4.3× for similar groups).
Moving the "normal kidney" reference from 90 to 120 barely helps (GMFE 1.46 →
1.40, and 1.43 when tuned on the other drugs and tested on the held-out one),
so we did not change it.

The **study-design decision** is the output that holds up. It picked the design
the real data justify for 16 of 17 drugs, and never recommended a reduced study
where the kidney turned out to matter — the error that costs a trial. The smoke
test guards that property.

### Module 3 — resistance model

| Split | Macro-F1 | Per-label accuracy | Exact match |
|---|---|---|---|
| Random split | 0.916 | 95.9% | 64.6% |
| **Cold-combination split** (unseen gene combinations) | **0.763** | 92.3% | 44.0% |

**These are CONSTRUCTED isolates.** On real genomes with laboratory
susceptibility results the model scores 0.526 — approximately chance. See
"What happens on real bacteria" below, and read that section before quoting
any number in this table.

Holding macro-F1 on gene combinations never seen in training shows the model
detects genes *compositionally* rather than memorising whole-isolate
fingerprints — the property that determines whether it works on a new field
isolate. Gene identification on the 12 bundled strains: **34/34 genes found,
zero false positives**, including a correctly clean susceptible control.

Gene detection is also the one layer that survives contact with real genomes:
on 42 real assemblies it found genuine determinants — mecA in 9 real
*S. aureus*, blaSHV in 5, plus blaNDM-1, blaVIM and blaCTX-M-15 — at a 13.8%
false-resistance rate. It is precise but narrow: 21 reference genes cannot
cover real resistance, so it misses 73% of laboratory-confirmed resistance.

### What happens on real bacteria

Every figure above is measured on isolates we constructed. The honest test is
real genomes with bench susceptibility results, and Module 3 was measured
against 42 of them (350 laboratory labels, BV-BRC, `evidence = "Laboratory
Method"` only — other groups' machine-learning predictions were filtered out
so we never score a prediction against a prediction).

| Test set | Accuracy | Very major | Major |
|---|---|---|---|
| Constructed isolates | 0.927 | 1.1% | 9.7% |
| **Real genomes + laboratory AST** | **0.526** | **47.2%** | **47.7%** |

With 176 resistant and 174 susceptible comparisons, **0.526 is chance**, and no
threshold rescues it — the full sweep peaks at 0.503.

The reason is architectural, not a tuning problem. The model trains on ~900 bp
cassettes planted in ~2.6 kb of background; a real genome is 2.5–7 Mb, where a
determinant is about 0.02% of the sequence, so a whole-genome 5-mer profile is
dominated by ordinary housekeeping DNA.

**So the defensible claim is narrow:** Module 3 detects 21 known resistance
genes in real genomes with high precision and resolves two point-mutation
determinants to the specific codon. It does not predict laboratory phenotype
on real bacteria. Reproduce with `python validate_real.py`; full write-up in
[reports/AUDIT_RESPONSE.md](reports/AUDIT_RESPONSE.md).

### Swapping to real sequences made these numbers worse. We kept the swap.

|  | synthetic cassettes | real NCBI sequences |
|---|---|---|
| macro-F1 (random) | 0.989 | **0.916** |
| macro-F1 (cold-combination) | 0.925 | **0.763** |
| exact match (random) | 0.891 | **0.646** |

The cold-combination figure fell again, from 0.841 to 0.763, when gene
sampling was reweighted to real ICMR prevalence. Rare determinants became
rare, which is harder and more honest.

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

**The classifier alone**, on held-out constructed isolates at the shipped
0.10 screening threshold:

| | Rate |
|---|---|
| Very major (predicted S, actually R) | 3.15% |
| Major (predicted R, actually S) | 14.39% |

The threshold is derived, not chosen: `derive_threshold.py` states an explicit
cost — one missed resistance equals ten unnecessary confirmatory assays — and
minimises expected cost over a sweep. It moved from 0.35 to 0.10, which brings
the very-major rate to the ~1.5% order commercial devices are held to on a
300-isolate held-out set. The previous 0.35 did not meet that, and the comment
justifying it quoted numbers that a retrain had invalidated.

**The deployed pipeline**, which is classifier + gene detection + mechanism
floor, across all 12 bundled strains (144 class calls):

| | Very major | Major | Correct |
|---|---|---|---|
| Classifier alone | 3 | 1 | 97.2% |
| **+ gene detection override** | **0** | 1 | **99.3%** |

The gene layer removes **3 very-major errors and introduces
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
| Expected calibration error (current model) | — | **0.017** |

#### 3. Clade-held-out validation

An entire species background removed from training, tested only on it —
population-structure control, rather than a random split that lets
near-identical isolates land on both sides.

**macro-F1 0.897 ± 0.021** across 8 held-out clades, worst clade
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

**The previous claim here — 0 false positives, 7 of 7 sensitivity — was
withdrawn.** All 19 of those cases came from the same generator that produced
the training data, so it showed the detector could separate one generator's
two modes. Tested on 42 real genomes, the shipped thresholds flagged something
in **every single one, at 21.75 segments per megabase** — roughly 100 flags on
a 5 Mb genome. Real genomes carry prophages, genomic islands and rRNA operons;
generated "clean" strains are uniform by construction.

Recalibrated by splitting the 42 genomes in half, sweeping thresholds on one
half and reporting on the other. **Held-out half:**

| thresholds | false alarms | sensitivity to a real foreign insert |
|---|---|---|
| 8.0 / 0.18 (previous) | 21.75 /Mb | 100% |
| **25.0 / 0.40 (shipped)** | **1.57 /Mb** | **52%** |

Sensitivity is measured with a real 1200 bp segment taken from the most
GC-distant genome in the set and inserted into a real recipient — both donor
and recipient are real sequence, so this does not repeat the original mistake
of testing a generator against itself.

**The trade-off is real and is not tuned away.** The frontier runs from 100%
sensitivity at ~22 flags/Mb to ~24% at 0.5 flags/Mb. Tetranucleotide
composition cannot distinguish an acquired element from a native genomic
island, because both are compositionally foreign to the host core. We chose
about eight flags per genome — reviewable by a person — and accept that this
misses roughly half of true insertions. Reproduce with
`python validate_novelty_real.py` and `python calibrate_novelty.py`.

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
| 1 | Lisinopril + Trimethoprim, elderly renal, ESBL *E. coli* | **Subgroup escalation** — Moderate → Severe; composite 64 → 82 |
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
validate_pk.py            Scores Module 2 against real renal impairment data
data/renal_validation.json  Sourced renal PK data (quote + URL per number)
smoke_test.py             57 end-to-end checks, no browser required
capture_screens.py        Headless-browser screenshots + live render check
trialsense/
  drugs.py                39 drugs: structures + pharmacology (self-validating)
  ddi.py                  Module 1 — mechanism KB + structure-only ML filter
  pk.py                   Module 2 — PK equations, risk escalation, study plan
  pk_validation.py        Module 2 — scoring against real renal data
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
