# TrialSense Module 3 — response to the technical audit

**Date:** 2026-09-29
**Scope:** all ten audit findings and all ten remediation priorities.
**Tests:** 72 passed, 0 failed.

---

## The headline finding

Module 3 was measured, for the first time, against **real bacterial genomes with laboratory-measured susceptibility results**. It does not predict them.

| Test set | Accuracy | Very major errors | Major errors |
|---|---|---|---|
| Constructed isolates | 0.927 | 1.1% | 9.7% |
| **Real genomes + laboratory AST** | **0.526** | **47.2%** | **47.7%** |

350 comparisons across 42 real isolates, near-balanced (176 resistant, 174 susceptible), so **0.526 is approximately chance.**

This confirms the audit's central claim. The old numbers measured whether the pipeline could recover gene content we ourselves planted, not whether it predicts resistance.

**Why it fails:** the model trains on ~900 bp cassettes planted in ~2.6 kb of generated background. A real genome is 2.5–7 Mb, where a determinant is ~0.02% of the sequence, so a whole-genome 5-mer profile is dominated by housekeeping DNA. **No threshold fixes this** — the sweep finds 0.503 at best across the entire range.

### Which layer works

| Layer | Accuracy | Very major | Major |
|---|---|---|---|
| Classifier + mechanism floor | 0.526 | 47.2% | 47.7% |
| **Gene detection alone** | 0.563 | 73.3% | **13.8%** |

Gene detection is **precise but narrow**. When it fires it is usually right, and it found real determinants in real assemblies — mecA in 9 *S. aureus*, blaSHV in 5, blaNDM-1, blaVIM, blaCTX-M-15. But 21 reference genes cannot cover real resistance, so it misses 73%.

---

## Audit findings

All ten verified against live code; **all ten accurate**, every cited number reproduced exactly.

| # | Finding | Outcome |
|---|---|---|
| 1 | Isolates generated, not real | **Fixed** — 42 real genomes, 350 bench labels |
| 2 | Threshold contradicts its own sweep | **Fixed** — derived from a stated cost model |
| 3 | Mechanism floor over-calls | **Fixed** — floor now measured, not asserted |
| 4 | Point mutations fire on locus, not allele | **Fixed** — codon is read directly |
| 5 | `locate_genes` O(genes × windows) | **Fixed** — 8.8× faster |
| 6 | Coarse gene→class mapping | **Fixed for blaSHV**; aac(6')-Ib defensible as-is |
| 7 | Uniform synthetic prevalence | **Fixed** — ICMR-weighted |
| 8 | Forecast tail hidden by MAE | **Fixed** — intervals + thin-series flag |
| 9 | DNABERT-2 unevaluated | **Blocked upstream** — see below |
| 10 | Novelty tuned on 19 same-generator cases | **Fixed** — recalibrated on real genomes |

**One the audit missed:** `amr.py` justified its threshold with `VME 1.45%` when the live value was **7.41%** — the comment predated a retrain, so the justification for the most important parameter was false.

---

## What changed

### 1. Real genomes and laboratory AST — `trialsense/ingest.py` (new)

42 real assemblies (188 MB, 7 organisms) and 350 bench labels across all 12 classes, balanced R/S, from BV-BRC.

**The trap avoided.** BV-BRC mixes bench measurements with *other groups' ML predictions*, flagged in an `evidence` column. Scoring against those would report agreement with another model as accuracy. Every query filters to `evidence == "Laboratory Method"`, and a test enforces it.

NCBI's `AST/` directories return 404 and its metadata carries no susceptibility column — an earlier revision concluded from this that bench labels were unobtainable. **That conclusion was wrong**; BV-BRC serves what NCBI no longer does. The probe is retained so the claim stays evidence.

### 2. Point-mutation allele specificity — `amr.py`

Every *E. coli* carries gyrA and our reference is wild-type K-12, so susceptible isolates matched at ~0.999 and were floored to 0.89. `read_allele()` now anchors on the conserved flank and reads the codon, on either strand. `UNDETERMINED` gets no floor — the floor asserts resistance, and we do not assert what we did not measure.

Validated against bench data: **10/12 agree**, where previously all 12 would be called resistant. The two disagreements are honest — one is real biology (S83L alone gives only reduced susceptibility), and six Salmonella read `undetermined` because our reference is *E. coli*.

This also exposed that `synthesize_isolate` planted **wild-type** gyrA while labelling it resistant — training data that contradicted itself. Fixed.

### 3. blaSHV cephalosporin claim — `amr.py`

Our reference is SHV-1, the ancestral narrow-spectrum penicillinase most *Klebsiella* carry chromosomally; only extended-spectrum variants defeat third-generation cephalosporins. The claim now requires the Gly238Ser codon, located via the conserved K-T-G motif at Ambler 234–236.

**Not changed:** `aac(6')-Ib` → Aminoglycosides is defensible at class level (it does defeat amikacin, tobramycin, kanamycin, sparing only gentamicin), and the lab labels are themselves class-level. `perdrug.py` handles the within-class detail.

### 4. Evidence-based mechanism floor — `derive_mechanism_floor.py` (new)

The floor asserted 0.90. We can now measure what it was guessing at:

> P(laboratory resistant | gene conferring that class detected) = **0.71**, not 0.90

Per-class, shrunk toward the pooled rate so three-for-three does not read as certainty: Penicillins 0.65, Cephalosporins 0.66, Carbapenems 0.61, Aminoglycosides 0.78, Trimethoprim-sulfonamides 0.87.

Point mutations are excluded from that table and get per-gene values, applied only after the codon is confirmed: **rpoB S450L 0.90** (used clinically as a rifampicin marker — Xpert MTB/RIF is built on it), **gyrA S83L 0.60** (single QRDR substitution gives only reduced susceptibility).

*Limits: 3–16 observations per class, from isolates selected for breadth of testing. Indicative, not precise — but measured rather than asserted.*

### 5. Threshold derived from a stated cost — `derive_threshold.py` (new)

A threshold is meaningless without saying how much worse a missed resistance is than a false alarm. We state it: **one very major error = ten unnecessary confirmatory assays**, and minimise `10 × P(VME) + P(ME)`.

| threshold | very major | major | accuracy | cost |
|---|---|---|---|---|
| 0.05 | 1.1% | 14.7% | 0.892 | 0.254 |
| **0.10** | **1.1%** | **9.7%** | 0.927 | **0.204** |
| 0.35 (old) | 4.7% | 3.0% | 0.965 | 0.504 |
| 0.50 | 7.7% | 1.4% | 0.968 | 0.780 |

Shipped threshold moved **0.35 → 0.10**, which meets the ~1.5% very-major target commercial devices are held to. The old value did not. `validation.json` now reports `recommended_threshold: None` — the sweep no longer disagrees with what ships.

Note accuracy is *highest* at 0.50: optimising accuracy on an imbalanced problem just predicts the majority class, which here means calling resistant isolates susceptible. That is why the cost model exists.

### 6. `locate_genes` — 8.8× faster

One shared prefix-sum table of k-mer counts per block replaces 21 independent sliding scans, then candidate peaks are re-scored exactly. **11.9 s → 1.35 s** per real genome; the real-data validation went 1197 s → 324 s.

Verified identical: same gene sets on all 12 demo strains and 5 real genomes, with scores never lower than the exhaustive scan's.

### 7. ICMR-weighted synthesis

Uniform sampling made blaKPC-2 as common as blaTEM-1; ICMR puts them at 15% and 54%. Genes are now sampled at measured prevalence, with the 13 genes ICMR did not genotype given the median of the measured ones rather than an invented figure.

Cold-combination macro-F1 moved **0.841 → 0.763** — an honest drop from a harder, more realistic training distribution.

### 8. Forecast uncertainty

Replaced an ad-hoc cone (fixed 1.96, arbitrary 18%/year widening) with the textbook prediction interval and Student's *t* on *n*−2 df, which widens properly when data is thin.

The split that matters:

| | MAE |
|---|---|
| Series with ≥6 points | **0.48 pp** |
| Series with <6 points | **5.13 pp** |

The headline 4.93 pp is almost entirely short series — all three worst are n=5. Forecasts now carry `n_points` and a `reliability` flag.

**Measured interval coverage: 82.6% against a nominal 95%.** Reported rather than widened to hit the target, since that would be fitting to the test. Treat the bands as roughly 80%.

### 9. Novelty recalibrated on real genomes

The prior claim — 0% false positives, 100% sensitivity — came from 19 cases from the same generator as the training data. On 42 real genomes the shipped thresholds flagged **something in every single one, at 21.75 segments per megabase**: roughly 100 flags on a 5 Mb genome.

Recalibrated by splitting the genomes in half, sweeping on one and reporting on the other. **Held-out test half:**

| thresholds | false alarms | sensitivity |
|---|---|---|
| 8.0 / 0.18 (old) | 21.75 /Mb | 100% |
| **25.0 / 0.40 (shipped)** | **1.57 /Mb** | **52%** |

**The trade-off is real and is not tuned away.** The frontier runs from 100% sensitivity at ~22 flags/Mb to ~24% at 0.5 flags/Mb. Tetranucleotide composition cannot separate an acquired element from a native genomic island, because both are foreign to the host core. We chose ~8 flags per genome, reviewable by a person, and accept missing about half of true insertions.

### 10. DNABERT-2 — blocked upstream, not skipped

PyTorch 2.14 and transformers 5.17 were installed and the backend reports available. The benchmark still cannot run: DNABERT-2's published modeling code hard-requires **Triton flash-attention, which is CUDA-only**, and the import fails on CPU. Confirmed directly with a minimal forward pass.

This is an upstream constraint, not a gap in our work. Recorded in `models/dnabert_benchmark.json` with `ran: false` and the reason. It needs a GPU; the Kaggle notebook path remains the way to run it.

---

## Files

**New:** `trialsense/ingest.py`, `validate_real.py`, `validate_novelty_real.py`, `calibrate_novelty.py`, `derive_mechanism_floor.py`, `derive_threshold.py`, `data/real/`, `models/{validation_real,novelty_calibration,mechanism_floor,threshold_derivation,dnabert_benchmark}.json`

**Modified:** `trialsense/amr.py` (allele reading, conditional classes, empirical floors, threshold, fast `locate_genes`, prevalence weighting, synthesis fix), `trialsense/surveillance.py` (prediction intervals, reliability, coverage), `trialsense/novelty.py` (recalibrated thresholds, `scan_statistics`), `smoke_test.py` (+9), `.gitignore`

`models/metrics.json` and `models/validation.json` were regenerated; the constructed-data baseline is preserved in this document, because the gap between the two is the finding.

## Tests added (9)

Computational predictions never become labels · Intermediate dropped not coerced · saved labels class-valid · wild-type gyrA does not trigger the floor · S83L does · reverse strand works · unreadable allele asserts nothing · ancestral SHV-1 makes no cephalosporin claim · SHV Gly238Ser does.

## Data classification

| Source | Use | Kind |
|---|---|---|
| BV-BRC genome_sequence | 42 assemblies | real observed |
| BV-BRC genome_amr (lab evidence only) | 350 labels | laboratory-confirmed |
| NCBI RefSeq | 21 reference genes | real observed |
| ICMR AMRSN 2016–2024 | forecasting, prevalence weights | population surveillance — **not isolate labels** |
| `synthesize_isolate` | 700 training isolates | **constructed synthetic** |

## Still missing

- **Indian isolates** — none specifically selected; ICMR remains our only Indian evidence, and it is population-level.
- **A random sample** — isolates chosen for breadth of bench testing, over-representing strains someone had reason to test.
- **Per-drug labels** — aggregated to class.
- **Prospective clinical outcomes** — none, and not obtainable this way.
- **A GPU** — for DNABERT-2.

## Remaining limitations

- The classifier does not transfer to real genomes. This is architectural, not a tuning problem.
- Empirical floors rest on 3–16 observations per class.
- Novelty misses ~half of true insertions at a reviewable false-alarm rate.
- Forecast intervals cover 83%, not the nominal 95%.
- `cross_check.mae_pp` (40.7) is **not an error metric** — it compares strain-specific predictions against national population prevalence, and a susceptible control sitting far below the national average inflates it while being exactly correct. Per instruction, ICMR percentages are not isolate labels. The aligned figure (6.7 pp) and the category counts are the meaningful summary.

## Claims now justified

- Gene detection finds genuine determinants in real assemblies at a 13.8% false-resistance rate.
- Point mutations are resolved to the codon, agreeing with bench phenotype on 10 of 12 readable real isolates.
- The shipped threshold meets the ~1.5% very-major target and is derived from a stated cost model.
- The mechanism floor is measured against laboratory outcomes rather than asserted.
- Novelty thresholds are calibrated on real genomes and reported on a held-out half.
- Forecast error is 0.48 pp where data is adequate.

## Claims that must not be made

- ❌ **"Clinically validated."** Never.
- ❌ **"Validated on real isolates"** without the figure — it is 0.526, approximately chance.
- ❌ **"Predicts antibiotic resistance."** On real genomes it does not. It detects specific known genes.
- ❌ **"144/144"** or any constructed figure as real-world accuracy.
- ❌ **Any accuracy claim for Indian isolates.** None were tested.
- ❌ **"0% false positives" for novelty.** Refuted: it was 100% of real genomes.
- ❌ **"Forecast accuracy 4.93 pp."** It is 0.48 pp on adequate series and 5.13 pp on thin ones, and MAE is not an accuracy percentage.

## What to say instead

> Module 3 detects 21 known resistance genes in real bacterial genomes with high precision and resolves two point-mutation determinants to the specific codon. We validated it against 350 laboratory susceptibility results from 42 real isolates. Gene detection transferred; the whole-genome classifier did not — it scores near chance on real genomes, because it was trained on short constructed sequences where the resistance gene dominates the signal. Every decision threshold in the system is now derived from measured outcomes with the cost trade-off stated, rather than chosen. We are reporting the real-data number rather than the constructed one, because the gap between them is the honest measure of what the system can currently do.

That is a stronger position than an unexamined 0.99, because it survives the question "how do you know?"
