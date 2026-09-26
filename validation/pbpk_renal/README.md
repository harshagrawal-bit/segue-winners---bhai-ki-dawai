# Module 2 — PBPK ground-truth validation (renal impairment)

Validates the Module 2 renal prediction (`CL_ratio = fe·KF + (1−fe)·HF`,
`AUC_ratio = 1/CL_ratio`) against whole-body PBPK simulations run in PK-Sim.

---

## 1. Environment — verified on this machine, 2026-09-26

| Component | Version | How it got here |
|---|---|---|
| OSP Suite installer | `OSPSuite-Full.12.3.263.exe` | already in `~/Downloads`, signature **valid** |
| PK-Sim | 12.3.173 | already installed at `C:\Program Files\Open Systems Pharmacology\PK-Sim 12.3` |
| MoBi | 12.3.136 | same installer |
| .NET runtime | 8.0.17 | present (required by `rSharp` 1.2.3) |
| R | 4.6.1 | installed via `winget install RProject.R` |
| `ospsuite` | 12.4.5 | GitHub release binary |
| `rSharp` | **1.2.3 — pinned** | GitHub release binary |
| `ospsuite.plots` | 1.3.0 | OSP r-universe |
| `tlf` | 1.6.2 | OSP r-universe |
| `ospsuite.utils` | 1.11.1 | OSP r-universe |

The downloaded installer was checked before use: Authenticode status **Valid**,
signed by *Computational Life Sciences Foundation gUG* (SSL.com EV Code
Signing), `ProductVersion 12.3.263`, 545.6 MB. It did not need to be re-run —
the suite was already installed from it.

> **Do not upgrade `rSharp`.** `ospsuite` 12.4.x requires `rSharp <= 1.2.3`
> (.NET 8). `rSharp` 2.0.0 targets .NET 10 and makes `loadSimulation()` fail on
> `.pkml` files. The OSP r-universe serves 2.0.0, so it is installed here from a
> pinned release zip instead. If you ever run `update.packages()`, re-pin it.

The full chain was smoke-tested end to end (load → run → `AUC_inf`) against
ospsuite's bundled example simulation before you were asked to do any GUI work.

---

## 2. Models — downloaded and SHA-pinned

In `models/`. Every file is pinned to the exact commit it was fetched from;
`models/SOURCES.tsv` records repo, branch, commit SHA and permalink, and the R
script reads it so the CSV provenance stays truthful if you re-download.

| Drug | File | Commit |
|---|---|---|
| Fluconazole | `Fluconazole-Model.json` | `fa23aa5a` |
| Metformin | `Metformin-Model.json` | `5586d01d` |
| Methotrexate | `Methotrexate-Model.json` (+ `.pksim5`) | `6cd01494` |
| Omeprazole | `Omeprazole-Model.json` | `c8b0f70a` |
| Atorvastatin | `Atorvastatin-Model.json` | `bf3befea` |
| Ketoconazole | `Ketoconazole-Model.json` (+ `.pksim5`) | `bb8d0029` |
| Theophylline | `Theophylline-Model.pksim5` **only** | `0610b095` |

All seven dedicated `<Drug>-Model` repos exist, so none were taken from
`OSP-PBPK-Model-Library`.

> **Theophylline is the risk case.** That repo has no snapshot JSON and was last
> touched in **March 2019**, so the `.pksim5` predates PK-Sim 8. PK-Sim 12.3 must
> convert it on open and may refuse. If it does, the script will write
> `status = missing_pkml` for all five Theophylline rows — leave it that way
> rather than substituting anything.

---

## 3. PK-Sim desktop steps

Three facts read directly out of the installed PK-Sim 12.3 assemblies and
`PKSimDB.sqlite`, before you start:

- The CKD model is **adults only**. PK-Sim enforces it:
  *"Chronic kidney disease model is only available for adult. Make sure the
  input age is greater than or equal to 18 years."* Keep age ≥ 18.
- The control is a **Disease State** combo box with a **CKD** entry, which
  reveals an **eGFR** parameter field.
- **The CKD eGFR field accepts 0–60 mL/min/1.73 m² only.** The database stores
  the ceiling as `0.00034683 l/min/dm²` = exactly **60.0**. This matches the
  clinical definition of CKD (stage 3+ is eGFR < 60). Entering 90 or 75 shows a
  red ✗ and greys out **OK**.

### What that means for the five planned levels

| Planned eGFR | Buildable? | How to build it |
|---|---|---|
| **90** (normal) | not via CKD | **Disease State = `Healthy`** — the matched healthy control |
| **75** | ✗ **no** | above the 60 ceiling, and not a healthy control either — **skip it** |
| **45** | ✓ | Disease State = `CKD`, eGFR `45` |
| **22** | ✓ | Disease State = `CKD`, eGFR `22` |
| **10** | ✓ | Disease State = `CKD`, eGFR `10` |

So it is **4 arms per drug, not 5**. Using a matched *Healthy* individual as the
normal reference is exactly what regulatory renal-impairment studies do, so the
design is sound — but note it is no longer "CKD at 90 vs CKD at 10". The
reference now differs from the impaired arms by disease model as well as by
eGFR. State that plainly when you write this up.

eGFR 75 is left as a documented gap. The only way to force it would be editing
GFR by hand, which would break the "no parameter changes" rule — so the script
records it as `ckd_range_exceeded` instead of inventing a number.

### Step A — open the model

**For the six drugs with a snapshot `.json`:**

1. Launch **PK-Sim 12.3** from the Start menu.
2. Menu **File → Load from Snapshot**.
3. Choose `validation\pbpk_renal\models\<Drug>-Model.json`.
4. Wait for PK-Sim to rebuild the project (large models take a minute).

**For Theophylline (`.pksim5` only):**

1. **File → Open Project**, choose `models\Theophylline-Model.pksim5`.
2. Accept the conversion prompt if PK-Sim offers to upgrade the old project.

### Step B — create four individuals

Do this **four times per drug**. Keep **Species, Population, Gender, Age,
Weight and Height identical every time** — the kidney setting is the only thing
that may differ between arms.

1. In the building-block tree on the left, right-click **Individuals** →
   **Create Individual**.
2. Set the common fields (leave the model's defaults; **age must be ≥ 18**):
   - **Species** — `Human`
   - **Population** — as the published model ships it (usually
     *European (ICRP 2002)*)
   - **Gender / Age / Weight / Height** — same values for all four arms
3. Then set the **Disease State** group:

   | Arm | Disease State | eGFR field | Name it |
   |---|---|---|---|
   | reference | `Healthy` | *(no field appears)* | `<Drug>_eGFR90` |
   | mild-mod | `Chronic Kidney Disease - No Dialysis (Malik et al, 2020)` | `45` | `<Drug>_eGFR45` |
   | severe | same CKD entry | `22` | `<Drug>_eGFR22` |
   | failure | same CKD entry | `10` | `<Drug>_eGFR10` |

4. **OK** to add it to the tree.

The reference file keeps the name `..._eGFR90` so the script and the CSV stay
consistent, even though it is built as *Healthy*. The script reads each
individual's **real** eGFR back out of the simulation and records it in the
`egfr_actual` column, so nothing rests on the filename. A healthy 30-year-old
comes out around **105 mL/min/1.73 m²** — expect that, not exactly 90.

### Step C — build and run one simulation per individual

1. Right-click **Simulations** → **Create Simulation**.
2. Select the individual `<Drug>_eGFR<value>`.
3. For **Compound**, **Formulation** and **Administration Protocol**, accept the
   ones the published model already defines — **identical for all five levels**.
   Same dose, same route, same schedule. The individual is the only thing that
   changes.
4. Finish, then **Run** (▶ / F5).

**You do not need to adjust the simulation end time.** Renal impairment
lengthens half-life several-fold, so a window suited to eGFR 90 is often far
too short at eGFR 10 — but the script handles this itself. It runs each
simulation exactly as exported, and only if more than 20% of `AUC_inf` came
from the extrapolated tail does it re-run once over a longer observation
window, keeping the better-supported result. The window used and the
before/after tail fraction are written to every row (`sim_end_hours`, `note`),
so the decision is auditable rather than silent.

This is safe because it changes only *when* the solver is observed, never what
is solved: `setOutputInterval()` rewrites the output schema, not a model
parameter, and dosing is still governed by the administration protocol.
Measured on ospsuite's Aciclovir example, extending 24 h → 240 h moved
`AUC_inf` by 0.09% while the extrapolated tail fell from 0.2% to 0.0%.

### Step D — export to `.pkml`

1. Right-click the simulation in the tree and choose the export entry ending in
   **…to pkml** (also reachable from the ribbon's Export group).
2. Save into **`validation\pbpk_renal\pkml\`**.
3. Name the file **exactly** `<Drug>_eGFR<value>.pkml` — the script matches on
   this (case-insensitively) and reports anything it can't find.

All 28 expected filenames (4 per drug — there is no `_eGFR75`):

```text
Fluconazole_eGFR90.pkml    Fluconazole_eGFR45.pkml    Fluconazole_eGFR22.pkml    Fluconazole_eGFR10.pkml
Metformin_eGFR90.pkml      Metformin_eGFR45.pkml      Metformin_eGFR22.pkml      Metformin_eGFR10.pkml
Methotrexate_eGFR90.pkml   Methotrexate_eGFR45.pkml   Methotrexate_eGFR22.pkml   Methotrexate_eGFR10.pkml
Omeprazole_eGFR90.pkml     Omeprazole_eGFR45.pkml     Omeprazole_eGFR22.pkml     Omeprazole_eGFR10.pkml
Atorvastatin_eGFR90.pkml   Atorvastatin_eGFR45.pkml   Atorvastatin_eGFR22.pkml   Atorvastatin_eGFR10.pkml
Ketoconazole_eGFR90.pkml   Ketoconazole_eGFR45.pkml   Ketoconazole_eGFR22.pkml   Ketoconazole_eGFR10.pkml
Theophylline_eGFR90.pkml   Theophylline_eGFR45.pkml   Theophylline_eGFR22.pkml   Theophylline_eGFR10.pkml
```

The CSV still carries a row for every eGFR 75 combination, marked
`ckd_range_exceeded` — the gap is recorded, not hidden.

You can export in batches — the script runs fine on a partial set and marks the
rest `missing_pkml`.

---

## 4. Run the analysis

```powershell
cd "D:\VIT PUNE\Trialsense\segue-winners---bhai-ki-dawai\validation\pbpk_renal"
& "C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe" run_pbpk_renal.R
```

Optional overrides: `Rscript run_pbpk_renal.R <pkml_dir> <out_csv>`.

Writes **`pksim_renal_predictions.csv`**.

### What the script does

For each `.pkml`: loads it, selects the **parent drug's** peripheral-venous
plasma observer, runs the simulation, and takes `AUC_inf` from
`calculatePKAnalyses()`. Ratios are `AUC(eGFR) / AUC(eGFR 90)` within each drug.

Two safeguards worth knowing:

- **Parent-only.** The observer's molecule segment must equal the drug name, so
  a metabolite curve can never be integrated and reported as parent exposure
  (this matters for omeprazole and atorvastatin, which both have metabolites in
  their published models). If the parent can't be identified the row fails
  loudly instead of silently using the wrong curve.
- **No parameter is ever written.** The only mutation is the output selection
  (`clearOutputs` / `setOutputs`). There is no `setParameterValuesByPath` call
  in the file — renal impairment enters solely through the CKD individual you
  built in the GUI.

### Columns

`drug`, `egfr`, `auc`, `auc_ratio`, `model_source_url`, `pksim_version` —
as specified, in that order — followed by diagnostics:

| Column | Meaning |
|---|---|
| `status` | `ok`, `missing_pkml`, `load_failed`, `no_parent_output`, `output_selection_failed`, `run_failed`, `pk_analysis_failed`, `auc_inf_absent`, `auc_inf_na` |
| `note` | the actual reason — the exception message for failures |
| `auc_unit` | unit of `auc` (e.g. `µmol*min/l`) |
| `frac_auc_extrapolated` | fraction of `AUC_inf` from the extrapolated tail; >0.20 is flagged in `note` |
| `output_path` | the exact observer integrated |
| `pkml_file`, `ospsuite_version`, `run_timestamp` | provenance |

`pksim_version` is read out of each `.pkml` itself (the `pKSimVersion`
attribute), not assumed — so the CSV records the version that actually produced
each simulation.

**Nothing is imputed.** Every one of the 35 drug × eGFR combinations always gets
a row. A failure yields `auc = NA`, `auc_ratio = NA` and a `note` saying why. If
a drug's eGFR 90 reference is unavailable, that drug's ratios stay `NA` and every
row says so rather than falling back to another baseline.

### Current state

The script has been run against the empty `pkml/` folder. The CSV therefore
holds 35 rows all marked `missing_pkml` — that is the correct starting state,
and it confirms the harness, provenance lookup and CSV schema work. Re-run it
after each batch of exports.
