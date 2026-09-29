# Real-data ingestion — what was obtained and what is missing

Generated 2026-09-29 by `python -m trialsense.ingest`.

## Obtained

- **42 real genome assemblies** (191,455,873 bp total) from BV-BRC.
- **350 laboratory susceptibility labels** across 12 antibiotic classes.

  Labels are bench measurements only. Rows whose `evidence` column read
  `Computational Method` — another group's machine-learning predictions —
  were filtered out at the server, so no prediction is ever scored against
  another prediction.

  Isolates per organism:

  - Acinetobacter baumannii: 6
  - Enterococcus faecium: 6
  - Escherichia coli: 6
  - Klebsiella pneumoniae: 6
  - Pseudomonas aeruginosa: 6
  - Salmonella enterica: 6
  - Staphylococcus aureus: 6

## Known limits of this set

- **Not a random sample.** Isolates were chosen for breadth of bench
  testing, so they skew towards strains somebody had reason to test —
  typically clinically interesting, often resistant ones. Resistance
  rates here describe this set, not any patient population.
- **Class-level, not drug-level.** A class counts as resistant if any
  drug in it failed at the bench. Within-class detail is lost.
- **Intermediate results dropped**, not coerced: 10,494 rows.
- **No Indian isolates were specifically selected.** BV-BRC geography is
  uneven, so this set does not speak to Indian resistance patterns; the
  ICMR surveillance layer remains the only Indian-specific evidence.

## What this permits us to claim

| Claim | Permitted? |
|---|---|
| Validated on constructed isolates | yes, with 'constructed' stated |
| Gene detection works on real genomes | yes, once the real-genome check runs |
| Measured against laboratory AST | yes, on this set, with its skew stated |
| Clinically validated | **no** |
| Accurate on Indian patient isolates | **no — never measured** |

The third row is the new one. It is not the same as the fourth: a bench
result on a curated public strain set is evidence, but it is not a
prospective clinical trial and must never be described as one.
