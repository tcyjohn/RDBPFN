# ADR-003: Reporting Budget for Data Analysis

## Status

Accepted — allocate 1–1.5 main-paper pages, with detailed diagnostics moved to supplementary material.

## Context

The no-new-training evidence plan can support several analysis layers:

1. **Pipeline funnel**: requested → complete → processed → direct tasks → filtered tasks.
2. **Filter sensitivity**: Direct R3 vs. filtered R3 performance and task removal.
3. **Final-H5 task profile**: rows, usable features, train/query split, categorical fraction, label balance, repetitive columns, and task redundancy.
4. **Raw relational structure**: table counts, schema edges/depth, row counts, FK topology, entity repetition, and temporal coverage.
5. **Benchmark alignment**: compare synthetic distributions with properties of the 19 real evaluation tasks.

All five layers are useful, but presenting every metric in the main paper would dilute the central argument and consume substantial space.

## Recommended packaging

### Main paper

- One compact pipeline/filter table.
- One distribution/alignment figure with 3–5 preselected metrics.
- One short paragraph interpreting the R3 filter sensitivity.

### Appendix or supplementary material

- Full metric definitions and extraction procedure.
- Complete distribution tables and effect sizes.
- Per-task filter changes and benchmark-alignment diagnostics.
- Data-quality caveats and failure cases.

## Decision to make

Resolved: 1–1.5 pages are available in the main paper.

## Main-paper allocation

- **0.25–0.35 page — compact evidence table**: pipeline completion, direct/filtered task counts, retention rates, and the existing R3 filter-sensitivity score.
- **0.45–0.60 page — two-panel figure**:
  - Panel A: R2 vs. R3 relational-structure distributions from raw/processed RDBs.
  - Panel B: task-level synthetic-to-RelBench coverage/alignment using properties available in the 19 evaluation tasks.
- **0.25–0.40 page — interpretation**: distinguish robustness evidence, descriptive structural differences, and non-causal alignment evidence.

## Space discipline

- Main text will carry at most 3–5 headline metrics per panel.
- Full distributions, metric definitions, effect sizes, and per-task details move to supplementary material.
- No correlation is called explanatory or causal merely because it matches the R3–R2 performance pattern.
