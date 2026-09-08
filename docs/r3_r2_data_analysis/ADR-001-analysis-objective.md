# ADR-001: R3 vs. R2 Data Analysis Objective

## Status

Accepted — dual-objective analysis, with ordering still constrained by experimental resources.

## Context

The paper reports a package-level performance difference between R3 and R2, but their data pipelines differ at several stages:

- R3: 1,024 requested databases → 946 complete RDBs → 946 processed RDBs → 1,600 direct-merge tasks, without the original repetitive-column filter.
- R2: 1,024 requested databases → 1,014 complete RDBs → 1,014 processed RDBs → 1,474 direct-merge tasks → 1,010 training tasks after the original repetitive-column filter.
- The final training corpora therefore differ in prior, generation success rate, task yield, filtering policy, and stored-task count.

## Decision

Use a two-layer analysis:

1. **Fairness and robustness gate**: quantify pipeline attrition, task yield, filtering effects, and stored-task-count differences; where resources permit, use matched filtering and matched task-count training controls.
2. **Structural richness and benchmark alignment**: compare schema, relational, temporal, feature, label, redundancy, and task-level distributions, then measure how each synthetic corpus aligns with the real evaluation tasks.

The second layer may help explain the result only after the first layer makes the comparison interpretable. Descriptive alignment alone is not treated as evidence that a property caused the AUROC difference.

## Consequences

- A mechanistic explanation requires controlled comparisons, especially applying a common filter or constructing matched subsets.
- A realism/alignment claim requires comparison to real benchmark task properties, not only R3 vs. R2.
- A fairness claim prioritizes pipeline attrition, matched task counts, common filters, and sensitivity analyses.
- A descriptive profile is lower risk but contributes less to the paper's central argument.

## Open question

Can the study afford additional 112K-step training runs, or must the analysis remain read-only over existing artifacts?
