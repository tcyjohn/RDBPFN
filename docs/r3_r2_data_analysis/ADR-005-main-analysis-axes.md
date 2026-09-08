# ADR-005: Main Structural Analysis Axes

- Status: Amended by ADR-013
- Date: 2026-07-18

## Context

R2 and R3 use the same schema DAGs. The main paper has only 1--1.5 pages for this analysis, so generic dataset descriptors cannot all be promoted to headline evidence.

## Decision

The paired structural analysis will cover three method-aligned axes:

1. temporal and entity structure;
2. foreign-key concentration and community structure;
3. feature signal-group diversity and target dependence.

The unit of comparison is the 946 RDBs completed by both generators and matched by `dag_rdb_i`.

Following ADR-013, these axes are supporting analyses rather than the main figure. The main figure prioritizes DFS-linearized correlation structure to match the original RDB-PFN paper's analysis style. Generic descriptors such as row count, column count, missingness, task size, and post-processing yield will be reported in a compact audit table or the appendix. Schema-topology descriptors are excluded as discriminative evidence because the schema DAG is controlled.

## Rationale

These axes correspond to mechanisms that can differ between R2 and R3 despite identical schema graphs. They therefore test whether R3 changes the realized database instances, rather than merely restating fixed input properties.

## Evidence Boundary

A difference in these metrics supports a descriptive claim about the generated corpus. It does not by itself establish that the difference caused the downstream AUROC gain.
