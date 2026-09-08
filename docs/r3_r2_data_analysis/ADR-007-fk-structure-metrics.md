# ADR-007: Foreign-Key Structure Metrics

- Status: Accepted
- Date: 2026-07-18

## Decision

Use two realized-data metrics as the main evidence for foreign-key structure.

### Normalized effective parent coverage

For each child-to-parent relationship, compute

\[
\frac{\exp(H(P_{FK}))}{N_{parent}},
\]

where `P_FK` is the empirical distribution of child rows over referenced parent rows and `N_parent` is the number of available parent rows. Lower values indicate stronger concentration on a subset of parent entities.

### Multi-parent FK dependence

For child tables with multiple foreign keys, compute normalized mutual information between pairs of FK assignments. This measures whether parent choices exhibit joint structure rather than behaving independently.

Aggregate first within each RDB, giving equal weight to each of the 946 paired RDBs, and then report paired R3-minus-R2 differences with uncertainty. Relationship-level sample counts and missingness are reported separately.

## Supporting Evidence

Generator parameters such as HSBM depth and cluster counts may be reported in the appendix but are not substitutes for realized-data measurements.

## Rationale

Generic community detection is unstable when each child row contributes only one edge to a given parent table. Parent-coverage entropy and multi-parent dependence are directly defined by the observed FK assignments and remain comparable across generator versions.
