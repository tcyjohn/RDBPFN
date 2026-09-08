# ADR-011: Statistical Reporting for Paired Structural Metrics

- Status: Accepted
- Date: 2026-07-18

## Decision

For each preregistered structural metric, report:

- the R2 and R3 median and interquartile range;
- the within-`dag_rdb_i` paired difference;
- the median paired difference;
- a 95% paired bootstrap confidence interval obtained by resampling RDB identifiers;
- the proportions of pairs in which R3 is higher, tied, or lower than R2.

Relationship- or table-level observations are summarized within each RDB before resampling, so schemas with more tables do not receive greater inferential weight.

Do not foreground null-hypothesis significance tests in the main paper. If requested for completeness, place paired tests with Holm correction across the five preregistered metrics in the appendix.

For binary temporal prevalence, report paired proportions and their paired difference with an RDB-level bootstrap interval; a McNemar test may appear only in the appendix.

## Rationale

With 946 paired RDBs, very small effects can produce small p-values. Effect magnitude, direction consistency, and uncertainty are more informative for corpus characterization.
