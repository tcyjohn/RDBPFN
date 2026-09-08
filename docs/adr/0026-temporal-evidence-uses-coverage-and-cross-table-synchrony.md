# ADR-0026: Use temporal coverage and cross-table synchrony as the main temporal evidence

## Status

Accepted

## Decision

RQ3 will characterize calendar-aware temporal structure with two complementary
raw-database measurements: the prevalence of timestamped tables and the
within-database synchrony of timestamped-table activity over calendar time. The
first measures how broadly temporal structure is instantiated; the second tests
for a measurable signature of shared calendar and database-level temporal
causes.

Parent--child timestamp-order satisfaction will be reported only as a
supplementary diagnostic. It is not suitable as the main R2--R3 comparison
because the R2 corpus contains very few relationships for which both the parent
and child tables are timestamped.

The manuscript will describe these measurements as evidence that R3 expands
calendar-aware temporal coverage and induces measurable cross-table temporal
synchrony. It will not claim that they validate every part of the temporal
generator or establish causal mediation through DFS.

## Final evidence

Across the full 946 paired completed source databases, the timestamped-table
fraction has median 0 for RDB-PFN-Small and 1 for SA-RDB-PFN. Cross-table
monthly-activity synchrony is defined for 52 paired databases; its median is
-0.030632 for RDB-PFN-Small and 0.226872 for SA-RDB-PFN, with a paired median
difference of 0.275107 and a 95% database-bootstrap interval of
[0.223446, 0.360123]. Parent--child ordering remains supplementary because the
original-prior eligible cohort is too small for a balanced headline comparison.
