# ADR-010: Preregistered Main Structural Metrics

- Status: Superseded by ADR-013
- Date: 2026-07-18

## Decision

The following five metrics remain preregistered supporting analyses, but ADR-013 moves them out of the main structural panel:

1. **Realized temporal prevalence:** whether each paired RDB contains a realized timestamp/temporal table after generation. Compare paired prevalence across all eligible RDBs.
2. **Normalized effective parent coverage:** concentration of realized FK assignments over available parent entities.
3. **Multi-parent FK dependence:** normalized mutual information between FK assignments in multi-parent child tables.
4. **Normalized feature effective rank:** effective dimensionality relative to the number of eligible features.
5. **Target-signal dispersion:** the effective number of target-associated features, using feature--target normalized-mutual-information weights and normalizing by eligible feature count.

If reported in the supplement, these metrics are included regardless of whether they favor R3.

## Temporal Supporting Analysis

Among timestamp-enabled tables, report temporal span, repeated observations per entity, and interval irregularity only as conditional appendix analyses. Do not treat non-temporal RDBs as having zero conditional temporal strength.

If a temporal mechanism is configured but does not survive into the realized table or H5, the main metric records it as absent. Configuration activation is reported separately as mechanism evidence.

## Rationale

Temporal prevalence is defined for both generators even when one rarely produces temporal data. Separating prevalence from conditional strength avoids conflating mechanism availability with the magnitude of a property conditional on that mechanism existing.
