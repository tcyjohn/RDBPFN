# ADR-008: Nonparametric Feature and Target Analysis

- Status: Accepted
- Date: 2026-07-18

## Decision

Do not train auxiliary probe models for the R3--R2 corpus analysis. Use nonparametric realized-data statistics:

- normalized effective rank for feature redundancy;
- near-duplicate and dominant-column rates, using thresholds aligned with the preprocessing/filter audit where possible;
- binned normalized mutual information between features and the target to characterize how target information is distributed across features.

Continuous variables are quantile-binned using a fixed rule shared by R2, R3, and benchmark tasks. Identifier-derived columns and the target itself are excluded from feature-feature summaries. Metrics are computed at task level, summarized within each RDB where an RDB contributes multiple tasks, and then compared using paired RDB statistics when pairing is available.

## Rationale

Auxiliary probes would introduce model class, optimization, and hyperparameter choices and would blur the distinction between corpus characterization and an additional predictive experiment. Nonparametric statistics directly address redundancy and signal distribution without violating the no-new-training constraint.

## Evidence Boundary

Mutual information is a dependence descriptor, not a causal attribution or a direct measure of downstream task difficulty. Binning sensitivity must be checked in the appendix.
