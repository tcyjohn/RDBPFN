# ADR-013: Filter-Aligned Corpus Analysis

- Status: Superseded by ADR-014
- Date: 2026-07-18

## Context

The reported R2 checkpoint was trained on the 1,010-task filtered corpus, whereas the reported standalone R3 checkpoint was trained on the 1,600-task direct-merge corpus. Comparing their feature-correlation statistics directly would mix generator differences with repetitive-column filtering.

An existing R3 checkpoint and evaluation result are available for the 1,169-task filtered R3 corpus, and its mean AUROC is close to the direct R3 result.

## Decision

Separate the performance and corpus-characterization comparisons.

### Performance comparison

- Primary: R2 filtered versus R3 direct, because these are the reported standalone operating points.
- Sensitivity: R3 direct versus R3 filtered, using the existing checkpoint only.

### Corpus-structure comparison

- Primary: R2 filtered versus R3 filtered, to align repetitive-column filtering as closely as the existing artifacts permit.
- Sensitivity: add R3 direct to show which correlation/redundancy differences are introduced or removed by filtering.

The main corpus analysis follows the original RDB-PFN paper's focus on DFS-linearized feature-correlation structure, but adds aggregate statistics across tasks rather than relying only on representative heatmaps.

## Main-Paper Evidence

1. A compact audit/performance table: completion rate, stored tasks, filter retention, mean AUROC, and paired evaluation uncertainty.
2. Clustered correlation heatmaps for representative R2-filtered, R3-filtered, and real benchmark tasks, selected by a documented non-performance-based rule.
3. Aggregate normalized effective rank and strong-correlation-pair rate for R2 filtered, R3 filtered, R3 direct, and real benchmark tasks.

## Supplementary Evidence

- simple schema-DAG attrition audit;
- temporal prevalence and conditional temporal diagnostics;
- FK effective parent coverage and multi-parent dependence;
- target-signal dispersion and threshold/binning sensitivity;
- benchmark coverage and robust Wasserstein-distance analyses.

## Interpretation Boundary

The filter-aligned corpus comparison characterizes the realized task distributions. The performance comparison remains package-level and does not identify which measured corpus property caused the AUROC difference.
