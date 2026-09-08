# ADR-014: Training-Corpus-First Reporting

- Status: Superseded in part by docs/adr/0011
- Date: 2026-07-18

## Context

The reported R2 model was trained on the 1,010-task filtered corpus, and the reported standalone R3 model was trained on the 1,600-task direct-merge corpus. These are the two operating points the paper seeks to explain. Elevating the 1,169-task filtered R3 corpus to an equal role would spend scarce space on a sensitivity run rather than the actual primary comparison.

## Decision

Use R2 filtered and R3 direct throughout the main performance and corpus analyses.

- Main performance comparison: R2 filtered versus R3 direct.
- Main data analysis: the corresponding R2-filtered and R3-direct H5 task distributions, with real benchmark tasks as context.
- R3 filtered: removed from the main table/text; optional appendix-only sensitivity evidence.

The sensitivity statement reports that filtering removes 26.9% of R3 tasks (1,600 to 1,169) while changing mean AUROC by approximately +0.0011, with a paired confidence interval spanning zero. It therefore supports robustness to filtering but is not presented as a third primary operating point.

## Interpretation Boundary

Because R2 and R3 use different final filtering policies, their corpus differences characterize the complete training packages and cannot be attributed solely to the generator. Correlation/redundancy results must be described as differences in realized training exposure. The filtered-R3 sensitivity analysis bounds concern about R3's own filter dependence but does not supply an unfiltered-R2 counterfactual.

## Consequence

ADR-013's filter-aligned corpus comparison is superseded. Filter-aligned R2-versus-R3 analysis may remain an appendix diagnostic but is not required for the 1--1.5-page main analysis.
