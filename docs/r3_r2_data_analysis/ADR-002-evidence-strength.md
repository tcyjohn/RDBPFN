# ADR-002: Evidence Strength and Retraining Scope

## Status

Superseded in part by `docs/adr/0011-filter-policy-is-a-version-boundary.md`: the no-new-training constraint remains accepted, but filtered R3 is no longer part of the main-paper analysis.

## Context

Existing artifacts allow detailed descriptive comparison, but they cannot establish that filtering or task count did not cause the observed performance difference:

- R2 unfiltered: 1,474 tasks; R2 trained: 1,010 filtered tasks.
- R3 trained: 1,600 unfiltered tasks.
- Evaluation-seed confidence intervals quantify inference variation for fixed checkpoints, not variation across training runs or generated corpora.

An existing filtered R3 run supplies a partial robustness control:

- H5: `/tmp/RDBPFN-ablation-r3/model_pretrain/pretrain_datasets/ablation_r3_data_prior_1024.h5`.
- Checkpoint: `/tmp/RDBPFN-ablation-r3/model_pretrain/checkpoints/ablation_r3_data_prior_1024_full128/model_eval00007.pt` (`step=112000`).
- Evaluation: `model_pretrain/results/ablation_r3_data_prior_1024_00112_full512.csv` and its `_per_seed.csv` companion.
- Corpus size: 1,169 tasks, retaining 73.1% of the 1,600-task direct merge.

## Adopted evidence level

### Read-only evidence

Analyze pipeline attrition, task removal by the R2 filter, corpus distributions, redundancy, learnability/leakage proxies, and benchmark alignment. This can identify confounds and plausible explanations but cannot close them experimentally.

### Existing within-R3 filter sensitivity

The existing R3 filtered checkpoint gives:

- Filtered R3 mean AUROC: 0.708660.
- Direct R3 mean AUROC: 0.707554.
- Paired difference: +0.001106, 95% CI [-0.001617, 0.003830].
- Task-level outcome: filtered R3 is higher on 9 tasks and lower on 10.

This supports a sensitivity statement that the R3 operating point is not strongly dependent on the repetitive-column filter. It does not estimate the R2 filter's performance effect because no unfiltered R2 checkpoint is available.

## Evidence boundary

- No additional training will be conducted.
- Corpus-level analysis can quantify what R2's filter removes, but cannot measure the counterfactual AUROC of unfiltered R2.
- The filtered-vs-direct R3 comparison is based on fixed checkpoints and evaluation seeds; it does not include training-seed uncertainty.
- Conclusions remain package-level and robustness-oriented, not causal component attribution.

## Next decision

Resolved: Direct R3 remains the primary reported operating point; filtered R3 is optional appendix-only sensitivity evidence.

## Reporting decision

- **Primary R3**: Direct R3, 1,600 tasks, mean AUROC 0.7076.
- **Sensitivity R3**: Filtered R3, 1,169 tasks, mean AUROC 0.7087, appendix only if retained.
- The filtered result will not replace the primary result merely because its observed mean is slightly higher.
- The sensitivity claim emphasizes the small paired difference and non-uniform 9/10 task split, not superiority of filtering.
