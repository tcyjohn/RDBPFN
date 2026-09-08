# R2--R3--v6.2 Data Analysis Blueprint

## 1. Purpose and manuscript role

The analysis explains what the successive design changes alter in the realized data and model interface without claiming that any one descriptive property caused an AUROC change.

The evidence chain is:

1. **R2 to R3:** test whether the new data prior produces observable temporal, FK-dependence, and feature-structure differences after generation and DFS preprocessing.
2. **R3 to v6.2:** test whether the aligned task format supplies valid entity identity and when same-entity attention is eligible at evaluation time.
3. **Performance:** show that R3 improves broadly over R2 while v6.2 has aggregate parity with R3 and redistributes task-wise performance.
4. **Mix:** retain the legacy Mix only as a secondary high-scoring checkpoint, not as a third generator or a target for distribution analysis.

The paired-delta performance forest belongs in the ablation-results subsection and is outside the 1--1.5-page data-analysis budget.

## 2. Operating points and artifacts

| Operating point | Meaning | Primary artifact |
|---|---|---|
| R2 | Original generator and original repetitive-column filtering policy | `/tmp/RDBPFN-ablation-r2/model_pretrain/pretrain_datasets/ablation_r2_original_style_1024.h5` (1,010 tasks) |
| R3 | Data-prior-only variant with the intended direct-merge policy | `/tmp/RDBPFN-ablation-r3/model_pretrain/pretrain_datasets/ablation_r3_data_prior_1024_unsampled.h5` (1,600 tasks) |
| v6.2 | Entity-aware package: R3 data prior, aligned task format, valid entity IDs, and same-entity attention bias | `model_pretrain/pretrain_datasets/v6.2.h5` (1,219 tasks) |
| Legacy Mix | v6.2 mixed with an earlier 1,169-task R3 materialization | aggregate result 0.7118; secondary main-table checkpoint |
| Corrected Mix | v6.2 mixed with the reported 1,600-task direct-R3 source | `model_pretrain/results/mix_v62_r3direct_entitybias_00112_full512_aligned.csv` (0.7079); disclosure and appendix |

Raw paired sources:

- R2: `/tmp/RDBPFN-ablation-r2/data_generation/RDB_datasets/ablation_r2_original_style_1024_raw`
- R3: `/tmp/RDBPFN-ablation-r3/data_generation/RDB_datasets/ablation_r3_data_prior_1024_raw`
- Primary paired cohort: the 946 complete `dag_rdb_i` identifiers shared by R2 and R3.
- Attrition cohort: all 1,024 requested identifiers; 1,014 R2 and 946 R3 instances completed.

The schema DAG is identical within each R2--R3 pair. Table count, edge count, and depth are controls or attrition descriptors, not evidence that R3 has richer relational structure.

## 3. Research questions

### RQ1: Does the R3 data prior survive realization?

On the 946 paired raw RDBs, does R3 change the realized prevalence of temporal data, FK concentration, and dependence between multiple parent choices relative to R2?

### RQ2: Does a distinct signature remain after DFS preprocessing?

Do the actual R2-filtered and R3-direct training corpora differ in feature-correlation structure, and how do their correlation signatures compare descriptively with real DFS tasks?

This is a comparison of realized training exposure. Because R2 and R3 intentionally use different filter policies, it is not a generator-only controlled contrast.

### RQ3: When can the v6.2 entity mechanism operate?

Do v6.2 training tasks contain valid entity IDs and repeated entities, and which evaluation tasks have query entities represented in the support set?

### RQ4: How do these changes relate to the observed performance pattern?

Does the task-wise performance profile support a broad R2-to-R3 change and an R3-to-v6.2 specialization pattern? The answer remains descriptive: no existing run isolates task format from entity bias.

## 4. Main-paper analysis: 1--1.5 pages

### 4.1 Compact pipeline statement

Use one short paragraph or two table rows:

- R2: 1,014/1,024 completed RDBs; 1,474 direct-merge tasks before the original filter; 1,010 training tasks after filtering.
- R3: 946/1,024 completed RDBs; 1,600 direct-merge training tasks; no original repetitive-column filter by design.
- Report the 6.6-percentage-point completion gap and only a coarse schema-DAG attrition summary. Do not infer failure causes.
- State the preprocessing boundary explicitly: the filter belongs to the original R2 pipeline, while R2.5 and later variants intentionally use direct merge.

### 4.2 Figure 1: R2-to-R3 mechanism survival

Use a compact paired-effect figure with four preregistered headline metrics:

1. **Realized temporal prevalence:** fraction of paired RDBs with a realized timestamp or temporal table.
2. **Normalized effective parent coverage:** for each child-to-parent FK,
   \[
   \exp(H(P_{FK}))/N_{parent}.
   \]
   Lower values indicate more concentrated use of parent entities.
3. **Multi-parent FK NMI:** normalized mutual information between FK assignments for pairs of parents in eligible multi-parent child tables.
4. **Normalized feature effective rank:** effective rank of the eligible feature-correlation matrix divided by eligible feature count.

For table- or relationship-level metrics, aggregate within each RDB first so every schema receives equal weight. For each metric report R2 and R3 median/IQR, median paired R3-minus-R2 difference, 95% paired RDB-bootstrap interval, and the proportions of pairs higher/tied/lower. For binary temporal prevalence, report paired proportions and the bootstrapped paired difference.

Do not foreground p-values: with 946 pairs, magnitude and consistency are more informative. Optional paired tests with Holm correction belong in the appendix.

### 4.3 Figure 2: DFS correlation context

Follow the visual language of the original RDB-PFN analysis, which compared representative feature-correlation heatmaps, but strengthen it with a corpus-level distribution:

- one clustered heatmap each for R2 filtered, R3 direct, and a real DFS task;
- choose each synthetic exemplar deterministically as the eligible task closest to its corpus median normalized effective rank;
- show the full normalized-effective-rank distributions beside or below the heatmaps;
- apply identical feature eligibility, missing-value handling, and correlation definitions to synthetic and real tasks.

The H5 files do not retain source RDB/task identifiers, so this panel is descriptive corpus-level evidence. Do not use task rows as independent inferential replications and do not claim comprehensive benchmark alignment. Real tasks provide only correlation-structure context.

### 4.4 v6.2 entity exposure

Use a compact paragraph plus a small strip/dot summary rather than another large figure.

Training-side facts to report:

- all 1,219 v6.2 H5 tasks store entity IDs and all actual rows have valid IDs;
- mean per-task query--support entity overlap is 95.17%, median 95.04%, and row-weighted overlap 95.06%;
- median same-entity support count is 5 and the conditional 90th percentile is 9.

Evaluation-side facts under the full-512 protocol:

- 15 of 19 tasks yield usable primary-key entity IDs;
- across those 15 tasks, mean query--support overlap is 6.76% but median is only 0.48%;
- exposure is concentrated in rel-event user-repeat (30.65%), rel-f1 driver-dnf (26.17%), and rel-f1 driver-top3 (34.04%); most other tasks are near zero.

Mechanism-use diagnostic:

- at the v6.2 step-112 checkpoint, the six positive same-entity bias strengths are approximately 0.011, 0.096, 0.249, 0.026, 0.091, and 1.565, compared with their 0.1 initialization.

Interpretation: v6.2 learned a nontrivial entity-bias parameter, but its inference-time opportunity is strongly task dependent. This is consistent with looking for task specialization rather than uniform improvement. It does not establish that entity exposure caused any task delta.

## 5. Performance evidence outside the data-analysis page budget

### 5.1 Main paired-delta forest plot

Replace the dense 19-task-by-variant main table with two aligned-seed task-delta panels:

- R3 minus R2;
- v6.2 minus R3.

For each task show the mean paired delta, paired-seed 95% interval, and Holm-corrected significance marker. Put the full raw per-task/per-seed values in the appendix.

The v6.2--R3 summary is:

- suite means: 0.707205 versus 0.707554;
- mean delta: -0.000349, 95% CI [-0.004683, 0.003985], exact paired sign-flip p = 0.8535;
- task wins: v6.2/R3 = 9/10;
- Holm-significant specialization: v6.2 is higher on rel-f1 driver-top3 (+0.02357), while R3 is higher on stackexchange upvote (+0.01976) and rel-event user-ignore (+0.02036).

Use “aggregate parity” as a descriptive phrase, not a formal equivalence claim.

### 5.2 Aggregate operating-point table and Mix disclosure

- Keep `Mix (earlier R3 source)` at 0.7118 as a secondary, visually bright checkpoint.
- Footnote that the source-aligned direct-R3 Mix obtained 0.7079.
- Put corrected Mix full results in the appendix.
- Do not analyze Mix as a separate data distribution; it samples two already characterized sources.
- Do not claim that mixing itself improves performance: corrected Mix differs from R3 by +0.000340 with an interval spanning zero and differs from v6.2 by +0.000688 with an interval spanning zero.

## 6. Appendix analyses

### Required

- Exact metric definitions, eligibility counts, and missingness for all four R2--R3 headline metrics.
- Conditional temporal span, repeated-observation, and interval-irregularity summaries among eligible tables only.
- Strong-correlation threshold and effective-rank sensitivity analyses.
- Full 19-task/per-seed performance table and exact sign-flip tests with Holm correction.
- Full evaluation entity-exposure table, including the four tasks without usable IDs.
- Corrected Mix task-level results and comparison with R3/v6.2.

### Optional, only if space permits

- Exposure-versus-v6.2-minus-R3 task-delta scatter, explicitly labeled exploratory and noncausal.
- Generator configuration activation rates as supporting mechanism evidence.
- Retrospective filtered-R3 sensitivity: 1,169 versus 1,600 tasks; mean AUROC +0.001106 relative to direct R3 with 95% CI [-0.001617, 0.003830]. This is not a third main operating point.
- Near-duplicate/dominant-column rates and alternative discretization thresholds.

Do not add broad benchmark-coverage or Wasserstein analyses. They consume space without directly explaining the staged interventions.

## 7. Reproducibility and statistical rules

- Freeze the 946 paired RDB identifier list before computing metric directions.
- Use the RDB identifier as the bootstrap unit for raw structural comparisons.
- Record the numerator, denominator, and missingness for every conditional metric.
- Use a fixed random seed and at least 10,000 paired bootstrap draws for reported structural intervals.
- Treat H5 analyses as descriptive corpus comparisons because source provenance is absent.
- For performance, pair on the same task and evaluation seed; use exact sign-flip tests and Holm correction across tasks.
- Retain every preregistered headline metric in the report even if its direction does not favor R3.
- Preserve machine-readable task-level and RDB-level output tables alongside plotting inputs.

## 8. Claim ladder

Supported claims, from strongest to weakest:

1. **Observed pipeline fact:** R2 and R3 have different completion, task-yield, and intended filtering policies.
2. **Mechanism-survival claim:** paired realized RDB metrics quantify whether R3 changes temporal prevalence and FK/feature structure.
3. **DFS-signature claim:** the reported training corpora differ descriptively in correlation/effective-rank structure, with real DFS tasks as context.
4. **Entity-eligibility claim:** v6.2 provides entity identity during training, while inference-time overlap is heterogeneous across tasks.
5. **Performance-pattern claim:** R3 shows the data-prior operating point; v6.2 has similar aggregate performance and meaningful task specialization.

Not supported:

- that any one measured data statistic caused the R3 performance change;
- that same-entity bias caused the v6.2 task deltas;
- that legacy Mix establishes a source-aligned mixing benefit;
- that representative heatmap similarity proves synthetic-data realism.

## 9. Execution order and stop conditions

1. Freeze cohorts and produce the pipeline/attrition audit.
2. Compute the four paired R2--R3 headline metrics and eligibility counts.
3. Validate metric implementations on a small fixed set of RDB pairs, then run the full paired cohort.
4. Compute H5 and real-task effective-rank distributions and select median exemplars deterministically.
5. Export v6.2 train/evaluation exposure and learned-bias diagnostics.
6. Generate manuscript-ready figures and machine-readable tables.
7. Update manuscript prose only after all headline outputs are frozen.

Stop rather than broaden the analysis if a metric cannot be defined identically across R2 and R3. Move unstable threshold-dependent results to sensitivity analysis. No additional model training is required.
