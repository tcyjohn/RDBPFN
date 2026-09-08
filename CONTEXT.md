# RDB-PFN Experimental Evidence

This context defines the operating points and evidence language used in the AAAI manuscript and its supporting data analysis.

## Language

**Task efficiency**:
The ability of a relational continuation method to attain a strong fixed-checkpoint result using a smaller corpus of stored relational tasks. It does not imply fewer generated source cells, optimizer steps, sampled exposures, FLOPs, energy, or wall-clock time. Requested and completed source databases and raw table cells are reported separately; the RDB-PFN paper does not directly report these source-level quantities, while its released generation schedule supports an explicitly labeled reconstruction.
_Avoid_: Data efficiency, training efficiency, compute efficiency, sample efficiency without naming the sampling unit

**Raw table cell**:
One scalar entry in a generated pre-DFS table, including keys, timestamps, identifiers, and generated features. Corpus totals are descriptive generation-volume accounting, not information content, effective sample size, storage bytes, FLOPs, or model exposure.
_Avoid_: Information cell, training example, token, compute unit

**Estimated full-RDB-PFN raw scale**:
Approximately 63.53B raw table cells reconstructed from the released schedule of 320K Small-Prior 1-hop, 80K Small-Prior 2-hop, and 80K Large-Prior 1-hop source RDBs. This is a code-derived estimate rather than a quantity directly reported in the RDB-PFN paper.
_Avoid_: Published raw-cell count, exact measured corpus size

**Published-versus-reconstructed RDB-PFN scale**:
The RDB-PFN paper reports 1.2M stored relational tasks but not its requested
source-database count or raw-cell total. The released schedule requests 480K
source RDBs and supports the approximately 63.53B raw-cell reconstruction.
Main tables must visibly distinguish these provenance levels.
_Avoid_: RDB-PFN reports 480K RDBs, RDB-PFN reports 63.53B cells

**Paper title**:
“SA-RDB-PFN: Structure-Aware Synthetic Priors for Task-Efficient Relational In-Context Learning.” Here, task-efficient is restricted to stored relational task-corpus scale.
_Avoid_: Data-efficient, compute-efficient, raw-row-efficient

**Paper scientific identity**:
SA-RDB-PFN is a relational synthetic-prior design-and-validation paper: it asks whether a relational prior spanning connectivity, time, shared feature sources, and source-row cardinality can improve learning from a compact continuation corpus, while tracing the three semantic structures through their available raw and DFS evidence. Task-corpus reduction is a headline empirical result, not the paper's scientific identity.
_Avoid_: Experimental report, 99.9%-fewer-tasks paper, task-count reduction as the sole research question

**Central research question**:
Can a relational synthetic prior that models connectivity, time, shared feature sources, and source-row cardinality produce task-relevant databases and improve relational in-context learning from a compact continuation corpus?
_Avoid_: Can 99.9% fewer tasks match RDB-PFN?, do all three mechanisms independently cause the performance gain?

**Scientific gap**:
For DFS-based relational in-context learning, the open design question is how a synthetic relational prior should organize the structures exposed to the learner and how those choices should be validated across stages. The paper addresses this gap through three DFS-facing information channels—reachability, availability, and content dependence—plus a source-row cardinality prior, and connects intended mechanisms to generated-database signatures, model-visible DFS representations where measured, and compact-corpus predictive evidence.
_Avoid_: Task count as the scientific gap, first relational prior, first HSBM generator, claiming that every mechanism is separately validated after DFS

**Novelty boundary**:
The paper contributes relational-prior design and cross-stage validation organized around the DFS task interface: it connects intended generator structure to raw-database signatures, model-visible DFS representations where measured, and compact-corpus predictive evidence. Relational priors, HSBM generation, and synthetic-prior design are established foundations rather than broad novelty claims.
_Avoid_: First relational prior, first HSBM relational generator, first synthetic-prior design study, patch list relative to RDB-PFN

**Affirmative scope language**:
State what each experiment measures and which evidence chain is available in direct, positive language. Express boundaries through the named comparison, sampling unit, or evidence stage instead of defensive disclaimers about what the result does not prove.
_Avoid_: We do not claim, this result does not show, cannot establish, should not be interpreted as

**Gap-narrowing performance claim**:
After the same number of optimizer steps, SA-RDB-PFN substantially improves over the original-prior compact-corpus control and reduces the observed performance gap to the published RDB-PFN checkpoint continued on 1.2M stored relational tasks. SA-RDB-PFN uses 1,600 stored relational tasks, approximately 99.9% fewer. This is not a statistical-equivalence or compute-efficiency claim.
_Avoid_: Matches RDB-PFN, equivalent performance, maintains the same performance

**Assumed reader**:
A machine-learning reader who understands supervised prediction and relational tables, primary keys, and foreign keys, but is not assumed to know PFNs, RDB-PFN, DFS feature synthesis, or relational in-context learning.
_Avoid_: RDB-PFN expert, PFN-specialist reader

**Deep Feature Synthesis (DFS)**:
The deterministic relational feature-generation method of Kanter and
Veeramachaneni (2015), which composes traversal, aggregation, and transformation
operations along relationship paths.
_Avoid_: Depth-first search, depth-first synthesis

**Original RDB-PFN adaptation schedule**:
A single-table warm-up followed by a mixed adaptation stage containing both
synthetic relational tasks and single-table tasks.
_Avoid_: Single-table pre-training followed by relational-only continuation

**Original RDB-PFN temporal prior**:
The original temporal vocabulary already contains trend, seasonal, spike, and
noise components that affect timestamp sampling and latent initialization.
SA-RDB-PFN adds explicit calendar-day semantics, day-of-week effects,
monthly/yearly Fourier components, database-level shared events, and
parent--child timestamp ordering.
_Avoid_: Saying the original prior has no temporal or calendar structure

**Original RDB-PFN foreign-key dependence**:
The original generator can construct and jointly score candidate parent tuples,
sample multiple parent choices together, and update later choices through
feedback. SA-RDB-PFN contributes an explicit hierarchical community prior; it
does not introduce multi-parent dependence from scratch.
_Avoid_: Independent foreign-key sampling, weakly coordinated foreign keys

**SA-RDB-PFN**:
The only method name used in the manuscript. It denotes the complete relational-prior operating point with three semantic structural mechanisms and a source-row cardinality prior, together with the inherited task construction and PFN learner. The internal experiment label is reserved for code-facing provenance and must not become a second public method name.
_Avoid_: Current branch, entity-aware package, v6.2, complete model

**Complete SA-RDB-PFN relational prior**:
The four-dimensional generative prior comprising hierarchical foreign-key connectivity, calendar-aware temporal structure, grouped latent feature sources, and source-row cardinality. The first three are semantic structural mechanisms; cardinality controls the generated row-count distribution.
_Avoid_: Only the three semantic mechanisms, density external to the prior, four semantic mechanisms

**RQ1 — complete-SA compact utility**:
“Does the Complete SA-RDB-PFN Improve Compact-Corpus Relational In-Context Learning?” RDB-PFN-Small provides the requested-RDB-matched original RDB-PFN compact control, while the published 1.2M-task checkpoint provides the stored-task scale reference. The main contrast compares complete compact operating points because their task-selection and materialization policies differ.
_Avoid_: How task-efficient is SA-RDB-PFN?, prior-only effect, attributing the package-level comparison to one prior dimension

**RQ2 — prior-signature validation**:
“Which Intended Prior Signatures Are Observable in Generated Databases and DFS Tasks?” Generated-RDB evidence covers the three semantic mechanisms and source-row cardinality at their intervention level; DFS-task evidence traces grouped feature dependence into the model-visible representation.
_Avoid_: Claiming every semantic mechanism survives DFS, raw diagnostic presented without its predefined Method signature

**RQ2 supplement organization**:
Expanded RQ2 evidence is organized by observation stage: Generated-Database
Diagnostics followed by DFS-Task Diagnostics. The raw stage covers all
predefined generator signatures; the DFS stage covers the grouped-source
dependence currently measurable in the learner-facing representation.
_Avoid_: Four artificially symmetric mechanism sections, implying post-DFS evidence for calendar time, hierarchical connectivity, or source-row cardinality

**Temporal RDB prevalence and table fraction**:
Temporal RDB prevalence is the indicator that an RDB has at least one metadata-
declared time column present in its table file. Temporal table fraction divides
the number of such tables by all metadata tables within that RDB.
_Avoid_: Ambiguous temporal coverage, denominator restricted to timestamp-eligible tables

**Monthly activity synchrony**:
Each eligible timestamp column is converted to normalized row-count frequencies
over 612 calendar-month bins from 1970 through 2020. Synchrony is Pearson
correlation for every unordered eligible table pair and is median-aggregated
within each RDB before paired-RDB analysis.
_Avoid_: Raw row-count correlation, daily bins, pooling table pairs across databases

**Effective parent coverage**:
For one foreign-key column, effective parent coverage is
\(\exp(H(p_{\mathrm{FK}}))/N_{\mathrm{parent}}\), where \(p_{\mathrm{FK}}\)
is the empirical valid-FK assignment distribution and
\(N_{\mathrm{parent}}\) is the number of available parent rows. Values are
median-aggregated over eligible FK columns within each RDB.
_Avoid_: Unique referenced parents divided by available parents

**Multi-parent FK NMI**:
Arithmetic normalized mutual information is computed for every unordered pair
of nonconstant FK columns in the same child table using rows where both
assignments are valid, then median-aggregated within each RDB.
_Avoid_: One preselected parent pair, pooled FK rows across databases, ordered FK pairs

**Normalized effective rank**:
For eligible numericized features, non-finite values are column-median imputed,
constant columns are removed, and normalized effective rank is
\(\exp[-\sum_j p_j\log p_j]/d\), where \(p_j\) are normalized eigenvalues of
the feature correlation matrix and \(d\) is the remaining feature count.
_Avoid_: Covariance-spectrum rank, raw rank without feature-count normalization

**Strong feature-correlation rate**:
Within one DFS task, the rate is the fraction of unordered off-diagonal
upper-triangle feature pairs whose absolute Pearson correlation is at least
0.9 after the same imputation and constant-column removal. Each distribution
point represents one stored DFS task; corpus summaries take the median across
tasks.
_Avoid_: Diagonal entries, ordered pairs, one point per feature or source database, comprehensive synthetic-real distribution match

**RQ3 — prior-dimension sensitivity**:
“How Is Predictive Performance Sensitive to Source-Row Cardinality and Grouped Feature Sources?” The four fixed operating points quantify conditional AUROC differences for a cardinality dimension and a semantic content dimension across independently materialized corpora.
_Avoid_: Random pair of ablations, causal factorial effect, temporal-history effect, named auxiliary variants

**Experimental Setup narrative order**:
Experimental Setup first introduces the three RQs and the experiment or evidence used to answer each one, then defines the operating points required by those experiments, followed by generation/continuation and evaluation protocols.
_Avoid_: Operating points before their experimental purpose, hyperparameters before research questions, repeating the full Preliminaries workflow

**Results paragraph contract**:
Each RQ proceeds through motivation or expectation, measurement or contrast, result, and scientific interpretation. Effect sizes and task-level evidence answer a predefined question rather than appearing as standalone observations.
_Avoid_: Number-first paragraph, post-hoc transition, interpretation before the comparison is identifiable, defensive closing disclaimer

**Contribution hierarchy**:
The Introduction presents three contributions in this order: the complete relational-prior design; compact-corpus predictive evidence for that prior; and analysis of the induced semantic mechanisms and selected prior dimensions across generated RDBs, DFS tasks, and sensitivity operating points.
_Avoid_: Ablation as a standalone contribution, task-count report before method, cross-stage analysis before establishing utility

**Introduction running example**:
Customer churn remains the running example across the opening and design-requirement paragraphs. Linked orders and products illustrate reachability; cutoff-valid records illustrate availability; shared attribute signals illustrate content dependence; and source-table population illustrates source-row cardinality without implying a larger PFN support set or guaranteed additional history for every target row.
_Avoid_: A disconnected one-sentence example, cardinality as support size, cardinality as guaranteed per-target history

**Introduction PFN exposition**:
The second Introduction paragraph explains PFNs and RDB-PFN only at the conceptual level: synthetic tasks, support-to-query prediction, and DFS-based relational continuation. The standalone Preliminaries section owns the complete task, continuation, and inference workflow.
_Avoid_: Transformer internals in the Introduction, full pipeline repetition, assuming prior PFN or RDB-PFN knowledge

**Introduction result preview**:
The fifth Introduction paragraph reports overall findings rather than experimental settings. It states that the complete prior improves compact-corpus prediction, narrows the observed gap to the published relational checkpoint with approximately 99.9% fewer stored relational tasks, and yields the intended mechanism signatures and positive sensitivity evidence. The 99.9% reduction is the only headline number in the Introduction.
_Avoid_: Requested-RDB counts, optimizer steps, exact corpus sizes, raw-cell counts, AUROC values, confidence intervals, sensitivity-grid numbers

**Abstract organizing insight**:
The Abstract introduces relational database foundation models as cross-schema, cross-task predictors constrained by scarce real pre-training databases. It then motivates the complete prior through the DFS-facing information channels of reachability, availability, and content dependence, together with source-row cardinality. The method follows from this interface-oriented principle; approximately 99.9% fewer stored relational tasks is the sole headline number supporting the conclusion that task-aligned prior design can reduce dependence on brute-force corpus scaling.
_Avoid_: Results-only abstract, list of mechanisms without a design rationale, experimental-setting details, introducing RDB-PFN before the relational foundation-model goal

**Discussion structure**:
Discussion and Limitations proceeds from scale progress to scientific interpretation and then to the remaining performance gap. It first compares SA-RDB-PFN with the complete RDB-PFN scale—approximately 99.9% fewer stored relational tasks and 98.4% fewer raw cells while substantially narrowing the observed performance gap—then interprets the DFS-facing prior and cross-stage evidence, and finally identifies the slightly lower mean AUROC than the larger checkpoint as the central limitation and motivation for stronger prior coverage and joint prior-quality/corpus-scale study.
_Avoid_: Opening with shortcomings, compact-control raw cells as the Discussion scale comparison, training-seed caveats, support-sampling caveats, a catalog of generic experimental limitations

**Conclusion scope**:
The Conclusion restates the DFS-facing prior-design principle, summarizes the complete prior and its cross-stage evidence, and closes with approximately 99.9% fewer stored relational tasks and 98.4% fewer raw cells while substantially narrowing the observed gap to the complete RDB-PFN checkpoint. It states once that the compact comparison also changes materialization policy and notes the larger checkpoint's slightly higher mean AUROC as the remaining target for future prior-coverage improvements.
_Avoid_: Requested-RDB control details, repeated Technical-Supplement references, R3 terminology, four-cell grid details, repeated conditional differences, a second limitations catalog

**Figure 2 role**:
The two-panel task-level delta forest explicitly maps its left panel to RQ1 (complete SA-RDB-PFN versus RDB-PFN-Small) and its right panel to RQ3 (complete SA-RDB-PFN versus the low-cardinality/grouped-off coordinate). Rows use a database/prediction-target label; the caption summarizes the represented task families and defines positive delta as favoring SA-RDB-PFN. Aggregate performance remains in the corresponding tables.
_Avoid_: Joint removal, unexplained task labels, an unlabeled comparison purpose, repeated aggregate AUROC in the figure

**Figure 3 and Figure 4 placement**:
The generated-RDB signature figure and the post-DFS grouped-feature figure remain unchanged in evidential content but move next to their RQ2 discussion. Figure 3 appears at the RQ2 opening or the next page top, Figure 4 appears adjacent to the DFS analysis, and both precede Discussion and Limitations rather than forming a detached end-of-results figure page.
_Avoid_: Both figures deferred until after Results, a standalone figure-only page detached from RQ2, changing measurements to solve a float-placement problem

**Related Work organization**:
Related Work follows four research trajectories: relational predictive learning from classical relational models and DFS through relational deep learning and task-specific architectures; relational foundation-model paradigms spanning graph-centric, native multi-table, feature-materialization, and relational-ICL analysis approaches; relational database generation spanning conditional real-database synthesis and prior-based generation for pre-training; and PFNs plus synthetic-prior design. The bibliography expands to roughly 35–40 functionally cited sources, with foundation-model, HSBM, and SCM foundations cited in Introduction or Method where they support the argument.
_Avoid_: One sentence per paper, benchmark comparison as Related Work, reference-count padding, omitting RT, omitting relational database synthesis, broad first-prior claims

**R3**:
The internal generator and experiment-provenance label underlying
SA-RDB-PFN. Use SA-RDB-PFN throughout the main manuscript; reserve R3 for
technical records, supplement provenance, and code-facing descriptions.
_Avoid_: A second method name, alternating R3 and SA-RDB-PFN in the main text

**Anonymous reproducibility package**:
The isolated, sanitized submission artifact under
`aaai2027_submission/anonymous_reproducibility/sa_rdbpfn_repro/` that contains
the complete code, configurations, final and initialization checkpoints,
per-seed results, validation records, and a format-complete smoke dataset needed
to inspect and exercise the reported SA-RDB-PFN workflow. Because the submission
portal limits this artifact to 50 MB, it provides the deterministic pipeline for
regenerating the full 1,600-task training H5 rather than embedding that H5; the
`/tmp` R3 repository remains an immutable source during package construction.
_Avoid_: Claiming the full training H5 is included, copying the full R3 worktree into the active repository, modifying the `/tmp` source, anonymous archive without a manifest

**Anonymous submission evidence scope**:
The main-paper evidence chain comprising SA-RDB-PFN, the requested-RDB-matched
RDB-PFN-Small control, the four-point source-row-cardinality by grouped-source
sensitivity grid, unified evaluation, and the raw/DFS analyses that support the
main claims. v6.2 and both Mix variants are excluded from every anonymous
submission artifact, including the technical supplement and reproducibility
package.
_Avoid_: Including exploratory entity-aware or mixture results anywhere in the anonymous submission, treating technical-supplement-only experiments as part of the submission scope

**Technical Supplement contract**:
The anonymous Technical Supplement has two top-level parts: Reproducibility
Protocol and Additional Evidence and Robustness Analyses. The first specifies
the complete experimental protocol; the second expands RQ1--RQ3 evidence and
reports RDB-PFN-Small (generation gate). It is a
self-contained supporting document rather than a response to the AI-generated
review checklist, and it does not include a separate supplement roadmap.
_Avoid_: Experimental details, additional experiments, reviewer response, evidence roadmap, exploratory-results archive

**Reproducibility Protocol**:
The first Technical-Supplement part follows the experimental lifecycle:
operating points; DAG construction; synthetic database and task construction;
generation-stage quality policy; DFS and materialization; continuation
training; real-task evaluation; analysis and uncertainty; and computing
environment plus reproducibility boundary. It gives the executed protocol
rather than an idealized design. It is written for a machine-learning reader
who has not read RDB-PFN and must make both RDB-PFN and SA-RDB-PFN understandable
before presenting their executable recipes.
_Avoid_: Unordered hyperparameter inventory, code-module tour, AI-review checklist, intended-but-unexecuted protocol

**Executable lifecycle recipe**:
The two-layer presentation used throughout the Reproducibility Protocol: each
stage first states the conceptual object transformation and then records its
inputs, outputs, executed configuration or command, randomness boundary, and
completion check. It begins with DAG construction and does not assume that the
reader already knows the inherited RDB-PFN pipeline.
_Avoid_: Command dump without object semantics, conceptual overview without runnable entry points, referring readers to RDB-PFN for the inherited stages

**Pipeline Step**:
The numbered Step 0--7 lifecycle used in the Reproducibility Protocol: artifact
setup; inherited-DAG inspection; DAG-to-RDB schema instantiation; raw-row and
candidate-task generation; task acceptance plus DFS materialization;
continuation training; real-task evaluation; and analysis plus completion
checks. This name is distinct from Stage A and Stage B inside the task-quality
gate.
_Avoid_: Calling lifecycle phases Stage A/B, a code-module ordering, omitting the object transformation between commands

**Lifecycle overview table**:
The compact Part-I table that maps Pipeline Steps 0--7 to their input object,
output object, original/SA policy difference, artifact command, and completion
check. It replaces a separate lifecycle figure; the customer--order--product
example and detailed recipes remain in prose and per-step tables.
_Avoid_: A duplicated method figure, a flowchart without executable checks, using the overview table as a substitute for explaining support/query formation

**Variant overlay recipe**:
The Reproducibility Protocol presents one expanded lifecycle for the original
RDB-PFN-Small path and one for SA-RDB-PFN, then records all other reported
operating points in a compact table of configuration, policy, and artifact
differences. RDB-PFN-Small (generation gate) remains a robustness control and
does not receive a separate end-to-end procedural narrative.
_Avoid_: Repeating the full pipeline for every variant, presenting the generation-gated control as a third main method, omitting variant-specific materialization policies

**Artifact-facing recipe**:
Every command shown in the Technical Supplement uses relative paths and
entry points from the Code Supplement or anonymous reproducibility artifact.
Executed settings are preserved through reported parameters and configurations;
development-worktree scripts, absolute temporary paths, internal run names,
tmux sessions, and local GPU assignments are excluded.
_Avoid_: Mixing development and artifact commands, undocumented internal paths, presenting a local orchestration script as the public reproduction interface

**Reported-pipeline wrapper**:
The stable Code-Supplement entry point for the two expanded lifecycle recipes:
RDB-PFN-Small (original filter) and SA-RDB-PFN. It composes the historical
generator, DFS/materialization, continuation-training, evaluation, and
completion checks while enforcing the reported policy overlay, including
a source-row multiplier sampled uniformly from 5 through 20, the generation gate, SNR 5.0, and direct merge
for SA-RDB-PFN.
_Avoid_: Asking readers to compose incompatible historical defaults, a separate generation-gated-control workflow, silently applying the repetitive-column filter to SA-RDB-PFN

**Inherited DAG corpus**:
The fixed 320-DAG `rdb_v1.pth` input inherited from RDB-PFN, represented by
source-edge lists, destination-edge lists, per-node dimension metadata, and
labels. The Technical Supplement paraphrases its upstream RDB-PFN construction
provenance, records its checksum and inspection recipe, and treats loading this
file as the first locally executable stage because the upstream DAG-generation
program is not present in the repository or artifact.
_Avoid_: Claiming local regeneration of the DAG corpus, inventing a DAG-generation command, omitting the inherited provenance, copying an extended passage from RDB-PFN

**Analysis and Uncertainty Protocol**:
The Technical-Supplement protocol that identifies the sampling unit,
eligibility rule, aggregation order, interval or test, multiplicity correction,
and uncertainty boundary for every predictive and structural analysis.
Numerical outcomes belong to Additional Evidence and Robustness Analyses rather
than being mixed into these definitions.
_Avoid_: Treating aligned support samplings as training repetitions, treating stored tasks as independent RDB replicates, method-and-result interleaving

**Additional Evidence and Robustness Analyses**:
The second Technical-Supplement part contains an expanded RQ1 evaluation-support
variability analysis and RDB-PFN-Small (generation gate), followed by expanded
RQ2 generated-database/DFS diagnostics and expanded RQ3 sensitivity results.
Its scope is frozen to evidence already aligned with the main study; it does
not add rushed task-specific baselines, training-seed repetitions, or new
connectivity/time ablations in response to the AI-generated review checklist.
_Avoid_: Open-ended additional-experiment archive, checklist-driven scope expansion, defensive missing-experiment section

**Main-paper checkpoint set**:
The seven checkpoints bundled in the anonymous package: the single-table
initialization, the published-scale RDB-PFN reference, RDB-PFN-Small,
SA-RDB-PFN, and the other three coordinates of the cardinality/grouped-source
sensitivity grid. This set supports direct reevaluation of every fixed
checkpoint reported in the main paper.
_Avoid_: SA-RDB-PFN checkpoint only, omitting initialization provenance, adding v6.2 or Mix checkpoints

**Evaluation-dataset provenance**:
The 13 processed `full-512` database directories used for the 19-task evaluation
were downloaded from the pre-existing public RDB-PFN dataset release. This work
did not locally reconstruct those evaluation files from raw 4DBInfer or RelBench
inputs, so the reproducibility package records their release identifiers and
checksums rather than claiming an unexecuted preprocessing pipeline.
_Avoid_: Claiming the evaluation datasets were locally preprocessed, presenting the heuristic RelBench converter as the paper's evaluation-data pipeline, treating the evaluation files as newly introduced SA-RDB-PFN data

**Evaluation Task Manifest**:
The Technical-Supplement table that identifies all 19 classification tasks by
processed-release dataset and exact task identifier, and records the primary
metric, loaded train/test pool sizes, DFS feature count, 512-row support cap,
and release/checksum provenance. It describes the supplied processed splits
rather than claiming local reconstruction from raw benchmark data.
_Avoid_: Display names without identifiers, reconstructed official-split claim, task list without pool/query sizes

**Generation reproducibility boundary**:
Training and evaluation can be deterministic when the materialized H5 is fixed,
but the historical R3 generator cannot reproduce that H5 byte-for-byte because
its timestamp sorting path uses an unseeded NumPy random state and several
seeds derive from Python's process-salted `hash()`. The package distinguishes
the exact historical source from any post-experiment determinism hardening.
The reported-pipeline wrapper defaults to the historical source for full runs,
offers an explicit `--deterministic-generation` option, and may enable that
option by default in smoke mode; its completion manifest records the choice
and the relevant input/configuration checksums.
_Avoid_: Inferring deterministic corpus generation from deterministic retraining on a fixed H5, claiming `random_seed=42` covers every generator random source

**Reproducibility-package license**:
Apache License 2.0, inherited from the original RDB-PFN repository and retained
for the SA-RDB-PFN modifications, bundled base code, and redistributable
single-table initialization checkpoint. The anonymous artifact includes the
license text and preserves non-identifying upstream notices.
_Avoid_: Re-licensing the package as MIT, omitting the license text, removing upstream attribution required by Apache-2.0

**Structural mechanisms**:
The three semantic interventions that organize the SA-RDB-PFN method: hierarchical foreign-key connectivity, calendar-aware temporal structure, and grouped latent feature sources.
_Avoid_: Persistent relational identity, entity-aware attention, temporal-table density setting

**Model-visible prior channels**:
The three DFS-facing information channels that motivate the structural mechanisms: reachability determines which linked records can be traversed, availability determines which records are valid at prediction time, and content dependence determines what shared signal their attributes carry. They map respectively to hierarchical foreign-key connectivity, calendar-aware temporal structure, and grouped latent feature sources.
_Avoid_: Three unrelated generator patches, exhaustive taxonomy of relational structure, four mechanisms including density

**Signal-group stage boundary**:
Grouped latent feature sources belong to content generation. They consume relational context already sampled upstream—including referenced parent rows and HSBM block paths—and project temporal, parent, path, and intrinsic signals into shared feature groups. They therefore carry relational structure into generated attributes without sampling schemas, foreign-key edges, or timestamps themselves.
_Avoid_: Signal groups generate relational structure, signal groups modify foreign-key sampling, a second copy of the mechanism in both pipeline stages

**Active feature-source set**:
The feature sources available to table \(T_v\) depend on its schema role: source tables use intrinsic signals; non-timestamp child tables use referenced-parent and HSBM-path signals; timestamped child tables additionally use calendar-time signals. A generated raw attribute is \(a_{ij}\), whereas \(x_i\) is reserved for the row produced by cutoff-aware DFS.
_Avoid_: Every table combines time, parent, path, and intrinsic sources, using \(x\) for both raw attributes and DFS features

**Mechanism explanation contract**:
Each structural mechanism is explained through its DFS-facing need, prior-design principle, generation mechanism, and predefined observable signature. Scientific motivation precedes implementation detail, and every signature named in Method maps to an explicit measurement in Results.
_Avoid_: Inherited-versus-added patch description, formula-first mechanism description, post-hoc metric without a Method prediction

**Source-row cardinality prior**:
The fourth dimension of the complete relational prior: for tables with schema out-degree at least one, \(S\sim U\{5,\ldots,20\}\) multiplies the parsed and fluctuated baseline table cardinality, \(N_{\mathrm{rows}}=N_{\mathrm{base}}S\), compared with the low-cardinality setting \(S=1\). It changes raw source-table row counts; the number of labeled support rows visible to the PFN is controlled separately.
Reviewer-facing configuration uses `source_row_cardinality_min` and
`source_row_cardinality_max`; historical code may retain the legacy
`snapshots_per_entity_*` spelling only as a provenance-preserving compatibility
alias.
_Avoid_: Entity snapshot, snapshots per entity in submission-facing documentation, fourth semantic mechanism, support-size increase, guaranteed per-target history increase, persistent identity, same-entity mechanism

**Calendar-aware temporal evidence**:
The combination of timestamped-table prevalence and within-database synchrony
between calendar-time activity curves. Coverage and synchrony measure different
properties and should be reported separately. Parent--child timestamp-order
satisfaction is supplementary because the R2 eligible cohort is very small.
_Avoid_: Temporal semantics are validated, all temporal mechanisms survive DFS

**Raw-to-DFS evidence boundary**:
All three SA-RDB-PFN mechanisms receive raw-database realization diagnostics,
but only grouped latent feature sources currently have a complete generator to
raw statistic to post-DFS statistic to predictive-ablation chain.
_Avoid_: All structures survive DFS, every mechanism remains measurable after DFS

**Self-contained relational PFN workflow**:
The standalone Preliminaries explanation from synthetic schema and row generation through target construction, cutoff-aware DFS materialization, support/query formation, continuation training, and frozen real-database inference. Readers should understand how a populated RDB and one stored relational task are produced without reading the RDB-PFN paper; Method then owns the structure-aware prior design.
_Avoid_: Inherited-pipeline subsection inside Method, generator stages left implicit, patch list, assumes familiarity with RDB-PFN, DFS left undefined

**Lifecycle running example**:
A compact customer--order--product schema used once across the Reproducibility
Protocol to show DAG nodes and edges becoming tables and foreign keys, generated
columns becoming a cutoff-aware churn task, DFS aggregates becoming flat
features, and stored task rows becoming PFN support and query examples. The
same example marks the SA-RDB-PFN generation changes while keeping the inherited
representation and learner fixed.
_Avoid_: A new example per pipeline step, explaining only commands, implying that SA-RDB-PFN changes the DFS interface or PFN learner

**Generated-row lifecycle**:
The shared Method narrative for one generated row: a sampled schema defines its table and key slots; topological table generation initializes the row, samples its coordinated foreign-key tuple and timestamp, constructs its attributes from relational and latent sources, and finally makes the populated row available to cutoff-aware DFS. The same row identity must connect the connectivity, temporal, and grouped-source mechanisms rather than treating their symbols as unrelated local examples.
_Avoid_: Three disconnected mechanism formulas, assuming the reader can reconstruct the inherited RDB-PFN row-generation order

**Low-cardinality sensitivity**:
The separately generated SA-RDB-PFN operating point with \(S=1\). Its observed AUROC provides sensitivity evidence for the source-row cardinality prior across separately materialized corpora with their recorded stored-task counts.
_Avoid_: Temporal-history effect, support-size effect, clean single-factor ablation, causal effect of temporal histories

**Two-by-two sensitivity grid**:
The four fixed checkpoints formed by crossing high versus low source-row cardinality with grouped sources enabled versus disabled. Each conditional contrast measures an operating-point difference across separately materialized corpora with their recorded stored-task counts.
_Avoid_: Factorial experiment, controlled interaction ablation, causal main effect

**Conditional operating-point benefit**:
The AUROC difference for one prior choice while holding the other named choice at either its high/on or low/off coordinate. Positive conditional differences at both coordinates summarize robustness of the choice across the grid.
_Avoid_: Independent causal benefit, component effect

**Complementary conditional pattern**:
Source-row cardinality and grouped feature sources each improve mean AUROC at both settings of the other dimension, with a larger conditional difference at the other dimension's low/off coordinate. Across the evaluated operating points, they provide complementary routes to constructing predictive relational tasks.
_Avoid_: Partial substitution, synergy, causal interaction, redundancy

**Descriptive contrast-of-contrasts**:
The difference between the grouped-source conditional AUROC difference at high
cardinality and the corresponding difference at low cardinality. It is reported
as a derived fixed-checkpoint summary without interpreting the independently
materialized, unequal-task corpora as a causal interaction, substitution
effect, or diminishing return.
_Avoid_: Difference-in-differences effect, interaction estimate, partial substitution, diminishing marginal return

**Low-cardinality/grouped-off coordinate**:
The lower-left sensitivity-grid checkpoint with \(S=1\) and grouped feature sources disabled. It is referenced by its two coordinates rather than assigned a method or variant name.
_Avoid_: Joint-removal operating point, remaining-mechanisms model, HSBM-plus-calendar model, double ablation, minimal SA-RDB-PFN

**Aligned-support interval**:
A 95% paired \(t\) interval over ten evaluation seeds, where each seed uses aligned support-row sampling and the fixed 19-task suite is averaged before comparison. It quantifies evaluation-support variation only, not uncertainty over training seeds, generated corpora, or benchmark suites.
_Avoid_: Training confidence interval, model uncertainty, full experimental confidence interval

**Evaluation-support variability**:
For each real DFS task and fixed checkpoint, the mean, sample standard
deviation (denominator \(n-1\)), and min--max AUROC across the ten aligned
support samplings. Task volatility is the median of that task's sample
standard deviations across published-scale RDB-PFN, RDB-PFN-Small (original
filter), and SA-RDB-PFN; the five largest values identify the most
support-sensitive tasks. The expanded RQ1 analysis also reports pairwise
Spearman correlations of task-wise standard deviations across the three
checkpoints and Spearman correlations between task volatility and log train
pool size, log test pool size, and DFS feature count over all 19 tasks.
_Avoid_: Variance as the primary reported scale, ranking tasks from the generation-gated or RQ3 sensitivity checkpoints, defensive caveats around the descriptive summary

**Entity-aware exploratory extension**:
The v6.2 package: aligned task materialization plus same-entity attention bias, studied only in the Technical Supplement under “Exploratory Entity-Aware Extension (v6.2; Not Part of SA-RDB-PFN).”
_Avoid_: Main-text method, core SA-RDB-PFN contribution, proven performance component

**Data-prior-only operating point (R3)**:
An experimental descriptor for SA-RDB-PFN, used when contrasting it with the separately materialized entity-aware package.
_Avoid_: A separate method from SA-RDB-PFN, full model, entity-aware model

**Requested-RDB-matched baseline (R2)**:
The RDB-PFN-Small baseline that, like SA-RDB-PFN, requests 1,024 source RDBs and uses the same checkpoint initialization, PFN architecture, and optimizer horizon. It anchors the effect of changing the complete compact-corpus operating point, while raw cells, completed RDBs, stored tasks, and materialization policy are reported separately.
It preserves the original generator and post-materialization repetitive-column
filter, so the main comparison is between end-to-end compact operating points
rather than an isolated prior effect.
R2 and RDB-PFN-Small (original filter) are exact synonyms; R2 never names
RDB-PFN-Small (generation gate).
_Avoid_: Removing R2, same raw-data volume, fixed-cell control, prior-only control, calling the 263-task control R2

**Generation-stage task-quality gate**:
The staged acceptance policy used by SA-RDB-PFN and its RQ3 operating points
before DFS materialization. It screens basic task validity and label balance,
sample and nonconstant-feature sufficiency, and then surrogate-model
learnability; tasks rejected early do not run the later model-based diagnostic.
For SA-RDB-PFN it is used in place of RDB-PFN's original
post-materialization repetitive-column filter; the two policies need not reject
the same tasks.
The historical call does not activate schema-based checks. Its nominal
identifier/timestamp leakage routine can reject a task, but because those
columns are already excluded from the full surrogate feature set, its
``without-column'' refit does not implement the intended removal comparison.
This task-level gate is followed by a separate RDB-level signal-to-noise
screen in SA-RDB-PFN.
_Avoid_: Repetitive-column filter, claiming every task runs every check, folding the RDB-level SNR screen into the task-level gate, claiming correct removal-based leakage attribution or active schema screening

**Stage A task screen**:
The rule-based first stage checks load/label validity, label balance, sample and
minority-class counts, and the presence of nonconstant generated features.
Schema-independence and child/parent-ratio branches exist in source but receive
no schema context in the historical generation call. Failure short-circuits
Stage B.
_Avoid_: Model-based gate, claiming all declared checks were active

**Stage B surrogate screen**:
The model-based stage fits a median-imputed feature-only ExtraTrees OOF
surrogate, applies the historical hard-reject/gray-zone/hard-accept thresholds
and gray-zone OOF bootstrap, then executes the nominal identifier/timestamp
leakage routine. It runs only for tasks that pass Stage A.
_Avoid_: Train/test benchmark evaluation, every-task diagnostic, correctly controlled leakage-removal test

**RDB-level signal-to-noise screen**:
The SA-RDB-PFN generation-stage screen applied after task generation and
feature-diagnostic export. Its proxy is the mean active time/parent/path
signal-group scale divided by mean residual standard deviation; an RDB is
discarded when the proxy is below 5.0. When no applicable grouped-signal
diagnostics exist, the historical implementation returns infinity and the
screen passes through, so the metric is not defined for the original-prior
control.
_Avoid_: Task-level SNR test, label SNR, claiming an SNR-matched original-prior control

**Generation-stage quality policy**:
The complete historical SA-RDB-PFN acceptance sequence: staged task-quality
screening with retry and best-attempt retention, followed by the RDB-level
signal-to-noise screen, then direct DFS-task merging without the original
repetitive-column filter.
_Avoid_: Using quality gate ambiguously when the task-level and RDB-level stages need to be distinguished

**Quality-policy reporting depth**:
The Technical Supplement defines Stage A, Stage B, retry/best-attempt
selection, RDB-level SNR screening, direct merge, and their executed thresholds
with concise pseudocode. It does not attempt a rejection-reason forensic audit
that the historical logs cannot support.
_Avoid_: Black-box gate description, idealized behavior, reconstructed historical attrition percentages

**Configuration disclosure boundary**:
Technical-Supplement tables enumerate every parameter that changes the
scientific meaning of prior mechanisms, task selection, DFS/materialization,
continuation, or evaluation. Exhaustive inherited and long-tail configuration
keys remain in executable artifact files identified by exact paths and
checksums.
_Avoid_: Five-row principal-settings placeholder, full configuration-file dump, sending behaviorally central parameters only to code

**RDB-PFN-Small (generation gate)**:
The Technical-Supplement robustness control that uses the RDB-PFN-Small
original relational prior with SA-RDB-PFN's task-level generation gate, retry
and best-attempt selection, and direct merge without repetitive-column
filtering. It is a materialization-matched original-prior control, not R2 or a
causal paired ablation; grouped-signal SNR is not applicable.
_Avoid_: R2, RDB-PFN-Small without a policy qualifier, post-hoc R2, prior-only ablation, filtered RDB-PFN-Small

**Generation-gate robustness artifacts**:
The Code-Supplement checkpoint, 19-task by 10-support-sampling output,
corpus-accounting table, paired-comparison table, and provenance/checksums for
RDB-PFN-Small (generation gate). The control is reevaluable through the unified
evaluation entry point but does not receive a separate generation recipe.
_Avoid_: Reporting the control without its checkpoint and per-seed output, presenting it as a third main pipeline, omitting it from the artifact manifest

**Original-prior robustness reporting commitment**:
RDB-PFN-Small (generation gate) is reported in the Technical Supplement
regardless of whether it improves, matches, or degrades relative to
RDB-PFN-Small (original filter) or SA-RDB-PFN. Its direction changes the
interpretation, not its eligibility for inclusion.
_Avoid_: Reporting only a favorable outcome, replacing it with the post-hoc screen after seeing performance, calling this control R2

**Post-hoc no-retry R2 screen**:
The diagnostic corpus obtained by applying the task-quality checks once to the
fixed historical R2 raw tasks. Its attrition is useful process evidence, but it
is not RDB-PFN-Small (generation gate) because it cannot regenerate failed
tasks. Keep it outside the submitted evidence unless the native control cannot
be completed, in which case it may appear only as an explicitly labeled
fallback sensitivity analysis.
_Avoid_: Gate-matched R2, native quality-gated R2

**Entity-aware package (v6.2)**:
The separately materialized variant that combines the structure-aware data prior, aligned entity-centered task format, valid entity identifiers, and same-entity attention bias.
_Avoid_: Mix, data-prior-only variant

**Legacy-materialization Mix**:
The 0.7118 training checkpoint that samples v6.2 and an earlier 1,169-task R3 materialization with equal source weights; same-entity bias is effective only on batches carrying valid entity identifiers.
_Avoid_: Source-aligned Mix, full model

**Source-aligned corrected Mix**:
The later 0.7079 training checkpoint that samples v6.2 and the reported 1,600-task direct-R3 source with equal weights.
_Avoid_: Legacy Mix, highest observed checkpoint

**Complete package**:
Informal conversational shorthand for v6.2 when contrasting it with data-prior-only R3. In the manuscript, prefer the precise term “entity-aware package” because “complete” does not identify the task format, corpus materialization, or bias exposure.
_Avoid_: Mix

**Aggregate parity**:
The observed near-equality of v6.2 and R3 suite-mean AUROC, with a paired uncertainty interval that includes zero; it is not a formal equivalence claim.
_Avoid_: Superiority, proven equivalence

**Task specialization**:
A redistribution of relative performance across evaluation tasks despite similar aggregate performance, including task-specific differences that remain after multiple-comparison correction.
_Avoid_: Uniform improvement

**Source distribution**:
One materialized training corpus sampled directly by the continuation loader, such as the v6.2 source or the earlier R3 source used by Mix.
_Avoid_: Generator when referring only to a materialized corpus

**Derived mixture**:
A training distribution created by sampling existing source distributions with fixed weights. Both Mix variants are derived mixtures and not third relational generators.
_Avoid_: New data prior

**Raw-to-DFS measurability**:
Evidence that an intended generator mechanism is present in realized raw database instances and remains measurable after transformation into the DFS-linearized training representation.
_Avoid_: Causal performance attribution

**DFS signature**:
A measurable property of the final linearized task features that can be traced descriptively to relational generation or aggregation, such as block-like feature correlation.
_Avoid_: Proof of downstream causality

**Benchmark alignment**:
Agreement between synthetic and real evaluation-task property distributions under identically defined measurements. It is supporting evidence and is not implied merely by wider synthetic coverage.
_Avoid_: Realism without an explicit metric

**Entity exposure**:
The opportunity for same-entity attention to operate, measured through valid entity identifiers and overlap between query entities and support entities.
_Avoid_: Entity-bias effectiveness

**Evaluation entity-exposure heterogeneity**:
Under the full-512 protocol, 15 of 19 classification tasks yield usable primary-key entity identifiers. Across those 15 tasks, query--support entity overlap has mean 6.76% and median 0.48%; it is concentrated in rel-event user-repeat (30.65%), rel-f1 driver-dnf (26.17%), and rel-f1 driver-top3 (34.04%), while most tasks are near zero. Report both the median and the task-wise distribution; the mean alone obscures this concentration.
_Avoid_: Saying that same-entity bias is uniformly active at evaluation time

**Learned entity-bias diagnostic**:
At the reported v6.2 step-112 checkpoint, the six transformer blocks have positive softplus-constrained same-entity bias strengths and several differ substantially from the 0.1 initialization (approximately 0.011, 0.096, 0.249, 0.026, 0.091, and 1.565). This establishes parameter adaptation, not a causal contribution to AUROC.
_Avoid_: Treating a nonzero learned parameter as an ablation

**Component-specific evidence**:
Measurements selected to match a component's actual intervention point, such as DFS feature correlations for the data prior and entity exposure for the entity-aware interface.
_Avoid_: One metric family applied indiscriminately to every variant

**Headline metric**:
A measurement committed for main-paper reporting before its observed direction is inspected. Headline metrics are reported even when they do not favor the proposed variant.
_Avoid_: Best-looking metric

**Paired-delta forest plot**:
A task-wise visualization of aligned-seed performance differences and confidence intervals, used in place of a dense multi-variant numeric table.
_Avoid_: Uncertainty-free win/loss chart

**Question-aligned task tables**:
The Technical Supplement reports complete 19-task predictive results in
separate RQ1 and RQ3 tables. The RQ1 table reports published-scale RDB-PFN,
RDB-PFN-Small (original filter), and SA-RDB-PFN as mean AUROC plus or minus
sample standard deviation across ten aligned support samplings; a companion
table gives min--max AUROC for the five tasks with largest evaluation-support
variability. The RQ3 table contains the four cardinality/grouped-source
coordinates and their conditional summaries. The generation-gated control is
summarized through corpus accounting, suite statistics, and its task-delta
forest rather than a duplicated four-model task-mean table.
_Avoid_: One omnibus table mixing the scale reference, compact controls, and sensitivity grid; selected favorable tasks

**Technical-Supplement figure increment**:
Supplement figures must add evidence beyond the main paper. The
RDB-PFN-Small (generation gate) section may add a two-contrast task-delta forest
after evaluation completes, RQ2 uses detailed numerical tables rather than
repeated main figures, and RQ3 retains the four-conditional-contrast forest
that expands the main comparison.
_Avoid_: Reprinting main figures, uncertainty-free win/loss graphic, aligned-support intervals labeled as training uncertainty

**Conditional metric**:
A measurement defined only for units where a mechanism exists, such as temporal span among timestamp-enabled tables. Absence of the mechanism is not encoded as zero conditional strength.
_Avoid_: Unconditional effect

**Mechanism eligibility**:
The presence of inputs and overlap conditions required for a mechanism to operate, such as valid entity identifiers shared between query and support rows.
_Avoid_: Mechanism effectiveness

**Mechanism use diagnostic**:
Evidence that a trainable mechanism changed from initialization or was active on eligible examples. It does not isolate the mechanism's contribution to predictive performance.
_Avoid_: Causal ablation

**Median exemplar**:
The eligible task whose predefined summary statistic is closest to its corpus median, used for deterministic representative visualization.
_Avoid_: Best-looking example, representative example without a selection rule

**Corpus-level H5 analysis**:
A comparison of final stored task distributions when source RDB identifiers are unavailable. It is descriptive and cannot recover raw-RDB pairing or clustering.
_Avoid_: Paired task analysis

**Identifiable sampling unit**:
The finest unit whose correspondence and dependence structure are retained well enough for inference, such as shared RDB identifiers or aligned evaluation seeds.
_Avoid_: Treating stored task rows as independent replications

**Original filter policy**:
The repetitive-column H5 filtering stage retained by R2 as part of the original RDB-PFN preprocessing pipeline.
_Avoid_: A filter that every later variant should use

**Direct-merge policy**:
The intended H5 materialization policy from R2.5 onward, in which accepted DFS tasks are merged without the original repetitive-column filter.
_Avoid_: Unfiltered by mistake

**DFS-correlation contextual alignment**:
A narrow comparison of synthetic and real linearized-task correlation structure using the original RDB-PFN visual language and aggregate effective-rank summaries.
_Avoid_: Comprehensive benchmark realism

**Four-pass manuscript polish**:
The final manuscript-wide check performed after the scientific structure is stable: paragraph purpose and logic; transitions and academic expression; terminology and claim boundaries; then grammar, notation, LaTeX, references, citations, floats, and self-contained captions.
_Avoid_: Sentence-level polishing before structural revision, defensive negative declarations
