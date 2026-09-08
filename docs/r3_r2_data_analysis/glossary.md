# R3 vs. R2 Data Analysis Glossary

## Confirmed terms

- **Requested database**: One intended synthetic RDB generation attempt; both variants start from 1,024 requests.
- **Complete RDB**: A raw generation directory that contains the artifacts required for downstream preprocessing.
- **Processed RDB**: A database that completed the DFS preprocessing stage.
- **Direct merge**: Merging all materialized tasks into H5 without applying the original repetitive-column filter.
- **Repetitive-column filter**: A stage of the original RDB-PFN H5 processing path retained by R2. It is intentionally absent from R2.5 and later direct-merge variants.
- **Identifier-derived feature exclusion**: Removal of raw or DFS-derived entity/table identifier columns during H5 merge. It is already present in both R3 direct and R3 filtered H5 files, so it is not the treatment difference in the filtered-vs-direct R3 comparison.
- **Filtered R3**: A retrospective 1,169-task sensitivity corpus obtained by applying the original R2 filter to R3; it is not the intended R3 pipeline policy.
- **Direct R3**: The 1,600-task R3 H5 produced by direct merge after identifier exclusion but without the additional repetitive-column filter.
- **Stored task**: One task instance stored in an H5 corpus. It measures corpus contents, not training exposures or unique semantic tasks.
- **Package-level comparison**: A comparison of complete R2 and R3 pipelines, allowing multiple simultaneous differences but not attributing the result to one component.
- **Matched analysis**: A comparison in which a chosen confound, such as filtering policy or task count, is held constant or adjusted explicitly.
- **Paired RDB cohort**: The 946 common `dag_rdb_i` identifiers with complete R2 and R3 artifacts; the same schema DAG underlies each pair.
- **Attrition cohort**: All 1,024 requested RDB identifiers analyzed by completion status rather than by generated-data properties.
- **Data-prior-only operating point (R3)**: The reported standalone structure-aware-prior variant with original-style task construction and no attention bias.
- **Entity-aware package (v6.2)**: The separately materialized variant combining the structure-aware data prior, aligned entity-centered task format, valid entity identifiers, and same-entity attention bias. This is what “完全体” referred to in planning discussions.
- **Legacy-materialization Mix**: The 0.7118 checkpoint combining v6.2 with an earlier 1,169-task R3 materialization; it is not the “完全体,” and entity bias is a no-op on R3 batches without valid identifiers.
- **Source-aligned corrected Mix**: The 0.7079 checkpoint combining v6.2 with the reported 1,600-task direct-R3 source; it is distinct from the legacy Mix.

## Terms requiring agreement

- **Data quality**: Must be operationalized; possible meanings include validity, non-redundancy, learnability, leakage avoidance, structural richness, or benchmark alignment.
- **Diversity**: Must specify the level: schema, RDB, task, feature distribution, label mechanism, or effective training diversity.
- **Structural richness**: Must identify measurable structures, such as schema depth, FK topology, repeated entities, temporal coverage, or source-depth diversity.
- **Fair comparison**: Must state which quantities are matched and which remain package-level differences.

## Resolved distinctions

- “Two filters in filtered R3” describes two processing protections present in the final H5, but only the repetitive-column filter differs from direct R3; identifier exclusion is shared.
- R2 uses the repetitive-column filter because it reproduces the original RDB-PFN processing path. R2.5 and later variants intentionally use direct merge, so filter alignment is not a design requirement for their primary comparison.
- A confidence interval across evaluation seeds is not a confidence interval across training runs.
