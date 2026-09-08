# ADR-004: Paired Unit and Structural Outcomes

## Status

Accepted.

## Decision

- Use `dag_rdb_i` as the paired unit for raw/processed RDB analysis.
- Restrict paired R2–R3 structural comparisons to the 946 identifiers complete in both pipelines.
- Treat the shared schema DAG as a controlled input, not an outcome.
- Analyze the 68 R2-complete/R3-incomplete identifiers separately as attrition cases.

## Excluded headline metrics

- number of tables;
- number of schema relationships;
- relationship density;
- DAG depth or longest schema path.

These quantities are expected to match because the schema DAG is shared. Showing them as if they evidenced R3 structural richness would be misleading.

## Eligible headline outcomes

- temporal coverage and temporal organization;
- repeated-entity structure;
- FK degree/concentration/community structure;
- feature correlation/effective-rank structure;
- label balance and target-feature dependence;
- post-DFS task redundancy and benchmark-property coverage.

## Statistical implication

Use paired differences and paired uncertainty summaries where the metric is defined on both versions of the same `dag_rdb_i`. Do not treat the two corpora as independent samples for these comparisons.
