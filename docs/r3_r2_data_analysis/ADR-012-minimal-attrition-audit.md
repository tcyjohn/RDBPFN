# ADR-012: Minimal Attrition Audit

- Status: Accepted
- Date: 2026-07-18

## Decision

Keep the R3--R2 generation-attrition analysis deliberately small.

The main paper reports only:

- requested and completed RDB counts;
- completion rates (R2: 1,014/1,024; R3: 946/1,024);
- the 6.6-percentage-point completion gap;
- a compact schema-DAG summary for completed versus R3-missing identifiers, using simple quantities such as table count, edge count, and DAG depth.

Do not attempt detailed failure-stage or root-cause attribution unless reliable logs and explicit failure records become available.

## Interpretation Boundary

The schema audit may indicate whether missing R3 cases have visibly more complex schemas. It cannot identify the operational failure mechanism. If no large schema difference is observed, state only that the completion gap is not obviously explained by these coarse schema descriptors.
