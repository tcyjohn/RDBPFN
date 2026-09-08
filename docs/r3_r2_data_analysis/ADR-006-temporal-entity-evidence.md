# ADR-006: Temporal and Entity Evidence

- Status: Accepted
- Date: 2026-07-18

## Decision

Use two evidence layers for the temporal/entity axis.

### Primary evidence: realized data

Compute comparable statistics from the generated tables in the 946 paired RDBs:

- temporal-column coverage;
- repeated observations per entity;
- normalized temporal span;
- irregularity of observation intervals, where entity and timestamp semantics are available.

Report paired R3-minus-R2 differences with uncertainty. Metrics that are undefined for an RDB are marked unavailable rather than replaced with zero.

### Supporting evidence: generator configuration

Report the activation rate and parameter distributions of temporal/entity mechanisms from `generation_schemas.yaml` in the appendix.

## Rationale

Configuration statistics establish that a mechanism was requested, while realized-data statistics test whether the mechanism produced an observable corpus difference. The main-paper claim must rely primarily on the latter.

## Reporting Boundary

Do not describe a configuration difference alone as evidence of richer realized temporal structure.
