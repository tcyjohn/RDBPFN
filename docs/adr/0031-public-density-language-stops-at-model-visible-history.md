---
status: superseded by ADR-0042
---

# ADR-0031: Describe density only through model-visible temporal history

## Status

Accepted

## Decision

The main manuscript and technical account of SA-RDB-PFN will describe
\(S\sim U(5,20)\) as a temporal-history density setting whose eligibility is
structural: it applies to tables with schema out-degree at least one, regardless
of whether timestamp generation is enabled. It increases source-table
cardinality and, together with the calendar-aware prior, the temporal context
available to DFS. The text will not name internal row-group assignment,
persistent identity, or entity history as an SA-RDB-PFN mechanism because the
standard PFN receives no reliable same-entity channel.

The separate v6.2 package may discuss entity identifiers only inside the
supplement section explicitly titled “Exploratory Entity-Aware Extension
(v6.2; Not Part of SA-RDB-PFN).”
