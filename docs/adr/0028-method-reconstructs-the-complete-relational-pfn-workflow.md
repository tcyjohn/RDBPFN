---
status: superseded by ADR-0036
---

# ADR-0028: Reconstruct the complete relational PFN workflow in Method

## Status

Accepted

## Decision

The Method section will be understandable without prior familiarity with
RDB-PFN or PFNs. It will proceed in five stages:

1. formally define the relational task interface, including the database,
   target table and rows, prediction cutoff, support/query construction, and
   DFS materialization;
2. summarize the inherited RDB-PFN pipeline from schema sampling through
   continuation training;
3. define the three structure-aware mechanisms, each in terms of the inherited
   behavior, the R3 change, its generation rule, and its expected observable
   statistic;
4. disclose the temporal-history density setting in a separate subsection;
5. restate task construction, identifier filtering, DFS materialization, and
   the unchanged PFN training objective.

Figure 1(a) will provide a concrete database and DFS example that is reused by
the prose. Table 1 will provide an inherited-versus-changed comparison so that
the prose need not repeat feature lists.

The additional Method space will come from moving entity-aware variants and
secondary results to the Technical Supplement and removing repetitive result
descriptions.
