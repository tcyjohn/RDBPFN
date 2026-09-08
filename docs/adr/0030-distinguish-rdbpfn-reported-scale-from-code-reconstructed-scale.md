# ADR-0030: Distinguish RDB-PFN reported scale from code-reconstructed scale

## Status

Accepted

## Decision

In the RQ1 scale table, the published RDB-PFN checkpoint will report:

- requested source databases: “not reported in the paper”;
- stored relational tasks: 1.2M;
- raw table cells: approximately 63.53B, marked as an estimate.

A table note will explain that the released generation schedule requests 480K
source databases and is the basis of the raw-cell reconstruction, but neither
the source-database count nor raw-cell total is reported in the RDB-PFN paper.
This preserves the distinction between author-reported quantities and
code-derived reconstruction.
