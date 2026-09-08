# Original RDB-PFN Data-Analysis Pattern

Source: Wang et al., *Relational In-Context Learning via Synthetic Pre-training with Structural Prior*, arXiv:2603.03805v5, Section 6.4 and Appendix A.1: <https://arxiv.org/html/2603.03805>

## What the original paper does

- Its main data-analysis figure shows representative correlation heatmaps for real single-table data, synthetic single-table data, real DFS-linearized RDB data, and synthetic RDB data.
- The interpretation is qualitative: DFS-derived feature families produce block-like correlation structure, and the synthetic relational prior reproduces a visually similar pattern.
- The benchmark appendix reports coarse dataset scale/topology statistics such as tables, columns, and rows.
- Component claims are supported mainly by performance ablations rather than a broad suite of corpus-distribution metrics.

## Implication for the R2--R3 analysis

A paper-consistent, page-efficient analysis should prioritize the data actually consumed by the model: DFS-linearized/H5 task features and their correlation-family structure. Raw-RDB temporal, FK, and signal-group metrics remain useful supporting diagnostics but need not all appear in the main paper.

Unlike the original representative-only visualization, the R2--R3 comparison should add an aggregate statistic across tasks or RDBs so that the conclusion does not depend on hand-picked heatmaps.
