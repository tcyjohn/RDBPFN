# Archive Note: RDBPFN Pre-HSBM Baseline

**Archive date:** 2026-05-11
**Source branch:** master (no commits — fresh repo, all files untracked)
**Baseline:** No commits exist; baseline is the working-tree state of the four source files below.

## Files Preserved

| File | MD5 |
|------|-----|
| `table_generation.py.bak` | `15a9ca327633d570985f5d4affe32f40` |
| `mlp_scm.py.bak` | `ebbcc7ae89ceeb8f37025c04949876d9` |
| `prior_config.py.bak` | `ff45f6ba5a28df3b1b813d99ec2dd08b` |
| `utils.py.bak` | `ab6e2d8eca07bb68e2da13b499d02bfc` |

## Current FK-Generation Mechanism (Selective SCM)

- Child tables oversample 10x rows (`sampling_ratio=10.0`), each paired with a randomly-sampled parent row (uniform or Zipf)
- SCM MLP allocates one output dimension for `EDGE_PROB` (via `MASK_TYPE.EDGE_PROB`)
- `forward_with_input()` in `mlp_scm.py:481` generates candidate rows + edge probabilities
- `sample_final_output()` in `mlp_scm.py:768` uses multinomial to select `seq_len` rows from the oversampled pool
- Surviving rows' parent indices become FK column values in the child table

## Planned HSBM Replacement

- Decouple FK generation from feature generation
- New `hsbm.py` module: hierarchical stochastic block model for bipartite parent-child connectivity
- FK connections determined before SCM runs; `forward_with_input()` receives pre-computed `fk_ids`
- No more EDGE_PROB, oversampling, or selective sampling in SCM

## Known Risks

- **Seed derivation**: HSBM needs deterministic per-relation seeding for reproducibility
- **HSBM hierarchy validation**: Must ensure `max_leaf_blocks <= min(parent_rows, child_rows)` or sampling degenerates
- **FK distribution drift**: HSBM produces block-clustered connections vs. uniform/Zipf parent sampling — downstream model performance may shift
- **Backward compat**: `MASK_TYPE.EDGE_PROB` kept in enum but unused; `parent_sampling_dist`/`alpha` kept but ignored

## Recovery Instructions

```bash
# Restore original files from backup
cd /data/caijunyu/RDBPFN
cp .archive/table_generation.py.bak data_generation/RDB/src/table_def/table_generation.py
cp .archive/mlp_scm.py.bak data_generation/RDB/src/prior/mlp_scm.py
cp .archive/prior_config.py.bak data_generation/RDB/src/prior/prior_config.py
cp .archive/utils.py.bak data_generation/RDB/src/prior/utils.py

# Remove hsbm.py if it was created
rm -f data_generation/RDB/src/prior/hsbm.py
```
