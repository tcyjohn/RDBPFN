# Task Plan: Temporal Generation Refactor

## Goal

Refactor time generation logic to fix three problems:

1. **虚假相关性**: MLP reads raw time value → learned shortcut between sampling density and feature values
2. **跨表时序不一致**: Child table timestamps independent of parent timestamps
3. **FK 组内无结构**: No differentiation of timestamps within same FK group

## Design Spec

`docs/superpowers/specs/2026-05-18-temporal-generation-refactor-design.md`

## Key Design Decisions

- `time_dim = 11` (fixed): trend(2) + seasonal(4) + spike(2) + gates(3)
- MLP reads basis vector B(t_i), not raw t_i
- Noise stays in sampler, NOT passed to MLP
- Child t_min = max(parent timestamps), with DAG-topology fallback
- Gamma decay uses normalized relative time (discrete tiers)
- Probabilistic per-FK-group sort (p_sort)
- No Fourier encoding, no latent state Z(t), no Hawkes process

## Phases

### Phase 1 — Basis Vector Input to MLP (Problem 1)

- [ ] Add `TemporalVocab.evaluate_basis(t)` — evaluate trend/seasonal/spike at arbitrary t
- [ ] Implement per-component eval methods (TrendVocab, SeasonalityVocab, SpikesVocab)
- [ ] Rename `_sample_time_embedding` → `_prepare_time_features` in MLPSCM
- [ ] Rewrite `_prepare_time_features` to return (N, 11) basis+gate tensor
- [ ] Build gate indicators in TableGenerator: [trend_active, seasonal_active, spike_active]
- [ ] Remove Fourier encoding and `time_embed_mode` config
- [ ] Update `time_dim` to fixed constant 11
- [ ] Remove `time_dim` from dynamic HP sampling (no longer sampled)
- **Verify:** Generate a table, check that time features shape = (N, 11), gates match component activation

### Phase 2 — Cross-Table Temporal Constraints (Problem 2)

- [ ] Modify `TemporalVocab.sample_time()` to accept `t_min: (N,)` tensor
- [ ] Implement per-row masking + batched multinomial (accept-reject on discretized intensity)
- [ ] In `_prepare_time_features`, collect parent timestamps via FK mapping, compute per-row t_min
- [ ] Implement DAG topology fallback (source/intermediate/leaf)
- [ ] Implement lifecycle decay with discrete gamma tiers + normalized tau
- [ ] Sample `gamma_tier` per table from {0.0, 0.5, 1.5, 3.0}
- **Verify:** Child timestamps >= max(parent timestamps) for all rows

### Phase 3 — Intra-Group Sort (Problem 3)

- [ ] Sample `p_sort` per table from Uniform(0.3, 0.8)
- [ ] After sampling, for each FK group: with prob p_sort, sort ascending
- **Verify:** Table with p_sort=1.0 has timestamps sorted within each FK group; p_sort=0.0 does not

### Phase 4 — Configuration & Cleanup

- [ ] Add new HP params to `prior_config.py`: trend_active, seasonal_active, spike_active, gamma_tier, p_sort
- [ ] Remove `time_embed_mode` from all config paths
- [ ] Update `dag_to_rdb_generator.py` timestamp_config defaults
- [ ] Pass DAG topology info for fallback
- **Verify:** Full pipeline run — no config errors, output data valid

## Open Questions

1. Exact probability values for trend_active, seasonal_active, spike_active Bernoulli params? ✅ Confirmed
2. Fallback t_min distribution ranges ✅ Finalized: Source U(0, 0.15T) / Intermediate U(0.15T, 0.45T) / Leaf U(0.45T, 0.85T)
3. Gamma tiers (0.0, 0.5, 1.5, 3.0) ✅ Confirmed
