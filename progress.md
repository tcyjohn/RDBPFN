# Progress Log

## Session: 2026-05-18 — Temporal Generation Refactor Brainstorming

### Context
User identified that current time generation logic is unreasonable. Three specific problems emerged during discussion.

### [x] Code Exploration
- Read all time-related code: `temporal_vocab.py`, `mlp_scm.py`, `table_generation.py`, `prior_config.py`, `dag_to_rdb_generator.py`
- Traced full timestamp generation pipeline: `TemporalVocab.generate()` → `sample_time()` → `_sample_time_embedding()` → MLP → `_convert_to_datetime64()`
- Confirmed HSBM determines FK mapping, not row counts: `child_rows = self.num_rows`
- Confirmed default timestamp prob=1.0 (all tables get timestamp)

### [x] Problem Definition
1. **虚假相关性**: raw time passed to MLP → shortcut learning from sampling intensity
2. **跨表时序不一致**: independent parent/child timestamp sampling
3. **FK 组内无结构**: no per-group temporal structure

### [x] Design Brainstorming
- Discussed Z(t) latent state approach (Gemini's proposal) — accepted core idea, simplified to basis vectors
- Rejected: full NHPP with thinning, Hawkes process, noise in MLP input, raw time fallback
- Agreed: fixed time_dim=11, gate indicators for 0-masking, DAG topology fallback, probabilistic sort
- Discussed gamma tiers and normalized relative time for lifecycle decay

### [x] Design Spec Written
- Spec: `docs/superpowers/specs/2026-05-18-temporal-generation-refactor-design.md`
- Three phases: (1) basis vector input, (2) cross-table constraints, (3) intra-group sort
- Each phase independently testable

### [x] Planning Files Updated
- `task_plan.md`, `findings.md`, `progress.md` updated for this task

### [ ] Phase 1 Implementation
- Pending user approval of spec

---

## Session: 2026-05-17 — HSBM FK Generation Bug Verification

### [x] Bug 1 Verification
- Confirmed `_clip_hierarchy` fails when `clusters_per_level > min_rows` even at 1 level

### [x] Bug 2 Verification  
- Confirmed already fixed in commit `ef4ba75` via `compute_hsbm_fk_ids_multi`

### [x] Bug 1 Fix Implemented
- Commit `82c8ebc`: clip clusters_per_level in HSBM hierarchy
