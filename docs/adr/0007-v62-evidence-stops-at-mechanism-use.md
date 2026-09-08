---
status: superseded by ADR-0017
---

# Limit v6.2 evidence to mechanism eligibility and task specialization

The v6.2 main analysis will report valid entity-ID coverage, query--support entity overlap, same-entity support counts, learned bias diagnostics, and the corrected R3--v6.2 task-wise performance profile. Evaluation exposure must be reported as heterogeneous: only 15 of 19 tasks yield usable primary-key entity identifiers, and overlap among those tasks is concentrated in a few benchmarks (mean 6.76%, median 0.48%). The step-112 checkpoint's six learned positive bias strengths (approximately 0.011, 0.096, 0.249, 0.026, 0.091, and 1.565 from a 0.1 initialization) establish parameter adaptation. These measurements show where the entity-aware mechanism is eligible and that its parameter changed during training, but they will not be used to claim that entity bias caused performance changes; exposure--performance associations remain exploratory appendix evidence.
