# Separate the historical generator from determinism hardening

The anonymous package preserves the exact R3 source used to create the reported
training corpus and provides seed-hardening changes as a separately identified
post-experiment patch. This keeps experimental provenance auditable while
allowing reviewers to remove Python hash randomization and the unseeded
timestamp-sort RNG for future deterministic generation; the patch is not
presented as code used to obtain the reported results.
