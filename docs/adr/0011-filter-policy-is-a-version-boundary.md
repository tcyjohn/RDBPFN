---
status: accepted
---

# Treat the repetitive-column filter as a versioned pipeline policy

R2 retains the repetitive-column H5 filter because it is the faithful original RDB-PFN processing path. R2.5 and later variants intentionally use direct merge without that filter. The analysis will report this as part of each complete package definition rather than treating filtering as a fairness control that later variants ought to match. Filtered R3 is removed from the main-paper analysis; at most, it remains a retrospective appendix sensitivity result outside the intended R3 pipeline.
