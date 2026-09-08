# ADR-009: Benchmark Alignment Rule

- Status: Accepted
- Date: 2026-07-18

## Decision

Evaluate benchmark alignment with two complementary quantities for each task-level property shared by synthetic and evaluation data.

1. **Benchmark coverage:** the proportion of the 19 evaluation tasks whose property value lies within the synthetic task distribution's 5th--95th percentile interval.
2. **Robust Wasserstein distance:** the one-dimensional Wasserstein distance between synthetic and benchmark task distributions after scaling the property by a robust benchmark scale (IQR, with a documented fallback for zero IQR).

Use identical property definitions and transformations for R2, R3, and benchmark tasks. Estimate uncertainty by resampling benchmark tasks and synthetic sampling units at their appropriate level.

## Interpretation Rule

Claim improved alignment only when R3 does not reduce benchmark coverage and also reduces robust distributional distance relative to R2. If coverage and distance disagree, report the result as a trade-off rather than a directional improvement.

## Rationale

Coverage alone rewards arbitrarily broad synthetic distributions, whereas distance alone can conceal missed tails. Requiring both prevents “broader” from being treated automatically as “better aligned.”

## Limitations

The 19 evaluation tasks are not a random sample of all relational learning problems. Alignment conclusions are therefore specific to the evaluated benchmark suite.
