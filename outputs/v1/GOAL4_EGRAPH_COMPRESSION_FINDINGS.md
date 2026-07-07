# Goal 4 E-Graph Compression Findings

This report mines v1 corpus e-graph compression outputs from saved Goal 4.6 and Goal 4.7 artifacts. The v0 corpus is pilot only and is not used for these result-bearing findings.

E-graphs here are non-ML compression: they search algebraic equivalences and then recompile the selected source expression through the official pure EML compiler. Improvements are structural compression results, not GNN evidence or model-performance evidence.

`safe` and `positive_real_formal` are separate modes. `positive_real_formal` uses branch-sensitive positive-real assumptions and must not be mixed with safe-mode results. Successful final EML outputs remain official pure EML after extraction; rows with timeout or validation failure are kept visible instead of being silently dropped.

`nontrivial_v1` and `identity_heavy_v1` are reported separately to avoid overstating easy simplifications from identities such as multiplication by one, log one, or log/exp cancellation.

## Mode Summary

| Rule mode | Rows | Valid non-timeout successes | Timeouts | Validation failures | Pure failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `safe` | 10000 | 9288 | 241 | 684 | 0 |
| `positive_real_formal` | 10000 | 8880 | 522 | 1059 | 0 |

## Top Safe Successes

| Rank | Index | Mode | Score | Gain | Optimized alpha | Subset | Expression |
| ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | 1491 | `safe` | 36.998 | 27 | 0.2 | `identity_heavy_v1` | `(1*1)*x` |
| 2 | 5655 | `safe` | 36.998 | 27 | 0.2 | `identity_heavy_v1` | `(1*1)*y` |
| 3 | 308 | `safe` | 33.998 | 24 | 0.2 | `identity_heavy_v1` | `(y*1)*1` |
| 4 | 2360 | `safe` | 33.998 | 24 | 0.2 | `identity_heavy_v1` | `(1*x)*1` |
| 5 | 50 | `safe` | 25.9967 | 16 | 0.333333 | `identity_heavy_v1` | `y*1` |

## Top Positive-Real Successes

| Rank | Index | Mode | Score | Gain | Optimized alpha | Subset | Expression |
| ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | 204 | `positive_real_formal` | 73.9993 | 64 | 0.0714286 | `identity_heavy_v1` | `1*x + ((y*y)*(x + y))*log(1)` |
| 2 | 4609 | `positive_real_formal` | 70.9992 | 61 | 0.0769231 | `identity_heavy_v1` | `exp((x*x + y*y)*log(1*1))` |
| 3 | 4458 | `positive_real_formal` | 68.9993 | 59 | 0.0714286 | `identity_heavy_v1` | `y + ((x + 1)*(y + 1))*log(1*1)` |
| 4 | 6286 | `positive_real_formal` | 68.9993 | 59 | 0.0714286 | `identity_heavy_v1` | `exp(((x + x)*log(1))*(y*y + exp(1)))` |
| 5 | 4481 | `positive_real_formal` | 65.9992 | 56 | 0.0769231 | `identity_heavy_v1` | `y + ((y + 1)*exp(x))*log(1*1)` |

## Top Safe Failures

| Rank | Index | Mode | Score | Gain | Optimized alpha | Subset | Expression |
| ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | 5487 | `safe` | 2105.56 |  |  | `identity_heavy_v1` | `y*(x*(1*(x*y)))` |
| 2 | 3428 | `safe` | 2105.46 |  |  | `identity_heavy_v1` | `y*(((1*x)*(x*y))*log(exp(x)))` |
| 3 | 5397 | `safe` | 2105.29 |  |  | `identity_heavy_v1` | `((1*x)*exp(1))*((x*y)*(x + y))` |
| 4 | 9523 | `safe` | 2105.25 |  |  | `identity_heavy_v1` | `y*(((1*x)*(y + y))*log(x))` |
| 5 | 4122 | `safe` | 2105.23 |  |  | `identity_heavy_v1` | `((x*1)*exp(x))*((x*y)*log(1))` |

## Top Positive-Real Failures

| Rank | Index | Mode | Score | Gain | Optimized alpha | Subset | Expression |
| ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | 6065 | `positive_real_formal` | 2105.62 |  |  | `identity_heavy_v1` | `y*((x*y)*log(1))` |
| 2 | 5487 | `positive_real_formal` | 2105.56 |  |  | `identity_heavy_v1` | `y*(x*(1*(x*y)))` |
| 3 | 3428 | `positive_real_formal` | 2105.46 |  |  | `identity_heavy_v1` | `y*(((1*x)*(x*y))*log(exp(x)))` |
| 4 | 2491 | `positive_real_formal` | 2105.33 |  |  | `identity_heavy_v1` | `((x*y)*exp(1))*log(1)` |
| 5 | 5397 | `positive_real_formal` | 2105.29 |  |  | `identity_heavy_v1` | `((1*x)*exp(1))*((x*y)*(x + y))` |

## Best Operator Signatures

| Mode | Signature | Count | Successes | Median optimized alpha | Median gain | Percent improved | Timeout rate | Validation failure rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `positive_real_formal` | `exp+log` | 29 | 28 | 0.6 | 2.3333333333333335 | 100.0 | 0.0 | 3.4482758620689653 |
| `positive_real_formal` | `Mul+log` | 185 | 172 | 3.625 | 1.4482758620689655 | 90.11627906976744 | 7.027027027027027 | 7.027027027027027 |
| `positive_real_formal` | `Mul+exp+log` | 563 | 493 | 3.25 | 1.4444444444444444 | 89.65517241379311 | 5.683836589698046 | 12.433392539964476 |
| `positive_real_formal` | `leaf_only` | 3 | 3 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| `safe` | `leaf_only` | 3 | 3 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| `positive_real_formal` | `Mul` | 60 | 58 | 4.285714285714286 | 1.3478260869565217 | 82.75862068965517 | 3.3333333333333335 | 3.3333333333333335 |
| `safe` | `Mul` | 60 | 58 | 4.285714285714286 | 1.3478260869565217 | 82.75862068965517 | 3.3333333333333335 | 3.3333333333333335 |
| `safe` | `Mul+log` | 185 | 180 | 4.25 | 1.2531746031746032 | 74.44444444444444 | 2.7027027027027026 | 2.7027027027027026 |

## Worst Operator Signatures

| Mode | Signature | Count | Successes | Median optimized alpha | Median gain | Percent improved | Timeout rate | Validation failure rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `positive_real_formal` | `exp` | 12 | 7 | 1.3333333333333333 | 1.0 | 0.0 | 0.0 | 41.666666666666664 |
| `safe` | `exp` | 12 | 7 | 1.3333333333333333 | 1.0 | 0.0 | 0.0 | 41.666666666666664 |
| `positive_real_formal` | `Mul+exp` | 307 | 225 | 3.8 | 1.3333333333333333 | 80.44444444444444 | 0.9771986970684039 | 26.710097719869708 |
| `safe` | `Mul+exp` | 307 | 248 | 4.0 | 1.2242857142857142 | 70.56451612903226 | 0.9771986970684039 | 19.218241042345277 |
| `positive_real_formal` | `Add+exp` | 290 | 252 | 3.1666666666666665 | 1.0 | 24.206349206349206 | 0.3448275862068966 | 13.10344827586207 |
| `safe` | `Add+exp` | 290 | 252 | 3.1666666666666665 | 1.0 | 24.206349206349206 | 0.0 | 13.10344827586207 |
| `positive_real_formal` | `Add+Mul+exp` | 1910 | 1598 | 3.5714285714285716 | 1.129319955406912 | 64.08010012515645 | 2.9842931937172774 | 16.335078534031414 |
| `safe` | `Add+Mul+exp` | 1910 | 1647 | 3.6666666666666665 | 1.0857142857142856 | 60.04857316332726 | 2.094240837696335 | 13.769633507853403 |

## Safe Regime Candidates

| Subset | Signature | Count | Percent below after | Percent improved | Timeout rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `nontrivial_v1` | `leaf_only` | 3 | 100.0 | 0.0 | 0.0 |
| `all_v1` | `leaf_only` | 3 | 100.0 | 0.0 | 0.0 |
| `nontrivial_v1` | `exp` | 12 | 100.0 | 0.0 | 0.0 |
| `all_v1` | `exp` | 12 | 100.0 | 0.0 | 0.0 |
| `all_v1` | `exp+log` | 29 | 25.0 | 0.0 | 0.0 |
| `identity_heavy_v1` | `exp+log` | 29 | 25.0 | 0.0 | 0.0 |
| `identity_heavy_v1` | `Mul` | 41 | 17.94871794871795 | 100.0 | 4.878048780487805 |
| `all_v1` | `Mul` | 60 | 12.068965517241379 | 82.75862068965517 | 3.3333333333333335 |

## Subset-Specific Successes

Top `nontrivial_v1` successes:

| Rank | Index | Mode | Score | Gain | Optimized alpha | Subset | Expression |
| ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | 9315 | `positive_real_formal` | 1.83381 | 1.85714 | 2.33333 | `nontrivial_v1` | `log(exp(1)*exp(y))` |
| 2 | 2939 | `positive_real_formal` | 1.72333 | 1.75 | 2.66667 | `nontrivial_v1` | `log(exp(x)*exp(y))` |
| 3 | 2465 | `positive_real_formal` | 1.70333 | 1.73333 | 3 | `nontrivial_v1` | `log(x*exp(x))` |
| 4 | 5874 | `positive_real_formal` | 1.70333 | 1.73333 | 3 | `nontrivial_v1` | `log(y*exp(y))` |
| 5 | 5834 | `positive_real_formal` | 1.69095 | 1.71429 | 2.33333 | `nontrivial_v1` | `log((y + 1)*exp(y)) + 1` |

Top `identity_heavy_v1` successes:

| Rank | Index | Mode | Score | Gain | Optimized alpha | Subset | Expression |
| ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | 204 | `positive_real_formal` | 73.9993 | 64 | 0.0714286 | `identity_heavy_v1` | `1*x + ((y*y)*(x + y))*log(1)` |
| 2 | 4609 | `positive_real_formal` | 70.9992 | 61 | 0.0769231 | `identity_heavy_v1` | `exp((x*x + y*y)*log(1*1))` |
| 3 | 4458 | `positive_real_formal` | 68.9993 | 59 | 0.0714286 | `identity_heavy_v1` | `y + ((x + 1)*(y + 1))*log(1*1)` |
| 4 | 6286 | `positive_real_formal` | 68.9993 | 59 | 0.0714286 | `identity_heavy_v1` | `exp(((x + x)*log(1))*(y*y + exp(1)))` |
| 5 | 4481 | `positive_real_formal` | 65.9992 | 56 | 0.0769231 | `identity_heavy_v1` | `y + ((y + 1)*exp(x))*log(1*1)` |

## Output Files

- `outputs/v1/top_egraph_compression_successes_safe.csv`
- `outputs/v1/top_egraph_compression_successes_positive_real.csv`
- `outputs/v1/top_egraph_compression_failures_safe.csv`
- `outputs/v1/top_egraph_compression_failures_positive_real.csv`
- `outputs/v1/best_egraph_operator_signatures.csv`
- `outputs/v1/worst_egraph_operator_signatures.csv`
- `outputs/v1/egraph_safe_regime_candidates.csv`
- `outputs/v1/egraph_nontrivial_successes.csv`
- `outputs/v1/egraph_identity_heavy_successes.csv`
