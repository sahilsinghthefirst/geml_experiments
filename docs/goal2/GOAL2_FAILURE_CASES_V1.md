# Goal 2 Failure Cases

This report mines raw official pure EML expansion failures from saved Goal 2 outputs.

Important: this is structural evidence about tree expansion only. It is not model-performance evidence.

Current alpha threshold used by raw rows: `1.5578858913022597`.

## Highest-Alpha Examples

| Rank | Index | Metric | Alpha | AST nodes | EML nodes | EML depth | Expression |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 8268 | 18.733333333333334 | 18.733333333333334 | 15 | 281 | 26 | `((1*x)*(y*y))*((x*y)*(y*y))` |
| 2 | 6515 | 18.272727272727273 | 18.272727272727273 | 11 | 201 | 26 | `((1*x)*y)*(y*(y*1))` |
| 3 | 4512 | 17.88888888888889 | 17.88888888888889 | 9 | 161 | 26 | `(x*(1*y))*(x*y)` |
| 4 | 5487 | 17.88888888888889 | 17.88888888888889 | 9 | 161 | 34 | `y*(x*(1*(x*y)))` |
| 5 | 7681 | 17.88888888888889 | 17.88888888888889 | 9 | 161 | 26 | `x*((1*1)*(x*x))` |

## Highest-Depth Examples

| Rank | Index | Metric | Alpha | AST nodes | EML nodes | EML depth | Expression |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 109 | 34 | 16.083333333333332 | 12 | 193 | 34 | `(((x*y)*(y + y))*1)*log(y)` |
| 2 | 151 | 34 | 15.842105263157896 | 19 | 301 | 34 | `((y*(x*y))*((y + 1)*exp(1)))*log(y*(x + y))` |
| 3 | 196 | 34 | 13.785714285714286 | 14 | 193 | 34 | `x*(((y*y)*exp(y))*(exp(y) + exp(1)))` |
| 4 | 243 | 34 | 15.352941176470589 | 17 | 261 | 34 | `(((x*x)*(y + 1))*exp(y + 1))*log(x*y)` |
| 5 | 251 | 34 | 16.166666666666668 | 18 | 291 | 34 | `(((1*x)*(y*y))*exp(x*y))*exp(y*log(y))` |

## Worst Operator Signatures

| Signature | Median alpha | P90 alpha | Count | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `Mul` | 17.285714285714285 | 17.88888888888889 | 60 | 0.0 |
| `Add+Mul` | 14.777777777777779 | 16.333333333333332 | 582 | 0.0 |
| `Mul+log` | 14.5 | 16.384615384615383 | 185 | 0.0 |
| `Mul+exp` | 13.833333333333334 | 15.375 | 307 | 0.0 |
| `Add+Mul+log` | 13.75 | 15.3 | 1675 | 0.0 |

## Depth Failure Modes

| AST depth | Mean alpha | P90 alpha | Mean EML/AST nodes | Count |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 12.26481908058203 | 15.285714285714286 | 12.608096633706758 | 3919 |
| 2 | 12.055288166768854 | 15.285714285714286 | 12.400298173686172 | 466 |
| 4 | 12.275239240317244 | 14.894736842105264 | 12.674345378460533 | 5594 |
| 1 | 8.38888888888889 | 13.666666666666666 | 9.125 | 18 |
| 0 | 1.0 | 1.0 | 1.0 | 3 |

## Common Structural Causes

- Add/Mul-heavy signatures dominate the worst median-alpha groups: `2` of the top `5` worst signatures contain both.
- `log` appears in `2` of the top `5` worst signatures.
- `exp` appears in `1` of the top `5` worst signatures.
- The strongest dominant operator family by median alpha is `Mul` with median alpha `14.454545454545455`.
- Repeated Add/Mul macro expansion is the main structural source of large pure EML trees; log and exp wrappers add depth and amplify nested products/sums.

## Safe Raw EML Regime

Some signatures have expressions below the current threshold.
| Signature | Median alpha | Median threshold gap | P90 alpha | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `leaf_only` | 1.0 | -0.5578858913022597 | 1.0 | 100.0 |
| `exp` | 1.7083333333333335 | 0.15044744203107374 | 1.8 | 25.0 |
| `exp+log` | 2.75 | 1.1921141086977403 | 3.4 | 0.0 |
| `log` | 3.5 | 1.9421141086977403 | 3.5 | 0.0 |
| `Add+exp+log` | 8.636363636363637 | 7.078477745061377 | 9.916666666666666 | 0.0 |

## Output Files

- `outputs/v1/top_alpha_explosions.csv`
- `outputs/v1/top_eml_node_explosions.csv`
- `outputs/v1/top_eml_depth_explosions.csv`
- `outputs/v1/worst_operator_signatures.csv`
- `outputs/v1/safest_operator_signatures.csv`
- `outputs/v1/depth_failure_modes.csv`
- `outputs/v1/safe_eml_regime_candidates.csv`
