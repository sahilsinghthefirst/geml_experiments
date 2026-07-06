# Goal 2 Failure Cases

This report mines raw official pure EML expansion failures from saved Goal 2 outputs.

Important: this is structural evidence about tree expansion only. It is not model-performance evidence.

Current alpha threshold used by raw rows: `1.5578858913022597`.

## Highest-Alpha Examples

| Rank | Index | Metric | Alpha | AST nodes | EML nodes | EML depth | Expression |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 8274 | 17.366666666666667 | 17.366666666666667 | 30 | 521 | 30 | `((x*x)*(y*y))*((x*x)*(x + 1)) + ((1*x)*(x*y) + (x*1)*exp(1))` |
| 2 | 5583 | 17.178571428571427 | 17.178571428571427 | 28 | 481 | 34 | `(((x*1)*(y*1))*((x*1)*(x + 1)))*((x*y + (x + y))*exp(y*y))` |
| 3 | 5946 | 17.035714285714285 | 17.035714285714285 | 28 | 477 | 34 | `(((1*x)*log(1))*((y*1)*(x + y)))*(((1*y)*exp(x))*((1*y)*exp(y)))` |
| 4 | 8672 | 16.9 | 16.9 | 30 | 507 | 30 | `((1*1)*(y + 1))*((x*y)*exp(x)) + ((y*1)*(x*y) + (1*x + x*1))` |
| 5 | 1916 | 16.8125 | 16.8125 | 16 | 269 | 27 | `exp(((1*y)*(x + 1))*((x*1)*(y*1)))` |

## Highest-Depth Examples

| Rank | Index | Metric | Alpha | AST nodes | EML nodes | EML depth | Expression |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 6 | 34 | 13.782608695652174 | 23 | 317 | 34 | `(((1*y)*(x*y))*((y + 1) + (1 + 1)))*(exp(exp(y)) + log(exp(x)))` |
| 2 | 29 | 34 | 13.814814814814815 | 27 | 373 | 34 | `(((1*1)*exp(y))*((y + 1) + exp(y)))*((1*x + (y + y)) + (log(1) + log(1)))` |
| 3 | 45 | 34 | 14.058823529411764 | 17 | 239 | 34 | `(((x*x)*(x*y))*log(exp(x)))*exp(exp(y + y))` |
| 4 | 48 | 34 | 13.470588235294118 | 17 | 229 | 34 | `(((x*y)*(1 + 1))*log(exp(x)))*log(exp(1 + 1))` |
| 5 | 49 | 34 | 13.961538461538462 | 26 | 363 | 34 | `(((y*1)*exp(x))*((y + 1)*exp(1)))*(((x + x) + (x + y)) + exp(y + 1))` |

## Worst Operator Signatures

| Signature | Median alpha | P90 alpha | Count | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `Add+Mul` | 16.677419354838708 | 16.677419354838708 | 3 | 0.0 |
| `Add+Mul+log` | 15.157407407407408 | 16.1 | 34 | 0.0 |
| `Add+Mul+exp` | 13.0 | 15.173913043478262 | 887 | 0.0 |
| `Add+Mul+exp+log` | 12.055555555555555 | 14.26923076923077 | 6174 | 0.0 |
| `Mul+exp` | 10.875 | 13.916666666666666 | 58 | 0.0 |

## Depth Failure Modes

| AST depth | Mean alpha | P90 alpha | Mean EML/AST nodes | Count |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 10.648995087903057 | 14.208333333333334 | 11.791297863820269 | 10000 |

## Common Structural Causes

- Add/Mul-heavy signatures dominate the worst median-alpha groups: `4` of the top `5` worst signatures contain both.
- `log` appears in `2` of the top `5` worst signatures.
- `exp` appears in `3` of the top `5` worst signatures.
- The strongest dominant operator family by median alpha is `Mul` with median alpha `13.947368421052632`.
- Repeated Add/Mul macro expansion is the main structural source of large pure EML trees; log and exp wrappers add depth and amplify nested products/sums.

## Safe Raw EML Regime

No robust safe regime appears under the current raw pure EML threshold. The closest signatures remain above threshold on median alpha.
| Signature | Median alpha | Median threshold gap | P90 alpha | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `exp` | 1.8 | 0.2421141086977403 | 1.8 | 0.0 |
| `exp+log` | 3.4 | 1.8421141086977402 | 3.4 | 0.0 |
| `Add+exp+log` | 6.916666666666667 | 5.358780775364407 | 9.583333333333334 | 0.0 |
| `Add+exp` | 7.375 | 5.817114108697741 | 9.266666666666667 | 0.0 |
| `Mul+exp+log` | 9.5 | 7.942114108697741 | 13.615384615384615 | 0.0 |

## Output Files

- `outputs/v0/top_alpha_explosions.csv`
- `outputs/v0/top_eml_node_explosions.csv`
- `outputs/v0/top_eml_depth_explosions.csv`
- `outputs/v0/worst_operator_signatures.csv`
- `outputs/v0/safest_operator_signatures.csv`
- `outputs/v0/depth_failure_modes.csv`
- `outputs/v0/safe_eml_regime_candidates.csv`
