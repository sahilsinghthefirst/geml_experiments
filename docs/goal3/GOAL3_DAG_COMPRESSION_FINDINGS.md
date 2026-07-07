# Goal 3 DAG Compression Findings

This report mines exact structural DAG compression results from saved Goal 3 outputs.
Compression success is ranked by high EML DAG compression and a large drop from `tree_alpha` to `dag_alpha_vs_ast_tree`.
Compression failure is ranked by weak EML DAG compression or DAG alpha values that remain high after sharing.

Important: these are structural representation findings only. They are not model-performance evidence.

## Threshold Context

For `current_grammar`, percent below threshold was `0.0` for tree alpha, `1.06` for DAG alpha vs AST tree, and `1.06` for DAG alpha vs AST DAG.

## Top DAG Compression Successes

| Rank | Index | Score | Tree alpha | DAG alpha | EML DAG compression | Signature | Expression |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2937 | 83.2051 | 13.75 | 1.95 | 7.05128 | `Add+Mul+exp+log` | `((1*y + 1*y) + exp(log(1))) + exp(y*1 + 1*y)` |
| 2 | 8996 | 75.7435 | 14.7917 | 2.41667 | 6.12069 | `Add+Mul+exp+log` | `((x*1 + y*1) + exp(1*y))*((x*1 + y*1) + exp(log(1)))` |
| 3 | 3479 | 73.2828 | 16.1 | 2.9 | 5.55172 | `Add+Mul+log` | `((x*x)*(x*x) + (x*y)*(y + 1)) + ((x*x)*(y + 1) + (y + 1)*log(1))` |
| 4 | 4541 | 71.3045 | 16.6774 | 3.16129 | 5.27551 | `Add+Mul` | `((1*y + (x + 1))*(x*y + x*y))*((x*1 + (1 + 1))*(x*y + 1*x))` |
| 5 | 5715 | 71.2935 | 15.72 | 2.84 | 5.53521 | `Add+Mul+exp+log` | `((y*y)*log(1))*(1*y + y*1) + ((y*y)*log(1))*log(exp(1))` |

## Top DAG Compression Failures

| Rank | Index | Score | Tree alpha | DAG alpha | EML DAG compression | Signature | Expression |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 261 | 12.9167 | 14.1667 | 5.16667 | 2.74194 | `Mul+exp+log` | `(((x*x)*exp(y))*((y*y)*exp(x)))*log(exp(exp(x)))` |
| 2 | 3384 | 12.4444 | 14.3333 | 5.44444 | 2.63265 | `Mul+exp+log` | `log(exp((x*x)*(y*y)))` |
| 3 | 3150 | 12.2768 | 16.3929 | 4.46429 | 3.672 | `Add+Mul+exp` | `(((1*1)*(x*y))*((x*1)*exp(y)))*(((x*y)*exp(1))*((y + 1) + exp(y)))` |
| 4 | 69 | 12.2051 | 15.5714 | 4.66667 | 3.33673 | `Add+Mul+exp+log` | `(((x*x)*(y*y))*((x + 1)*(x + y)))*log(exp(x + y))` |
| 5 | 6054 | 11.9925 | 14.68 | 4.68 | 3.13675 | `Add+Mul+exp` | `((x*x)*(x + y))*((y*y)*(x + 1)) + (exp(x)*exp(y))*exp(exp(y))` |

## Best Operator Signatures

| Signature | Count | Median DAG alpha | Median compression | Median improvement | Percent below DAG threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Add+Mul` | 3 | 3.161290322580645 | 5.275510204081633 | 5.275510204081632 | 0.0 |
| `Add+Mul+log` | 34 | 3.339080459770115 | 4.4415760869565215 | 4.4415760869565215 | 0.0 |
| `Add+Mul+exp` | 887 | 3.7142857142857144 | 3.4941176470588236 | 3.4941176470588236 | 0.0 |
| `Add+Mul+exp+log` | 6174 | 3.625 | 3.305084745762712 | 3.305084745762712 | 0.0 |
| `Mul+exp` | 58 | 4.111111111111111 | 2.8005001389274797 | 2.8005001389274797 | 0.0 |

## Worst Operator Signatures

| Signature | Count | Median DAG alpha | Median compression | Median improvement | Percent below DAG threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Add+Mul` | 3 | 3.161290322580645 | 5.275510204081633 | 5.275510204081632 | 0.0 |
| `Add+Mul+log` | 34 | 3.339080459770115 | 4.4415760869565215 | 4.4415760869565215 | 0.0 |
| `Add+Mul+exp` | 887 | 3.7142857142857144 | 3.4941176470588236 | 3.4941176470588236 | 0.0 |
| `Mul+exp` | 58 | 4.111111111111111 | 2.8005001389274797 | 2.8005001389274797 | 0.0 |
| `Add+Mul+exp+log` | 6174 | 3.625 | 3.305084745762712 | 3.305084745762712 | 0.0 |

## Candidate Safe Regimes

Some operator signatures cross the current threshold after exact DAG sharing.
| Signature | Count | Percent below DAG threshold | Median DAG alpha | Median compression |
| --- | ---: | ---: | ---: | ---: |
| `exp` | 25 | 100.0 | 1.2 | 1.5 |
| `exp+log` | 828 | 9.782608695652174 | 1.8 | 1.8571428571428572 |
| `Add+exp` | 87 | 0.0 | 2.857142857142857 | 2.4583333333333335 |
| `Add+exp+log` | 953 | 0.0 | 3.0 | 2.466666666666667 |
| `Add+Mul` | 3 | 0.0 | 3.161290322580645 | 5.275510204081633 |

## Interpretation

- DAG compression helps most where the source tree contains repeated exact structural subtrees.
- DAG compression does not rewrite algebra, commute arguments, or create hidden macro nodes; high remaining DAG alpha is therefore evidence of real structural cost in the official pure EML representation.
- If DAG helps but alpha remains above threshold, the correct conclusion is that sharing reduced tree redundancy but did not by itself make the representation compact enough under that threshold.

## Output Files

- `outputs/v0/top_dag_compression_successes.csv`
- `outputs/v0/top_dag_compression_failures.csv`
- `outputs/v0/best_dag_operator_signatures.csv`
- `outputs/v0/worst_dag_operator_signatures.csv`
- `outputs/v0/dag_safe_regime_candidates.csv`
- `outputs/v0/plots_goal3/`
