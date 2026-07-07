# Goal 3 DAG Compression Findings

This report mines exact structural DAG compression results from saved Goal 3 outputs.
Compression success is ranked by high EML DAG compression and a large drop from `tree_alpha` to `dag_alpha_vs_ast_tree`.
Compression failure is ranked by weak EML DAG compression or DAG alpha values that remain high after sharing.

Important: these are structural representation findings only. They are not model-performance evidence.

## Threshold Context

For `current_grammar`, percent below threshold was `0.06` for tree alpha, `0.22` for DAG alpha vs AST tree, and `0.22` for DAG alpha vs AST DAG.

## Top DAG Compression Successes

| Rank | Index | Score | Tree alpha | DAG alpha | EML DAG compression | Signature | Expression |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2957 | 64.7146 | 15.7727 | 3.09091 | 5.10294 | `Add+Mul+exp` | `((y*1)*(x*y) + exp(y*y)) + ((x*y + y*y) + 1)` |
| 2 | 1267 | 60.3961 | 14.3043 | 2.73913 | 5.22222 | `Add+Mul+log` | `((x*x + 1*x) + (x*x + (y + 1))) + (log(y) + log(x*1))` |
| 3 | 7960 | 60.2129 | 15.3 | 3.1 | 4.93548 | `Add+Mul+log` | `(1*y)*(1*y + log(y))` |
| 4 | 840 | 60.1799 | 17.2857 | 3.85714 | 4.48148 | `Mul` | `(1*x)*(x*1)` |
| 5 | 3858 | 60.1799 | 17.2857 | 3.85714 | 4.48148 | `Mul` | `(y*1)*(y*1)` |

## Top DAG Compression Failures

| Rank | Index | Score | Tree alpha | DAG alpha | EML DAG compression | Signature | Expression |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 3 | 15.8333 | 13.6667 | 6.33333 | 2.15789 | `Mul` | `x*x` |
| 2 | 275 | 15.8333 | 13.6667 | 6.33333 | 2.15789 | `Mul` | `y*y` |
| 3 | 47 | 15.5507 | 13.6667 | 7.66667 | 1.78261 | `Mul` | `x*y` |
| 4 | 6023 | 15.4286 | 17.2857 | 6.42857 | 2.68889 | `Mul` | `(x*x)*(y*y)` |
| 5 | 9 | 15.3 | 16.2 | 6.8 | 2.38235 | `Mul` | `x*(y*y)` |

## Best Operator Signatures

| Signature | Count | Median DAG alpha | Median compression | Median improvement | Percent below DAG threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Add+Mul` | 582 | 4.428571428571429 | 3.326923076923077 | 3.3269230769230766 | 0.0 |
| `Add+Mul+log` | 1675 | 4.181818181818182 | 3.2285714285714286 | 3.2285714285714286 | 0.0 |
| `Add` | 69 | 3.6 | 3.16 | 3.16 | 0.0 |
| `Add+Mul+exp` | 1910 | 4.076923076923077 | 3.13953488372093 | 3.1395348837209305 | 0.0 |
| `Add+Mul+exp+log` | 3449 | 3.92 | 3.097560975609756 | 3.097560975609756 | 0.0 |

## Worst Operator Signatures

| Signature | Count | Median DAG alpha | Median compression | Median improvement | Percent below DAG threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Mul` | 60 | 5.428571428571429 | 3.0 | 2.9999999999999996 | 0.0 |
| `Mul+log` | 185 | 5.090909090909091 | 2.9 | 2.9 | 0.0 |
| `Add+Mul` | 582 | 4.428571428571429 | 3.326923076923077 | 3.3269230769230766 | 0.0 |
| `Mul+exp` | 307 | 4.714285714285714 | 2.6595744680851063 | 2.6595744680851063 | 0.0 |
| `Mul+exp+log` | 563 | 4.636363636363637 | 2.5428571428571427 | 2.5428571428571427 | 0.0 |

## Candidate Safe Regimes

Some operator signatures cross the current threshold after exact DAG sharing.
| Signature | Count | Percent below DAG threshold | Median DAG alpha | Median compression |
| --- | ---: | ---: | ---: | ---: |
| `leaf_only` | 3 | 100.0 | 1.0 | 1.0 |
| `exp` | 12 | 100.0 | 1.225 | 1.45 |
| `exp+log` | 29 | 24.137931034482758 | 1.6666666666666667 | 1.625 |
| `log` | 3 | 0.0 | 2.5 | 1.4 |
| `Add+exp+log` | 643 | 0.0 | 3.1818181818181817 | 2.625 |

## Interpretation

- DAG compression helps most where the source tree contains repeated exact structural subtrees.
- DAG compression does not rewrite algebra, commute arguments, or create hidden macro nodes; high remaining DAG alpha is therefore evidence of real structural cost in the official pure EML representation.
- If DAG helps but alpha remains above threshold, the correct conclusion is that sharing reduced tree redundancy but did not by itself make the representation compact enough under that threshold.

## Output Files

- `outputs/v1/top_dag_compression_successes.csv`
- `outputs/v1/top_dag_compression_failures.csv`
- `outputs/v1/best_dag_operator_signatures.csv`
- `outputs/v1/worst_dag_operator_signatures.csv`
- `outputs/v1/dag_safe_regime_candidates.csv`
