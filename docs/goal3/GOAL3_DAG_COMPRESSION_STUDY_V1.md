# Goal 3 DAG Compression Study

## Goal 3 Question

Goal 3 asks whether exact structural DAG compression can make official pure EML structurally competitive with AST representations after Goal 2 showed raw EML trees are representation-complete but expensive.

The comparison is structural only. It does not make a neural model performance claim.

## Relation To Goal 2

Goal 2 measured raw official pure EML trees on the fixed-seed 10k expression distribution. It found every expression could compile to pure EML, but tree alpha was far above threshold. Goal 3 keeps the same distribution first so the DAG result is directly comparable.

- processed expressions: `10000`
- supported expressions: `10000`
- unsupported expressions: `0`

## Exact Structural DAG Definition

A Goal 3 DAG node represents one unique structural subtree. Two tree subtrees may share a DAG node only when their full canonical structural signatures are identical.

Allowed sharing:

- identical leaf signatures: kind plus label/value
- identical unary signatures: kind, label, and child signature
- identical binary signatures: kind, label, ordered left/right child signatures
- repeated child references such as `EML(a, a)`, with both references kept explicit

Forbidden sharing:

- derived leaves
- hidden compound-expression leaves
- macro or template nodes
- parameterized macro sharing
- algebraic simplification for compression
- pattern sharing with holes such as `EML(1, z)`
- treating `x + y` and `y + x` as identical unless upstream AST normalization already made them structurally identical
- treating `x*x` and `x**2` as identical unless the source converter represents them identically

## AST DAG vs EML DAG

AST DAG compression shares repeated source AST subtrees. EML DAG compression shares repeated official pure EML subtrees after macro expansion. EML has more repeated structural material, especially the constant `1` and repeated macro-expansion subtrees, so EML DAG compression is real. The key question is whether that sharing is enough to cross alpha thresholds.

## Aggregate Metrics

| Metric | Definition | Mean | Median | P90 | P95 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tree_alpha` | T_EML nodes / T_AST nodes | 12.250527857284991 | 12.454545454545455 | 15.0 | 15.642857142857142 | 18.733333333333334 |
| `dag_alpha_vs_ast_tree` | D_EML nodes / T_AST nodes | 4.036107723777943 | 4.0 | 4.9 | 5.166666666666667 | 7.666666666666667 |
| `dag_alpha_vs_ast_dag` | D_EML nodes / D_AST nodes | 5.24252227317617 | 5.25 | 6.5 | 6.857142857142857 | 10.0 |
| `eml_dag_compression` | T_EML nodes / D_EML nodes | 3.0608322420245515 | 3.0634920634920637 | 3.72 | 3.9358974358974357 | 5.382352941176471 |

Tree alpha falls substantially after EML DAG sharing, but the DAG alpha is still measured both against the AST tree and AST DAG baselines:

- `dag_alpha_vs_ast_tree = D_EML_nodes / T_AST_nodes`
- `dag_alpha_vs_ast_dag = D_EML_nodes / D_AST_nodes`

## Threshold Scenarios

| Scenario | K | L | Threshold | Tree below | DAG vs AST tree below | DAG vs AST DAG below |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_grammar` | 4 | 3 | 1.5578858913022597 | 0.06 | 0.22 | 0.22 |
| `generous_operator_vocab` | 20 | 3 | 2.2055713536802566 | 0.15 | 0.51 | 0.43 |
| `larger_operator_vocab` | 50 | 3 | 2.574313870407124 | 0.15 | 1.04 | 0.54 |

Current-grammar threshold result:

- tree alpha below threshold: `0.06%`
- DAG alpha vs AST tree below threshold: `0.22%`
- DAG alpha vs AST DAG below threshold: `0.22%`

## Stratified Findings

- strongest median EML DAG compression family: `Add` with median compression `3.2857142857142856`
- highest median DAG-alpha family: `Mul` with median DAG alpha `4.571428571428571`
- top compression-success signature: `Add+Mul`
- top compression-failure signature: `Mul`

AST-size bucket summary:

| AST node bucket | Count | Median tree alpha | Median DAG alpha | Median compression |
| --- | ---: | ---: | ---: | ---: |
| `1-3` | 30 | 3.25 | 2.0 | 1.6770833333333335 |
| `4-7` | 1956 | 10.714285714285714 | 4.142857142857143 | 2.5172413793103448 |
| `8-15` | 6482 | 12.538461538461538 | 4.071428571428571 | 3.0727272727272728 |
| `16-31` | 1532 | 13.61111111111111 | 3.764705882352941 | 3.5796971670465645 |

Operator-family summary:

| Family | Count | Median DAG alpha | Median compression | Median improvement | Below threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Mul` | 2741 | 4.571428571428571 | 3.1785714285714284 | 3.1785714285714284 | 0.0 |
| `mixed_Mul+exp+log` | 90 | 4.55050505050505 | 2.418382352941176 | 2.4183823529411765 | 0.0 |
| `mixed_Mul+log` | 205 | 4.5 | 2.816666666666667 | 2.816666666666667 | 0.0 |
| `mixed_Mul+exp` | 342 | 4.333333333333333 | 2.7857142857142856 | 2.7857142857142856 | 0.0 |
| `mixed_Add+Mul+log` | 212 | 4.2727272727272725 | 2.92 | 2.9199999999999995 | 0.0 |
| `mixed_Add+Mul` | 807 | 4.2 | 3.2790697674418605 | 3.2790697674418605 | 0.0 |
| `mixed_Add+Mul+exp+log` | 224 | 4.142857142857143 | 2.586206896551724 | 2.5862068965517238 | 0.0 |
| `mixed_Add+Mul+exp` | 280 | 4.1042780748663095 | 2.9148936170212765 | 2.914893617021277 | 0.0 |
| `log` | 253 | 3.8333333333333335 | 2.5483870967741935 | 2.5483870967741935 | 0.0 |
| `mixed_Add+log` | 220 | 3.7 | 2.8378378378378377 | 2.837837837837838 | 0.0 |
| `Add` | 3193 | 3.6666666666666665 | 3.2857142857142856 | 3.2857142857142856 | 0.0 |
| `exp` | 911 | 3.6363636363636362 | 2.4482758620689653 | 2.4482758620689653 | 2.0856201975850714 |
| `mixed_Add+exp+log` | 85 | 3.6 | 2.1875 | 2.1875 | 0.0 |
| `mixed_exp+log` | 58 | 3.5555555555555554 | 2.2432432432432434 | 2.2432432432432434 | 0.0 |
| `mixed_Add+exp` | 376 | 3.4444444444444446 | 2.7777777777777777 | 2.7777777777777777 | 0.0 |
| `leaf_only` | 3 | 1.0 | 1.0 | 1.0 | 100.0 |

Selected boolean-feature summary:

| Feature | Count | Median DAG alpha | Median compression | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `contains_Add` | 8838 | 3.933333333333333 | 3.108108108108108 | 0.0 |
| `contains_Mul` | 8731 | 4.111111111111111 | 3.1052631578947367 | 0.0 |
| `contains_log` | 6767 | 4.0 | 3.046511627906977 | 0.10344318013890941 |
| `contains_exp` | 7203 | 3.9285714285714284 | 3.0 | 0.2637789809801472 |

## Plots

- `outputs/v1/plots_goal3/tree_alpha_vs_dag_alpha.png`
- `outputs/v1/plots_goal3/eml_tree_nodes_vs_eml_dag_nodes.png`
- `outputs/v1/plots_goal3/eml_dag_compression_histogram.png`
- `outputs/v1/plots_goal3/dag_alpha_vs_ast_tree_histogram.png`
- `outputs/v1/plots_goal3/dag_alpha_vs_ast_dag_histogram.png`
- `outputs/v1/plots_goal3/median_dag_alpha_by_operator_family.png`
- `outputs/v1/plots_goal3/median_eml_dag_compression_by_operator_family.png`
- `outputs/v1/plots_goal3/percent_below_threshold_tree_vs_dag.png`
- `outputs/v1/plots_goal3/dag_improvement_by_ast_size_bucket.png`

## Success And Failure Cases

Top compression successes are ranked by high EML DAG compression and large drop from tree alpha to DAG alpha. Top failures are ranked by weak compression or high remaining DAG alpha.

Top success examples:

| Rank | Index | Score | Tree alpha | DAG alpha | EML DAG compression | Signature | Expression |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2957 | 64.71457219251337 | 15.772727272727273 | 3.090909090909091 | 5.102941176470588 | `Add+Mul+exp` | `((y*1)*(x*y) + exp(y*y)) + ((x*y + y*y) + 1)` |
| 2 | 1267 | 60.39613526570049 | 14.304347826086957 | 2.739130434782609 | 5.222222222222222 | `Add+Mul+log` | `((x*x + 1*x) + (x*x + (y + 1))) + (log(y) + log(x*1))` |
| 3 | 7960 | 60.21290322580646 | 15.3 | 3.1 | 4.935483870967742 | `Add+Mul+log` | `(1*y)*(1*y + log(y))` |
| 4 | 840 | 60.17989417989418 | 17.285714285714285 | 3.857142857142857 | 4.481481481481482 | `Mul` | `(1*x)*(x*1)` |
| 5 | 3858 | 60.17989417989418 | 17.285714285714285 | 3.857142857142857 | 4.481481481481482 | `Mul` | `(y*1)*(y*1)` |

Top failure examples:

| Rank | Index | Score | Tree alpha | DAG alpha | EML DAG compression | Signature | Expression |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 3 | 15.833333333333332 | 13.666666666666666 | 6.333333333333333 | 2.1578947368421053 | `Mul` | `x*x` |
| 2 | 275 | 15.833333333333332 | 13.666666666666666 | 6.333333333333333 | 2.1578947368421053 | `Mul` | `y*y` |
| 3 | 47 | 15.55072463768116 | 13.666666666666666 | 7.666666666666667 | 1.7826086956521738 | `Mul` | `x*y` |
| 4 | 6023 | 15.428571428571429 | 17.285714285714285 | 6.428571428571429 | 2.688888888888889 | `Mul` | `(x*x)*(y*y)` |
| 5 | 9 | 15.3 | 16.2 | 6.8 | 2.3823529411764706 | `Mul` | `x*(y*y)` |

Best operator-signature compression groups:

| Signature | Count | Median DAG alpha | Median compression | Median improvement | Below threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Add+Mul` | 582 | 4.428571428571429 | 3.326923076923077 | 3.3269230769230766 | 0.0 |
| `Add+Mul+log` | 1675 | 4.181818181818182 | 3.2285714285714286 | 3.2285714285714286 | 0.0 |
| `Add` | 69 | 3.6 | 3.16 | 3.16 | 0.0 |
| `Add+Mul+exp` | 1910 | 4.076923076923077 | 3.13953488372093 | 3.1395348837209305 | 0.0 |
| `Add+Mul+exp+log` | 3449 | 3.92 | 3.097560975609756 | 3.097560975609756 | 0.0 |

Worst remaining DAG-alpha groups:

| Signature | Count | Median DAG alpha | Median compression | Median improvement | Below threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Mul` | 60 | 5.428571428571429 | 3.0 | 2.9999999999999996 | 0.0 |
| `Mul+log` | 185 | 5.090909090909091 | 2.9 | 2.9 | 0.0 |
| `Add+Mul` | 582 | 4.428571428571429 | 3.326923076923077 | 3.3269230769230766 | 0.0 |
| `Mul+exp` | 307 | 4.714285714285714 | 2.6595744680851063 | 2.6595744680851063 | 0.0 |
| `Mul+exp+log` | 563 | 4.636363636363637 | 2.5428571428571427 | 2.5428571428571427 | 0.0 |

Candidate safe regimes:

| Signature | Count | Percent below threshold | Median DAG alpha | Median compression |
| --- | ---: | ---: | ---: | ---: |
| `leaf_only` | 3 | 100.0 | 1.0 | 1.0 |
| `exp` | 12 | 100.0 | 1.225 | 1.45 |
| `exp+log` | 29 | 24.137931034482758 | 1.6666666666666667 | 1.625 |
| `log` | 3 | 0.0 | 2.5 | 1.4 |
| `Add+exp+log` | 643 | 0.0 | 3.1818181818181817 | 2.625 |

## Semantic Audit Results

- audit expressions: `12`
- structurally valid EML DAGs: `12`
- numerically valid EML DAGs: `12`
- audit JSON: `outputs/v1/goal3_dag_semantic_audit.json`
- audit CSV: `outputs/v1/goal3_dag_semantic_audit.csv`
- audit docs: `docs/goal3/GOAL3_DAG_SEMANTIC_AUDIT_V1.md`

The audit verifies no derived leaves, hidden compound leaves, macro/template nodes, unsupported final EML labels, invalid child slots, or collapsed duplicate child references. It also compares original SymPy, EML tree, and EML DAG numeric values on safe positive real inputs.

## Conclusion

Exact structural DAG sharing reduces the median alpha from `12.454545454545455` to `4.0` versus AST tree size, with median EML DAG compression `3.0634920634920637`. The current-threshold pass rate improves from `0.06%` before DAG sharing to `0.22%` after DAG sharing. This helps materially, but it does not broadly rescue raw official pure EML under the current structural threshold.

DAG compression helps, but under the current fixed-seed distribution it does not rescue raw official pure EML structurally as a general representation. EML DAGs are much smaller than raw EML trees, yet the median DAG alpha remains above the current threshold and only a small slice of operator families crosses it.

## Recommendation For Goal 4

Goal 4 should move from size-only analysis to fair graph-representation baselines: AST-tree/AST-DAG baselines, EML-DAG baselines, and eventually AST-GNN versus EML-DAG-GNN comparisons. The EML side must keep the Goal 3 contract: exact structural DAG sharing only, no hidden macro nodes, no derived leaves, and no algebraic simplification used as compression.

Do not introduce equivalence-pair generation or neural models until the graph baseline task is explicitly scoped.

## Reproducible Commands

```bash
.venv/bin/python -m geml.experiments.run_goal3_dag_pipeline --config configs/dag_compression_v0.yaml
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format . --check
```
