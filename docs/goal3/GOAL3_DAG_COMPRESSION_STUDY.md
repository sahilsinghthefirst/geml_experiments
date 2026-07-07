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
| `tree_alpha` | T_EML nodes / T_AST nodes | 10.648995087903057 | 11.375 | 14.208333333333334 | 14.733333333333333 | 17.366666666666667 |
| `dag_alpha_vs_ast_tree` | D_EML nodes / T_AST nodes | 3.4323505588013528 | 3.5 | 4.25 | 4.444444444444445 | 5.444444444444445 |
| `dag_alpha_vs_ast_dag` | D_EML nodes / D_AST nodes | 4.550634066244213 | 4.714285714285714 | 6.0 | 6.3076923076923075 | 7.875 |
| `eml_dag_compression` | T_EML nodes / D_EML nodes | 3.060560619100973 | 3.1176470588235294 | 3.9146341463414633 | 4.159574468085107 | 7.051282051282051 |

Tree alpha falls substantially after EML DAG sharing, but the DAG alpha is still measured both against the AST tree and AST DAG baselines:

- `dag_alpha_vs_ast_tree = D_EML_nodes / T_AST_nodes`
- `dag_alpha_vs_ast_dag = D_EML_nodes / D_AST_nodes`

## Threshold Scenarios

| Scenario | K | L | Threshold | Tree below | DAG vs AST tree below | DAG vs AST DAG below |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_grammar` | 4 | 3 | 1.5578858913022597 | 0.0 | 1.06 | 1.06 |
| `generous_operator_vocab` | 20 | 3 | 2.2055713536802566 | 0.25 | 9.3 | 8.54 |
| `larger_operator_vocab` | 50 | 3 | 2.574313870407124 | 0.25 | 11.48 | 8.87 |

Current-grammar threshold result:

- tree alpha below threshold: `0.0%`
- DAG alpha vs AST tree below threshold: `1.06%`
- DAG alpha vs AST DAG below threshold: `1.06%`

## Stratified Findings

- strongest median EML DAG compression family: `mixed_Add+Mul` with median compression `3.6451612903225805`
- highest median DAG-alpha family: `mixed_Mul+exp` with median DAG alpha `4.0625`
- top compression-success signature: `Add+Mul`
- top compression-failure signature: `Add+Mul`

AST-size bucket summary:

| AST node bucket | Count | Median tree alpha | Median DAG alpha | Median compression |
| --- | ---: | ---: | ---: | ---: |
| `4-7` | 1590 | 3.4 | 2.0 | 1.8888888888888888 |
| `8-15` | 3985 | 10.333333333333334 | 3.6 | 2.8666666666666667 |
| `16-31` | 4425 | 12.875 | 3.619047619047619 | 3.526315789473684 |

Operator-family summary:

| Family | Count | Median DAG alpha | Median compression | Median improvement | Below threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mixed_Mul+exp` | 436 | 4.0625 | 3.0 | 3.0 | 0.0 |
| `Mul` | 2088 | 4.045454545454546 | 3.445945945945946 | 3.4459459459459456 | 0.0 |
| `mixed_Mul+log` | 129 | 3.875 | 3.064516129032258 | 3.064516129032258 | 0.0 |
| `mixed_Mul+exp+log` | 129 | 3.8666666666666667 | 2.9285714285714284 | 2.9285714285714284 | 0.0 |
| `mixed_Add+Mul+exp` | 166 | 3.797554347826087 | 3.2388059701492535 | 3.2388059701492535 | 0.0 |
| `mixed_Add+Mul` | 373 | 3.7083333333333335 | 3.6451612903225805 | 3.6451612903225805 | 0.0 |
| `mixed_Add+Mul+exp+log` | 60 | 3.6153846153846154 | 3.1702127659574466 | 3.1702127659574466 | 0.0 |
| `mixed_Add+Mul+log` | 38 | 3.582125603864734 | 3.453125 | 3.4531250000000004 | 0.0 |
| `log` | 413 | 3.375 | 3.0 | 3.0 | 0.0 |
| `Add` | 2080 | 3.3636363636363638 | 3.569605943152455 | 3.5696059431524545 | 0.0 |
| `exp` | 2584 | 3.3636363636363638 | 2.4857142857142858 | 2.4857142857142858 | 4.102167182662539 |
| `mixed_Add+exp+log` | 132 | 3.3333333333333335 | 2.9183673469387754 | 2.9183673469387754 | 0.0 |
| `mixed_Add+exp` | 412 | 3.2817460317460316 | 3.1052631578947367 | 3.1052631578947367 | 0.0 |
| `mixed_Add+log` | 121 | 3.2 | 2.9672131147540983 | 2.9672131147540983 | 0.0 |
| `mixed_exp+log` | 839 | 2.0 | 1.8888888888888888 | 1.8888888888888888 | 0.0 |

Selected boolean-feature summary:

| Feature | Count | Median DAG alpha | Median compression | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `contains_Add` | 8138 | 3.5555555555555554 | 3.2580645161290325 | 0.0 |
| `contains_Mul` | 8107 | 3.652173913043478 | 3.2567567567567566 | 0.0 |
| `contains_log` | 8940 | 3.5 | 3.0806451612903225 | 0.9060402684563759 |
| `contains_exp` | 9963 | 3.5 | 3.1132075471698113 | 1.0639365652915789 |

## Plots

- `outputs/v0/plots_goal3/tree_alpha_vs_dag_alpha.png`
- `outputs/v0/plots_goal3/eml_tree_nodes_vs_eml_dag_nodes.png`
- `outputs/v0/plots_goal3/eml_dag_compression_histogram.png`
- `outputs/v0/plots_goal3/dag_alpha_vs_ast_tree_histogram.png`
- `outputs/v0/plots_goal3/dag_alpha_vs_ast_dag_histogram.png`
- `outputs/v0/plots_goal3/median_dag_alpha_by_operator_family.png`
- `outputs/v0/plots_goal3/median_eml_dag_compression_by_operator_family.png`
- `outputs/v0/plots_goal3/percent_below_threshold_tree_vs_dag.png`
- `outputs/v0/plots_goal3/dag_improvement_by_ast_size_bucket.png`

## Success And Failure Cases

Top compression successes are ranked by high EML DAG compression and large drop from tree alpha to DAG alpha. Top failures are ranked by weak compression or high remaining DAG alpha.

Top success examples:

| Rank | Index | Score | Tree alpha | DAG alpha | EML DAG compression | Signature | Expression |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2937 | 83.2051282051282 | 13.75 | 1.95 | 7.051282051282051 | `Add+Mul+exp+log` | `((1*y + 1*y) + exp(log(1))) + exp(y*1 + 1*y)` |
| 2 | 8996 | 75.74353448275862 | 14.791666666666666 | 2.4166666666666665 | 6.120689655172414 | `Add+Mul+exp+log` | `((x*1 + y*1) + exp(1*y))*((x*1 + y*1) + exp(log(1)))` |
| 3 | 3479 | 73.28275862068966 | 16.1 | 2.9 | 5.551724137931035 | `Add+Mul+log` | `((x*x)*(x*x) + (x*y)*(y + 1)) + ((x*x)*(y + 1) + (y + 1)*log(1))` |
| 4 | 4541 | 71.30447662936143 | 16.677419354838708 | 3.161290322580645 | 5.275510204081633 | `Add+Mul` | `((1*y + (x + 1))*(x*y + x*y))*((x*1 + (1 + 1))*(x*y + 1*x))` |
| 5 | 5715 | 71.29352112676057 | 15.72 | 2.84 | 5.535211267605634 | `Add+Mul+exp+log` | `((y*y)*log(1))*(1*y + y*1) + ((y*y)*log(1))*log(exp(1))` |

Top failure examples:

| Rank | Index | Score | Tree alpha | DAG alpha | EML DAG compression | Signature | Expression |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 261 | 12.916666666666668 | 14.166666666666666 | 5.166666666666667 | 2.7419354838709675 | `Mul+exp+log` | `(((x*x)*exp(y))*((y*y)*exp(x)))*log(exp(exp(x)))` |
| 2 | 3384 | 12.444444444444445 | 14.333333333333334 | 5.444444444444445 | 2.63265306122449 | `Mul+exp+log` | `log(exp((x*x)*(y*y)))` |
| 3 | 3150 | 12.276785714285715 | 16.392857142857142 | 4.464285714285714 | 3.672 | `Add+Mul+exp` | `(((1*1)*(x*y))*((x*1)*exp(y)))*(((x*y)*exp(1))*((y + 1) + exp(y)))` |
| 4 | 69 | 12.205128205128204 | 15.571428571428571 | 4.666666666666667 | 3.336734693877551 | `Add+Mul+exp+log` | `(((x*x)*(y*y))*((x + 1)*(x + y)))*log(exp(x + y))` |
| 5 | 6054 | 11.9925 | 14.68 | 4.68 | 3.1367521367521367 | `Add+Mul+exp` | `((x*x)*(x + y))*((y*y)*(x + 1)) + (exp(x)*exp(y))*exp(exp(y))` |

Best operator-signature compression groups:

| Signature | Count | Median DAG alpha | Median compression | Median improvement | Below threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Add+Mul` | 3 | 3.161290322580645 | 5.275510204081633 | 5.275510204081632 | 0.0 |
| `Add+Mul+log` | 34 | 3.339080459770115 | 4.4415760869565215 | 4.4415760869565215 | 0.0 |
| `Add+Mul+exp` | 887 | 3.7142857142857144 | 3.4941176470588236 | 3.4941176470588236 | 0.0 |
| `Add+Mul+exp+log` | 6174 | 3.625 | 3.305084745762712 | 3.305084745762712 | 0.0 |
| `Mul+exp` | 58 | 4.111111111111111 | 2.8005001389274797 | 2.8005001389274797 | 0.0 |

Worst remaining DAG-alpha groups:

| Signature | Count | Median DAG alpha | Median compression | Median improvement | Below threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Add+Mul` | 3 | 3.161290322580645 | 5.275510204081633 | 5.275510204081632 | 0.0 |
| `Add+Mul+log` | 34 | 3.339080459770115 | 4.4415760869565215 | 4.4415760869565215 | 0.0 |
| `Add+Mul+exp` | 887 | 3.7142857142857144 | 3.4941176470588236 | 3.4941176470588236 | 0.0 |
| `Mul+exp` | 58 | 4.111111111111111 | 2.8005001389274797 | 2.8005001389274797 | 0.0 |
| `Add+Mul+exp+log` | 6174 | 3.625 | 3.305084745762712 | 3.305084745762712 | 0.0 |

Candidate safe regimes:

| Signature | Count | Percent below threshold | Median DAG alpha | Median compression |
| --- | ---: | ---: | ---: | ---: |
| `exp` | 25 | 100.0 | 1.2 | 1.5 |
| `exp+log` | 828 | 9.782608695652174 | 1.8 | 1.8571428571428572 |
| `Add+exp` | 87 | 0.0 | 2.857142857142857 | 2.4583333333333335 |
| `Add+exp+log` | 953 | 0.0 | 3.0 | 2.466666666666667 |
| `Add+Mul` | 3 | 0.0 | 3.161290322580645 | 5.275510204081633 |

## Semantic Audit Results

- audit expressions: `12`
- structurally valid EML DAGs: `12`
- numerically valid EML DAGs: `12`
- audit JSON: `outputs/v0/goal3_dag_semantic_audit.json`
- audit CSV: `outputs/v0/goal3_dag_semantic_audit.csv`
- audit docs: `docs/goal3/GOAL3_DAG_SEMANTIC_AUDIT.md`

The audit verifies no derived leaves, hidden compound leaves, macro/template nodes, unsupported final EML labels, invalid child slots, or collapsed duplicate child references. It also compares original SymPy, EML tree, and EML DAG numeric values on safe positive real inputs.

## Conclusion

Exact structural DAG sharing reduces the median alpha from `11.375` to `3.5` versus AST tree size, with median EML DAG compression `3.1176470588235294`. The current-threshold pass rate improves from `0.0%` before DAG sharing to `1.06%` after DAG sharing. This helps materially, but it does not broadly rescue raw official pure EML under the current structural threshold.

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
