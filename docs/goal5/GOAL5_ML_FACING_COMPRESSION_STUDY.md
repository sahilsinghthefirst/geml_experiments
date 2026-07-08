# Goal 5 ML-Facing Compression Study

Goal 5 asks whether transparent ML-facing compressed graph representations can reduce graph size enough to make later GNN training practical while preserving expandability back to official pure EML.

Goal 5 does not train final symbolic-reasoning GNNs and does not claim downstream reasoning improvement.

## Relation to Previous Goals

- Goal 2 showed that raw official pure EML is valid but structurally expensive.
- Goal 3 added exact structural EML-DAG sharing, which helped but did not rescue EML.
- Goal 3R repaired the result-bearing v1 corpus and made `outputs/v1` the baseline.
- Goal 4 added non-ML e-graph compression with separately labeled `safe` and `positive_real_formal` modes.
- Goal 5 adds ML-facing compression layers before any final GNN training.

## Integrity Boundary

- Macro, motif, and learned motif nodes are not pure EML nodes.
- Every compressed node must have an expansion path back to official pure EML.
- Compressed graph metrics are reported separately from pure EML-DAG metrics.
- Safe and positive-real e-graph modes remain separately labeled.
- Reconstruction and validation failures are reported rather than dropped.

Integrated reconstruction failure count: 0.

## Comparison Table

| Mode | Processed | Success | Median nodes | Median gain vs Goal 3 | Nontrivial gain | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Goal 3 pure EML-DAG | 10,000 | 10,000 | 42.000 | 1.000 |  | Official pure EML-DAG baseline from Goal 3. |
| Goal 4 safe e-graph optimized EML-DAG | 10,000 | 9,316 | 38.000 | 1.045 | 1.000 | Non-ML e-graph extraction with safe rules. |
| Goal 4 positive-real optimized EML-DAG | 10,000 | 8,941 | 34.000 | 1.169 | 1.000 | Non-ML e-graph extraction with positive-real assumptions. |
| Goal 5 macro graph | 10,000 | 10,000 | 8.000 | 5.250 | 5.375 | Transparent compiler macro nodes; not pure EML alpha. |
| Goal 5 frequent motif graph | 10,000 | 10,000 | 6.000 | 7.400 | 7.750 | Greedy motif replacement from mined frequent motifs. |
| Goal 5 learned motif graph | 10,000 | 10,000 | 6.000 | 7.125 | 7.429 | Deterministic learned motif selection; exact reconstruction required. |
| Goal 5 neural e-graph extractor | 20,000 | 18,871 | 37.000 | 1.074 | 1.000 | Learned ranking model; output still compiled to official pure EML-DAG. |
| Goal 5 hierarchical graph | 88,257 | 88,257 | 73.000 |  |  | Audit/export container spanning AST, macro, EML-DAG, and motif levels. |

## Macro Graph Results

The macro graph baseline processed 10,000 expressions. Median macro graph alpha was 0.778, and median compression gain vs Goal 3 EML-DAG was 5.250.

On `nontrivial_v1`, the median macro gain was 5.375. Expansion validation failures: 0.

Interpretation: macro graphs are the cleanest transparent abstraction because each macro is an official compiler concept with a known expansion. They are compressed graph features, not pure EML alpha measurements.

## Frequent Motif Results

The frequent motif baseline selected a vocabulary of 70 motifs. Median compression gain vs Goal 3 was 7.400, with median motif coverage 57.143%.

On `nontrivial_v1`, the median frequent motif gain was 7.750. Expansion validation failures: 0.

Top motifs by support:

| Motif | Type | Nodes | Support | Covered nodes | Macro |
| --- | --- | --- | --- | --- | --- |
| pure_eml_dag_0009 | pure_eml_dag | 1 | 421,977 | 421,977 |  |
| pure_eml_dag_0000 | pure_eml_dag | 2 | 250,481 | 500,962 |  |
| pure_eml_dag_0001 | pure_eml_dag | 2 | 216,562 | 433,124 |  |
| pure_eml_dag_0002 | pure_eml_dag | 2 | 180,024 | 360,048 | eml_exp |
| pure_eml_dag_0003 | pure_eml_dag | 2 | 154,675 | 309,350 | eml_exp |

Top motifs by compression saved:

| Motif | Type | Nodes | Support | Covered nodes | Macro |
| --- | --- | --- | --- | --- | --- |
| pure_eml_dag_0000 | pure_eml_dag | 2 | 250,481 | 500,962 |  |
| pure_eml_dag_0001 | pure_eml_dag | 2 | 216,562 | 433,124 |  |
| pure_eml_dag_0002 | pure_eml_dag | 2 | 180,024 | 360,048 | eml_exp |
| pure_eml_dag_0003 | pure_eml_dag | 2 | 154,675 | 309,350 | eml_exp |
| pure_eml_dag_0004 | pure_eml_dag | 2 | 9,967 | 19,934 | eml_exp |

## Learned Motif Results

The learned motif selector chose 30 motifs and used a random baseline of 30 motifs. Median learned gain vs Goal 3 was 7.125. Median learned-vs-frequent compression was 1.000, and median learned-vs-random compression was 1.000.

The random vocabulary median gain vs Goal 3 was 7.250. In this v1 run the learned selector preserved exact reconstruction but did not clearly beat the random baseline at the median.

Train/validation/test results:

| Split | Processed | Success | Median gain | Failures |
| --- | --- | --- | --- | --- |
| train | 7,021 | 7,021 | 7.143 | 0 |
| validation | 1,491 | 1,491 | 7.000 | 0 |
| test | 1,488 | 1,488 | 7.000 | 0 |

On `nontrivial_v1`, the learned motif median gain was 7.429. Reconstruction failures: 0.

## Neural E-Graph Extractor Results

The neural e-graph extractor evaluated 20,000 expression/rule mode groups. Median regret vs exact best was 0.000, p90 regret was 3.000, and exact-best match rate was 64.236%.

Median neural compression gain vs Goal 3 was 1.074. Median speedup vs exact beam cost scoring was 109.305x. Validation failures: 1,129.

On `nontrivial_v1`, median neural gain was 1.000 and exact-best match rate was 62.590%.

Interpretation: the neural model is a learned ranking/cost tool. It does not define mathematical truth, and selected candidates still compile through the official EML compiler.

## Hierarchical Graph Export

The hierarchical export wrote 88,257 graph records across these modes: ast_dag_graph, ast_tree_graph, egraph_positive_real_eml_dag_graph, egraph_safe_eml_dag_graph, frequent_motif_graph, hierarchical_eml_graph, learned_motif_graph, macro_graph, pure_eml_dag_graph.

Expansion validation rate was 100.000%, reconstruction validation rate was 100.000%, and missing expansion count was 0.

Node/edge statistics by mode:

| Mode | Graphs | Median nodes | Median edges | Reconstruction % |
| --- | --- | --- | --- | --- |
| ast_dag_graph | 10,000 | 8.000 | 9.000 | 100.000 |
| ast_tree_graph | 10,000 | 10.000 | 9.000 | 100.000 |
| egraph_positive_real_eml_dag_graph | 8,941 | 35.000 | 64.000 | 100.000 |
| egraph_safe_eml_dag_graph | 9,316 | 38.000 | 70.000 | 100.000 |
| frequent_motif_graph | 10,000 | 6.000 | 9.000 | 100.000 |
| hierarchical_eml_graph | 10,000 | 73.000 | 128.000 | 100.000 |
| learned_motif_graph | 10,000 | 6.000 | 9.000 | 100.000 |
| macro_graph | 10,000 | 8.000 | 17.000 | 100.000 |
| pure_eml_dag_graph | 10,000 | 42.000 | 79.000 | 100.000 |

The hierarchical graph is a dataset/export format, not a compression score by itself. It keeps AST, macro, pure EML-DAG, frequent motif, and learned motif levels available for audit and future multi-level modeling.

## Final Recommendation for Goal 6

Train initial Goal 6 graph models on three clearly separated tracks:

1. `macro_graph` as the most transparent official-compiler abstraction.
2. `learned_motif_graph` and `frequent_motif_graph` as compact motif baselines.
3. `pure_eml_dag_graph` as the required official pure EML control.

Use Goal 4 e-graph optimized EML-DAGs as non-ML compression baselines. Use the neural e-graph extractor as a learned extraction/ranking baseline, not as evidence of reasoning performance. Use `hierarchical_eml_graph` after the single-mode baselines are stable, because it is richer but larger.

Do not overclaim: compression makes later GNN training more practical, but it does not prove symbolic reasoning ability.

## Limitations

- Compression does not prove reasoning ability.
- Motif and learned motif nodes are not pure EML.
- Learned compression may overfit observed motifs.
- The v1 grammar is still limited to Add/Mul/exp/log.
- Trig, powers, and broader algebra need later stress tests.
- Positive-real e-graph results depend on explicit assumptions and must stay separate from safe-mode results.

## Reproducibility

```bash
.venv/bin/python -m geml.experiments.run_goal5_compression_pipeline --config configs/goal5_compression_v1.yaml
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format . --check
```
