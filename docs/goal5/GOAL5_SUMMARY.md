# Goal 5 Summary

Goal 5 implemented ML-facing compression before final GNN training. It created macro graphs, frequent motif graphs, learned motif graphs, a neural e-graph cost model, and hierarchical graph exports.

## Headline Results

- Macro graph median alpha: 0.778
- Macro graph median gain vs Goal 3: 5.250
- Frequent motif median gain vs Goal 3: 7.400
- Learned motif median gain vs Goal 3: 7.125
- Learned vs random motif median compression: 1.000
- Neural e-graph median regret: 0.000
- Neural e-graph median speedup: 109.305x
- Hierarchical export validation rate: 100.000%
- Reconstruction failure count: 0

## Nontrivial v1

- Macro median gain: 5.375
- Frequent motif median gain: 7.750
- Learned motif median gain: 7.429
- Neural e-graph median gain: 1.000

## Recommendation

For Goal 6, start with `macro_graph`, `learned_motif_graph`, `frequent_motif_graph`, and `pure_eml_dag_graph` controls. Keep Goal 4 e-graph outputs and the neural extractor as compression/ranking baselines. Treat `hierarchical_eml_graph` as the audit-rich export for later multi-level modeling.

Goal 5 makes graph sizes more practical for future ML work, but it does not claim symbolic reasoning performance.
