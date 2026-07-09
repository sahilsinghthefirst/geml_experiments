# Goal 5 Summary

Goal 5 implemented ML-facing compression before final GNN training. Macro graphs are validated and useful; frequent motif compression is the strongest simple compression result; learned motif selection does not beat frequent/random baselines at the median; and the neural e-graph ranker mainly provides speed/ranking utility, not major compression.

## Headline Results

- Macro graph median alpha: 0.778
- Macro graph median gain vs Goal 3: 5.250
- Frequent motif median gain vs Goal 3: 7.400
- Learned motif median gain vs Goal 3: 7.111
- Learned vs random motif median compression: 1.000
- Learned vs random motif mean compression: 1.004
- Neural e-graph median regret: 0.000
- Neural e-graph median speedup: 109.305x
- Hierarchical export validation rate: 100.000%
- Reconstruction failure count: 0

## Learned And Neural Baseline Check

| Metric | Value |
| --- | --- |
| learned vs frequent motif median | 1.000 |
| learned vs random motif median | 1.000 |
| learned vs random motif mean | 1.004 |
| neural exact-match rate | 64.236 |
| estimated heuristic exact-match rate | 80.504 |
| AST baseline exact-match rate | 78.321 |
| neural mean regret | 0.506 |
| heuristic mean regret | 0.597 |
| AST mean regret | 0.698 |

The learned motif gain vs Goal 3 is mostly due to motif compression itself, not learned selection.

Learned motif candidate discovery now uses the train-only motif vocabulary; test rows used for candidate discovery: False.

The neural extractor’s 109x speedup is scoped to candidate cost scoring only.

## Denominator Audit

After-threshold e-graph rates report both `success_only_after_rate` and `all_processed_after_rate`; failed or timeout rows count as not below threshold in the all-processed denominator.

| Mode | Processed | Success | Timeout | Validation failed | Extraction failed | Official compile failed | Before rate | Success-only after rate | All-processed after rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| safe | 10,000 | 9,316 | 241 | 471 | 0 | 0 | 0.220 | 1.020 | 0.950 |
| positive_real_formal | 10,000 | 8,941 | 522 | 583 | 15 | 0 | 0.220 | 5.827 | 5.210 |

## Nontrivial v1

- Macro median gain: 5.375
- Frequent motif median gain: 7.750
- Learned motif median gain: 7.400
- Neural e-graph median gain: 1.000

## Recommendation

For Goal 6, start with `macro_graph`, `learned_motif_graph`, `frequent_motif_graph`, and `pure_eml_dag_graph` controls. Keep Goal 4 e-graph outputs, learned motif selection, and the neural extractor as baselines. Treat `hierarchical_eml_graph` as the audit-rich export for later multi-level modeling.

Goal 5 makes graph sizes more practical for future ML work, but it does not claim symbolic reasoning performance.
