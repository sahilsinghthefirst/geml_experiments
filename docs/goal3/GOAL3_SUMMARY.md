# Goal 3 Summary

Goal 3 implemented exact structural DAG compression for AST and official pure EML trees, measured it on the fixed-seed Goal 2 distribution, mined where it helps and fails, and audited that compression does not hide complexity.

## Main Result

- raw tree alpha remains the Goal 2 baseline
- EML DAG compression substantially reduces pure EML size
- current-threshold pass rate improves but remains low
- no derived leaves, hidden compound leaves, macro nodes, or semantic simplification are used

## Headline Numbers

- processed: `10000`
- supported: `10000`
- mean tree alpha: `10.648995087903057`
- median tree alpha: `11.375`
- p90 tree alpha: `14.208333333333334`
- mean DAG alpha vs AST tree: `3.4323505588013528`
- median DAG alpha vs AST tree: `3.5`
- p90 DAG alpha vs AST tree: `4.25`
- mean EML DAG compression: `3.060560619100973`
- median EML DAG compression: `3.1176470588235294`
- p90 EML DAG compression: `3.9146341463414633`
- current threshold below before DAG: `0.0%`
- current threshold below after DAG vs AST tree: `1.06%`

## Best And Worst Families

- top compression-success signature: `Add+Mul`
- top compression-failure signature: `Add+Mul`

## Semantic Audit

- audit expressions: `12`
- structurally valid: `12`
- numerically valid: `12`

## Conclusion

Exact structural DAG sharing reduces the median alpha from `11.375` to `3.5` versus AST tree size, with median EML DAG compression `3.1176470588235294`. The current-threshold pass rate improves from `0.0%` before DAG sharing to `1.06%` after DAG sharing. This helps materially, but it does not broadly rescue raw official pure EML under the current structural threshold.

## Primary Artifacts

- `docs/goal3/GOAL3_DAG_COMPRESSION_STUDY.md`
- `outputs/v0/dag_compression_summary.json`
- `outputs/v0/dag_compression_metrics.csv`
- `outputs/v0/dag_alpha_threshold_summary.json`
- `docs/goal3/GOAL3_DAG_COMPRESSION_FINDINGS.md`
- `docs/goal3/GOAL3_DAG_SEMANTIC_AUDIT.md`
