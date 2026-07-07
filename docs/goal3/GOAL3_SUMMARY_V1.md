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
- mean tree alpha: `12.250527857284991`
- median tree alpha: `12.454545454545455`
- p90 tree alpha: `15.0`
- mean DAG alpha vs AST tree: `4.036107723777943`
- median DAG alpha vs AST tree: `4.0`
- p90 DAG alpha vs AST tree: `4.9`
- mean EML DAG compression: `3.0608322420245515`
- median EML DAG compression: `3.0634920634920637`
- p90 EML DAG compression: `3.72`
- current threshold below before DAG: `0.06%`
- current threshold below after DAG vs AST tree: `0.22%`

## Best And Worst Families

- top compression-success signature: `Add+Mul`
- top compression-failure signature: `Mul`

## Semantic Audit

- audit expressions: `12`
- structurally valid: `12`
- numerically valid: `12`

## Conclusion

Exact structural DAG sharing reduces the median alpha from `12.454545454545455` to `4.0` versus AST tree size, with median EML DAG compression `3.0634920634920637`. The current-threshold pass rate improves from `0.06%` before DAG sharing to `0.22%` after DAG sharing. This helps materially, but it does not broadly rescue raw official pure EML under the current structural threshold.

## Primary Artifacts

- `docs/goal3/GOAL3_DAG_COMPRESSION_STUDY_V1.md`
- `outputs/v1/dag_compression_summary.json`
- `outputs/v1/dag_compression_metrics.csv`
- `outputs/v1/dag_alpha_threshold_summary.json`
- `docs/goal3/GOAL3_DAG_COMPRESSION_FINDINGS_V1.md`
- `docs/goal3/GOAL3_DAG_SEMANTIC_AUDIT_V1.md`
