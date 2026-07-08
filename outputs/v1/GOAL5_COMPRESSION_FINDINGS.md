# Goal 5 Compression Findings

## Findings

1. Macro graphs gave a transparent official-compiler abstraction with median alpha 0.778 and median gain 5.250 vs Goal 3.
2. Frequent motifs were the strongest simple compression baseline at the median, with gain 7.400 vs Goal 3.
3. Learned motifs preserved exact reconstruction but did not clearly beat the random motif baseline at the median (1.000).
4. The neural e-graph extractor had median zero regret and large scoring-speed improvement (109.305x), but it still had validation failures and does not prove reasoning ability.
5. Hierarchical graph export produced an audit-ready dataset with 100.000% reconstruction validation and zero missing expansion mappings.

## Integrity

- Reconstruction failure count: 0
- Neural validation failure count: 1,129
- Hidden pure-EML violations: False
- Final symbolic-reasoning GNN trained: False

## Output Artifacts

- `outputs/v1/goal5_compression_comparison.csv`
- `outputs/v1/goal5_compression_summary.json`
- `docs/goal5/GOAL5_ML_FACING_COMPRESSION_STUDY.md`
- `docs/goal5/GOAL5_SUMMARY.md`
- `outputs/v1/GOAL5_COMPRESSION_FINDINGS.md`
