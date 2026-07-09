# Goal 4 Summary

Goal 4 implemented non-ML e-graph compression for the repaired v1 corpus. It runs safe and positive-real formal rewrite modes, uses EML-aware extraction, compares against the Goal 3 exact EML-DAG baseline, mines successes/failures, plots summary artifacts, and audits semantics, structural purity, and rewrite provenance.

## Headline Result

| Mode | Processed | Success | Timeout | Validation failed | Extraction failures | Official compile failures | Median Goal 3 alpha | Median optimized alpha | Median gain | Below before | Below after success-only | Below after all processed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `safe` | 10000 | 9316 | 241 | 471 | 0 | 0 | 4.0 | 3.6363636363636362 | 1.0454545454545454 | 0.220% | 1.020% | 0.950% |
| `positive_real_formal` | 10000 | 8941 | 522 | 583 | 15 | 0 | 4.0 | 3.3636363636363638 | 1.1692307692307693 | 0.220% | 5.827% | 5.210% |

The safe mode improves many rows but only modestly changes the threshold pass rate. The positive-real formal mode improves more rows, but its branch-sensitive assumptions must be reported separately. Neither mode broadly rescues official pure EML-DAG size under the current threshold.

## Subsets

| Subset | Mode | Count | Success | Median Goal 3 alpha | Median optimized alpha | Median gain | Improved | Unchanged | Worse | Below before | Below after success-only | Below after all processed | Timeout | Validation failure | Branch-sensitive usage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all_v1` | `safe` | 10000 | 9316 | 4.0 | 3.6363636363636362 | 1.0454545454545454 | 53.027% | 45.298% | 1.675% | 0.220% | 1.020% | 0.950% | 2.410% | 6.840% | 0.000% |
| `all_v1` | `positive_real_formal` | 10000 | 8941 | 4.0 | 3.3636363636363638 | 1.1692307692307693 | 72.934% | 25.791% | 1.275% | 0.220% | 5.827% | 5.210% | 5.220% | 10.590% | 100.000% |
| `nontrivial_v1` | `safe` | 3733 | 3428 | 4.111111111111111 | 4.071428571428571 | 1.0 | 25.554% | 71.120% | 3.326% | 0.402% | 0.292% | 0.268% | 0.884% | 8.170% | 0.000% |
| `nontrivial_v1` | `positive_real_formal` | 3733 | 3359 | 4.111111111111111 | 3.933333333333333 | 1.0 | 37.571% | 59.244% | 3.185% | 0.402% | 0.298% | 0.268% | 1.286% | 10.019% | 100.000% |
| `identity_heavy_v1` | `safe` | 6267 | 5888 | 4.0 | 3.4166666666666665 | 1.1538461538461537 | 69.022% | 30.265% | 0.713% | 0.112% | 1.444% | 1.356% | 3.319% | 6.048% | 0.000% |
| `identity_heavy_v1` | `positive_real_formal` | 6267 | 5582 | 4.0 | 3.0444664031620556 | 1.2692307692307692 | 94.214% | 5.661% | 0.125% | 0.112% | 9.154% | 8.154% | 7.563% | 10.930% | 100.000% |

Identity-heavy rows drive the largest gains. Nontrivial rows remain difficult.

## Best And Worst Families

- top success family: `positive_real_formal:mixed_exp+log[log+exp]`
- top failure family: `positive_real_formal:exp[Mul+exp]`
- top success signature: `exp+log`
- top failure signature: `exp`

## Semantic Audit

- audit rows: `28`
- all semantic validation valid: `True`
- all EML-DAG validation valid: `True`
- all structural purity valid: `True`
- safe branch-sensitive applications: `0`
- positive-real branch-sensitive applications: `5`
- provenance invalid count: `0`
- SymPy simplify rewrite path free: `True`
- audit JSON: `outputs/v1/goal4_egraph_semantic_audit.json`
- audit CSV: `outputs/v1/goal4_egraph_semantic_audit.csv`
- audit docs: `docs/goal4/GOAL4_EGRAPH_SEMANTIC_AUDIT.md`

## Primary Artifacts

- `outputs/v1/egraph_compression_summary.json`
- `outputs/v1/GOAL4_EGRAPH_COMPRESSION_FINDINGS.md`
- `docs/goal4/GOAL4_NONML_COMPRESSION_STUDY.md`
- `docs/goal4/GOAL4_SUMMARY.md`
- `docs/goal4/GOAL4_EGRAPH_SEMANTIC_AUDIT.md`
- `outputs/v1/plots_goal4`

## Goal 5 Recommendation

Non-ML e-graph compression is a useful baseline and should remain in the evaluation stack, but it is not enough by itself. The threshold pass rate remains low when failed rows are kept in the denominator: safe mode `0.950%` and positive-real mode `5.210%`. Nontrivial positive-real median gain is `1.0`. Goal 5 should therefore still investigate ML-facing motif or macro compression, with honest separation between structural compression results and future model-performance claims.
