# Goal 3R V1 Corpus Comparison

## Question

Goal 3R repairs the expression generator before any e-graph or neural work. The
v0 seed-0 corpus was too degenerate: every expression had exact max depth, many
`srepr` rows were duplicates, log arguments were dominated by `exp(...)` and
`1`, and trivial identities appeared frequently. These artifacts could make
later compression results look stronger than they are.

The v1 run keeps the same high-level operator family (`add`, `mul`, `exp`,
`log`), symbols (`x`, `y`), count (`10,000`), seed (`0`), and max depth (`4`),
but adds target-depth sampling, intermediate leaves, positive-domain log
arguments, srepr deduplication, and a cap on heavily trivial generated rows.

## Baseline Switch

The v1 corpus is now the default corpus for all future compression and
ML-facing experiments. V0 is a pilot corpus and is deprecated for result claims.
Existing v0 Goal 2/3 outputs remain useful for historical comparison and
diagnostics, but Goal 4 e-graph compression, Goal 5 ML-facing compression, and
Goal 6 GNN training/evaluation must use v1.

Any e-graph result computed on v0 must be labeled diagnostic-only.

Reserved future subset labels:

- `all_v1`: the full repaired v1 corpus
- `nontrivial_v1`: a future filtered v1 subset with low triviality
- `identity_heavy_v1`: a future diagnostic v1 subset enriched for identity
  simplification opportunities

## Corpus Quality

| Metric | v0 | v1 |
| --- | ---: | ---: |
| generated rows | 10000 | 10000 |
| unique sreprs | 7297 | 10000 |
| duplicate srepr rate | 27.03% | 0.00% |
| exact max-depth rows | 10000 | 5594 |

Depth distribution:

| Depth | v0 count | v1 count |
| ---: | ---: | ---: |
| 0 | 0 | 3 |
| 1 | 0 | 18 |
| 2 | 0 | 466 |
| 3 | 0 | 3919 |
| 4 | 10000 | 5594 |

## Log Argument Distribution

v0 log arguments were mostly exp-wrapped or constant `1`.

| Log argument class | v0 count | v1 count |
| --- | ---: | ---: |
| `add` | 0 | 1742 |
| `exp` | 9121 | 637 |
| `mul` | 0 | 1363 |
| `one` | 6669 | 1444 |
| `symbol` | 0 | 4067 |

The v1 generator assumes the configured variables are evaluated on the positive
real domain. Under that domain, symbols, `1`, sums/products of positive
expressions, and `exp(anything)` are valid log arguments without blanket
exp-wrapping.

## Triviality Metrics

The v1 config does not eliminate all trivial expressions. It caps heavily
trivial generated rows while still reporting the residual triviality rate.

| Triviality count | v0 total | v1 total |
| --- | ---: | ---: |
| Mul-by-1 | 5401 | 4368 |
| constant-only Add/Mul subtrees | 4019 | 1415 |
| `log(1)` | 6669 | 1444 |
| `exp(log(...))` | 2732 | 783 |
| `log(exp(...))` | 9121 | 637 |

## Raw Alpha Summary

| Metric | v0 | v1 |
| --- | ---: | ---: |
| processed | 10000 | 10000 |
| supported | 10000 | 10000 |
| mean tree alpha | 10.648995087903057 | 12.250527857284991 |
| median tree alpha | 11.375 | 12.454545454545455 |
| p90 tree alpha | 14.208333333333334 | 15.0 |
| p95 tree alpha | 14.909090909090908 | 15.642857142857142 |
| max tree alpha | 18.733333333333334 | 18.733333333333334 |
| current-threshold percent below | 0.0% | 0.06% |

## DAG Alpha Summary

| Metric | v0 | v1 |
| --- | ---: | ---: |
| mean DAG alpha vs AST tree | 3.4323505588013528 | 4.036107723777943 |
| median DAG alpha vs AST tree | 3.5 | 4.0 |
| p90 DAG alpha vs AST tree | 4.25 | 4.9 |
| mean DAG alpha vs AST DAG | 4.535780564778956 | 5.24252227317617 |
| median DAG alpha vs AST DAG | 4.666666666666667 | 5.25 |
| p90 DAG alpha vs AST DAG | 5.666666666666667 | 6.5 |
| mean EML DAG compression | 3.110470058705476 | 3.0608322420245515 |
| median EML DAG compression | 3.0806451612903225 | 3.0634920634920637 |
| p90 EML DAG compression | 3.759493670886076 | 3.72 |
| current-threshold below before DAG | 0.0% | 0.06% |
| current-threshold below after DAG vs AST tree | 0.03% | 0.22% |
| current-threshold below after DAG vs AST DAG | 0.03% | 0.22% |

## Interpretation

The v1 generator fixes the corpus pathologies that could have distorted Goal 2
and Goal 3:

- actual depths are variable instead of all exactly max depth
- srepr duplicates are removed from the generated corpus
- log arguments are no longer almost all `exp(...)` or `1`
- trivial identities are reduced and explicitly measured
- `Pow` parsing is shared and consistent across dataset, stratified, and DAG
  code paths

The stronger corpus does not rescue raw official pure EML structurally. Raw tree
alpha is higher on v1 than v0, and DAG alpha is also higher. Exact structural DAG
sharing still helps substantially, but the current-threshold pass rate remains
well below 1% after DAG sharing.

This is structural evidence only. It does not claim model-performance
improvement or degradation.

## Reproducible Commands

```bash
.venv/bin/python -m geml.data.generate_exprs --config configs/expression_v1.yaml
.venv/bin/python -m geml.experiments.run_goal2_expansion_pipeline --config configs/expansion_v1.yaml
.venv/bin/python -m geml.experiments.run_goal3_dag_pipeline --config configs/dag_compression_v1.yaml
```

Primary generated summaries:

- `outputs/v1/expression_generation_summary.json`
- `outputs/v1/expansion_generation_summary.json`
- `outputs/v1/official_eml_compiler_summary.json`
- `outputs/v1/dag_compression_summary.json`
- `outputs/v1/v0_v1_comparison_summary.json`
