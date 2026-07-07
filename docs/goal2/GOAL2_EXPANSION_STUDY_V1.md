# Goal 2 Expansion Study

## Goal And Scientific Question

Goal 2 asks whether the official pure recursive EML representation is structurally smaller than a standard expression AST before any compression or modeling. The measured quantity is alpha:

```text
alpha = |T_EML| / |T_AST|
```

If raw pure EML alpha is usually far above the theoretical threshold, then uncompressed EML trees are unlikely to be computationally smaller than ASTs.

## Official Compiler And Representation

The pure compiler ports macro definitions from:

- Repository: `VA00/SymbolicRegressionPackage`
- File: `EML_toolkit/EmL_compiler/eml_compiler_v4.py`

Core primitive:

```text
EML(a, b) = exp(a) - log(b)
```

Goal 2 pure EML grammar:

```text
P ::= variable | 1 | eml(P, P)
```

Every internal node must be `eml`; leaves may only be variables or constant `1`. Derived leaves are invalid for alpha because they can hide compound source expressions inside a single leaf and artificially reduce expansion.

## Dataset And Run Configuration

- expression count: `10000`
- seed: `0`
- max source depth: `4`
- representation mode: `restricted_eml_pure`
- supported official pure EML count: `10000`
- unsupported count: `0`

## Threshold Model

Threshold formula:

```text
alpha_threshold = 1 + log(K) / log(4L)
```

Primary row-level K/L values: `K=4`, `L=3`.

| Scenario | K | L | Alpha threshold | Percent below | Percent above |
| --- | ---: | ---: | ---: | ---: | ---: |
| `current_grammar` | 4 | 3 | 1.5578858913022597 | 0.06 | 99.94 |
| `generous_operator_vocab` | 20 | 3 | 2.2055713536802566 | 0.15 | 99.85 |
| `larger_operator_vocab` | 50 | 3 | 2.574313870407124 | 0.15 | 99.85 |

## Aggregate Alpha Results

- mean alpha: `12.250527857284991`
- median alpha: `12.454545454545455`
- p90 alpha: `15.0`
- p95 alpha: `15.642857142857142`
- max alpha: `18.733333333333334`
- current-threshold percent below: `0.06`

## Stratified Findings

- highest median-alpha dominant family: `Mul` with median alpha `14.454545454545455`
- worst operator signature by failure mining: `Mul` with median alpha `17.285714285714285`
- AST-size bucket summaries show alpha rising as source trees get larger:

| AST node bucket | Count | Median alpha | P90 alpha | Mean EML nodes |
| --- | ---: | ---: | ---: | ---: |
| `1-3` | 30 | 3.25 | 13.666666666666666 | 17.0 |
| `4-7` | 1956 | 10.714285714285714 | 13.833333333333334 | 66.79856850715747 |
| `8-15` | 6482 | 12.538461538461538 | 14.916666666666666 | 137.11292810860846 |
| `16-31` | 1532 | 13.61111111111111 | 15.318181818181818 | 263.65143603133157 |

- Boolean feature summaries show Add/Mul participation is the main structural risk:

| Feature | Count | Median alpha | P90 alpha | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `contains_Mul` | 8731 | 12.8125 | 15.285714285714286 | 0.0 |
| `contains_Add` | 8838 | 12.446022727272727 | 14.9 | 0.0 |

## Plots

- `outputs/v1/plots/alpha_histogram.png`
- `outputs/v1/plots/alpha_histogram_log_scale.png`
- `outputs/v1/plots/ast_nodes_vs_eml_nodes.png`
- `outputs/v1/plots/ast_depth_vs_alpha.png`
- `outputs/v1/plots/eml_depth_vs_alpha.png`
- `outputs/v1/plots/alpha_by_ast_depth.png`
- `outputs/v1/plots/alpha_by_operator_family.png`
- `outputs/v1/plots/percent_below_threshold_by_ast_depth.png`
- `outputs/v1/plots/percent_below_threshold_by_operator_family.png`
- `outputs/v1/plots/eml_nodes_by_ast_nodes.png`

## Failure Modes

The top failure-mode tables are:

- `outputs/v1/top_alpha_explosions.csv`
- `outputs/v1/top_eml_node_explosions.csv`
- `outputs/v1/top_eml_depth_explosions.csv`
- `outputs/v1/worst_operator_signatures.csv`
- `outputs/v1/depth_failure_modes.csv`

Worst signature preview:

| Signature | Median alpha | P90 alpha | Count | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `Mul` | 17.285714285714285 | 17.88888888888889 | 60 | 0.0 |
| `Add+Mul` | 14.777777777777779 | 16.333333333333332 | 582 | 0.0 |
| `Mul+log` | 14.5 | 16.384615384615383 | 185 | 0.0 |
| `Mul+exp` | 13.833333333333334 | 15.375 | 307 | 0.0 |
| `Add+Mul+log` | 13.75 | 15.3 | 1675 | 0.0 |

Depth failure-mode preview:

| AST depth | Mean alpha | P90 alpha | Mean EML/AST nodes | Count |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 12.26481908058203 | 15.285714285714286 | 12.608096633706758 | 3919 |
| 2 | 12.055288166768854 | 15.285714285714286 | 12.400298173686172 | 466 |
| 4 | 12.275239240317244 | 14.894736842105264 | 12.674345378460533 | 5594 |
| 1 | 8.38888888888889 | 13.666666666666666 | 9.125 | 18 |
| 0 | 1.0 | 1.0 | 1.0 | 3 |

## Safe-Regime Candidates

Closest raw pure EML candidate: `leaf_only` with median alpha `1.0` and median threshold gap `-0.5578858913022597`.

| Signature | Median alpha | Median threshold gap | P90 alpha | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `leaf_only` | 1.0 | -0.5578858913022597 | 1.0 | 100.0 |
| `exp` | 1.7083333333333335 | 0.15044744203107374 | 1.8 | 25.0 |
| `exp+log` | 2.75 | 1.1921141086977403 | 3.4 | 0.0 |
| `log` | 3.5 | 1.9421141086977403 | 3.5 | 0.0 |
| `Add+exp+log` | 8.636363636363637 | 7.078477745061377 | 9.916666666666666 | 0.0 |

No robust raw pure EML safe regime appears under the current threshold when using these generated expressions.

## Conclusion

Raw official pure EML expansion is scientifically valid but structurally expensive. The representation removes operator vocabulary but expands common Add/Mul/Pow/log/exp source patterns into much larger trees. The 10k run shows alpha far above all tested thresholds for almost every expression, so raw pure EML trees are unlikely to be computationally smaller than ASTs without a separate compression mechanism.

This is structural evidence only. It is not model-performance evidence.

## Recommendation For Goal 3

Goal 3 should study DAG compression or shared-subexpression compression for pure EML before introducing GNNs, neural models, or equivalence-pair generation. Compression should be measured against the same AST baseline, threshold scenarios, and failure strata from Goal 2.

## Reproducible Commands

```bash
.venv/bin/python -m geml.experiments.run_goal2_expansion_pipeline --config configs/expansion_v0.yaml
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format . --check
```
