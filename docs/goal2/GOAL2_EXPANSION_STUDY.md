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
| `current_grammar` | 4 | 3 | 1.5578858913022597 | 0.0 | 100.0 |
| `generous_operator_vocab` | 20 | 3 | 2.2055713536802566 | 0.25 | 99.75 |
| `larger_operator_vocab` | 50 | 3 | 2.574313870407124 | 0.25 | 99.75 |

## Aggregate Alpha Results

- mean alpha: `10.648995087903057`
- median alpha: `11.375`
- p90 alpha: `14.208333333333334`
- p95 alpha: `14.733333333333333`
- max alpha: `17.366666666666667`
- current-threshold percent below: `0.0`

## Stratified Findings

- highest median-alpha dominant family: `Mul` with median alpha `13.947368421052632`
- worst operator signature by failure mining: `Add+Mul` with median alpha `16.677419354838708`
- AST-size bucket summaries show alpha rising as source trees get larger:

| AST node bucket | Count | Median alpha | P90 alpha | Mean EML nodes |
| --- | ---: | ---: | ---: | ---: |
| `4-7` | 1590 | 3.4 | 8.5 | 29.559748427672957 |
| `8-15` | 3985 | 10.333333333333334 | 12.785714285714286 | 117.35382685069008 |
| `16-31` | 4425 | 12.875 | 14.791666666666666 | 269.64361581920906 |

- Boolean feature summaries show Add/Mul participation is the main structural risk:

| Feature | Count | Median alpha | P90 alpha | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `contains_Mul` | 8107 | 11.95 | 14.368421052631579 | 0.0 |
| `contains_Add` | 8138 | 11.857142857142858 | 14.318181818181818 | 0.0 |

## Plots

- `outputs/v0/plots/alpha_histogram.png`
- `outputs/v0/plots/alpha_histogram_log_scale.png`
- `outputs/v0/plots/ast_nodes_vs_eml_nodes.png`
- `outputs/v0/plots/ast_depth_vs_alpha.png`
- `outputs/v0/plots/eml_depth_vs_alpha.png`
- `outputs/v0/plots/alpha_by_ast_depth.png`
- `outputs/v0/plots/alpha_by_operator_family.png`
- `outputs/v0/plots/percent_below_threshold_by_ast_depth.png`
- `outputs/v0/plots/percent_below_threshold_by_operator_family.png`
- `outputs/v0/plots/eml_nodes_by_ast_nodes.png`

## Failure Modes

The top failure-mode tables are:

- `outputs/v0/top_alpha_explosions.csv`
- `outputs/v0/top_eml_node_explosions.csv`
- `outputs/v0/top_eml_depth_explosions.csv`
- `outputs/v0/worst_operator_signatures.csv`
- `outputs/v0/depth_failure_modes.csv`

Worst signature preview:

| Signature | Median alpha | P90 alpha | Count | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `Add+Mul` | 16.677419354838708 | 16.677419354838708 | 3 | 0.0 |
| `Add+Mul+log` | 15.157407407407408 | 16.1 | 34 | 0.0 |
| `Add+Mul+exp` | 13.0 | 15.173913043478262 | 887 | 0.0 |
| `Add+Mul+exp+log` | 12.055555555555555 | 14.26923076923077 | 6174 | 0.0 |
| `Mul+exp` | 10.875 | 13.916666666666666 | 58 | 0.0 |

Depth failure-mode preview:

| AST depth | Mean alpha | P90 alpha | Mean EML/AST nodes | Count |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 10.648995087903057 | 14.208333333333334 | 11.791297863820269 | 10000 |

## Safe-Regime Candidates

Closest raw pure EML candidate: `exp` with median alpha `1.8` and median threshold gap `0.2421141086977403`.

| Signature | Median alpha | Median threshold gap | P90 alpha | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `exp` | 1.8 | 0.2421141086977403 | 1.8 | 0.0 |
| `exp+log` | 3.4 | 1.8421141086977402 | 3.4 | 0.0 |
| `Add+exp+log` | 6.916666666666667 | 5.358780775364407 | 9.583333333333334 | 0.0 |
| `Add+exp` | 7.375 | 5.817114108697741 | 9.266666666666667 | 0.0 |
| `Mul+exp+log` | 9.5 | 7.942114108697741 | 13.615384615384615 | 0.0 |

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
