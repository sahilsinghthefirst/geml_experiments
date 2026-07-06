# Goal 2 Summary

This document summarizes the Goal 2 expansion-factor work completed so far. It consolidates the representation decisions, official pure EML compiler, 10,000-expression expansion run, alpha-threshold analysis, stratified analysis, plots, failure mining, integration pipeline, and decision memo.

## Executive Summary

Goal 2 asked whether the official pure recursive EML representation is structurally smaller than ordinary ASTs before compression or modeling.

The answer is no for raw trees. The official pure EML compiler is representation-complete for the supported arithmetic/log/exp subset, but it expands common source expressions substantially. In the fixed-seed 10,000-expression run:

- processed expressions: `10000`
- official pure EML supported: `10000`
- unsupported: `0`
- mean alpha: `10.648995087903057`
- median alpha: `11.375`
- p90 alpha: `14.208333333333334`
- p95 alpha: `14.733333333333333`
- max alpha: `17.366666666666667`

Alpha almost never falls below the theoretical threshold:

| Scenario | K | L | Alpha threshold | Percent below | Percent above |
| --- | ---: | ---: | ---: | ---: | ---: |
| `current_grammar` | 4 | 3 | 1.5578858913022597 | 0.0 | 100.0 |
| `generous_operator_vocab` | 20 | 3 | 2.2055713536802566 | 0.25 | 99.75 |
| `larger_operator_vocab` | 50 | 3 | 2.574313870407124 | 0.25 | 99.75 |

Central conclusion:

> Raw pure EML is representation-complete but structurally expensive; therefore, the next research step should test whether DAG compression and learned graph models recover enough structure to make EML useful.

Goal 2 does not kill GEML. It rules out the naive claim that raw pure EML trees are compact by themselves. The remaining plausible hypothesis is that EML may become useful after graph sharing, DAG compression, and fair learned baselines.

## Scope And Non-Scope

Goal 2 includes:

- formal restricted EML representation semantics
- official pure recursive EML compiler port
- fixed-seed 10,000-expression expansion run
- alpha threshold computation
- stratified expansion analysis
- reproducible plots
- failure-case mining
- final integrated pipeline
- decision memo for the next research step

Goal 2 explicitly does not include:

- DAG compression
- GNNs or neural models
- equivalence-pair generation
- model-performance claims

All results are structural evidence about representation size and expansion, not evidence about downstream ML performance.

## Goal 2.0: Representation Semantics

Goal 2 began by locking the restricted EML semantics before expansion-factor experiments.

The key issue from Goal 1 was the derived lift rule:

```text
E -> eml(log(E), 1)
```

For `Add` and `Mul`, this rule could hide a compound expression inside a derived leaf such as `log(x + y)`. That made alpha artificially small because the hidden source subtree counted as one EML leaf.

The policy established in Goal 2.0:

- hidden compound derived leaves are not valid atomic EML leaves
- derived leaves must not contribute to serious alpha measurements
- alpha-valid rows must use pure, valid restricted EML only
- derived-mode rows must have `alpha_valid=false`

Representation modes:

- `ast`: normal binary AST representation
- `restricted_eml_pure`: official recursive pure EML representation
- `restricted_eml_with_derived`: diagnostic representation that may contain derived leaves

Pure restricted EML grammar:

```text
P ::= variable | 1 | eml(P, P)
```

where every internal node is `eml`, every `eml` node has exactly two children, and leaves are only variables or constant `1`.

Implementation/docs:

- `geml/symbolic/representations.py`
- `geml/symbolic/eml_nodes.py`
- `geml/symbolic/eml_transpile.py`
- `docs/goal2_representation_semantics.md`

## Goal 2.1: Scale-Test Expression Generation And Metrics Export

Goal 2.1 added the fixed-seed expansion input generation and raw metrics export.

Configuration:

- `configs/expansion_v0.yaml`
- seed: `0`
- count: `10000`
- max depth: `4`
- representation mode: `restricted_eml_pure`
- operators: `add`, `mul`, `exp`, `log`
- symbols: `x`, `y`

Important correctness point:

- generated rows use `srepr` as the authoritative structural representation
- human-readable `expression` strings are retained for display only

Primary outputs:

- `outputs/v0/expansion_inputs.jsonl`
- `outputs/v0/expansion_raw_metrics.jsonl`
- `outputs/v0/expansion_raw_metrics.csv`

Per-row raw metrics include:

- expression string
- `srepr`
- source serialization mode
- representation mode
- AST node/edge/depth/leaf/operator counts
- EML node/edge/depth/leaf/operator counts
- normal/derived/hidden EML leaf counts
- alpha
- alpha validity
- support flag
- error message

Implementation/tests:

- `geml/experiments/expansion_study.py`
- `geml/data/dataset.py`
- `tests/test_expansion_study.py`
- `tests/test_dataset.py`

## Goal 2.1b: Official Pure Recursive EML Compiler

Goal 2.1b replaced the temporary Add/Mul unsupported/derived behavior with the official recursive EML compiler.

Official source:

- repository: `VA00/SymbolicRegressionPackage`
- file: `EML_toolkit/EmL_compiler/eml_compiler_v4.py`
- URL: `https://github.com/VA00/SymbolicRegressionPackage/blob/master/EML_toolkit/EmL_compiler/eml_compiler_v4.py`

Core primitive:

```text
EML(a, b) = exp(a) - log(b)
```

Final tree constraints:

- every internal node is `eml`
- every leaf is a source variable or constant `1`
- no final `Add`, `Mul`, `Sub`, `Div`, `Pow`, `Exp`, `Log`, `Sin`, `Cos`, `Derived`, or hidden compound-expression node
- macro helper names such as `eml_log`, `eml_exp`, `eml_add`, and `eml_mul` never appear as final node labels

Ported macro definitions:

```text
eml_exp(z)      = EML(z, 1)
eml_log(z)      = EML(1, EML(EML(1, z), 1))
eml_zero()      = eml_log(1)
eml_sub(a, b)   = EML(eml_log(a), eml_exp(b))
eml_neg(z)      = eml_sub(eml_zero(), z)
eml_add(a, b)   = eml_sub(a, eml_neg(b))
eml_inv(z)      = eml_exp(eml_neg(eml_log(z)))
eml_mul(a, b)   = eml_exp(eml_add(eml_log(a), eml_log(b)))
eml_div(a, b)   = eml_mul(a, eml_inv(b))
eml_pow(a, b)   = eml_exp(eml_mul(b, eml_log(a)))
eml_one()       = 1
```

Constants:

- only primitive constant leaf is `1`
- `0` compiles as `eml_zero()`
- negative integers use `eml_neg`
- positive integers use official binary repeated doubling/addition
- rationals use official numerator/inverse-denominator construction
- floats are converted through `Rational(str(x))`

Supported SymPy nodes:

- `Symbol`
- `Integer`
- `Rational`
- `Float`
- `Add`
- `Mul`
- `Pow`
- `exp`
- `log`

Unsupported for now:

- trigonometric functions
- inverse trig
- hyperbolic functions
- `Abs`
- arbitrary unsupported SymPy nodes

Implementation/tests/docs:

- `geml/symbolic/official_eml_compiler.py`
- `tests/test_official_eml_compiler.py`
- `docs/GOAL2_OFFICIAL_EML_COMPILER.md`

The tests cover structural purity, tree-not-DAG invariants, official-style string emission, exact small formulas, and numeric equivalence over safe positive real inputs.

## Goal 2.1c: Expansion Size Audit

Goal 2.1c audited pure expansion size.

Additional outputs:

- `outputs/v0/official_eml_top20_alpha.json`
- `outputs/v0/official_eml_top20_depth.json`
- `outputs/v0/official_eml_simple_examples.json`

Simple-expression audit:

| Expression | AST nodes | EML nodes | Alpha | EML depth | Derived leaves |
| --- | ---: | ---: | ---: | ---: | ---: |
| `x+y` | 3 | 27 | 9.0 | 9 | 0 |
| `x*y` | 3 | 41 | 13.666666666666666 | 10 | 0 |
| `log(x)` | 2 | 7 | 3.5 | 3 | 0 |
| `exp(x)` | 2 | 3 | 1.5 | 1 | 0 |
| `x**2` | 3 | 75 | 25.0 | 18 | 0 |

Interpretation:

- `exp` is relatively cheap because `eml_exp(z) = EML(z, 1)`
- `log` adds nested EML structure
- `Add` expands through subtraction and negation
- `Mul` expands through log, add, and exp
- `Pow` expands through multiplication and log
- integer constants are also expanded recursively from `1`

These expansions are correct by official macro semantics, but tree size is high by construction.

## Goal 2.2: Alpha Thresholds

Goal 2.2 added threshold computation and per-row classification.

Threshold formula:

```text
alpha_threshold = 1 + log(K) / log(4L)
```

Raw metric rows now include:

- `alpha`
- `alpha_threshold`
- `below_threshold`

Threshold summary outputs:

- `outputs/v0/expansion_alpha_summary.csv`
- `outputs/v0/expansion_alpha_summary.json`

Scenarios:

| Scenario | K | L | Alpha threshold | Below count | Above count | Percent below |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_grammar` | 4 | 3 | 1.5578858913022597 | 0 | 10000 | 0.0 |
| `generous_operator_vocab` | 20 | 3 | 2.2055713536802566 | 25 | 9975 | 0.25 |
| `larger_operator_vocab` | 50 | 3 | 2.574313870407124 | 25 | 9975 | 0.25 |

Conclusion:

Raw pure EML trees are far above the threshold almost everywhere. This makes uncompressed pure EML tree size unfavorable relative to AST size.

## Goal 2.3: Stratified Expansion Analysis

Goal 2.3 asked where expansion is worst and which families are least bad.

Module:

- `geml/experiments/stratified_expansion.py`

Inputs:

- `outputs/v0/expansion_raw_metrics.csv`
- `outputs/v0/expansion_alpha_summary.csv`
- `outputs/v0/expansion_alpha_summary.json`

Feature extraction from authoritative `srepr`:

- `count_Add`
- `count_Mul`
- `count_Pow`
- `count_exp`
- `count_log`
- `count_symbols`
- `count_constants`
- `contains_Add`
- `contains_Mul`
- `contains_Pow`
- `contains_exp`
- `contains_log`

Bucket features:

- `ast_nodes_bucket`
- `ast_depth_bucket`
- `eml_nodes_bucket`
- `alpha_bucket`

Grouping dimensions:

- AST depth
- AST node-count bucket
- dominant operator family
- exact operator signature
- boolean operator features

Stratified outputs:

- `outputs/v0/alpha_by_ast_depth.csv`
- `outputs/v0/alpha_by_ast_size_bucket.csv`
- `outputs/v0/alpha_by_operator_family.csv`
- `outputs/v0/alpha_by_operator_signature.csv`
- `outputs/v0/alpha_by_boolean_features.csv`

Key findings:

- highest median-alpha dominant family: `Mul`
- `Mul` median alpha: `13.947368421052632`
- worst signature by failure mining: `Add+Mul`
- `Add+Mul` median alpha: `16.677419354838708`
- Add/Mul participation is the primary structural risk

AST-size bucket findings:

| AST node bucket | Count | Median alpha | P90 alpha | Mean EML nodes |
| --- | ---: | ---: | ---: | ---: |
| `4-7` | 1590 | 3.4 | 8.5 | 29.559748427672957 |
| `8-15` | 3985 | 10.333333333333334 | 12.785714285714286 | 117.35382685069008 |
| `16-31` | 4425 | 12.875 | 14.791666666666666 | 269.64361581920906 |

Boolean feature findings:

| Feature | Count | Median alpha | P90 alpha | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `contains_Mul` | 8107 | 11.95 | 14.368421052631579 | 0.0 |
| `contains_Add` | 8138 | 11.857142857142858 | 14.318181818181818 | 0.0 |

## Goal 2.4: Reproducible Plots

Goal 2.4 produced reproducible Matplotlib plots from saved CSV/JSON artifacts only.

Module:

- `geml/experiments/plot_expansion_study.py`

Rules:

- use Matplotlib only
- no seaborn
- read only saved artifacts from `outputs/v0`
- save plots under `outputs/v0/plots`
- include clear title, x-axis label, y-axis label, and filename for each plot

Plot outputs:

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

Top-expression table outputs:

- `outputs/v0/top_20_alpha_expressions.csv`
- `outputs/v0/top_20_eml_node_expressions.csv`
- `outputs/v0/top_20_eml_depth_expressions.csv`

Dependency update:

- `matplotlib>=3.9` added to `pyproject.toml`

Test:

- `tests/test_plot_expansion_study.py`

The smoke test creates tiny fake saved CSV/JSON files, runs the plot exporter, and verifies all plot/table files are created.

## Goal 2.5: Failure-Case Mining

Goal 2.5 identified expressions and structural patterns where raw pure EML expansion is worst.

Module:

- `geml/experiments/expansion_failure_mining.py`

Inputs:

- `outputs/v0/expansion_raw_metrics.csv`
- `outputs/v0/alpha_by_operator_family.csv`
- `outputs/v0/alpha_by_operator_signature.csv`
- `outputs/v0/alpha_by_ast_depth.csv`

Outputs:

- `outputs/v0/top_alpha_explosions.csv`
- `outputs/v0/top_eml_node_explosions.csv`
- `outputs/v0/top_eml_depth_explosions.csv`
- `outputs/v0/worst_operator_signatures.csv`
- `outputs/v0/safest_operator_signatures.csv`
- `outputs/v0/depth_failure_modes.csv`
- `outputs/v0/safe_eml_regime_candidates.csv`
- `outputs/v0/GOAL2_FAILURE_CASES.md`

Top expression failure tables include:

- expression
- `srepr`
- AST counts
- EML counts
- alpha
- alpha threshold
- threshold gap
- official-style EML snippet
- full official-style EML string length
- truncation flag

Worst operator signatures:

| Signature | Median alpha | P90 alpha | Count | Percent below threshold |
| --- | ---: | ---: | ---: | ---: |
| `Add+Mul` | 16.677419354838708 | 16.677419354838708 | 3 | 0.0 |
| `Add+Mul+log` | 15.157407407407408 | 16.1 | 34 | 0.0 |
| `Add+Mul+exp` | 13.0 | 15.173913043478262 | 887 | 0.0 |
| `Add+Mul+exp+log` | 12.055555555555555 | 14.26923076923077 | 6174 | 0.0 |
| `Mul+exp` | 10.875 | 13.916666666666666 | 58 | 0.0 |

Highest-alpha example:

- index: `8274`
- expression: `((x*x)*(y*y))*((x*x)*(x + 1)) + ((1*x)*(x*y) + (x*1)*exp(1))`
- AST nodes: `30`
- EML nodes: `521`
- EML depth: `30`
- alpha: `17.366666666666667`

Highest-depth examples reached EML depth `34`.

Common structural causes:

- Add/Mul-heavy signatures dominate failures
- `Mul` is the strongest dominant operator family by median alpha
- log and exp wrappers add depth
- repeated macro expansion creates large repeated tree patterns

Least bad families:

- `exp`
- `exp+log`

Closest safe-regime candidate:

- signature: `exp`
- median alpha: `1.8`
- median threshold gap: `0.2421141086977403`
- percent below current threshold: `0.0`

Conclusion:

No robust raw pure EML safe regime appears under the current threshold for this generated dataset.

## Goal 2.6: End-To-End Pipeline And Final Report

Goal 2.6 integrated all stages into one reproducible command.

Module:

- `geml/experiments/run_goal2_expansion_pipeline.py`

Command:

```bash
.venv/bin/python -m geml.experiments.run_goal2_expansion_pipeline --config configs/expansion_v0.yaml
```

Pipeline order:

1. generate expansion inputs
2. compute raw AST/official-EML metrics
3. compute alpha threshold summary
4. compute stratified analysis
5. generate plots
6. mine failure cases
7. write final report

Final report:

- `docs/GOAL2_EXPANSION_STUDY.md`

The final report includes:

- goal and scientific question
- official EML compiler source and attribution
- pure EML grammar
- why derived leaves are invalid for alpha
- dataset size and seed
- supported/unsupported count
- K and L values
- alpha-threshold formula
- threshold scenarios tested
- aggregate alpha summary
- percent below threshold
- stratified findings
- plot references
- top failure modes
- safe-regime candidates
- conclusion on raw pure EML tree expansion
- recommendation for Goal 3 and DAG compression

Test:

- `tests/test_goal2_expansion_pipeline.py`

The test runs the entire pipeline with `count=25` and verifies output files plus final report creation.

## Goal 2.7: Decision Memo

Goal 2.7 distilled the research decision before moving to the next phase.

Memo:

- `docs/GOAL2_DECISION_MEMO.md`

Central decision:

> Raw pure EML is representation-complete but structurally expensive; therefore, the next research step should test whether DAG compression and learned graph models recover enough structure to make EML useful.

The memo answers:

- raw pure EML is not smaller than AST
- alpha almost never falls below threshold
- worst families are Add/Mul-heavy
- least bad families are `exp` and `exp+log`
- this does not kill GEML
- DAG compression is now necessary
- AST-GNN and EML-DAG-GNN baselines are required

## Generated Artifact Inventory

Primary expansion outputs:

- `outputs/v0/expansion_inputs.jsonl`
- `outputs/v0/expansion_raw_metrics.jsonl`
- `outputs/v0/expansion_raw_metrics.csv`
- `outputs/v0/expansion_alpha_summary.csv`
- `outputs/v0/expansion_alpha_summary.json`
- `outputs/v0/official_eml_compiler_summary.json`
- `outputs/v0/official_eml_top20_alpha.json`
- `outputs/v0/official_eml_top20_depth.json`
- `outputs/v0/official_eml_simple_examples.json`

Stratified outputs:

- `outputs/v0/alpha_by_ast_depth.csv`
- `outputs/v0/alpha_by_ast_size_bucket.csv`
- `outputs/v0/alpha_by_operator_family.csv`
- `outputs/v0/alpha_by_operator_signature.csv`
- `outputs/v0/alpha_by_boolean_features.csv`

Plot outputs:

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

Top and failure-mining tables:

- `outputs/v0/top_20_alpha_expressions.csv`
- `outputs/v0/top_20_eml_node_expressions.csv`
- `outputs/v0/top_20_eml_depth_expressions.csv`
- `outputs/v0/top_alpha_explosions.csv`
- `outputs/v0/top_eml_node_explosions.csv`
- `outputs/v0/top_eml_depth_explosions.csv`
- `outputs/v0/worst_operator_signatures.csv`
- `outputs/v0/safest_operator_signatures.csv`
- `outputs/v0/depth_failure_modes.csv`
- `outputs/v0/safe_eml_regime_candidates.csv`

Reports and docs:

- `docs/goal2_representation_semantics.md`
- `docs/GOAL2_OFFICIAL_EML_COMPILER.md`
- `docs/GOAL2_EXPANSION_STUDY.md`
- `docs/GOAL2_DECISION_MEMO.md`
- `docs/GOAL2_SUMMARY.md`
- `outputs/v0/GOAL2_FAILURE_CASES.md`

## Implementation Inventory

Core symbolic/representation modules:

- `geml/symbolic/representations.py`
- `geml/symbolic/eml_nodes.py`
- `geml/symbolic/eml_transpile.py`
- `geml/symbolic/official_eml_compiler.py`
- `geml/symbolic/ast_graph.py`

Data and metrics modules:

- `geml/data/generate_exprs.py`
- `geml/data/dataset.py`

Goal 2 experiment modules:

- `geml/experiments/expansion_study.py`
- `geml/experiments/stratified_expansion.py`
- `geml/experiments/plot_expansion_study.py`
- `geml/experiments/expansion_failure_mining.py`
- `geml/experiments/run_goal2_expansion_pipeline.py`

Configuration:

- `configs/expansion_v0.yaml`

Dependency update:

- `matplotlib>=3.9` added for Goal 2.4 plots

## Test Coverage Added Or Extended

Key test files:

- `tests/test_eml_transpile.py`
- `tests/test_official_eml_compiler.py`
- `tests/test_dataset.py`
- `tests/test_expansion_study.py`
- `tests/test_stratified_expansion.py`
- `tests/test_plot_expansion_study.py`
- `tests/test_expansion_failure_mining.py`
- `tests/test_goal2_expansion_pipeline.py`
- `tests/test_imports.py`

Coverage includes:

- representation-mode validation
- hidden derived leaves not counted as alpha-valid leaves
- official pure compiler structural purity
- no final unsupported node labels in pure EML
- official-style EML string output
- numeric verification for supported expressions
- scale pipeline smoke tests
- threshold math and below-threshold classification
- operator counting and bucket assignment
- group summary math
- plot smoke generation
- failure-mining rankings and report generation
- small end-to-end Goal 2 pipeline

Latest verification after Goal 2.7:

```text
.venv/bin/python -m pytest -> 70 passed
.venv/bin/python -m ruff check . -> passed
.venv/bin/python -m ruff format . --check -> passed
```

## Scientific Interpretation

Goal 2 demonstrates that raw official pure EML trees are structurally expensive. The official compiler is correct and pure, but Add/Mul/Pow/log/exp source operations expand into many EML nodes because all operations are expressed through a single primitive. This creates large repeated macro patterns in tree form.

The important distinction is:

- raw pure EML tree size is poor
- compressed pure EML graph size remains untested

Therefore, Goal 2 narrows the research hypothesis. GEML should not claim that raw pure EML trees are compact. The next plausible claim is that EML may be useful after graph sharing and learning because the representation has a uniform primitive structure.

## Recommendation For The Next Research Step

Goal 3 should test DAG compression before any strong ML claims.

Minimum next comparisons:

- AST tree or AST DAG baseline
- pure EML tree baseline
- pure EML DAG after structural sharing
- AST-GNN baseline
- EML-DAG-GNN baseline

Why these are needed:

- AST-GNN is required to show whether ordinary symbolic graph learning already works
- EML-DAG-GNN is required because raw EML trees are too large
- DAG compression is required to test whether repeated EML macro structure can be shared enough to make EML competitive

The key next metric should be something like:

```text
alpha_dag = |DAG_EML| / |AST or DAG_AST|
```

Only after DAG compression is measured should the project move toward neural equivalence modeling or larger ML experiments.
