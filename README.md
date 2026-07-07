# GEML Experiments

GEML is an experimental benchmark for testing when EML-native symbolic
representations help or hurt mathematical reasoning.

The current work compares standard expression ASTs against restricted official
pure EML trees and exact structural DAGs. The main measured tradeoff is whether
EML's reduced operator vocabulary compensates for its larger structural size.

## Current Status

- Goal 1: expression generation, AST conversion, restricted EML conversion,
  metrics export, and a small sample pipeline.
- Goal 2: official pure EML expansion study on the fixed seed-0 v0 corpus.
- Goal 3: exact structural DAG compression for AST and official pure EML trees.
- Goal 3R: repaired expression generation and rerun Goal 2/3 studies on a
  stronger v1 corpus.
- Goal 4: non-ML e-graph compression baseline work must use v1 for any
  result-bearing run.
- Goal 5 and Goal 6 ML-facing compression/GNN work must use v1.

Goal 3R does not implement e-graphs or neural models.

## Baseline Corpus Policy

`outputs/v1` is the default corpus for all future compression and ML
experiments. It is the only corpus that should be used for new result claims.

`outputs/v0` is now a pilot corpus and is deprecated for result-bearing
analysis. Existing v0 Goal 2/3 reports remain useful for historical comparison
and diagnostics, but v0 should not be used as the default baseline for Goal 4,
Goal 5, or Goal 6. Any e-graph result computed on v0 is diagnostic only.

Reserved future subset labels:

- `all_v1`: the full repaired v1 corpus.
- `nontrivial_v1`: future low-triviality v1 subset.
- `identity_heavy_v1`: future diagnostic v1 subset enriched for identity
  simplification opportunities.

## Setup

Use the repo-local virtual environment:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format . --check
```

## Expression Generation

Generate the stronger Goal 3R v1 corpus with:

```bash
.venv/bin/python -m geml.data.generate_exprs --config configs/expression_v1.yaml
```

The v1 generator adds:

- target-depth sampling instead of forcing every expression to exact max depth
- intermediate leaf probability
- positive-domain log argument grammar without blanket `exp(...)` wrapping
- srepr-based deduplication with rejection sampling
- duplicate-rate, depth-histogram, log-argument, and triviality reporting
- optional triviality capping without eliminating all trivial expressions

The generator supports `x`, `y`, `1`, `Add`, `Mul`, `Pow`, `exp`, and `log`
across parsing and downstream metrics. The random generator operator set remains
configurable through YAML.

## AST Binary Trees

`geml.symbolic.ast_graph.sympy_to_ast_tree` converts supported SymPy expressions into a serializable rooted tree with nodes, directed edges, root id, node labels, metadata, and structural statistics.

Supported AST nodes are symbols, integer constants, `Add`, `Mul`, `Pow`, `exp`, and `log`. N-ary `Add` and `Mul` nodes are normalized into deterministic binary operator trees.

## Representation Modes

All exported representations use an explicit `representation_mode`:

- `ast`: normal binary AST representation.
- `restricted_eml_pure`: official recursive pure EML tree used for valid alpha measurements.
- `restricted_eml_with_derived`: diagnostic EML tree that may contain derived leaves.

Goal 2 expansion-factor work must use only rows with `representation_mode=restricted_eml_pure` and `alpha_valid=true` for serious alpha plots.

## Restricted EML Trees

`geml.symbolic.eml_transpile.sympy_to_eml_tree` converts the initial supported expression subset into a rooted binary tree whose internal nodes are all `eml`, where:

```text
eml(x, y) = exp(x) - log(y)
```

The pure restricted EML representation grammar is:

```text
P ::= variable | 1 | eml(P, P)
```

Only variables and constant `1` are normal EML leaves. `restricted_eml_pure` now uses the official recursive compiler port from `VA00/SymbolicRegressionPackage/EML_toolkit/EmL_compiler/eml_compiler_v4.py`, supporting variables, integers, rationals, floats, `Add`, `Mul`, `Pow`, `exp`, and `log` through pure EML macro expansion.

`restricted_eml_with_derived` preserves the previous diagnostic lift rule `E -> eml(log(E), 1)` for `Add` and `Mul`, but the `log(E)` child is a `derived` leaf that can hide a compound expression. Such leaves are counted separately as derived/hidden leaves, not normal EML leaves, and `alpha` is `null` with `alpha_valid=false`.

## Shared Srepr Parsing

Generated corpora use SymPy `srepr` as the authoritative structural
serialization. Shared parsing lives in `geml.symbolic.srepr` and reconstructs
`Add`, `Mul`, `Pow`, `exp`, and `log` with `evaluate=False` so Goal 2 and Goal 3
measure the same structure.

## Dataset Metrics Export

After generating expressions, export integrated AST/EML metrics with:

```bash
.venv/bin/python -m geml.data.dataset --config configs/dataset_v0.yaml
```

This writes one JSONL row per expression plus a flattened CSV summary under `outputs/v0/`. Unsupported expressions are retained with `supported=false` and an error message. The default export mode is `restricted_eml_pure`; pass `--representation-mode restricted_eml_with_derived` only for diagnostic inspection of derived leaves.

Metrics export prefers `srepr` as the authoritative structural input and falls back to the human-readable `expression` string only when no `srepr` is available. Output rows include `source_serialization`.

## Goal 2 Expansion Pipeline

Run the repaired v1 study with:

```bash
.venv/bin/python -m geml.experiments.run_goal2_expansion_pipeline --config configs/expansion_v1.yaml
```

The original v0 study is retained as a pilot/deprecated diagnostic:

```bash
.venv/bin/python -m geml.experiments.run_goal2_expansion_pipeline --config configs/expansion_v0.yaml
```

The pipeline generates the configured expression set, computes AST and official
pure EML metrics, computes alpha-threshold summaries, runs stratified analysis,
writes plots, mines failure cases, and writes Goal 2 reports.

The component raw expansion-factor scale pipeline can be run with:

```bash
.venv/bin/python -m geml.experiments.expansion_study --config configs/expansion_v1.yaml
```

Both v0 and v1 compute the Goal 2 threshold:

```text
alpha_threshold = 1 + log(K) / log(4L)
```

Additional Goal 2 component commands use their config defaults unless given
explicit paths:

```bash
.venv/bin/python -m geml.experiments.stratified_expansion --config configs/expansion_v1.yaml
.venv/bin/python -m geml.experiments.plot_expansion_study
.venv/bin/python -m geml.experiments.expansion_failure_mining
```

## Goal 3 DAG Compression Pipeline

Run the repaired v1 DAG study with:

```bash
.venv/bin/python -m geml.experiments.run_goal3_dag_pipeline --config configs/dag_compression_v1.yaml
```

The original v0 DAG study is retained as a pilot/deprecated diagnostic:

```bash
.venv/bin/python -m geml.experiments.run_goal3_dag_pipeline --config configs/dag_compression_v0.yaml
```

This computes AST tree, AST DAG, official pure EML tree, and official pure EML
DAG metrics, writes DAG threshold summaries and stratified analyses, generates
plots, mines compression successes/failures, and runs the semantic audit.

Goal 3 DAG sharing is exact structural sharing only. It does not introduce derived
leaves, macro/template nodes, parameterized sharing, algebraic simplification, or
pattern sharing with holes.

Additional Goal 3 component commands:

```bash
.venv/bin/python -m geml.experiments.dag_compression_study --config configs/dag_compression_v1.yaml
.venv/bin/python -m geml.experiments.stratified_dag_compression --config configs/dag_compression_v1.yaml
.venv/bin/python -m geml.experiments.plot_dag_compression
.venv/bin/python -m geml.experiments.dag_compression_mining
.venv/bin/python -m geml.experiments.dag_semantic_audit
```

## Goal 3R V1 Corpus Repair

Primary artifacts:

- `configs/expression_v1.yaml`
- `configs/expansion_v1.yaml`
- `configs/dag_compression_v1.yaml`
- `outputs/v1/expression_generation_summary.json`
- `outputs/v1/expansion_generation_summary.json`
- `outputs/v1/official_eml_compiler_summary.json`
- `outputs/v1/dag_compression_summary.json`
- `outputs/v1/v0_v1_comparison_summary.json`
- `docs/goal3/GOAL3R_V1_CORPUS_COMPARISON.md`

Generated JSONL, CSV, and PNG artifacts under `outputs/` are ignored by git, but
the local pipeline writes them reproducibly.

## Goal 4 Baseline Switch

Goal 4 e-graph compression must use v1 inputs by default. The current future
experiment placeholders, `configs/equiv_ast.yaml` and `configs/equiv_eml.yaml`,
point at `outputs/v1` and select `all_v1`. V0 e-graph runs may be useful while
debugging the engine, but they are diagnostic only.

## Goal 4 E-Graph Compression Pipeline

Run the complete v1 non-ML e-graph compression study with:

```bash
.venv/bin/python -m geml.experiments.run_goal4_egraph_pipeline --config configs/egraph_compression_v1.yaml
```

This loads the v1 Goal 3 exact EML-DAG baseline, runs or loads e-graph
compression in `safe` and `positive_real_formal` modes, regenerates stratified
analysis, plots, success/failure mining, semantic/provenance audit, and writes
the final Goal 4 reports.

Component commands:

```bash
.venv/bin/python -m geml.experiments.egraph_compression_study --config configs/egraph_compression_v1.yaml
.venv/bin/python -m geml.experiments.stratified_egraph_compression
.venv/bin/python -m geml.experiments.plot_egraph_compression
.venv/bin/python -m geml.experiments.egraph_compression_mining
.venv/bin/python -m geml.experiments.egraph_semantic_audit
```

Headline v1 findings from the current 10k artifacts:

- `safe`: 10,000 processed, 9,316 successful extractions, 241 timeouts, 607
  validation failures, median optimized EML-DAG alpha 3.636, median compression
  gain 1.045, and threshold pass rate 1.020% after e-graph extraction.
- `positive_real_formal`: 10,000 processed, 8,941 successful extractions, 522
  timeouts, 878 validation failures, median optimized EML-DAG alpha 3.364,
  median compression gain 1.169, and threshold pass rate 5.827% after e-graph
  extraction.
- `identity_heavy_v1` gains are much larger than `nontrivial_v1` gains, so the
  subset split is required to avoid overstating easy identity simplifications.
- Final extracted outputs still compile through the official pure EML compiler
  and remain pure EML DAGs. These are structural non-ML compression results, not
  neural model or GNN evidence.

## Goal 1 Sample Pipeline

Run the small end-to-end Goal 1 pipeline with:

```bash
.venv/bin/python -m geml.experiments.goal1_sample
```

By default this:

- generates 100 expressions
- converts each expression to the normal AST binary tree
- converts supported expressions to the selected restricted EML binary tree
- computes `alpha = |T_EML| / |T_AST|` only when `alpha_valid=true`
- writes `outputs/v0/goal1_sample.jsonl`
- writes `outputs/v0/goal1_summary.csv`

Optional arguments:

```bash
.venv/bin/python -m geml.experiments.goal1_sample --count 100 --seed 0 --max-depth 4
```

## Documentation

- Goal 1 docs: `docs/goal1/`
- Goal 2 docs: `docs/goal2/`
- Goal 3 docs: `docs/goal3/`
- Goal 4 docs: `docs/goal4/`

See `docs/goal2/GOAL2_OFFICIAL_EML_COMPILER.md` for the official compiler port,
`docs/goal2/goal2_representation_semantics.md` for representation-mode policy,
`docs/goal3/GOAL3_DAG_SEMANTICS.md` for exact DAG semantics, and
`docs/goal4/GOAL4_NONML_COMPRESSION_STUDY.md` for the final Goal 4 e-graph
compression report.
