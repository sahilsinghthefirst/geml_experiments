# GEML-v0

GEML-v0 is an experimental benchmark for testing when EML-native graph representations help or hurt symbolic mathematical reasoning.

The project compares standard expression representations against restricted EML representations while measuring the tradeoff between reduced operator diversity and increased tree size.

## Goal 1 Scaffold

This repository is currently scaffolded for Goal 1: core expression representation and data generation.

The Goal 1 implementation includes:

- Python 3.12 package metadata
- pytest configuration
- ruff linting and formatting configuration
- YAML configs
- package directories for data generation, symbolic graph conversion, models, training, and experiments
- JSONL/CSV output directories under `outputs/v0/`
- tests covering imports, generation, AST conversion, EML conversion, dataset export, and the sample pipeline

Expression generation, AST conversion, restricted EML conversion, metrics, and dataset export are implemented. Model logic will be implemented in later goals.

## Expression Generation

Generate the initial Goal 1 expression dataset with:

```bash
python -m geml.data.generate_exprs --config configs/data_v0.yaml
```

The generator supports `x`, `y`, `1`, `Add`, `Mul`, `Exp`, and `Log`, with configurable depth, seed, expression count, and operator probabilities. It writes JSONL and CSV outputs under `outputs/v0/` by default.

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

## Dataset Metrics Export

After generating expressions, export integrated AST/EML metrics with:

```bash
python -m geml.data.dataset --config configs/dataset_v0.yaml
```

This writes one JSONL row per expression plus a flattened CSV summary under `outputs/v0/`. Unsupported expressions are retained with `supported=false` and an error message. The default export mode is `restricted_eml_pure`; pass `--representation-mode restricted_eml_with_derived` only for diagnostic inspection of derived leaves.

Metrics export prefers `srepr` as the authoritative structural input and falls back to the human-readable `expression` string only when no `srepr` is available. Output rows include `source_serialization`.

## Goal 2 Expansion Scale Pipeline

Run the complete Goal 2 expansion-factor study with:

```bash
python -m geml.experiments.run_goal2_expansion_pipeline --config configs/expansion_v0.yaml
```

This generates the fixed-seed expression set, computes AST and official pure EML metrics, computes alpha-threshold summaries, runs stratified analysis, writes plots, mines failure cases, and refreshes `docs/GOAL2_EXPANSION_STUDY.md`.

The component raw expansion-factor scale pipeline can be run with:

```bash
python -m geml.experiments.expansion_study --config configs/expansion_v0.yaml
```

The default config generates 10,000 expressions with fixed seed `0`, writes `outputs/v0/expansion_inputs.jsonl`, exports raw metrics to `outputs/v0/expansion_raw_metrics.jsonl` and `outputs/v0/expansion_raw_metrics.csv`, and writes official compiler audit JSON files for the run summary, top 20 alpha expressions, top 20 deepest EML trees, and simple expression examples. It also computes the Goal 2.2 threshold

```text
alpha_threshold = 1 + log(K) / log(4L)
```

for configurable `K` and `L` values, annotates raw metric rows with `alpha_threshold` and `below_threshold`, and writes:

- `outputs/v0/expansion_alpha_summary.csv`
- `outputs/v0/expansion_alpha_summary.json`

The included threshold scenarios are `current_grammar` with `K=4, L=3`, `generous_operator_vocab` with `K=20, L=3`, and `larger_operator_vocab` with `K=50, L=3`. This step intentionally does not generate plots.

Additional Goal 2 component commands:

```bash
python -m geml.experiments.stratified_expansion --config configs/expansion_v0.yaml
python -m geml.experiments.plot_expansion_study
python -m geml.experiments.expansion_failure_mining
```

## Goal 1 Sample Pipeline

Run the small end-to-end Goal 1 pipeline with:

```bash
python -m geml.experiments.goal1_sample
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
python -m geml.experiments.goal1_sample --count 100 --seed 0 --max-depth 4
```

See `docs/GOAL2_OFFICIAL_EML_COMPILER.md` for the official compiler port and `docs/goal2_representation_semantics.md` for the representation-mode policy.
