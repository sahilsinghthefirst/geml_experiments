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

## Restricted EML Trees

`geml.symbolic.eml_transpile.sympy_to_eml_tree` converts the initial supported expression subset into a rooted binary tree whose internal nodes are all `eml`, where:

```text
eml(x, y) = exp(x) - log(y)
```

The current restricted converter supports variables, constant `1`, `Add`, `Mul`, `exp`, and `log`. Direct rules are used for `exp` and `log`; `Add` and `Mul` use a restricted lift rule `E -> eml(log(E), 1)` so the EML evaluator simplifies back to the source expression. The converter reports EML tree statistics and `alpha = |T_EML| / |T_AST|`.

## Dataset Metrics Export

After generating expressions, export integrated AST/EML metrics with:

```bash
python -m geml.data.dataset --config configs/dataset_v0.yaml
```

This writes one JSONL row per expression plus a flattened CSV summary under `outputs/v0/`. Unsupported expressions are retained with `supported=false` and an error message.

## Goal 1 Sample Pipeline

Run the small end-to-end Goal 1 pipeline with:

```bash
python -m geml.experiments.goal1_sample
```

By default this:

- generates 100 expressions
- converts each expression to the normal AST binary tree
- converts supported expressions to the restricted EML binary tree
- computes `alpha = |T_EML| / |T_AST|`
- writes `outputs/v0/goal1_sample.jsonl`
- writes `outputs/v0/goal1_summary.csv`

Optional arguments:

```bash
python -m geml.experiments.goal1_sample --count 100 --seed 0 --max-depth 4
```
