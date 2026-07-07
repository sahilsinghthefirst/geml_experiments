# Goal 1 Completion Report

This document summarizes the technical state of GEML-v0 Goal 1 after the completed implementation stages through commit `737ba6c`.

## Goal 1 Scope

Goal 1 established the core representation and data-generation foundation for GEML-v0:

- Generate bounded-depth SymPy expressions.
- Convert expressions into normal AST binary trees.
- Convert supported expressions into restricted EML binary trees.
- Compute tree statistics and alpha.
- Export JSONL/CSV datasets and metrics.
- Provide a small end-to-end sample pipeline.
- Add tests for the implemented components.

Neural models, DAG conversion, trig support, training loops, equivalence-pair generation, and proof/rewrite modeling were intentionally not implemented in Goal 1.

## Repository And Tooling

Implemented files:

- `pyproject.toml`
- `.gitignore`
- `README.md`
- `AGENTS.md`
- `configs/data_v0.yaml`
- `configs/dataset_v0.yaml`
- `configs/expansion_v0.yaml`
- `configs/equiv_ast.yaml`
- `configs/equiv_eml.yaml`
- `outputs/.gitkeep`
- `outputs/v0/.gitkeep`

Tooling configured:

- Python package metadata for `geml-experiments`
- Python requirement: `>=3.12`
- Runtime dependencies:
  - `pydantic`
  - `pyyaml`
  - `sympy`
- Dev dependencies:
  - `pytest`
  - `ruff`
- Pytest config:
  - strict config
  - strict markers
  - `tests/` as test root
- Ruff config:
  - Python 3.12 target
  - line length 100
  - lint families: `E`, `F`, `I`, `B`, `UP`, `ANN`

## Expression Generator

Implemented in:

- `geml/data/generate_exprs.py`
- `tests/test_generate_exprs.py`

Core types and functions:

- `ExpressionGeneratorConfig`
- `GeneratedExpression`
- `SympyExpressionGenerator`
- `expression_depth`
- `load_config`
- `write_jsonl`
- `write_csv`
- `generate_dataset`
- CLI entry point: `python -m geml.data.generate_exprs --config configs/data_v0.yaml`

Supported expression subset:

- Symbols: `x`, `y`
- Constant: `1`
- Operators:
  - `Add`
  - `Mul`
  - `exp`
  - `log`

Implemented generator behavior:

- Deterministic generation with a configurable seed.
- Configurable expression count.
- Configurable maximum depth.
- Configurable operator probabilities.
- Bounded-depth recursive generation.
- SymPy object construction instead of string construction.
- `evaluate=False` for generated operator nodes where applicable.
- Log-argument guard:
  - terminal log argument falls back to `1`
  - non-terminal log argument is generated as `exp(...)`
- JSONL and CSV export.

Generated row fields:

- `index`
- `expression`
- `srepr`
- `depth`
- `metadata`

Tested behavior:

- Reproducibility with fixed seed.
- Generated depth bounds.
- Valid SymPy parsing.
- JSONL/CSV output schema.

## Normal AST Binary-Tree Converter

Implemented in:

- `geml/symbolic/ast_graph.py`
- `tests/test_ast_graph.py`

Core types and functions:

- `UnsupportedExpressionError`
- `AstNode`
- `AstEdge`
- `AstTree`
- `sympy_to_ast_tree`

Supported AST node types:

- Symbols
- Integer constants
- `Add`
- `Mul`
- `Pow`
- `exp`
- `log`

AST output structure:

- `nodes`
- `edges`
- `root_id`
- `node_labels`
- `metadata`
- `statistics`

AST node kinds:

- `symbol`
- `constant`
- `operator`

Binary-tree handling:

- `Add` with more than two arguments is normalized into deterministic left-associative binary operator nodes.
- `Mul` with more than two arguments is normalized into deterministic left-associative binary operator nodes.
- Unary operators `exp` and `log` keep one child.
- `Pow` keeps two children.

Tested expressions:

- `x + 1`
- `x * y`
- `exp(x)`
- `log(x + 1)`
- `(x + 1) * (y + 1)`
- n-ary `Add`
- `Pow(x + 1, 2)`

## Tree Statistics

Implemented in:

- `geml/symbolic/metrics.py`

Core types and functions:

- `TreeStatistics`
- `compute_tree_statistics`

Computed statistics:

- `node_count`
- `edge_count`
- `depth`
- `leaf_count`
- `operator_count`

Validation added:

- root id must exist
- all edge endpoints must exist
- root must have no parent
- every non-root node must have exactly one parent
- every node must be reachable from the root
- repeated paths/cycles are rejected

Depth convention:

- leaf nodes have depth `0`

## Restricted EML Binary-Tree Converter

Implemented in:

- `geml/symbolic/eml_nodes.py`
- `geml/symbolic/eml_transpile.py`
- `tests/test_eml_transpile.py`

Core types:

- `EmlNode`
- `EmlEdge`
- `EmlTree`
- `UnsupportedExpressionError`

Core functions:

- `eml_operator`
- `sympy_to_eml_tree`
- `evaluate_eml_tree`
- `simplify_eml_tree`
- `eml_alpha`

EML operator:

```text
eml(x, y) = exp(x) - log(y)
```

Supported official pure EML input subset:

- variables
- integer constants
- rational constants
- floats through decimal-string rationalization
- constant `1`
- `Add`
- `Mul`
- `Pow`
- `exp`
- `log`

Diagnostic derived mode additionally accepts `Add` and `Mul` through the old lift rule, but those trees are not valid for alpha.

Unsupported in the restricted EML converter:

- trig functions
- unsupported SymPy node types

EML output structure:

- `nodes`
- `edges`
- `root_id`
- `node_labels`
- `metadata`
- `statistics`
- `normal_leaf_count`
- `derived_leaf_count`
- `hidden_compound_leaf_count`
- `ast_statistics`
- `alpha`
- `alpha_valid`

EML node kinds:

- `eml`
- `variable`
- `constant`
- `derived`

Implemented restricted rules:

- `x -> x`
- `y -> y`
- `1 -> 1`
- official recursive macro definitions ported from `VA00/SymbolicRegressionPackage/EML_toolkit/EmL_compiler/eml_compiler_v4.py`
- `exp(a) -> EML(a, 1)`
- `log(a) -> EML(1, EML(EML(1, a), 1))`
- `Add`, `Mul`, `Pow`, subtraction, division, inverse, negation, integers, and rationals expand through official recursive macros.
- `restricted_eml_with_derived` can still use the diagnostic lift rule:

```text
E -> eml(log(E), 1)
```

Alpha computation:

```text
alpha = |T_EML| / |T_AST|
```

where `|T_*|` is the tree node count.

Alpha is valid only when `representation_mode=restricted_eml_pure`, `alpha_valid=true`, and there are no derived leaves.

Tested behavior:

- variable leaf conversion
- constant `1` conversion
- `exp(x)`
- `log(x)`
- official pure conversion of `x + 1`
- official pure conversion of `x * y`
- official pure conversion of `log(x + 1)`
- official pure conversion of `Pow`
- official pure conversion of integer constants other than `1`
- derived-mode classification of `x + 1`
- derived-mode classification of `(x + 1) * (y + 1)`
- valid pure alpha ratio calculation
- derived-mode alpha rejection
- unsupported `sin(x)`
- hidden compound derived leaves are not counted as normal EML leaves
- all internal nodes with children are `eml`
- converted structures are connected trees

## Dataset Metrics Export

Implemented in:

- `geml/data/dataset.py`
- `configs/dataset_v0.yaml`
- `tests/test_dataset.py`

Core types and functions:

- `DatasetExportConfig`
- `GeneratedExpressionInput`
- `DatasetMetricsRow`
- `load_generated_expressions`
- `compute_metrics_rows`
- `write_metrics_jsonl`
- `write_metrics_csv`
- `export_dataset_metrics`
- CLI entry point: `python -m geml.data.dataset --config configs/dataset_v0.yaml`

Input:

- generated expression JSONL, default `outputs/v0/dataset.jsonl`

JSONL output:

- default `outputs/v0/dataset_metrics.jsonl`

CSV summary output:

- default `outputs/v0/dataset_metrics.csv`

Per-expression JSONL fields:

- `index`
- `expression`
- `srepr`
- `representation_mode`
- `ast_stats`
- `eml_stats`
- `eml_normal_leaf_count`
- `eml_derived_leaf_count`
- `eml_hidden_compound_leaf_count`
- `alpha`
- `alpha_valid`
- `supported`
- `error`
- `metadata`

CSV summary fields:

- `index`
- `expression`
- `srepr`
- `representation_mode`
- `supported`
- `error`
- `ast_node_count`
- `ast_edge_count`
- `ast_depth`
- `ast_leaf_count`
- `ast_operator_count`
- `eml_node_count`
- `eml_edge_count`
- `eml_depth`
- `eml_leaf_count`
- `eml_normal_leaf_count`
- `eml_derived_leaf_count`
- `eml_hidden_compound_leaf_count`
- `eml_operator_count`
- `alpha`
- `alpha_valid`

Unsupported-row behavior:

- If parsing fails, both AST and EML stats are empty.
- If AST conversion fails, both AST and EML stats are empty.
- If AST conversion succeeds but EML conversion fails, AST stats are retained and EML stats/alpha are empty.
- If derived-mode conversion succeeds with hidden compound leaves, EML stats are retained but alpha is empty and `alpha_valid=false`.
- Unsupported rows are retained with `supported=false` and an error message.

## Goal 1 Sample Pipeline

Implemented in:

- `geml/experiments/goal1_sample.py`
- `tests/test_goal1_sample.py`

CLI:

```bash
python -m geml.experiments.goal1_sample
```

Default behavior:

- generate 100 expressions
- convert each expression to AST
- convert supported expressions to restricted EML
- compute alpha
- write `outputs/v0/goal1_sample.jsonl`
- write `outputs/v0/goal1_summary.csv`

Configurable CLI arguments:

- `--count`
- `--seed`
- `--max-depth`
- `--output-jsonl`
- `--output-csv`

## Tests

Implemented test files:

- `tests/test_imports.py`
- `tests/test_generate_exprs.py`
- `tests/test_ast_graph.py`
- `tests/test_eml_transpile.py`
- `tests/test_dataset.py`
- `tests/test_goal1_sample.py`

Latest verification command:

```bash
.venv/bin/python -m pytest
```

Latest result:

```text
26 passed
```

Latest lint command:

```bash
.venv/bin/python -m ruff check .
```

Latest result:

```text
All checks passed
```

Latest format command:

```bash
.venv/bin/python -m ruff format . --check
```

Latest result:

```text
29 files already formatted
```

## Generated Artifacts

The following artifacts are produced locally under `outputs/v0/`:

- `dataset.jsonl`
- `dataset.csv`
- `dataset_metrics.jsonl`
- `dataset_metrics.csv`
- `goal1_sample.jsonl`
- `goal1_summary.csv`

These output files are ignored by git. Only `.gitkeep` files are versioned under `outputs/`.

## How To Run Goal 1

Install the project in editable mode with dev dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
python -m ruff check .
python -m ruff format . --check
```

Generate the main expression dataset:

```bash
python -m geml.data.generate_exprs --config configs/data_v0.yaml
```

Export AST/EML metrics for the generated dataset:

```bash
python -m geml.data.dataset --config configs/dataset_v0.yaml
```

Run the 100-expression sample pipeline:

```bash
python -m geml.experiments.goal1_sample
```

## Known Uncertainties And Potential Issues

### Restricted EML Add/Mul Encoding Is Classified And Superseded

Goal 2.0 initially resolved the largest technical uncertainty in the Goal 1 converter by splitting EML conversion into pure and derived modes.

The old diagnostic rule:

```text
E -> eml(log(E), 1)
```

keeps every internal tree node as `eml`, but it introduces a `derived` leaf labeled `log(expr)` that can contain a compound expression. This is not a valid pure EML representation because pure leaves must be only variables and constant `1`.

Current decision:

- Goal 2.1b ports the official recursive pure compiler from `VA00/SymbolicRegressionPackage/EML_toolkit/EmL_compiler/eml_compiler_v4.py`.
- `restricted_eml_pure` now compiles `Add`, `Mul`, `Pow`, numeric constants, `exp`, and `log` through official pure recursive macros.
- `restricted_eml_with_derived` keeps the old lift only for diagnostic inspection.
- Derived hidden compound leaves are counted separately and are not normal EML leaves.
- Alpha is empty and `alpha_valid=false` for derived-mode trees.
- Goal 2 serious alpha plots must use only `restricted_eml_pure` rows with `alpha_valid=true`.

The locked semantics are documented in `docs/goal2/goal2_representation_semantics.md` and the official compiler port is documented in `docs/goal2/GOAL2_OFFICIAL_EML_COMPILER.md`.

### EML Simplification Uses Formal Inverse Rules

`simplify_eml_tree` uses:

```python
sympy.simplify(..., inverse=True)
sympy.expand_log(..., force=True)
```

This is formal symbolic simplification. It can ignore domain restrictions that matter over real or complex domains. For example, `log(exp(x)) = x` is not generally valid over the full complex plane without assumptions.

Consequences:

- EML equivalence tests are formal algebraic checks, not domain-safe mathematical proofs.
- Later experiment claims should state the assumed symbolic domain.

### Generated Expression String Is Not Structural Serialization

Generated rows store both `expression` and `srepr`. The human-readable `expression` string is convenient, but reparsing it with SymPy can flatten or reorder `Add` and `Mul`.

Observed consequence:

- The parsed expression may be semantically equivalent but structurally different from the original generated tree.
- AST stats and alpha computed from reparsed strings may differ from stats computed from the original generated object.

Mitigation implemented in Goal 2.1:

- Metrics export now prefers `srepr` as the authoritative structural input and falls back to `expression` only when no `srepr` is available.

### Python Version Mismatch In Local Verification

The project targets Python 3.12. The local verification environment used `.venv` backed by Python 3.14.2.

Consequences:

- Tests pass locally, but CI or future runs should verify Python 3.12 specifically.
- Any Python 3.14-specific behavior should be avoided.

### Remaining EML Support Gaps

Goal 2.1b added official pure EML support for `Pow`, arithmetic, and numeric constants.

Remaining unsupported cases include:

- trigonometric functions
- inverse trigonometric functions
- hyperbolic functions
- `Abs`
- arbitrary unsupported SymPy nodes

### Log-Argument Validity Is Structural, Not Domain-Proven

The generator avoids obviously invalid log arguments by using `1` or `exp(...)`, but this is a structural guard, not a complete mathematical domain analysis.

Consequences:

- generated expressions are reasonable for formal symbolic experiments
- they should not be treated as fully domain-certified real-valued expressions

### No DAG Or Shared-Subexpression Compression Yet

All statistics currently assume tree structures.

Consequences:

- repeated subexpressions are duplicated
- no AST-DAG or EML-DAG compression is measured
- DAG compression remains a later goal

### Goal 1 Does Not Yet Generate Equivalence Pairs

The broader Goal 1 roadmap mentions expression pairs, but the implemented pipeline currently generates single-expression rows with AST/EML metrics.

Consequences:

- equivalence/non-equivalence pair generation remains future work
- model training datasets are not ready

### Large-Scale Generation Success Criterion Not Fully Exercised

The project brief mentions generating 10,000+ expressions. The current checked local artifacts include:

- 1,000-expression generated dataset from `configs/data_v0.yaml`
- 100-expression sample pipeline output

The code is configurable for larger counts, but the 10,000+ scale has not been verified in this stage.

### Output Artifacts Are Not Versioned

Generated outputs are ignored by git.

Consequences:

- reproducibility depends on configs, code, and seed
- output files must be regenerated locally or in CI

## Recommended Next Technical Fixes

Before moving to serious experiments:

1. Make `srepr` or another structural format the authoritative dataset representation.
2. Run verification on Python 3.12.
3. Scale-test generation and metrics export at 10,000+ expressions.
4. Add equivalence-pair generation only after representation semantics are locked.
