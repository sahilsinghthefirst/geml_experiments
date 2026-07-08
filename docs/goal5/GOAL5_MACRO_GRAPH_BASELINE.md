# Goal 5.1 Macro Graph Baseline

Goal 5.1 introduces an ML-facing compression baseline that exposes the official
compiler's macro structure as transparent graph nodes.

This representation is not pure EML. It is a compressed, auditable macro graph
whose nodes record official compiler macro names and whose expansion rules point
back to the official pure EML compiler implementation.

## Representation Contract

- Representation mode: `macro_graph_v1`
- Pure EML status: `is_pure_eml=false`
- Expansion status: every macro node has `expansion_to_pure_eml_available=true`
- Expansion target: `restricted_eml_pure`
- Expansion authority: `geml.symbolic.official_eml_compiler`
- Size policy: macro graph node counts are reported separately from Goal 3 pure
  EML tree and DAG metrics
- Scope: v1 only, using `outputs/v1/dag_compression_inputs.jsonl` and
  `outputs/v1/dag_compression_metrics.csv`

Implemented macro nodes include `eml_exp`, `eml_log`, `eml_add`, `eml_mul`,
`eml_sub`, `eml_neg`, `eml_inv`, `eml_div`, `eml_pow`, `eml_zero`,
`eml_variable`, `eml_integer`, and `eml_rational`.

Each node records:

- `macro_name`
- `arity`
- ordered `input_slots`
- `source_subtree_id`
- source expression and `srepr`
- `expansion_rule_name`
- `expansion_to_pure_eml_available`
- `pure_eml_expansion_node_count`
- `pure_eml_expansion_dag_node_count`

## Integrity Boundary

Macro graph construction mirrors the official compiler dispatch:

- `exp(a)` uses `official_eml_compiler.eml_exp`
- `log(a)` uses `official_eml_compiler.eml_log`
- `Add` terms use the official ordering and subtraction detection
- `Mul` factors use the official division detection
- `Pow` uses the official `eml_pow(base, exponent)` path
- integer and rational leaves expand through `eml_int` and `eml_rational`

No official pure EML compiler formulas are modified. Expansion validation
compares the macro graph expansion against the official compiler's pure EML
string. The macro graph is a compressed feature representation, not a substitute
pure EML DAG or pure EML alpha.

## Outputs

The v1 run writes:

- `outputs/v1/goal5_macro_graph_metrics.csv`
- `outputs/v1/goal5_macro_graph_metrics.jsonl`
- `outputs/v1/goal5_macro_graph_summary.json`

Per-expression rows include the required source AST counts, Goal 3 EML tree/DAG
counts, macro graph node/reference/depth counts, macro graph alpha ratios,
compression gain against the Goal 3 EML-DAG baseline, and expansion validation
flags.

The summary reports processed count, success count, expansion validation
failures, median macro graph alpha, median compression gain against Goal 3
EML-DAG, required subset summaries for `all_v1`, `nontrivial_v1`, and
`identity_heavy_v1`, plus operator-family summaries.

## V1 Run Summary

The current v1 artifact run completed with:

- Processed expressions: 10,000
- Successful macro expansions: 10,000
- Expansion validation failures: 0
- Median macro graph alpha vs AST tree: 0.7777777777777778
- Median compression gain vs Goal 3 EML-DAG: 5.25

Subset medians:

| subset | processed | success | median macro alpha | median gain vs Goal 3 EML-DAG |
| --- | ---: | ---: | ---: | ---: |
| `all_v1` | 10,000 | 10,000 | 0.7777777777777778 | 5.25 |
| `nontrivial_v1` | 3,666 | 3,666 | 0.7777777777777778 | 5.375 |
| `identity_heavy_v1` | 6,334 | 6,334 | 0.7777777777777778 | 5.166666666666667 |

## Reproducibility

Run:

```bash
.venv/bin/python -m geml.experiments.goal5_macro_graph_baseline --config configs/macro_graph_v1.yaml
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format . --check
```

This baseline does not train GNNs and does not make downstream model-performance
claims.
