# Goal 3 DAG Semantic Audit

This audit checks that exact structural DAG compression does not reintroduce hidden complexity into official pure EML representations.

The audit is structural first: it verifies that DAG sharing is exact subtree sharing, not macro creation, derived leaves, parameterized templates, or algebraic simplification.

## Scope

- AST and EML trees are built with the existing converters.
- AST and EML DAGs are built with exact structural hashing.
- Numeric checks compare the original SymPy expression, the official pure EML tree, and the pure EML DAG on safe positive real inputs.
- No official compiler formulas are changed by this audit.

## Summary

- Expressions audited: `12`
- Structurally valid EML DAGs: `12`
- Numerically valid EML DAGs: `12`
- JSON output: `outputs/v0/goal3_dag_semantic_audit.json`
- CSV output: `outputs/v0/goal3_dag_semantic_audit.csv`

## Audit Table

| Expression | AST tree | AST DAG | EML tree | EML DAG | Tree alpha | DAG alpha | EML DAG compression | Shared records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `x+y` | 3 | 3 | 27 | 16 | 9 | 5.33333 | 1.6875 | 1 |
| `x*y` | 3 | 3 | 41 | 23 | 13.6667 | 7.66667 | 1.78261 | 1 |
| `log(x)` | 2 | 2 | 7 | 5 | 3.5 | 2.5 | 1.4 | 1 |
| `exp(x)` | 2 | 2 | 3 | 3 | 1.5 | 1.5 | 1 | 0 |
| `x**2` | 3 | 3 | 75 | 29 | 25 | 9.66667 | 2.58621 | 7 |
| `x+x` | 3 | 2 | 27 | 15 | 9 | 5 | 1.8 | 2 |
| `x*x` | 3 | 2 | 41 | 19 | 13.6667 | 6.33333 | 2.15789 | 5 |
| `(x+1)*(x+1)` | 7 | 4 | 93 | 25 | 13.2857 | 3.57143 | 3.72 | 10 |
| `(x*x)*(x*x)` | 7 | 3 | 121 | 30 | 17.2857 | 4.28571 | 4.03333 | 10 |
| `log(x)+log(x)` | 5 | 3 | 39 | 18 | 7.8 | 3.6 | 2.16667 | 5 |
| `exp(x)+exp(x)` | 5 | 3 | 31 | 16 | 6.2 | 3.2 | 1.9375 | 3 |
| `((x*x)*(y*y))*((x*x)*(x+1))` | 15 | 9 | 267 | 70 | 17.8 | 4.66667 | 3.81429 | 10 |

## Hidden-Complexity Checks

- No derived leaves are allowed.
- No hidden compound leaves are allowed.
- No macro or template DAG nodes are allowed.
- Pure EML DAG internal nodes must be exactly `eml`.
- Pure EML DAG leaves must be variables or constant `1`.
- Child slots must be preserved as `left` and `right` for binary EML nodes.
- Duplicate child references must remain explicit references.

## Result

All audited expressions passed structural purity and numeric semantic checks.

This remains a representation-level audit. It does not claim neural model performance improvement.
