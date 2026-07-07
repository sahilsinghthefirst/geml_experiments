# Goal 4.9 E-Graph Semantic Audit

This audit checks the fixed Goal 4 selected expressions on the v1 artifact path.
It is a semantic, structural-purity, and provenance audit for non-ML e-graph
compression. It does not use v0 as the primary corpus and does not make neural
model claims.

## Scope

- run modes: `safe`, `positive_real_formal`
- extractor: `exact_eml_dag_beam_cost`
- semantic checks: positive-real numeric probes only; no complex-domain validity is claimed
- final metrics: official pure EML compilation followed by exact structural EML-DAG conversion
- rewrite provenance: actual applied rewrite names and tiers are recorded
- extraction provenance: candidate exact EML-DAG metrics are recorded
- shortcut check: rewrite/e-graph source files are scanned for `simplify` usage

## Artifacts

- JSON: `outputs/v1/goal4_egraph_semantic_audit.json`
- CSV: `outputs/v1/goal4_egraph_semantic_audit.csv`
- report: `docs/goal4/GOAL4_EGRAPH_SEMANTIC_AUDIT.md`

## Summary

- rows: `28`
- all structural purity checks valid: `True`
- all positive-real semantic checks valid: `True`
- all extracted EML-DAG evaluator checks valid: `True`
- branch-sensitive rule applications in safe mode: `0`
- branch-sensitive rule applications in positive-real mode: `5`
- rewrite path free of SymPy simplify shortcut: `True`

## Required Cases

- `log(exp(x))` safe extraction: `Log(Exp(x))`; branch-sensitive applied: `False`
- `log(exp(x))` positive-real extraction: `x`; branch-sensitive applied: `True`
- `exp(log(x))` safe extraction: `Exp(Log(x))`; branch-sensitive applied: `False`
- `exp(log(x))` positive-real extraction: `x`; branch-sensitive applied: `True`
- `x+2-1` safe EML-DAG nodes: `22 -> 14`

## Audit Table

| Expression | Mode | Extracted | Rules applied | Branch-sensitive | Original D_EML | Extracted D_EML | Gain | Semantic | Pure |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `x+y` | `safe` | `Add(x,y)` | `add_commutativity` | `False` | 16 | 16 | 1 | `valid` | `True` |
| `x+y` | `positive_real_formal` | `Add(x,y)` | `add_commutativity` | `False` | 16 | 16 | 1 | `valid` | `True` |
| `y+x` | `safe` | `Add(x,y)` | `add_commutativity` | `False` | 16 | 16 | 1 | `valid` | `True` |
| `y+x` | `positive_real_formal` | `Add(x,y)` | `add_commutativity` | `False` | 16 | 16 | 1 | `valid` | `True` |
| `x+1` | `safe` | `Add(1,x)` | `add_commutativity` | `False` | 14 | 14 | 1 | `valid` | `True` |
| `x+1` | `positive_real_formal` | `Add(1,x)` | `add_commutativity` | `False` | 14 | 14 | 1 | `valid` | `True` |
| `x+2-1` | `safe` | `Add(1,x)` | `add_associativity, add_commutativity, fold_add_constants, fold_neg_constant, sub_lowering` | `False` | 22 | 14 | 1.57143 | `valid` | `True` |
| `x+2-1` | `positive_real_formal` | `Add(1,x)` | `add_associativity, add_commutativity, fold_add_constants, fold_neg_constant, sub_lowering` | `False` | 22 | 14 | 1.57143 | `valid` | `True` |
| `x*1` | `safe` | `x` | `mul_associativity, mul_commutativity, mul_identity_right` | `False` | 19 | 1 | 19 | `valid` | `True` |
| `x*1` | `positive_real_formal` | `x` | `mul_associativity, mul_commutativity, mul_identity_right` | `False` | 19 | 1 | 19 | `valid` | `True` |
| `x*x` | `safe` | `Mul(x,x)` | `none` | `False` | 19 | 19 | 1 | `valid` | `True` |
| `x*x` | `positive_real_formal` | `Mul(x,x)` | `none` | `False` | 19 | 19 | 1 | `valid` | `True` |
| `x**2` | `safe` | `Pow(x,2)` | `none` | `False` | 29 | 29 | 1 | `valid` | `True` |
| `x**2` | `positive_real_formal` | `Pow(x,2)` | `none` | `False` | 29 | 29 | 1 | `valid` | `True` |
| `(x+1)*(x+1)` | `safe` | `Mul(Add(1,x),Add(1,x))` | `add_commutativity` | `False` | 25 | 25 | 1 | `valid` | `True` |
| `(x+1)*(x+1)` | `positive_real_formal` | `Mul(Add(1,x),Add(1,x))` | `add_commutativity` | `False` | 25 | 25 | 1 | `valid` | `True` |
| `log(x)+log(x)` | `safe` | `Add(Log(x),Log(x))` | `none` | `False` | 18 | 18 | 1 | `valid` | `True` |
| `log(x)+log(x)` | `positive_real_formal` | `Add(Log(x),Log(x))` | `log_product` | `True` | 18 | 18 | 1 | `valid` | `True` |
| `log(exp(x))` | `safe` | `Log(Exp(x))` | `none` | `False` | 6 | 6 | 1 | `valid` | `True` |
| `log(exp(x))` | `positive_real_formal` | `x` | `log_exp_inverse` | `True` | 6 | 1 | 6 | `valid` | `True` |
| `exp(log(x))` | `safe` | `Exp(Log(x))` | `none` | `False` | 6 | 6 | 1 | `valid` | `True` |
| `exp(log(x))` | `positive_real_formal` | `x` | `exp_log_inverse` | `True` | 6 | 1 | 6 | `valid` | `True` |
| `log(x*y)` | `safe` | `Log(Mul(x,y))` | `mul_commutativity` | `False` | 26 | 26 | 1 | `valid` | `True` |
| `log(x*y)` | `positive_real_formal` | `Add(Log(x),Log(y))` | `log_product, mul_commutativity` | `True` | 26 | 22 | 1.18182 | `valid` | `True` |
| `exp(x+y)` | `safe` | `Exp(Add(x,y))` | `add_commutativity` | `False` | 17 | 17 | 1 | `valid` | `True` |
| `exp(x+y)` | `positive_real_formal` | `Exp(Add(x,y))` | `add_commutativity, exp_sum` | `True` | 17 | 17 | 1 | `valid` | `True` |
| `((x*x)*(y*y))*((x*x)*(x + 1))` | `safe` | `Mul(Add(1,x),Mul(Mul(Mul(Mul(x,x),Mul(x,x)),y),y))` | `add_commutativity, mul_associativity, mul_commutativity` | `False` | 70 | 70 | 1 | `valid` | `True` |
| `((x*x)*(y*y))*((x*x)*(x + 1))` | `positive_real_formal` | `Mul(Add(1,x),Mul(Mul(Mul(Mul(x,x),Mul(x,x)),y),y))` | `add_commutativity, mul_associativity, mul_commutativity` | `False` | 70 | 70 | 1 | `valid` | `True` |

## Integrity Boundary

Every successful extracted expression is compiled through the official pure EML
compiler before metrics are recorded. The final EML tree/DAG must contain only
`eml` internal nodes and variable or constant-`1` leaves. Derived leaves, hidden
compound leaves, macro/template nodes, invalid child slots, and collapsed duplicate
child references are audit failures.

The positive-real mode is branch-sensitive formal algebra. It is reported separately
from safe mode and is not universal complex-domain algebra.
