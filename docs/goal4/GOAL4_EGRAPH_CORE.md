# Goal 4.1 E-Graph Core

Goal 4.1 adds a small controlled e-graph core for the GEML source-expression
subset. It is intentionally transparent and local to `geml/egraph/`.

This stage does not implement EML-aware extraction, neural models,
visualization, or any change to the official pure EML compiler formulas.

## Implemented Modules

- `geml/egraph/ir.py`: immutable source-expression IR, exact rational constants,
  SymPy conversion, and canonical display strings.
- `geml/egraph/egraph.py`: ordered e-nodes, e-classes, union-find, e-node
  insertion, expression insertion, rebuild, congruence closure, and node/class
  counts.
- `geml/egraph/patterns.py`: e-class pattern matching with variables such as
  `?a`, `?b`, and `?c`.
- `geml/egraph/rewrites.py`: rewrite rules, guards, simple safe rule
  constructors, positive-real log/exp placeholders, and a resource-limited
  saturation loop.
- `geml/egraph/extractor.py`: ordinary AST-node-count extraction baseline only.
- `geml/egraph/validation.py`: final SymPy-based semantic validation helpers.

## Supported Source IR

The e-graph IR supports:

- `Var(name)`
- `Const(value)` using exact `Fraction` values
- `Add(a, b)`
- `Mul(a, b)`
- `Neg(a)`
- `Sub(a, b)`
- `Div(a, b)`
- `Pow(a, b)`
- `Exp(a)`
- `Log(a)`

Rewrite logic uses exact rational constants and does not use floats.

## Core Semantics

E-nodes are ordered. `Add(x, y)` and `Add(y, x)` are different e-nodes until a
commutativity rewrite explicitly introduces the swapped form and unions the
classes.

Pattern matching operates over e-classes. A concrete pattern can match any
e-node inside an e-class, and pattern variables bind to canonical e-class ids.

The saturation loop records one of these statuses:

- `saturated`
- `iteration_limit`
- `enode_limit`
- `eclass_limit`
- `timeout`

## Deliberate Non-Use Of SymPy Simplification

The e-graph core does not call `SymPy.simplify` as a substitute for equality
saturation. SymPy is used for conversion and final validation only.

Positive-real log/exp rewrites are present only as inactive placeholders in
Goal 4.1. They are not returned by the implemented safe rule set.

## Current Extractor Status

`extract_min_ast_size` is an ordinary AST-node-count baseline extractor. It must
not be reported as EML-optimal. The EML-aware extractor required by Goal 4.0
will be implemented in a later stage.

## Large-Scale Corpus Note

Large-scale Goal 4 compression results must use the v1 corpus by default.
The v0 corpus is pilot/deprecated-for-results after Goal 3R. Any e-graph result
computed on v0 is diagnostic only and must not be reported as the authoritative
Goal 4 result set.

Reserved future subset labels are:

- `all_v1`
- `nontrivial_v1`
- `identity_heavy_v1`
