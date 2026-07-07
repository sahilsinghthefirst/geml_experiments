# Goal 2.0 Representation Semantics

Goal 2 expansion-factor experiments use alpha only for pure, valid restricted EML representations. The previous `Add`/`Mul` lift rule is retained only as a diagnostic representation and must not be used for serious alpha plots.

## Representation Modes

Every representation should declare one of these `representation_mode` values:

- `ast`: normal binary AST representation.
- `restricted_eml_pure`: official recursive pure restricted EML representation.
- `restricted_eml_with_derived`: diagnostic EML representation that may contain derived leaves.

## Formal Restricted EML Grammar

Pure restricted EML trees for Goal 2 use:

```text
P ::= variable | 1 | eml(P, P)
```

where:

- every internal node is `eml`
- every `eml` node has exactly two children
- normal leaves are only source variables and constant `1`
- `derived` leaves are not part of the pure grammar

The EML operator is:

```text
eml(x, y) = exp(x) - log(y)
```

Current pure translation support, after Goal 2.1b:

- `variable -> variable`
- `1 -> 1`
- integer constants via official `eml_int`
- rational constants via official `eml_rational`
- floats via `Rational(str(x))`
- `exp(a) -> EML(P(a), 1)`
- `log(a) -> EML(1, EML(EML(1, P(a)), 1))`
- `Add`, `Mul`, `Pow`, subtraction, division, inverse, and negation through the official recursive macros

The previous Goal 2.0 stopgap marked `Add` and `Mul` unsupported for pure alpha. Goal 2.1b replaces that stopgap with the official recursive compiler, so generated arithmetic/log/exp expressions can now be pure alpha-valid without derived leaves.

## Derived Mode Classification

`restricted_eml_with_derived` permits the previous lift:

```text
E -> eml(log(E), 1)
```

for `Add` and `Mul`. The `log(E)` child is a `derived` leaf. If `E` is compound, that leaf hides an expression subtree and can make the EML tree artificially small.

Derived-mode trees therefore report:

- `normal_leaf_count`: variables and constant `1` only
- `derived_leaf_count`: derived leaves
- `hidden_compound_leaf_count`: derived leaves containing compound source expressions
- `alpha = null`
- `alpha_valid = false`

## Alpha Policy

Serious Goal 2 plots must filter to:

```text
representation_mode == "restricted_eml_pure"
alpha_valid == true
alpha is not null
```

Rows with hidden compound derived leaves must never be treated as valid EML alpha measurements. They may be inspected only to debug or compare against the old Goal 1 derived-lift behavior.

No neural model work or DAG compression is included in Goal 2.0 or Goal 2.1b.
