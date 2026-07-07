# Goal 4.2 Safe E-Graph Rule Sets

Goal 4.2 adds named rewrite rule sets for the Goal 4 e-graph core. These rules
operate on the GEML source-expression IR, not on official pure EML trees.

This stage does not implement EML-aware extraction, neural models,
visualization, or changes to the official pure EML compiler formulas.

## Corpus Baseline

Goal 4 e-graph compression runs that produce result claims must use the v1
corpus. V0 corpus runs are diagnostic only. Rule-set names such as
`safe_core_v0`, `constants_v0`, and `positive_real_logexp_v0` are rule-version
labels; they do not imply that the v0 corpus is an accepted result baseline.

## Rule Sets

### `safe_core_v0`

`safe_core_v0` is enabled by default and contains branch-insensitive algebraic
rules:

- add commutativity: `add(?a, ?b) <-> add(?b, ?a)`
- mul commutativity: `mul(?a, ?b) <-> mul(?b, ?a)`
- add associativity: `add(add(?a, ?b), ?c) <-> add(?a, add(?b, ?c))`
- mul associativity: `mul(mul(?a, ?b), ?c) <-> mul(?a, mul(?b, ?c))`
- add identity: `add(?a, 0) -> ?a`, `add(0, ?a) -> ?a`
- mul identity: `mul(?a, 1) -> ?a`, `mul(1, ?a) -> ?a`
- mul zero: `mul(?a, 0) -> 0`, `mul(0, ?a) -> 0`
- sub lowering: `sub(?a, ?b) -> add(?a, neg(?b))`
- double negation: `neg(neg(?a)) -> ?a`
- add inverse: `add(?a, neg(?a)) -> 0`, `add(neg(?a), ?a) -> 0`
- pow identity: `pow(?a, 1) -> ?a`
- guarded pow zero: `pow(?a, 0) -> 1` only when
  `assume_nonzero_for_pow_zero=true`

### `constants_v0`

`constants_v0` is enabled by default and performs exact rational constant
folding for:

- `add(Const, Const)`
- `mul(Const, Const)`
- `sub(Const, Const)`
- `div(Const, Const)` when the denominator is nonzero
- `neg(Const)`
- `pow(Const, Const)` only for exact integer exponents that stay within the
  configured constant bounds

Constants use exact `Fraction` values. Rewrite logic does not use floats.

### `positive_real_logexp_v0`

`positive_real_logexp_v0` is a disabled placeholder. It records future
positive-real log/exp rules, but contributes no active rewrites by default.

The safe default rule sets do not include:

- `log(exp(x)) -> x`
- `exp(log(x)) -> x`
- `log(a*b) -> log(a)+log(b)`
- `exp(a+b) -> exp(a)*exp(b)`
- `a/a -> 1`

## Guards And Bounds

Rule-set configuration records:

- `assume_nonzero_for_pow_zero`
- `max_abs_constant`
- `max_constant_bit_length`

Division by zero is rejected. Constant folds that would produce values outside
the configured magnitude or bit-length bounds are rejected. Constant powers are
folded only when the result is exact, rational, and bounded.

## Rule Provenance

Every active rewrite records:

- `rule_name`
- `rule_tier`
- `direction`
- `guard_status`

This metadata is required so later Goal 4 outputs can report which rules were
allowed during saturation.
