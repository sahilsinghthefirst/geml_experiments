# Goal 4.3 Positive-Real Formal Rules

Goal 4.3 adds a separate `positive_real_formal` rewrite mode for branch-sensitive
log/exp rules. This mode is not universal complex-domain algebra.

These rules rely on the v1 corpus/domain convention: generated log arguments
are intended to be interpreted under a positive-real domain policy. Results from
this mode must be reported separately from safe-mode results.

## Rule Modes

- `safe`: the default mode. It includes `safe_core_v0` and `constants_v0`.
- `positive_real_formal`: explicit opt-in mode. It includes `safe_core_v0`,
  `constants_v0`, and `positive_real_logexp_v0`.

Safe mode remains the default. Branch-sensitive positive-real rules must never
be silently included in safe mode.

## Active Positive-Real Rules

`positive_real_logexp_v0` contains:

- `log(1) -> 0`
- `exp(0) -> 1`
- `log(exp(?a)) -> ?a`
- `exp(log(?a)) -> ?a`
- `log(mul(?a, ?b)) <-> add(log(?a), log(?b))`
- `exp(add(?a, ?b)) <-> mul(exp(?a), exp(?b))`

Optional guarded rules are present but disabled by default:

- `div(?a, ?a) -> 1` only when `assume_nonzero_symbols=true`
- `log(pow(?a, ?b)) -> mul(?b, log(?a))` only when
  `enable_log_power_rule=true`

## Provenance

Every positive-real rule records:

- `rule_name`
- `rule_tier`
- `branch_sensitive=true`
- `assumptions=positive_real_formal`

Output rows produced under `positive_real_formal` mode must include:

- `rule_mode`
- `assumptions`
- `branch_sensitive_rules_used`
- `branch_sensitive_rule_count`
- `branch_sensitive_rule_names`

## Validation

`positive_real_numeric_validation(original, extracted)` samples only positive
real values for variables such as `x` and `y`. It reports:

- `validation_status`
- `max_abs_error`
- `sample_count`
- `assumptions=positive_real_formal`

This validation is numeric evidence under the positive-real assumption. It does
not prove complex-domain validity.

## Forbidden Safe-Mode Rewrites

Safe mode does not include:

- `log(exp(?a)) -> ?a`
- `exp(log(?a)) -> ?a`
- `log(mul(?a, ?b)) <-> add(log(?a), log(?b))`
- `exp(add(?a, ?b)) <-> mul(exp(?a), exp(?b))`
- `div(?a, ?a) -> 1`

Goal 4.3 does not modify official EML compiler formulas, introduce macro EML
nodes, hide operators in leaves, run the 10k pipeline, or implement neural
models.
