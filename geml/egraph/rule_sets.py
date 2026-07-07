"""Named Goal 4 e-graph rule sets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from geml.egraph.egraph import EGraph
from geml.egraph.ir import Add, Const, Div, Exp, Log, Mul, Neg, Pow, Sub
from geml.egraph.patterns import Substitution, pvar
from geml.egraph.rewrites import RewriteRule, RulePlaceholder

type RuleMode = Literal["safe", "positive_real_formal"]

DEFAULT_RULE_MODE: RuleMode = "safe"
POSITIVE_REAL_ASSUMPTIONS = "positive_real_formal"


@dataclass(frozen=True, slots=True)
class RuleSetConfig:
    """Configuration for guarded safe rule sets."""

    assume_nonzero_for_pow_zero: bool = False
    assume_nonzero_symbols: bool = False
    enable_log_power_rule: bool = False
    max_abs_constant: Fraction = Fraction(1_000_000, 1)
    max_constant_bit_length: int = 64

    def __post_init__(self) -> None:
        if self.max_abs_constant <= 0:
            raise ValueError("max_abs_constant must be positive")
        if self.max_constant_bit_length <= 0:
            raise ValueError("max_constant_bit_length must be positive")


@dataclass(frozen=True, slots=True)
class NamedRuleSet:
    """A named collection of active rules and disabled placeholders."""

    name: str
    version: str
    rules: tuple[RewriteRule, ...]
    placeholders: tuple[RulePlaceholder, ...] = ()
    enabled_by_default: bool = True


@dataclass(frozen=True, slots=True)
class RuleModeSummary:
    """Metadata required on outputs produced with a rule mode."""

    rule_mode: RuleMode
    assumptions: str | None
    branch_sensitive_rules_used: bool
    branch_sensitive_rule_count: int
    branch_sensitive_rule_names: tuple[str, ...]


def safe_core_v0(config: RuleSetConfig | None = None) -> NamedRuleSet:
    """Return the default safe algebraic rule set without log/exp branch rules."""
    rule_config = config or RuleSetConfig()
    a = pvar("a")
    b = pvar("b")
    c = pvar("c")
    zero = Const(0)
    one = Const(1)

    return NamedRuleSet(
        name="safe_core_v0",
        version="v0",
        rules=(
            RewriteRule(
                name="add_commutativity",
                tier="safe",
                direction="bidirectional",
                left=Add(a, b),
                right=Add(b, a),
            ),
            RewriteRule(
                name="mul_commutativity",
                tier="safe",
                direction="bidirectional",
                left=Mul(a, b),
                right=Mul(b, a),
            ),
            RewriteRule(
                name="add_associativity",
                tier="safe",
                direction="forward",
                left=Add(Add(a, b), c),
                right=Add(a, Add(b, c)),
            ),
            RewriteRule(
                name="add_associativity",
                tier="safe",
                direction="backward",
                left=Add(a, Add(b, c)),
                right=Add(Add(a, b), c),
            ),
            RewriteRule(
                name="mul_associativity",
                tier="safe",
                direction="forward",
                left=Mul(Mul(a, b), c),
                right=Mul(a, Mul(b, c)),
            ),
            RewriteRule(
                name="mul_associativity",
                tier="safe",
                direction="backward",
                left=Mul(a, Mul(b, c)),
                right=Mul(Mul(a, b), c),
            ),
            RewriteRule(
                name="add_identity_right",
                tier="safe",
                left=Add(a, zero),
                right=a,
            ),
            RewriteRule(
                name="add_identity_left",
                tier="safe",
                left=Add(zero, a),
                right=a,
            ),
            RewriteRule(
                name="mul_identity_right",
                tier="safe",
                left=Mul(a, one),
                right=a,
            ),
            RewriteRule(
                name="mul_identity_left",
                tier="safe",
                left=Mul(one, a),
                right=a,
            ),
            RewriteRule(
                name="mul_zero_right",
                tier="safe",
                left=Mul(a, zero),
                right=zero,
            ),
            RewriteRule(
                name="mul_zero_left",
                tier="safe",
                left=Mul(zero, a),
                right=zero,
            ),
            RewriteRule(
                name="sub_lowering",
                tier="safe",
                left=Sub(a, b),
                right=Add(a, Neg(b)),
            ),
            RewriteRule(
                name="double_negation",
                tier="safe",
                left=Neg(Neg(a)),
                right=a,
            ),
            RewriteRule(
                name="add_inverse_right",
                tier="safe",
                left=Add(a, Neg(a)),
                right=zero,
            ),
            RewriteRule(
                name="add_inverse_left",
                tier="safe",
                left=Add(Neg(a), a),
                right=zero,
            ),
            RewriteRule(
                name="pow_one",
                tier="safe",
                left=Pow(a, one),
                right=a,
            ),
            RewriteRule(
                name="pow_zero_assume_nonzero",
                tier="guarded",
                left=Pow(a, zero),
                right=one,
                guard=lambda _graph, _subst: rule_config.assume_nonzero_for_pow_zero,
            ),
        ),
    )


def constants_v0(config: RuleSetConfig | None = None) -> NamedRuleSet:
    """Return bounded exact rational constant-folding rules."""
    rule_config = config or RuleSetConfig()
    a = pvar("a")
    b = pvar("b")

    return NamedRuleSet(
        name="constants_v0",
        version="v0",
        rules=(
            RewriteRule(
                name="fold_add_constants",
                tier="safe",
                left=Add(a, b),
                right=lambda graph, subst: _add_const(
                    graph,
                    _required_constant(graph, subst["?a"]) + _required_constant(graph, subst["?b"]),
                ),
                guard=_bounded_binary_constant_guard(rule_config, "?a", "?b", _add_values),
            ),
            RewriteRule(
                name="fold_mul_constants",
                tier="safe",
                left=Mul(a, b),
                right=lambda graph, subst: _add_const(
                    graph,
                    _required_constant(graph, subst["?a"]) * _required_constant(graph, subst["?b"]),
                ),
                guard=_bounded_binary_constant_guard(rule_config, "?a", "?b", _mul_values),
            ),
            RewriteRule(
                name="fold_sub_constants",
                tier="safe",
                left=Sub(a, b),
                right=lambda graph, subst: _add_const(
                    graph,
                    _required_constant(graph, subst["?a"]) - _required_constant(graph, subst["?b"]),
                ),
                guard=_bounded_binary_constant_guard(rule_config, "?a", "?b", _sub_values),
            ),
            RewriteRule(
                name="fold_div_constants",
                tier="safe",
                left=Div(a, b),
                right=lambda graph, subst: _add_const(
                    graph,
                    _required_constant(graph, subst["?a"]) / _required_constant(graph, subst["?b"]),
                ),
                guard=_bounded_binary_constant_guard(rule_config, "?a", "?b", _div_values),
            ),
            RewriteRule(
                name="fold_neg_constant",
                tier="safe",
                left=Neg(a),
                right=lambda graph, subst: _add_const(
                    graph,
                    -_required_constant(graph, subst["?a"]),
                ),
                guard=_bounded_unary_constant_guard(rule_config, "?a", lambda value: -value),
            ),
            RewriteRule(
                name="fold_pow_constants",
                tier="safe",
                left=Pow(a, b),
                right=lambda graph, subst: _add_const(
                    graph,
                    _required_pow_value(graph, subst, rule_config),
                ),
                guard=_bounded_binary_constant_guard(rule_config, "?a", "?b", _pow_values),
            ),
        ),
    )


def positive_real_logexp_v0() -> NamedRuleSet:
    """Return branch-sensitive positive-real formal log/exp rules."""
    a = pvar("a")
    b = pvar("b")
    zero = Const(0)
    one = Const(1)

    return NamedRuleSet(
        name="positive_real_logexp_v0",
        version="v0",
        rules=(
            _positive_real_rule(
                name="log_one",
                left=Log(one),
                right=zero,
            ),
            _positive_real_rule(
                name="exp_zero",
                left=Exp(zero),
                right=one,
            ),
            _positive_real_rule(
                name="log_exp_inverse",
                left=Log(Exp(a)),
                right=a,
            ),
            _positive_real_rule(
                name="exp_log_inverse",
                left=Exp(Log(a)),
                right=a,
            ),
            _positive_real_rule(
                name="log_product",
                direction="forward",
                left=Log(Mul(a, b)),
                right=Add(Log(a), Log(b)),
            ),
            _positive_real_rule(
                name="log_product",
                direction="backward",
                left=Add(Log(a), Log(b)),
                right=Log(Mul(a, b)),
            ),
            _positive_real_rule(
                name="exp_sum",
                direction="forward",
                left=Exp(Add(a, b)),
                right=Mul(Exp(a), Exp(b)),
            ),
            _positive_real_rule(
                name="exp_sum",
                direction="backward",
                left=Mul(Exp(a), Exp(b)),
                right=Exp(Add(a, b)),
            ),
            _positive_real_rule(
                name="div_self_assume_nonzero_symbols",
                tier="guarded",
                left=Div(a, a),
                right=one,
                guard=lambda _graph, _subst: False,
            ),
            _positive_real_rule(
                name="log_power",
                tier="guarded",
                left=Log(Pow(a, b)),
                right=Mul(b, Log(a)),
                guard=lambda _graph, _subst: False,
            ),
        ),
        enabled_by_default=False,
    )


def positive_real_logexp_v0_with_config(config: RuleSetConfig | None = None) -> NamedRuleSet:
    """Return positive-real formal rules with optional guarded rules configured."""
    rule_config = config or RuleSetConfig()
    base = positive_real_logexp_v0()
    configured_rules = tuple(
        _configure_positive_real_guard(rule, rule_config) for rule in base.rules
    )
    return NamedRuleSet(
        name=base.name,
        version=base.version,
        rules=configured_rules,
        placeholders=base.placeholders,
        enabled_by_default=base.enabled_by_default,
    )


def default_safe_rule_sets(config: RuleSetConfig | None = None) -> tuple[NamedRuleSet, ...]:
    """Return rule sets enabled by default for safe algebraic saturation."""
    return (safe_core_v0(config), constants_v0(config))


def default_safe_rules(config: RuleSetConfig | None = None) -> tuple[RewriteRule, ...]:
    """Return active rules from the default safe rule sets."""
    return tuple(rule for rule_set in default_safe_rule_sets(config) for rule in rule_set.rules)


def rule_sets_for_mode(
    mode: RuleMode = DEFAULT_RULE_MODE,
    config: RuleSetConfig | None = None,
) -> tuple[NamedRuleSet, ...]:
    """Return active rule sets for a named rule mode."""
    if mode == "safe":
        return default_safe_rule_sets(config)
    if mode == "positive_real_formal":
        return (
            safe_core_v0(config),
            constants_v0(config),
            positive_real_logexp_v0_with_config(config),
        )
    raise ValueError(f"unknown rule mode: {mode!r}")


def rules_for_mode(
    mode: RuleMode = DEFAULT_RULE_MODE,
    config: RuleSetConfig | None = None,
) -> tuple[RewriteRule, ...]:
    """Return active rules for a named rule mode."""
    return tuple(rule for rule_set in rule_sets_for_mode(mode, config) for rule in rule_set.rules)


def rule_mode_summary(
    mode: RuleMode = DEFAULT_RULE_MODE,
    config: RuleSetConfig | None = None,
) -> RuleModeSummary:
    """Return output-row metadata for a rule mode."""
    rules = rules_for_mode(mode, config)
    branch_sensitive_names = tuple(rule.name for rule in rules if rule.branch_sensitive)
    return RuleModeSummary(
        rule_mode=mode,
        assumptions=POSITIVE_REAL_ASSUMPTIONS if mode == "positive_real_formal" else None,
        branch_sensitive_rules_used=bool(branch_sensitive_names),
        branch_sensitive_rule_count=len(branch_sensitive_names),
        branch_sensitive_rule_names=branch_sensitive_names,
    )


def _positive_real_rule(
    *,
    name: str,
    left: object,
    right: object,
    direction: str = "forward",
    tier: str = "positive_real_formal_rules",
    guard: Callable[[EGraph, Substitution], bool] | None = None,
) -> RewriteRule:
    return RewriteRule(
        name=name,
        tier=tier,
        direction=direction,
        left=left,
        right=right,
        guard=guard,
        branch_sensitive=True,
        assumptions=POSITIVE_REAL_ASSUMPTIONS,
    )


def _configure_positive_real_guard(rule: RewriteRule, config: RuleSetConfig) -> RewriteRule:
    if rule.name == "div_self_assume_nonzero_symbols":
        return RewriteRule(
            name=rule.name,
            tier=rule.tier,
            direction=rule.direction,
            left=rule.left,
            right=rule.right,
            guard=lambda _graph, _subst: config.assume_nonzero_symbols,
            branch_sensitive=rule.branch_sensitive,
            assumptions=rule.assumptions,
        )
    if rule.name == "log_power":
        return RewriteRule(
            name=rule.name,
            tier=rule.tier,
            direction=rule.direction,
            left=rule.left,
            right=rule.right,
            guard=lambda _graph, _subst: config.enable_log_power_rule,
            branch_sensitive=rule.branch_sensitive,
            assumptions=rule.assumptions,
        )
    return rule


def _bounded_binary_constant_guard(
    config: RuleSetConfig,
    left_name: str,
    right_name: str,
    operation: Callable[[Fraction, Fraction, RuleSetConfig], Fraction | None],
) -> Callable[[EGraph, Substitution], bool]:
    def guard(egraph: EGraph, substitution: Substitution) -> bool:
        left = _constant(egraph, substitution[left_name])
        right = _constant(egraph, substitution[right_name])
        if left is None or right is None:
            return False
        result = operation(left, right, config)
        return result is not None and _constant_is_bounded(result, config)

    return guard


def _bounded_unary_constant_guard(
    config: RuleSetConfig,
    value_name: str,
    operation: Callable[[Fraction], Fraction | None],
) -> Callable[[EGraph, Substitution], bool]:
    def guard(egraph: EGraph, substitution: Substitution) -> bool:
        value = _constant(egraph, substitution[value_name])
        if value is None:
            return False
        result = operation(value)
        return result is not None and _constant_is_bounded(result, config)

    return guard


def _constant(egraph: EGraph, eclass_id: int) -> Fraction | None:
    for node in egraph.get_eclass_nodes(eclass_id):
        if node.op == "const" and isinstance(node.value, Fraction):
            return node.value
    return None


def _required_constant(egraph: EGraph, eclass_id: int) -> Fraction:
    value = _constant(egraph, eclass_id)
    if value is None:
        raise ValueError(f"e-class {eclass_id} does not contain a constant")
    return value


def _required_pow_value(
    egraph: EGraph,
    substitution: Substitution,
    config: RuleSetConfig,
) -> Fraction:
    result = _pow_values(
        _required_constant(egraph, substitution["?a"]),
        _required_constant(egraph, substitution["?b"]),
        config,
    )
    if result is None:
        raise ValueError("pow constant fold is not exact or bounded")
    return result


def _add_const(egraph: EGraph, value: Fraction) -> int:
    return egraph.add_expr(Const(value))


def _add_values(left: Fraction, right: Fraction, _config: RuleSetConfig) -> Fraction:
    return left + right


def _mul_values(left: Fraction, right: Fraction, _config: RuleSetConfig) -> Fraction:
    return left * right


def _sub_values(left: Fraction, right: Fraction, _config: RuleSetConfig) -> Fraction:
    return left - right


def _div_values(left: Fraction, right: Fraction, _config: RuleSetConfig) -> Fraction | None:
    if right == 0:
        return None
    return left / right


def _pow_values(left: Fraction, right: Fraction, config: RuleSetConfig) -> Fraction | None:
    if right.denominator != 1:
        return None
    exponent = right.numerator
    if abs(exponent) > config.max_constant_bit_length:
        return None
    if left == 0 and exponent <= 0:
        return None
    try:
        return left**exponent
    except ZeroDivisionError:
        return None


def _constant_is_bounded(value: Fraction, config: RuleSetConfig) -> bool:
    return (
        abs(value) <= config.max_abs_constant
        and abs(value.numerator).bit_length() <= config.max_constant_bit_length
        and abs(value.denominator).bit_length() <= config.max_constant_bit_length
    )
