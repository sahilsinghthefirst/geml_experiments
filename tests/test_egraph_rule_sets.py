from __future__ import annotations

from fractions import Fraction

from geml.egraph.egraph import EGraph
from geml.egraph.ir import Add, Const, Div, Exp, Log, Mul, Neg, Pow, Sub, Var
from geml.egraph.rewrites import RewriteRule, SaturationLimits, saturate
from geml.egraph.rule_sets import (
    DEFAULT_RULE_MODE,
    POSITIVE_REAL_ASSUMPTIONS,
    RuleSetConfig,
    constants_v0,
    default_safe_rules,
    positive_real_logexp_v0,
    rule_mode_summary,
    rule_sets_for_mode,
    rules_for_mode,
    safe_core_v0,
)
from geml.egraph.validation import positive_real_numeric_validation


def test_safe_core_has_rule_provenance() -> None:
    rules = safe_core_v0().rules

    assert rules
    for rule in rules:
        assert rule.rule_name
        assert rule.rule_tier in {"safe", "guarded"}
        assert rule.direction in {"forward", "backward", "bidirectional"}
        assert rule.guard_status in {"guarded", "unguarded"}


def test_constants_rules_are_guarded() -> None:
    rules = constants_v0().rules

    assert rules
    assert all(rule.guard_status == "guarded" for rule in rules)


def test_default_rule_mode_is_safe() -> None:
    assert DEFAULT_RULE_MODE == "safe"


def test_positive_real_formal_mode_includes_required_rule_sets() -> None:
    rule_sets = rule_sets_for_mode("positive_real_formal")

    assert [rule_set.name for rule_set in rule_sets] == [
        "safe_core_v0",
        "constants_v0",
        "positive_real_logexp_v0",
    ]


def test_positive_real_logexp_rules_are_disabled_by_default_not_empty() -> None:
    rule_set = positive_real_logexp_v0()

    assert rule_set.enabled_by_default is False
    assert rule_set.rules
    assert {rule.name for rule in rule_set.rules} >= {
        "exp_log_inverse",
        "exp_sum",
        "log_exp_inverse",
        "log_product",
    }


def test_x_plus_y_equivalent_to_y_plus_x() -> None:
    assert equivalent(
        Add(Var("x"), Var("y")),
        Add(Var("y"), Var("x")),
        safe_core_v0().rules,
    )


def test_x_plus_one_equivalent_to_x_plus_two_minus_one_with_constants() -> None:
    assert equivalent(
        Add(Var("x"), Const(1)),
        Add(Var("x"), Sub(Const(2), Const(1))),
        default_safe_rules(),
        max_iterations=10,
    )


def test_mul_one_rewrites_to_x() -> None:
    assert equivalent(Mul(Var("x"), Const(1)), Var("x"), safe_core_v0().rules)


def test_add_zero_rewrites_to_x() -> None:
    assert equivalent(Add(Var("x"), Const(0)), Var("x"), safe_core_v0().rules)


def test_mul_zero_rewrites_to_zero() -> None:
    assert equivalent(Mul(Var("x"), Const(0)), Const(0), safe_core_v0().rules)


def test_add_inverse_rewrites_to_zero() -> None:
    assert equivalent(Add(Var("x"), Neg(Var("x"))), Const(0), safe_core_v0().rules)


def test_pow_zero_rewrite_requires_config_assumption() -> None:
    without_assumption = safe_core_v0(RuleSetConfig(assume_nonzero_for_pow_zero=False)).rules
    with_assumption = safe_core_v0(RuleSetConfig(assume_nonzero_for_pow_zero=True)).rules

    assert not equivalent(Pow(Var("x"), Const(0)), Const(1), without_assumption)
    assert equivalent(Pow(Var("x"), Const(0)), Const(1), with_assumption)


def test_a_over_a_is_not_rewritten_in_safe_core() -> None:
    assert not equivalent(Div(Var("a"), Var("a")), Const(1), safe_core_v0().rules)


def test_log_exp_is_not_rewritten_in_safe_core() -> None:
    assert not equivalent(Log(Exp(Var("x"))), Var("x"), safe_core_v0().rules)


def test_exp_log_is_not_rewritten_in_safe_core() -> None:
    assert not equivalent(Exp(Log(Var("x"))), Var("x"), safe_core_v0().rules)


def test_log_product_is_not_rewritten_in_safe_mode() -> None:
    assert not equivalent(
        Log(Mul(Var("x"), Var("y"))),
        Add(Log(Var("x")), Log(Var("y"))),
        rules_for_mode("safe"),
    )


def test_a_over_a_is_not_rewritten_in_positive_real_mode_by_default() -> None:
    assert not equivalent(Div(Var("a"), Var("a")), Const(1), rules_for_mode("positive_real_formal"))


def test_a_over_a_rewrites_in_positive_real_mode_when_nonzero_symbols_assumed() -> None:
    rules = rules_for_mode(
        "positive_real_formal",
        RuleSetConfig(assume_nonzero_symbols=True),
    )

    assert equivalent(Div(Var("a"), Var("a")), Const(1), rules)


def test_log_power_rewrite_is_opt_in_positive_real_mode() -> None:
    expression = Log(Pow(Var("x"), Var("y")))
    expanded = Mul(Var("y"), Log(Var("x")))

    assert not equivalent(expression, expanded, rules_for_mode("positive_real_formal"))
    assert equivalent(
        expression,
        expanded,
        rules_for_mode("positive_real_formal", RuleSetConfig(enable_log_power_rule=True)),
    )


def test_log_exp_rewrites_in_positive_real_formal_mode() -> None:
    assert equivalent(Log(Exp(Var("x"))), Var("x"), rules_for_mode("positive_real_formal"))


def test_exp_log_rewrites_in_positive_real_formal_mode() -> None:
    assert equivalent(Exp(Log(Var("x"))), Var("x"), rules_for_mode("positive_real_formal"))


def test_log_product_rewrites_in_positive_real_formal_mode() -> None:
    assert equivalent(
        Log(Mul(Var("x"), Var("y"))),
        Add(Log(Var("x")), Log(Var("y"))),
        rules_for_mode("positive_real_formal"),
    )


def test_all_branch_sensitive_rules_have_positive_real_provenance() -> None:
    positive_rules = positive_real_logexp_v0().rules

    assert positive_rules
    assert all(rule.branch_sensitive is True for rule in positive_rules)
    assert all(rule.assumptions == POSITIVE_REAL_ASSUMPTIONS for rule in positive_rules)
    assert all(
        rule.rule_tier in {"positive_real_formal_rules", "guarded"} for rule in positive_rules
    )


def test_positive_real_mode_summary_reports_branch_sensitive_fields() -> None:
    summary = rule_mode_summary("positive_real_formal")

    assert summary.rule_mode == "positive_real_formal"
    assert summary.assumptions == POSITIVE_REAL_ASSUMPTIONS
    assert summary.branch_sensitive_rules_used is True
    assert summary.branch_sensitive_rule_count == len(summary.branch_sensitive_rule_names)
    assert "log_exp_inverse" in summary.branch_sensitive_rule_names


def test_positive_real_numeric_validation_passes_for_selected_examples() -> None:
    examples = [
        (Log(Exp(Var("x"))), Var("x")),
        (Exp(Log(Var("x"))), Var("x")),
        (Log(Mul(Var("x"), Var("y"))), Add(Log(Var("x")), Log(Var("y")))),
    ]

    for original, extracted in examples:
        result = positive_real_numeric_validation(original, extracted)
        assert result.validation_status == "valid"
        assert result.max_abs_error is not None
        assert result.max_abs_error < 1e-9
        assert result.assumptions == POSITIVE_REAL_ASSUMPTIONS


def test_constant_division_by_zero_is_guarded_off() -> None:
    assert not equivalent(Div(Const(1), Const(0)), Const(0), constants_v0().rules)


def test_constant_integer_power_folds_when_exact_and_bounded() -> None:
    assert equivalent(Pow(Const(2), Const(3)), Const(8), constants_v0().rules)


def test_constant_fractional_power_is_not_folded() -> None:
    assert not equivalent(Pow(Const(4), Const(Fraction(1, 2))), Const(2), constants_v0().rules)


def test_huge_constant_folding_is_guarded_off() -> None:
    config = RuleSetConfig(max_abs_constant=Fraction(10, 1), max_constant_bit_length=8)

    assert not equivalent(Mul(Const(100), Const(100)), Const(10_000), constants_v0(config).rules)


def equivalent(
    left: object,
    right: object,
    rules: tuple[RewriteRule, ...],
    *,
    max_iterations: int = 6,
) -> bool:
    egraph = EGraph()
    left_id = egraph.add_expr(left)
    right_id = egraph.add_expr(right)
    saturate(
        egraph,
        rules,
        limits=SaturationLimits(
            max_iterations=max_iterations,
            max_enodes=10_000,
            max_eclasses=10_000,
            timeout_seconds=5,
        ),
    )
    return egraph.find(left_id) == egraph.find(right_id)
