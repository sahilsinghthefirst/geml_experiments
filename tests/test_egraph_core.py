from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import sympy as sp
from geml.egraph.egraph import EGraph
from geml.egraph.extractor import extract_min_ast_size
from geml.egraph.ir import Add, Const, Var, display, from_sympy, to_sympy
from geml.egraph.patterns import match_eclass, parse_pattern
from geml.egraph.rewrites import (
    SaturationLimits,
    add_associativity_rule,
    add_commutativity_rule,
    additive_identity_rules,
    positive_real_formal_rule_placeholders,
    saturate,
)


def test_add_commutativity_equivalence_requires_rule() -> None:
    x = Var("x")
    y = Var("y")
    egraph = EGraph()
    left_id = egraph.add_expr(Add(x, y))
    right_id = egraph.add_expr(Add(y, x))

    assert egraph.find(left_id) != egraph.find(right_id)

    saturate(
        egraph,
        [add_commutativity_rule()],
        limits=SaturationLimits(max_iterations=3),
    )

    assert egraph.find(left_id) == egraph.find(right_id)


def test_add_associativity_equivalence_requires_rule() -> None:
    x = Var("x")
    y = Var("y")
    z = Var("z")
    egraph = EGraph()
    left_id = egraph.add_expr(Add(Add(x, y), z))
    right_id = egraph.add_expr(Add(x, Add(y, z)))

    assert egraph.find(left_id) != egraph.find(right_id)

    saturate(
        egraph,
        [add_associativity_rule()],
        limits=SaturationLimits(max_iterations=3),
    )

    assert egraph.find(left_id) == egraph.find(right_id)


def test_additive_identity_equivalence_requires_rule() -> None:
    x = Var("x")
    egraph = EGraph()
    left_id = egraph.add_expr(Add(x, Const(0)))
    right_id = egraph.add_expr(x)

    assert egraph.find(left_id) != egraph.find(right_id)

    saturate(
        egraph,
        list(additive_identity_rules()),
        limits=SaturationLimits(max_iterations=3),
    )

    assert egraph.find(left_id) == egraph.find(right_id)


def test_pattern_matching_operates_over_eclasses() -> None:
    x = Var("x")
    y = Var("y")
    z = Var("z")
    egraph = EGraph()
    x_id = egraph.add_expr(x)
    y_id = egraph.add_expr(y)
    egraph.union(x_id, y_id)
    egraph.rebuild()
    root_id = egraph.add_expr(Add(y, z))

    matches = match_eclass(egraph, parse_pattern("Add(x, ?b)"), root_id)

    assert len(matches) == 1
    assert egraph.find(matches[0]["?b"]) == egraph.find(egraph.add_expr(z))


def test_saturation_stops_at_limits() -> None:
    egraph = EGraph()
    egraph.add_expr(Add(Var("x"), Var("y")))

    result = saturate(
        egraph,
        [add_commutativity_rule()],
        limits=SaturationLimits(max_iterations=0),
    )

    assert result.status == "iteration_limit"


def test_no_sympy_simplify_shortcut_is_used_in_egraph_core() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    core_files = [
        repo_root / "geml/egraph/egraph.py",
        repo_root / "geml/egraph/patterns.py",
        repo_root / "geml/egraph/rewrites.py",
        repo_root / "geml/egraph/rule_sets.py",
        repo_root / "geml/egraph/extractor.py",
        repo_root / "geml/egraph/ir.py",
    ]

    for path in core_files:
        source = path.read_text(encoding="utf-8")
        assert ".simplify" not in source
        assert "simplify(" not in source


def test_conversion_round_trip_for_supported_subset() -> None:
    x, y = sp.symbols("x y")
    sympy_exprs = [
        x,
        sp.Rational(2, 3),
        sp.Add(x, y, evaluate=False),
        sp.Mul(x, y, evaluate=False),
        sp.exp(x, evaluate=False),
        sp.log(x, evaluate=False),
        sp.Pow(x, 2, evaluate=False),
    ]

    for sympy_expr in sympy_exprs:
        ir_expr = from_sympy(sympy_expr)
        round_tripped = from_sympy(to_sympy(ir_expr))
        assert display(round_tripped) == display(ir_expr)


def test_exact_rational_constants_are_preserved() -> None:
    expr = Const(Fraction(3, 5))

    assert display(expr) == "3/5"
    assert to_sympy(expr) == sp.Rational(3, 5)


def test_positive_real_log_exp_rules_are_placeholders_only() -> None:
    placeholders = positive_real_formal_rule_placeholders()

    assert {placeholder.name for placeholder in placeholders} == {
        "exp_log_inverse",
        "exp_sum",
        "log_exp_inverse",
        "log_product",
    }


def test_ast_baseline_extractor_does_not_claim_eml_cost() -> None:
    egraph = EGraph()
    root_id = egraph.add_expr(Add(Var("x"), Const(0)))
    saturate(
        egraph,
        list(additive_identity_rules()),
        limits=SaturationLimits(max_iterations=3),
    )

    extracted = extract_min_ast_size(egraph, root_id)

    assert extracted.status == "completed"
    assert extracted.cost_model == "ast_node_count"
    assert extracted.expression == Var("x")
