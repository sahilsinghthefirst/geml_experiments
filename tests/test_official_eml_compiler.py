"""Tests for the official recursive pure EML compiler port."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pytest
import sympy as sp
from geml.symbolic.eml_transpile import sympy_to_eml_tree
from geml.symbolic.official_eml_compiler import (
    OfficialEmlCompilerError,
    compile_to_official_eml_subtree,
    emit_official_eml_string,
    eml_mul,
    eml_neg,
    eml_sub,
    eml_variable,
    evaluate_official_eml_tree,
)

BAD_FINAL_LABELS = {
    "Add",
    "Mul",
    "Sub",
    "Div",
    "Pow",
    "Exp",
    "Log",
    "Derived",
    "derived",
    "add",
    "mul",
    "pow",
    "exp",
    "log",
}


def assert_pure_eml_tree(expr: sp.Expr | int | float) -> None:
    tree = sympy_to_eml_tree(expr)
    node_ids = {node.id for node in tree.nodes}
    parent_counts = {node.id: 0 for node in tree.nodes}
    child_counts = {node.id: 0 for node in tree.nodes}

    for edge in tree.edges:
        assert edge.source in node_ids
        assert edge.target in node_ids
        parent_counts[edge.target] += 1
        child_counts[edge.source] += 1

    assert parent_counts[tree.root_id] == 0
    for node in tree.nodes:
        if node.id != tree.root_id:
            assert parent_counts[node.id] == 1
        assert node.label not in BAD_FINAL_LABELS
        assert node.kind not in {"derived", "Add", "Mul", "Pow", "Exp", "Log"}
        if child_counts[node.id]:
            assert node.kind == "eml"
            assert node.label == "eml"
            assert child_counts[node.id] == 2
        else:
            assert node.kind in {"variable", "constant"}
            if node.kind == "constant":
                assert node.label == "1"
            else:
                assert node.metadata["expression"] == node.label

    assert tree.statistics.edge_count == tree.statistics.node_count - 1
    assert tree.normal_leaf_count == tree.statistics.leaf_count
    assert tree.derived_leaf_count == 0
    assert tree.hidden_compound_leaf_count == 0
    assert tree.alpha_valid is True


def test_exact_official_style_strings_for_small_macros() -> None:
    x, y = sp.symbols("x y")

    assert emit_official_eml_string(sympy_to_eml_tree(sp.exp(x, evaluate=False))) == "EML[x,1]"
    assert (
        emit_official_eml_string(sympy_to_eml_tree(sp.log(x, evaluate=False)))
        == "EML[1,EML[EML[1,x],1]]"
    )
    assert emit_official_eml_string(sympy_to_eml_tree(0)) == "EML[1,EML[EML[1,1],1]]"

    sub_expr = sp.Add(x, sp.Mul(-1, y, evaluate=False), evaluate=False)
    assert emit_official_eml_string(sympy_to_eml_tree(sub_expr)) == emit_official_eml_string(
        eml_sub(eml_variable("x"), eml_variable("y"))
    )

    add_expr = sp.Add(x, y, evaluate=False)
    assert emit_official_eml_string(sympy_to_eml_tree(add_expr)) == emit_official_eml_string(
        eml_sub(eml_variable("x"), eml_neg(eml_variable("y")))
    )

    mul_expr = sp.Mul(x, y, evaluate=False)
    assert emit_official_eml_string(sympy_to_eml_tree(mul_expr)) == emit_official_eml_string(
        eml_mul(eml_variable("x"), eml_variable("y"))
    )


def test_all_supported_compilations_are_structurally_pure() -> None:
    x, y = sp.symbols("x y")
    expressions: Iterable[sp.Expr | int] = [
        x,
        y,
        1,
        0,
        2,
        3,
        -1,
        sp.Rational(1, 2),
        sp.Add(x, y, evaluate=False),
        sp.Add(x, sp.Mul(-1, y, evaluate=False), evaluate=False),
        sp.Mul(x, y, evaluate=False),
        sp.Mul(x, sp.Pow(y, -1, evaluate=False), evaluate=False),
        sp.exp(x, evaluate=False),
        sp.log(x, evaluate=False),
        sp.Pow(x, 2, evaluate=False),
        sp.Pow(x, y, evaluate=False),
        sp.Mul(sp.Add(x, 1, evaluate=False), sp.Add(y, 1, evaluate=False), evaluate=False),
        sp.log(sp.Add(x, 1, evaluate=False), evaluate=False),
        sp.exp(sp.Mul(x, y, evaluate=False), evaluate=False),
    ]

    for expr in expressions:
        assert_pure_eml_tree(expr)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (sp.Symbol("x"), 1.3),
        (sp.Symbol("y"), 2.1),
        (sp.Integer(1), 1.0),
        (sp.Integer(0), 0.0),
        (sp.Integer(2), 2.0),
        (sp.Integer(3), 3.0),
        (sp.Integer(-1), -1.0),
        (sp.Rational(1, 2), 0.5),
        (sp.Add(sp.Symbol("x"), sp.Symbol("y"), evaluate=False), 3.4),
        (
            sp.Add(sp.Symbol("x"), sp.Mul(-1, sp.Symbol("y"), evaluate=False), evaluate=False),
            -0.8,
        ),
        (
            sp.Add(sp.Symbol("y"), sp.Mul(-1, sp.Symbol("x"), evaluate=False), evaluate=False),
            0.8,
        ),
        (sp.Mul(sp.Symbol("x"), sp.Symbol("y"), evaluate=False), 2.73),
        (
            sp.Mul(sp.Symbol("x"), sp.Pow(sp.Symbol("y"), -1, evaluate=False), evaluate=False),
            1.3 / 2.1,
        ),
        (
            sp.Mul(sp.Symbol("y"), sp.Pow(sp.Symbol("x"), -1, evaluate=False), evaluate=False),
            2.1 / 1.3,
        ),
        (sp.exp(sp.Symbol("x"), evaluate=False), math.exp(1.3)),
        (sp.log(sp.Symbol("x"), evaluate=False), math.log(1.3)),
        (sp.Pow(sp.Symbol("x"), 2, evaluate=False), 1.3**2),
        (sp.Pow(sp.Symbol("x"), sp.Symbol("y"), evaluate=False), 1.3**2.1),
        (
            sp.Mul(
                sp.Add(sp.Symbol("x"), 1, evaluate=False),
                sp.Add(sp.Symbol("y"), 1, evaluate=False),
                evaluate=False,
            ),
            (1.3 + 1.0) * (2.1 + 1.0),
        ),
        (sp.log(sp.Add(sp.Symbol("x"), 1, evaluate=False), evaluate=False), math.log(2.3)),
        (
            sp.exp(sp.Mul(sp.Symbol("x"), sp.Symbol("y"), evaluate=False), evaluate=False),
            math.exp(1.3 * 2.1),
        ),
    ],
)
def test_official_pure_eml_numeric_equivalence(expr: sp.Expr, expected: float) -> None:
    tree = sympy_to_eml_tree(expr)
    actual = evaluate_official_eml_tree(tree, {"x": 1.3, "y": 2.1, "z": 3.2})

    assert math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8)


def test_float_compiles_through_rational_decimal_string() -> None:
    assert_pure_eml_tree(sp.Float("0.5"))
    actual = evaluate_official_eml_tree(sympy_to_eml_tree(sp.Float("0.5")), {})

    assert math.isclose(actual, 0.5, rel_tol=1e-8, abs_tol=1e-8)


def test_trig_remains_unsupported() -> None:
    x = sp.Symbol("x")

    with pytest.raises(OfficialEmlCompilerError, match="unsupported"):
        compile_to_official_eml_subtree(sp.sin(x))
