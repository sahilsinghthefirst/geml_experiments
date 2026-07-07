"""Tests for shared srepr parsing."""

from __future__ import annotations

import sympy as sp
from geml.symbolic.srepr import build_symbol_locals, parse_expression_or_srepr, parse_srepr


def test_parse_srepr_preserves_pow_structure() -> None:
    expr = parse_srepr("Pow(Symbol('x'), Integer(2))")

    assert expr.func == sp.Pow
    assert expr.args[0] == sp.Symbol("x")
    assert expr.args[1] == sp.Integer(2)
    assert sp.srepr(expr) == "Pow(Symbol('x'), Integer(2))"


def test_parse_expression_or_srepr_prefers_pow_srepr() -> None:
    expr, source = parse_expression_or_srepr(
        expression="x*x",
        srepr="Pow(Symbol('x'), Integer(2))",
        symbol_locals=build_symbol_locals(("x",)),
    )

    assert source == "srepr"
    assert expr.func == sp.Pow
    assert sp.srepr(expr) == "Pow(Symbol('x'), Integer(2))"
