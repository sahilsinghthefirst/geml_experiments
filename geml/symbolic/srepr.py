"""Shared SymPy ``srepr`` parsing utilities.

Generated corpora use ``srepr`` as the authoritative structural serialization.
This module reconstructs that structure with unevaluated Add/Mul/Pow/exp/log
nodes so later AST, EML, DAG, and stratified analyses all see the same tree.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import sympy as sp

type SourceSerialization = Literal["srepr", "expression"]


def build_symbol_locals(symbol_names: Iterable[str]) -> dict[str, sp.Symbol]:
    """Build SymPy parser locals for generated symbol names."""
    return {name: sp.Symbol(name) for name in symbol_names}


def build_srepr_locals() -> dict[str, Any]:
    """Build locals that reconstruct generated srepr without evaluating operators."""

    def add(*args: sp.Expr, **_: object) -> sp.Expr:
        return sp.Add(*args, evaluate=False)

    def mul(*args: sp.Expr, **_: object) -> sp.Expr:
        return sp.Mul(*args, evaluate=False)

    def pow_expr(base: sp.Expr, exponent: sp.Expr, **_: object) -> sp.Expr:
        return sp.Pow(base, exponent, evaluate=False)

    def exp_expr(arg: sp.Expr, **_: object) -> sp.Expr:
        return sp.exp(arg, evaluate=False)

    def log_expr(arg: sp.Expr, **_: object) -> sp.Expr:
        return sp.log(arg, evaluate=False)

    return {
        "Add": add,
        "Float": sp.Float,
        "Integer": sp.Integer,
        "Mul": mul,
        "Pow": pow_expr,
        "Rational": sp.Rational,
        "Symbol": sp.Symbol,
        "exp": exp_expr,
        "log": log_expr,
    }


def parse_srepr(srepr: str) -> sp.Expr:
    """Parse generated SymPy ``srepr`` while preserving unevaluated operators."""
    return sp.sympify(srepr, locals=build_srepr_locals(), evaluate=False)


def parse_expression_or_srepr(
    *,
    expression: str,
    srepr: str | None,
    symbol_locals: dict[str, sp.Symbol],
) -> tuple[sp.Expr, SourceSerialization]:
    """Parse generated input, preferring authoritative srepr over display strings."""
    if srepr:
        return parse_srepr(srepr), "srepr"
    return sp.sympify(expression, locals=symbol_locals, evaluate=False), "expression"
