"""Source-expression IR for Goal 4 e-graph experiments."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import sympy as sp


@dataclass(frozen=True, slots=True)
class Var:
    """A source variable."""

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("variable name must not be empty")


@dataclass(frozen=True, slots=True)
class Const:
    """An exact integer or rational constant."""

    value: Fraction

    def __init__(self, value: int | Fraction) -> None:
        object.__setattr__(self, "value", exact_fraction(value))


@dataclass(frozen=True, slots=True)
class Add:
    """Binary addition."""

    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class Mul:
    """Binary multiplication."""

    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class Neg:
    """Unary negation."""

    value: Expr


@dataclass(frozen=True, slots=True)
class Sub:
    """Binary subtraction."""

    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class Div:
    """Binary division."""

    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class Pow:
    """Binary exponentiation."""

    base: Expr
    exponent: Expr


@dataclass(frozen=True, slots=True)
class Exp:
    """Unary exponential."""

    value: Expr


@dataclass(frozen=True, slots=True)
class Log:
    """Unary logarithm."""

    value: Expr


type Expr = Var | Const | Add | Mul | Neg | Sub | Div | Pow | Exp | Log
type UnaryExpr = Neg | Exp | Log
type BinaryExpr = Add | Mul | Sub | Div | Pow


def exact_fraction(value: int | Fraction | sp.Integer | sp.Rational | sp.Float) -> Fraction:
    """Convert numeric input into an exact Fraction for rewrite logic."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, sp.Integer):
        return Fraction(int(value), 1)
    if isinstance(value, sp.Rational):
        return Fraction(int(value.p), int(value.q))
    if isinstance(value, sp.Float):
        return Fraction(str(value))
    raise TypeError(f"unsupported constant value type: {type(value).__name__}")


def from_sympy(expr: sp.Expr | int | Fraction) -> Expr:
    """Convert a supported SymPy expression into GEML e-graph IR without rewriting it."""
    sympy_expr = sp.sympify(expr)
    if sympy_expr.is_Symbol:
        return Var(str(sympy_expr))
    if isinstance(sympy_expr, sp.Integer):
        return Const(exact_fraction(sympy_expr))
    if isinstance(sympy_expr, sp.Rational):
        return Const(exact_fraction(sympy_expr))
    if isinstance(sympy_expr, sp.Float):
        return Const(exact_fraction(sympy_expr))

    if sympy_expr.func == sp.Add:
        return _fold_binary(sympy_expr.args, Add, "Add")
    if sympy_expr.func == sp.Mul:
        return _fold_binary(sympy_expr.args, Mul, "Mul")
    if sympy_expr.func == sp.Pow and len(sympy_expr.args) == 2:
        return Pow(from_sympy(sympy_expr.args[0]), from_sympy(sympy_expr.args[1]))
    if sympy_expr.func == sp.exp and len(sympy_expr.args) == 1:
        return Exp(from_sympy(sympy_expr.args[0]))
    if sympy_expr.func == sp.log and len(sympy_expr.args) == 1:
        return Log(from_sympy(sympy_expr.args[0]))

    raise ValueError(f"unsupported SymPy expression for e-graph IR: {sympy_expr!r}")


def to_sympy(expr: Expr) -> sp.Expr:
    """Convert GEML e-graph IR back to a SymPy expression without intentional rewriting."""
    if isinstance(expr, Var):
        return sp.Symbol(expr.name)
    if isinstance(expr, Const):
        if expr.value.denominator == 1:
            return sp.Integer(expr.value.numerator)
        return sp.Rational(expr.value.numerator, expr.value.denominator)
    if isinstance(expr, Add):
        return sp.Add(to_sympy(expr.left), to_sympy(expr.right), evaluate=False)
    if isinstance(expr, Mul):
        return sp.Mul(to_sympy(expr.left), to_sympy(expr.right), evaluate=False)
    if isinstance(expr, Neg):
        return sp.Mul(sp.Integer(-1), to_sympy(expr.value), evaluate=False)
    if isinstance(expr, Sub):
        return sp.Add(
            to_sympy(expr.left),
            sp.Mul(sp.Integer(-1), to_sympy(expr.right), evaluate=False),
            evaluate=False,
        )
    if isinstance(expr, Div):
        return sp.Mul(
            to_sympy(expr.left),
            sp.Pow(to_sympy(expr.right), sp.Integer(-1), evaluate=False),
            evaluate=False,
        )
    if isinstance(expr, Pow):
        return sp.Pow(to_sympy(expr.base), to_sympy(expr.exponent), evaluate=False)
    if isinstance(expr, Exp):
        return sp.exp(to_sympy(expr.value), evaluate=False)
    if isinstance(expr, Log):
        return sp.log(to_sympy(expr.value), evaluate=False)
    raise TypeError(f"unsupported IR expression type: {type(expr).__name__}")


def display(expr: Expr) -> str:
    """Return a stable canonical display string for the IR tree."""
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Const):
        return _display_fraction(expr.value)
    if isinstance(expr, Add):
        return f"Add({display(expr.left)},{display(expr.right)})"
    if isinstance(expr, Mul):
        return f"Mul({display(expr.left)},{display(expr.right)})"
    if isinstance(expr, Neg):
        return f"Neg({display(expr.value)})"
    if isinstance(expr, Sub):
        return f"Sub({display(expr.left)},{display(expr.right)})"
    if isinstance(expr, Div):
        return f"Div({display(expr.left)},{display(expr.right)})"
    if isinstance(expr, Pow):
        return f"Pow({display(expr.base)},{display(expr.exponent)})"
    if isinstance(expr, Exp):
        return f"Exp({display(expr.value)})"
    if isinstance(expr, Log):
        return f"Log({display(expr.value)})"
    raise TypeError(f"unsupported IR expression type: {type(expr).__name__}")


def node_count(expr: Expr) -> int:
    """Count IR tree nodes."""
    return 1 + sum(node_count(child) for child in children(expr))


def children(expr: Expr) -> tuple[Expr, ...]:
    """Return ordered child expressions."""
    if isinstance(expr, Var | Const):
        return ()
    if isinstance(expr, Neg | Exp | Log):
        return (expr.value,)
    if isinstance(expr, Add | Mul | Sub | Div):
        return (expr.left, expr.right)
    if isinstance(expr, Pow):
        return (expr.base, expr.exponent)
    raise TypeError(f"unsupported IR expression type: {type(expr).__name__}")


def _fold_binary(
    args: tuple[sp.Expr, ...],
    constructor: type[Add] | type[Mul],
    label: str,
) -> Expr:
    if len(args) < 2:
        raise ValueError(f"{label} requires at least two arguments")
    current = constructor(from_sympy(args[0]), from_sympy(args[1]))
    for arg in args[2:]:
        current = constructor(current, from_sympy(arg))
    return current


def _display_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
