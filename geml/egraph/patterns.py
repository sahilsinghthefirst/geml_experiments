"""Pattern matching over e-classes."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from fractions import Fraction

from geml.egraph.egraph import EGraph, ENode
from geml.egraph.ir import Add, Const, Div, Exp, Expr, Log, Mul, Neg, Pow, Sub, Var


@dataclass(frozen=True, slots=True)
class PatternVar:
    """A pattern variable such as ?a."""

    name: str

    def __post_init__(self) -> None:
        if not self.name.startswith("?") or len(self.name) == 1:
            raise ValueError("pattern variable names must look like '?a'")


type Pattern = PatternVar | Expr
type Substitution = dict[str, int]

_PATTERN_VAR_RE = re.compile(r"\?([A-Za-z_][A-Za-z0-9_]*)")


def pvar(name: str) -> PatternVar:
    """Create a pattern variable, accepting either 'a' or '?a'."""
    return PatternVar(name if name.startswith("?") else f"?{name}")


def parse_pattern(source: str) -> Pattern:
    """Parse a small prefix pattern such as Add(?a, ?b)."""
    rewritten = _PATTERN_VAR_RE.sub(r'P("\1")', source)
    parsed = ast.parse(rewritten, mode="eval")
    return _from_ast(parsed.body)


def match_eclass(
    egraph: EGraph,
    pattern: Pattern,
    eclass_id: int,
    substitution: Substitution | None = None,
) -> list[Substitution]:
    """Match a pattern against every node in an e-class."""
    initial = {} if substitution is None else dict(substitution)
    return _match(egraph, pattern, egraph.find(eclass_id), initial)


def add_pattern(egraph: EGraph, pattern: Pattern, substitution: Substitution) -> int:
    """Instantiate a pattern into an e-graph and return its e-class id."""
    if isinstance(pattern, PatternVar):
        try:
            return egraph.find(substitution[pattern.name])
        except KeyError as exc:
            raise ValueError(f"unbound pattern variable {pattern.name}") from exc
    enode, child_patterns = pattern_to_enode_shape(pattern)
    child_ids = tuple(add_pattern(egraph, child, substitution) for child in child_patterns)
    return egraph.add_enode(ENode(op=enode.op, children=child_ids, value=enode.value))


def pattern_to_enode_shape(pattern: Pattern) -> tuple[ENode, tuple[Pattern, ...]]:
    """Return the e-node shape and ordered child patterns for a non-variable pattern."""
    if isinstance(pattern, PatternVar):
        raise TypeError("pattern variables do not have e-node shapes")
    if isinstance(pattern, Var):
        return ENode("var", value=pattern.name), ()
    if isinstance(pattern, Const):
        return ENode("const", value=pattern.value), ()
    if isinstance(pattern, Add):
        return ENode("add"), (pattern.left, pattern.right)
    if isinstance(pattern, Mul):
        return ENode("mul"), (pattern.left, pattern.right)
    if isinstance(pattern, Neg):
        return ENode("neg"), (pattern.value,)
    if isinstance(pattern, Sub):
        return ENode("sub"), (pattern.left, pattern.right)
    if isinstance(pattern, Div):
        return ENode("div"), (pattern.left, pattern.right)
    if isinstance(pattern, Pow):
        return ENode("pow"), (pattern.base, pattern.exponent)
    if isinstance(pattern, Exp):
        return ENode("exp"), (pattern.value,)
    if isinstance(pattern, Log):
        return ENode("log"), (pattern.value,)
    raise TypeError(f"unsupported pattern type: {type(pattern).__name__}")


def _match(
    egraph: EGraph,
    pattern: Pattern,
    eclass_id: int,
    substitution: Substitution,
) -> list[Substitution]:
    root_id = egraph.find(eclass_id)
    if isinstance(pattern, PatternVar):
        existing = substitution.get(pattern.name)
        if existing is None:
            next_substitution = dict(substitution)
            next_substitution[pattern.name] = root_id
            return [next_substitution]
        return [substitution] if egraph.find(existing) == root_id else []

    pattern_shape, child_patterns = pattern_to_enode_shape(pattern)
    matches: list[Substitution] = []
    for node in egraph.get_eclass_nodes(root_id):
        if not _same_shape(pattern_shape, node) or len(node.children) != len(child_patterns):
            continue
        partials = [dict(substitution)]
        for child_pattern, child_id in zip(child_patterns, node.children, strict=True):
            next_partials: list[Substitution] = []
            for partial in partials:
                next_partials.extend(_match(egraph, child_pattern, child_id, partial))
            partials = next_partials
            if not partials:
                break
        matches.extend(partials)
    return matches


def _same_shape(pattern_node: ENode, node: ENode) -> bool:
    return pattern_node.op == node.op and pattern_node.value == node.value


def _from_ast(node: ast.AST) -> Pattern:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "P":
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant):
                raise ValueError("P expects one variable-name string")
            value = node.args[0].value
            if not isinstance(value, str):
                raise ValueError("P expects one variable-name string")
            return pvar(value)
        args = [_from_ast(arg) for arg in node.args]
        return _call_pattern(node.func.id, args)
    if isinstance(node, ast.Name):
        return Var(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return Const(Fraction(node.value, 1))
    raise ValueError(f"unsupported pattern syntax: {ast.dump(node)}")


def _call_pattern(name: str, args: list[Pattern]) -> Pattern:
    if name == "Add" and len(args) == 2:
        return Add(args[0], args[1])
    if name == "Mul" and len(args) == 2:
        return Mul(args[0], args[1])
    if name == "Neg" and len(args) == 1:
        return Neg(args[0])
    if name == "Sub" and len(args) == 2:
        return Sub(args[0], args[1])
    if name == "Div" and len(args) == 2:
        return Div(args[0], args[1])
    if name == "Pow" and len(args) == 2:
        return Pow(args[0], args[1])
    if name == "Exp" and len(args) == 1:
        return Exp(args[0])
    if name == "Log" and len(args) == 1:
        return Log(args[0])
    if name == "Const" and len(args) == 1 and isinstance(args[0], Const):
        return args[0]
    raise ValueError(f"unsupported pattern call: {name}/{len(args)}")
