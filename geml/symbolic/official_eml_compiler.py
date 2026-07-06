"""Official recursive pure EML compiler port.

The macro definitions in this module are ported from:
VA00/SymbolicRegressionPackage/EML_toolkit/EmL_compiler/eml_compiler_v4.py

They are implemented as recursive constructors for GEML's native EmlTree
representation. Helper names such as eml_log, eml_exp, eml_add, and eml_mul are
macros only; the final tree contains only internal ``eml`` nodes plus variable
and constant-one leaves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import sympy as sp

from geml.symbolic.ast_graph import sympy_to_ast_tree
from geml.symbolic.eml_nodes import EmlEdge, EmlNode, EmlTree, MetadataValue
from geml.symbolic.metrics import compute_tree_statistics

PureNodeKind = Literal["eml", "variable", "constant"]


class OfficialEmlCompilerError(ValueError):
    """Raised when an expression cannot be compiled into official pure EML."""


@dataclass(frozen=True)
class PureEmlNode:
    """Immutable pre-materialization pure EML subtree."""

    label: str
    kind: PureNodeKind
    children: tuple[PureEmlNode, ...] = field(default_factory=tuple)


def eml_primitive(a: PureEmlNode, b: PureEmlNode) -> PureEmlNode:
    """EML(a, b) primitive."""
    return PureEmlNode(label="eml", kind="eml", children=(a, b))


def eml_one() -> PureEmlNode:
    """Official eml_one() = "1"."""
    return PureEmlNode(label="1", kind="constant")


def eml_variable(name: str) -> PureEmlNode:
    """Create a variable leaf."""
    return PureEmlNode(label=name, kind="variable")


def eml_exp(z: PureEmlNode) -> PureEmlNode:
    """Official eml_exp(z) = EML(z, 1)."""
    return eml_primitive(z, eml_one())


def eml_log(z: PureEmlNode) -> PureEmlNode:
    """Official eml_log(z) = EML(1, eml_exp(EML(1, z)))."""
    return eml_primitive(eml_one(), eml_exp(eml_primitive(eml_one(), z)))


def eml_zero() -> PureEmlNode:
    """Official eml_zero() = eml_log(1)."""
    return eml_log(eml_one())


def eml_sub(a: PureEmlNode, b: PureEmlNode) -> PureEmlNode:
    """Official eml_sub(a, b) = EML(eml_log(a), eml_exp(b))."""
    return eml_primitive(eml_log(a), eml_exp(b))


def eml_neg(z: PureEmlNode) -> PureEmlNode:
    """Official eml_neg(z) = eml_sub(eml_zero(), z)."""
    return eml_sub(eml_zero(), z)


def eml_add(a: PureEmlNode, b: PureEmlNode) -> PureEmlNode:
    """Official eml_add(a, b) = eml_sub(a, eml_neg(b))."""
    return eml_sub(a, eml_neg(b))


def eml_inv(z: PureEmlNode) -> PureEmlNode:
    """Official eml_inv(z) = eml_exp(eml_neg(eml_log(z)))."""
    return eml_exp(eml_neg(eml_log(z)))


def eml_mul(a: PureEmlNode, b: PureEmlNode) -> PureEmlNode:
    """Official eml_mul(a, b) = eml_exp(eml_add(eml_log(a), eml_log(b)))."""
    return eml_exp(eml_add(eml_log(a), eml_log(b)))


def eml_div(a: PureEmlNode, b: PureEmlNode) -> PureEmlNode:
    """Official eml_div(a, b) = eml_mul(a, eml_inv(b))."""
    return eml_mul(a, eml_inv(b))


def eml_pow(a: PureEmlNode, b: PureEmlNode) -> PureEmlNode:
    """Official eml_pow(a, b) = eml_exp(eml_mul(b, eml_log(a)))."""
    return eml_exp(eml_mul(b, eml_log(a)))


def eml_int(n: int) -> PureEmlNode:
    """Port of official eml_int(n) binary repeated-doubling construction."""
    if n == 1:
        return eml_one()
    if n == 0:
        return eml_zero()
    if n < 0:
        return eml_neg(eml_int(-n))

    acc: PureEmlNode | None = None
    term = eml_one()
    k = n
    while k > 0:
        if k & 1:
            acc = term if acc is None else eml_add(acc, term)
        term = eml_add(term, term)
        k >>= 1
    if acc is None:
        raise OfficialEmlCompilerError(f"failed to compile integer constant {n}")
    return acc


def eml_rational(p: int, q: int) -> PureEmlNode:
    """Port of official eml_rational(p, q)."""
    if q == 0:
        raise OfficialEmlCompilerError("rational denominator must not be zero")
    if q < 0:
        p = -p
        q = -q
    if q == 1:
        return eml_int(p)

    num = eml_int(abs(p))
    den = eml_int(q)
    val = eml_mul(num, eml_inv(den))
    return eml_neg(val) if p < 0 else val


def compile_to_official_eml_subtree(expr: sp.Expr | str | int | float) -> PureEmlNode:
    """Compile a supported SymPy expression into a pure EML subtree."""
    sympy_expr = sp.sympify(expr)
    return _compile_expr(sympy_expr)


def sympy_to_official_eml_tree(expr: sp.Expr | str | int | float) -> EmlTree:
    """Compile a supported SymPy expression into a native pure EML tree."""
    sympy_expr = sp.sympify(expr)
    root_subtree = compile_to_official_eml_subtree(sympy_expr)
    builder = _OfficialEmlTreeMaterializer()
    root_id = builder.materialize(root_subtree)
    edge_pairs = [(edge.source, edge.target) for edge in builder.edges]
    operator_node_ids = [node.id for node in builder.nodes if node.kind == "eml"]
    statistics = compute_tree_statistics(
        root_id=root_id,
        node_ids=[node.id for node in builder.nodes],
        edges=edge_pairs,
        operator_node_ids=operator_node_ids,
    )
    ast_tree = sympy_to_ast_tree(sympy_expr)
    alpha = statistics.node_count / ast_tree.statistics.node_count

    return EmlTree(
        representation_mode="restricted_eml_pure",
        nodes=builder.nodes,
        edges=builder.edges,
        root_id=root_id,
        node_labels={node.id: node.label for node in builder.nodes},
        metadata={
            "converter": "official_recursive_eml_compiler_v4_port",
            "representation_mode": "restricted_eml_pure",
            "official_source": (
                "VA00/SymbolicRegressionPackage/EML_toolkit/EmL_compiler/eml_compiler_v4.py"
            ),
            "expression": str(sympy_expr),
            "srepr": sp.srepr(sympy_expr),
            "eml_operator": "eml(x, y) = exp(x) - log(y)",
            "alpha_policy": "alpha is valid only for pure trees with no derived leaves",
            "alpha_valid": True,
        },
        statistics=statistics,
        normal_leaf_count=statistics.leaf_count,
        derived_leaf_count=0,
        hidden_compound_leaf_count=0,
        ast_statistics=ast_tree.statistics,
        alpha=alpha,
        alpha_valid=True,
    )


def emit_official_eml_string(tree: EmlTree | PureEmlNode) -> str:
    """Emit a pure EML tree as official-style ``EML[left,right]`` text."""
    if isinstance(tree, PureEmlNode):
        return _emit_subtree(tree)

    nodes_by_id = {node.id: node for node in tree.nodes}
    children_by_id: dict[int, list[tuple[int, int]]] = {node.id: [] for node in tree.nodes}
    for edge in tree.edges:
        children_by_id[edge.source].append((edge.position, edge.target))

    def emit_node(node_id: int) -> str:
        node = nodes_by_id[node_id]
        if node.kind != "eml":
            return node.label
        children = sorted(children_by_id[node_id])
        if len(children) != 2:
            raise ValueError(f"eml node {node_id} must have exactly two children")
        left = emit_node(children[0][1])
        right = emit_node(children[1][1])
        return f"EML[{left},{right}]"

    return emit_node(tree.root_id)


def evaluate_official_eml_tree(tree: EmlTree, values: dict[str, float]) -> float:
    """Numerically evaluate a native pure EML tree using real exp/log."""
    nodes_by_id = {node.id: node for node in tree.nodes}
    children_by_id: dict[int, list[tuple[int, int]]] = {node.id: [] for node in tree.nodes}
    for edge in tree.edges:
        children_by_id[edge.source].append((edge.position, edge.target))

    def evaluate_node(node_id: int) -> float:
        node = nodes_by_id[node_id]
        if node.kind == "constant":
            if node.label != "1":
                raise ValueError(f"unsupported constant leaf {node.label!r}")
            return 1.0
        if node.kind == "variable":
            if node.label not in values:
                raise ValueError(f"missing numeric value for variable {node.label!r}")
            return values[node.label]
        children = sorted(children_by_id[node_id])
        if len(children) != 2:
            raise ValueError(f"eml node {node_id} must have exactly two children")
        left = evaluate_node(children[0][1])
        right = evaluate_node(children[1][1])
        return math.exp(left) - _real_log_with_formal_zero(right)

    return evaluate_node(tree.root_id)


def _compile_expr(expr: sp.Expr) -> PureEmlNode:
    if expr.is_Atom:
        if isinstance(expr, sp.Integer):
            return eml_int(int(expr))
        if isinstance(expr, sp.Rational):
            return eml_rational(int(expr.p), int(expr.q))
        if isinstance(expr, sp.Float):
            rational = sp.Rational(str(expr))
            return eml_rational(int(rational.p), int(rational.q))
        if isinstance(expr, sp.Symbol):
            return eml_variable(expr.name)
        raise OfficialEmlCompilerError(f"unsupported atomic expression: {expr!r}")

    if expr.func == sp.exp and len(expr.args) == 1:
        return eml_exp(_compile_expr(expr.args[0]))

    if expr.func == sp.log and len(expr.args) == 1:
        return eml_log(_compile_expr(expr.args[0]))

    if expr.func == sp.Pow and len(expr.args) == 2:
        base, power = expr.as_base_exp()
        return eml_pow(_compile_expr(base), _compile_expr(power))

    if expr.func == sp.Mul:
        factors = list(expr.args)
        if not factors:
            raise OfficialEmlCompilerError("Mul requires at least one factor")
        division = _binary_division_factors(factors)
        if division is not None:
            numerator, denominator = division
            return eml_div(_compile_expr(numerator), _compile_expr(denominator))
        acc = _compile_expr(factors[0])
        for factor in factors[1:]:
            acc = eml_mul(acc, _compile_expr(factor))
        return acc

    if expr.func == sp.Add:
        terms = _ordered_add_terms(expr)
        if not terms:
            raise OfficialEmlCompilerError("Add requires at least one term")
        subtraction = _binary_subtraction_terms(terms)
        if subtraction is not None:
            left, right = subtraction
            return eml_sub(_compile_expr(left), _compile_expr(right))
        acc = _compile_expr(terms[0])
        for term in terms[1:]:
            acc = eml_add(acc, _compile_expr(term))
        return acc

    raise OfficialEmlCompilerError(
        f"unsupported SymPy expression node {expr.func.__name__}: {expr}"
    )


def _ordered_add_terms(expr: sp.Expr) -> list[sp.Expr]:
    return sorted(expr.args, key=lambda term: (bool(term.is_number), term.sort_key()))


def _binary_subtraction_terms(terms: list[sp.Expr]) -> tuple[sp.Expr, sp.Expr] | None:
    if len(terms) != 2:
        return None
    left, right = terms
    if right.could_extract_minus_sign() and not left.could_extract_minus_sign():
        return left, -right
    if left.could_extract_minus_sign() and not right.could_extract_minus_sign():
        return right, -left
    return None


def _binary_division_factors(factors: list[sp.Expr]) -> tuple[sp.Expr, sp.Expr] | None:
    if len(factors) != 2:
        return None
    left, right = factors
    right_denominator = _inverse_base(right)
    if right_denominator is not None:
        return left, right_denominator
    left_denominator = _inverse_base(left)
    if left_denominator is not None:
        return right, left_denominator
    return None


def _inverse_base(expr: sp.Expr) -> sp.Expr | None:
    if expr.func == sp.Pow and len(expr.args) == 2 and expr.args[1] == sp.Integer(-1):
        return expr.args[0]
    return None


def _emit_subtree(node: PureEmlNode) -> str:
    if node.kind != "eml":
        return node.label
    if len(node.children) != 2:
        raise ValueError("eml subtrees must have exactly two children")
    left = _emit_subtree(node.children[0])
    right = _emit_subtree(node.children[1])
    return f"EML[{left},{right}]"


def _real_log_with_formal_zero(value: float) -> float:
    if value == 0:
        return -math.inf
    return math.log(value)


class _OfficialEmlTreeMaterializer:
    def __init__(self) -> None:
        self.nodes: list[EmlNode] = []
        self.edges: list[EmlEdge] = []
        self._next_node_id = 0

    def materialize(self, subtree: PureEmlNode) -> int:
        metadata: dict[str, MetadataValue]
        if subtree.kind == "constant":
            metadata = {"expression": "1", "sympy_func": "One", "value": 1}
        elif subtree.kind == "variable":
            metadata = {"expression": subtree.label, "sympy_func": "Symbol"}
        else:
            metadata = {"expression": "exp(left) - log(right)", "arity": 2}

        node_id = self._add_node(label=subtree.label, kind=subtree.kind, metadata=metadata)
        for position, child in enumerate(subtree.children):
            child_id = self.materialize(child)
            self.edges.append(EmlEdge(source=node_id, target=child_id, position=position))
        return node_id

    def _add_node(
        self,
        *,
        label: str,
        kind: PureNodeKind,
        metadata: dict[str, MetadataValue],
    ) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        self.nodes.append(EmlNode(id=node_id, label=label, kind=kind, metadata=metadata))
        return node_id
