"""Restricted EML binary-tree conversion from SymPy expressions."""

from __future__ import annotations

import sympy as sp

from geml.symbolic.ast_graph import sympy_to_ast_tree
from geml.symbolic.eml_nodes import EmlEdge, EmlNode, EmlNodeKind, EmlTree, MetadataValue
from geml.symbolic.metrics import compute_tree_statistics
from geml.symbolic.official_eml_compiler import (
    OfficialEmlCompilerError,
    sympy_to_official_eml_tree,
)
from geml.symbolic.representations import EML_REPRESENTATION_MODES, EmlRepresentationMode


class UnsupportedExpressionError(ValueError):
    """Raised when a SymPy expression cannot be represented in restricted EML form."""


def eml_operator(left: sp.Expr, right: sp.Expr) -> sp.Expr:
    """Evaluate the EML operator eml(left, right) = exp(left) - log(right)."""
    return sp.exp(left, evaluate=False) - sp.log(right, evaluate=False)


def sympy_to_eml_tree(
    expr: sp.Expr | str | int | float,
    *,
    representation_mode: EmlRepresentationMode = "restricted_eml_pure",
) -> EmlTree:
    """Convert a supported SymPy expression into restricted EML binary-tree form."""
    _validate_representation_mode(representation_mode)
    sympy_expr = sp.sympify(expr)
    if representation_mode == "restricted_eml_pure":
        try:
            return sympy_to_official_eml_tree(sympy_expr)
        except OfficialEmlCompilerError as exc:
            raise UnsupportedExpressionError(str(exc)) from exc

    builder = _EmlTreeBuilder(representation_mode=representation_mode)
    root_id = builder.convert(sympy_expr)
    edge_pairs = [(edge.source, edge.target) for edge in builder.edges]
    operator_node_ids = [node.id for node in builder.nodes if node.kind == "eml"]
    statistics = compute_tree_statistics(
        root_id=root_id,
        node_ids=[node.id for node in builder.nodes],
        edges=edge_pairs,
        operator_node_ids=operator_node_ids,
    )
    ast_tree = sympy_to_ast_tree(sympy_expr)
    normal_leaf_count = sum(1 for node in builder.nodes if node.kind in {"variable", "constant"})
    derived_leaf_count = sum(1 for node in builder.nodes if node.kind == "derived")
    hidden_compound_leaf_count = sum(
        1
        for node in builder.nodes
        if node.kind == "derived" and node.metadata.get("contains_hidden_compound") is True
    )
    alpha_valid = (
        representation_mode == "restricted_eml_pure"
        and derived_leaf_count == 0
        and hidden_compound_leaf_count == 0
    )
    alpha = statistics.node_count / ast_tree.statistics.node_count if alpha_valid else None

    return EmlTree(
        representation_mode=representation_mode,
        nodes=builder.nodes,
        edges=builder.edges,
        root_id=root_id,
        node_labels={node.id: node.label for node in builder.nodes},
        metadata={
            "converter": "restricted_eml_binary_v0",
            "representation_mode": representation_mode,
            "expression": str(sympy_expr),
            "srepr": sp.srepr(sympy_expr),
            "eml_operator": "eml(x, y) = exp(x) - log(y)",
            "formal_restricted_eml_grammar": (
                "pure ::= variable | 1 | eml(pure, pure); "
                "derived leaves are excluded from pure alpha"
            ),
            "restricted_rules": _restricted_rules(representation_mode),
            "alpha_policy": (
                "alpha is valid only for restricted_eml_pure trees with no derived leaves"
            ),
            "alpha_valid": alpha_valid,
        },
        statistics=statistics,
        normal_leaf_count=normal_leaf_count,
        derived_leaf_count=derived_leaf_count,
        hidden_compound_leaf_count=hidden_compound_leaf_count,
        ast_statistics=ast_tree.statistics,
        alpha=alpha,
        alpha_valid=alpha_valid,
    )


def _restricted_rules(representation_mode: EmlRepresentationMode) -> list[str]:
    rules = [
        "var -> var",
        "1 -> 1",
        "exp(a) -> eml(a, 1)",
        "log(a) -> eml(0, eml(eml(0, a), 1)) using pure EML zero construction",
    ]
    if representation_mode == "restricted_eml_pure":
        return [
            *rules,
            "Add/Mul source operators are unsupported until a valid pure expansion exists",
        ]
    return [
        *rules,
        "Add/Mul -> eml(log(expr), 1) diagnostic derived lift",
    ]


def _validate_representation_mode(representation_mode: str) -> None:
    if representation_mode not in EML_REPRESENTATION_MODES:
        allowed = ", ".join(EML_REPRESENTATION_MODES)
        raise ValueError(
            f"unsupported EML representation mode {representation_mode!r}; use {allowed}"
        )


def evaluate_eml_tree(tree: EmlTree) -> sp.Expr:
    """Evaluate an EML tree back into a SymPy expression using the EML operator."""
    nodes_by_id = {node.id: node for node in tree.nodes}
    children_by_id: dict[int, list[tuple[int, int]]] = {node.id: [] for node in tree.nodes}
    for edge in tree.edges:
        children_by_id[edge.source].append((edge.position, edge.target))

    def evaluate_node(node_id: int) -> sp.Expr:
        node = nodes_by_id[node_id]
        if node.kind != "eml":
            expression = node.metadata.get("expression")
            if not isinstance(expression, str):
                raise ValueError(f"leaf node {node_id} is missing expression metadata")
            return sp.sympify(expression)

        children = sorted(children_by_id[node_id])
        if len(children) != 2:
            raise ValueError(f"eml node {node_id} must have exactly two children")
        left = evaluate_node(children[0][1])
        right = evaluate_node(children[1][1])
        return eml_operator(left, right)

    return evaluate_node(tree.root_id)


def simplify_eml_tree(tree: EmlTree) -> sp.Expr:
    """Simplify an evaluated EML tree using formal inverse simplification."""
    simplified = sp.simplify(evaluate_eml_tree(tree), inverse=True)
    return sp.simplify(sp.expand_log(simplified, force=True), inverse=True)


def eml_alpha(
    expr: sp.Expr | str | int | float,
    *,
    representation_mode: EmlRepresentationMode = "restricted_eml_pure",
) -> float:
    """Return pure/valid alpha = |T_EML| / |T_AST| for a supported expression."""
    tree = sympy_to_eml_tree(expr, representation_mode=representation_mode)
    if tree.alpha is None or not tree.alpha_valid:
        raise ValueError("alpha is valid only for restricted_eml_pure trees without derived leaves")
    return tree.alpha


class _EmlTreeBuilder:
    def __init__(self, *, representation_mode: EmlRepresentationMode) -> None:
        self.representation_mode = representation_mode
        self.nodes: list[EmlNode] = []
        self.edges: list[EmlEdge] = []
        self._next_node_id = 0

    def convert(self, expr: sp.Expr) -> int:
        if expr.is_Symbol:
            return self._add_leaf(
                label=str(expr),
                kind="variable",
                metadata={"expression": str(expr), "sympy_func": expr.func.__name__},
            )

        if expr == sp.Integer(1):
            return self._one()

        if expr.is_Integer:
            raise UnsupportedExpressionError(
                f"restricted EML supports only integer constant 1, got {expr}"
            )

        if expr.func == sp.exp:
            return self._exp(self.convert_single_arg(expr, "exp"))

        if expr.func == sp.log:
            return self._log(self.convert_single_arg(expr, "log"))

        if expr.func == sp.Add:
            return self._convert_add_or_mul(expr, source_operator="add")

        if expr.func == sp.Mul:
            return self._convert_add_or_mul(expr, source_operator="mul")

        raise UnsupportedExpressionError(
            f"unsupported SymPy expression node {expr.func.__name__}: {expr}"
        )

    def _convert_add_or_mul(self, expr: sp.Expr, *, source_operator: str) -> int:
        if self.representation_mode == "restricted_eml_pure":
            raise UnsupportedExpressionError(
                "restricted_eml_pure does not support Add/Mul source operators; "
                "use restricted_eml_with_derived only for diagnostic conversion until "
                "a valid pure expansion exists"
            )
        self._validate_nary_args(expr, source_operator)
        return self._restricted_lift(expr, source_operator=source_operator)

    def convert_single_arg(self, expr: sp.Expr, operator_name: str) -> int:
        if len(expr.args) != 1:
            raise UnsupportedExpressionError(
                f"{operator_name} expected 1 argument, got {len(expr.args)}"
            )
        return self.convert(expr.args[0])

    def _validate_nary_args(self, expr: sp.Expr, operator_name: str) -> None:
        if len(expr.args) < 2:
            raise UnsupportedExpressionError(f"{operator_name} requires at least 2 arguments")
        for arg in expr.args:
            self._validate_supported(arg)

    def _validate_supported(self, expr: sp.Expr) -> None:
        if expr.is_Symbol or expr == sp.Integer(1):
            return
        if expr.is_Integer:
            raise UnsupportedExpressionError(
                f"restricted EML supports only integer constant 1, got {expr}"
            )
        if expr.func in {sp.Add, sp.Mul}:
            self._validate_nary_args(expr, expr.func.__name__.lower())
            return
        if expr.func in {sp.exp, sp.log}:
            if len(expr.args) != 1:
                raise UnsupportedExpressionError(
                    f"{expr.func.__name__} expected 1 argument, got {len(expr.args)}"
                )
            self._validate_supported(expr.args[0])
            return
        raise UnsupportedExpressionError(
            f"unsupported SymPy expression node {expr.func.__name__}: {expr}"
        )

    def _restricted_lift(self, expr: sp.Expr, *, source_operator: str) -> int:
        left_id = self._add_leaf(
            label="log(expr)",
            kind="derived",
            metadata={
                "expression": str(sp.log(expr, evaluate=False)),
                "source_operator": source_operator,
                "source_expression": str(expr),
                "contains_hidden_compound": True,
                "alpha_valid": False,
            },
        )
        return self._eml(left_id, self._one())

    def _zero(self) -> int:
        inner_exp_arg_id = self._eml(self._one(), self._one())
        right_id = self._eml(inner_exp_arg_id, self._one())
        return self._eml(self._one(), right_id)

    def _exp(self, value_id: int) -> int:
        return self._eml(value_id, self._one())

    def _log(self, value_id: int) -> int:
        one_minus_log_id = self._eml(self._zero(), value_id)
        exp_one_minus_log_id = self._exp(one_minus_log_id)
        return self._eml(self._zero(), exp_one_minus_log_id)

    def _one(self) -> int:
        return self._add_leaf(
            label="1",
            kind="constant",
            metadata={"expression": "1", "sympy_func": "One", "value": 1},
        )

    def _eml(self, left_id: int, right_id: int) -> int:
        node_id = self._add_leaf(
            label="eml",
            kind="eml",
            metadata={"expression": "exp(left) - log(right)", "arity": 2},
        )
        self._add_edge(source=node_id, target=left_id, position=0)
        self._add_edge(source=node_id, target=right_id, position=1)
        return node_id

    def _add_leaf(
        self,
        *,
        label: str,
        kind: EmlNodeKind,
        metadata: dict[str, MetadataValue],
    ) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        self.nodes.append(EmlNode(id=node_id, label=label, kind=kind, metadata=metadata))
        return node_id

    def _add_edge(self, *, source: int, target: int, position: int) -> None:
        self.edges.append(EmlEdge(source=source, target=target, position=position))
