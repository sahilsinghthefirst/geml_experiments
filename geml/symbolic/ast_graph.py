"""Normal AST binary-tree conversion from SymPy expressions."""

from __future__ import annotations

from typing import Literal

import sympy as sp
from pydantic import BaseModel, Field

from geml.symbolic.metrics import TreeStatistics, compute_tree_statistics

type MetadataValue = str | int | float | bool | list[str]

NodeKind = Literal["symbol", "constant", "operator"]


class UnsupportedExpressionError(ValueError):
    """Raised when a SymPy expression contains unsupported syntax."""


class AstNode(BaseModel):
    """A node in a normal AST binary tree."""

    id: int = Field(ge=0)
    label: str
    kind: NodeKind
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class AstEdge(BaseModel):
    """A directed parent-to-child AST edge."""

    source: int = Field(ge=0)
    target: int = Field(ge=0)
    position: int = Field(ge=0)


class AstTree(BaseModel):
    """Serializable normal AST tree representation."""

    nodes: list[AstNode]
    edges: list[AstEdge]
    root_id: int
    node_labels: dict[int, str]
    metadata: dict[str, MetadataValue]
    statistics: TreeStatistics


def sympy_to_ast_tree(expr: sp.Expr | str | int) -> AstTree:
    """Convert a supported SymPy expression into a deterministic binary AST tree."""
    sympy_expr = sp.sympify(expr)
    builder = _AstTreeBuilder()
    root_id = builder.convert(sympy_expr)
    edge_pairs = [(edge.source, edge.target) for edge in builder.edges]
    operator_node_ids = [node.id for node in builder.nodes if node.kind == "operator"]
    statistics = compute_tree_statistics(
        root_id=root_id,
        node_ids=[node.id for node in builder.nodes],
        edges=edge_pairs,
        operator_node_ids=operator_node_ids,
    )

    return AstTree(
        nodes=builder.nodes,
        edges=builder.edges,
        root_id=root_id,
        node_labels={node.id: node.label for node in builder.nodes},
        metadata={
            "converter": "ast_binary_v0",
            "expression": str(sympy_expr),
            "srepr": sp.srepr(sympy_expr),
            "supported_operators": ["add", "mul", "pow", "exp", "log"],
        },
        statistics=statistics,
    )


class _AstTreeBuilder:
    def __init__(self) -> None:
        self.nodes: list[AstNode] = []
        self.edges: list[AstEdge] = []
        self._next_node_id = 0

    def convert(self, expr: sp.Expr) -> int:
        if expr.is_Symbol:
            return self._add_node(
                label=str(expr),
                kind="symbol",
                metadata={"sympy_func": expr.func.__name__},
            )

        if expr.is_Integer:
            return self._add_node(
                label=str(expr),
                kind="constant",
                metadata={"sympy_func": expr.func.__name__, "value": int(expr)},
            )

        if expr.func == sp.Add:
            return self._convert_nary_operator(label="add", args=expr.args)

        if expr.func == sp.Mul:
            return self._convert_nary_operator(label="mul", args=expr.args)

        if expr.func == sp.Pow:
            return self._convert_fixed_arity_operator(label="pow", args=expr.args, expected_arity=2)

        if expr.func == sp.exp:
            return self._convert_fixed_arity_operator(label="exp", args=expr.args, expected_arity=1)

        if expr.func == sp.log:
            return self._convert_fixed_arity_operator(label="log", args=expr.args, expected_arity=1)

        raise UnsupportedExpressionError(
            f"unsupported SymPy expression node {expr.func.__name__}: {expr}"
        )

    def _convert_nary_operator(self, *, label: str, args: tuple[sp.Expr, ...]) -> int:
        if len(args) < 2:
            raise UnsupportedExpressionError(f"{label} requires at least 2 arguments")

        current_root_id = self._convert_binary_operator(label=label, left=args[0], right=args[1])
        for arg in args[2:]:
            parent_id = self._add_operator_node(label=label, arity=2)
            self._add_edge(source=parent_id, target=current_root_id, position=0)
            right_id = self.convert(arg)
            self._add_edge(source=parent_id, target=right_id, position=1)
            current_root_id = parent_id
        return current_root_id

    def _convert_binary_operator(self, *, label: str, left: sp.Expr, right: sp.Expr) -> int:
        parent_id = self._add_operator_node(label=label, arity=2)
        left_id = self.convert(left)
        right_id = self.convert(right)
        self._add_edge(source=parent_id, target=left_id, position=0)
        self._add_edge(source=parent_id, target=right_id, position=1)
        return parent_id

    def _convert_fixed_arity_operator(
        self,
        *,
        label: str,
        args: tuple[sp.Expr, ...],
        expected_arity: int,
    ) -> int:
        if len(args) != expected_arity:
            raise UnsupportedExpressionError(
                f"{label} expected {expected_arity} arguments, got {len(args)}"
            )

        parent_id = self._add_operator_node(label=label, arity=expected_arity)
        for position, arg in enumerate(args):
            child_id = self.convert(arg)
            self._add_edge(source=parent_id, target=child_id, position=position)
        return parent_id

    def _add_operator_node(self, *, label: str, arity: int) -> int:
        return self._add_node(
            label=label,
            kind="operator",
            metadata={"arity": arity, "sympy_func": label},
        )

    def _add_node(
        self,
        *,
        label: str,
        kind: NodeKind,
        metadata: dict[str, MetadataValue],
    ) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        self.nodes.append(AstNode(id=node_id, label=label, kind=kind, metadata=metadata))
        return node_id

    def _add_edge(self, *, source: int, target: int, position: int) -> None:
        self.edges.append(AstEdge(source=source, target=target, position=position))
