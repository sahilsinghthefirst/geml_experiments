"""Tests for SymPy to normal AST binary-tree conversion."""

from __future__ import annotations

import sympy as sp
from geml.symbolic.ast_graph import AstTree, sympy_to_ast_tree


def child_labels(tree: AstTree, node_id: int) -> list[str]:
    edges = sorted(
        (edge for edge in tree.edges if edge.source == node_id),
        key=lambda edge: edge.position,
    )
    return [tree.node_labels[edge.target] for edge in edges]


def assert_binary_operator_nodes(tree: AstTree) -> None:
    for node in tree.nodes:
        if node.kind != "operator":
            continue
        child_count = sum(1 for edge in tree.edges if edge.source == node.id)
        assert child_count <= 2


def test_convert_x_plus_one() -> None:
    x = sp.Symbol("x")

    tree = sympy_to_ast_tree(sp.Add(x, 1, evaluate=False))

    assert tree.node_labels[tree.root_id] == "add"
    assert child_labels(tree, tree.root_id) == ["x", "1"]
    assert tree.statistics.node_count == 3
    assert tree.statistics.edge_count == 2
    assert tree.statistics.depth == 1
    assert tree.statistics.leaf_count == 2
    assert tree.statistics.operator_count == 1


def test_convert_x_times_y() -> None:
    x, y = sp.symbols("x y")

    tree = sympy_to_ast_tree(sp.Mul(x, y, evaluate=False))

    assert tree.node_labels[tree.root_id] == "mul"
    assert child_labels(tree, tree.root_id) == ["x", "y"]
    assert tree.statistics.node_count == 3
    assert tree.statistics.edge_count == 2
    assert tree.statistics.depth == 1
    assert tree.statistics.leaf_count == 2
    assert tree.statistics.operator_count == 1


def test_convert_exp_x() -> None:
    x = sp.Symbol("x")

    tree = sympy_to_ast_tree(sp.exp(x, evaluate=False))

    assert tree.node_labels[tree.root_id] == "exp"
    assert child_labels(tree, tree.root_id) == ["x"]
    assert tree.statistics.node_count == 2
    assert tree.statistics.edge_count == 1
    assert tree.statistics.depth == 1
    assert tree.statistics.leaf_count == 1
    assert tree.statistics.operator_count == 1


def test_convert_log_x_plus_one() -> None:
    x = sp.Symbol("x")
    expr = sp.log(sp.Add(x, 1, evaluate=False), evaluate=False)

    tree = sympy_to_ast_tree(expr)

    assert tree.node_labels[tree.root_id] == "log"
    assert child_labels(tree, tree.root_id) == ["add"]
    assert tree.statistics.node_count == 4
    assert tree.statistics.edge_count == 3
    assert tree.statistics.depth == 2
    assert tree.statistics.leaf_count == 2
    assert tree.statistics.operator_count == 2


def test_convert_product_of_two_sums() -> None:
    x, y = sp.symbols("x y")
    expr = sp.Mul(
        sp.Add(x, 1, evaluate=False),
        sp.Add(y, 1, evaluate=False),
        evaluate=False,
    )

    tree = sympy_to_ast_tree(expr)

    assert tree.node_labels[tree.root_id] == "mul"
    assert child_labels(tree, tree.root_id) == ["add", "add"]
    assert tree.statistics.node_count == 7
    assert tree.statistics.edge_count == 6
    assert tree.statistics.depth == 2
    assert tree.statistics.leaf_count == 4
    assert tree.statistics.operator_count == 3


def test_nary_add_is_converted_to_binary_tree() -> None:
    x, y = sp.symbols("x y")
    expr = sp.Add(x, y, 1, evaluate=False)

    tree = sympy_to_ast_tree(expr)

    assert tree.node_labels[tree.root_id] == "add"
    assert tree.statistics.node_count == 5
    assert tree.statistics.edge_count == 4
    assert tree.statistics.depth == 2
    assert tree.statistics.leaf_count == 3
    assert tree.statistics.operator_count == 2
    assert_binary_operator_nodes(tree)


def test_convert_pow() -> None:
    x = sp.Symbol("x")
    expr = sp.Pow(sp.Add(x, 1, evaluate=False), 2, evaluate=False)

    tree = sympy_to_ast_tree(expr)

    assert tree.node_labels[tree.root_id] == "pow"
    assert child_labels(tree, tree.root_id) == ["add", "2"]
    assert tree.statistics.node_count == 5
    assert tree.statistics.edge_count == 4
    assert tree.statistics.depth == 2
    assert tree.statistics.leaf_count == 3
    assert tree.statistics.operator_count == 2
