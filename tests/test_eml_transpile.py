"""Tests for restricted EML binary-tree conversion."""

from __future__ import annotations

import pytest
import sympy as sp
from geml.symbolic.ast_graph import sympy_to_ast_tree
from geml.symbolic.eml_transpile import (
    UnsupportedExpressionError,
    eml_alpha,
    simplify_eml_tree,
    sympy_to_eml_tree,
)


def assert_eml_equivalent(expr: sp.Expr) -> None:
    tree = sympy_to_eml_tree(expr)
    simplified = simplify_eml_tree(tree)
    assert sp.simplify(simplified - expr, inverse=True) == 0


def assert_internal_nodes_are_eml(expr: sp.Expr) -> None:
    tree = sympy_to_eml_tree(expr)
    internal_node_ids = {edge.source for edge in tree.edges}
    assert internal_node_ids
    for node in tree.nodes:
        if node.id in internal_node_ids:
            assert node.kind == "eml"
            assert node.label == "eml"


def assert_connected_tree_shape(expr: sp.Expr) -> None:
    tree = sympy_to_eml_tree(expr)
    assert tree.statistics.edge_count == tree.statistics.node_count - 1

    child_counts = {node.id: 0 for node in tree.nodes}
    for edge in tree.edges:
        child_counts[edge.source] += 1

    for node in tree.nodes:
        if node.kind == "eml":
            assert child_counts[node.id] == 2
        else:
            assert child_counts[node.id] == 0


def test_convert_variable_leaf() -> None:
    x = sp.Symbol("x")

    tree = sympy_to_eml_tree(x)

    assert tree.node_labels[tree.root_id] == "x"
    assert tree.statistics.node_count == 1
    assert tree.statistics.edge_count == 0
    assert tree.statistics.depth == 0
    assert tree.statistics.leaf_count == 1
    assert tree.alpha == 1.0
    assert simplify_eml_tree(tree) == x


def test_convert_constant_one_leaf() -> None:
    tree = sympy_to_eml_tree(1)

    assert tree.node_labels[tree.root_id] == "1"
    assert tree.statistics.node_count == 1
    assert tree.statistics.leaf_count == 1
    assert tree.alpha == 1.0
    assert simplify_eml_tree(tree) == 1


def test_convert_x_plus_one() -> None:
    x = sp.Symbol("x")
    expr = sp.Add(x, 1, evaluate=False)

    tree = sympy_to_eml_tree(expr)

    assert tree.node_labels[tree.root_id] == "eml"
    assert tree.statistics.node_count == 3
    assert tree.statistics.edge_count == 2
    assert tree.statistics.depth == 1
    assert tree.statistics.leaf_count == 2
    assert tree.alpha == 1.0
    assert_internal_nodes_are_eml(expr)
    assert_connected_tree_shape(expr)
    assert_eml_equivalent(expr)


def test_convert_x_times_y() -> None:
    x, y = sp.symbols("x y")
    expr = sp.Mul(x, y, evaluate=False)

    tree = sympy_to_eml_tree(expr)

    assert tree.node_labels[tree.root_id] == "eml"
    assert tree.statistics.node_count == 3
    assert tree.statistics.edge_count == 2
    assert tree.statistics.depth == 1
    assert tree.statistics.leaf_count == 2
    assert tree.alpha == 1.0
    assert_internal_nodes_are_eml(expr)
    assert_connected_tree_shape(expr)
    assert_eml_equivalent(expr)


def test_convert_exp_x() -> None:
    x = sp.Symbol("x")
    expr = sp.exp(x, evaluate=False)

    tree = sympy_to_eml_tree(expr)

    assert tree.node_labels[tree.root_id] == "eml"
    assert tree.statistics.node_count == 3
    assert tree.statistics.edge_count == 2
    assert tree.statistics.depth == 1
    assert tree.statistics.leaf_count == 2
    assert tree.alpha == 1.5
    assert_internal_nodes_are_eml(expr)
    assert_connected_tree_shape(expr)
    assert_eml_equivalent(expr)


def test_convert_log_x_plus_one() -> None:
    x = sp.Symbol("x")
    expr = sp.log(sp.Add(x, 1, evaluate=False), evaluate=False)

    tree = sympy_to_eml_tree(expr)

    assert tree.node_labels[tree.root_id] == "eml"
    assert tree.statistics.node_count > sympy_to_ast_tree(expr).statistics.node_count
    assert tree.alpha > 1.0
    assert_internal_nodes_are_eml(expr)
    assert_connected_tree_shape(expr)
    assert_eml_equivalent(expr)


def test_convert_product_of_two_sums() -> None:
    x, y = sp.symbols("x y")
    expr = sp.Mul(
        sp.Add(x, 1, evaluate=False),
        sp.Add(y, 1, evaluate=False),
        evaluate=False,
    )

    tree = sympy_to_eml_tree(expr)
    ast_tree = sympy_to_ast_tree(expr)

    assert tree.node_labels[tree.root_id] == "eml"
    assert tree.statistics.node_count == 3
    assert tree.alpha == tree.statistics.node_count / ast_tree.statistics.node_count
    assert_internal_nodes_are_eml(expr)
    assert_connected_tree_shape(expr)
    assert_eml_equivalent(expr)


def test_alpha_matches_node_count_ratio() -> None:
    x = sp.Symbol("x")
    expr = sp.exp(x, evaluate=False)

    eml_tree = sympy_to_eml_tree(expr)
    ast_tree = sympy_to_ast_tree(expr)

    assert eml_tree.alpha == eml_tree.statistics.node_count / ast_tree.statistics.node_count
    assert eml_alpha(expr) == eml_tree.alpha


def test_unsupported_expression_raises_clear_error() -> None:
    x = sp.Symbol("x")

    with pytest.raises(UnsupportedExpressionError, match="unsupported SymPy expression node"):
        sympy_to_eml_tree(sp.sin(x))


def test_unsupported_pow_raises_clear_error() -> None:
    x = sp.Symbol("x")

    with pytest.raises(UnsupportedExpressionError, match="unsupported SymPy expression node"):
        sympy_to_eml_tree(sp.Pow(x, 2, evaluate=False))


def test_unsupported_integer_constant_raises_clear_error() -> None:
    with pytest.raises(UnsupportedExpressionError, match="supports only integer constant 1"):
        sympy_to_eml_tree(2)
