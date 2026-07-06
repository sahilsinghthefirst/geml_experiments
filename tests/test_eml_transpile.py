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
from geml.symbolic.representations import EmlRepresentationMode


def assert_eml_equivalent(
    expr: sp.Expr,
    *,
    representation_mode: EmlRepresentationMode = "restricted_eml_pure",
) -> None:
    tree = sympy_to_eml_tree(expr, representation_mode=representation_mode)
    simplified = simplify_eml_tree(tree)
    assert sp.simplify(simplified - expr, inverse=True) == 0


def assert_internal_nodes_are_eml(
    expr: sp.Expr,
    *,
    representation_mode: EmlRepresentationMode = "restricted_eml_pure",
) -> None:
    tree = sympy_to_eml_tree(expr, representation_mode=representation_mode)
    internal_node_ids = {edge.source for edge in tree.edges}
    assert internal_node_ids
    for node in tree.nodes:
        if node.id in internal_node_ids:
            assert node.kind == "eml"
            assert node.label == "eml"


def assert_connected_tree_shape(
    expr: sp.Expr,
    *,
    representation_mode: EmlRepresentationMode = "restricted_eml_pure",
) -> None:
    tree = sympy_to_eml_tree(expr, representation_mode=representation_mode)
    assert tree.statistics.edge_count == tree.statistics.node_count - 1

    child_counts = {node.id: 0 for node in tree.nodes}
    for edge in tree.edges:
        child_counts[edge.source] += 1

    for node in tree.nodes:
        if node.kind == "eml":
            assert child_counts[node.id] == 2
        else:
            assert child_counts[node.id] == 0


def test_convert_variable_leaf_in_pure_mode() -> None:
    x = sp.Symbol("x")

    tree = sympy_to_eml_tree(x)

    assert tree.representation_mode == "restricted_eml_pure"
    assert tree.node_labels[tree.root_id] == "x"
    assert tree.statistics.node_count == 1
    assert tree.statistics.edge_count == 0
    assert tree.statistics.depth == 0
    assert tree.statistics.leaf_count == 1
    assert tree.normal_leaf_count == 1
    assert tree.derived_leaf_count == 0
    assert tree.hidden_compound_leaf_count == 0
    assert tree.alpha == 1.0
    assert tree.alpha_valid is True
    assert simplify_eml_tree(tree) == x


def test_convert_constant_one_leaf_in_pure_mode() -> None:
    tree = sympy_to_eml_tree(1)

    assert tree.representation_mode == "restricted_eml_pure"
    assert tree.node_labels[tree.root_id] == "1"
    assert tree.statistics.node_count == 1
    assert tree.statistics.leaf_count == 1
    assert tree.normal_leaf_count == 1
    assert tree.derived_leaf_count == 0
    assert tree.hidden_compound_leaf_count == 0
    assert tree.alpha == 1.0
    assert tree.alpha_valid is True
    assert simplify_eml_tree(tree) == 1


def test_convert_exp_x_in_pure_mode() -> None:
    x = sp.Symbol("x")
    expr = sp.exp(x, evaluate=False)

    tree = sympy_to_eml_tree(expr)

    assert tree.node_labels[tree.root_id] == "eml"
    assert tree.statistics.node_count == 3
    assert tree.statistics.edge_count == 2
    assert tree.statistics.depth == 1
    assert tree.statistics.leaf_count == 2
    assert tree.normal_leaf_count == 2
    assert tree.derived_leaf_count == 0
    assert tree.hidden_compound_leaf_count == 0
    assert tree.alpha == 1.5
    assert tree.alpha_valid is True
    assert_internal_nodes_are_eml(expr)
    assert_connected_tree_shape(expr)
    assert_eml_equivalent(expr)


def test_convert_log_x_in_pure_mode() -> None:
    x = sp.Symbol("x")
    expr = sp.log(x, evaluate=False)

    tree = sympy_to_eml_tree(expr)
    ast_tree = sympy_to_ast_tree(expr)

    assert tree.node_labels[tree.root_id] == "eml"
    assert tree.statistics.node_count > ast_tree.statistics.node_count
    assert tree.normal_leaf_count == tree.statistics.leaf_count
    assert tree.derived_leaf_count == 0
    assert tree.hidden_compound_leaf_count == 0
    assert tree.alpha == tree.statistics.node_count / ast_tree.statistics.node_count
    assert tree.alpha_valid is True
    assert_internal_nodes_are_eml(expr)
    assert_connected_tree_shape(expr)
    assert_eml_equivalent(expr)


def test_convert_x_plus_one_in_official_pure_mode() -> None:
    x = sp.Symbol("x")
    expr = sp.Add(x, 1, evaluate=False)

    tree = sympy_to_eml_tree(expr)

    assert tree.representation_mode == "restricted_eml_pure"
    assert tree.node_labels[tree.root_id] == "eml"
    assert tree.normal_leaf_count == tree.statistics.leaf_count
    assert tree.derived_leaf_count == 0
    assert tree.hidden_compound_leaf_count == 0
    assert tree.alpha == tree.statistics.node_count / sympy_to_ast_tree(expr).statistics.node_count
    assert tree.alpha_valid is True
    assert_internal_nodes_are_eml(expr)
    assert_connected_tree_shape(expr)


def test_convert_x_times_y_in_official_pure_mode() -> None:
    x, y = sp.symbols("x y")
    expr = sp.Mul(x, y, evaluate=False)

    tree = sympy_to_eml_tree(expr)

    assert tree.representation_mode == "restricted_eml_pure"
    assert tree.node_labels[tree.root_id] == "eml"
    assert tree.normal_leaf_count == tree.statistics.leaf_count
    assert tree.derived_leaf_count == 0
    assert tree.hidden_compound_leaf_count == 0
    assert tree.alpha == tree.statistics.node_count / sympy_to_ast_tree(expr).statistics.node_count
    assert tree.alpha_valid is True
    assert_internal_nodes_are_eml(expr)
    assert_connected_tree_shape(expr)


def test_convert_log_x_plus_one_in_official_pure_mode() -> None:
    x = sp.Symbol("x")
    expr = sp.log(sp.Add(x, 1, evaluate=False), evaluate=False)

    tree = sympy_to_eml_tree(expr)

    assert tree.representation_mode == "restricted_eml_pure"
    assert tree.node_labels[tree.root_id] == "eml"
    assert tree.normal_leaf_count == tree.statistics.leaf_count
    assert tree.derived_leaf_count == 0
    assert tree.hidden_compound_leaf_count == 0
    assert tree.alpha_valid is True
    assert_internal_nodes_are_eml(expr)
    assert_connected_tree_shape(expr)


def test_with_derived_mode_classifies_add_lift_as_diagnostic() -> None:
    x = sp.Symbol("x")
    expr = sp.Add(x, 1, evaluate=False)

    tree = sympy_to_eml_tree(expr, representation_mode="restricted_eml_with_derived")

    assert tree.representation_mode == "restricted_eml_with_derived"
    assert tree.node_labels[tree.root_id] == "eml"
    assert tree.statistics.node_count == 3
    assert tree.statistics.edge_count == 2
    assert tree.statistics.depth == 1
    assert tree.statistics.leaf_count == 2
    assert tree.normal_leaf_count == 1
    assert tree.derived_leaf_count == 1
    assert tree.hidden_compound_leaf_count == 1
    assert tree.alpha is None
    assert tree.alpha_valid is False

    derived_nodes = [node for node in tree.nodes if node.kind == "derived"]
    assert len(derived_nodes) == 1
    assert derived_nodes[0].metadata["contains_hidden_compound"] is True
    assert derived_nodes[0].metadata["source_operator"] == "add"

    assert_internal_nodes_are_eml(expr, representation_mode="restricted_eml_with_derived")
    assert_connected_tree_shape(expr, representation_mode="restricted_eml_with_derived")
    assert_eml_equivalent(expr, representation_mode="restricted_eml_with_derived")


def test_hidden_compound_derived_leaf_is_not_counted_as_normal_leaf() -> None:
    x, y = sp.symbols("x y")
    expr = sp.Mul(
        sp.Add(x, 1, evaluate=False),
        sp.Add(y, 1, evaluate=False),
        evaluate=False,
    )

    tree = sympy_to_eml_tree(expr, representation_mode="restricted_eml_with_derived")

    assert tree.statistics.leaf_count == 2
    assert tree.normal_leaf_count == 1
    assert tree.derived_leaf_count == 1
    assert tree.hidden_compound_leaf_count == 1
    assert tree.alpha is None
    assert tree.alpha_valid is False
    assert_eml_equivalent(expr, representation_mode="restricted_eml_with_derived")


def test_alpha_matches_node_count_ratio_for_valid_pure_representation() -> None:
    x = sp.Symbol("x")
    expr = sp.exp(x, evaluate=False)

    eml_tree = sympy_to_eml_tree(expr)
    ast_tree = sympy_to_ast_tree(expr)

    assert eml_tree.alpha == eml_tree.statistics.node_count / ast_tree.statistics.node_count
    assert eml_tree.alpha_valid is True
    assert eml_alpha(expr) == eml_tree.alpha


def test_eml_alpha_rejects_derived_mode() -> None:
    x = sp.Symbol("x")
    expr = sp.Add(x, 1, evaluate=False)

    with pytest.raises(ValueError, match="alpha is valid only"):
        eml_alpha(expr, representation_mode="restricted_eml_with_derived")


def test_unsupported_expression_raises_clear_error() -> None:
    x = sp.Symbol("x")

    with pytest.raises(UnsupportedExpressionError, match="unsupported SymPy expression node"):
        sympy_to_eml_tree(sp.sin(x))


def test_pow_is_supported_by_official_pure_compiler() -> None:
    x = sp.Symbol("x")

    tree = sympy_to_eml_tree(sp.Pow(x, 2, evaluate=False))

    assert tree.representation_mode == "restricted_eml_pure"
    assert tree.alpha_valid is True
    assert tree.derived_leaf_count == 0


def test_integer_constants_compile_without_non_one_leaves() -> None:
    tree = sympy_to_eml_tree(2)

    assert tree.representation_mode == "restricted_eml_pure"
    assert tree.alpha_valid is True
    assert {node.label for node in tree.nodes if node.kind == "constant"} == {"1"}
