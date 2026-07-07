"""Tests for single-expression DAG metrics."""

from __future__ import annotations

import sympy as sp
from geml.symbolic.dag_metrics import (
    build_simple_expression_dag_audit,
    compute_expression_dag_analysis,
    compute_expression_dag_metrics,
)


def test_exp_x_has_no_dag_compression() -> None:
    x = sp.Symbol("x")

    analysis = compute_expression_dag_analysis(sp.exp(x, evaluate=False))
    metrics = analysis.metrics

    assert metrics.ast_tree_node_count == 2
    assert metrics.ast_dag_node_count == 2
    assert metrics.ast_dag_compression == 1.0
    assert metrics.eml_tree_node_count == 3
    assert metrics.eml_dag_node_count == 3
    assert metrics.eml_dag_compression == 1.0
    assert metrics.tree_alpha == 1.5
    assert metrics.tree_alpha == analysis.eml_tree.alpha


def test_repeated_subexpressions_show_ast_and_eml_dag_compression() -> None:
    x = sp.Symbol("x")
    expr = sp.Mul(
        sp.Add(x, 1, evaluate=False),
        sp.Add(x, 1, evaluate=False),
        evaluate=False,
    )

    metrics = compute_expression_dag_metrics(expr)

    assert metrics.ast_tree_node_count == 7
    assert metrics.ast_dag_node_count == 4
    assert metrics.ast_dag_compression > 1.0
    assert metrics.eml_dag_node_count < metrics.eml_tree_node_count
    assert metrics.eml_dag_compression > 1.0
    assert metrics.dag_alpha_vs_ast_tree < metrics.tree_alpha


def test_dag_node_counts_are_never_greater_than_tree_node_counts() -> None:
    x, y = sp.symbols("x y")
    expressions = [
        sp.Add(x, y, evaluate=False),
        sp.Mul(x, y, evaluate=False),
        sp.log(x, evaluate=False),
        sp.exp(x, evaluate=False),
        sp.Pow(x, 2, evaluate=False),
        sp.Mul(
            sp.Mul(x, x, evaluate=False),
            sp.Mul(x, x, evaluate=False),
            evaluate=False,
        ),
    ]

    for expr in expressions:
        metrics = compute_expression_dag_metrics(expr)
        assert metrics.ast_dag_node_count <= metrics.ast_tree_node_count
        assert metrics.eml_dag_node_count <= metrics.eml_tree_node_count


def test_pure_eml_dag_contains_only_eml_internal_nodes_and_variable_or_one_leaves() -> None:
    x = sp.Symbol("x")
    expr = sp.Mul(
        sp.Add(x, 1, evaluate=False),
        sp.Add(x, 1, evaluate=False),
        evaluate=False,
    )

    analysis = compute_expression_dag_analysis(expr)

    internal_node_ids = {ref.parent_id for ref in analysis.eml_dag.child_refs}
    for node in analysis.eml_dag.nodes:
        if node.id in internal_node_ids:
            assert node.kind == "eml"
            assert node.label == "eml"
        else:
            assert node.kind in {"variable", "constant"}
            if node.kind == "constant":
                assert node.label == "1"
        assert node.kind != "derived"
        assert not node.label.startswith("eml_")


def test_simple_expression_dag_audit_contains_required_rows_and_fields() -> None:
    rows = build_simple_expression_dag_audit()
    rows_by_name = {row.name: row for row in rows}

    assert set(rows_by_name) == {
        "x+y",
        "x*y",
        "log(x)",
        "exp(x)",
        "x**2",
        "(x+1)*(x+1)",
        "(x*x)*(x*x)",
    }
    for row in rows:
        assert row.ast_tree_nodes >= row.ast_dag_nodes
        assert row.eml_tree_nodes >= row.eml_dag_nodes
        assert row.tree_alpha == row.eml_tree_nodes / row.ast_tree_nodes
        assert row.dag_alpha_vs_ast_tree == row.eml_dag_nodes / row.ast_tree_nodes
        assert row.dag_alpha_vs_ast_dag == row.eml_dag_nodes / row.ast_dag_nodes
        assert row.eml_dag_compression == row.eml_tree_nodes / row.eml_dag_nodes

    assert rows_by_name["(x+1)*(x+1)"].ast_tree_nodes > rows_by_name["(x+1)*(x+1)"].ast_dag_nodes
    assert rows_by_name["(x*x)*(x*x)"].eml_dag_compression > 1.0
