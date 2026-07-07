"""Single-expression AST-DAG and pure EML-DAG metrics."""

from __future__ import annotations

import sympy as sp
from pydantic import BaseModel

from geml.symbolic.ast_graph import AstTree, sympy_to_ast_tree
from geml.symbolic.dag_graph import DagGraph, tree_to_dag
from geml.symbolic.eml_nodes import EmlTree
from geml.symbolic.eml_transpile import sympy_to_eml_tree


class ExpressionDagMetrics(BaseModel):
    """Tree and exact structural DAG metrics for one expression."""

    expression: str
    srepr: str
    ast_tree_node_count: int
    ast_tree_edge_count: int
    ast_tree_depth: int
    ast_dag_node_count: int
    ast_dag_child_ref_count: int
    ast_dag_depth: int
    ast_dag_compression: float
    eml_tree_node_count: int
    eml_tree_edge_count: int
    eml_tree_depth: int
    eml_dag_node_count: int
    eml_dag_child_ref_count: int
    eml_dag_depth: int
    eml_dag_compression: float
    tree_alpha: float
    dag_alpha_vs_ast_tree: float
    dag_alpha_vs_ast_dag: float


class ExpressionDagAnalysis(BaseModel):
    """Full tree/DAG analysis bundle for one expression."""

    expression: str
    srepr: str
    ast_tree: AstTree
    ast_dag: DagGraph
    eml_tree: EmlTree
    eml_dag: DagGraph
    metrics: ExpressionDagMetrics


class SimpleExpressionDagAuditRow(BaseModel):
    """Compact Goal 3.2 simple-expression audit row."""

    name: str
    expression: str
    srepr: str
    ast_tree_nodes: int
    ast_dag_nodes: int
    eml_tree_nodes: int
    eml_dag_nodes: int
    tree_alpha: float
    dag_alpha_vs_ast_tree: float
    dag_alpha_vs_ast_dag: float
    eml_dag_compression: float


def compute_expression_dag_analysis(expr: sp.Expr | str | int | float) -> ExpressionDagAnalysis:
    """Compute AST/EML tree and exact structural DAG metrics for one expression."""
    sympy_expr = sp.sympify(expr)
    ast_tree = sympy_to_ast_tree(sympy_expr)
    ast_dag = tree_to_dag(ast_tree)
    eml_tree = sympy_to_eml_tree(sympy_expr, representation_mode="restricted_eml_pure")
    eml_dag = tree_to_dag(eml_tree)
    metrics = _build_metrics(
        sympy_expr=sympy_expr,
        ast_tree=ast_tree,
        ast_dag=ast_dag,
        eml_tree=eml_tree,
        eml_dag=eml_dag,
    )

    return ExpressionDagAnalysis(
        expression=str(sympy_expr),
        srepr=sp.srepr(sympy_expr),
        ast_tree=ast_tree,
        ast_dag=ast_dag,
        eml_tree=eml_tree,
        eml_dag=eml_dag,
        metrics=metrics,
    )


def compute_expression_dag_metrics(expr: sp.Expr | str | int | float) -> ExpressionDagMetrics:
    """Compute only the numeric tree/DAG metric row for one expression."""
    return compute_expression_dag_analysis(expr).metrics


def build_simple_expression_dag_audit() -> list[SimpleExpressionDagAuditRow]:
    """Build the Goal 3.2 simple-expression DAG audit."""
    x, y = sp.symbols("x y")
    repeated_sum = sp.Add(x, 1, evaluate=False)
    repeated_product = sp.Mul(x, x, evaluate=False)
    examples = [
        ("x+y", sp.Add(x, y, evaluate=False)),
        ("x*y", sp.Mul(x, y, evaluate=False)),
        ("log(x)", sp.log(x, evaluate=False)),
        ("exp(x)", sp.exp(x, evaluate=False)),
        ("x**2", sp.Pow(x, 2, evaluate=False)),
        (
            "(x+1)*(x+1)",
            sp.Mul(repeated_sum, sp.Add(x, 1, evaluate=False), evaluate=False),
        ),
        (
            "(x*x)*(x*x)",
            sp.Mul(repeated_product, sp.Mul(x, x, evaluate=False), evaluate=False),
        ),
    ]

    rows: list[SimpleExpressionDagAuditRow] = []
    for name, expr in examples:
        metrics = compute_expression_dag_metrics(expr)
        rows.append(
            SimpleExpressionDagAuditRow(
                name=name,
                expression=metrics.expression,
                srepr=metrics.srepr,
                ast_tree_nodes=metrics.ast_tree_node_count,
                ast_dag_nodes=metrics.ast_dag_node_count,
                eml_tree_nodes=metrics.eml_tree_node_count,
                eml_dag_nodes=metrics.eml_dag_node_count,
                tree_alpha=metrics.tree_alpha,
                dag_alpha_vs_ast_tree=metrics.dag_alpha_vs_ast_tree,
                dag_alpha_vs_ast_dag=metrics.dag_alpha_vs_ast_dag,
                eml_dag_compression=metrics.eml_dag_compression,
            )
        )
    return rows


def _build_metrics(
    *,
    sympy_expr: sp.Expr,
    ast_tree: AstTree,
    ast_dag: DagGraph,
    eml_tree: EmlTree,
    eml_dag: DagGraph,
) -> ExpressionDagMetrics:
    ast_tree_nodes = ast_tree.statistics.node_count
    ast_dag_nodes = ast_dag.statistics.unique_node_count
    eml_tree_nodes = eml_tree.statistics.node_count
    eml_dag_nodes = eml_dag.statistics.unique_node_count

    return ExpressionDagMetrics(
        expression=str(sympy_expr),
        srepr=sp.srepr(sympy_expr),
        ast_tree_node_count=ast_tree_nodes,
        ast_tree_edge_count=ast_tree.statistics.edge_count,
        ast_tree_depth=ast_tree.statistics.depth,
        ast_dag_node_count=ast_dag_nodes,
        ast_dag_child_ref_count=ast_dag.statistics.child_reference_count,
        ast_dag_depth=ast_dag.statistics.depth,
        ast_dag_compression=ast_tree_nodes / ast_dag_nodes,
        eml_tree_node_count=eml_tree_nodes,
        eml_tree_edge_count=eml_tree.statistics.edge_count,
        eml_tree_depth=eml_tree.statistics.depth,
        eml_dag_node_count=eml_dag_nodes,
        eml_dag_child_ref_count=eml_dag.statistics.child_reference_count,
        eml_dag_depth=eml_dag.statistics.depth,
        eml_dag_compression=eml_tree_nodes / eml_dag_nodes,
        tree_alpha=eml_tree_nodes / ast_tree_nodes,
        dag_alpha_vs_ast_tree=eml_dag_nodes / ast_tree_nodes,
        dag_alpha_vs_ast_dag=eml_dag_nodes / ast_dag_nodes,
    )
