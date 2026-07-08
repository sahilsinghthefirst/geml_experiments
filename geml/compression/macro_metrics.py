"""Metrics for official macro graph compression baselines."""

from __future__ import annotations

import sympy as sp
from pydantic import BaseModel

from geml.compression.macro_expansions import validate_expansion_against_official
from geml.compression.macro_graph import MacroGraph, build_macro_graph
from geml.symbolic.ast_graph import AstTree, sympy_to_ast_tree
from geml.symbolic.dag_graph import DagGraph, tree_to_dag
from geml.symbolic.eml_nodes import EmlTree
from geml.symbolic.official_eml_compiler import sympy_to_official_eml_tree


class MacroGraphMetrics(BaseModel):
    """Per-expression macro graph metrics, separate from pure EML-DAG metrics."""

    expression: str
    srepr: str
    representation_mode: str
    source_ast_tree_nodes: int
    source_ast_dag_nodes: int
    goal3_eml_tree_nodes: int
    goal3_eml_dag_nodes: int
    macro_graph_nodes: int
    macro_graph_edges_or_child_refs: int
    macro_graph_depth: int
    macro_graph_alpha_vs_ast_tree: float
    macro_graph_alpha_vs_ast_dag: float
    compression_gain_vs_goal3_eml_dag: float
    expansion_valid: bool
    pure_eml_equivalent: bool
    validation_error: str | None = None


class MacroGraphAnalysis(BaseModel):
    """Full macro graph analysis bundle for one expression."""

    expression: str
    srepr: str
    ast_tree: AstTree
    ast_dag: DagGraph
    official_eml_tree: EmlTree
    official_eml_dag: DagGraph
    macro_graph: MacroGraph
    metrics: MacroGraphMetrics


def compute_macro_graph_analysis(expr: sp.Expr | str | int | float) -> MacroGraphAnalysis:
    """Compute AST, official pure EML, and macro graph metrics for one expression."""
    sympy_expr = sp.sympify(expr)
    ast_tree = sympy_to_ast_tree(sympy_expr)
    ast_dag = tree_to_dag(ast_tree)
    official_eml_tree = sympy_to_official_eml_tree(sympy_expr)
    official_eml_dag = tree_to_dag(official_eml_tree)
    macro_graph = build_macro_graph(sympy_expr)
    metrics = build_macro_graph_metrics(
        sympy_expr,
        macro_graph=macro_graph,
        source_ast_tree_nodes=ast_tree.statistics.node_count,
        source_ast_dag_nodes=ast_dag.statistics.unique_node_count,
        goal3_eml_tree_nodes=official_eml_tree.statistics.node_count,
        goal3_eml_dag_nodes=official_eml_dag.statistics.unique_node_count,
    )
    return MacroGraphAnalysis(
        expression=str(sympy_expr),
        srepr=sp.srepr(sympy_expr),
        ast_tree=ast_tree,
        ast_dag=ast_dag,
        official_eml_tree=official_eml_tree,
        official_eml_dag=official_eml_dag,
        macro_graph=macro_graph,
        metrics=metrics,
    )


def compute_macro_graph_metrics(expr: sp.Expr | str | int | float) -> MacroGraphMetrics:
    """Compute only per-expression macro graph metrics."""
    return compute_macro_graph_analysis(expr).metrics


def build_macro_graph_metrics(
    expr: sp.Expr,
    *,
    macro_graph: MacroGraph,
    source_ast_tree_nodes: int,
    source_ast_dag_nodes: int,
    goal3_eml_tree_nodes: int,
    goal3_eml_dag_nodes: int,
) -> MacroGraphMetrics:
    """Build metric row from a macro graph and caller-supplied baselines."""
    validation = validate_expansion_against_official(macro_graph, expr)
    macro_node_count = macro_graph.statistics.node_count
    return MacroGraphMetrics(
        expression=str(expr),
        srepr=sp.srepr(expr),
        representation_mode=macro_graph.representation_mode,
        source_ast_tree_nodes=source_ast_tree_nodes,
        source_ast_dag_nodes=source_ast_dag_nodes,
        goal3_eml_tree_nodes=goal3_eml_tree_nodes,
        goal3_eml_dag_nodes=goal3_eml_dag_nodes,
        macro_graph_nodes=macro_node_count,
        macro_graph_edges_or_child_refs=macro_graph.statistics.child_reference_count,
        macro_graph_depth=macro_graph.statistics.depth,
        macro_graph_alpha_vs_ast_tree=macro_node_count / source_ast_tree_nodes,
        macro_graph_alpha_vs_ast_dag=macro_node_count / source_ast_dag_nodes,
        compression_gain_vs_goal3_eml_dag=goal3_eml_dag_nodes / macro_node_count,
        expansion_valid=validation.expansion_valid,
        pure_eml_equivalent=validation.pure_eml_equivalent,
        validation_error=validation.error,
    )
