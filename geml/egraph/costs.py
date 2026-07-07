"""Cost models for Goal 4 e-graph extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import geml.symbolic.ast_graph as ast_graph
import geml.symbolic.dag_graph as dag_graph
import geml.symbolic.official_eml_compiler as official_eml_compiler
from geml.egraph.ir import (
    Add,
    Const,
    Div,
    Exp,
    Expr,
    Log,
    Mul,
    Neg,
    Pow,
    Sub,
    Var,
    display,
    to_sympy,
)
from geml.symbolic.dag_graph import DagGraph
from geml.symbolic.eml_nodes import EmlTree

type ExtractorMode = Literal["ast_node_cost", "estimated_eml_cost", "exact_eml_dag_beam_cost"]


@dataclass(frozen=True, slots=True)
class ExactEmlDagCost:
    """Exact candidate metrics after official pure EML compilation."""

    expression: Expr
    expression_string: str
    ast_tree_nodes: int
    ast_dag_nodes: int
    eml_tree_nodes: int
    eml_dag_nodes: int
    eml_tree: EmlTree
    eml_dag: DagGraph
    integrity_valid: bool
    integrity_errors: tuple[str, ...]

    @property
    def tie_break_key(self) -> tuple[int, int, int, int, str]:
        """Required Goal 4.4 tie-break order."""
        return (
            self.eml_dag_nodes,
            self.eml_tree_nodes,
            self.ast_dag_nodes,
            self.ast_tree_nodes,
            self.expression_string,
        )


def ast_node_cost(expr: Expr) -> int:
    """Count source IR tree nodes.

    This is a baseline source-AST objective. It must not be reported as
    EML-optimal.
    """
    if isinstance(expr, Var | Const):
        return 1
    if isinstance(expr, Neg | Exp | Log):
        return 1 + ast_node_cost(expr.value)
    if isinstance(expr, Add | Mul | Sub | Div):
        return 1 + ast_node_cost(expr.left) + ast_node_cost(expr.right)
    if isinstance(expr, Pow):
        return 1 + ast_node_cost(expr.base) + ast_node_cost(expr.exponent)
    raise TypeError(f"unsupported expression type: {type(expr).__name__}")


def estimated_eml_cost(expr: Expr) -> int:
    """Return a local approximate EML cost.

    The estimate is only a beam-ordering and baseline objective. It does not
    compile candidates, does not compute structural EML-DAG sharing, and is not
    suitable for final Goal 4 headline numbers.
    """
    if isinstance(expr, Var | Const):
        return 1
    if isinstance(expr, Exp):
        return 3 + estimated_eml_cost(expr.value)
    if isinstance(expr, Log):
        return 6 + estimated_eml_cost(expr.value)
    if isinstance(expr, Neg):
        return 12 + estimated_eml_cost(expr.value)
    if isinstance(expr, Add):
        return 26 + estimated_eml_cost(expr.left) + estimated_eml_cost(expr.right)
    if isinstance(expr, Sub):
        return 10 + estimated_eml_cost(expr.left) + estimated_eml_cost(expr.right)
    if isinstance(expr, Mul):
        return 34 + estimated_eml_cost(expr.left) + estimated_eml_cost(expr.right)
    if isinstance(expr, Div):
        return 48 + estimated_eml_cost(expr.left) + estimated_eml_cost(expr.right)
    if isinstance(expr, Pow):
        return 60 + estimated_eml_cost(expr.base) + estimated_eml_cost(expr.exponent)
    raise TypeError(f"unsupported expression type: {type(expr).__name__}")


def exact_eml_dag_cost(expr: Expr) -> ExactEmlDagCost:
    """Compile a candidate and compute exact official pure EML-DAG size."""
    sympy_expr = to_sympy(expr)
    ast_tree = ast_graph.sympy_to_ast_tree(sympy_expr)
    ast_dag = dag_graph.tree_to_dag(ast_tree)
    eml_tree = official_eml_compiler.sympy_to_official_eml_tree(sympy_expr)
    eml_dag = dag_graph.tree_to_dag(eml_tree)
    integrity_errors = validate_pure_eml_integrity(eml_tree, eml_dag)
    return ExactEmlDagCost(
        expression=expr,
        expression_string=display(expr),
        ast_tree_nodes=ast_tree.statistics.node_count,
        ast_dag_nodes=ast_dag.statistics.unique_node_count,
        eml_tree_nodes=eml_tree.statistics.node_count,
        eml_dag_nodes=eml_dag.statistics.unique_node_count,
        eml_tree=eml_tree,
        eml_dag=eml_dag,
        integrity_valid=not integrity_errors,
        integrity_errors=integrity_errors,
    )


def validate_pure_eml_integrity(eml_tree: EmlTree, eml_dag: DagGraph) -> tuple[str, ...]:
    """Validate the GEML pure-EML integrity contract for an extracted candidate."""
    errors: list[str] = []
    if eml_tree.representation_mode != "restricted_eml_pure":
        errors.append(f"unexpected EML mode {eml_tree.representation_mode!r}")
    if eml_tree.derived_leaf_count != 0:
        errors.append("derived leaves are present")
    if eml_tree.hidden_compound_leaf_count != 0:
        errors.append("hidden compound leaves are present")
    if eml_tree.alpha_valid is not True:
        errors.append("pure EML alpha is not marked valid")

    child_counts: dict[int, int] = {node.id: 0 for node in eml_tree.nodes}
    for edge in eml_tree.edges:
        child_counts[edge.source] = child_counts.get(edge.source, 0) + 1

    for node in eml_tree.nodes:
        if node.kind in dag_graph.FORBIDDEN_DAG_KINDS:
            errors.append(f"forbidden EML tree node kind {node.kind!r}")
        if node.label in dag_graph.FORBIDDEN_DAG_LABELS or node.label.startswith("eml_"):
            errors.append(f"forbidden EML tree node label {node.label!r}")
        child_count = child_counts.get(node.id, 0)
        if node.kind == "eml":
            if node.label != "eml":
                errors.append(f"invalid EML internal label {node.label!r}")
            if child_count != 2:
                errors.append(f"eml tree node {node.id} has {child_count} children")
        elif child_count:
            errors.append(f"non-eml internal tree node {node.id}")
        elif node.kind == "constant" and node.label != "1":
            errors.append(f"non-one constant EML leaf {node.label!r}")
        elif node.kind not in {"variable", "constant"}:
            errors.append(f"invalid EML tree leaf kind {node.kind!r}")

    dag_child_counts: dict[int, int] = {node.id: 0 for node in eml_dag.nodes}
    for ref in eml_dag.child_refs:
        dag_child_counts[ref.parent_id] = dag_child_counts.get(ref.parent_id, 0) + 1

    for node in eml_dag.nodes:
        if node.kind in dag_graph.FORBIDDEN_DAG_KINDS:
            errors.append(f"forbidden EML DAG node kind {node.kind!r}")
        if node.label in dag_graph.FORBIDDEN_DAG_LABELS or node.label.startswith("eml_"):
            errors.append(f"forbidden EML DAG node label {node.label!r}")
        child_count = dag_child_counts.get(node.id, 0)
        if node.kind == "eml" and child_count != 2:
            errors.append(f"eml DAG node {node.id} has {child_count} children")
        if node.kind != "eml" and child_count:
            errors.append(f"non-eml internal DAG node {node.id}")
        if node.kind == "constant" and node.label != "1":
            errors.append(f"non-one constant EML DAG leaf {node.label!r}")
        if node.kind not in {"eml", "variable", "constant"}:
            errors.append(f"invalid EML DAG node kind {node.kind!r}")

    return tuple(dict.fromkeys(errors))
