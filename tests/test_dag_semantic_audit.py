"""Tests for Goal 3.6 DAG semantic audit."""

from __future__ import annotations

import math

import pytest
import sympy as sp
from geml.experiments.dag_semantic_audit import (
    assert_eml_dag_structurally_pure,
    audit_eml_dag_structure,
    evaluate_eml_dag,
    top_shared_subtree_signatures,
)
from geml.symbolic.dag_graph import DagChildRef, DagGraph, DagNode, DagStatistics, tree_to_dag
from geml.symbolic.dag_metrics import compute_expression_dag_analysis
from geml.symbolic.eml_nodes import EmlEdge, EmlNode, EmlTree
from geml.symbolic.metrics import TreeStatistics, compute_tree_statistics
from geml.symbolic.official_eml_compiler import evaluate_official_eml_tree


def test_eml_dag_evaluator_matches_eml_tree_evaluator() -> None:
    x = sp.Symbol("x")
    expr = sp.Mul(
        sp.Add(x, 1, evaluate=False),
        sp.Add(x, 1, evaluate=False),
        evaluate=False,
    )
    analysis = compute_expression_dag_analysis(expr)
    values = {"x": 1.3, "y": 2.1, "z": 3.2}

    tree_value = evaluate_official_eml_tree(analysis.eml_tree, values)
    dag_value = evaluate_eml_dag(analysis.eml_dag, values)

    assert math.isclose(tree_value, dag_value, rel_tol=1e-8, abs_tol=1e-8)
    assert_eml_dag_structurally_pure(analysis.eml_dag)


def test_duplicate_child_ref_case_evaluates_both_slots() -> None:
    dag = build_eml_x_x_dag()

    result = audit_eml_dag_structure(dag)
    value = evaluate_eml_dag(dag, {"x": 1.3})

    assert result.valid is True
    assert result.duplicate_child_ref_parent_count == 1
    assert math.isclose(value, math.exp(1.3) - math.log(1.3), rel_tol=1e-8, abs_tol=1e-8)
    assert len(dag.child_refs) == 2
    assert [ref.child_slot for ref in sorted(dag.child_refs, key=lambda ref: ref.slot_index)] == [
        "left",
        "right",
    ]


@pytest.mark.parametrize("bad_dag", ["derived", "macro"])
def test_structural_purity_checks_fail_if_derived_or_macro_nodes_appear(bad_dag: str) -> None:
    dag = build_bad_dag(bad_dag)

    with pytest.raises(ValueError):
        assert_eml_dag_structurally_pure(dag)

    result = audit_eml_dag_structure(dag)
    assert result.valid is False
    assert result.derived_leaf_count or result.macro_template_node_count


def test_shared_subtree_counts_are_computed_correctly() -> None:
    tree = build_repeated_eml_one_x_tree()
    dag = tree_to_dag(tree)

    shared = top_shared_subtree_signatures(dag, limit=5)

    assert shared[0].label == "eml"
    assert shared[0].kind == "eml"
    assert shared[0].reuse_count == 2
    assert shared[0].incoming_child_ref_count == 2
    assert any(record.label == "x" and record.reuse_count == 2 for record in shared)
    assert any(record.label == "1" and record.reuse_count == 2 for record in shared)


def build_eml_x_x_dag() -> DagGraph:
    """Build the pure EML DAG EML(x, x) with two explicit child refs."""
    nodes = [
        DagNode(id=0, label="eml", kind="eml"),
        DagNode(id=1, label="x", kind="variable", metadata={"source_tree_node_ids": [1, 2]}),
    ]
    child_refs = [
        DagChildRef(parent_id=0, child_id=1, child_slot="left", slot_index=0),
        DagChildRef(parent_id=0, child_id=1, child_slot="right", slot_index=1),
    ]
    return DagGraph(
        nodes=nodes,
        child_refs=child_refs,
        root_id=0,
        node_labels={node.id: node.label for node in nodes},
        node_kinds={node.id: node.kind for node in nodes},
        metadata={
            "dag_mode": "restricted_eml_pure_dag",
            "source_representation_mode": "restricted_eml_pure",
        },
        statistics=DagStatistics(
            unique_node_count=2,
            child_reference_count=2,
            depth=1,
            leaf_count=1,
            shared_node_count=1,
        ),
    )


def build_bad_dag(kind: str) -> DagGraph:
    """Build a structurally connected DAG with one forbidden node."""
    if kind == "derived":
        bad_node = DagNode(
            id=1,
            label="log(expr)",
            kind="derived",
            metadata={"contains_hidden_compound": True, "source_tree_node_ids": [1]},
        )
    elif kind == "macro":
        bad_node = DagNode(id=1, label="eml_add", kind="macro")
    else:
        raise ValueError(kind)

    nodes = [
        DagNode(id=0, label="eml", kind="eml"),
        bad_node,
        DagNode(id=2, label="1", kind="constant", metadata={"value": 1}),
    ]
    child_refs = [
        DagChildRef(parent_id=0, child_id=1, child_slot="left", slot_index=0),
        DagChildRef(parent_id=0, child_id=2, child_slot="right", slot_index=1),
    ]
    return DagGraph(
        nodes=nodes,
        child_refs=child_refs,
        root_id=0,
        node_labels={node.id: node.label for node in nodes},
        node_kinds={node.id: node.kind for node in nodes},
        metadata={
            "dag_mode": "restricted_eml_pure_dag",
            "source_representation_mode": "restricted_eml_pure",
        },
        statistics=DagStatistics(
            unique_node_count=3,
            child_reference_count=2,
            depth=1,
            leaf_count=2,
            shared_node_count=0,
        ),
    )


def build_repeated_eml_one_x_tree() -> EmlTree:
    """Build the pure EML tree EML(EML(1, x), EML(1, x))."""
    nodes = [
        EmlNode(id=0, label="eml", kind="eml"),
        EmlNode(id=1, label="eml", kind="eml"),
        EmlNode(id=2, label="1", kind="constant", metadata={"value": 1}),
        EmlNode(id=3, label="x", kind="variable"),
        EmlNode(id=4, label="eml", kind="eml"),
        EmlNode(id=5, label="1", kind="constant", metadata={"value": 1}),
        EmlNode(id=6, label="x", kind="variable"),
    ]
    edges = [
        EmlEdge(source=0, target=1, position=0),
        EmlEdge(source=0, target=4, position=1),
        EmlEdge(source=1, target=2, position=0),
        EmlEdge(source=1, target=3, position=1),
        EmlEdge(source=4, target=5, position=0),
        EmlEdge(source=4, target=6, position=1),
    ]
    statistics = compute_tree_statistics(
        root_id=0,
        node_ids=[node.id for node in nodes],
        edges=[(edge.source, edge.target) for edge in edges],
        operator_node_ids=[node.id for node in nodes if node.kind == "eml"],
    )
    return EmlTree(
        representation_mode="restricted_eml_pure",
        nodes=nodes,
        edges=edges,
        root_id=0,
        node_labels={node.id: node.label for node in nodes},
        metadata={"representation_mode": "restricted_eml_pure"},
        statistics=statistics,
        normal_leaf_count=statistics.leaf_count,
        derived_leaf_count=0,
        hidden_compound_leaf_count=0,
        ast_statistics=TreeStatistics(
            node_count=1,
            edge_count=0,
            depth=0,
            leaf_count=1,
            operator_count=0,
        ),
        alpha=None,
        alpha_valid=True,
    )
