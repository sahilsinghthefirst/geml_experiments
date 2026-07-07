"""Tests for exact structural DAG conversion."""

from __future__ import annotations

import pytest
import sympy as sp
from geml.symbolic.ast_graph import AstTree, sympy_to_ast_tree
from geml.symbolic.dag_graph import (
    DagChildRef,
    DagGraph,
    DagNode,
    DagStatistics,
    canonical_structural_signature,
    tree_to_dag,
    validate_dag_graph,
)
from geml.symbolic.eml_nodes import EmlEdge, EmlNode, EmlTree
from geml.symbolic.eml_transpile import sympy_to_eml_tree
from geml.symbolic.metrics import TreeStatistics, compute_tree_statistics


def child_refs_for(dag: DagGraph, parent_id: int) -> list[DagChildRef]:
    """Return child references for a DAG parent in slot order."""
    return sorted(
        [ref for ref in dag.child_refs if ref.parent_id == parent_id],
        key=lambda ref: ref.slot_index,
    )


def ast_child_labels(tree: AstTree, node_id: int) -> list[str]:
    """Return AST child labels in slot order."""
    return [
        tree.node_labels[edge.target]
        for edge in sorted(
            [edge for edge in tree.edges if edge.source == node_id],
            key=lambda edge: edge.position,
        )
    ]


def test_x_plus_x_ast_tree_compresses_repeated_x_leaf() -> None:
    x = sp.Symbol("x")
    tree = sympy_to_ast_tree(sp.Add(x, x, evaluate=False))

    dag = tree_to_dag(tree)
    root_refs = child_refs_for(dag, dag.root_id)

    assert dag.metadata["dag_mode"] == "ast_dag"
    assert dag.statistics.unique_node_count == 2
    assert dag.statistics.child_reference_count == 2
    assert [ref.child_slot for ref in root_refs] == ["left", "right"]
    assert root_refs[0].child_id == root_refs[1].child_id
    assert dag.node_labels[root_refs[0].child_id] == "x"


def test_repeated_ast_subtree_is_shared_when_structurally_identical() -> None:
    x = sp.Symbol("x")
    left = sp.Add(x, 1, evaluate=False)
    right = sp.Add(x, 1, evaluate=False)
    tree = sympy_to_ast_tree(sp.Mul(left, right, evaluate=False))

    dag = tree_to_dag(tree)
    root_refs = child_refs_for(dag, dag.root_id)
    shared_add_id = root_refs[0].child_id

    assert dag.statistics.unique_node_count == 4
    assert dag.node_labels[dag.root_id] == "mul"
    assert root_refs[0].child_id == root_refs[1].child_id
    assert dag.node_labels[shared_add_id] == "add"
    assert [dag.node_labels[ref.child_id] for ref in child_refs_for(dag, shared_add_id)] == [
        "x",
        "1",
    ]


def test_repeated_eml_one_x_subtree_is_shared() -> None:
    tree = build_repeated_eml_one_x_tree()

    dag = tree_to_dag(tree)
    root_refs = child_refs_for(dag, dag.root_id)
    shared_eml_id = root_refs[0].child_id

    assert dag.metadata["dag_mode"] == "restricted_eml_pure_dag"
    assert dag.statistics.unique_node_count == 4
    assert root_refs[0].child_id == root_refs[1].child_id
    assert dag.node_labels[shared_eml_id] == "eml"
    assert [dag.node_labels[ref.child_id] for ref in child_refs_for(dag, shared_eml_id)] == [
        "1",
        "x",
    ]


def test_eml_a_a_preserves_two_child_refs_to_same_child() -> None:
    tree = build_eml_x_x_tree()

    dag = tree_to_dag(tree)
    root_refs = child_refs_for(dag, dag.root_id)

    assert dag.statistics.unique_node_count == 2
    assert dag.statistics.child_reference_count == 2
    assert [ref.child_slot for ref in root_refs] == ["left", "right"]
    assert [ref.slot_index for ref in root_refs] == [0, 1]
    assert root_refs[0].child_id == root_refs[1].child_id
    assert dag.node_labels[root_refs[0].child_id] == "x"


def test_x_plus_y_and_y_plus_x_are_not_forcibly_merged() -> None:
    x, y = sp.symbols("x y")
    xy_tree = sympy_to_ast_tree(sp.Add(x, y, evaluate=False))
    yx_tree = sympy_to_ast_tree(sp.Add(y, x, evaluate=False))

    xy_children = ast_child_labels(xy_tree, xy_tree.root_id)
    yx_children = ast_child_labels(yx_tree, yx_tree.root_id)
    if xy_children != yx_children:
        assert canonical_structural_signature(xy_tree) != canonical_structural_signature(yx_tree)


def test_dag_validation_rejects_cycles() -> None:
    dag = DagGraph(
        nodes=[
            DagNode(id=0, label="root", kind="operator"),
            DagNode(id=1, label="x", kind="symbol"),
        ],
        child_refs=[
            DagChildRef(parent_id=0, child_id=1, child_slot="arg0", slot_index=0),
            DagChildRef(parent_id=1, child_id=0, child_slot="arg0", slot_index=0),
        ],
        root_id=0,
        node_labels={0: "root", 1: "x"},
        node_kinds={0: "operator", 1: "symbol"},
        metadata={"dag_mode": "ast_dag"},
        statistics=DagStatistics(
            unique_node_count=2,
            child_reference_count=2,
            depth=0,
            leaf_count=0,
            shared_node_count=0,
        ),
    )

    with pytest.raises(ValueError, match="cycle detected"):
        validate_dag_graph(dag)


def test_pure_eml_dag_has_no_derived_or_macro_nodes() -> None:
    x = sp.Symbol("x")
    tree = sympy_to_eml_tree(
        sp.Add(x, 1, evaluate=False), representation_mode="restricted_eml_pure"
    )

    dag = tree_to_dag(tree)

    assert all(node.kind != "derived" for node in dag.nodes)
    assert all(not node.label.startswith("eml_") for node in dag.nodes)
    assert {node.kind for node in dag.nodes} <= {"eml", "variable", "constant"}
    assert {node.label for node in dag.nodes if node.kind == "constant"} == {"1"}


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
    return build_eml_tree(nodes=nodes, edges=edges)


def build_eml_x_x_tree() -> EmlTree:
    """Build the pure EML tree EML(x, x)."""
    nodes = [
        EmlNode(id=0, label="eml", kind="eml"),
        EmlNode(id=1, label="x", kind="variable"),
        EmlNode(id=2, label="x", kind="variable"),
    ]
    edges = [
        EmlEdge(source=0, target=1, position=0),
        EmlEdge(source=0, target=2, position=1),
    ]
    return build_eml_tree(nodes=nodes, edges=edges)


def build_eml_tree(*, nodes: list[EmlNode], edges: list[EmlEdge]) -> EmlTree:
    """Build a valid pure EML test tree from nodes and edges."""
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
