"""Exact structural DAG conversion for symbolic tree representations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from geml.symbolic.ast_graph import AstTree
from geml.symbolic.eml_nodes import EmlTree

type MetadataValue = object
type StructuralSignature = tuple[object, ...]
type TreeRepresentation = AstTree | EmlTree

FORBIDDEN_DAG_KINDS = frozenset({"derived", "macro", "template"})
FORBIDDEN_DAG_LABELS = frozenset(
    {
        "derived",
        "eml_add",
        "eml_div",
        "eml_exp",
        "eml_int",
        "eml_inv",
        "eml_log",
        "eml_mul",
        "eml_neg",
        "eml_one",
        "eml_pow",
        "eml_rational",
        "eml_sub",
        "eml_zero",
        "macro",
        "template",
    }
)


class _TreeNode(Protocol):
    id: int
    label: str
    kind: str
    metadata: dict[str, object]


class DagNode(BaseModel):
    """One unique structural subtree in a DAG."""

    id: int = Field(ge=0)
    label: str
    kind: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class DagChildRef(BaseModel):
    """An ordered parent-to-child reference in a DAG."""

    parent_id: int = Field(ge=0)
    child_id: int = Field(ge=0)
    child_slot: str
    slot_index: int = Field(ge=0)


DagEdge = DagChildRef


class DagStatistics(BaseModel):
    """Structural DAG statistics."""

    unique_node_count: int = Field(ge=0)
    child_reference_count: int = Field(ge=0)
    depth: int = Field(ge=0)
    leaf_count: int = Field(ge=0)
    shared_node_count: int = Field(ge=0)


class DagGraph(BaseModel):
    """Serializable exact structural DAG representation."""

    nodes: list[DagNode]
    child_refs: list[DagChildRef]
    root_id: int
    node_labels: dict[int, str]
    node_kinds: dict[int, str]
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    statistics: DagStatistics


@dataclass(frozen=True)
class _TreeChild:
    position: int
    node_id: int


@dataclass(frozen=True)
class _TreeIndex:
    source_mode: str
    nodes_by_id: dict[int, _TreeNode]
    children_by_parent: dict[int, list[_TreeChild]]


def tree_to_dag(tree: TreeRepresentation) -> DagGraph:
    """Compress an AST or pure EML tree into an exact structural DAG."""
    index = _build_tree_index(tree)
    signature_to_dag_id: dict[StructuralSignature, int] = {}
    dag_nodes: list[DagNode] = []
    child_refs: list[DagChildRef] = []

    def convert_node(tree_node_id: int) -> tuple[int, StructuralSignature]:
        node = index.nodes_by_id[tree_node_id]
        children = _ordered_children(index, tree_node_id)
        child_results = [convert_node(child.node_id) for child in children]
        child_signatures = tuple(
            (child.position, child_signature)
            for child, (_, child_signature) in zip(children, child_results, strict=True)
        )
        signature = _node_signature(
            source_mode=index.source_mode,
            node=node,
            child_signatures=child_signatures,
        )

        existing_dag_id = signature_to_dag_id.get(signature)
        if existing_dag_id is not None:
            source_ids = dag_nodes[existing_dag_id].metadata["source_tree_node_ids"]
            if not isinstance(source_ids, list):
                raise TypeError("source_tree_node_ids metadata must be a list")
            source_ids.append(tree_node_id)
            return existing_dag_id, signature

        dag_id = len(dag_nodes)
        signature_to_dag_id[signature] = dag_id
        dag_nodes.append(
            DagNode(
                id=dag_id,
                label=node.label,
                kind=node.kind,
                metadata={
                    **dict(node.metadata),
                    "source_representation_mode": index.source_mode,
                    "source_tree_node_ids": [tree_node_id],
                },
            )
        )

        child_count = len(children)
        for child, (child_dag_id, _) in zip(children, child_results, strict=True):
            child_refs.append(
                DagChildRef(
                    parent_id=dag_id,
                    child_id=child_dag_id,
                    child_slot=_slot_name(child.position, child_count),
                    slot_index=child.position,
                )
            )
        return dag_id, signature

    root_id, root_signature = convert_node(tree.root_id)
    statistics = compute_dag_statistics(
        root_id=root_id,
        node_ids=[node.id for node in dag_nodes],
        child_refs=child_refs,
        source_tree_node_ids_by_dag_id={node.id: _source_tree_node_ids(node) for node in dag_nodes},
    )
    dag = DagGraph(
        nodes=dag_nodes,
        child_refs=child_refs,
        root_id=root_id,
        node_labels={node.id: node.label for node in dag_nodes},
        node_kinds={node.id: node.kind for node in dag_nodes},
        metadata={
            "converter": "exact_structural_dag_v0",
            "source_representation_mode": index.source_mode,
            "dag_mode": _dag_mode(index.source_mode),
            "root_structural_signature": repr(root_signature),
            "sharing_policy": "exact canonical structural subtree signature equality",
        },
        statistics=statistics,
    )
    validate_dag_graph(dag)
    return dag


def canonical_structural_signature(
    tree: TreeRepresentation,
    node_id: int | None = None,
) -> StructuralSignature:
    """Return the canonical structural signature for a tree node."""
    index = _build_tree_index(tree)
    target_node_id = tree.root_id if node_id is None else node_id
    if target_node_id not in index.nodes_by_id:
        raise ValueError(f"node_id {target_node_id} is not present in tree nodes")

    def build_signature(current_node_id: int) -> StructuralSignature:
        node = index.nodes_by_id[current_node_id]
        children = _ordered_children(index, current_node_id)
        child_signatures = tuple(
            (child.position, build_signature(child.node_id)) for child in children
        )
        return _node_signature(
            source_mode=index.source_mode,
            node=node,
            child_signatures=child_signatures,
        )

    return build_signature(target_node_id)


def validate_dag_graph(dag: DagGraph) -> None:
    """Validate a rooted exact structural DAG."""
    compute_dag_statistics(
        root_id=dag.root_id,
        node_ids=[node.id for node in dag.nodes],
        child_refs=dag.child_refs,
        source_tree_node_ids_by_dag_id={node.id: _source_tree_node_ids(node) for node in dag.nodes},
    )


def compute_dag_statistics(
    *,
    root_id: int,
    node_ids: Sequence[int],
    child_refs: Sequence[DagChildRef],
    source_tree_node_ids_by_dag_id: dict[int, list[int]] | None = None,
) -> DagStatistics:
    """Compute DAG statistics and validate rooted DAG invariants."""
    node_id_set = set(node_ids)
    if len(node_id_set) != len(node_ids):
        raise ValueError("DAG node ids must be unique")
    if root_id not in node_id_set:
        raise ValueError(f"root_id {root_id} is not present in DAG nodes")

    refs_by_parent: dict[int, list[DagChildRef]] = defaultdict(list)
    incoming_counts: dict[int, int] = {node_id: 0 for node_id in node_id_set}
    for ref in child_refs:
        if ref.parent_id not in node_id_set:
            raise ValueError(f"child ref parent {ref.parent_id} is not present in DAG nodes")
        if ref.child_id not in node_id_set:
            raise ValueError(f"child ref child {ref.child_id} is not present in DAG nodes")
        refs_by_parent[ref.parent_id].append(ref)
        incoming_counts[ref.child_id] += 1

    for parent_id, refs in refs_by_parent.items():
        _validate_child_slots(parent_id, refs)

    visited: set[int] = set()
    active: set[int] = set()

    def visit(node_id: int) -> None:
        if node_id in active:
            raise ValueError(f"cycle detected at DAG node {node_id}")
        if node_id in visited:
            return
        active.add(node_id)
        for ref in refs_by_parent.get(node_id, []):
            visit(ref.child_id)
        active.remove(node_id)
        visited.add(node_id)

    visit(root_id)
    if visited != node_id_set:
        unreachable = sorted(node_id_set - visited)
        raise ValueError(f"DAG contains unreachable nodes: {unreachable}")

    depth_cache: dict[int, int] = {}

    def depth_from(node_id: int) -> int:
        if node_id in depth_cache:
            return depth_cache[node_id]
        refs = refs_by_parent.get(node_id, [])
        if not refs:
            depth_cache[node_id] = 0
            return 0
        depth_cache[node_id] = 1 + max(depth_from(ref.child_id) for ref in refs)
        return depth_cache[node_id]

    source_ids = source_tree_node_ids_by_dag_id or {}
    shared_node_count = sum(
        1
        for node_id in node_id_set
        if incoming_counts[node_id] > 1 or len(source_ids.get(node_id, [])) > 1
    )
    leaf_count = sum(1 for node_id in node_id_set if not refs_by_parent.get(node_id))

    return DagStatistics(
        unique_node_count=len(node_id_set),
        child_reference_count=len(child_refs),
        depth=depth_from(root_id),
        leaf_count=leaf_count,
        shared_node_count=shared_node_count,
    )


def _build_tree_index(tree: TreeRepresentation) -> _TreeIndex:
    source_mode = _source_mode(tree)
    nodes_by_id = {node.id: node for node in tree.nodes}
    if len(nodes_by_id) != len(tree.nodes):
        raise ValueError("tree node ids must be unique")
    if tree.root_id not in nodes_by_id:
        raise ValueError(f"tree root_id {tree.root_id} is not present in tree nodes")

    children_by_parent: dict[int, list[_TreeChild]] = {node.id: [] for node in tree.nodes}
    for edge in tree.edges:
        if edge.source not in nodes_by_id:
            raise ValueError(f"tree edge source {edge.source} is not present in tree nodes")
        if edge.target not in nodes_by_id:
            raise ValueError(f"tree edge target {edge.target} is not present in tree nodes")
        children_by_parent[edge.source].append(
            _TreeChild(position=edge.position, node_id=edge.target)
        )

    for node in tree.nodes:
        _validate_node_allowed(source_mode=source_mode, kind=node.kind, label=node.label)

    return _TreeIndex(
        source_mode=source_mode,
        nodes_by_id=nodes_by_id,
        children_by_parent=children_by_parent,
    )


def _source_mode(tree: TreeRepresentation) -> str:
    representation_mode = getattr(tree, "representation_mode", None)
    if isinstance(tree, EmlTree):
        if representation_mode != "restricted_eml_pure":
            raise ValueError(
                "DAG compression supports only restricted_eml_pure EML trees; "
                f"got {representation_mode!r}"
            )
        return "restricted_eml_pure"
    if isinstance(tree, AstTree):
        return "ast"
    raise TypeError(f"unsupported tree type {type(tree).__name__}")


def _dag_mode(source_mode: str) -> str:
    if source_mode == "ast":
        return "ast_dag"
    if source_mode == "restricted_eml_pure":
        return "restricted_eml_pure_dag"
    raise ValueError(f"unsupported source representation mode {source_mode!r}")


def _ordered_children(index: _TreeIndex, node_id: int) -> list[_TreeChild]:
    children = sorted(index.children_by_parent[node_id], key=lambda child: child.position)
    if len(children) > 2:
        raise ValueError(
            f"n-ary tree node {node_id} has {len(children)} children; "
            "Goal 3 DAG compression expects binary-normalized trees"
        )
    expected_positions = list(range(len(children)))
    positions = [child.position for child in children]
    if positions != expected_positions:
        raise ValueError(
            f"tree node {node_id} has child positions {positions}; expected {expected_positions}"
        )
    return children


def _node_signature(
    *,
    source_mode: str,
    node: _TreeNode,
    child_signatures: tuple[tuple[int, StructuralSignature], ...],
) -> StructuralSignature:
    if not child_signatures:
        return (
            source_mode,
            "leaf",
            node.kind,
            node.label,
            _leaf_structural_value(node),
        )
    if len(child_signatures) == 1:
        return (
            source_mode,
            "unary",
            node.kind,
            node.label,
            child_signatures[0],
        )
    if len(child_signatures) == 2:
        return (
            source_mode,
            "binary",
            node.kind,
            node.label,
            child_signatures[0],
            child_signatures[1],
        )
    raise ValueError(
        f"n-ary structural signature for node {node.id} has {len(child_signatures)} children"
    )


def _leaf_structural_value(node: _TreeNode) -> tuple[tuple[str, object], ...]:
    if node.kind not in {"constant"}:
        return ()
    structural_keys = ("denominator", "numerator", "value")
    return tuple(
        (key, _freeze_value(node.metadata[key])) for key in structural_keys if key in node.metadata
    )


def _freeze_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_value(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _validate_node_allowed(*, source_mode: str, kind: str, label: str) -> None:
    if kind in FORBIDDEN_DAG_KINDS:
        raise ValueError(f"DAG compression forbids node kind {kind!r}")
    if label in FORBIDDEN_DAG_LABELS:
        raise ValueError(f"DAG compression forbids node label {label!r}")

    if source_mode == "restricted_eml_pure":
        if kind == "eml":
            if label != "eml":
                raise ValueError(f"pure EML internal nodes must be labeled 'eml', got {label!r}")
            return
        if kind == "constant":
            if label != "1":
                raise ValueError(f"pure EML constant leaves must be '1', got {label!r}")
            return
        if kind == "variable":
            return
        raise ValueError(f"pure EML DAG compression forbids node kind {kind!r}")


def _validate_child_slots(parent_id: int, refs: Sequence[DagChildRef]) -> None:
    if len(refs) > 2:
        raise ValueError(
            f"n-ary DAG node {parent_id} has {len(refs)} child refs; "
            "Goal 3 DAG compression supports only leaf, unary, and binary nodes"
        )
    sorted_refs = sorted(refs, key=lambda ref: ref.slot_index)
    slot_indices = [ref.slot_index for ref in sorted_refs]
    if len(set(slot_indices)) != len(slot_indices):
        raise ValueError(f"DAG node {parent_id} has duplicate child slot indices")
    expected_indices = list(range(len(sorted_refs)))
    if slot_indices != expected_indices:
        raise ValueError(
            f"DAG node {parent_id} has child slot indices {slot_indices}; "
            f"expected {expected_indices}"
        )
    for ref in sorted_refs:
        expected_slot = _slot_name(ref.slot_index, len(sorted_refs))
        if ref.child_slot != expected_slot:
            raise ValueError(
                f"DAG child ref {parent_id}->{ref.child_id} has slot {ref.child_slot!r}; "
                f"expected {expected_slot!r}"
            )


def _slot_name(position: int, child_count: int) -> str:
    if child_count == 1 and position == 0:
        return "arg0"
    if child_count == 2 and position == 0:
        return "left"
    if child_count == 2 and position == 1:
        return "right"
    raise ValueError(f"unsupported child position {position} for child count {child_count}")


def _source_tree_node_ids(node: DagNode) -> list[int]:
    source_ids = node.metadata.get("source_tree_node_ids", [])
    if not isinstance(source_ids, list):
        raise TypeError("source_tree_node_ids metadata must be a list")
    return [int(source_id) for source_id in source_ids]
