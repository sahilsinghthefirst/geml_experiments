"""Neutral graph export schema for future GNN dataset preparation."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]

RepresentationMode = Literal[
    "ast_tree_graph",
    "ast_dag_graph",
    "pure_eml_dag_graph",
    "egraph_safe_eml_dag_graph",
    "egraph_positive_real_eml_dag_graph",
    "macro_graph",
    "frequent_motif_graph",
    "learned_motif_graph",
    "hierarchical_eml_graph",
]

EdgeType = Literal[
    "ast_child",
    "ast_to_macro",
    "macro_child",
    "macro_expands_to_eml",
    "eml_child",
    "motif_instance",
    "motif_expands_to_eml",
    "learned_motif_instance",
    "hierarchy_parent_child",
]

REPRESENTATION_MODES: tuple[str, ...] = (
    "ast_tree_graph",
    "ast_dag_graph",
    "pure_eml_dag_graph",
    "egraph_safe_eml_dag_graph",
    "egraph_positive_real_eml_dag_graph",
    "macro_graph",
    "frequent_motif_graph",
    "learned_motif_graph",
    "hierarchical_eml_graph",
)

EDGE_TYPES: tuple[str, ...] = (
    "ast_child",
    "ast_to_macro",
    "macro_child",
    "macro_expands_to_eml",
    "eml_child",
    "motif_instance",
    "motif_expands_to_eml",
    "learned_motif_instance",
    "hierarchy_parent_child",
)

COMPRESSED_NODE_TYPES = frozenset({"macro", "motif", "learned_motif"})


class GraphExportNode(BaseModel):
    """One neutral exported graph node."""

    node_id: str
    graph_id: str
    representation_mode: RepresentationMode
    node_type: str
    label: str
    arity: int = Field(ge=0)
    child_slot: str | None = None
    source_expression_id: int = Field(ge=0)
    expansion_available: bool = False
    expansion_target_ids: list[str] = Field(default_factory=list)
    pure_eml_valid: bool = False
    motif_id: str | None = None
    macro_name: str | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class GraphExportEdge(BaseModel):
    """One neutral exported graph edge."""

    edge_id: str
    graph_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    child_slot: str
    slot_index: int = Field(ge=0)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class GraphValidationStatus(BaseModel):
    """Validation status attached to one exported graph record."""

    schema_valid: bool
    expansion_valid: bool
    reconstruction_valid: bool
    missing_expansion_count: int = Field(ge=0)
    pure_eml_valid: bool
    errors: list[str] = Field(default_factory=list)


class GraphExportRecord(BaseModel):
    """One exported graph in a neutral JSONL schema."""

    graph_id: str
    source_expression_id: int = Field(ge=0)
    representation_mode: RepresentationMode
    root_node_id: str
    nodes: list[GraphExportNode]
    edges: list[GraphExportEdge]
    subset_label: str
    split: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    validation: GraphValidationStatus


def validate_graph_record(record: GraphExportRecord) -> GraphValidationStatus:
    """Validate graph schema invariants and compressed-node expansion maps."""
    errors: list[str] = []
    node_ids = [node.node_id for node in record.nodes]
    node_id_set = set(node_ids)
    if len(node_id_set) != len(node_ids):
        errors.append("node ids are not unique")
    if record.root_node_id not in node_id_set:
        errors.append("root node id is missing")
    for node in record.nodes:
        if node.graph_id != record.graph_id:
            errors.append(f"node {node.node_id} graph_id mismatch")
        if node.representation_mode != record.representation_mode:
            errors.append(f"node {node.node_id} representation_mode mismatch")
        if node.source_expression_id != record.source_expression_id:
            errors.append(f"node {node.node_id} source_expression_id mismatch")
        if node.node_type in COMPRESSED_NODE_TYPES and not node.expansion_target_ids:
            errors.append(f"compressed node {node.node_id} has no expansion targets")
        if node.node_type in COMPRESSED_NODE_TYPES and not node.expansion_available:
            errors.append(f"compressed node {node.node_id} has expansion_available=False")

    edge_ids = [edge.edge_id for edge in record.edges]
    if len(set(edge_ids)) != len(edge_ids):
        errors.append("edge ids are not unique")
    refs_by_parent: dict[str, list[GraphExportEdge]] = defaultdict(list)
    for edge in record.edges:
        if edge.graph_id != record.graph_id:
            errors.append(f"edge {edge.edge_id} graph_id mismatch")
        if edge.source_id not in node_id_set:
            errors.append(f"edge {edge.edge_id} source missing")
        if edge.target_id not in node_id_set:
            errors.append(f"edge {edge.edge_id} target missing")
        refs_by_parent[edge.source_id].append(edge)

    for parent_id, edges in refs_by_parent.items():
        ordered_child_edges = [
            edge for edge in edges if edge.edge_type in {"ast_child", "macro_child", "eml_child"}
        ]
        if not ordered_child_edges:
            continue
        slot_indices = [edge.slot_index for edge in sorted(ordered_child_edges, key=_edge_sort_key)]
        if len(set(slot_indices)) != len(slot_indices):
            errors.append(f"parent {parent_id} has duplicate child slot indices")

    missing_expansion_count = sum(
        1
        for node in record.nodes
        if node.node_type in COMPRESSED_NODE_TYPES and not node.expansion_target_ids
    )
    expansion_valid = missing_expansion_count == 0 and not any(
        "expansion_available=False" in error for error in errors
    )
    if "reconstruction_valid" not in record.metadata:
        errors.append("record metadata is missing reconstruction_valid")
    schema_valid = not errors
    reconstruction_valid = bool(record.metadata.get("reconstruction_valid", False))
    pure_eml_valid = bool(record.metadata.get("pure_eml_valid", False))
    return GraphValidationStatus(
        schema_valid=schema_valid,
        expansion_valid=expansion_valid,
        reconstruction_valid=reconstruction_valid and expansion_valid,
        missing_expansion_count=missing_expansion_count,
        pure_eml_valid=pure_eml_valid,
        errors=errors,
    )


def graph_schema_document() -> dict[str, object]:
    """Return the versioned JSON schema descriptor for exported graph records."""
    return {
        "schema_version": "goal5_hierarchical_graph_schema_v1",
        "format": "jsonl",
        "graph_record_model": GraphExportRecord.model_json_schema(),
        "representation_modes": list(REPRESENTATION_MODES),
        "edge_types": list(EDGE_TYPES),
        "required_node_metadata_fields": [
            "node_id",
            "graph_id",
            "representation_mode",
            "node_type",
            "label",
            "arity",
            "child_slot",
            "source_expression_id",
            "expansion_available",
            "expansion_target_ids",
            "pure_eml_valid",
            "motif_id",
            "macro_name",
        ],
        "integrity_contract": {
            "compressed_nodes_are_pure_eml": False,
            "compressed_nodes_require_expansion_targets": True,
            "safe_and_positive_real_modes_labeled_explicitly": True,
            "contains_gnn_training": False,
            "contains_hidden_target_labels": False,
        },
    }


def write_graph_schema(path: Path) -> None:
    """Write the neutral graph schema descriptor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(graph_schema_document(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_graph_records(records: Sequence[GraphExportRecord]) -> tuple[GraphExportRecord, ...]:
    """Revalidate graph records and return copies with updated validation fields."""
    return tuple(
        record.model_copy(update={"validation": validate_graph_record(record)})
        for record in records
    )


def _edge_sort_key(edge: GraphExportEdge) -> tuple[int, str, str]:
    return (edge.slot_index, edge.source_id, edge.target_id)
