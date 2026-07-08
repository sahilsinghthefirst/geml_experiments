"""Converters and writers for neutral graph export records."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from geml.compression.graph_schema import (
    EdgeType,
    GraphExportEdge,
    GraphExportNode,
    GraphExportRecord,
    GraphValidationStatus,
    RepresentationMode,
    validate_graph_record,
)
from geml.compression.macro_graph import MacroGraph
from geml.compression.motif_mining import MiningGraph, MiningNode
from geml.compression.motif_rewrite import MotifCompressedGraph
from geml.symbolic.ast_graph import AstTree
from geml.symbolic.dag_graph import DagGraph

type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]


def graph_record_from_ast_tree(
    tree: AstTree,
    *,
    graph_id: str,
    source_expression_id: int,
    subset_label: str,
    split: str,
) -> GraphExportRecord:
    """Convert a source AST tree into a neutral export graph."""
    tree_edges = [(edge.source, edge.target, edge.position) for edge in tree.edges]
    child_counts = _tree_child_counts(tree_edges)
    incoming_slots = _tree_incoming_slots(tree_edges)
    nodes = [
        GraphExportNode(
            node_id=_node_id(graph_id, node.id),
            graph_id=graph_id,
            representation_mode="ast_tree_graph",
            node_type=node.kind,
            label=node.label,
            arity=child_counts[node.id],
            child_slot=incoming_slots.get(node.id),
            source_expression_id=source_expression_id,
            expansion_available=False,
            expansion_target_ids=[],
            pure_eml_valid=False,
            metadata=dict(node.metadata),
        )
        for node in tree.nodes
    ]
    edges = [
        GraphExportEdge(
            edge_id=_edge_id(graph_id, edge.source, edge.target, edge.position),
            graph_id=graph_id,
            source_id=_node_id(graph_id, edge.source),
            target_id=_node_id(graph_id, edge.target),
            edge_type="ast_child",
            child_slot=_slot_name(edge.position, child_counts[edge.source]),
            slot_index=edge.position,
        )
        for edge in tree.edges
    ]
    return _record(
        graph_id=graph_id,
        source_expression_id=source_expression_id,
        representation_mode="ast_tree_graph",
        root_node_id=_node_id(graph_id, tree.root_id),
        nodes=nodes,
        edges=edges,
        subset_label=subset_label,
        split=split,
        metadata={
            "source": "sympy_ast_tree",
            "pure_eml_valid": False,
            "reconstruction_valid": True,
            "contains_hidden_target_labels": False,
        },
    )


def graph_record_from_dag(
    dag: DagGraph,
    *,
    graph_id: str,
    representation_mode: RepresentationMode,
    source_expression_id: int,
    subset_label: str,
    split: str,
    source_label: str,
) -> GraphExportRecord:
    """Convert an exact structural DAG into a neutral export graph."""
    edge_type: EdgeType = "ast_child" if representation_mode == "ast_dag_graph" else "eml_child"
    child_counts = _child_counts(
        (ref.parent_id, ref.child_id, ref.slot_index) for ref in dag.child_refs
    )
    incoming_slots = _incoming_slots(
        (ref.parent_id, ref.child_id, ref.child_slot, ref.slot_index) for ref in dag.child_refs
    )
    pure_eml_valid = representation_mode in {
        "pure_eml_dag_graph",
        "egraph_safe_eml_dag_graph",
        "egraph_positive_real_eml_dag_graph",
    }
    nodes = [
        GraphExportNode(
            node_id=_node_id(graph_id, node.id),
            graph_id=graph_id,
            representation_mode=representation_mode,
            node_type=node.kind,
            label=node.label,
            arity=child_counts[node.id],
            child_slot=incoming_slots.get(node.id),
            source_expression_id=source_expression_id,
            expansion_available=False,
            expansion_target_ids=[],
            pure_eml_valid=pure_eml_valid,
            metadata=_clean_metadata(node.metadata),
        )
        for node in dag.nodes
    ]
    edges = [
        GraphExportEdge(
            edge_id=_edge_id(graph_id, ref.parent_id, ref.child_id, ref.slot_index),
            graph_id=graph_id,
            source_id=_node_id(graph_id, ref.parent_id),
            target_id=_node_id(graph_id, ref.child_id),
            edge_type=edge_type,
            child_slot=ref.child_slot,
            slot_index=ref.slot_index,
        )
        for ref in dag.child_refs
    ]
    return _record(
        graph_id=graph_id,
        source_expression_id=source_expression_id,
        representation_mode=representation_mode,
        root_node_id=_node_id(graph_id, dag.root_id),
        nodes=nodes,
        edges=edges,
        subset_label=subset_label,
        split=split,
        metadata={
            "source": source_label,
            "pure_eml_valid": pure_eml_valid,
            "reconstruction_valid": True,
            "contains_hidden_target_labels": False,
            "dag_source_representation_mode": dag.metadata.get("source_representation_mode"),
        },
    )


def graph_record_from_macro_graph(
    graph: MacroGraph,
    *,
    graph_id: str,
    source_expression_id: int,
    subset_label: str,
    split: str,
    pure_eml_target_ids: Sequence[str],
) -> GraphExportRecord:
    """Convert a macro graph into a neutral export graph."""
    child_counts = _child_counts(
        (ref.parent_id, ref.child_id, ref.slot_index) for ref in graph.child_refs
    )
    incoming_slots = _incoming_slots(
        (ref.parent_id, ref.child_id, ref.child_slot, ref.slot_index) for ref in graph.child_refs
    )
    nodes = [
        GraphExportNode(
            node_id=_node_id(graph_id, node.id),
            graph_id=graph_id,
            representation_mode="macro_graph",
            node_type="macro",
            label=node.macro_name,
            arity=child_counts[node.id],
            child_slot=incoming_slots.get(node.id),
            source_expression_id=source_expression_id,
            expansion_available=node.expansion_to_pure_eml_available,
            expansion_target_ids=list(pure_eml_target_ids),
            pure_eml_valid=False,
            macro_name=node.macro_name,
            metadata={
                **_clean_metadata(node.metadata),
                "expansion_rule_name": node.expansion_rule_name,
                "pure_eml_expansion_node_count": node.pure_eml_expansion_node_count,
                "pure_eml_expansion_dag_node_count": node.pure_eml_expansion_dag_node_count,
            },
        )
        for node in graph.nodes
    ]
    edges = [
        GraphExportEdge(
            edge_id=_edge_id(graph_id, ref.parent_id, ref.child_id, ref.slot_index),
            graph_id=graph_id,
            source_id=_node_id(graph_id, ref.parent_id),
            target_id=_node_id(graph_id, ref.child_id),
            edge_type="macro_child",
            child_slot=ref.child_slot,
            slot_index=ref.slot_index,
        )
        for ref in graph.child_refs
    ]
    for node in graph.nodes:
        for target_index, target_id in enumerate(pure_eml_target_ids):
            edges.append(
                GraphExportEdge(
                    edge_id=f"{graph_id}:macro_expand:{node.id}:{target_index}",
                    graph_id=graph_id,
                    source_id=_node_id(graph_id, node.id),
                    target_id=target_id,
                    edge_type="macro_expands_to_eml",
                    child_slot="expansion",
                    slot_index=target_index,
                    metadata={"expansion_rule_name": node.expansion_rule_name},
                )
            )
    return _record(
        graph_id=graph_id,
        source_expression_id=source_expression_id,
        representation_mode="macro_graph",
        root_node_id=_node_id(graph_id, graph.root_id),
        nodes=nodes,
        edges=edges,
        subset_label=subset_label,
        split=split,
        metadata={
            "source": "goal5_macro_graph_v1",
            "pure_eml_valid": False,
            "reconstruction_valid": True,
            "contains_hidden_target_labels": False,
            "is_pure_eml": False,
        },
        validate_targets=False,
    )


def graph_record_from_mining_graph(
    graph: MiningGraph,
    *,
    graph_id: str,
    representation_mode: RepresentationMode,
    source_expression_id: int,
    subset_label: str,
    split: str,
    edge_type: EdgeType,
    pure_eml_target_ids: Sequence[str],
) -> GraphExportRecord:
    """Convert a motif-mining graph into a neutral graph record."""
    child_counts = _child_counts(
        (ref.parent_id, ref.child_id, ref.slot_index) for ref in graph.child_refs
    )
    incoming_slots = _incoming_slots(
        (ref.parent_id, ref.child_id, ref.child_slot, ref.slot_index) for ref in graph.child_refs
    )
    nodes = [
        _node_from_mining_node(
            node,
            graph_id=graph_id,
            representation_mode=representation_mode,
            source_expression_id=source_expression_id,
            arity=child_counts[node.id],
            child_slot=incoming_slots.get(node.id),
            pure_eml_target_ids=pure_eml_target_ids,
            learned=representation_mode == "learned_motif_graph",
        )
        for node in graph.nodes
    ]
    edges = [
        GraphExportEdge(
            edge_id=_edge_id(graph_id, ref.parent_id, ref.child_id, ref.slot_index),
            graph_id=graph_id,
            source_id=_node_id(graph_id, ref.parent_id),
            target_id=_node_id(graph_id, ref.child_id),
            edge_type=edge_type,
            child_slot=ref.child_slot,
            slot_index=ref.slot_index,
        )
        for ref in graph.child_refs
    ]
    return _record(
        graph_id=graph_id,
        source_expression_id=source_expression_id,
        representation_mode=representation_mode,
        root_node_id=_node_id(graph_id, graph.root_id),
        nodes=nodes,
        edges=edges,
        subset_label=subset_label,
        split=split,
        metadata={
            "source": graph.metadata.get("source_graph_type", "mining_graph"),
            "pure_eml_valid": graph.graph_type == "pure_eml_dag",
            "reconstruction_valid": True,
            "contains_hidden_target_labels": False,
            "graph_type": graph.graph_type,
        },
    )


def graph_record_from_motif_compressed_graph(
    graph: MotifCompressedGraph,
    *,
    graph_id: str,
    representation_mode: RepresentationMode,
    source_expression_id: int,
    subset_label: str,
    split: str,
    pure_eml_target_ids: Sequence[str],
    learned: bool = False,
) -> GraphExportRecord:
    """Convert a materialized motif-compressed graph into a neutral graph record."""
    edge_type: EdgeType = "learned_motif_instance" if learned else "motif_instance"
    mining_record = graph_record_from_mining_graph(
        MiningGraph(
            graph_id=graph_id,
            graph_type=graph.source_graph_type,  # type: ignore[arg-type]
            nodes=graph.nodes,
            child_refs=graph.child_refs,
            root_id=graph.root_id,
            metadata={
                **dict(graph.metadata),
                "source_graph_type": graph.source_graph_type,
            },
        ),
        graph_id=graph_id,
        representation_mode=representation_mode,
        source_expression_id=source_expression_id,
        subset_label=subset_label,
        split=split,
        edge_type=edge_type,
        pure_eml_target_ids=pure_eml_target_ids,
    )
    replacement_edges = []
    replacement_by_node = {
        replacement.motif_node_id: replacement for replacement in graph.motif_replacements
    }
    nodes = []
    for node in mining_record.nodes:
        source_node_id = _source_node_int(node.node_id)
        replacement = replacement_by_node.get(source_node_id)
        if replacement is not None:
            node_type = "learned_motif" if learned else "motif"
            node = node.model_copy(
                update={
                    "node_type": node_type,
                    "motif_id": replacement.motif_id,
                    "expansion_available": True,
                    "expansion_target_ids": list(pure_eml_target_ids),
                    "pure_eml_valid": False,
                    "metadata": {
                        **node.metadata,
                        "motif_id": replacement.motif_id,
                        "motif_type": replacement.motif_type,
                        "expansion_map_to_original_graph": (
                            replacement.expansion_map_to_original_graph
                        ),
                        "internal_node_ids": list(replacement.internal_node_ids),
                    },
                }
            )
            for target_index, target_id in enumerate(pure_eml_target_ids):
                replacement_edges.append(
                    GraphExportEdge(
                        edge_id=f"{graph_id}:motif_expand:{replacement.replacement_id}:{target_index}",
                        graph_id=graph_id,
                        source_id=node.node_id,
                        target_id=target_id,
                        edge_type="motif_expands_to_eml"
                        if not learned
                        else "learned_motif_instance",
                        child_slot="expansion",
                        slot_index=target_index,
                        metadata={"motif_id": replacement.motif_id},
                    )
                )
        nodes.append(node)
    return _record(
        graph_id=graph_id,
        source_expression_id=source_expression_id,
        representation_mode=representation_mode,
        root_node_id=mining_record.root_node_id,
        nodes=nodes,
        edges=[*mining_record.edges, *replacement_edges],
        subset_label=subset_label,
        split=split,
        metadata={
            **mining_record.metadata,
            "source": "motif_compressed_graph_v1",
            "reconstruction_valid": True,
            "pure_eml_valid": False,
            "selected_replacement_count": len(graph.motif_replacements),
            "source_graph_type": graph.source_graph_type,
        },
        validate_targets=False,
    )


def write_graph_records_jsonl(records: Iterable[GraphExportRecord], path: Path) -> int:
    """Write graph records to JSONL and return the written count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
            count += 1
    return count


def write_splits_json(
    *,
    graph_ids_by_split: dict[str, list[str]],
    expression_ids_by_split: dict[str, list[int]],
    path: Path,
) -> None:
    """Write deterministic train/validation/test split metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "goal5_graph_splits_v1",
        "contains_gnn_training": False,
        "graph_ids_by_split": {key: sorted(values) for key, values in graph_ids_by_split.items()},
        "expression_ids_by_split": {
            key: sorted(set(values)) for key, values in expression_ids_by_split.items()
        },
        "split_counts": {key: len(set(values)) for key, values in expression_ids_by_split.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def graph_can_reconstruct_pure_eml_dag(record: GraphExportRecord) -> bool:
    """Return whether a graph carries a path back to official pure EML-DAG."""
    if record.representation_mode in {
        "pure_eml_dag_graph",
        "egraph_safe_eml_dag_graph",
        "egraph_positive_real_eml_dag_graph",
    }:
        return record.validation.schema_valid and all(node.pure_eml_valid for node in record.nodes)
    return record.validation.reconstruction_valid and record.validation.expansion_valid


def summarize_graph_records(records: Sequence[GraphExportRecord]) -> dict[str, object]:
    """Summarize exported graph records by representation mode."""
    by_mode: dict[str, list[GraphExportRecord]] = defaultdict(list)
    for record in records:
        by_mode[record.representation_mode].append(record)
    return {
        mode: {
            "graph_count": len(mode_records),
            "node_count": _distribution(len(record.nodes) for record in mode_records),
            "edge_count": _distribution(len(record.edges) for record in mode_records),
            "expansion_validation_rate": _percent(
                sum(record.validation.expansion_valid for record in mode_records),
                len(mode_records),
            ),
            "reconstruction_validation_rate": _percent(
                sum(record.validation.reconstruction_valid for record in mode_records),
                len(mode_records),
            ),
            "missing_expansion_count": sum(
                record.validation.missing_expansion_count for record in mode_records
            ),
        }
        for mode, mode_records in sorted(by_mode.items())
    }


def _node_from_mining_node(
    node: MiningNode,
    *,
    graph_id: str,
    representation_mode: RepresentationMode,
    source_expression_id: int,
    arity: int,
    child_slot: str | None,
    pure_eml_target_ids: Sequence[str],
    learned: bool,
) -> GraphExportNode:
    label = node.label
    kind = node.kind
    metadata = _clean_metadata(dict(node.metadata))
    is_macro = kind == "macro"
    is_motif = kind == "motif"
    node_type = "learned_motif" if learned and is_motif else kind
    motif_id = metadata.get("motif_id") if isinstance(metadata.get("motif_id"), str) else None
    macro_name = label if is_macro else None
    expansion_available = bool(is_macro or is_motif)
    pure_eml_valid = representation_mode == "frequent_motif_graph" and not expansion_available
    return GraphExportNode(
        node_id=_node_id(graph_id, node.id),
        graph_id=graph_id,
        representation_mode=representation_mode,
        node_type=node_type,
        label=label,
        arity=arity,
        child_slot=child_slot,
        source_expression_id=source_expression_id,
        expansion_available=expansion_available,
        expansion_target_ids=list(pure_eml_target_ids) if expansion_available else [],
        pure_eml_valid=pure_eml_valid,
        motif_id=motif_id,
        macro_name=macro_name,
        metadata=metadata,
    )


def _record(
    *,
    graph_id: str,
    source_expression_id: int,
    representation_mode: RepresentationMode,
    root_node_id: str,
    nodes: Sequence[GraphExportNode],
    edges: Sequence[GraphExportEdge],
    subset_label: str,
    split: str,
    metadata: dict[str, MetadataValue],
    validate_targets: bool = True,
) -> GraphExportRecord:
    validation = GraphValidationStatus(
        schema_valid=True,
        expansion_valid=True,
        reconstruction_valid=True,
        missing_expansion_count=0,
        pure_eml_valid=bool(metadata.get("pure_eml_valid", False)),
        errors=[],
    )
    record = GraphExportRecord(
        graph_id=graph_id,
        source_expression_id=source_expression_id,
        representation_mode=representation_mode,
        root_node_id=root_node_id,
        nodes=list(nodes),
        edges=list(edges),
        subset_label=subset_label,
        split=split,
        metadata=metadata,
        validation=validation,
    )
    if validate_targets:
        return record.model_copy(update={"validation": validate_graph_record(record)})
    return record.model_copy(update={"validation": _validate_external_targets_record(record)})


def _validate_external_targets_record(record: GraphExportRecord) -> GraphValidationStatus:
    validation = validate_graph_record(
        record.model_copy(
            update={
                "edges": [
                    edge
                    for edge in record.edges
                    if edge.edge_type
                    not in {
                        "macro_expands_to_eml",
                        "motif_expands_to_eml",
                        "learned_motif_instance",
                    }
                ]
            }
        )
    )
    return validation


def _tree_child_counts(edges: Iterable[tuple[int, int, int]]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for source, _target, _position in edges:
        counts[source] += 1
    return counts


def _tree_incoming_slots(edges: Iterable[tuple[int, int, int]]) -> dict[int, str]:
    edge_list = list(edges)
    incoming: dict[int, str] = {}
    child_counts = _tree_child_counts(edge_list)
    for source, target, position in edge_list:
        incoming.setdefault(target, _slot_name(position, child_counts[source]))
    return incoming


def _child_counts(edges: Iterable[tuple[int, int, int]]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for source, _target, _position in edges:
        counts[source] += 1
    return counts


def _incoming_slots(edges: Iterable[tuple[int, int, str, int]]) -> dict[int, str]:
    incoming: dict[int, str] = {}
    for _source, target, child_slot, _slot_index in edges:
        incoming.setdefault(target, child_slot)
    return incoming


def _node_id(graph_id: str, node_id: int) -> str:
    return f"{graph_id}:n{node_id}"


def _edge_id(graph_id: str, parent_id: int, child_id: int, slot_index: int) -> str:
    return f"{graph_id}:e{parent_id}:{slot_index}:{child_id}"


def _source_node_int(export_node_id: str) -> int:
    suffix = export_node_id.rsplit(":", maxsplit=1)[1]
    if suffix.startswith("n"):
        suffix = suffix[1:]
    return int(suffix)


def _slot_name(position: int, child_count: int) -> str:
    if child_count == 1:
        return "arg0"
    if child_count == 2 and position == 0:
        return "left"
    if child_count == 2 and position == 1:
        return "right"
    return f"arg{position}"


def _clean_metadata(metadata: dict[str, object]) -> dict[str, MetadataValue]:
    return {str(key): _clean_value(value) for key, value in metadata.items()}


def _clean_value(value: object) -> MetadataValue:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_clean_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _clean_value(item) for key, item in value.items()}
    return str(value)


def _distribution(values: Iterable[int | float]) -> dict[str, float | None]:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return {"mean": None, "median": None, "p90": None}
    return {
        "mean": statistics.fmean(numeric_values),
        "median": statistics.median(numeric_values),
        "p90": _quantile(numeric_values, 0.9),
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * float(numerator) / float(denominator)
