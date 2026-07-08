"""Frequent connected motif mining over pure EML-DAGs and macro graphs."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from geml.compression.macro_graph import MacroGraph
from geml.compression.motif_vocab import (
    MotifBoundaryRefTemplate,
    MotifChildRefTemplate,
    MotifNodeTemplate,
    MotifRecord,
    MotifType,
    MotifVocabulary,
)
from geml.symbolic.dag_graph import DagGraph

type GraphType = Literal["pure_eml_dag", "macro_graph"]
type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]

OFFICIAL_MACRO_LABELS = frozenset(
    {
        "eml_exp",
        "eml_log",
        "eml_add",
        "eml_mul",
        "eml_sub",
        "eml_neg",
        "eml_inv",
        "eml_div",
        "eml_pow",
        "eml_zero",
        "eml_integer",
        "eml_rational",
        "eml_variable",
    }
)


class MiningNode(BaseModel):
    """A graph node normalized for motif mining."""

    id: int = Field(ge=0)
    label: str
    kind: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class MiningChildRef(BaseModel):
    """A directed ordered child reference normalized for motif mining."""

    parent_id: int = Field(ge=0)
    child_id: int = Field(ge=0)
    child_slot: str
    slot_index: int = Field(ge=0)


class MiningGraph(BaseModel):
    """Graph normalized for structural motif enumeration."""

    graph_id: str
    graph_type: GraphType
    nodes: list[MiningNode]
    child_refs: list[MiningChildRef]
    root_id: int
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class MotifBoundaryOccurrenceRef(BaseModel):
    """A concrete boundary mapping for one motif occurrence."""

    parent_local_id: int = Field(ge=0)
    boundary_slot_index: int = Field(ge=0)
    boundary_slot: str
    child_slot: str
    slot_index: int = Field(ge=0)
    external_child_id: int = Field(ge=0)


class MotifOccurrence(BaseModel):
    """One concrete motif occurrence in one source graph."""

    graph_id: str
    graph_type: GraphType
    expression_index: int
    subset_label: str
    signature: str
    root_node_id: int = Field(ge=0)
    internal_node_ids: tuple[int, ...]
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    internal_nodes: tuple[MotifNodeTemplate, ...]
    internal_child_refs: tuple[MotifChildRefTemplate, ...] = ()
    boundary_child_refs: tuple[MotifBoundaryRefTemplate, ...] = ()
    boundary_occurrence_refs: tuple[MotifBoundaryOccurrenceRef, ...] = ()


@dataclass
class _MotifAggregate:
    signature: str
    motif_type: MotifType
    node_count: int
    edge_count: int
    internal_nodes: tuple[MotifNodeTemplate, ...]
    internal_child_refs: tuple[MotifChildRefTemplate, ...]
    boundary_child_refs: tuple[MotifBoundaryRefTemplate, ...]
    support_count: int = 0
    sample_occurrences: list[MotifOccurrence] = field(default_factory=list)
    graph_ids: set[str] = field(default_factory=set)
    support_by_subset: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    covered_by_subset: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def mining_graph_from_dag(dag: DagGraph, *, graph_id: str, expression_index: int) -> MiningGraph:
    """Convert an official pure EML-DAG into the motif-mining graph schema."""
    return MiningGraph(
        graph_id=graph_id,
        graph_type="pure_eml_dag",
        nodes=[
            MiningNode(
                id=node.id,
                label=node.label,
                kind=node.kind,
                metadata=dict(node.metadata),
            )
            for node in dag.nodes
        ],
        child_refs=[
            MiningChildRef(
                parent_id=ref.parent_id,
                child_id=ref.child_id,
                child_slot=ref.child_slot,
                slot_index=ref.slot_index,
            )
            for ref in dag.child_refs
        ],
        root_id=dag.root_id,
        metadata={
            "source_graph_type": "restricted_eml_pure_dag",
            "expression_index": expression_index,
            "is_pure_eml": True,
        },
    )


def mining_graph_from_macro_graph(
    graph: MacroGraph,
    *,
    graph_id: str,
    expression_index: int,
) -> MiningGraph:
    """Convert a macro graph into the motif-mining graph schema."""
    return MiningGraph(
        graph_id=graph_id,
        graph_type="macro_graph",
        nodes=[
            MiningNode(
                id=node.id,
                label=node.macro_name,
                kind="macro",
                metadata={
                    **dict(node.metadata),
                    "expansion_rule_name": node.expansion_rule_name,
                    "expansion_to_pure_eml_available": (node.expansion_to_pure_eml_available),
                    "is_pure_eml": False,
                },
            )
            for node in graph.nodes
        ],
        child_refs=[
            MiningChildRef(
                parent_id=ref.parent_id,
                child_id=ref.child_id,
                child_slot=ref.child_slot,
                slot_index=ref.slot_index,
            )
            for ref in graph.child_refs
        ],
        root_id=graph.root_id,
        metadata={
            "source_graph_type": "macro_graph_v1",
            "expression_index": expression_index,
            "is_pure_eml": False,
        },
    )


def enumerate_motif_occurrences(
    graph: MiningGraph,
    *,
    min_motif_nodes: int,
    max_motif_nodes: int,
    expression_index: int,
    subset_label: str,
) -> list[MotifOccurrence]:
    """Enumerate rooted connected descendant subgraph motifs with boundary refs."""
    if min_motif_nodes <= 0:
        raise ValueError("min_motif_nodes must be positive")
    if max_motif_nodes < min_motif_nodes:
        raise ValueError("max_motif_nodes must be >= min_motif_nodes")

    nodes_by_id = {node.id: node for node in graph.nodes}
    children_by_parent = _children_by_parent(graph.child_refs)
    occurrences: list[MotifOccurrence] = []
    seen_global: set[tuple[int, tuple[int, ...]]] = set()

    for root_id in sorted(nodes_by_id):
        root_set = frozenset({root_id})
        stack = [root_set]
        seen_for_root: set[frozenset[int]] = set()
        while stack:
            internal = stack.pop()
            if internal in seen_for_root:
                continue
            seen_for_root.add(internal)

            key = (root_id, tuple(sorted(internal)))
            if key not in seen_global:
                seen_global.add(key)
                if min_motif_nodes <= len(internal) <= max_motif_nodes:
                    occurrence = build_occurrence(
                        graph,
                        root_id=root_id,
                        internal_node_ids=internal,
                        expression_index=expression_index,
                        subset_label=subset_label,
                    )
                    if occurrence.edge_count > 0:
                        occurrences.append(occurrence)

            if len(internal) >= max_motif_nodes:
                continue
            for child_id in reversed(_frontier_child_ids(internal, children_by_parent)):
                expanded = frozenset((*internal, child_id))
                if expanded not in seen_for_root:
                    stack.append(expanded)
    return occurrences


def build_occurrence(
    graph: MiningGraph,
    *,
    root_id: int,
    internal_node_ids: frozenset[int],
    expression_index: int,
    subset_label: str,
) -> MotifOccurrence:
    """Build a canonical motif occurrence from a rooted internal node set."""
    nodes_by_id = {node.id: node for node in graph.nodes}
    children_by_parent = _children_by_parent(graph.child_refs)
    local_ids = _canonical_local_ids(root_id, internal_node_ids, children_by_parent)
    internal_nodes = tuple(
        MotifNodeTemplate(
            local_id=local_id,
            label=nodes_by_id[node_id].label,
            kind=nodes_by_id[node_id].kind,
            metadata=_template_node_metadata(nodes_by_id[node_id]),
        )
        for node_id, local_id in sorted(local_ids.items(), key=lambda item: item[1])
    )
    internal_refs: list[MotifChildRefTemplate] = []
    boundary_templates: list[MotifBoundaryRefTemplate] = []
    boundary_occurrences: list[MotifBoundaryOccurrenceRef] = []
    boundary_index = 0

    for node_id, parent_local_id in sorted(local_ids.items(), key=lambda item: item[1]):
        for ref in children_by_parent.get(node_id, []):
            if ref.child_id in local_ids:
                internal_refs.append(
                    MotifChildRefTemplate(
                        parent_local_id=parent_local_id,
                        child_local_id=local_ids[ref.child_id],
                        child_slot=ref.child_slot,
                        slot_index=ref.slot_index,
                    )
                )
            else:
                boundary_slot = f"boundary_{boundary_index}"
                boundary_templates.append(
                    MotifBoundaryRefTemplate(
                        parent_local_id=parent_local_id,
                        boundary_slot_index=boundary_index,
                        boundary_slot=boundary_slot,
                        child_slot=ref.child_slot,
                        slot_index=ref.slot_index,
                    )
                )
                boundary_occurrences.append(
                    MotifBoundaryOccurrenceRef(
                        parent_local_id=parent_local_id,
                        boundary_slot_index=boundary_index,
                        boundary_slot=boundary_slot,
                        child_slot=ref.child_slot,
                        slot_index=ref.slot_index,
                        external_child_id=ref.child_id,
                    )
                )
                boundary_index += 1

    signature = motif_signature(
        graph_type=graph.graph_type,
        internal_nodes=internal_nodes,
        internal_child_refs=tuple(internal_refs),
        boundary_child_refs=tuple(boundary_templates),
    )
    return MotifOccurrence(
        graph_id=graph.graph_id,
        graph_type=graph.graph_type,
        expression_index=expression_index,
        subset_label=subset_label,
        signature=signature,
        root_node_id=root_id,
        internal_node_ids=tuple(sorted(internal_node_ids)),
        node_count=len(internal_node_ids),
        edge_count=len(internal_refs) + len(boundary_templates),
        internal_nodes=internal_nodes,
        internal_child_refs=tuple(internal_refs),
        boundary_child_refs=tuple(boundary_templates),
        boundary_occurrence_refs=tuple(boundary_occurrences),
    )


def mine_frequent_motifs(
    graphs: Sequence[MiningGraph],
    *,
    min_motif_nodes: int,
    max_motif_nodes: int,
    min_support: int,
    max_vocab_size: int,
    subset_labels_by_graph_id: dict[str, str],
    expression_indices_by_graph_id: dict[str, int],
) -> list[MotifRecord]:
    """Mine frequent motifs for one graph family."""
    aggregates: dict[str, _MotifAggregate] = {}
    for graph in graphs:
        subset_label = subset_labels_by_graph_id.get(graph.graph_id, "all_v1")
        expression_index = expression_indices_by_graph_id.get(graph.graph_id, -1)
        occurrences = enumerate_motif_occurrences(
            graph,
            min_motif_nodes=min_motif_nodes,
            max_motif_nodes=max_motif_nodes,
            expression_index=expression_index,
            subset_label=subset_label,
        )
        for occurrence in occurrences:
            aggregate = aggregates.get(occurrence.signature)
            if aggregate is None:
                aggregate = _MotifAggregate(
                    signature=occurrence.signature,
                    motif_type=occurrence.graph_type,
                    node_count=occurrence.node_count,
                    edge_count=occurrence.edge_count,
                    internal_nodes=occurrence.internal_nodes,
                    internal_child_refs=occurrence.internal_child_refs,
                    boundary_child_refs=occurrence.boundary_child_refs,
                )
                aggregates[occurrence.signature] = aggregate
            aggregate.support_count += 1
            if len(aggregate.sample_occurrences) < 5:
                aggregate.sample_occurrences.append(occurrence)
            aggregate.graph_ids.add(occurrence.graph_id)
            aggregate.support_by_subset[subset_label] += 1
            aggregate.covered_by_subset[subset_label] += occurrence.node_count

    records = [
        _aggregate_to_record(aggregate, motif_id=f"{aggregate.motif_type}_{position:04d}")
        for position, aggregate in enumerate(
            sorted(
                (
                    aggregate
                    for aggregate in aggregates.values()
                    if aggregate.support_count >= min_support
                ),
                key=_aggregate_sort_key,
            )
        )
    ]
    return records[:max_vocab_size]


def build_motif_vocabulary(
    *,
    pure_records: Sequence[MotifRecord],
    macro_records: Sequence[MotifRecord],
    max_vocab_size: int,
    config: dict[str, MetadataValue],
) -> MotifVocabulary:
    """Build a mixed vocabulary with pure, macro, and mixed expansion records."""
    if max_vocab_size <= 0:
        raise ValueError("max_vocab_size must be positive")
    pure_limit = max(1, max_vocab_size // 3)
    macro_limit = max(1, max_vocab_size // 3)
    mixed_limit = max(0, max_vocab_size - pure_limit - macro_limit)
    selected_pure = list(pure_records[:pure_limit])
    selected_macro = list(macro_records[:macro_limit])
    mixed = derive_mixed_motifs(selected_macro, max_count=mixed_limit)
    motifs = _renumber_motifs([*selected_pure, *selected_macro, *mixed])
    return MotifVocabulary(
        motifs=motifs,
        config=config,
        metadata={
            "representation_mode": "frequent_motif_vocab_v1",
            "is_pure_eml": False,
            "motif_types": ["pure_eml_dag", "macro_graph", "mixed_macro_expansion"],
            "selection_policy": (
                "top scored pure motifs, top scored macro motifs, then mixed macro expansions"
            ),
        },
    )


def derive_mixed_motifs(
    macro_records: Sequence[MotifRecord],
    *,
    max_count: int,
) -> list[MotifRecord]:
    """Derive mixed macro-to-pure expansion motif records from macro motifs."""
    mixed: list[MotifRecord] = []
    for position, macro_motif in enumerate(macro_records[:max_count]):
        raw = macro_motif.model_dump(mode="json")
        raw["motif_id"] = f"mixed_macro_expansion_{position:04d}"
        raw["motif_type"] = "mixed_macro_expansion"
        raw["signature"] = f"mixed:{macro_motif.signature}"
        raw["expansion_map_to_pure_eml_available"] = True
        raw["expansion_map_to_original_graph"] = {
            "source_macro_motif_id": macro_motif.motif_id,
            "macro_template": macro_motif.expansion_map_to_original_graph,
            "pure_eml_expansion_source": "official_eml_compiler macro expansion",
        }
        raw["metadata"] = {
            **dict(macro_motif.metadata),
            "source_macro_motif_id": macro_motif.motif_id,
            "mixed_motif_policy": "macro motif with official pure EML expansion map",
        }
        mixed.append(MotifRecord.model_validate(raw))
    return mixed


def graph_structural_signature(graph: MiningGraph) -> str:
    """Return a deterministic whole-graph structural signature."""
    node_rows = sorted((node.id, node.label, node.kind) for node in graph.nodes)
    ref_rows = sorted(
        (ref.parent_id, ref.child_id, ref.child_slot, ref.slot_index) for ref in graph.child_refs
    )
    return json.dumps(
        {
            "graph_type": graph.graph_type,
            "root_id": graph.root_id,
            "nodes": node_rows,
            "child_refs": ref_rows,
        },
        sort_keys=True,
    )


def motif_signature(
    *,
    graph_type: GraphType,
    internal_nodes: Sequence[MotifNodeTemplate],
    internal_child_refs: Sequence[MotifChildRefTemplate],
    boundary_child_refs: Sequence[MotifBoundaryRefTemplate],
) -> str:
    """Return a deterministic motif signature preserving ordered child slots."""
    payload = {
        "graph_type": graph_type,
        "nodes": [
            {
                "local_id": node.local_id,
                "label": node.label,
                "kind": node.kind,
                "metadata": _signature_metadata(node),
            }
            for node in internal_nodes
        ],
        "internal_refs": [
            {
                "parent": ref.parent_local_id,
                "child": ref.child_local_id,
                "child_slot": ref.child_slot,
                "slot_index": ref.slot_index,
            }
            for ref in internal_child_refs
        ],
        "boundary_refs": [
            {
                "parent": ref.parent_local_id,
                "boundary": ref.boundary_slot_index,
                "child_slot": ref.child_slot,
                "slot_index": ref.slot_index,
            }
            for ref in boundary_child_refs
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _aggregate_to_record(aggregate: _MotifAggregate, *, motif_id: str) -> MotifRecord:
    support_count = aggregate.support_count
    total_covered_nodes = support_count * aggregate.node_count
    compression_score = support_count * max(aggregate.node_count - 1, 0)
    official_macro_name = _official_macro_name(aggregate)
    sample_occurrences = [
        {
            "graph_id": occurrence.graph_id,
            "expression_index": occurrence.expression_index,
            "root_node_id": occurrence.root_node_id,
            "internal_node_ids": list(occurrence.internal_node_ids),
            "boundary_external_child_ids": [
                ref.external_child_id for ref in occurrence.boundary_occurrence_refs
            ],
        }
        for occurrence in aggregate.sample_occurrences
    ]
    return MotifRecord(
        motif_id=motif_id,
        motif_type=aggregate.motif_type,
        signature=aggregate.signature,
        node_count=aggregate.node_count,
        edge_count=aggregate.edge_count,
        support_count=support_count,
        support_graph_count=len(aggregate.graph_ids),
        total_covered_nodes=total_covered_nodes,
        compression_score=float(compression_score),
        expansion_map_to_original_graph={
            "root_local_node_id": 0,
            "internal_nodes": [node.model_dump(mode="json") for node in aggregate.internal_nodes],
            "internal_child_refs": [
                ref.model_dump(mode="json") for ref in aggregate.internal_child_refs
            ],
            "boundary_child_refs": [
                ref.model_dump(mode="json") for ref in aggregate.boundary_child_refs
            ],
            "output": "root node replaces the occurrence root",
        },
        expansion_map_to_pure_eml_available=aggregate.motif_type in {"pure_eml_dag", "macro_graph"},
        internal_nodes=list(aggregate.internal_nodes),
        internal_child_refs=list(aggregate.internal_child_refs),
        boundary_child_refs=list(aggregate.boundary_child_refs),
        support_by_subset_label=dict(aggregate.support_by_subset),
        covered_nodes_by_subset_label=dict(aggregate.covered_by_subset),
        official_macro_name=official_macro_name,
        is_obvious_official_macro=official_macro_name is not None,
        sample_occurrences=sample_occurrences,
        metadata={
            "motif_is_pure_eml_node": False,
            "boundary_input_slot_count": len(aggregate.boundary_child_refs),
        },
    )


def _aggregate_sort_key(aggregate: _MotifAggregate) -> tuple[float, int, int, str]:
    support_count = aggregate.support_count
    compression_score = support_count * max(aggregate.node_count - 1, 0)
    return (-float(compression_score), -support_count, -aggregate.node_count, aggregate.signature)


def _renumber_motifs(records: Sequence[MotifRecord]) -> list[MotifRecord]:
    counters: dict[str, int] = defaultdict(int)
    renumbered: list[MotifRecord] = []
    for record in records:
        prefix = record.motif_type
        new_id = f"{prefix}_{counters[prefix]:04d}"
        counters[prefix] += 1
        raw = record.model_dump(mode="json")
        raw["motif_id"] = new_id
        renumbered.append(MotifRecord.model_validate(raw))
    return renumbered


def _children_by_parent(child_refs: Iterable[MiningChildRef]) -> dict[int, list[MiningChildRef]]:
    grouped: dict[int, list[MiningChildRef]] = defaultdict(list)
    for ref in child_refs:
        grouped[ref.parent_id].append(ref)
    return {
        parent_id: sorted(refs, key=lambda ref: (ref.slot_index, ref.child_slot, ref.child_id))
        for parent_id, refs in grouped.items()
    }


def _frontier_child_ids(
    internal: frozenset[int],
    children_by_parent: dict[int, list[MiningChildRef]],
) -> list[int]:
    child_ids = {
        ref.child_id
        for parent_id in internal
        for ref in children_by_parent.get(parent_id, [])
        if ref.child_id not in internal
    }
    return sorted(child_ids)


def _canonical_local_ids(
    root_id: int,
    internal: frozenset[int],
    children_by_parent: dict[int, list[MiningChildRef]],
) -> dict[int, int]:
    local_ids: dict[int, int] = {}

    def visit(node_id: int) -> None:
        if node_id in local_ids:
            return
        local_ids[node_id] = len(local_ids)
        for ref in children_by_parent.get(node_id, []):
            if ref.child_id in internal:
                visit(ref.child_id)

    visit(root_id)
    for node_id in sorted(internal):
        if node_id not in local_ids:
            local_ids[node_id] = len(local_ids)
    return local_ids


def _template_node_metadata(node: MiningNode) -> dict[str, MetadataValue]:
    if node.kind == "constant":
        return {
            key: node.metadata[key]
            for key in ("value", "numerator", "denominator")
            if key in node.metadata
        }
    if node.kind == "macro":
        keys = (
            "symbol_name",
            "value",
            "numerator",
            "denominator",
            "expansion_rule_name",
            "expansion_to_pure_eml_available",
        )
        return {key: node.metadata[key] for key in keys if key in node.metadata}
    return {}


def _signature_metadata(node: MotifNodeTemplate) -> dict[str, MetadataValue]:
    structural_keys = (
        "value",
        "numerator",
        "denominator",
        "symbol_name",
        "expansion_rule_name",
    )
    return {key: node.metadata[key] for key in structural_keys if key in node.metadata}


def _official_macro_name(aggregate: _MotifAggregate) -> str | None:
    labels = [node.label for node in aggregate.internal_nodes]
    if aggregate.motif_type == "macro_graph" and labels and labels[0] in OFFICIAL_MACRO_LABELS:
        return labels[0]
    if aggregate.motif_type == "pure_eml_dag":
        return _pure_eml_macro_heuristic(aggregate)
    return None


def _pure_eml_macro_heuristic(aggregate: _MotifAggregate) -> str | None:
    labels = [node.label for node in aggregate.internal_nodes]
    if labels == ["eml", "1"] and aggregate.edge_count == 2:
        return "eml_exp"
    if labels and labels[0] == "eml" and aggregate.node_count >= 3:
        constant_count = sum(1 for label in labels if label == "1")
        if constant_count >= 1:
            return "official_eml_macro_fragment"
    return None
