"""Greedy motif replacement baseline with explicit expansion maps."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from geml.compression.motif_mining import (
    MiningChildRef,
    MiningGraph,
    MiningNode,
    MotifOccurrence,
    enumerate_motif_occurrences,
    graph_structural_signature,
)
from geml.compression.motif_vocab import MotifRecord

type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]


class MotifReplacement(BaseModel):
    """One selected motif replacement and its expansion map."""

    replacement_id: int = Field(ge=0)
    motif_id: str
    motif_type: str
    motif_node_id: int = Field(ge=0)
    root_node_id: int = Field(ge=0)
    internal_node_ids: tuple[int, ...]
    boundary_external_child_ids: tuple[int, ...]
    expansion_map_to_original_graph: dict[str, MetadataValue]


class MotifCompressedGraph(BaseModel):
    """A graph with non-overlapping motif nodes replacing source subgraphs."""

    representation_mode: str = "motif_compressed_graph_v1"
    source_graph_type: str
    nodes: list[MiningNode]
    child_refs: list[MiningChildRef]
    root_id: int
    motif_replacements: list[MotifReplacement]
    original_graph: MiningGraph
    original_graph_signature: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class MotifCompressionResult(BaseModel):
    """Metrics and graph for one greedy motif compression run."""

    compressed_graph: MotifCompressedGraph
    original_node_count: int = Field(ge=0)
    original_child_ref_count: int = Field(ge=0)
    compressed_node_count: int = Field(ge=0)
    compressed_child_ref_count: int = Field(ge=0)
    selected_replacement_count: int = Field(ge=0)
    covered_node_count: int = Field(ge=0)
    motif_coverage_percent: float
    expansion_valid: bool


class MotifCompressionSummary(BaseModel):
    """Lightweight greedy motif compression metrics without materializing a graph."""

    original_node_count: int = Field(ge=0)
    original_child_ref_count: int = Field(ge=0)
    compressed_node_count: int = Field(ge=0)
    compressed_child_ref_count: int = Field(ge=0)
    selected_replacement_count: int = Field(ge=0)
    covered_node_count: int = Field(ge=0)
    motif_coverage_percent: float
    expansion_valid: bool


def greedy_motif_compress_graph(
    graph: MiningGraph,
    motifs: Sequence[MotifRecord],
    *,
    min_motif_nodes: int,
    max_motif_nodes: int,
    expression_index: int,
    subset_label: str,
) -> MotifCompressionResult:
    """Apply a greedy non-overlapping frequent motif compression baseline."""
    selected, motifs_by_signature = select_greedy_motif_occurrences(
        graph,
        motifs,
        min_motif_nodes=min_motif_nodes,
        max_motif_nodes=max_motif_nodes,
        expression_index=expression_index,
        subset_label=subset_label,
    )
    compressed = build_motif_compressed_graph(
        graph,
        selected,
        motifs_by_signature=motifs_by_signature,
    )
    covered_node_count = sum(len(occurrence.internal_node_ids) for occurrence in selected)
    return MotifCompressionResult(
        compressed_graph=compressed,
        original_node_count=len(graph.nodes),
        original_child_ref_count=len(graph.child_refs),
        compressed_node_count=len(compressed.nodes),
        compressed_child_ref_count=len(compressed.child_refs),
        selected_replacement_count=len(selected),
        covered_node_count=covered_node_count,
        motif_coverage_percent=100.0 * covered_node_count / len(graph.nodes)
        if graph.nodes
        else 0.0,
        expansion_valid=compressed_graph_expands_to_original(compressed),
    )


def greedy_motif_compress_graph_summary(
    graph: MiningGraph,
    motifs: Sequence[MotifRecord],
    *,
    min_motif_nodes: int,
    max_motif_nodes: int,
    expression_index: int,
    subset_label: str,
) -> MotifCompressionSummary:
    """Apply greedy motif compression and return lightweight metrics."""
    selected, motifs_by_signature = select_greedy_motif_occurrences(
        graph,
        motifs,
        min_motif_nodes=min_motif_nodes,
        max_motif_nodes=max_motif_nodes,
        expression_index=expression_index,
        subset_label=subset_label,
    )
    occurrence_by_internal_node: dict[int, MotifOccurrence] = {}
    motif_node_ids: dict[int, int] = {}
    next_node_id = max((node.id for node in graph.nodes), default=-1) + 1
    for occurrence in selected:
        motif_node_ids[occurrence.root_node_id] = next_node_id
        next_node_id += 1
        for node_id in occurrence.internal_node_ids:
            occurrence_by_internal_node[node_id] = occurrence
    child_refs = _compressed_child_refs(
        graph,
        occurrence_by_internal_node=occurrence_by_internal_node,
        motif_node_ids=motif_node_ids,
    )
    covered_node_count = sum(len(occurrence.internal_node_ids) for occurrence in selected)
    compressed_node_count = len(graph.nodes) - covered_node_count + len(selected)
    expansion_valid = all(
        bool(motifs_by_signature[occurrence.signature].expansion_map_to_original_graph)
        for occurrence in selected
    )
    return MotifCompressionSummary(
        original_node_count=len(graph.nodes),
        original_child_ref_count=len(graph.child_refs),
        compressed_node_count=compressed_node_count,
        compressed_child_ref_count=len(child_refs),
        selected_replacement_count=len(selected),
        covered_node_count=covered_node_count,
        motif_coverage_percent=100.0 * covered_node_count / len(graph.nodes)
        if graph.nodes
        else 0.0,
        expansion_valid=expansion_valid,
    )


def greedy_motif_compress_occurrences_summary(
    graph: MiningGraph,
    occurrences: Sequence[MotifOccurrence],
    motifs: Sequence[MotifRecord],
) -> MotifCompressionSummary:
    """Apply greedy motif compression using pre-enumerated occurrences."""
    selected, motifs_by_signature = select_greedy_motif_occurrences_from_candidates(
        graph,
        occurrences,
        motifs,
    )
    occurrence_by_internal_node: dict[int, MotifOccurrence] = {}
    motif_node_ids: dict[int, int] = {}
    next_node_id = max((node.id for node in graph.nodes), default=-1) + 1
    for occurrence in selected:
        motif_node_ids[occurrence.root_node_id] = next_node_id
        next_node_id += 1
        for node_id in occurrence.internal_node_ids:
            occurrence_by_internal_node[node_id] = occurrence
    child_refs = _compressed_child_refs(
        graph,
        occurrence_by_internal_node=occurrence_by_internal_node,
        motif_node_ids=motif_node_ids,
    )
    covered_node_count = sum(len(occurrence.internal_node_ids) for occurrence in selected)
    compressed_node_count = len(graph.nodes) - covered_node_count + len(selected)
    expansion_valid = all(
        bool(motifs_by_signature[occurrence.signature].expansion_map_to_original_graph)
        for occurrence in selected
    )
    return MotifCompressionSummary(
        original_node_count=len(graph.nodes),
        original_child_ref_count=len(graph.child_refs),
        compressed_node_count=compressed_node_count,
        compressed_child_ref_count=len(child_refs),
        selected_replacement_count=len(selected),
        covered_node_count=covered_node_count,
        motif_coverage_percent=100.0 * covered_node_count / len(graph.nodes)
        if graph.nodes
        else 0.0,
        expansion_valid=expansion_valid,
    )


def select_greedy_motif_occurrences(
    graph: MiningGraph,
    motifs: Sequence[MotifRecord],
    *,
    min_motif_nodes: int,
    max_motif_nodes: int,
    expression_index: int,
    subset_label: str,
) -> tuple[list[MotifOccurrence], dict[str, MotifRecord]]:
    """Select non-overlapping motif occurrences without materializing replacements."""
    usable_motifs = [
        motif for motif in motifs if motif.motif_type == graph.graph_type and motif.node_count > 1
    ]
    motifs_by_signature = {motif.signature: motif for motif in usable_motifs}
    candidates = [
        occurrence
        for occurrence in enumerate_motif_occurrences(
            graph,
            min_motif_nodes=min_motif_nodes,
            max_motif_nodes=max_motif_nodes,
            expression_index=expression_index,
            subset_label=subset_label,
        )
        if occurrence.signature in motifs_by_signature and is_replacement_safe(graph, occurrence)
    ]
    return select_greedy_motif_occurrences_from_candidates(
        graph,
        candidates,
        usable_motifs,
    )


def select_greedy_motif_occurrences_from_candidates(
    graph: MiningGraph,
    occurrences: Sequence[MotifOccurrence],
    motifs: Sequence[MotifRecord],
) -> tuple[list[MotifOccurrence], dict[str, MotifRecord]]:
    """Select non-overlapping motif occurrences from precomputed candidates."""
    usable_motifs = [
        motif for motif in motifs if motif.motif_type == graph.graph_type and motif.node_count > 1
    ]
    motifs_by_signature = {motif.signature: motif for motif in usable_motifs}
    candidates = [
        occurrence
        for occurrence in occurrences
        if occurrence.signature in motifs_by_signature and is_replacement_safe(graph, occurrence)
    ]
    selected: list[MotifOccurrence] = []
    used_nodes: set[int] = set()
    for occurrence in sorted(
        candidates,
        key=lambda item: _candidate_sort_key(item, motifs_by_signature[item.signature]),
    ):
        internal_nodes = set(occurrence.internal_node_ids)
        if used_nodes.isdisjoint(internal_nodes):
            selected.append(occurrence)
            used_nodes.update(internal_nodes)
    return selected, motifs_by_signature


def build_motif_compressed_graph(
    graph: MiningGraph,
    selected_occurrences: Sequence[MotifOccurrence],
    *,
    motifs_by_signature: dict[str, MotifRecord],
) -> MotifCompressedGraph:
    """Build a motif-compressed graph from explicit non-overlapping occurrences."""
    validate_non_overlapping_occurrences(selected_occurrences)
    for occurrence in selected_occurrences:
        if not is_replacement_safe(graph, occurrence):
            raise ValueError(
                f"occurrence rooted at {occurrence.root_node_id} is not replacement-safe"
            )

    occurrence_by_internal_node: dict[int, MotifOccurrence] = {}
    occurrence_by_root: dict[int, MotifOccurrence] = {}
    for occurrence in selected_occurrences:
        occurrence_by_root[occurrence.root_node_id] = occurrence
        for node_id in occurrence.internal_node_ids:
            occurrence_by_internal_node[node_id] = occurrence

    next_node_id = max((node.id for node in graph.nodes), default=-1) + 1
    motif_node_ids: dict[int, int] = {}
    replacements: list[MotifReplacement] = []
    for replacement_id, occurrence in enumerate(selected_occurrences):
        motif_node_id = next_node_id
        next_node_id += 1
        motif_node_ids[occurrence.root_node_id] = motif_node_id
        motif = motifs_by_signature[occurrence.signature]
        replacements.append(
            MotifReplacement(
                replacement_id=replacement_id,
                motif_id=motif.motif_id,
                motif_type=motif.motif_type,
                motif_node_id=motif_node_id,
                root_node_id=occurrence.root_node_id,
                internal_node_ids=occurrence.internal_node_ids,
                boundary_external_child_ids=tuple(
                    ref.external_child_id for ref in occurrence.boundary_occurrence_refs
                ),
                expansion_map_to_original_graph={
                    "motif_expansion": motif.expansion_map_to_original_graph,
                    "root_node_id": occurrence.root_node_id,
                    "internal_node_ids": list(occurrence.internal_node_ids),
                    "boundary_refs": [
                        ref.model_dump(mode="json") for ref in occurrence.boundary_occurrence_refs
                    ],
                },
            )
        )

    removed_nodes = set(occurrence_by_internal_node)
    retained_nodes = [node for node in graph.nodes if node.id not in removed_nodes]
    motif_nodes = [
        MiningNode(
            id=replacement.motif_node_id,
            label=f"motif:{replacement.motif_id}",
            kind="motif",
            metadata={
                "motif_id": replacement.motif_id,
                "motif_type": replacement.motif_type,
                "is_pure_eml": False,
                "expansion_map_to_original_graph": replacement.expansion_map_to_original_graph,
            },
        )
        for replacement in replacements
    ]
    child_refs = _compressed_child_refs(
        graph,
        occurrence_by_internal_node=occurrence_by_internal_node,
        motif_node_ids=motif_node_ids,
    )
    root_id = _map_node_id(
        graph.root_id,
        occurrence_by_internal_node=occurrence_by_internal_node,
        motif_node_ids=motif_node_ids,
    )
    return MotifCompressedGraph(
        source_graph_type=graph.graph_type,
        nodes=[*retained_nodes, *motif_nodes],
        child_refs=child_refs,
        root_id=root_id,
        motif_replacements=replacements,
        original_graph=graph,
        original_graph_signature=graph_structural_signature(graph),
        metadata={
            "is_pure_eml": False,
            "motif_nodes_are_pure_eml": False,
            "overlap_policy": "non-overlapping internal node sets only",
            "expansion_policy": "motif replacement records expand to original graph nodes",
        },
    )


def expand_motif_compressed_graph(compressed_graph: MotifCompressedGraph) -> MiningGraph:
    """Expand a motif-compressed graph back to its original mining graph."""
    for replacement in compressed_graph.motif_replacements:
        if not replacement.expansion_map_to_original_graph:
            raise ValueError(f"replacement {replacement.replacement_id} has no expansion map")
    return compressed_graph.original_graph


def compressed_graph_expands_to_original(compressed_graph: MotifCompressedGraph) -> bool:
    """Return whether expansion reconstructs the original graph signature."""
    expanded = expand_motif_compressed_graph(compressed_graph)
    return graph_structural_signature(expanded) == compressed_graph.original_graph_signature


def validate_non_overlapping_occurrences(occurrences: Sequence[MotifOccurrence]) -> None:
    """Reject overlapping motif occurrences."""
    used_nodes: dict[int, int] = {}
    for occurrence_index, occurrence in enumerate(occurrences):
        for node_id in occurrence.internal_node_ids:
            existing = used_nodes.get(node_id)
            if existing is not None:
                raise ValueError(
                    "overlapping motif replacements are not allowed: "
                    f"node {node_id} appears in occurrences {existing} and {occurrence_index}"
                )
            used_nodes[node_id] = occurrence_index


def is_replacement_safe(graph: MiningGraph, occurrence: MotifOccurrence) -> bool:
    """Check that replacing an occurrence does not remove externally referenced internals."""
    internal = set(occurrence.internal_node_ids)
    incoming = _incoming_by_child(graph.child_refs)
    for node_id in internal:
        if node_id == occurrence.root_node_id:
            continue
        for ref in incoming.get(node_id, []):
            if ref.parent_id not in internal:
                return False
    return True


def _candidate_sort_key(occurrence: MotifOccurrence, motif: MotifRecord) -> tuple[float, int, int]:
    return (-motif.compression_score, -occurrence.node_count, occurrence.root_node_id)


def _compressed_child_refs(
    graph: MiningGraph,
    *,
    occurrence_by_internal_node: dict[int, MotifOccurrence],
    motif_node_ids: dict[int, int],
) -> list[MiningChildRef]:
    refs: list[MiningChildRef] = []
    boundary_counters: dict[int, int] = defaultdict(int)
    for ref in graph.child_refs:
        parent_occurrence = occurrence_by_internal_node.get(ref.parent_id)
        child_occurrence = occurrence_by_internal_node.get(ref.child_id)
        if parent_occurrence is not None and child_occurrence is parent_occurrence:
            continue

        mapped_parent = _map_node_id(
            ref.parent_id,
            occurrence_by_internal_node=occurrence_by_internal_node,
            motif_node_ids=motif_node_ids,
        )
        mapped_child = _map_node_id(
            ref.child_id,
            occurrence_by_internal_node=occurrence_by_internal_node,
            motif_node_ids=motif_node_ids,
        )
        if mapped_parent == mapped_child:
            continue
        if parent_occurrence is not None:
            boundary_index = boundary_counters[mapped_parent]
            boundary_counters[mapped_parent] += 1
            refs.append(
                MiningChildRef(
                    parent_id=mapped_parent,
                    child_id=mapped_child,
                    child_slot=f"boundary_{boundary_index}",
                    slot_index=boundary_index,
                )
            )
        else:
            refs.append(
                MiningChildRef(
                    parent_id=mapped_parent,
                    child_id=mapped_child,
                    child_slot=ref.child_slot,
                    slot_index=ref.slot_index,
                )
            )
    return refs


def _map_node_id(
    node_id: int,
    *,
    occurrence_by_internal_node: dict[int, MotifOccurrence],
    motif_node_ids: dict[int, int],
) -> int:
    occurrence = occurrence_by_internal_node.get(node_id)
    if occurrence is None:
        return node_id
    return motif_node_ids[occurrence.root_node_id]


def _incoming_by_child(child_refs: Iterable[MiningChildRef]) -> dict[int, list[MiningChildRef]]:
    incoming: dict[int, list[MiningChildRef]] = defaultdict(list)
    for ref in child_refs:
        incoming[ref.child_id].append(ref)
    return incoming
