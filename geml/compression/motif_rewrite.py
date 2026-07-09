"""Greedy motif replacement baseline with explicit expansion maps."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from geml.compression.motif_mining import (
    MiningChildRef,
    MiningGraph,
    MiningNode,
    MotifBoundaryOccurrenceRef,
    MotifOccurrence,
    enumerate_motif_occurrences,
    graph_structural_signature,
)
from geml.compression.motif_vocab import MotifBoundaryRefTemplate, MotifRecord, MotifVocabulary

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
    original_graph: MiningGraph | None = None
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
    reconstruction_valid: bool


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
    reconstruction_valid: bool


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
    reconstruction_valid = compressed_graph_expands_to_original(
        compressed,
        motifs_by_signature.values(),
    )
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
        expansion_valid=reconstruction_valid,
        reconstruction_valid=reconstruction_valid,
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
    compressed = build_motif_compressed_graph(
        graph,
        selected,
        motifs_by_signature=motifs_by_signature,
    )
    covered_node_count = sum(len(occurrence.internal_node_ids) for occurrence in selected)
    reconstruction_valid = compressed_graph_expands_to_original(
        compressed,
        motifs_by_signature.values(),
    )
    return MotifCompressionSummary(
        original_node_count=len(graph.nodes),
        original_child_ref_count=len(graph.child_refs),
        compressed_node_count=len(compressed.nodes),
        compressed_child_ref_count=len(compressed.child_refs),
        selected_replacement_count=len(selected),
        covered_node_count=covered_node_count,
        motif_coverage_percent=100.0 * covered_node_count / len(graph.nodes)
        if graph.nodes
        else 0.0,
        expansion_valid=reconstruction_valid,
        reconstruction_valid=reconstruction_valid,
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
    compressed = build_motif_compressed_graph(
        graph,
        selected,
        motifs_by_signature=motifs_by_signature,
    )
    covered_node_count = sum(len(occurrence.internal_node_ids) for occurrence in selected)
    reconstruction_valid = compressed_graph_expands_to_original(
        compressed,
        motifs_by_signature.values(),
    )
    return MotifCompressionSummary(
        original_node_count=len(graph.nodes),
        original_child_ref_count=len(graph.child_refs),
        compressed_node_count=len(compressed.nodes),
        compressed_child_ref_count=len(compressed.child_refs),
        selected_replacement_count=len(selected),
        covered_node_count=covered_node_count,
        motif_coverage_percent=100.0 * covered_node_count / len(graph.nodes)
        if graph.nodes
        else 0.0,
        expansion_valid=reconstruction_valid,
        reconstruction_valid=reconstruction_valid,
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
        local_node_bindings = _local_node_bindings(graph, occurrence)
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
                    "local_node_bindings": [
                        {"local_id": local_id, "original_node_id": original_node_id}
                        for local_id, original_node_id in sorted(local_node_bindings.items())
                    ],
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
    compressed = MotifCompressedGraph(
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
    reconstruction_valid = compressed_graph_expands_to_original(
        compressed,
        motifs_by_signature.values(),
    )
    return compressed.model_copy(
        update={
            "metadata": {
                **compressed.metadata,
                "reconstruction_valid": reconstruction_valid,
            }
        }
    )


def reconstruct_from_motif_graph(
    compressed_graph: MotifCompressedGraph,
    motif_vocab: MotifVocabulary | Sequence[MotifRecord] | Mapping[str, MotifRecord],
) -> MiningGraph:
    """Reconstruct the original graph from compressed nodes and motif templates."""
    motifs_by_id = _motifs_by_id(motif_vocab)
    replacements_by_node = {
        replacement.motif_node_id: replacement
        for replacement in compressed_graph.motif_replacements
    }
    compressed_node_ids = {node.id for node in compressed_graph.nodes}
    for replacement in compressed_graph.motif_replacements:
        if replacement.motif_node_id not in compressed_node_ids:
            raise ValueError(
                f"replacement {replacement.replacement_id} motif node is missing from graph"
            )
        if not replacement.expansion_map_to_original_graph:
            raise ValueError(f"replacement {replacement.replacement_id} has no expansion map")
        if replacement.motif_id not in motifs_by_id:
            raise ValueError(f"replacement {replacement.replacement_id} motif is missing")

    retained_nodes = [
        node for node in compressed_graph.nodes if node.id not in replacements_by_node
    ]
    reconstructed_nodes: list[MiningNode] = list(retained_nodes)
    reconstructed_refs: list[MiningChildRef] = []

    expanded_roots_by_motif_node = {
        replacement.motif_node_id: replacement.root_node_id
        for replacement in compressed_graph.motif_replacements
    }
    original_to_compressed_node = {
        original_node_id: replacement.motif_node_id
        for replacement in compressed_graph.motif_replacements
        for original_node_id in replacement.internal_node_ids
    }

    for ref in compressed_graph.child_refs:
        parent_replacement = replacements_by_node.get(ref.parent_id)
        if parent_replacement is not None:
            continue
        child_id = expanded_roots_by_motif_node.get(ref.child_id, ref.child_id)
        reconstructed_refs.append(
            MiningChildRef(
                parent_id=ref.parent_id,
                child_id=child_id,
                child_slot=ref.child_slot,
                slot_index=ref.slot_index,
            )
        )

    for replacement in compressed_graph.motif_replacements:
        motif = motifs_by_id[replacement.motif_id]
        local_to_original = _replacement_local_node_bindings(replacement)
        if set(local_to_original) != {node.local_id for node in motif.internal_nodes}:
            raise ValueError(
                f"replacement {replacement.replacement_id} local node bindings do not "
                "match motif template nodes"
            )
        boundary_occurrences = _replacement_boundary_refs(replacement)
        boundary_templates = _ordered_boundary_templates(motif)
        if len(boundary_occurrences) != len(boundary_templates):
            raise ValueError(
                f"replacement {replacement.replacement_id} boundary count does not match motif"
            )
        if tuple(ref.external_child_id for ref in boundary_occurrences) != (
            replacement.boundary_external_child_ids
        ):
            raise ValueError(
                f"replacement {replacement.replacement_id} boundary external ids do not "
                "match expansion map"
            )
        _validate_compressed_boundary_edges(
            compressed_graph,
            replacement,
            boundary_occurrences,
            original_to_compressed_node=original_to_compressed_node,
        )

        for template in sorted(motif.internal_nodes, key=lambda node: node.local_id):
            reconstructed_nodes.append(
                MiningNode(
                    id=local_to_original[template.local_id],
                    label=template.label,
                    kind=template.kind,
                    metadata=dict(template.metadata),
                )
            )
        for template in sorted(
            motif.internal_child_refs,
            key=lambda ref: (
                ref.parent_local_id,
                ref.slot_index,
                ref.child_slot,
                ref.child_local_id,
            ),
        ):
            reconstructed_refs.append(
                MiningChildRef(
                    parent_id=local_to_original[template.parent_local_id],
                    child_id=local_to_original[template.child_local_id],
                    child_slot=template.child_slot,
                    slot_index=template.slot_index,
                )
            )
        for template, occurrence in zip(
            boundary_templates,
            boundary_occurrences,
            strict=True,
        ):
            _validate_boundary_template_match(replacement, template, occurrence)
            reconstructed_refs.append(
                MiningChildRef(
                    parent_id=local_to_original[template.parent_local_id],
                    child_id=occurrence.external_child_id,
                    child_slot=template.child_slot,
                    slot_index=template.slot_index,
                )
            )

    reconstructed_root_id = expanded_roots_by_motif_node.get(
        compressed_graph.root_id,
        compressed_graph.root_id,
    )
    return MiningGraph(
        graph_id=f"{compressed_graph.source_graph_type}:reconstructed",
        graph_type=compressed_graph.source_graph_type,  # type: ignore[arg-type]
        nodes=reconstructed_nodes,
        child_refs=reconstructed_refs,
        root_id=reconstructed_root_id,
        metadata={
            "source_graph_type": compressed_graph.source_graph_type,
            "reconstructed_from_motif_graph": True,
            "used_original_graph": False,
        },
    )


def expand_motif_compressed_graph(
    compressed_graph: MotifCompressedGraph,
    motif_vocab: MotifVocabulary | Sequence[MotifRecord] | Mapping[str, MotifRecord],
) -> MiningGraph:
    """Expand a motif-compressed graph through its motif vocabulary templates."""
    return reconstruct_from_motif_graph(compressed_graph, motif_vocab)


def compressed_graph_expands_to_original(
    compressed_graph: MotifCompressedGraph,
    motif_vocab: MotifVocabulary | Sequence[MotifRecord] | Mapping[str, MotifRecord],
) -> bool:
    """Return whether expansion reconstructs the original graph signature."""
    try:
        reconstructed = reconstruct_from_motif_graph(compressed_graph, motif_vocab)
    except ValueError:
        return False
    return graph_structural_signature(reconstructed) == compressed_graph.original_graph_signature


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
            boundary_ref = _boundary_occurrence_for_graph_ref(graph, parent_occurrence, ref)
            boundary_index = boundary_ref.boundary_slot_index
            boundary_counters[mapped_parent] += 1
            refs.append(
                MiningChildRef(
                    parent_id=mapped_parent,
                    child_id=mapped_child,
                    child_slot=boundary_ref.boundary_slot,
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
    expected_boundary_counts: dict[int, int] = defaultdict(int)
    for occurrence in occurrence_by_internal_node.values():
        expected_boundary_counts[motif_node_ids[occurrence.root_node_id]] = len(
            occurrence.boundary_occurrence_refs
        )
    for motif_node_id, expected_count in expected_boundary_counts.items():
        if boundary_counters[motif_node_id] != expected_count:
            raise ValueError(
                f"motif node {motif_node_id} has {boundary_counters[motif_node_id]} "
                f"boundary refs, expected {expected_count}"
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


def _motifs_by_id(
    motif_vocab: MotifVocabulary | Sequence[MotifRecord] | Mapping[str, MotifRecord],
) -> dict[str, MotifRecord]:
    if isinstance(motif_vocab, MotifVocabulary):
        return {motif.motif_id: motif for motif in motif_vocab.motifs}
    if isinstance(motif_vocab, Mapping):
        return dict(motif_vocab)
    return {motif.motif_id: motif for motif in motif_vocab}


def _local_node_bindings(graph: MiningGraph, occurrence: MotifOccurrence) -> dict[int, int]:
    children_by_parent = _children_by_parent(graph.child_refs)
    local_by_original = _canonical_local_ids(
        occurrence.root_node_id,
        frozenset(occurrence.internal_node_ids),
        children_by_parent,
    )
    return {local_id: original_id for original_id, local_id in local_by_original.items()}


def _replacement_local_node_bindings(replacement: MotifReplacement) -> dict[int, int]:
    raw_bindings = replacement.expansion_map_to_original_graph.get("local_node_bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise ValueError(f"replacement {replacement.replacement_id} is missing local node bindings")
    bindings: dict[int, int] = {}
    for raw in raw_bindings:
        if not isinstance(raw, dict):
            raise ValueError(
                f"replacement {replacement.replacement_id} has invalid local node binding"
            )
        local_id = raw.get("local_id")
        original_node_id = raw.get("original_node_id")
        if not isinstance(local_id, int) or not isinstance(original_node_id, int):
            raise ValueError(
                f"replacement {replacement.replacement_id} has invalid local node binding ids"
            )
        bindings[local_id] = original_node_id
    if set(bindings.values()) != set(replacement.internal_node_ids):
        raise ValueError(
            f"replacement {replacement.replacement_id} local node bindings do not match "
            "internal node ids"
        )
    return bindings


def _replacement_boundary_refs(
    replacement: MotifReplacement,
) -> tuple[MotifBoundaryOccurrenceRef, ...]:
    raw_refs = replacement.expansion_map_to_original_graph.get("boundary_refs")
    if raw_refs is None:
        raise ValueError(f"replacement {replacement.replacement_id} is missing boundary refs")
    if not isinstance(raw_refs, list):
        raise ValueError(f"replacement {replacement.replacement_id} boundary refs are invalid")
    refs = tuple(MotifBoundaryOccurrenceRef.model_validate(raw) for raw in raw_refs)
    return tuple(sorted(refs, key=_boundary_occurrence_sort_key))


def _ordered_boundary_templates(motif: MotifRecord) -> tuple[MotifBoundaryRefTemplate, ...]:
    return tuple(
        sorted(
            motif.boundary_child_refs,
            key=lambda ref: (
                ref.boundary_slot_index,
                ref.boundary_slot,
                ref.parent_local_id,
                ref.slot_index,
                ref.child_slot,
            ),
        )
    )


def _validate_boundary_template_match(
    replacement: MotifReplacement,
    template: MotifBoundaryRefTemplate,
    occurrence: MotifBoundaryOccurrenceRef,
) -> None:
    if (
        template.parent_local_id != occurrence.parent_local_id
        or template.boundary_slot_index != occurrence.boundary_slot_index
        or template.boundary_slot != occurrence.boundary_slot
        or template.child_slot != occurrence.child_slot
        or template.slot_index != occurrence.slot_index
    ):
        raise ValueError(
            f"replacement {replacement.replacement_id} boundary order does not match motif template"
        )


def _validate_compressed_boundary_edges(
    compressed_graph: MotifCompressedGraph,
    replacement: MotifReplacement,
    boundary_occurrences: Sequence[MotifBoundaryOccurrenceRef],
    *,
    original_to_compressed_node: dict[int, int],
) -> None:
    refs_by_boundary_slot = {
        (ref.slot_index, ref.child_slot): ref
        for ref in compressed_graph.child_refs
        if ref.parent_id == replacement.motif_node_id
    }
    if len(refs_by_boundary_slot) != len(boundary_occurrences):
        raise ValueError(
            f"replacement {replacement.replacement_id} compressed boundary edge count "
            "does not match expansion map"
        )
    for occurrence in boundary_occurrences:
        compressed_ref = refs_by_boundary_slot.get(
            (occurrence.boundary_slot_index, occurrence.boundary_slot)
        )
        if compressed_ref is None:
            raise ValueError(
                f"replacement {replacement.replacement_id} is missing compressed boundary "
                f"edge {occurrence.boundary_slot}"
            )
        expected_child_id = original_to_compressed_node.get(
            occurrence.external_child_id,
            occurrence.external_child_id,
        )
        if compressed_ref.child_id != expected_child_id:
            raise ValueError(
                f"replacement {replacement.replacement_id} compressed boundary child "
                "does not match expansion map"
            )


def _boundary_occurrence_for_graph_ref(
    graph: MiningGraph,
    occurrence: MotifOccurrence,
    ref: MiningChildRef,
) -> MotifBoundaryOccurrenceRef:
    local_by_original = {
        original_id: local_id
        for local_id, original_id in _local_node_bindings(graph, occurrence).items()
    }
    parent_local_id = local_by_original[ref.parent_id]
    matches = [
        boundary_ref
        for boundary_ref in occurrence.boundary_occurrence_refs
        if boundary_ref.parent_local_id == parent_local_id
        and boundary_ref.external_child_id == ref.child_id
        and boundary_ref.child_slot == ref.child_slot
        and boundary_ref.slot_index == ref.slot_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"could not identify boundary ref for occurrence rooted at {occurrence.root_node_id}"
        )
    return matches[0]


def _children_by_parent(child_refs: Iterable[MiningChildRef]) -> dict[int, list[MiningChildRef]]:
    grouped: dict[int, list[MiningChildRef]] = defaultdict(list)
    for ref in child_refs:
        grouped[ref.parent_id].append(ref)
    return {
        parent_id: sorted(refs, key=lambda ref: (ref.slot_index, ref.child_slot, ref.child_id))
        for parent_id, refs in grouped.items()
    }


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


def _boundary_occurrence_sort_key(
    ref: MotifBoundaryOccurrenceRef,
) -> tuple[int, str, int, int, str, int]:
    return (
        ref.boundary_slot_index,
        ref.boundary_slot,
        ref.parent_local_id,
        ref.slot_index,
        ref.child_slot,
        ref.external_child_id,
    )
