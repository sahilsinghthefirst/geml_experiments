from __future__ import annotations

import pytest
import sympy as sp
from geml.compression.macro_graph import build_macro_graph
from geml.compression.motif_mining import (
    build_motif_vocabulary,
    enumerate_motif_occurrences,
    graph_structural_signature,
    mine_frequent_motifs,
    mining_graph_from_macro_graph,
)
from geml.compression.motif_rewrite import (
    MotifCompressedGraph,
    build_motif_compressed_graph,
    compressed_graph_expands_to_original,
    expand_motif_compressed_graph,
    greedy_motif_compress_graph,
    reconstruct_from_motif_graph,
    validate_non_overlapping_occurrences,
)
from geml.compression.motif_vocab import MotifRecord


def test_motif_occurrence_detection_preserves_child_slots() -> None:
    x, y = sp.symbols("x y")
    graph = mining_graph_from_macro_graph(
        build_macro_graph(sp.Add(x, y, evaluate=False)),
        graph_id="macro:0",
        expression_index=0,
    )

    occurrences = enumerate_motif_occurrences(
        graph,
        min_motif_nodes=1,
        max_motif_nodes=1,
        expression_index=0,
        subset_label="nontrivial_v1",
    )
    add_occurrence = next(
        occurrence for occurrence in occurrences if occurrence.internal_nodes[0].label == "eml_add"
    )

    assert [ref.child_slot for ref in add_occurrence.boundary_child_refs] == ["left", "right"]
    assert [ref.slot_index for ref in add_occurrence.boundary_child_refs] == [0, 1]


def test_repeated_child_references_are_not_collapsed_in_boundary_maps() -> None:
    x = sp.Symbol("x")
    graph = mining_graph_from_macro_graph(
        build_macro_graph(sp.Add(x, x, evaluate=False)),
        graph_id="macro:0",
        expression_index=0,
    )

    occurrences = enumerate_motif_occurrences(
        graph,
        min_motif_nodes=1,
        max_motif_nodes=1,
        expression_index=0,
        subset_label="nontrivial_v1",
    )
    add_occurrence = next(
        occurrence for occurrence in occurrences if occurrence.internal_nodes[0].label == "eml_add"
    )

    assert len(add_occurrence.boundary_occurrence_refs) == 2
    assert (
        add_occurrence.boundary_occurrence_refs[0].external_child_id
        == add_occurrence.boundary_occurrence_refs[1].external_child_id
    )


def test_motif_replacement_preserves_expandability() -> None:
    x, y = sp.symbols("x y")
    expr = sp.Add(
        sp.Add(x, y, evaluate=False),
        sp.Add(x, y, evaluate=False),
        evaluate=False,
    )
    graph = mining_graph_from_macro_graph(
        build_macro_graph(expr),
        graph_id="macro:0",
        expression_index=0,
    )
    records = mine_frequent_motifs(
        [graph],
        min_motif_nodes=1,
        max_motif_nodes=3,
        min_support=1,
        max_vocab_size=20,
        subset_labels_by_graph_id={"macro:0": "nontrivial_v1"},
        expression_indices_by_graph_id={"macro:0": 0},
    )
    vocab = build_motif_vocabulary(
        pure_records=[],
        macro_records=records,
        max_vocab_size=20,
        config={},
    )

    result = greedy_motif_compress_graph(
        graph,
        vocab.motifs_by_type("macro_graph"),
        min_motif_nodes=1,
        max_motif_nodes=3,
        expression_index=0,
        subset_label="nontrivial_v1",
    )

    assert result.expansion_valid is True
    macro_motifs = vocab.motifs_by_type("macro_graph")
    assert compressed_graph_expands_to_original(result.compressed_graph, macro_motifs) is True
    reconstructed = expand_motif_compressed_graph(result.compressed_graph, macro_motifs)
    assert graph_structural_signature(reconstructed) == graph_structural_signature(graph)


def test_simple_motif_reconstructs_exactly() -> None:
    x = sp.Symbol("x")
    graph, motifs, compressed = _single_occurrence_compression(
        sp.exp(x),
        node_label="eml_exp",
        node_count=2,
    )

    reconstructed = reconstruct_from_motif_graph(compressed, motifs)

    assert graph_structural_signature(reconstructed) == graph_structural_signature(graph)
    assert compressed_graph_expands_to_original(compressed, motifs) is True


def test_two_boundary_input_motif_reconstructs_exactly() -> None:
    x, y = sp.symbols("x y")
    graph, motifs, compressed = _single_occurrence_compression(
        sp.Add(x, y, evaluate=False),
        node_label="eml_add",
        node_count=1,
    )

    reconstructed = reconstruct_from_motif_graph(compressed, motifs)

    assert graph_structural_signature(reconstructed) == graph_structural_signature(graph)
    replacement = compressed.motif_replacements[0]
    boundary_refs = replacement.expansion_map_to_original_graph["boundary_refs"]
    assert isinstance(boundary_refs, list)
    assert [ref["boundary_slot"] for ref in boundary_refs] == ["boundary_0", "boundary_1"]


def test_repeated_child_ref_motif_reconstructs_exactly() -> None:
    x = sp.Symbol("x")
    graph, motifs, compressed = _single_occurrence_compression(
        sp.Add(x, x, evaluate=False),
        node_label="eml_add",
        node_count=1,
    )

    reconstructed = reconstruct_from_motif_graph(compressed, motifs)

    assert graph_structural_signature(reconstructed) == graph_structural_signature(graph)
    replacement = compressed.motif_replacements[0]
    assert len(set(replacement.boundary_external_child_ids)) == 1


def test_wrong_boundary_order_fails_reconstruction_validation() -> None:
    x, y = sp.symbols("x y")
    _graph, motifs, compressed = _single_occurrence_compression(
        sp.Add(x, y, evaluate=False),
        node_label="eml_add",
        node_count=1,
    )
    corrupted = _swap_first_two_boundary_external_ids(compressed)

    assert compressed_graph_expands_to_original(corrupted, motifs) is False


def test_missing_expansion_map_fails_reconstruction() -> None:
    x, y = sp.symbols("x y")
    _graph, motifs, compressed = _single_occurrence_compression(
        sp.Add(x, y, evaluate=False),
        node_label="eml_add",
        node_count=1,
    )
    replacement = compressed.motif_replacements[0].model_copy(
        update={"expansion_map_to_original_graph": {}}
    )
    corrupted = compressed.model_copy(update={"motif_replacements": [replacement]})

    with pytest.raises(ValueError, match="has no expansion map"):
        reconstruct_from_motif_graph(corrupted, motifs)


def test_reconstruction_does_not_validate_by_returning_original_graph() -> None:
    x, y = sp.symbols("x y")
    graph, motifs, compressed = _single_occurrence_compression(
        sp.Add(x, y, evaluate=False),
        node_label="eml_add",
        node_count=1,
    )
    corrupted = _swap_first_two_boundary_external_ids(compressed).model_copy(
        update={"original_graph": graph}
    )

    assert compressed_graph_expands_to_original(corrupted, motifs) is False


def test_no_motif_node_is_labeled_pure_eml() -> None:
    x, y = sp.symbols("x y")
    graph = mining_graph_from_macro_graph(
        build_macro_graph(sp.Mul(sp.Add(x, y, evaluate=False), y, evaluate=False)),
        graph_id="macro:0",
        expression_index=0,
    )
    records = mine_frequent_motifs(
        [graph],
        min_motif_nodes=1,
        max_motif_nodes=3,
        min_support=1,
        max_vocab_size=20,
        subset_labels_by_graph_id={"macro:0": "nontrivial_v1"},
        expression_indices_by_graph_id={"macro:0": 0},
    )
    result = greedy_motif_compress_graph(
        graph,
        records,
        min_motif_nodes=1,
        max_motif_nodes=3,
        expression_index=0,
        subset_label="nontrivial_v1",
    )

    motif_nodes = [node for node in result.compressed_graph.nodes if node.kind == "motif"]
    assert motif_nodes
    assert all(node.label.startswith("motif:") for node in motif_nodes)
    assert all(node.metadata["is_pure_eml"] is False for node in motif_nodes)


def test_overlapping_motif_replacements_are_rejected() -> None:
    x, y = sp.symbols("x y")
    graph = mining_graph_from_macro_graph(
        build_macro_graph(sp.Add(sp.Add(x, y, evaluate=False), y, evaluate=False)),
        graph_id="macro:0",
        expression_index=0,
    )
    occurrences = enumerate_motif_occurrences(
        graph,
        min_motif_nodes=1,
        max_motif_nodes=2,
        expression_index=0,
        subset_label="nontrivial_v1",
    )
    overlapping = [
        occurrence for occurrence in occurrences if graph.root_id in occurrence.internal_node_ids
    ][:2]

    assert len(overlapping) == 2
    with pytest.raises(ValueError, match="overlapping motif replacements"):
        validate_non_overlapping_occurrences(overlapping)
    with pytest.raises(ValueError, match="overlapping motif replacements"):
        build_motif_compressed_graph(
            graph,
            overlapping,
            motifs_by_signature={},
        )


def _single_occurrence_compression(
    expr: sp.Expr,
    *,
    node_label: str,
    node_count: int,
) -> tuple[object, tuple[MotifRecord, ...], MotifCompressedGraph]:
    graph = mining_graph_from_macro_graph(
        build_macro_graph(expr),
        graph_id="macro:0",
        expression_index=0,
    )
    occurrences = enumerate_motif_occurrences(
        graph,
        min_motif_nodes=1,
        max_motif_nodes=max(node_count, 1),
        expression_index=0,
        subset_label="nontrivial_v1",
    )
    occurrence = next(
        item
        for item in occurrences
        if item.internal_nodes[0].label == node_label and item.node_count == node_count
    )
    records = mine_frequent_motifs(
        [graph],
        min_motif_nodes=1,
        max_motif_nodes=max(node_count, 1),
        min_support=1,
        max_vocab_size=20,
        subset_labels_by_graph_id={"macro:0": "nontrivial_v1"},
        expression_indices_by_graph_id={"macro:0": 0},
    )
    motif = next(record for record in records if record.signature == occurrence.signature)
    compressed = build_motif_compressed_graph(
        graph,
        [occurrence],
        motifs_by_signature={occurrence.signature: motif},
    )
    return graph, (motif,), compressed


def _swap_first_two_boundary_external_ids(
    compressed: MotifCompressedGraph,
) -> MotifCompressedGraph:
    replacement = compressed.motif_replacements[0]
    raw_map = dict(replacement.expansion_map_to_original_graph)
    raw_refs = raw_map["boundary_refs"]
    if not isinstance(raw_refs, list) or len(raw_refs) < 2:
        raise AssertionError("test requires at least two boundary refs")
    boundary_refs = [dict(ref) for ref in raw_refs]
    boundary_refs[0]["external_child_id"], boundary_refs[1]["external_child_id"] = (
        boundary_refs[1]["external_child_id"],
        boundary_refs[0]["external_child_id"],
    )
    raw_map["boundary_refs"] = boundary_refs
    corrupted_replacement = replacement.model_copy(
        update={
            "boundary_external_child_ids": tuple(ref["external_child_id"] for ref in boundary_refs),
            "expansion_map_to_original_graph": raw_map,
        }
    )
    return compressed.model_copy(update={"motif_replacements": [corrupted_replacement]})
