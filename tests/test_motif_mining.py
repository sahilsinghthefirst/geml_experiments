from __future__ import annotations

import pytest
import sympy as sp
from geml.compression.macro_graph import build_macro_graph
from geml.compression.motif_mining import (
    build_motif_vocabulary,
    enumerate_motif_occurrences,
    mine_frequent_motifs,
    mining_graph_from_macro_graph,
)
from geml.compression.motif_rewrite import (
    build_motif_compressed_graph,
    compressed_graph_expands_to_original,
    expand_motif_compressed_graph,
    greedy_motif_compress_graph,
    validate_non_overlapping_occurrences,
)


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
    assert compressed_graph_expands_to_original(result.compressed_graph) is True
    assert expand_motif_compressed_graph(result.compressed_graph) == graph


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
