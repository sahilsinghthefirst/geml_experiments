from __future__ import annotations

import sympy as sp
from geml.compression.learned_motifs import (
    build_learned_motif_vocabulary,
    select_random_motif_ids,
)
from geml.compression.macro_graph import build_macro_graph
from geml.compression.motif_dataset import SplitConfig, assign_split
from geml.compression.motif_mining import mine_frequent_motifs, mining_graph_from_macro_graph
from geml.compression.motif_rewrite import (
    compressed_graph_expands_to_original,
    greedy_motif_compress_graph,
)
from geml.compression.motif_selection_model import (
    MotifScoringWeights,
    MotifSplitFeature,
    optimize_motif_selection,
    score_motif,
)


def test_train_validation_test_split_is_deterministic() -> None:
    config = SplitConfig(seed=13)

    first = [assign_split(index, config) for index in range(100)]
    second = [assign_split(index, config) for index in range(100)]

    assert first == second
    assert set(first) == {"train", "validation", "test"}


def test_learned_motif_vocabulary_is_reproducible_with_fixed_seed() -> None:
    features = synthetic_features()
    weights = (
        MotifScoringWeights(),
        MotifScoringWeights(coverage_bonus=0.01, nontrivial_coverage_bonus=0.02),
    )

    first = optimize_motif_selection(features, vocab_sizes=(2, 3), weight_grid=weights)
    second = optimize_motif_selection(features, vocab_sizes=(2, 3), weight_grid=weights)

    assert first.selected_motif_ids == second.selected_motif_ids
    assert first.selected_weights == second.selected_weights
    assert first.trained_final_reasoning_gnn is False


def test_compressed_graphs_reconstruct_exactly_for_learned_motifs() -> None:
    x, y = sp.symbols("x y")
    graph = mining_graph_from_macro_graph(
        build_macro_graph(sp.Add(x, y, evaluate=False)),
        graph_id="macro:0",
        expression_index=0,
    )
    motifs = mine_frequent_motifs(
        [graph],
        min_motif_nodes=1,
        max_motif_nodes=2,
        min_support=1,
        max_vocab_size=10,
        subset_labels_by_graph_id={"macro:0": "nontrivial_v1"},
        expression_indices_by_graph_id={"macro:0": 0},
    )

    result = greedy_motif_compress_graph(
        graph,
        motifs[:1],
        min_motif_nodes=1,
        max_motif_nodes=2,
        expression_index=0,
        subset_label="nontrivial_v1",
    )

    assert result.expansion_valid is True
    assert compressed_graph_expands_to_original(result.compressed_graph) is True


def test_learned_motif_nodes_store_expansion_maps() -> None:
    x, y = sp.symbols("x y")
    graph = mining_graph_from_macro_graph(
        build_macro_graph(sp.Mul(x, y, evaluate=False)),
        graph_id="macro:0",
        expression_index=0,
    )
    motifs = mine_frequent_motifs(
        [graph],
        min_motif_nodes=1,
        max_motif_nodes=2,
        min_support=1,
        max_vocab_size=10,
        subset_labels_by_graph_id={"macro:0": "nontrivial_v1"},
        expression_indices_by_graph_id={"macro:0": 0},
    )
    feature = MotifSplitFeature(
        motif_id=motifs[0].motif_id,
        motif_type=motifs[0].motif_type,
        node_count=motifs[0].node_count,
        edge_count=motifs[0].edge_count,
        support_count=1,
        train_support=1,
        validation_support=0,
        test_support=0,
        node_savings=1,
        train_node_savings=1,
        validation_node_savings=0,
        test_node_savings=0,
        coverage=motifs[0].node_count,
        train_coverage=motifs[0].node_count,
        validation_coverage=0,
        test_coverage=0,
        nontrivial_coverage=motifs[0].node_count,
        train_nontrivial_coverage=motifs[0].node_count,
        validation_nontrivial_coverage=0,
        test_nontrivial_coverage=0,
        expansion_complexity=motifs[0].edge_count,
    )
    learned = build_learned_motif_vocabulary(
        selected_motif_ids=[motifs[0].motif_id],
        selected_scores={
            motifs[0].motif_id: score_motif(feature, MotifScoringWeights(), split="train")
        },
        selected_weights=MotifScoringWeights().to_json_dict(),
        candidate_features_by_id={feature.motif_id: feature},
        candidate_motifs_by_id={motifs[0].motif_id: motifs[0]},
        random_baseline_motif_ids=[motifs[0].motif_id],
    )

    assert learned.motifs[0].expansion_map
    assert learned.motifs[0].expansion_valid is True
    assert learned.motifs[0].metadata["is_pure_eml"] is False


def test_random_baseline_uses_same_vocab_size() -> None:
    candidates = [f"motif_{index}" for index in range(10)]

    selected = select_random_motif_ids(candidates, vocab_size=4, seed=0)

    assert len(selected) == 4
    assert selected == select_random_motif_ids(candidates, vocab_size=4, seed=0)


def test_no_final_reasoning_gnn_is_trained() -> None:
    result = optimize_motif_selection(
        synthetic_features(),
        vocab_sizes=(2,),
        weight_grid=(MotifScoringWeights(),),
    )

    assert result.model_type == "deterministic_linear_motif_scorer"
    assert result.trained_final_reasoning_gnn is False


def synthetic_features() -> tuple[MotifSplitFeature, ...]:
    return tuple(
        MotifSplitFeature(
            motif_id=f"motif_{index}",
            motif_type="macro_graph",
            node_count=2,
            edge_count=2,
            support_count=10 + index,
            train_support=7 + index,
            validation_support=2 + index,
            test_support=1,
            node_savings=10 + index,
            train_node_savings=7 + index,
            validation_node_savings=2 + index,
            test_node_savings=1,
            coverage=20 + index,
            train_coverage=14 + index,
            validation_coverage=4 + index,
            test_coverage=2,
            nontrivial_coverage=10 + index,
            train_nontrivial_coverage=7 + index,
            validation_nontrivial_coverage=2 + index,
            test_nontrivial_coverage=1,
            expansion_complexity=2,
        )
        for index in range(5)
    )
