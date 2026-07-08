"""Tests for Goal 5.5 hierarchical graph exports."""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp
from geml.compression.graph_export import (
    graph_can_reconstruct_pure_eml_dag,
    graph_record_from_dag,
)
from geml.compression.graph_schema import GraphExportRecord, graph_schema_document
from geml.experiments.goal5_hierarchical_export import (
    HierarchicalGraphExportConfig,
    load_config,
    run_goal5_hierarchical_export,
)
from geml.symbolic.ast_graph import sympy_to_ast_tree
from geml.symbolic.dag_graph import tree_to_dag


def test_hierarchical_export_small_end_to_end_schema_validates(tmp_path: Path) -> None:
    config = _small_config(tmp_path, count=5)

    result = run_goal5_hierarchical_export(config)

    for path in result.output_paths:
        assert path.exists()
    records = _read_graph_records(config.graphs_jsonl_path)
    assert records
    assert result.summary["graph_count"] == len(records)
    assert set(result.summary["representation_modes_exported"]) == {
        "ast_tree_graph",
        "ast_dag_graph",
        "pure_eml_dag_graph",
        "egraph_safe_eml_dag_graph",
        "egraph_positive_real_eml_dag_graph",
        "macro_graph",
        "frequent_motif_graph",
        "learned_motif_graph",
        "hierarchical_eml_graph",
    }
    assert all(record.validation.schema_valid for record in records)


def test_all_graph_ids_are_unique(tmp_path: Path) -> None:
    config = _small_config(tmp_path, count=4)
    run_goal5_hierarchical_export(config)

    records = _read_graph_records(config.graphs_jsonl_path)
    graph_ids = [record.graph_id for record in records]

    assert len(graph_ids) == len(set(graph_ids))


def test_every_compressed_node_has_expansion_mapping(tmp_path: Path) -> None:
    config = _small_config(tmp_path, count=4)
    run_goal5_hierarchical_export(config)

    records = _read_graph_records(config.graphs_jsonl_path)
    compressed_nodes = [
        node
        for record in records
        for node in record.nodes
        if node.node_type in {"macro", "motif", "learned_motif"}
    ]

    assert compressed_nodes
    assert all(node.expansion_available for node in compressed_nodes)
    assert all(node.expansion_target_ids for node in compressed_nodes)


def test_child_slots_and_duplicate_references_are_preserved() -> None:
    x = sp.Symbol("x")
    ast_tree = sympy_to_ast_tree(sp.Add(x, x, evaluate=False))
    ast_dag = tree_to_dag(ast_tree)

    record = graph_record_from_dag(
        ast_dag,
        graph_id="test:ast_dag",
        representation_mode="ast_dag_graph",
        source_expression_id=0,
        subset_label="nontrivial_v1",
        split="train",
        source_label="test_ast_dag",
    )

    root_edges = [edge for edge in record.edges if edge.source_id == record.root_node_id]
    assert [edge.child_slot for edge in sorted(root_edges, key=lambda item: item.slot_index)] == [
        "left",
        "right",
    ]
    assert len({edge.target_id for edge in root_edges}) == 1


def test_graph_can_reconstruct_pure_eml_dag(tmp_path: Path) -> None:
    config = _small_config(tmp_path, count=3)
    run_goal5_hierarchical_export(config)

    records = _read_graph_records(config.graphs_jsonl_path)

    assert records
    assert all(graph_can_reconstruct_pure_eml_dag(record) for record in records)


def test_no_gnn_training_occurs(tmp_path: Path) -> None:
    config = _small_config(tmp_path, count=3)
    result = run_goal5_hierarchical_export(config)

    schema = graph_schema_document()
    assert schema["integrity_contract"]["contains_gnn_training"] is False
    assert result.summary["integrity"]["trained_final_symbolic_reasoning_gnn"] is False
    assert result.summary["integrity"]["hidden_target_labels_in_graph_records"] is False


def test_hierarchical_export_config_loads_v1_yaml() -> None:
    config = load_config(Path("configs/hierarchical_graph_export_v1.yaml"))

    assert config.count == 10_000
    assert config.graphs_jsonl_path.as_posix() == "outputs/v1/goal5_hierarchical_graphs.jsonl"
    assert config.schema_json_path.as_posix() == "outputs/v1/goal5_graph_schema.json"


def _small_config(tmp_path: Path, *, count: int) -> HierarchicalGraphExportConfig:
    output_dir = tmp_path / "outputs" / "v1"
    return HierarchicalGraphExportConfig(
        count=count,
        graphs_jsonl_path=output_dir / "goal5_hierarchical_graphs.jsonl",
        splits_json_path=output_dir / "goal5_graph_splits.json",
        schema_json_path=output_dir / "goal5_graph_schema.json",
        summary_json_path=output_dir / "goal5_hierarchical_export_summary.json",
    )


def _read_graph_records(path: Path) -> list[GraphExportRecord]:
    records: list[GraphExportRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(GraphExportRecord.model_validate(json.loads(line)))
    return records
