"""Tests for the Goal 3.3 DAG compression study pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.dag_compression_study import (
    DagCompressionStudyConfig,
    run_dag_compression_study,
)


def test_dag_compression_study_small_count_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "v0"
    config = DagCompressionStudyConfig(
        seed=13,
        count=25,
        max_depth=2,
        output_dir=output_dir,
        input_jsonl_path=output_dir / "dag_compression_inputs.jsonl",
        metrics_jsonl_path=output_dir / "dag_compression_metrics.jsonl",
        metrics_csv_path=output_dir / "dag_compression_metrics.csv",
        summary_json_path=output_dir / "dag_compression_summary.json",
    )

    rows = run_dag_compression_study(config)

    assert len(rows) == 25
    assert config.input_jsonl_path.exists()
    assert config.metrics_jsonl_path.exists()
    assert config.metrics_csv_path.exists()
    assert config.summary_json_path.exists()

    jsonl_rows = [
        json.loads(line)
        for line in config.metrics_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    with config.metrics_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    summary = json.loads(config.summary_json_path.read_text(encoding="utf-8"))

    assert len(jsonl_rows) == 25
    assert len(csv_rows) == 25
    assert summary["processed_count"] == 25
    assert summary["supported_count"] == 25
    assert summary["unsupported_count"] == 0
    assert summary["mean_tree_alpha"] is not None
    assert summary["mean_dag_alpha_vs_ast_tree"] is not None
    assert summary["mean_dag_alpha_vs_ast_dag"] is not None
    assert summary["mean_eml_dag_compression"] is not None
    assert summary["percent_below_threshold_tree_alpha"] is not None
    assert summary["percent_below_threshold_dag_alpha_vs_ast_tree"] is not None
    assert summary["percent_below_threshold_dag_alpha_vs_ast_dag"] is not None

    required_fields = {
        "index",
        "expression",
        "srepr",
        "source_serialization",
        "ast_tree_node_count",
        "ast_dag_node_count",
        "ast_dag_child_ref_count",
        "ast_tree_depth",
        "ast_dag_depth",
        "ast_dag_compression",
        "eml_tree_node_count",
        "eml_dag_node_count",
        "eml_dag_child_ref_count",
        "eml_tree_depth",
        "eml_dag_depth",
        "eml_dag_compression",
        "tree_alpha",
        "dag_alpha_vs_ast_tree",
        "dag_alpha_vs_ast_dag",
        "alpha_threshold_current",
        "below_threshold_tree",
        "below_threshold_dag_vs_ast_tree",
        "below_threshold_dag_vs_ast_dag",
        "pure_eml_valid",
        "derived_leaf_count",
        "hidden_compound_leaf_count",
        "error",
    }
    assert required_fields <= set(jsonl_rows[0])
    assert required_fields <= set(csv_rows[0])

    for row in jsonl_rows:
        assert row["source_serialization"] == "srepr"
        assert row["pure_eml_valid"] is True
        assert row["derived_leaf_count"] == 0
        assert row["hidden_compound_leaf_count"] == 0
        assert row["error"] is None
        assert row["eml_dag_node_count"] <= row["eml_tree_node_count"]
        assert row["ast_dag_node_count"] <= row["ast_tree_node_count"]
