"""Tests for the integrated Goal 5 compression pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.run_goal5_compression_pipeline import (
    Goal5CompressionPipelineConfig,
    load_config,
    load_goal5_artifacts,
    run_goal5_compression_pipeline,
)


def test_goal5_compression_pipeline_small_end_to_end(tmp_path: Path) -> None:
    """A small artifact-only run writes comparison, summary, and docs."""
    config = _write_small_artifacts(tmp_path)

    result = run_goal5_compression_pipeline(config)

    assert len(result.comparison_rows) == 8
    assert config.comparison_csv_path.exists()
    assert config.summary_json_path.exists()
    assert config.final_report_path.exists()
    assert config.summary_doc_path.exists()
    assert config.findings_report_path.exists()

    rows = list(csv.DictReader(config.comparison_csv_path.open(encoding="utf-8")))
    assert {row["mode"] for row in rows} >= {
        "Goal 5 macro graph",
        "Goal 5 frequent motif graph",
        "Goal 5 learned motif graph",
        "Goal 5 neural e-graph extractor",
        "Goal 5 hierarchical graph",
    }

    final_report = config.final_report_path.read_text(encoding="utf-8")
    assert "Goal 5 asks whether" in final_report
    assert "Final Recommendation for Goal 6" in final_report
    assert "does not train final symbolic-reasoning GNNs" in final_report


def test_goal5_compression_pipeline_loads_prior_goal5_artifacts(tmp_path: Path) -> None:
    """All report-sized prior Goal 5 artifacts are loaded and validated."""
    config = _write_small_artifacts(tmp_path)

    artifacts = load_goal5_artifacts(config)

    assert set(artifacts.summaries) == {
        "goal3",
        "goal4",
        "macro_graph",
        "frequent_motifs",
        "learned_motifs",
        "neural_egraph",
        "hierarchical_export",
    }
    assert set(artifacts.json_artifacts) == {
        "frequent_motif_vocab",
        "learned_motif_vocab",
        "learned_motif_train_log",
        "neural_egraph_train_log",
        "hierarchical_schema",
        "hierarchical_splits",
    }
    assert "macro_graph_metrics" in artifacts.csv_headers


def test_goal5_compression_pipeline_integrity_flags(tmp_path: Path) -> None:
    """The integrated summary keeps Goal 5 integrity boundaries explicit."""
    config = _write_small_artifacts(tmp_path)

    result = run_goal5_compression_pipeline(config)
    integrity = result.summary["integrity"]

    assert result.summary["reconstruction_failure_count"] == 0
    assert integrity["no_missing_expansion_maps"] is True
    assert integrity["no_hidden_pure_eml_violations"] is True
    assert integrity["trained_final_symbolic_reasoning_gnn"] is False
    assert integrity["compressed_graph_metrics_are_pure_eml_alpha"] is False
    assert integrity["safe_and_positive_real_modes_separately_labeled"] is True


def test_goal5_compression_pipeline_config_loads_yaml(tmp_path: Path) -> None:
    """The v1-style integration config can be loaded from YAML."""
    config = _write_small_artifacts(tmp_path)
    config_path = tmp_path / "goal5_compression.yaml"
    config_path.write_text(
        "\n".join(
            [
                "count: 2",
                "reuse_existing_artifacts: true",
                "run_missing_artifacts: false",
                f"goal3_summary_json_path: {config.goal3_summary_json_path}",
                f"goal3_metrics_csv_path: {config.goal3_metrics_csv_path}",
                f"goal4_summary_json_path: {config.goal4_summary_json_path}",
                f"goal4_safe_metrics_csv_path: {config.goal4_safe_metrics_csv_path}",
                "goal4_positive_real_metrics_csv_path: "
                f"{config.goal4_positive_real_metrics_csv_path}",
                f"macro_summary_json_path: {config.macro_summary_json_path}",
                f"macro_metrics_csv_path: {config.macro_metrics_csv_path}",
                f"frequent_motif_summary_json_path: {config.frequent_motif_summary_json_path}",
                f"frequent_motif_metrics_csv_path: {config.frequent_motif_metrics_csv_path}",
                f"frequent_motif_vocab_json_path: {config.frequent_motif_vocab_json_path}",
                f"learned_motif_summary_json_path: {config.learned_motif_summary_json_path}",
                f"learned_motif_metrics_csv_path: {config.learned_motif_metrics_csv_path}",
                f"learned_motif_vocab_json_path: {config.learned_motif_vocab_json_path}",
                f"learned_motif_train_log_json_path: {config.learned_motif_train_log_json_path}",
                f"neural_egraph_summary_json_path: {config.neural_egraph_summary_json_path}",
                f"neural_egraph_metrics_csv_path: {config.neural_egraph_metrics_csv_path}",
                f"neural_egraph_train_log_json_path: {config.neural_egraph_train_log_json_path}",
                f"hierarchical_splits_json_path: {config.hierarchical_splits_json_path}",
                f"hierarchical_schema_json_path: {config.hierarchical_schema_json_path}",
                f"hierarchical_summary_json_path: {config.hierarchical_summary_json_path}",
                f"comparison_csv_path: {config.comparison_csv_path}",
                f"summary_json_path: {config.summary_json_path}",
                f"findings_report_path: {config.findings_report_path}",
                f"final_report_path: {config.final_report_path}",
                f"summary_doc_path: {config.summary_doc_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.count == 2
    assert loaded.run_missing_artifacts is False
    assert loaded.goal3_summary_json_path == config.goal3_summary_json_path


def _write_small_artifacts(tmp_path: Path) -> Goal5CompressionPipelineConfig:
    output_dir = tmp_path / "outputs" / "v1"
    docs_dir = tmp_path / "docs" / "goal5"
    config = Goal5CompressionPipelineConfig(
        count=2,
        reuse_existing_artifacts=True,
        run_missing_artifacts=False,
        macro_config_path=tmp_path / "configs" / "macro.yaml",
        frequent_motif_config_path=tmp_path / "configs" / "frequent.yaml",
        learned_motif_config_path=tmp_path / "configs" / "learned.yaml",
        neural_egraph_config_path=tmp_path / "configs" / "neural.yaml",
        hierarchical_export_config_path=tmp_path / "configs" / "hierarchy.yaml",
        goal3_summary_json_path=output_dir / "dag_compression_summary.json",
        goal3_metrics_csv_path=output_dir / "dag_compression_metrics.csv",
        goal4_summary_json_path=output_dir / "egraph_compression_summary.json",
        goal4_safe_metrics_csv_path=output_dir / "egraph_compression_metrics_safe.csv",
        goal4_positive_real_metrics_csv_path=(
            output_dir / "egraph_compression_metrics_positive_real.csv"
        ),
        macro_summary_json_path=output_dir / "goal5_macro_graph_summary.json",
        macro_metrics_csv_path=output_dir / "goal5_macro_graph_metrics.csv",
        macro_metrics_jsonl_path=output_dir / "goal5_macro_graph_metrics.jsonl",
        frequent_motif_summary_json_path=output_dir / "goal5_frequent_motif_summary.json",
        frequent_motif_metrics_csv_path=output_dir / "goal5_frequent_motif_metrics.csv",
        frequent_motif_metrics_jsonl_path=output_dir / "goal5_frequent_motif_metrics.jsonl",
        frequent_motif_vocab_json_path=output_dir / "goal5_frequent_motif_vocab.json",
        learned_motif_summary_json_path=output_dir / "goal5_learned_motif_summary.json",
        learned_motif_metrics_csv_path=output_dir / "goal5_learned_motif_metrics.csv",
        learned_motif_metrics_jsonl_path=output_dir / "goal5_learned_motif_metrics.jsonl",
        learned_motif_vocab_json_path=output_dir / "goal5_learned_motif_vocab.json",
        learned_motif_train_log_json_path=output_dir / "goal5_learned_motif_train_log.json",
        neural_egraph_summary_json_path=output_dir / "goal5_neural_egraph_summary.json",
        neural_egraph_metrics_csv_path=output_dir / "goal5_neural_egraph_metrics.csv",
        neural_egraph_candidate_dataset_jsonl_path=(
            output_dir / "goal5_neural_egraph_candidate_dataset.jsonl"
        ),
        neural_egraph_train_log_json_path=output_dir / "goal5_neural_egraph_train_log.json",
        hierarchical_graphs_jsonl_path=output_dir / "goal5_hierarchical_graphs.jsonl",
        hierarchical_splits_json_path=output_dir / "goal5_graph_splits.json",
        hierarchical_schema_json_path=output_dir / "goal5_graph_schema.json",
        hierarchical_summary_json_path=output_dir / "goal5_hierarchical_export_summary.json",
        comparison_csv_path=output_dir / "goal5_compression_comparison.csv",
        summary_json_path=output_dir / "goal5_compression_summary.json",
        findings_report_path=output_dir / "GOAL5_COMPRESSION_FINDINGS.md",
        final_report_path=docs_dir / "GOAL5_ML_FACING_COMPRESSION_STUDY.md",
        summary_doc_path=docs_dir / "GOAL5_SUMMARY.md",
    )
    _write_json(config.goal3_summary_json_path, {"processed_count": 2, "supported_count": 2})
    _write_json(config.goal4_summary_json_path, _small_goal4_summary())
    _write_json(config.macro_summary_json_path, _small_macro_summary())
    _write_json(config.frequent_motif_summary_json_path, _small_frequent_summary())
    _write_json(config.learned_motif_summary_json_path, _small_learned_summary())
    _write_json(config.neural_egraph_summary_json_path, _small_neural_summary())
    _write_json(config.hierarchical_summary_json_path, _small_hierarchical_summary())
    _write_json(config.frequent_motif_vocab_json_path, {"motifs": []})
    _write_json(config.learned_motif_vocab_json_path, {"motifs": []})
    _write_json(config.learned_motif_train_log_json_path, {"trained_final_reasoning_gnn": False})
    _write_json(config.neural_egraph_train_log_json_path, {"trained_final_reasoning_gnn": False})
    _write_json(config.hierarchical_schema_json_path, {"schema_version": "goal5_graph_v1"})
    _write_json(config.hierarchical_splits_json_path, {"train": [], "validation": [], "test": []})
    for path in [
        config.goal3_metrics_csv_path,
        config.goal4_safe_metrics_csv_path,
        config.goal4_positive_real_metrics_csv_path,
        config.macro_metrics_csv_path,
        config.frequent_motif_metrics_csv_path,
        config.learned_motif_metrics_csv_path,
        config.neural_egraph_metrics_csv_path,
    ]:
        _write_csv(path)
    return config


def _small_goal4_summary() -> dict[str, object]:
    return {
        "rule_modes": {
            "safe": _small_egraph_mode_summary(nodes=38.0, gain=1.05),
            "positive_real_formal": _small_egraph_mode_summary(nodes=34.0, gain=1.17),
        }
    }


def _small_egraph_mode_summary(*, nodes: float, gain: float) -> dict[str, object]:
    return {
        "processed_count": 2,
        "success_count": 2,
        "validation_failure_count": 0,
        "original_eml_dag_nodes": {"median": 42.0},
        "extracted_eml_dag_nodes": {"median": nodes},
        "compression_gain_vs_goal3_dag": {"median": gain},
        "results_by_subset_label": {
            "nontrivial_v1": {
                "processed_count": 1,
                "success_count": 1,
                "compression_gain_vs_goal3_dag": {"median": 1.0},
            }
        },
    }


def _small_macro_summary() -> dict[str, object]:
    return {
        "processed_count": 2,
        "success_count": 2,
        "median_macro_graph_alpha": 0.8,
        "median_compression_gain_vs_goal3_eml_dag": 5.0,
        "expansion_validation_failure_count": 0,
        "representation_contract": {"is_pure_eml": False},
        "results_by_subset_label": {
            "all_v1": {"macro_graph_nodes": {"median": 8.0}},
            "nontrivial_v1": {
                "processed_count": 1,
                "success_count": 1,
                "macro_graph_nodes": {"median": 7.0},
                "compression_gain_vs_goal3_eml_dag": {"median": 5.3},
                "expansion_validation_failure_count": 0,
            },
        },
    }


def _small_frequent_summary() -> dict[str, object]:
    motif = {
        "motif_id": "pure_eml_dag_0000",
        "motif_type": "pure_eml_dag",
        "node_count": 2,
        "support_count": 5,
        "total_covered_nodes": 10,
        "official_macro_name": "eml_exp",
    }
    return {
        "processed_count": 2,
        "success_count": 2,
        "motif_vocab_size": 4,
        "motif_counts_by_type": {"pure_eml_dag": 2, "macro_graph": 2},
        "motif_compressed_nodes": {"median": 6.0},
        "motif_coverage_percent": {"median": 57.0},
        "compression_gain_vs_goal3_eml_dag": {"median": 7.4},
        "expansion_validation_failure_count": 0,
        "representation_contract": {"motif_nodes_are_pure_eml": False},
        "top_motifs_by_support": [motif],
        "top_motifs_by_compression_saved": [motif],
        "motifs_that_correspond_to_official_macros": [motif],
        "motifs_not_obvious_official_macros": [],
        "results_by_subset_label": {
            "nontrivial_v1": {
                "processed_count": 1,
                "success_count": 1,
                "compression_gain_vs_goal3_eml_dag": {"median": 7.7},
                "motif_coverage_percent": {"median": 60.0},
                "expansion_validation_failure_count": 0,
            },
        },
    }


def _small_learned_summary() -> dict[str, object]:
    split = {
        "processed_count": 1,
        "success_count": 1,
        "learned_gain_vs_goal3_eml_dag": {"median": 7.0},
        "reconstruction_failure_count": 0,
    }
    return {
        "processed_count": 2,
        "success_count": 2,
        "learned_vocab_size": 3,
        "random_vocab_size": 3,
        "learned_gain_vs_goal3_eml_dag": {"median": 7.1},
        "learned_vs_frequent_motif_compression": {"median": 1.0},
        "learned_vs_random_motif_compression": {"median": 1.0},
        "random_gain_vs_goal3_eml_dag": {"median": 7.2},
        "reconstruction_failure_count": 0,
        "integrity": {
            "motif_ids_are_pure_eml_nodes": False,
            "modified_official_eml_compiler": False,
            "trained_final_symbolic_reasoning_gnn": False,
        },
        "results_by_split": {"train": split, "validation": split, "test": split},
        "results_by_subset_label": {
            "all_v1": {"learned_motif_nodes": {"median": 6.0}},
            "nontrivial_v1": {
                "processed_count": 1,
                "success_count": 1,
                "learned_motif_nodes": {"median": 5.0},
                "learned_gain_vs_goal3_eml_dag": {"median": 7.4},
                "reconstruction_failure_count": 0,
            },
        },
    }


def _small_neural_summary() -> dict[str, object]:
    subset = {
        "processed_count": 2,
        "success_count": 2,
        "validation_failure_count": 0,
        "median_compression_gain": 1.07,
        "percent_matching_exact_best": 64.0,
        "neural_regret_vs_exact_best": {"median": 0.0, "p90": 3.0},
        "neural_speedup_vs_exact_scoring": {"median": 100.0},
    }
    return {
        "processed_group_count": 2,
        "success_count": 2,
        "validation_failure_count": 0,
        "neural_vs_exact_beam": {
            "top1_eml_dag_nodes": {"median": 37.0},
            "regret_vs_exact_best": {"median": 0.0, "p90": 3.0},
            "percent_matching_exact_best": 64.0,
        },
        "runtime_tradeoff": {"neural_speedup_vs_exact_scoring": {"median": 100.0}},
        "compression_gain_vs_goal3_dag": {"neural": {"median": 1.07}},
        "integrity": {
            "modified_official_eml_compiler": False,
            "trained_final_symbolic_reasoning_gnn": False,
        },
        "results_by_rule_mode": {"safe": subset, "positive_real_formal": subset},
        "results_by_split": {"train": subset, "validation": subset, "test": subset},
        "results_by_subset_label": {"nontrivial_v1": subset},
    }


def _small_hierarchical_summary() -> dict[str, object]:
    return {
        "graph_count": 5,
        "representation_modes_exported": ["macro_graph", "hierarchical_eml_graph"],
        "expansion_validation_rate": 100.0,
        "reconstruction_validation_rate": 100.0,
        "missing_expansion_count": 0,
        "node_edge_stats_by_mode": {
            "hierarchical_eml_graph": {
                "graph_count": 2,
                "node_count": {"median": 73.0},
                "edge_count": {"median": 128.0},
                "reconstruction_validation_rate": 100.0,
            }
        },
        "train_val_test_counts": {},
        "integrity": {
            "compressed_graph_nodes_are_pure_eml": False,
            "modified_official_eml_compiler": False,
            "trained_final_symbolic_reasoning_gnn": False,
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("index\n0\n", encoding="utf-8")
