"""Tests for the Goal 5.4 neural e-graph extractor experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.goal5_neural_egraph_extractor import (
    NeuralEgraphExtractorConfig,
    load_config,
    run_goal5_neural_egraph_extractor,
)


def test_goal5_neural_egraph_extractor_small_end_to_end(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "v1"
    config = NeuralEgraphExtractorConfig(
        count=5,
        run_modes=("safe",),
        max_iterations=2,
        max_enodes=1_000,
        max_eclasses=1_000,
        saturation_timeout_seconds=1,
        row_timeout_seconds=2,
        beam_size=4,
        max_candidate_depth=5,
        max_candidates_evaluated=4,
        hidden_size=4,
        epochs=2,
        max_pairs_per_group=3,
        reuse_existing_candidate_dataset=False,
        candidate_dataset_jsonl_path=output_dir / "goal5_neural_egraph_candidate_dataset.jsonl",
        metrics_csv_path=output_dir / "goal5_neural_egraph_metrics.csv",
        summary_json_path=output_dir / "goal5_neural_egraph_summary.json",
        train_log_json_path=output_dir / "goal5_neural_egraph_train_log.json",
    )

    result = run_goal5_neural_egraph_extractor(config)

    for path in result.output_paths:
        assert path.exists()
        assert "outputs/v1" in path.as_posix()

    candidate_rows = [
        json.loads(line)
        for line in config.candidate_dataset_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert candidate_rows
    assert all(row["true_official_eml_dag_nodes"] is not None for row in candidate_rows)

    with config.metrics_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        metric_rows = list(csv.DictReader(csv_file))
    assert metric_rows
    assert "neural_eml_dag_nodes" in metric_rows[0]
    assert "exact_best_eml_dag_nodes" in metric_rows[0]

    summary = json.loads(config.summary_json_path.read_text(encoding="utf-8"))
    assert summary["processed_group_count"] == len(metric_rows)
    assert summary["integrity"]["ground_truth_cost"] == "official_pure_eml_dag_nodes"
    assert summary["integrity"]["trained_final_symbolic_reasoning_gnn"] is False

    train_log = json.loads(config.train_log_json_path.read_text(encoding="utf-8"))
    assert train_log["trained_final_reasoning_gnn"] is False
    assert train_log["test_set_used_for_training"] is False
    assert train_log["model"]["model_type"] == "feature_mlp_pairwise_ranker"


def test_goal5_neural_egraph_extractor_config_loads_v1_yaml() -> None:
    config = load_config(Path("configs/neural_egraph_extractor_v1.yaml"))

    assert config.count == 10_000
    assert set(config.run_modes) == {"safe", "positive_real_formal"}
    assert config.candidate_dataset_jsonl_path.as_posix() == (
        "outputs/v1/goal5_neural_egraph_candidate_dataset.jsonl"
    )
    assert config.summary_json_path.as_posix() == "outputs/v1/goal5_neural_egraph_summary.json"
