from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.goal5_learned_motif_compression import (
    LearnedMotifCompressionConfig,
    load_config,
    run_goal5_learned_motif_compression,
)


def test_goal5_learned_motif_small_pipeline_writes_outputs(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=25)

    result = run_goal5_learned_motif_compression(config)

    for path in result.output_paths:
        assert path.exists()
        assert "outputs/v1" in path.as_posix()
    assert result.summary["processed_count"] == 25
    assert result.summary["success_count"] == 25
    assert result.summary["reconstruction_failure_count"] == 0
    assert result.summary["learned_vocab_size"] == result.summary["random_vocab_size"]
    assert result.summary["integrity"]["trained_final_symbolic_reasoning_gnn"] is False


def test_goal5_learned_motif_rows_separate_graph_sizes(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=20)

    run_goal5_learned_motif_compression(config)

    rows = read_jsonl(config.metrics_jsonl_path)
    assert len(rows) == 20
    assert all(row["reconstruction_valid"] is True for row in rows)
    for row in rows:
        assert row["original_eml_dag_nodes"] >= row["learned_motif_nodes"]
        assert row["macro_graph_nodes"] >= 1
        assert row["frequent_motif_nodes"] >= 1
        assert row["learned_motif_nodes"] >= 1
        assert row["random_motif_nodes"] >= 1


def test_goal5_learned_motif_random_vocab_same_size(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=25)

    run_goal5_learned_motif_compression(config)

    vocab = json.loads(config.learned_vocab_json_path.read_text(encoding="utf-8"))
    assert len(vocab["motifs"]) == len(vocab["random_baseline_motif_ids"])
    assert vocab["trained_final_reasoning_gnn"] is False


def test_goal5_learned_motif_metrics_csv_rows_are_not_dropped(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=12)

    run_goal5_learned_motif_compression(config)

    with config.metrics_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 12
    assert {int(row["index"]) for row in rows} == set(range(12))


def test_goal5_learned_motif_config_loads_v1_yaml() -> None:
    config = load_config(Path("configs/learned_motifs_v1.yaml"))

    assert config.count == 10_000
    assert config.seed == 0
    assert config.max_depth == 4
    assert config.operator_set == ("add", "mul", "exp", "log")
    assert config.symbol_names == ("x", "y")
    assert config.source_serialization == "srepr"
    assert "outputs/v1" in config.metrics_jsonl_path.as_posix()


def small_config(tmp_path: Path, *, count: int) -> LearnedMotifCompressionConfig:
    output_dir = tmp_path / "outputs" / "v1"
    return LearnedMotifCompressionConfig(
        count=count,
        learned_vocab_sizes=(5, 8),
        coverage_bonuses=(0.0, 0.01),
        nontrivial_coverage_bonuses=(0.0,),
        vocab_complexity_penalties=(0.0,),
        expansion_complexity_penalties=(0.0,),
        learned_vocab_json_path=output_dir / "goal5_learned_motif_vocab.json",
        metrics_csv_path=output_dir / "goal5_learned_motif_metrics.csv",
        metrics_jsonl_path=output_dir / "goal5_learned_motif_metrics.jsonl",
        summary_json_path=output_dir / "goal5_learned_motif_summary.json",
        train_log_json_path=output_dir / "goal5_learned_motif_train_log.json",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw_row = json.loads(line)
            if not isinstance(raw_row, dict):
                raise TypeError("JSONL row must be an object")
            rows.append(raw_row)
    return rows
