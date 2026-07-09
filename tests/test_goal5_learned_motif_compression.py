from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.goal5_frequent_motif_mining import (
    FrequentMotifMiningConfig,
    run_goal5_frequent_motif_mining,
)
from geml.experiments.goal5_learned_motif_compression import (
    LearnedMotifCompressionConfig,
    load_config,
    run_goal5_learned_motif_compression,
)

from tests.goal5_fixture_builders import ensure_frequent_fixture, ensure_macro_fixture


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


def test_goal5_learned_motif_uses_train_only_candidate_discovery(
    tmp_path: Path,
) -> None:
    paths = ensure_macro_fixture(tmp_path, count=25)
    train_only_frequent_config = FrequentMotifMiningConfig(
        count=25,
        min_support=2,
        max_vocab_size=30,
        candidate_discovery_split="train",
        train_fraction=0.5,
        validation_fraction=0.25,
        full_corpus_metrics_csv_path=None,
        input_jsonl_path=paths.input_jsonl_path,
        goal3_metrics_csv_path=paths.goal3_metrics_csv_path,
        macro_graph_metrics_csv_path=paths.macro_metrics_csv_path,
        vocab_json_path=paths.output_dir / "train_only_vocab.json",
        metrics_csv_path=paths.output_dir / "train_only_metrics.csv",
        metrics_jsonl_path=paths.output_dir / "train_only_metrics.jsonl",
        summary_json_path=paths.output_dir / "train_only_summary.json",
    )
    run_goal5_frequent_motif_mining(train_only_frequent_config)
    config = LearnedMotifCompressionConfig(
        count=25,
        train_fraction=0.5,
        validation_fraction=0.25,
        learned_vocab_sizes=(5,),
        coverage_bonuses=(0.0, 0.01),
        nontrivial_coverage_bonuses=(0.0,),
        vocab_complexity_penalties=(0.0,),
        expansion_complexity_penalties=(0.0,),
        input_jsonl_path=paths.input_jsonl_path,
        goal3_metrics_csv_path=paths.goal3_metrics_csv_path,
        macro_graph_metrics_csv_path=paths.macro_metrics_csv_path,
        frequent_motif_vocab_json_path=train_only_frequent_config.vocab_json_path,
        frequent_motif_metrics_csv_path=train_only_frequent_config.metrics_csv_path,
        learned_vocab_json_path=paths.output_dir / "learned_train_only_vocab.json",
        metrics_csv_path=paths.output_dir / "learned_train_only_metrics.csv",
        metrics_jsonl_path=paths.output_dir / "learned_train_only_metrics.jsonl",
        summary_json_path=paths.output_dir / "learned_train_only_summary.json",
        train_log_json_path=paths.output_dir / "learned_train_only_log.json",
    )

    result = run_goal5_learned_motif_compression(config)

    train_log = json.loads(config.train_log_json_path.read_text(encoding="utf-8"))
    assert train_log["candidate_discovery"]["candidate_discovery_mode"] == "train_only"
    assert train_log["test_set_used_for_candidate_discovery"] is False
    assert result.summary["integrity"]["test_set_used_for_candidate_discovery"] is False
    rows = read_jsonl(config.metrics_jsonl_path)
    heldout_rows = [row for row in rows if row["split"] in {"validation", "test"}]
    assert {row["split"] for row in heldout_rows} == {"validation", "test"}
    assert all(row["reconstruction_valid"] is True for row in heldout_rows)


def test_goal5_learned_motif_config_loads_v1_yaml() -> None:
    config = load_config(Path("configs/learned_motifs_v1.yaml"))

    assert config.count == 10_000
    assert config.seed == 0
    assert config.max_depth == 4
    assert config.operator_set == ("add", "mul", "exp", "log")
    assert config.symbol_names == ("x", "y")
    assert config.source_serialization == "srepr"
    assert "train_only" in config.frequent_motif_vocab_json_path.as_posix()
    assert "outputs/v1" in config.metrics_jsonl_path.as_posix()


def small_config(tmp_path: Path, *, count: int) -> LearnedMotifCompressionConfig:
    paths = ensure_frequent_fixture(tmp_path)
    return LearnedMotifCompressionConfig(
        count=count,
        learned_vocab_sizes=(5, 8),
        coverage_bonuses=(0.0, 0.01),
        nontrivial_coverage_bonuses=(0.0,),
        vocab_complexity_penalties=(0.0,),
        expansion_complexity_penalties=(0.0,),
        input_jsonl_path=paths.input_jsonl_path,
        goal3_metrics_csv_path=paths.goal3_metrics_csv_path,
        macro_graph_metrics_csv_path=paths.macro_metrics_csv_path,
        frequent_motif_vocab_json_path=paths.frequent_vocab_json_path,
        frequent_motif_metrics_csv_path=paths.frequent_metrics_csv_path,
        learned_vocab_json_path=paths.learned_vocab_json_path,
        metrics_csv_path=paths.learned_metrics_csv_path,
        metrics_jsonl_path=paths.learned_metrics_jsonl_path,
        summary_json_path=paths.learned_summary_json_path,
        train_log_json_path=paths.learned_train_log_json_path,
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
