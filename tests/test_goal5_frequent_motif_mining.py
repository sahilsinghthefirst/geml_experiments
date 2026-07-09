from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.compression.motif_dataset import SplitConfig, assign_split
from geml.compression.motif_vocab import load_motif_vocabulary
from geml.experiments.goal5_frequent_motif_mining import (
    FrequentMotifMiningConfig,
    load_config,
    run_goal5_frequent_motif_mining,
)

from tests.goal5_fixture_builders import ensure_macro_fixture


def test_goal5_frequent_motif_small_pipeline_writes_outputs(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=25)

    result = run_goal5_frequent_motif_mining(config)

    for path in result.output_paths:
        assert path.exists()
        assert "outputs/v1" in path.as_posix()
    assert result.summary["processed_count"] == 25
    assert result.summary["success_count"] == 25
    assert result.summary["expansion_validation_failure_count"] == 0
    assert result.summary["motif_vocab_size"] > 0


def test_goal5_frequent_motif_rows_have_required_metrics(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=20)

    run_goal5_frequent_motif_mining(config)

    rows = read_jsonl(config.metrics_jsonl_path)
    assert len(rows) == 20
    assert all(row["expansion_valid"] is True for row in rows)
    assert all(row["motif_vocab_size"] > 0 for row in rows)
    assert all(row["motif_compressed_nodes"] is not None for row in rows)
    assert {row["subset_label"] for row in rows} <= {
        "all_v1",
        "nontrivial_v1",
        "identity_heavy_v1",
    }


def test_goal5_frequent_motif_csv_rows_are_not_dropped(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=12)

    run_goal5_frequent_motif_mining(config)

    with config.metrics_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 12
    assert {int(row["index"]) for row in rows} == set(range(12))


def test_goal5_frequent_motif_summary_has_required_analysis(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=25)

    result = run_goal5_frequent_motif_mining(config)

    assert set(result.summary["results_by_subset_label"]) == {
        "all_v1",
        "nontrivial_v1",
        "identity_heavy_v1",
    }
    assert result.summary["top_motifs_by_support"]
    assert result.summary["top_motifs_by_compression_saved"]
    assert result.summary["top_motifs_by_nontrivial_v1_coverage"]
    assert "motifs_that_correspond_to_official_macros" in result.summary
    assert "motifs_not_obvious_official_macros" in result.summary


def test_train_only_motif_candidate_mining_excludes_test_expressions(
    tmp_path: Path,
) -> None:
    paths = ensure_macro_fixture(tmp_path, count=25)
    split_config = SplitConfig(seed=0, train_fraction=0.5, validation_fraction=0.25)
    full_config = frequent_config_for_paths(
        paths,
        count=25,
        vocab_json_path=paths.output_dir / "full_vocab.json",
        metrics_csv_path=paths.output_dir / "full_metrics.csv",
        metrics_jsonl_path=paths.output_dir / "full_metrics.jsonl",
        summary_json_path=paths.output_dir / "full_summary.json",
        candidate_discovery_split="all",
        full_corpus_metrics_csv_path=None,
        train_fraction=split_config.train_fraction,
        validation_fraction=split_config.validation_fraction,
    )
    run_goal5_frequent_motif_mining(full_config)
    train_only_config = frequent_config_for_paths(
        paths,
        count=25,
        vocab_json_path=paths.output_dir / "train_only_vocab.json",
        metrics_csv_path=paths.output_dir / "train_only_metrics.csv",
        metrics_jsonl_path=paths.output_dir / "train_only_metrics.jsonl",
        summary_json_path=paths.output_dir / "train_only_summary.json",
        candidate_discovery_split="train",
        full_corpus_metrics_csv_path=full_config.metrics_csv_path,
        train_fraction=split_config.train_fraction,
        validation_fraction=split_config.validation_fraction,
    )

    result = run_goal5_frequent_motif_mining(train_only_config)

    split_by_index = {index: assign_split(index, split_config) for index in range(25)}
    assert {"train", "validation", "test"} <= set(split_by_index.values())
    vocabulary = load_motif_vocabulary(train_only_config.vocab_json_path)
    discovery_indices = set(vocabulary.metadata["candidate_discovery_expression_indices"])
    assert discovery_indices
    assert all(split_by_index[index] == "train" for index in discovery_indices)
    assert vocabulary.metadata["test_set_used_for_candidate_discovery"] is False
    assert vocabulary.metadata["validation_set_used_for_candidate_discovery"] is False
    assert result.summary["candidate_discovery"]["test_set_used_for_candidate_discovery"] is False
    assert result.summary["full_corpus_comparison"]["comparison_available"] is True


def test_train_only_vocab_compresses_validation_and_test_rows(tmp_path: Path) -> None:
    paths = ensure_macro_fixture(tmp_path, count=25)
    config = frequent_config_for_paths(
        paths,
        count=25,
        vocab_json_path=paths.output_dir / "train_only_vocab.json",
        metrics_csv_path=paths.output_dir / "train_only_metrics.csv",
        metrics_jsonl_path=paths.output_dir / "train_only_metrics.jsonl",
        summary_json_path=paths.output_dir / "train_only_summary.json",
        candidate_discovery_split="train",
        full_corpus_metrics_csv_path=None,
        train_fraction=0.5,
        validation_fraction=0.25,
    )

    result = run_goal5_frequent_motif_mining(config)

    rows = read_jsonl(config.metrics_jsonl_path)
    heldout_rows = [row for row in rows if row["split"] in {"validation", "test"}]
    assert {row["split"] for row in heldout_rows} == {"validation", "test"}
    assert all(row["expansion_valid"] is True for row in heldout_rows)
    assert all(row["motif_compressed_nodes"] is not None for row in heldout_rows)
    assert result.summary["results_by_split"]["validation"]["success_count"] > 0
    assert result.summary["results_by_split"]["test"]["success_count"] > 0
    assert result.summary["expansion_validation_failure_count"] == 0


def test_goal5_frequent_motif_config_loads_v1_yaml() -> None:
    config = load_config(Path("configs/frequent_motifs_v1.yaml"))

    assert config.count == 10_000
    assert config.seed == 0
    assert config.max_depth == 4
    assert config.operator_set == ("add", "mul", "exp", "log")
    assert config.symbol_names == ("x", "y")
    assert config.source_serialization == "srepr"
    assert config.min_motif_nodes == 1
    assert config.max_motif_nodes == 2
    assert "outputs/v1" in config.metrics_jsonl_path.as_posix()


def test_goal5_frequent_motif_train_only_config_loads_v1_yaml() -> None:
    config = load_config(Path("configs/frequent_motifs_train_only_v1.yaml"))

    assert config.count == 10_000
    assert config.candidate_discovery_split == "train"
    assert "train_only" in config.vocab_json_path.as_posix()
    assert "outputs/v1" in config.metrics_jsonl_path.as_posix()


def small_config(tmp_path: Path, *, count: int) -> FrequentMotifMiningConfig:
    paths = ensure_macro_fixture(tmp_path)
    return FrequentMotifMiningConfig(
        count=count,
        min_support=2,
        max_vocab_size=30,
        full_corpus_metrics_csv_path=None,
        input_jsonl_path=paths.input_jsonl_path,
        goal3_metrics_csv_path=paths.goal3_metrics_csv_path,
        macro_graph_metrics_csv_path=paths.macro_metrics_csv_path,
        vocab_json_path=paths.frequent_vocab_json_path,
        metrics_csv_path=paths.frequent_metrics_csv_path,
        metrics_jsonl_path=paths.frequent_metrics_jsonl_path,
        summary_json_path=paths.frequent_summary_json_path,
    )


def frequent_config_for_paths(
    paths: object,
    *,
    count: int,
    vocab_json_path: Path,
    metrics_csv_path: Path,
    metrics_jsonl_path: Path,
    summary_json_path: Path,
    candidate_discovery_split: str,
    full_corpus_metrics_csv_path: Path | None,
    train_fraction: float,
    validation_fraction: float,
) -> FrequentMotifMiningConfig:
    return FrequentMotifMiningConfig(
        count=count,
        min_support=2,
        max_vocab_size=30,
        candidate_discovery_split=candidate_discovery_split,  # type: ignore[arg-type]
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        full_corpus_metrics_csv_path=full_corpus_metrics_csv_path,
        input_jsonl_path=paths.input_jsonl_path,
        goal3_metrics_csv_path=paths.goal3_metrics_csv_path,
        macro_graph_metrics_csv_path=paths.macro_metrics_csv_path,
        vocab_json_path=vocab_json_path,
        metrics_csv_path=metrics_csv_path,
        metrics_jsonl_path=metrics_jsonl_path,
        summary_json_path=summary_json_path,
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
