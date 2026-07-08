from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.goal5_frequent_motif_mining import (
    FrequentMotifMiningConfig,
    load_config,
    run_goal5_frequent_motif_mining,
)


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


def small_config(tmp_path: Path, *, count: int) -> FrequentMotifMiningConfig:
    output_dir = tmp_path / "outputs" / "v1"
    return FrequentMotifMiningConfig(
        count=count,
        min_support=2,
        max_vocab_size=30,
        vocab_json_path=output_dir / "goal5_frequent_motif_vocab.json",
        metrics_csv_path=output_dir / "goal5_frequent_motif_metrics.csv",
        metrics_jsonl_path=output_dir / "goal5_frequent_motif_metrics.jsonl",
        summary_json_path=output_dir / "goal5_frequent_motif_summary.json",
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
