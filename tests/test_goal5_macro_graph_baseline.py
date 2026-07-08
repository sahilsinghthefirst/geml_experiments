from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.goal5_macro_graph_baseline import (
    MacroGraphBaselineConfig,
    load_config,
    run_goal5_macro_graph_baseline,
)


def test_goal5_macro_graph_small_pipeline_writes_outputs(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=25)

    result = run_goal5_macro_graph_baseline(config)

    for path in result.output_paths:
        assert path.exists()
        assert "outputs/v1" in path.as_posix()

    rows = read_jsonl(config.metrics_jsonl_path)
    assert len(rows) == 25
    assert result.summary["processed_count"] == 25
    assert result.summary["success_count"] == 25
    assert result.summary["expansion_validation_failure_count"] == 0
    assert result.summary["representation_contract"]["is_pure_eml"] is False


def test_goal5_macro_graph_rows_have_required_labels_and_validation(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=10)

    run_goal5_macro_graph_baseline(config)

    rows = read_jsonl(config.metrics_jsonl_path)
    assert {row["representation_mode"] for row in rows} == {"macro_graph_v1"}
    assert {row["subset_label"] for row in rows} <= {
        "all_v1",
        "nontrivial_v1",
        "identity_heavy_v1",
    }
    assert all(row["success"] is True for row in rows)
    assert all(row["expansion_valid"] is True for row in rows)
    assert all(row["pure_eml_equivalent"] is True for row in rows)


def test_goal5_macro_graph_csv_rows_are_not_dropped(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=12)

    run_goal5_macro_graph_baseline(config)

    with config.metrics_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))

    assert len(csv_rows) == 12
    assert {int(row["index"]) for row in csv_rows} == set(range(12))


def test_goal5_macro_graph_summary_has_required_subsets_and_operator_summaries(
    tmp_path: Path,
) -> None:
    config = small_config(tmp_path, count=25)

    result = run_goal5_macro_graph_baseline(config)

    assert set(result.summary["results_by_subset_label"]) == {
        "all_v1",
        "nontrivial_v1",
        "identity_heavy_v1",
    }
    operator_summaries = result.summary["operator_family_summaries"]
    assert operator_summaries["by_dominant_operator_family"]
    assert operator_summaries["by_operator_signature"]


def test_goal5_macro_graph_config_loads_v1_yaml() -> None:
    config = load_config(Path("configs/macro_graph_v1.yaml"))

    assert config.count == 10_000
    assert config.seed == 0
    assert config.max_depth == 4
    assert config.operator_set == ("add", "mul", "exp", "log")
    assert config.symbol_names == ("x", "y")
    assert config.source_serialization == "srepr"
    assert "outputs/v1" in config.metrics_jsonl_path.as_posix()


def small_config(tmp_path: Path, *, count: int) -> MacroGraphBaselineConfig:
    output_dir = tmp_path / "outputs" / "v1"
    return MacroGraphBaselineConfig(
        count=count,
        metrics_csv_path=output_dir / "goal5_macro_graph_metrics.csv",
        metrics_jsonl_path=output_dir / "goal5_macro_graph_metrics.jsonl",
        summary_json_path=output_dir / "goal5_macro_graph_summary.json",
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
