from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.egraph_compression_study import (
    EgraphCompressionStudyConfig,
    load_config,
    run_egraph_compression_study,
)


def test_egraph_compression_small_pipeline_count_25_writes_outputs(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=25)

    result = run_egraph_compression_study(config)

    for path in result.output_paths:
        assert path.exists()
        assert "outputs/v1" in path.as_posix()

    safe_rows = read_jsonl(config.safe_metrics_jsonl_path)
    positive_rows = read_jsonl(config.positive_real_metrics_jsonl_path)
    assert len(safe_rows) == 25
    assert len(positive_rows) == 25
    assert result.summary["rule_modes"]["safe"]["processed_count"] == 25
    assert result.summary["rule_modes"]["positive_real_formal"]["processed_count"] == 25


def test_egraph_compression_rows_are_not_silently_dropped(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=25, checkpoint_interval=7)

    run_egraph_compression_study(config)

    with config.safe_metrics_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        safe_csv_rows = list(csv.DictReader(csv_file))
    with config.positive_real_metrics_csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        positive_csv_rows = list(csv.DictReader(csv_file))

    assert len(safe_csv_rows) == 25
    assert len(positive_csv_rows) == 25
    assert {int(row["index"]) for row in safe_csv_rows} == set(range(25))
    assert {int(row["index"]) for row in positive_csv_rows} == set(range(25))


def test_egraph_compression_timeout_rows_are_retained(tmp_path: Path) -> None:
    config = small_config(
        tmp_path,
        count=5,
        timeout_seconds=1e-12,
        checkpoint_interval=2,
    )

    run_egraph_compression_study(config)

    safe_rows = read_jsonl(config.safe_metrics_jsonl_path)
    positive_rows = read_jsonl(config.positive_real_metrics_jsonl_path)
    assert len(safe_rows) == 5
    assert len(positive_rows) == 5
    assert any(row["timeout"] is True for row in safe_rows + positive_rows)


def test_egraph_compression_rows_have_required_mode_subset_validation_fields(
    tmp_path: Path,
) -> None:
    config = small_config(tmp_path, count=10)

    run_egraph_compression_study(config)

    for row in read_jsonl(config.safe_metrics_jsonl_path) + read_jsonl(
        config.positive_real_metrics_jsonl_path
    ):
        assert row["rule_mode"] in {"safe", "positive_real_formal"}
        assert row["subset_label"] in {"all_v1", "nontrivial_v1", "identity_heavy_v1"}
        assert row["validation_status"] in {"valid", "invalid", "error", None}


def test_egraph_compression_integrity_and_branch_sensitive_labels(tmp_path: Path) -> None:
    config = small_config(tmp_path, count=25)

    run_egraph_compression_study(config)

    safe_rows = read_jsonl(config.safe_metrics_jsonl_path)
    positive_rows = read_jsonl(config.positive_real_metrics_jsonl_path)

    assert all(row["structural_purity_valid"] is True for row in safe_rows + positive_rows)
    assert all(row["assumptions"] is None for row in safe_rows)
    assert all(row["branch_sensitive_rules_used"] is False for row in safe_rows)
    assert all(row["branch_sensitive_rule_count"] == 0 for row in safe_rows)
    assert all(row["assumptions"] == "positive_real_formal" for row in positive_rows)
    assert all(row["branch_sensitive_rules_used"] is True for row in positive_rows)


def test_egraph_compression_config_loads_v1_yaml() -> None:
    config = load_config(Path("configs/egraph_compression_v1.yaml"))

    assert config.count == 10_000
    assert config.seed == 0
    assert config.max_depth == 4
    assert config.operator_set == ("add", "mul", "exp", "log")
    assert config.symbol_names == ("x", "y")
    assert config.source_serialization == "srepr"
    assert "outputs/v1" in config.safe_metrics_jsonl_path.as_posix()


def small_config(
    tmp_path: Path,
    *,
    count: int,
    timeout_seconds: float = 0.25,
    checkpoint_interval: int = 10,
) -> EgraphCompressionStudyConfig:
    output_dir = tmp_path / "outputs" / "v1"
    return EgraphCompressionStudyConfig(
        count=count,
        output_dir=output_dir,
        safe_metrics_csv_path=output_dir / "egraph_compression_metrics_safe.csv",
        safe_metrics_jsonl_path=output_dir / "egraph_compression_metrics_safe.jsonl",
        positive_real_metrics_csv_path=(
            output_dir / "egraph_compression_metrics_positive_real.csv"
        ),
        positive_real_metrics_jsonl_path=(
            output_dir / "egraph_compression_metrics_positive_real.jsonl"
        ),
        summary_json_path=output_dir / "egraph_compression_summary.json",
        run_metadata_json_path=output_dir / "egraph_compression_run_metadata.json",
        timeout_seconds=timeout_seconds,
        beam_size=8,
        max_candidate_depth=7,
        max_candidates_evaluated=8,
        checkpoint_interval=checkpoint_interval,
        resume=False,
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
