from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.egraph_compression_study import (
    EgraphCompressionRow,
    EgraphCompressionStudyConfig,
    load_config,
    run_egraph_compression_study,
    summarize_rows,
)

from tests.goal5_fixture_builders import ensure_goal3_fixture


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


def test_egraph_summary_after_threshold_denominators_include_hard_failures() -> None:
    rows = [
        fake_egraph_row(index=0, below_after=True, extracted_eml_dag_nodes=4),
        fake_egraph_row(index=1, below_after=False, extracted_eml_dag_nodes=8),
        fake_egraph_row(
            index=2,
            saturation_status="timeout",
            extraction_status="timeout",
            validation_status="error",
            timeout=True,
            below_after=None,
            extracted_eml_dag_nodes=None,
        ),
        fake_egraph_row(
            index=3,
            extraction_status="failed",
            validation_status="error",
            below_after=None,
            extracted_eml_dag_nodes=None,
        ),
        fake_egraph_row(
            index=4,
            validation_status="valid",
            below_after=None,
            extracted_eml_dag_nodes=None,
        ),
    ]

    summary = summarize_rows(rows)

    assert summary["processed"] == 5
    assert summary["success"] == 2
    assert summary["timeout"] == 1
    assert summary["extraction_failed"] == 1
    assert summary["official_compilation_failed"] == 1
    assert summary["success_only_after_rate"] == 50.0
    assert summary["all_processed_after_rate"] == 20.0
    assert summary["percent_below_threshold_after_egraph"] == 50.0
    assert summary["percent_below_threshold_after_egraph_all_processed"] == 20.0


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
    completed_rows = [
        row for row in safe_rows + positive_rows if row["extraction_status"] == "completed"
    ]

    assert completed_rows
    assert all(row["structural_purity_valid"] is True for row in completed_rows)
    assert all(row["assumptions"] is None for row in safe_rows)
    assert all(row["branch_sensitive_rules_used"] is False for row in safe_rows)
    assert all(row["branch_sensitive_rule_count"] == 0 for row in safe_rows)
    assert all(row["assumptions"] == "positive_real_formal" for row in positive_rows)
    assert all("branch_sensitive_rules_used" in row for row in positive_rows)


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
    paths = ensure_goal3_fixture(tmp_path)
    return EgraphCompressionStudyConfig(
        count=count,
        input_jsonl_path=paths.input_jsonl_path,
        goal3_metrics_csv_path=paths.goal3_metrics_csv_path,
        goal3_summary_json_path=paths.goal3_summary_json_path,
        v0_v1_comparison_summary_json_path=None,
        output_dir=paths.output_dir,
        safe_metrics_csv_path=paths.egraph_safe_metrics_csv_path,
        safe_metrics_jsonl_path=paths.egraph_safe_metrics_jsonl_path,
        positive_real_metrics_csv_path=paths.egraph_positive_metrics_csv_path,
        positive_real_metrics_jsonl_path=paths.egraph_positive_metrics_jsonl_path,
        summary_json_path=paths.egraph_summary_json_path,
        run_metadata_json_path=paths.egraph_run_metadata_json_path,
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


def fake_egraph_row(
    *,
    index: int,
    saturation_status: str = "saturated",
    extraction_status: str = "completed",
    validation_status: str = "valid",
    timeout: bool = False,
    below_after: bool | None,
    extracted_eml_dag_nodes: int | None,
) -> EgraphCompressionRow:
    return EgraphCompressionRow(
        index=index,
        original_expression="x + y",
        original_srepr="Add(Symbol('x'), Symbol('y'))",
        rule_mode="safe",
        assumptions=None,
        saturation_status=saturation_status,
        extraction_status=extraction_status,
        validation_status=validation_status,
        timeout=timeout,
        eclass_count=1,
        enode_count=1,
        iterations_run=1,
        total_rules_applied=0,
        branch_sensitive_rules_used=False,
        branch_sensitive_rule_count=0,
        branch_sensitive_rule_names=(),
        extracted_expression="Add(x,y)" if extracted_eml_dag_nodes is not None else None,
        extracted_srepr="Add(Symbol('x'), Symbol('y'))"
        if extracted_eml_dag_nodes is not None
        else None,
        validation_error=None if validation_status == "valid" else "failed",
        max_abs_error=0.0 if validation_status == "valid" else None,
        original_ast_tree_nodes=3,
        original_ast_dag_nodes=3,
        original_eml_tree_nodes=12,
        original_eml_dag_nodes=8,
        extracted_ast_tree_nodes=3 if extracted_eml_dag_nodes is not None else None,
        extracted_ast_dag_nodes=3 if extracted_eml_dag_nodes is not None else None,
        extracted_eml_tree_nodes=12 if extracted_eml_dag_nodes is not None else None,
        extracted_eml_dag_nodes=extracted_eml_dag_nodes,
        goal3_tree_alpha=4.0,
        goal3_dag_alpha_vs_ast_tree=8 / 3,
        goal3_dag_alpha_vs_ast_dag=8 / 3,
        optimized_tree_alpha=4.0 if extracted_eml_dag_nodes is not None else None,
        optimized_dag_alpha_vs_ast_tree=(
            extracted_eml_dag_nodes / 3 if extracted_eml_dag_nodes is not None else None
        ),
        optimized_dag_alpha_vs_ast_dag=(
            extracted_eml_dag_nodes / 3 if extracted_eml_dag_nodes is not None else None
        ),
        compression_gain_vs_goal3_dag=(
            8 / extracted_eml_dag_nodes if extracted_eml_dag_nodes else None
        ),
        alpha_threshold_current=2.0,
        below_threshold_goal3_dag=False,
        below_threshold_optimized_dag=below_after,
        subset_label="nontrivial_v1",
        structural_purity_valid=True,
        runtime_seconds=0.01,
        error=None if validation_status == "valid" else "failed",
    )
