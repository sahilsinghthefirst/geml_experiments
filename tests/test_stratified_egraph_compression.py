"""Tests for Goal 4.7 stratified e-graph compression analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.stratified_egraph_compression import (
    StratifiedEgraphCompressionConfig,
    build_stratified_egraph_row,
    compute_triviality_features,
    group_by_rule_mode,
    group_by_subset_label,
    group_by_triviality_feature,
    run_stratified_egraph_compression_analysis,
    summarize_egraph_group,
)


def raw_egraph_row(
    *,
    index: str = "0",
    expression: str = "x + y",
    srepr: str = "Add(Symbol('x'), Symbol('y'))",
    rule_mode: str = "safe",
    saturation_status: str = "saturated",
    extraction_status: str = "completed",
    validation_status: str = "valid",
    timeout: str = "False",
    branch_sensitive: str = "False",
    original_ast_nodes: str = "3",
    original_eml_dag_nodes: str = "9",
    extracted_eml_dag_nodes: str = "7",
    goal3_alpha: str = "3.0",
    optimized_alpha: str = "2.3333333333333335",
    gain: str = "1.2857142857142858",
    below_before: str = "False",
    below_after: str = "False",
    structural_purity_valid: str = "True",
) -> dict[str, str]:
    """Build a minimal raw Goal 4.6 e-graph metric row."""
    return {
        "index": index,
        "original_expression": expression,
        "original_srepr": srepr,
        "rule_mode": rule_mode,
        "saturation_status": saturation_status,
        "extraction_status": extraction_status,
        "validation_status": validation_status,
        "timeout": timeout,
        "branch_sensitive_rules_used": branch_sensitive,
        "original_ast_tree_nodes": original_ast_nodes,
        "original_eml_dag_nodes": original_eml_dag_nodes,
        "extracted_eml_dag_nodes": extracted_eml_dag_nodes,
        "goal3_dag_alpha_vs_ast_tree": goal3_alpha,
        "optimized_dag_alpha_vs_ast_tree": optimized_alpha,
        "compression_gain_vs_goal3_dag": gain,
        "alpha_threshold_current": "1.5578858913022597",
        "below_threshold_goal3_dag": below_before,
        "below_threshold_optimized_dag": below_after,
        "structural_purity_valid": structural_purity_valid,
    }


def test_triviality_features_and_subset_label() -> None:
    features = compute_triviality_features(
        "Add(Mul(Symbol('x'), Integer(1)), log(exp(Symbol('y'))))"
    )

    assert features.has_mul_by_one is True
    assert features.has_log_exp is True
    assert features.has_log_one is False
    assert features.triviality_score == 2
    assert features.subset_label == "identity_heavy_v1"

    nontrivial = compute_triviality_features("Add(Symbol('x'), Symbol('y'))")
    assert nontrivial.triviality_score == 0
    assert nontrivial.subset_label == "nontrivial_v1"


def test_operator_feature_enrichment_from_authoritative_srepr() -> None:
    row = build_stratified_egraph_row(
        raw_egraph_row(
            expression="log(x*y) + exp(x)",
            srepr="Add(log(Mul(Symbol('x'), Symbol('y'))), exp(Symbol('x')))",
            original_ast_nodes="7",
            original_eml_dag_nodes="31",
        )
    )

    assert row.features.count_Add == 1
    assert row.features.count_Mul == 1
    assert row.features.count_log == 1
    assert row.features.count_exp == 1
    assert row.features.contains_Add is True
    assert row.features.contains_Mul is True
    assert row.features.contains_log is True
    assert row.features.contains_exp is True
    assert row.operator_signature == "Add+Mul+exp+log"
    assert row.dominant_operator_family == "mixed_Add+Mul+exp+log"


def test_improvement_unchanged_worse_and_failure_rates() -> None:
    rows = [
        build_stratified_egraph_row(
            raw_egraph_row(index="0", original_eml_dag_nodes="10", extracted_eml_dag_nodes="8")
        ),
        build_stratified_egraph_row(
            raw_egraph_row(index="1", original_eml_dag_nodes="10", extracted_eml_dag_nodes="10")
        ),
        build_stratified_egraph_row(
            raw_egraph_row(index="2", original_eml_dag_nodes="10", extracted_eml_dag_nodes="12")
        ),
        build_stratified_egraph_row(
            raw_egraph_row(
                index="3",
                extraction_status="timeout",
                validation_status="error",
                timeout="True",
                extracted_eml_dag_nodes="",
                optimized_alpha="",
                gain="",
                below_after="",
            )
        ),
    ]

    summary = summarize_egraph_group(rows)

    assert summary["count"] == 4
    assert summary["success_count"] == 3
    assert summary["percent_improved"] == 100 / 3
    assert summary["percent_unchanged"] == 100 / 3
    assert summary["percent_worse"] == 100 / 3
    assert summary["timeout_rate"] == 25.0
    assert summary["validation_failure_rate"] == 25.0


def test_branch_sensitive_rate_and_subset_aggregates() -> None:
    rows = [
        build_stratified_egraph_row(
            raw_egraph_row(index="0", rule_mode="positive_real_formal", branch_sensitive="True")
        ),
        build_stratified_egraph_row(
            raw_egraph_row(
                index="1",
                srepr="Mul(Symbol('x'), Integer(1))",
                rule_mode="positive_real_formal",
                branch_sensitive="False",
            )
        ),
    ]

    all_summary = group_by_rule_mode(rows)
    positive_all = next(
        row
        for row in all_summary
        if row["rule_mode"] == "positive_real_formal" and row["subset_label"] == "all_v1"
    )
    subset_summary = group_by_subset_label(rows)
    subset_labels = {row["subset_label"] for row in subset_summary}

    assert positive_all["branch_sensitive_rule_usage_rate"] == 50.0
    assert {"all_v1", "nontrivial_v1", "identity_heavy_v1"} <= subset_labels


def test_triviality_effect_grouping_uses_measured_features() -> None:
    rows = [
        build_stratified_egraph_row(raw_egraph_row(index="0")),
        build_stratified_egraph_row(
            raw_egraph_row(index="1", srepr="log(Integer(1))", original_ast_nodes="2")
        ),
    ]

    summaries = group_by_triviality_feature(rows)
    log_one_true = next(
        row
        for row in summaries
        if row["subset_label"] == "all_v1"
        and row["triviality_feature"] == "has_log_one"
        and row["feature_value"] == "True"
    )

    assert log_one_true["count"] == 1
    assert log_one_true["success_count"] == 1


def test_stratified_egraph_compression_small_export(tmp_path: Path) -> None:
    safe_path = tmp_path / "outputs" / "v1" / "safe.csv"
    positive_path = tmp_path / "outputs" / "v1" / "positive.csv"
    baseline_path = tmp_path / "outputs" / "v1" / "dag.csv"
    dag_summary_path = tmp_path / "outputs" / "v1" / "dag_summary.json"
    generation_summary_path = tmp_path / "outputs" / "v1" / "generation_summary.json"

    safe_rows = [
        raw_egraph_row(index="0", srepr="Add(Symbol('x'), Symbol('y'))", rule_mode="safe"),
        raw_egraph_row(
            index="1",
            srepr="Mul(Symbol('x'), Integer(1))",
            rule_mode="safe",
            original_eml_dag_nodes="10",
            extracted_eml_dag_nodes="10",
            gain="1.0",
        ),
    ]
    positive_rows = [
        raw_egraph_row(
            index="0",
            srepr="Add(Symbol('x'), Symbol('y'))",
            rule_mode="positive_real_formal",
            branch_sensitive="True",
        ),
        raw_egraph_row(
            index="1",
            srepr="Mul(Symbol('x'), Integer(1))",
            rule_mode="positive_real_formal",
            branch_sensitive="True",
        ),
    ]
    write_egraph_rows(safe_path, safe_rows)
    write_egraph_rows(positive_path, positive_rows)
    write_baseline_rows(
        baseline_path,
        {
            0: "Add(Symbol('x'), Symbol('y'))",
            1: "Mul(Symbol('x'), Integer(1))",
        },
    )
    dag_summary_path.write_text(json.dumps({"processed_count": 2}), encoding="utf-8")
    generation_summary_path.write_text(json.dumps({"generated_count": 2}), encoding="utf-8")

    config = StratifiedEgraphCompressionConfig(
        safe_metrics_csv_path=safe_path,
        positive_real_metrics_csv_path=positive_path,
        goal3_metrics_csv_path=baseline_path,
        dag_summary_json_path=dag_summary_path,
        expression_generation_summary_json_path=generation_summary_path,
        alpha_by_operator_signature_csv_path=tmp_path / "outputs/v1/by_signature.csv",
        alpha_by_operator_family_csv_path=tmp_path / "outputs/v1/by_family.csv",
        alpha_by_size_bucket_csv_path=tmp_path / "outputs/v1/by_size.csv",
        alpha_by_rule_mode_csv_path=tmp_path / "outputs/v1/by_mode.csv",
        alpha_by_subset_label_csv_path=tmp_path / "outputs/v1/by_subset.csv",
        timeout_failure_summary_csv_path=tmp_path / "outputs/v1/timeout.csv",
        triviality_effect_summary_csv_path=tmp_path / "outputs/v1/triviality.csv",
    )

    result = run_stratified_egraph_compression_analysis(config)

    assert result.input_count == 4
    assert result.baseline_count == 2
    assert result.dag_summary_processed_count == 2
    assert result.generation_summary_count == 2
    for path in result.output_paths:
        assert path.exists()
        assert "outputs/v1" in path.as_posix()

    with config.alpha_by_subset_label_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        subset_rows = list(csv.DictReader(csv_file))
    assert {row["subset_label"] for row in subset_rows} == {
        "all_v1",
        "nontrivial_v1",
        "identity_heavy_v1",
    }


def write_egraph_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Write fake Goal 4.6 e-graph metric rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_baseline_rows(path: Path, sreprs_by_index: dict[int, str]) -> None:
    """Write fake Goal 3 baseline rows with authoritative srepr values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["index", "srepr"])
        writer.writeheader()
        for index, srepr in sorted(sreprs_by_index.items()):
            writer.writerow({"index": index, "srepr": srepr})
