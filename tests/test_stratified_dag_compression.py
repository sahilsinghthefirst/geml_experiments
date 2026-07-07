"""Tests for Goal 3.4 stratified DAG compression analysis."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from geml.experiments.stratified_dag_compression import (
    DagThresholdScenario,
    StratifiedDagCompressionConfig,
    build_stratified_dag_row,
    build_threshold_summaries,
    group_by_operator_family,
    group_by_operator_signature,
    run_stratified_dag_compression_analysis,
    summarize_dag_group,
)


def raw_row(
    *,
    expression: str = "x + y",
    srepr: str = "Add(Symbol('x'), Symbol('y'))",
    ast_nodes: str = "3",
    ast_depth: str = "1",
    tree_alpha: str = "9.0",
    dag_alpha_tree: str = "3.0",
    dag_alpha_dag: str = "3.0",
    eml_compression: str = "3.0",
) -> dict[str, str]:
    """Build a minimal raw DAG metric CSV row for tests."""
    return {
        "expression": expression,
        "srepr": srepr,
        "ast_tree_node_count": ast_nodes,
        "ast_dag_node_count": ast_nodes,
        "ast_tree_depth": ast_depth,
        "ast_dag_depth": ast_depth,
        "eml_tree_node_count": "27",
        "eml_dag_node_count": "9",
        "tree_alpha": tree_alpha,
        "dag_alpha_vs_ast_tree": dag_alpha_tree,
        "dag_alpha_vs_ast_dag": dag_alpha_dag,
        "eml_dag_compression": eml_compression,
    }


def test_threshold_scenario_math() -> None:
    rows = [
        build_stratified_dag_row(
            raw_row(tree_alpha="2.0", dag_alpha_tree="1.5", dag_alpha_dag="1.5")
        ),
        build_stratified_dag_row(
            raw_row(tree_alpha="3.0", dag_alpha_tree="2.0", dag_alpha_dag="2.0")
        ),
    ]

    summaries = build_threshold_summaries(
        rows,
        [DagThresholdScenario(name="current_grammar", k=4, l=3)],
    )

    threshold = 1 + (math.log(4) / math.log(12))
    assert summaries[0]["alpha_threshold"] == threshold
    assert summaries[0]["percent_below_tree_alpha"] == 0.0
    assert summaries[0]["percent_below_dag_alpha_vs_ast_tree"] == 50.0
    assert summaries[0]["percent_below_dag_alpha_vs_ast_dag"] == 50.0


def test_improvement_ratio_math() -> None:
    rows = [
        build_stratified_dag_row(raw_row(tree_alpha="10.0", dag_alpha_tree="2.0")),
        build_stratified_dag_row(raw_row(tree_alpha="9.0", dag_alpha_tree="3.0")),
    ]

    summary = summarize_dag_group(rows)

    assert summary["median_improvement"] == 4.0
    assert rows[0].improvement == 5.0
    assert rows[1].improvement == 3.0


def test_group_summaries() -> None:
    rows = [
        build_stratified_dag_row(
            raw_row(
                srepr="Add(Symbol('x'), Symbol('y'))",
                tree_alpha="9.0",
                dag_alpha_tree="3.0",
                dag_alpha_dag="3.0",
                eml_compression="3.0",
            )
        ),
        build_stratified_dag_row(
            raw_row(
                expression="x*y",
                srepr="Mul(Symbol('x'), Symbol('y'))",
                tree_alpha="13.0",
                dag_alpha_tree="4.0",
                dag_alpha_dag="4.0",
                eml_compression="3.25",
            )
        ),
    ]

    family_rows = group_by_operator_family(rows)
    family_by_name = {row["dominant_operator_family"]: row for row in family_rows}
    signature_rows = group_by_operator_signature(rows)
    signature_by_name = {row["operator_signature"]: row for row in signature_rows}

    assert family_by_name["Add"]["count"] == 1
    assert family_by_name["Mul"]["count"] == 1
    assert family_by_name["Add"]["median_tree_alpha"] == 9.0
    assert family_by_name["Mul"]["median_dag_alpha_vs_ast_tree"] == 4.0
    assert signature_by_name["Add"]["count"] == 1
    assert signature_by_name["Mul"]["count"] == 1


def test_operator_signature_extraction() -> None:
    row = build_stratified_dag_row(
        raw_row(
            expression="log(x*y + 1)",
            srepr=("log(Add(Mul(Symbol('x'), Symbol('y')), Integer(1)))"),
            ast_nodes="6",
            ast_depth="3",
        )
    )

    assert row.features.count_Add == 1
    assert row.features.count_Mul == 1
    assert row.features.count_log == 1
    assert row.features.contains_Add is True
    assert row.features.contains_Mul is True
    assert row.features.contains_log is True
    assert row.operator_signature == "Add+Mul+log"
    assert row.dominant_operator_family == "mixed_Add+Mul+log"


def test_stratified_dag_compression_small_export(tmp_path: Path) -> None:
    metrics_path = tmp_path / "dag_compression_metrics.csv"
    summary_path = tmp_path / "dag_compression_summary.json"
    write_fake_dag_metrics(metrics_path)
    summary_path.write_text(json.dumps({"processed_count": 2}), encoding="utf-8")

    config = StratifiedDagCompressionConfig(
        dag_metrics_csv_path=metrics_path,
        dag_summary_json_path=summary_path,
        dag_alpha_threshold_summary_csv_path=tmp_path / "dag_alpha_threshold_summary.csv",
        dag_alpha_threshold_summary_json_path=tmp_path / "dag_alpha_threshold_summary.json",
        dag_alpha_by_ast_size_bucket_csv_path=tmp_path / "dag_alpha_by_ast_size_bucket.csv",
        dag_alpha_by_ast_depth_csv_path=tmp_path / "dag_alpha_by_ast_depth.csv",
        dag_alpha_by_operator_family_csv_path=tmp_path / "dag_alpha_by_operator_family.csv",
        dag_alpha_by_operator_signature_csv_path=tmp_path / "dag_alpha_by_operator_signature.csv",
        dag_alpha_by_boolean_features_csv_path=tmp_path / "dag_alpha_by_boolean_features.csv",
    )

    result = run_stratified_dag_compression_analysis(config)

    assert result.input_count == 2
    assert result.summary_processed_count == 2
    assert result.threshold_summary_count == 3
    for path in result.output_paths:
        assert path.exists()

    threshold_rows = json.loads(
        config.dag_alpha_threshold_summary_json_path.read_text(encoding="utf-8")
    )
    assert {row["scenario"] for row in threshold_rows} == {
        "current_grammar",
        "generous_operator_vocab",
        "larger_operator_vocab",
    }

    with config.dag_alpha_by_operator_signature_csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        signature_rows = list(csv.DictReader(csv_file))
    assert {row["operator_signature"] for row in signature_rows} == {"Add", "Mul"}


def write_fake_dag_metrics(path: Path) -> None:
    """Write a tiny supported pure EML DAG metric CSV."""
    fieldnames = [
        "expression",
        "srepr",
        "supported",
        "pure_eml_valid",
        "ast_tree_node_count",
        "ast_dag_node_count",
        "ast_tree_depth",
        "ast_dag_depth",
        "eml_tree_node_count",
        "eml_dag_node_count",
        "tree_alpha",
        "dag_alpha_vs_ast_tree",
        "dag_alpha_vs_ast_dag",
        "eml_dag_compression",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                **raw_row(
                    srepr="Add(Symbol('x'), Symbol('y'))",
                    tree_alpha="9.0",
                    dag_alpha_tree="3.0",
                    dag_alpha_dag="3.0",
                    eml_compression="3.0",
                ),
                "supported": "True",
                "pure_eml_valid": "True",
            }
        )
        writer.writerow(
            {
                **raw_row(
                    expression="x*y",
                    srepr="Mul(Symbol('x'), Symbol('y'))",
                    tree_alpha="13.0",
                    dag_alpha_tree="4.0",
                    dag_alpha_dag="4.0",
                    eml_compression="3.25",
                ),
                "supported": "True",
                "pure_eml_valid": "True",
            }
        )
