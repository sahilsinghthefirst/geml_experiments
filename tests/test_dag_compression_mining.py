"""Tests for Goal 3.5 DAG compression mining."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.dag_compression_mining import (
    DagCompressionMiningConfig,
    DagMetricMiningRow,
    DagSignatureMiningRow,
    run_dag_compression_mining,
    select_top_failures,
    select_top_successes,
)


def test_top_n_success_and_failure_selection() -> None:
    rows = [
        make_mining_row(
            index=0,
            expression="success",
            tree_alpha=10.0,
            dag_alpha_tree=2.0,
            dag_alpha_dag=2.0,
            eml_compression=5.0,
        ),
        make_mining_row(
            index=1,
            expression="moderate",
            tree_alpha=12.0,
            dag_alpha_tree=4.0,
            dag_alpha_dag=5.0,
            eml_compression=2.0,
        ),
        make_mining_row(
            index=2,
            expression="failure",
            tree_alpha=9.0,
            dag_alpha_tree=7.0,
            dag_alpha_dag=8.0,
            eml_compression=1.1,
        ),
    ]

    successes = select_top_successes(rows, limit=2)
    failures = select_top_failures(rows, limit=2)

    assert [row.expression for row in successes] == ["success", "moderate"]
    assert successes[0].success_score == 40.0
    assert [row.expression for row in failures] == ["failure", "moderate"]
    assert failures[0].failure_score == 15.9


def test_dag_compression_mining_report_generation(tmp_path: Path) -> None:
    metrics_path = tmp_path / "dag_compression_metrics.csv"
    signature_path = tmp_path / "dag_alpha_by_operator_signature.csv"
    threshold_path = tmp_path / "dag_alpha_threshold_summary.json"
    write_fake_dag_metrics(metrics_path)
    write_fake_signature_metrics(signature_path)
    threshold_path.write_text(
        json.dumps(
            [
                {
                    "scenario": "current_grammar",
                    "alpha_threshold": 1.55,
                    "percent_below_tree_alpha": 0.0,
                    "percent_below_dag_alpha_vs_ast_tree": 33.3,
                    "percent_below_dag_alpha_vs_ast_dag": 33.3,
                }
            ]
        ),
        encoding="utf-8",
    )

    config = DagCompressionMiningConfig(
        dag_metrics_csv_path=metrics_path,
        dag_operator_signature_csv_path=signature_path,
        dag_threshold_summary_json_path=threshold_path,
        top_successes_csv_path=tmp_path / "top_dag_compression_successes.csv",
        top_failures_csv_path=tmp_path / "top_dag_compression_failures.csv",
        best_operator_signatures_csv_path=tmp_path / "best_dag_operator_signatures.csv",
        worst_operator_signatures_csv_path=tmp_path / "worst_dag_operator_signatures.csv",
        safe_regime_candidates_csv_path=tmp_path / "dag_safe_regime_candidates.csv",
        report_md_path=tmp_path / "GOAL3_DAG_COMPRESSION_FINDINGS.md",
        top_n=2,
    )

    result = run_dag_compression_mining(config)

    assert result.dag_metric_count == 3
    assert result.operator_signature_count == 3
    for output_path in result.output_paths:
        assert output_path.exists()

    with config.top_successes_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        success_rows = list(csv.DictReader(csv_file))
    assert len(success_rows) == 2
    assert {
        "expression",
        "srepr",
        "tree_alpha",
        "dag_alpha_vs_ast_tree",
        "dag_alpha_vs_ast_dag",
        "eml_dag_compression",
        "ast_dag_compression",
        "ast_tree_node_count",
        "ast_dag_node_count",
        "eml_tree_node_count",
        "eml_dag_node_count",
        "operator_signature",
    } <= set(success_rows[0])

    report = config.report_md_path.read_text(encoding="utf-8")
    assert "Goal 3 DAG Compression Findings" in report
    assert "structural representation findings only" in report
    assert "not model-performance evidence" in report
    assert "Top DAG Compression Successes" in report


def make_mining_row(
    *,
    index: int,
    expression: str,
    tree_alpha: float,
    dag_alpha_tree: float,
    dag_alpha_dag: float,
    eml_compression: float,
) -> DagMetricMiningRow:
    return DagMetricMiningRow(
        index=index,
        expression=expression,
        srepr="Add(Symbol('x'), Symbol('y'))",
        tree_alpha=tree_alpha,
        dag_alpha_vs_ast_tree=dag_alpha_tree,
        dag_alpha_vs_ast_dag=dag_alpha_dag,
        eml_dag_compression=eml_compression,
        ast_dag_compression=1.0,
        ast_tree_node_count=3,
        ast_dag_node_count=3,
        eml_tree_node_count=30,
        eml_dag_node_count=10,
        operator_signature="Add",
    )


def make_signature_row(
    signature: str,
    *,
    median_dag_alpha: float,
    median_compression: float,
    percent_below: float,
) -> DagSignatureMiningRow:
    return DagSignatureMiningRow(
        operator_signature=signature,
        count=10,
        median_tree_alpha=median_dag_alpha * median_compression,
        median_dag_alpha_vs_ast_tree=median_dag_alpha,
        median_dag_alpha_vs_ast_dag=median_dag_alpha,
        median_eml_dag_compression=median_compression,
        p90_eml_dag_compression=median_compression,
        percent_below_threshold_after_dag=percent_below,
        percent_below_threshold_dag_vs_ast_tree=percent_below,
        percent_below_threshold_dag_vs_ast_dag=percent_below,
        median_improvement=median_compression,
    )


def write_fake_dag_metrics(path: Path) -> None:
    rows = [
        make_raw_dag_row(
            index=0,
            expression="exp(x)",
            srepr="exp(Symbol('x'))",
            tree_alpha=1.5,
            dag_alpha_tree=1.5,
            dag_alpha_dag=1.5,
            eml_compression=1.0,
        ),
        make_raw_dag_row(
            index=1,
            expression="(x + 1)*(x + 1)",
            srepr=("Mul(Add(Symbol('x'), Integer(1)), Add(Symbol('x'), Integer(1)))"),
            tree_alpha=10.0,
            dag_alpha_tree=2.5,
            dag_alpha_dag=5.0,
            eml_compression=4.0,
        ),
        make_raw_dag_row(
            index=2,
            expression="x*y",
            srepr="Mul(Symbol('x'), Symbol('y'))",
            tree_alpha=13.0,
            dag_alpha_tree=7.0,
            dag_alpha_dag=7.0,
            eml_compression=1.2,
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_raw_dag_row(
    *,
    index: int,
    expression: str,
    srepr: str,
    tree_alpha: float,
    dag_alpha_tree: float,
    dag_alpha_dag: float,
    eml_compression: float,
) -> dict[str, str | int | float]:
    return {
        "index": index,
        "expression": expression,
        "srepr": srepr,
        "source_serialization": "srepr",
        "supported": "True",
        "ast_tree_node_count": 6,
        "ast_dag_node_count": 4,
        "ast_dag_child_ref_count": 5,
        "ast_tree_depth": 2,
        "ast_dag_depth": 2,
        "ast_dag_compression": 1.5,
        "eml_tree_node_count": int(tree_alpha * 6),
        "eml_dag_node_count": int(dag_alpha_tree * 6),
        "eml_dag_child_ref_count": 20,
        "eml_tree_depth": 4,
        "eml_dag_depth": 4,
        "eml_dag_compression": eml_compression,
        "tree_alpha": tree_alpha,
        "dag_alpha_vs_ast_tree": dag_alpha_tree,
        "dag_alpha_vs_ast_dag": dag_alpha_dag,
        "alpha_threshold_current": 1.55,
        "below_threshold_tree": str(tree_alpha < 1.55),
        "below_threshold_dag_vs_ast_tree": str(dag_alpha_tree < 1.55),
        "below_threshold_dag_vs_ast_dag": str(dag_alpha_dag < 1.55),
        "pure_eml_valid": "True",
        "derived_leaf_count": 0,
        "hidden_compound_leaf_count": 0,
        "error": "",
    }


def write_fake_signature_metrics(path: Path) -> None:
    rows = [
        signature_to_csv_row(
            make_signature_row(
                "exp",
                median_dag_alpha=1.5,
                median_compression=1.2,
                percent_below=100,
            )
        ),
        signature_to_csv_row(
            make_signature_row("Add", median_dag_alpha=3.0, median_compression=3.0, percent_below=0)
        ),
        signature_to_csv_row(
            make_signature_row("Mul", median_dag_alpha=7.0, median_compression=1.2, percent_below=0)
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def signature_to_csv_row(row: DagSignatureMiningRow) -> dict[str, str | int | float]:
    return {
        "operator_signature": row.operator_signature,
        "count": row.count,
        "median_tree_alpha": row.median_tree_alpha,
        "median_dag_alpha_vs_ast_tree": row.median_dag_alpha_vs_ast_tree,
        "median_dag_alpha_vs_ast_dag": row.median_dag_alpha_vs_ast_dag,
        "median_eml_dag_compression": row.median_eml_dag_compression,
        "p90_eml_dag_compression": row.p90_eml_dag_compression,
        "percent_below_threshold_after_dag": row.percent_below_threshold_after_dag,
        "percent_below_threshold_dag_vs_ast_tree": (row.percent_below_threshold_dag_vs_ast_tree),
        "percent_below_threshold_dag_vs_ast_dag": row.percent_below_threshold_dag_vs_ast_dag,
        "median_improvement": row.median_improvement,
    }
