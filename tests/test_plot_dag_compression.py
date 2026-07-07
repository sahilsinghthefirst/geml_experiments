"""Smoke tests for Goal 3.5 DAG compression plotting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.plot_dag_compression import (
    GOAL3_PLOT_FILENAMES,
    DagCompressionPlotConfig,
    run_dag_compression_plots,
)


def test_dag_compression_plot_smoke_writes_pngs(tmp_path: Path) -> None:
    metrics_path = tmp_path / "dag_compression_metrics.csv"
    threshold_path = tmp_path / "dag_alpha_threshold_summary.json"
    operator_family_path = tmp_path / "dag_alpha_by_operator_family.csv"
    ast_size_bucket_path = tmp_path / "dag_alpha_by_ast_size_bucket.csv"
    plots_dir = tmp_path / "plots_goal3"

    write_fake_dag_metrics(metrics_path)
    threshold_path.write_text(
        json.dumps(
            [
                {
                    "scenario": "current_grammar",
                    "alpha_threshold": 1.55,
                    "percent_below_tree_alpha": 0.0,
                    "percent_below_dag_alpha_vs_ast_tree": 33.3,
                    "percent_below_dag_alpha_vs_ast_dag": 33.3,
                },
                {
                    "scenario": "generous_operator_vocab",
                    "alpha_threshold": 2.2,
                    "percent_below_tree_alpha": 0.0,
                    "percent_below_dag_alpha_vs_ast_tree": 66.7,
                    "percent_below_dag_alpha_vs_ast_dag": 33.3,
                },
            ]
        ),
        encoding="utf-8",
    )
    write_fake_group_csv(
        operator_family_path,
        group_field="dominant_operator_family",
        rows=[
            {
                "dominant_operator_family": "exp",
                "count": "1",
                "median_dag_alpha_vs_ast_tree": "1.5",
                "median_eml_dag_compression": "2.0",
                "median_improvement": "2.0",
            },
            {
                "dominant_operator_family": "Add",
                "count": "2",
                "median_dag_alpha_vs_ast_tree": "4.0",
                "median_eml_dag_compression": "3.5",
                "median_improvement": "3.0",
            },
        ],
    )
    write_fake_group_csv(
        ast_size_bucket_path,
        group_field="ast_nodes_bucket",
        rows=[
            {
                "ast_nodes_bucket": "4-7",
                "count": "1",
                "median_dag_alpha_vs_ast_tree": "1.5",
                "median_eml_dag_compression": "2.0",
                "median_improvement": "2.0",
            },
            {
                "ast_nodes_bucket": "8-15",
                "count": "2",
                "median_dag_alpha_vs_ast_tree": "4.0",
                "median_eml_dag_compression": "3.5",
                "median_improvement": "3.0",
            },
        ],
    )

    config = DagCompressionPlotConfig(
        dag_metrics_csv_path=metrics_path,
        dag_threshold_summary_json_path=threshold_path,
        dag_operator_family_csv_path=operator_family_path,
        dag_ast_size_bucket_csv_path=ast_size_bucket_path,
        plots_dir=plots_dir,
    )

    result = run_dag_compression_plots(config)

    assert result.dag_metric_count == 3
    assert result.threshold_summary_count == 2
    assert {path.name for path in result.plot_paths} == set(GOAL3_PLOT_FILENAMES)
    for plot_path in result.plot_paths:
        assert plot_path.exists()
        assert plot_path.read_bytes().startswith(b"\x89PNG")


def write_fake_dag_metrics(path: Path) -> None:
    rows = [
        make_dag_row(
            index=0,
            expression="exp(x)",
            srepr="exp(Symbol('x'))",
            ast_tree_nodes=2,
            ast_dag_nodes=2,
            eml_tree_nodes=3,
            eml_dag_nodes=3,
            tree_alpha=1.5,
            dag_alpha_tree=1.5,
            dag_alpha_dag=1.5,
            eml_compression=1.0,
        ),
        make_dag_row(
            index=1,
            expression="(x + 1)*(x + 1)",
            srepr=("Mul(Add(Symbol('x'), Integer(1)), Add(Symbol('x'), Integer(1)))"),
            ast_tree_nodes=7,
            ast_dag_nodes=4,
            eml_tree_nodes=70,
            eml_dag_nodes=20,
            tree_alpha=10.0,
            dag_alpha_tree=2.857,
            dag_alpha_dag=5.0,
            eml_compression=3.5,
        ),
        make_dag_row(
            index=2,
            expression="x + y",
            srepr="Add(Symbol('x'), Symbol('y'))",
            ast_tree_nodes=3,
            ast_dag_nodes=3,
            eml_tree_nodes=27,
            eml_dag_nodes=9,
            tree_alpha=9.0,
            dag_alpha_tree=3.0,
            dag_alpha_dag=3.0,
            eml_compression=3.0,
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_dag_row(
    *,
    index: int,
    expression: str,
    srepr: str,
    ast_tree_nodes: int,
    ast_dag_nodes: int,
    eml_tree_nodes: int,
    eml_dag_nodes: int,
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
        "ast_tree_node_count": ast_tree_nodes,
        "ast_dag_node_count": ast_dag_nodes,
        "ast_dag_child_ref_count": ast_dag_nodes - 1,
        "ast_tree_depth": 2,
        "ast_dag_depth": 2,
        "ast_dag_compression": ast_tree_nodes / ast_dag_nodes,
        "eml_tree_node_count": eml_tree_nodes,
        "eml_dag_node_count": eml_dag_nodes,
        "eml_dag_child_ref_count": eml_dag_nodes - 1,
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


def write_fake_group_csv(
    path: Path,
    *,
    group_field: str,
    rows: list[dict[str, str]],
) -> None:
    fieldnames = [
        group_field,
        "count",
        "median_tree_alpha",
        "median_dag_alpha_vs_ast_tree",
        "median_dag_alpha_vs_ast_dag",
        "median_eml_dag_compression",
        "p90_eml_dag_compression",
        "percent_below_threshold_after_dag",
        "percent_below_threshold_dag_vs_ast_tree",
        "percent_below_threshold_dag_vs_ast_dag",
        "median_improvement",
    ]
    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "median_tree_alpha": "9.0",
                "median_dag_alpha_vs_ast_dag": row["median_dag_alpha_vs_ast_tree"],
                "p90_eml_dag_compression": row["median_eml_dag_compression"],
                "percent_below_threshold_after_dag": "50.0",
                "percent_below_threshold_dag_vs_ast_tree": "50.0",
                "percent_below_threshold_dag_vs_ast_dag": "50.0",
                **row,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)
