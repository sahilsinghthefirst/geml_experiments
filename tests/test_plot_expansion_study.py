"""Smoke tests for Goal 2.4 expansion plotting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.plot_expansion_study import (
    PLOT_FILENAMES,
    ExpansionPlotConfig,
    run_expansion_plots,
)


def test_expansion_plot_smoke_writes_pngs_and_tables(tmp_path: Path) -> None:
    raw_metrics_path = tmp_path / "expansion_raw_metrics.csv"
    alpha_summary_path = tmp_path / "expansion_alpha_summary.json"
    ast_depth_path = tmp_path / "alpha_by_ast_depth.csv"
    operator_family_path = tmp_path / "alpha_by_operator_family.csv"
    plots_dir = tmp_path / "plots"

    write_fake_raw_metrics(raw_metrics_path)
    alpha_summary_path.write_text(
        json.dumps([{"scenario": "current_grammar", "alpha_threshold": 2.0}]),
        encoding="utf-8",
    )
    write_fake_group_csv(
        ast_depth_path,
        group_field="ast_depth",
        rows=[
            {"ast_depth": "1", "count": "1", "mean_alpha": "1.5", "median_alpha": "1.5"},
            {"ast_depth": "2", "count": "2", "mean_alpha": "5.0", "median_alpha": "5.0"},
        ],
    )
    write_fake_group_csv(
        operator_family_path,
        group_field="dominant_operator_family",
        rows=[
            {
                "dominant_operator_family": "exp",
                "count": "1",
                "mean_alpha": "1.5",
                "median_alpha": "1.5",
            },
            {
                "dominant_operator_family": "Add",
                "count": "2",
                "mean_alpha": "5.0",
                "median_alpha": "5.0",
            },
        ],
    )

    config = ExpansionPlotConfig(
        raw_metrics_csv_path=raw_metrics_path,
        alpha_summary_json_path=alpha_summary_path,
        alpha_by_ast_depth_csv_path=ast_depth_path,
        alpha_by_operator_family_csv_path=operator_family_path,
        plots_dir=plots_dir,
        top_alpha_csv_path=tmp_path / "top_20_alpha_expressions.csv",
        top_eml_node_csv_path=tmp_path / "top_20_eml_node_expressions.csv",
        top_eml_depth_csv_path=tmp_path / "top_20_eml_depth_expressions.csv",
    )

    result = run_expansion_plots(config)

    assert result.raw_metric_count == 3
    assert result.alpha_summary_count == 1
    assert {path.name for path in result.plot_paths} == set(PLOT_FILENAMES)
    for plot_path in result.plot_paths:
        assert plot_path.exists()
        assert plot_path.read_bytes().startswith(b"\x89PNG")
    for table_path in result.table_paths:
        assert table_path.exists()
        with table_path.open("r", encoding="utf-8", newline="") as csv_file:
            table_rows = list(csv.DictReader(csv_file))
        assert len(table_rows) == 3
        assert {"rank", "expression", "alpha", "eml_node_count", "eml_depth"} <= set(table_rows[0])


def write_fake_raw_metrics(path: Path) -> None:
    rows = [
        make_raw_row(
            index=0,
            expression="exp(x)",
            srepr="exp(Symbol('x'))",
            ast_node_count=2,
            ast_depth=1,
            eml_node_count=3,
            eml_depth=1,
            alpha=1.5,
            below_threshold=True,
        ),
        make_raw_row(
            index=1,
            expression="x + y",
            srepr="Add(Symbol('x'), Symbol('y'))",
            ast_node_count=3,
            ast_depth=1,
            eml_node_count=27,
            eml_depth=9,
            alpha=9.0,
            below_threshold=False,
        ),
        make_raw_row(
            index=2,
            expression="log(x)",
            srepr="log(Symbol('x'))",
            ast_node_count=2,
            ast_depth=2,
            eml_node_count=7,
            eml_depth=3,
            alpha=3.5,
            below_threshold=False,
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_raw_row(
    *,
    index: int,
    expression: str,
    srepr: str,
    ast_node_count: int,
    ast_depth: int,
    eml_node_count: int,
    eml_depth: int,
    alpha: float,
    below_threshold: bool,
) -> dict[str, str | int | float]:
    return {
        "index": index,
        "expression": expression,
        "srepr": srepr,
        "source_serialization": "srepr",
        "representation_mode": "restricted_eml_pure",
        "supported": "True",
        "error": "",
        "ast_node_count": ast_node_count,
        "ast_edge_count": ast_node_count - 1,
        "ast_depth": ast_depth,
        "ast_leaf_count": 1,
        "ast_operator_count": 1,
        "eml_node_count": eml_node_count,
        "eml_edge_count": eml_node_count - 1,
        "eml_depth": eml_depth,
        "eml_leaf_count": (eml_node_count + 1) // 2,
        "eml_normal_leaf_count": (eml_node_count + 1) // 2,
        "eml_derived_leaf_count": 0,
        "eml_hidden_compound_leaf_count": 0,
        "eml_operator_count": eml_node_count // 2,
        "alpha": alpha,
        "alpha_threshold": 2.0,
        "below_threshold": str(below_threshold),
        "alpha_valid": "True",
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
        "mean_alpha",
        "median_alpha",
        "p90_alpha",
        "p95_alpha",
        "max_alpha",
        "mean_ast_nodes",
        "mean_eml_nodes",
        "percent_below_threshold",
    ]
    normalized_rows = []
    for row in rows:
        normalized = {
            "p90_alpha": row["mean_alpha"],
            "p95_alpha": row["mean_alpha"],
            "max_alpha": row["mean_alpha"],
            "mean_ast_nodes": "2.0",
            "mean_eml_nodes": "10.0",
            "percent_below_threshold": "50.0",
            **row,
        }
        normalized_rows.append(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)
