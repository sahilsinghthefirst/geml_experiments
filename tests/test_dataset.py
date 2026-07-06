"""Tests for dataset metrics JSONL/CSV export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import sympy as sp
from geml.data.dataset import DatasetExportConfig, export_dataset_metrics


def write_input_jsonl(path: Path) -> None:
    x, y = sp.symbols("x y")
    expressions = [
        sp.Add(x, 1, evaluate=False),
        sp.sin(x),
        sp.Pow(x, 2, evaluate=False),
        sp.Mul(sp.Add(x, 1, evaluate=False), sp.Add(y, 1, evaluate=False), evaluate=False),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for index, expr in enumerate(expressions):
            row = {
                "index": index,
                "expression": str(expr),
                "srepr": sp.srepr(expr),
                "metadata": {"test_case": True},
            }
            jsonl_file.write(json.dumps(row))
            jsonl_file.write("\n")


def test_dataset_metrics_jsonl_schema(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_jsonl_path = tmp_path / "metrics.jsonl"
    output_csv_path = tmp_path / "metrics.csv"
    write_input_jsonl(input_path)

    rows = export_dataset_metrics(
        DatasetExportConfig(
            input_jsonl_path=input_path,
            output_jsonl_path=output_jsonl_path,
            output_csv_path=output_csv_path,
        )
    )

    jsonl_rows = [
        json.loads(line)
        for line in output_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert len(rows) == 4
    assert len(jsonl_rows) == 4
    assert {
        "index",
        "expression",
        "srepr",
        "ast_stats",
        "eml_stats",
        "alpha",
        "supported",
        "error",
        "metadata",
    } <= set(jsonl_rows[0])
    assert jsonl_rows[0]["supported"] is True
    assert jsonl_rows[0]["ast_stats"]["node_count"] == 3
    assert jsonl_rows[0]["eml_stats"]["node_count"] == 3
    assert jsonl_rows[0]["alpha"] == 1.0
    assert jsonl_rows[1]["supported"] is False
    assert jsonl_rows[1]["error"]
    assert jsonl_rows[1]["ast_stats"] is None
    assert jsonl_rows[1]["eml_stats"] is None
    assert jsonl_rows[1]["alpha"] is None
    assert jsonl_rows[2]["supported"] is False
    assert jsonl_rows[2]["ast_stats"]["node_count"] == 3
    assert jsonl_rows[2]["eml_stats"] is None
    assert jsonl_rows[2]["alpha"] is None


def test_dataset_metrics_csv_summary_schema(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_jsonl_path = tmp_path / "metrics.jsonl"
    output_csv_path = tmp_path / "metrics.csv"
    write_input_jsonl(input_path)

    export_dataset_metrics(
        DatasetExportConfig(
            input_jsonl_path=input_path,
            output_jsonl_path=output_jsonl_path,
            output_csv_path=output_csv_path,
        )
    )

    with output_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))

    assert len(csv_rows) == 4
    assert {
        "index",
        "expression",
        "srepr",
        "supported",
        "error",
        "ast_node_count",
        "ast_edge_count",
        "ast_depth",
        "ast_leaf_count",
        "ast_operator_count",
        "eml_node_count",
        "eml_edge_count",
        "eml_depth",
        "eml_leaf_count",
        "eml_operator_count",
        "alpha",
    } <= set(csv_rows[0])
    assert csv_rows[0]["supported"] == "True"
    assert csv_rows[0]["ast_node_count"] == "3"
    assert csv_rows[0]["eml_node_count"] == "3"
    assert csv_rows[0]["alpha"] == "1.0"
    assert csv_rows[1]["supported"] == "False"
    assert csv_rows[1]["error"]
    assert csv_rows[1]["alpha"] == ""
    assert csv_rows[2]["supported"] == "False"
    assert csv_rows[2]["ast_node_count"] == "3"
    assert csv_rows[2]["eml_node_count"] == ""
    assert csv_rows[2]["alpha"] == ""
