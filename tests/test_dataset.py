"""Tests for dataset metrics JSONL/CSV export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import sympy as sp
from geml.data.dataset import (
    DatasetExportConfig,
    GeneratedExpressionInput,
    compute_metrics_rows,
    export_dataset_metrics,
)


def write_input_jsonl(path: Path) -> None:
    x, y = sp.symbols("x y")
    expressions = [
        sp.exp(x, evaluate=False),
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


def test_dataset_metrics_jsonl_schema_for_pure_mode(tmp_path: Path) -> None:
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

    assert len(rows) == 5
    assert len(jsonl_rows) == 5
    assert {
        "index",
        "expression",
        "srepr",
        "source_serialization",
        "representation_mode",
        "ast_stats",
        "eml_stats",
        "eml_normal_leaf_count",
        "eml_derived_leaf_count",
        "eml_hidden_compound_leaf_count",
        "alpha",
        "alpha_valid",
        "supported",
        "error",
        "metadata",
    } <= set(jsonl_rows[0])

    assert jsonl_rows[0]["source_serialization"] == "srepr"
    assert jsonl_rows[0]["representation_mode"] == "restricted_eml_pure"
    assert jsonl_rows[0]["supported"] is True
    assert jsonl_rows[0]["alpha_valid"] is True
    assert jsonl_rows[0]["ast_stats"]["node_count"] == 2
    assert jsonl_rows[0]["eml_stats"]["node_count"] == 3
    assert jsonl_rows[0]["eml_normal_leaf_count"] == 2
    assert jsonl_rows[0]["eml_derived_leaf_count"] == 0
    assert jsonl_rows[0]["eml_hidden_compound_leaf_count"] == 0
    assert jsonl_rows[0]["alpha"] == 1.5

    assert jsonl_rows[1]["supported"] is True
    assert jsonl_rows[1]["alpha_valid"] is True
    assert jsonl_rows[1]["ast_stats"]["node_count"] == 3
    assert jsonl_rows[1]["eml_stats"]["node_count"] > jsonl_rows[1]["ast_stats"]["node_count"]
    assert jsonl_rows[1]["alpha"] is not None
    assert jsonl_rows[1]["error"] is None

    assert jsonl_rows[2]["supported"] is False
    assert jsonl_rows[2]["error"]
    assert jsonl_rows[2]["ast_stats"] is None
    assert jsonl_rows[2]["eml_stats"] is None
    assert jsonl_rows[2]["alpha"] is None

    assert jsonl_rows[3]["supported"] is True
    assert jsonl_rows[3]["alpha_valid"] is True
    assert jsonl_rows[3]["ast_stats"]["node_count"] == 3
    assert jsonl_rows[3]["eml_stats"]["node_count"] > jsonl_rows[3]["ast_stats"]["node_count"]
    assert jsonl_rows[3]["alpha"] is not None

    assert jsonl_rows[4]["supported"] is True
    assert jsonl_rows[4]["alpha_valid"] is True
    assert jsonl_rows[4]["eml_stats"]["node_count"] > jsonl_rows[4]["ast_stats"]["node_count"]


def test_dataset_metrics_csv_summary_schema_for_pure_mode(tmp_path: Path) -> None:
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

    assert len(csv_rows) == 5
    assert {
        "index",
        "expression",
        "srepr",
        "source_serialization",
        "representation_mode",
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
        "eml_normal_leaf_count",
        "eml_derived_leaf_count",
        "eml_hidden_compound_leaf_count",
        "eml_operator_count",
        "alpha",
        "alpha_valid",
    } <= set(csv_rows[0])
    assert csv_rows[0]["source_serialization"] == "srepr"
    assert csv_rows[0]["representation_mode"] == "restricted_eml_pure"
    assert csv_rows[0]["supported"] == "True"
    assert csv_rows[0]["alpha_valid"] == "True"
    assert csv_rows[0]["ast_node_count"] == "2"
    assert csv_rows[0]["eml_node_count"] == "3"
    assert csv_rows[0]["eml_normal_leaf_count"] == "2"
    assert csv_rows[0]["eml_derived_leaf_count"] == "0"
    assert csv_rows[0]["eml_hidden_compound_leaf_count"] == "0"
    assert csv_rows[0]["alpha"] == "1.5"
    assert csv_rows[1]["supported"] == "True"
    assert csv_rows[1]["error"] == ""
    assert csv_rows[1]["alpha"] != ""
    assert csv_rows[1]["alpha_valid"] == "True"
    assert csv_rows[2]["supported"] == "False"
    assert csv_rows[2]["alpha"] == ""
    assert csv_rows[3]["supported"] == "True"
    assert csv_rows[3]["ast_node_count"] == "3"
    assert csv_rows[3]["eml_node_count"] != ""
    assert csv_rows[3]["alpha"] != ""
    assert csv_rows[4]["supported"] == "True"
    assert csv_rows[4]["alpha_valid"] == "True"


def test_dataset_derived_mode_keeps_hidden_leaves_out_of_alpha() -> None:
    x = sp.Symbol("x")
    rows = compute_metrics_rows(
        [
            GeneratedExpressionInput(
                index=0,
                expression=str(sp.Add(x, 1, evaluate=False)),
                srepr=None,
            )
        ],
        representation_mode="restricted_eml_with_derived",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.representation_mode == "restricted_eml_with_derived"
    assert row.supported is True
    assert row.eml_stats is not None
    assert row.eml_stats.leaf_count == 2
    assert row.eml_normal_leaf_count == 1
    assert row.eml_derived_leaf_count == 1
    assert row.eml_hidden_compound_leaf_count == 1
    assert row.alpha is None
    assert row.alpha_valid is False


def test_dataset_metrics_use_srepr_as_authoritative_structure() -> None:
    x = sp.Symbol("x")
    rows = compute_metrics_rows(
        [
            GeneratedExpressionInput(
                index=0,
                expression=str(x),
                srepr=sp.srepr(sp.Add(x, 1, evaluate=False)),
            )
        ],
        representation_mode="ast",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.source_serialization == "srepr"
    assert row.supported is True
    assert row.ast_stats is not None
    assert row.ast_stats.node_count == 3
    assert row.srepr == "Add(Symbol('x'), Integer(1))"


def test_dataset_metrics_parse_rational_srepr() -> None:
    rows = compute_metrics_rows(
        [
            GeneratedExpressionInput(
                index=0,
                expression="0.5",
                srepr=sp.srepr(sp.Rational(1, 2)),
            )
        ],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.source_serialization == "srepr"
    assert row.supported is True
    assert row.alpha_valid is True
    assert row.srepr == "Rational(1, 2)"
