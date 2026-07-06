"""Tests for Goal 2.3 stratified expansion analysis."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import sympy as sp
from geml.experiments.stratified_expansion import (
    OperatorFeatures,
    StratifiedExpansionConfig,
    StratifiedExpressionRow,
    bucket_alpha,
    bucket_ast_depth,
    bucket_ast_nodes,
    bucket_eml_nodes,
    count_operator_features,
    dominant_operator_family,
    operator_signature,
    run_stratified_expansion_analysis,
    summarize_group,
)


def test_operator_counting_from_srepr() -> None:
    x, y = sp.symbols("x y")
    expr = sp.Mul(
        sp.Add(x, 1, evaluate=False),
        sp.log(sp.exp(y, evaluate=False), evaluate=False),
        evaluate=False,
    )

    features = count_operator_features(sp.srepr(expr))

    assert features.count_Add == 1
    assert features.count_Mul == 1
    assert features.count_Pow == 0
    assert features.count_exp == 1
    assert features.count_log == 1
    assert features.count_symbols == 2
    assert features.count_constants == 1
    assert features.contains_Add is True
    assert features.contains_Mul is True
    assert features.contains_exp is True
    assert features.contains_log is True


def test_bucket_assignment() -> None:
    assert bucket_ast_nodes(3) == "1-3"
    assert bucket_ast_nodes(4) == "4-7"
    assert bucket_ast_depth(1) == "0-1"
    assert bucket_ast_depth(5) == "5+"
    assert bucket_eml_nodes(63) == "32-63"
    assert bucket_eml_nodes(256) == "256+"
    assert bucket_alpha(1.5) == "0-<2"
    assert bucket_alpha(2.0) == "2-<5"
    assert bucket_alpha(15.0) == "15+"


def test_dominant_family_detection_and_operator_signature() -> None:
    leaf = OperatorFeatures(count_symbols=1)
    add_heavy = OperatorFeatures(count_Add=2, count_Mul=1)
    tied = OperatorFeatures(count_Add=1, count_Mul=1, count_log=1)

    assert dominant_operator_family(leaf) == "leaf_only"
    assert operator_signature(leaf) == "leaf_only"
    assert dominant_operator_family(add_heavy) == "Add"
    assert operator_signature(add_heavy) == "Add+Mul"
    assert dominant_operator_family(tied) == "mixed_Add+Mul+log"
    assert operator_signature(tied) == "Add+Mul+log"


def test_group_summary_math() -> None:
    rows = [
        make_summary_row(alpha=1.0, below_threshold=True, ast_nodes=3, eml_nodes=3),
        make_summary_row(alpha=3.0, below_threshold=True, ast_nodes=5, eml_nodes=15),
        make_summary_row(alpha=5.0, below_threshold=False, ast_nodes=7, eml_nodes=35),
    ]

    summary = summarize_group(rows)

    assert summary["count"] == 3
    assert summary["mean_alpha"] == 3.0
    assert summary["median_alpha"] == 3.0
    assert summary["p90_alpha"] == 5.0
    assert summary["p95_alpha"] == 5.0
    assert summary["max_alpha"] == 5.0
    assert summary["mean_ast_nodes"] == 5.0
    assert math.isclose(summary["mean_eml_nodes"], 53 / 3)
    assert math.isclose(summary["percent_below_threshold"], 200 / 3)


def test_stratified_analysis_writes_group_outputs(tmp_path: Path) -> None:
    raw_metrics_path = tmp_path / "expansion_raw_metrics.csv"
    alpha_summary_json_path = tmp_path / "expansion_alpha_summary.json"
    alpha_summary_csv_path = tmp_path / "expansion_alpha_summary.csv"
    write_small_raw_metrics(raw_metrics_path)
    alpha_summary = [
        {
            "scenario": "current_grammar",
            "k": 4,
            "l": 3,
            "alpha_threshold": 1.5578858913022597,
            "alpha_valid_count": 2,
        }
    ]
    alpha_summary_json_path.write_text(json.dumps(alpha_summary), encoding="utf-8")
    with alpha_summary_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["scenario", "k", "l"])
        writer.writeheader()
        writer.writerow({"scenario": "current_grammar", "k": 4, "l": 3})

    config = StratifiedExpansionConfig(
        raw_metrics_csv_path=raw_metrics_path,
        alpha_summary_json_path=alpha_summary_json_path,
        alpha_summary_csv_path=alpha_summary_csv_path,
        alpha_by_ast_depth_csv_path=tmp_path / "alpha_by_ast_depth.csv",
        alpha_by_ast_size_bucket_csv_path=tmp_path / "alpha_by_ast_size_bucket.csv",
        alpha_by_operator_family_csv_path=tmp_path / "alpha_by_operator_family.csv",
        alpha_by_operator_signature_csv_path=tmp_path / "alpha_by_operator_signature.csv",
        alpha_by_boolean_features_csv_path=tmp_path / "alpha_by_boolean_features.csv",
    )

    result = run_stratified_expansion_analysis(config)

    assert result.input_count == 2
    assert result.alpha_summary_json_count == 1
    assert result.alpha_summary_csv_count == 1
    for output_path in result.output_paths:
        assert output_path.exists()

    with config.alpha_by_operator_signature_csv_path.open(
        "r", encoding="utf-8", newline=""
    ) as csv_file:
        signature_rows = list(csv.DictReader(csv_file))

    assert {row["operator_signature"] for row in signature_rows} == {"Add", "exp"}


def make_summary_row(
    *,
    alpha: float,
    below_threshold: bool,
    ast_nodes: int,
    eml_nodes: int,
) -> StratifiedExpressionRow:
    return StratifiedExpressionRow(
        expression="x",
        srepr="Symbol('x')",
        ast_node_count=ast_nodes,
        ast_depth=1,
        ast_operator_count=0,
        ast_leaf_count=1,
        eml_node_count=eml_nodes,
        eml_depth=1,
        eml_operator_count=0,
        eml_leaf_count=1,
        alpha=alpha,
        alpha_threshold=2.0,
        below_threshold=below_threshold,
        features=OperatorFeatures(count_symbols=1),
        ast_nodes_bucket=bucket_ast_nodes(ast_nodes),
        ast_depth_bucket=bucket_ast_depth(1),
        eml_nodes_bucket=bucket_eml_nodes(eml_nodes),
        alpha_bucket=bucket_alpha(alpha),
        dominant_operator_family="leaf_only",
        operator_signature="leaf_only",
    )


def write_small_raw_metrics(path: Path) -> None:
    x, y = sp.symbols("x y")
    rows = [
        {
            "index": 0,
            "expression": "exp(x)",
            "srepr": sp.srepr(sp.exp(x, evaluate=False)),
            "source_serialization": "srepr",
            "representation_mode": "restricted_eml_pure",
            "supported": "True",
            "error": "",
            "ast_node_count": "2",
            "ast_edge_count": "1",
            "ast_depth": "1",
            "ast_leaf_count": "1",
            "ast_operator_count": "1",
            "eml_node_count": "3",
            "eml_edge_count": "2",
            "eml_depth": "1",
            "eml_leaf_count": "2",
            "eml_normal_leaf_count": "2",
            "eml_derived_leaf_count": "0",
            "eml_hidden_compound_leaf_count": "0",
            "eml_operator_count": "1",
            "alpha": "1.5",
            "alpha_threshold": "1.5578858913022597",
            "below_threshold": "True",
            "alpha_valid": "True",
        },
        {
            "index": 1,
            "expression": "x + y",
            "srepr": sp.srepr(sp.Add(x, y, evaluate=False)),
            "source_serialization": "srepr",
            "representation_mode": "restricted_eml_pure",
            "supported": "True",
            "error": "",
            "ast_node_count": "3",
            "ast_edge_count": "2",
            "ast_depth": "1",
            "ast_leaf_count": "2",
            "ast_operator_count": "1",
            "eml_node_count": "27",
            "eml_edge_count": "26",
            "eml_depth": "9",
            "eml_leaf_count": "14",
            "eml_normal_leaf_count": "14",
            "eml_derived_leaf_count": "0",
            "eml_hidden_compound_leaf_count": "0",
            "eml_operator_count": "13",
            "alpha": "9.0",
            "alpha_threshold": "1.5578858913022597",
            "below_threshold": "False",
            "alpha_valid": "True",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
