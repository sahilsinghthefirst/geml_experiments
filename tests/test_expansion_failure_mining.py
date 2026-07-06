"""Tests for Goal 2.5 expansion failure mining."""

from __future__ import annotations

import csv
from pathlib import Path

from geml.experiments.expansion_failure_mining import (
    FailureMiningConfig,
    GroupMetricRow,
    MetricRow,
    rank_safest_operator_signatures,
    rank_worst_operator_signatures,
    run_failure_mining,
    select_top_rows,
)


def test_top_n_selection_uses_metric_and_index_tie_breaker() -> None:
    rows = [
        make_metric_row(index=2, alpha=4.0, eml_nodes=40, eml_depth=8),
        make_metric_row(index=1, alpha=4.0, eml_nodes=30, eml_depth=7),
        make_metric_row(index=0, alpha=2.0, eml_nodes=50, eml_depth=9),
    ]

    top_alpha = select_top_rows(rows, key=lambda row: row.alpha, limit=2)
    top_nodes = select_top_rows(rows, key=lambda row: row.eml_node_count, limit=2)

    assert [row.index for row in top_alpha] == [1, 2]
    assert [row.index for row in top_nodes] == [0, 2]


def test_safe_and_worst_signature_ranking() -> None:
    rows = [
        make_group_row("exp", median=1.8, p90=1.8, percent_below=10.0),
        make_group_row("Add+Mul", median=16.0, p90=17.0, percent_below=0.0),
        make_group_row("Add+Mul+log", median=15.0, p90=18.0, percent_below=0.0),
    ]

    worst = rank_worst_operator_signatures(rows, limit=2)
    safest = rank_safest_operator_signatures(rows, threshold=1.55, limit=2)

    assert [row["operator_signature"] for row in worst] == ["Add+Mul", "Add+Mul+log"]
    assert worst[0]["median_alpha_rank"] == 1
    assert worst[1]["p90_alpha_rank"] == 1
    assert safest[0]["operator_signature"] == "exp"
    assert safest[0]["median_threshold_gap"] == 0.25


def test_failure_report_generation_on_tiny_fake_data(tmp_path: Path) -> None:
    raw_metrics_path = tmp_path / "expansion_raw_metrics.csv"
    operator_family_path = tmp_path / "alpha_by_operator_family.csv"
    operator_signature_path = tmp_path / "alpha_by_operator_signature.csv"
    ast_depth_path = tmp_path / "alpha_by_ast_depth.csv"
    write_fake_raw_metrics(raw_metrics_path)
    write_group_metrics(operator_family_path, "dominant_operator_family")
    write_group_metrics(operator_signature_path, "operator_signature")
    write_depth_metrics(ast_depth_path)

    config = FailureMiningConfig(
        raw_metrics_csv_path=raw_metrics_path,
        alpha_by_operator_family_csv_path=operator_family_path,
        alpha_by_operator_signature_csv_path=operator_signature_path,
        alpha_by_ast_depth_csv_path=ast_depth_path,
        top_alpha_csv_path=tmp_path / "top_alpha_explosions.csv",
        top_eml_node_csv_path=tmp_path / "top_eml_node_explosions.csv",
        top_eml_depth_csv_path=tmp_path / "top_eml_depth_explosions.csv",
        worst_operator_signatures_csv_path=tmp_path / "worst_operator_signatures.csv",
        safest_operator_signatures_csv_path=tmp_path / "safest_operator_signatures.csv",
        depth_failure_modes_csv_path=tmp_path / "depth_failure_modes.csv",
        safe_eml_regime_candidates_csv_path=tmp_path / "safe_eml_regime_candidates.csv",
        report_md_path=tmp_path / "GOAL2_FAILURE_CASES.md",
        top_n=2,
        snippet_max_chars=80,
    )

    result = run_failure_mining(config)

    assert result.raw_metric_count == 3
    for output_path in result.output_paths:
        assert output_path.exists()
    report = config.report_md_path.read_text(encoding="utf-8")
    assert "Highest-Alpha Examples" in report
    assert "Common Structural Causes" in report
    assert "structural evidence" in report
    with config.top_alpha_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        top_rows = list(csv.DictReader(csv_file))
    assert len(top_rows) == 2
    assert top_rows[0]["official_eml_snippet"]
    assert top_rows[0]["official_eml_truncated"] in {"True", "False"}


def make_metric_row(
    *,
    index: int,
    alpha: float,
    eml_nodes: int,
    eml_depth: int,
) -> MetricRow:
    return MetricRow(
        index=index,
        expression="x + y",
        srepr="Add(Symbol('x'), Symbol('y'))",
        ast_node_count=3,
        ast_depth=1,
        ast_operator_count=1,
        ast_leaf_count=2,
        eml_node_count=eml_nodes,
        eml_depth=eml_depth,
        eml_operator_count=eml_nodes // 2,
        eml_leaf_count=(eml_nodes + 1) // 2,
        alpha=alpha,
        alpha_threshold=1.55,
        below_threshold=alpha < 1.55,
    )


def make_group_row(
    key: str,
    *,
    median: float,
    p90: float,
    percent_below: float,
) -> GroupMetricRow:
    return GroupMetricRow(
        key=key,
        count=10,
        mean_alpha=median,
        median_alpha=median,
        p90_alpha=p90,
        p95_alpha=p90,
        max_alpha=p90,
        mean_ast_nodes=5.0,
        mean_eml_nodes=5.0 * median,
        percent_below_threshold=percent_below,
    )


def write_fake_raw_metrics(path: Path) -> None:
    rows = [
        make_raw_row(
            index=0,
            expression="exp(x)",
            srepr="exp(Symbol('x'))",
            alpha=1.5,
            eml_nodes=3,
            eml_depth=1,
            below_threshold=True,
        ),
        make_raw_row(
            index=1,
            expression="x + y",
            srepr="Add(Symbol('x'), Symbol('y'))",
            alpha=9.0,
            eml_nodes=27,
            eml_depth=9,
            below_threshold=False,
        ),
        make_raw_row(
            index=2,
            expression="x*y",
            srepr="Mul(Symbol('x'), Symbol('y'))",
            alpha=13.0,
            eml_nodes=39,
            eml_depth=10,
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
    alpha: float,
    eml_nodes: int,
    eml_depth: int,
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
        "ast_node_count": 3,
        "ast_edge_count": 2,
        "ast_depth": 1,
        "ast_leaf_count": 2,
        "ast_operator_count": 1,
        "eml_node_count": eml_nodes,
        "eml_edge_count": eml_nodes - 1,
        "eml_depth": eml_depth,
        "eml_leaf_count": (eml_nodes + 1) // 2,
        "eml_normal_leaf_count": (eml_nodes + 1) // 2,
        "eml_derived_leaf_count": 0,
        "eml_hidden_compound_leaf_count": 0,
        "eml_operator_count": eml_nodes // 2,
        "alpha": alpha,
        "alpha_threshold": 1.55,
        "below_threshold": str(below_threshold),
        "alpha_valid": "True",
    }


def write_group_metrics(path: Path, key_field: str) -> None:
    rows = [
        make_group_csv_row(key_field, "exp", median=1.5, p90=1.5, percent_below=100.0),
        make_group_csv_row(key_field, "Add+Mul", median=13.0, p90=14.0, percent_below=0.0),
        make_group_csv_row(key_field, "Add+Mul+log", median=12.0, p90=15.0, percent_below=0.0),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_group_csv_row(
    key_field: str,
    key: str,
    *,
    median: float,
    p90: float,
    percent_below: float,
) -> dict[str, str | float | int]:
    return {
        key_field: key,
        "count": 10,
        "mean_alpha": median,
        "median_alpha": median,
        "p90_alpha": p90,
        "p95_alpha": p90,
        "max_alpha": p90,
        "mean_ast_nodes": 5.0,
        "mean_eml_nodes": 5.0 * median,
        "percent_below_threshold": percent_below,
    }


def write_depth_metrics(path: Path) -> None:
    rows = [
        make_group_csv_row("ast_depth", "1", median=3.0, p90=4.0, percent_below=10.0),
        make_group_csv_row("ast_depth", "2", median=8.0, p90=9.0, percent_below=0.0),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
