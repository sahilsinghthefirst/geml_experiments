"""Tests for Goal 4.8 e-graph compression mining."""

from __future__ import annotations

import csv
from pathlib import Path

from geml.experiments.egraph_compression_mining import (
    EgraphCompressionMiningConfig,
    EgraphMetricMiningRow,
    EgraphSignatureMiningRow,
    rank_best_operator_signatures,
    rank_safe_regime_candidates,
    run_egraph_compression_mining,
    select_subset_successes,
    select_top_failures,
    select_top_successes,
)


def test_top_n_success_and_failure_selection() -> None:
    rows = [
        make_metric_row(
            index=0,
            expression="best",
            original_nodes=20,
            extracted_nodes=5,
            gain=4.0,
            optimized_alpha=1.0,
            below_before=False,
            below_after=True,
        ),
        make_metric_row(
            index=1,
            expression="moderate",
            original_nodes=20,
            extracted_nodes=10,
            gain=2.0,
            optimized_alpha=2.0,
        ),
        make_metric_row(
            index=2,
            expression="timeout",
            original_nodes=20,
            extracted_nodes=None,
            gain=None,
            optimized_alpha=None,
            timeout=True,
            validation_status="error",
        ),
    ]

    successes = select_top_successes(rows, limit=2)
    failures = select_top_failures(rows, limit=2)

    assert [row.original_expression for row in successes] == ["best", "moderate"]
    assert successes[0].threshold_status_improved is True
    assert [row.original_expression for row in failures] == ["timeout", "moderate"]


def test_subset_specific_mining() -> None:
    rows = [
        make_metric_row(index=0, expression="nontrivial", subset_label="nontrivial_v1", gain=2.0),
        make_metric_row(
            index=1,
            expression="identity",
            subset_label="identity_heavy_v1",
            gain=5.0,
        ),
    ]

    nontrivial = select_subset_successes(rows, subset_label="nontrivial_v1", limit=5)
    identity = select_subset_successes(rows, subset_label="identity_heavy_v1", limit=5)

    assert [row.original_expression for row in nontrivial] == ["nontrivial"]
    assert [row.original_expression for row in identity] == ["identity"]


def test_signature_rankings() -> None:
    rows = [
        make_signature_row("Add", rule_mode="safe", gain=1.2, percent_improved=20.0),
        make_signature_row(
            "log",
            rule_mode="positive_real_formal",
            gain=3.0,
            percent_improved=90.0,
        ),
        make_signature_row(
            "leaf_only",
            rule_mode="safe",
            subset_label="nontrivial_v1",
            gain=1.1,
            percent_improved=10.0,
            percent_below_after=80.0,
            timeout_rate=0.0,
            validation_failure_rate=0.0,
        ),
    ]

    best = rank_best_operator_signatures(rows, limit=2)
    safe_candidates = rank_safe_regime_candidates(rows, limit=2)

    assert best[0]["operator_signature"] == "log"
    assert safe_candidates[0]["operator_signature"] == "leaf_only"
    assert safe_candidates[0]["candidate_kind"] == "operator_signature"


def test_egraph_compression_mining_report_generation(tmp_path: Path) -> None:
    safe_path = tmp_path / "outputs/v1/egraph_compression_metrics_safe.csv"
    positive_path = tmp_path / "outputs/v1/egraph_compression_metrics_positive_real.csv"
    signature_path = tmp_path / "outputs/v1/egraph_alpha_by_operator_signature.csv"
    family_path = tmp_path / "outputs/v1/egraph_alpha_by_operator_family.csv"
    subset_path = tmp_path / "outputs/v1/egraph_alpha_by_subset_label.csv"

    write_metric_rows(safe_path, "safe", branch_sensitive=False)
    write_metric_rows(positive_path, "positive_real_formal", branch_sensitive=True)
    write_signature_rows(signature_path)
    write_generic_summary_rows(family_path)
    write_generic_summary_rows(subset_path)

    config = EgraphCompressionMiningConfig(
        safe_metrics_csv_path=safe_path,
        positive_real_metrics_csv_path=positive_path,
        operator_signature_csv_path=signature_path,
        operator_family_csv_path=family_path,
        subset_label_csv_path=subset_path,
        top_successes_safe_csv_path=tmp_path / "outputs/v1/top_safe.csv",
        top_successes_positive_real_csv_path=tmp_path / "outputs/v1/top_positive.csv",
        top_failures_safe_csv_path=tmp_path / "outputs/v1/fail_safe.csv",
        top_failures_positive_real_csv_path=tmp_path / "outputs/v1/fail_positive.csv",
        best_operator_signatures_csv_path=tmp_path / "outputs/v1/best.csv",
        worst_operator_signatures_csv_path=tmp_path / "outputs/v1/worst.csv",
        safe_regime_candidates_csv_path=tmp_path / "outputs/v1/safe_regimes.csv",
        nontrivial_successes_csv_path=tmp_path / "outputs/v1/nontrivial.csv",
        identity_heavy_successes_csv_path=tmp_path / "outputs/v1/identity.csv",
        report_md_path=tmp_path / "outputs/v1/GOAL4_EGRAPH_COMPRESSION_FINDINGS.md",
        top_n=2,
    )

    result = run_egraph_compression_mining(config)

    assert result.safe_metric_count == 3
    assert result.positive_real_metric_count == 3
    for output_path in result.output_paths:
        assert output_path.exists()
        assert "outputs/v1" in output_path.as_posix()

    with config.top_successes_safe_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        success_rows = list(csv.DictReader(csv_file))
    assert len(success_rows) == 2
    assert {"rule_mode", "subset_label", "compression_gain_vs_goal3_dag"} <= set(success_rows[0])

    report = config.report_md_path.read_text(encoding="utf-8")
    assert "v1 corpus" in report
    assert "v0 corpus is pilot only" in report
    assert "non-ML compression" in report
    assert "branch-sensitive positive-real assumptions" in report
    assert "not GNN evidence" in report
    assert "official pure EML" in report
    assert "nontrivial_v1" in report
    assert "identity_heavy_v1" in report


def make_metric_row(
    *,
    index: int,
    expression: str,
    original_nodes: int = 20,
    extracted_nodes: int | None = 10,
    gain: float | None = 2.0,
    optimized_alpha: float | None = 2.0,
    rule_mode: str = "safe",
    subset_label: str = "nontrivial_v1",
    timeout: bool = False,
    validation_status: str = "valid",
    below_before: bool = False,
    below_after: bool | None = False,
) -> EgraphMetricMiningRow:
    """Build a mining row for ranking tests."""
    return EgraphMetricMiningRow(
        index=index,
        original_expression=expression,
        original_srepr="Add(Symbol('x'), Symbol('y'))",
        rule_mode=rule_mode,
        saturation_status="saturated" if not timeout else "timeout",
        extraction_status="completed" if not timeout else "timeout",
        validation_status=validation_status,
        timeout=timeout,
        branch_sensitive_rules_used=rule_mode == "positive_real_formal",
        branch_sensitive_rule_names="[]",
        original_eml_dag_nodes=original_nodes,
        extracted_eml_dag_nodes=extracted_nodes,
        goal3_dag_alpha_vs_ast_tree=4.0,
        optimized_dag_alpha_vs_ast_tree=optimized_alpha,
        compression_gain_vs_goal3_dag=gain,
        below_threshold_goal3_dag=below_before,
        below_threshold_optimized_dag=below_after,
        subset_label=subset_label,
        structural_purity_valid=True,
        eclass_count=5,
        enode_count=10,
        total_rules_applied=3,
        validation_error=None,
        error=None,
    )


def make_signature_row(
    signature: str,
    *,
    rule_mode: str,
    subset_label: str = "all_v1",
    gain: float,
    percent_improved: float,
    percent_below_after: float = 10.0,
    timeout_rate: float = 0.0,
    validation_failure_rate: float = 0.0,
) -> EgraphSignatureMiningRow:
    """Build a fake signature mining row."""
    return EgraphSignatureMiningRow(
        rule_mode=rule_mode,
        subset_label=subset_label,
        operator_signature=signature,
        count=10,
        success_count=9,
        median_goal3_dag_alpha_vs_ast_tree=4.0,
        median_optimized_dag_alpha_vs_ast_tree=2.0,
        median_compression_gain_vs_goal3_dag=gain,
        p90_compression_gain_vs_goal3_dag=gain,
        percent_improved=percent_improved,
        percent_unchanged=100.0 - percent_improved,
        percent_worse=0.0,
        percent_below_threshold_before=0.0,
        percent_below_threshold_after=percent_below_after,
        timeout_rate=timeout_rate,
        validation_failure_rate=validation_failure_rate,
        branch_sensitive_rule_usage_rate=100.0 if rule_mode == "positive_real_formal" else 0.0,
    )


def write_metric_rows(path: Path, rule_mode: str, *, branch_sensitive: bool) -> None:
    """Write fake per-expression Goal 4.6 rows."""
    rows = [
        csv_metric_row(0, rule_mode, "x+y", 20, 10, 2.0, "nontrivial_v1", branch_sensitive),
        csv_metric_row(1, rule_mode, "x*1", 30, 6, 5.0, "identity_heavy_v1", branch_sensitive),
        csv_metric_row(
            2,
            rule_mode,
            "timeout",
            30,
            None,
            None,
            "identity_heavy_v1",
            branch_sensitive,
            timeout=True,
            validation_status="error",
        ),
    ]
    write_rows(path, rows)


def csv_metric_row(
    index: int,
    rule_mode: str,
    expression: str,
    original_nodes: int,
    extracted_nodes: int | None,
    gain: float | None,
    subset_label: str,
    branch_sensitive: bool,
    *,
    timeout: bool = False,
    validation_status: str = "valid",
) -> dict[str, str]:
    """Build a fake per-expression CSV row."""
    return {
        "index": str(index),
        "original_expression": expression,
        "original_srepr": "Add(Symbol('x'), Symbol('y'))",
        "rule_mode": rule_mode,
        "saturation_status": "timeout" if timeout else "saturated",
        "extraction_status": "timeout" if timeout else "completed",
        "validation_status": validation_status,
        "timeout": str(timeout),
        "branch_sensitive_rules_used": str(branch_sensitive),
        "branch_sensitive_rule_names": "[]",
        "original_eml_dag_nodes": str(original_nodes),
        "extracted_eml_dag_nodes": "" if extracted_nodes is None else str(extracted_nodes),
        "goal3_dag_alpha_vs_ast_tree": "4.0",
        "optimized_dag_alpha_vs_ast_tree": "" if extracted_nodes is None else "2.0",
        "compression_gain_vs_goal3_dag": "" if gain is None else str(gain),
        "below_threshold_goal3_dag": "False",
        "below_threshold_optimized_dag": "False" if extracted_nodes is not None else "",
        "subset_label": subset_label,
        "structural_purity_valid": "True",
        "eclass_count": "" if timeout else "4",
        "enode_count": "" if timeout else "9",
        "total_rules_applied": "" if timeout else "2",
        "validation_error": "timeout" if timeout else "",
        "error": "timeout" if timeout else "",
    }


def write_signature_rows(path: Path) -> None:
    """Write fake operator-signature summary rows."""
    rows = [
        signature_to_csv_row(
            make_signature_row("Add", rule_mode="safe", gain=2.0, percent_improved=60.0)
        ),
        signature_to_csv_row(
            make_signature_row(
                "log",
                rule_mode="positive_real_formal",
                gain=4.0,
                percent_improved=90.0,
            )
        ),
        signature_to_csv_row(
            make_signature_row(
                "leaf_only",
                rule_mode="safe",
                subset_label="nontrivial_v1",
                gain=1.5,
                percent_improved=25.0,
                percent_below_after=75.0,
            )
        ),
    ]
    write_rows(path, rows)


def write_generic_summary_rows(path: Path) -> None:
    """Write a non-empty generic summary CSV for mining input checks."""
    write_rows(
        path,
        [
            signature_to_csv_row(
                make_signature_row("Add", rule_mode="safe", gain=2.0, percent_improved=60.0)
            )
        ],
    )


def signature_to_csv_row(row: EgraphSignatureMiningRow) -> dict[str, str]:
    """Serialize a signature row to fake CSV fields."""
    return {
        "rule_mode": row.rule_mode,
        "subset_label": row.subset_label,
        "operator_signature": row.operator_signature,
        "count": str(row.count),
        "success_count": str(row.success_count),
        "median_goal3_dag_alpha_vs_ast_tree": str(row.median_goal3_dag_alpha_vs_ast_tree),
        "median_optimized_dag_alpha_vs_ast_tree": str(row.median_optimized_dag_alpha_vs_ast_tree),
        "median_compression_gain_vs_goal3_dag": str(row.median_compression_gain_vs_goal3_dag),
        "p90_compression_gain_vs_goal3_dag": str(row.p90_compression_gain_vs_goal3_dag),
        "percent_improved": str(row.percent_improved),
        "percent_unchanged": str(row.percent_unchanged),
        "percent_worse": str(row.percent_worse),
        "percent_below_threshold_before": str(row.percent_below_threshold_before),
        "percent_below_threshold_after": str(row.percent_below_threshold_after),
        "timeout_rate": str(row.timeout_rate),
        "validation_failure_rate": str(row.validation_failure_rate),
        "branch_sensitive_rule_usage_rate": str(row.branch_sensitive_rule_usage_rate),
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Write CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
