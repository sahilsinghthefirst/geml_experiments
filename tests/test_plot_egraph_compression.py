"""Smoke tests for Goal 4.8 e-graph compression plotting."""

from __future__ import annotations

import csv
from pathlib import Path

from geml.experiments.plot_egraph_compression import (
    GOAL4_PLOT_FILENAMES,
    EgraphCompressionPlotConfig,
    run_egraph_compression_plots,
)


def test_egraph_compression_plot_smoke_writes_pngs(tmp_path: Path) -> None:
    safe_path = tmp_path / "outputs/v1/egraph_compression_metrics_safe.csv"
    positive_path = tmp_path / "outputs/v1/egraph_compression_metrics_positive_real.csv"
    signature_path = tmp_path / "outputs/v1/egraph_alpha_by_operator_signature.csv"
    family_path = tmp_path / "outputs/v1/egraph_alpha_by_operator_family.csv"
    subset_path = tmp_path / "outputs/v1/egraph_alpha_by_subset_label.csv"
    plots_dir = tmp_path / "outputs/v1/plots_goal4"

    write_metric_rows(safe_path, "safe", branch_sensitive=False)
    write_metric_rows(positive_path, "positive_real_formal", branch_sensitive=True)
    write_signature_rows(signature_path)
    write_family_rows(family_path)
    write_subset_rows(subset_path)

    config = EgraphCompressionPlotConfig(
        safe_metrics_csv_path=safe_path,
        positive_real_metrics_csv_path=positive_path,
        operator_signature_csv_path=signature_path,
        operator_family_csv_path=family_path,
        subset_label_csv_path=subset_path,
        plots_dir=plots_dir,
    )

    result = run_egraph_compression_plots(config)

    assert result.safe_metric_count == 3
    assert result.positive_real_metric_count == 3
    assert {path.name for path in result.plot_paths} == set(GOAL4_PLOT_FILENAMES)
    for plot_path in result.plot_paths:
        assert plot_path.exists()
        assert plot_path.read_bytes().startswith(b"\x89PNG")


def write_metric_rows(path: Path, rule_mode: str, *, branch_sensitive: bool) -> None:
    """Write fake per-expression Goal 4.6 rows."""
    rows = [
        metric_row(0, rule_mode, 3.0, 2.0, 1.5, "nontrivial_v1", branch_sensitive),
        metric_row(1, rule_mode, 4.0, 4.0, 1.0, "identity_heavy_v1", branch_sensitive),
        metric_row(2, rule_mode, 5.0, 6.0, 0.8333333333, "identity_heavy_v1", branch_sensitive),
    ]
    write_rows(path, rows)


def metric_row(
    index: int,
    rule_mode: str,
    goal3_alpha: float,
    optimized_alpha: float,
    gain: float,
    subset_label: str,
    branch_sensitive: bool,
) -> dict[str, str]:
    """Build a fake per-expression e-graph metric row."""
    return {
        "index": str(index),
        "rule_mode": rule_mode,
        "validation_status": "valid",
        "extraction_status": "completed",
        "timeout": "False",
        "enode_count": str(20 + index),
        "goal3_dag_alpha_vs_ast_tree": str(goal3_alpha),
        "optimized_dag_alpha_vs_ast_tree": str(optimized_alpha),
        "compression_gain_vs_goal3_dag": str(gain),
        "subset_label": subset_label,
        "structural_purity_valid": "True",
        "branch_sensitive_rules_used": str(branch_sensitive),
    }


def write_signature_rows(path: Path) -> None:
    """Write fake operator-signature summary rows."""
    write_rows(
        path,
        [
            summary_row("safe", "all_v1", operator_signature="Add"),
            summary_row("positive_real_formal", "all_v1", operator_signature="Add"),
        ],
    )


def write_family_rows(path: Path) -> None:
    """Write fake operator-family summary rows."""
    rows = []
    for rule_mode in ("safe", "positive_real_formal"):
        for subset_label in ("all_v1", "nontrivial_v1", "identity_heavy_v1"):
            rows.append(
                {
                    **summary_row(rule_mode, subset_label),
                    "dominant_operator_family": "Add",
                    "contains_Add": "True",
                    "contains_Mul": "False",
                    "contains_log": "False",
                    "contains_exp": "False",
                }
            )
    write_rows(path, rows)


def write_subset_rows(path: Path) -> None:
    """Write fake subset-label summary rows."""
    rows = []
    for rule_mode in ("safe", "positive_real_formal"):
        for subset_label in ("all_v1", "nontrivial_v1", "identity_heavy_v1"):
            rows.append(summary_row(rule_mode, subset_label))
    write_rows(path, rows)


def summary_row(
    rule_mode: str,
    subset_label: str,
    *,
    operator_signature: str = "Add",
) -> dict[str, str]:
    """Build a fake Goal 4.7 summary row."""
    return {
        "rule_mode": rule_mode,
        "subset_label": subset_label,
        "operator_signature": operator_signature,
        "count": "3",
        "success_count": "3",
        "median_goal3_dag_alpha_vs_ast_tree": "4.0",
        "median_optimized_dag_alpha_vs_ast_tree": "2.5",
        "median_compression_gain_vs_goal3_dag": "1.4",
        "p90_compression_gain_vs_goal3_dag": "2.0",
        "percent_improved": "66.7",
        "percent_unchanged": "33.3",
        "percent_worse": "0.0",
        "percent_below_threshold_before": "0.0",
        "percent_below_threshold_after": "10.0",
        "timeout_rate": "0.0",
        "validation_failure_rate": "0.0",
        "branch_sensitive_rule_usage_rate": "100.0"
        if rule_mode == "positive_real_formal"
        else "0.0",
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Write CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
