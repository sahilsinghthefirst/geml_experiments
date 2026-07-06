"""Goal 2 expansion-factor scale pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import sympy as sp
import yaml
from pydantic import BaseModel, Field, model_validator

from geml.data.dataset import (
    DatasetMetricsRow,
    compute_metrics_rows,
    load_generated_expressions,
    write_metrics_csv,
    write_metrics_jsonl,
)
from geml.data.generate_exprs import (
    DEFAULT_OPERATOR_PROBABILITIES,
    DEFAULT_SYMBOL_NAMES,
    ExpressionGeneratorConfig,
    SympyExpressionGenerator,
    write_jsonl,
)
from geml.symbolic.eml_transpile import sympy_to_eml_tree
from geml.symbolic.official_eml_compiler import emit_official_eml_string
from geml.symbolic.representations import REPRESENTATION_MODES, RepresentationMode


class ThresholdScenario(BaseModel):
    """Alpha-threshold scenario parameters."""

    name: str
    k: int = Field(gt=0)
    ell: int = Field(gt=0, alias="l")


class ExpansionStudyConfig(BaseModel):
    """Configuration for Goal 2.1 expression generation and raw metrics export."""

    seed: int = 0
    count: int = Field(default=10_000, gt=0)
    max_depth: int = Field(default=4, ge=0)
    output_dir: Path = Path("outputs/v0")
    input_jsonl_path: Path = Path("outputs/v0/expansion_inputs.jsonl")
    raw_metrics_jsonl_path: Path = Path("outputs/v0/expansion_raw_metrics.jsonl")
    raw_metrics_csv_path: Path = Path("outputs/v0/expansion_raw_metrics.csv")
    summary_json_path: Path = Path("outputs/v0/official_eml_compiler_summary.json")
    alpha_summary_csv_path: Path = Path("outputs/v0/expansion_alpha_summary.csv")
    alpha_summary_json_path: Path = Path("outputs/v0/expansion_alpha_summary.json")
    top_alpha_json_path: Path = Path("outputs/v0/official_eml_top20_alpha.json")
    top_depth_json_path: Path = Path("outputs/v0/official_eml_top20_depth.json")
    simple_examples_json_path: Path = Path("outputs/v0/official_eml_simple_examples.json")
    alpha_threshold_k: int = Field(default=4, gt=0)
    alpha_threshold_l: int = Field(default=3, gt=0)
    threshold_scenarios: tuple[ThresholdScenario, ...] = Field(
        default_factory=lambda: (
            ThresholdScenario(name="current_grammar", k=4, l=3),
            ThresholdScenario(name="generous_operator_vocab", k=20, l=3),
            ThresholdScenario(name="larger_operator_vocab", k=50, l=3),
        )
    )
    representation_mode: RepresentationMode = "restricted_eml_pure"
    operator_probabilities: dict[str, float] = Field(
        default_factory=lambda: DEFAULT_OPERATOR_PROBABILITIES.copy()
    )
    symbol_names: tuple[str, ...] = DEFAULT_SYMBOL_NAMES

    @model_validator(mode="after")
    def validate_symbol_names(self) -> Self:
        if not self.symbol_names:
            raise ValueError("symbol_names must not be empty")
        return self


def run_expansion_study(config: ExpansionStudyConfig) -> list[DatasetMetricsRow]:
    """Generate expressions and export raw AST/EML metrics for Goal 2."""
    generator_config = ExpressionGeneratorConfig(
        seed=config.seed,
        count=config.count,
        max_depth=config.max_depth,
        operator_probabilities=config.operator_probabilities,
        symbol_names=config.symbol_names,
    )
    records = SympyExpressionGenerator(generator_config).generate()
    write_jsonl(records, config.input_jsonl_path)

    input_rows = load_generated_expressions(config.input_jsonl_path)
    metrics_rows = compute_metrics_rows(
        input_rows,
        symbol_names=config.symbol_names,
        representation_mode=config.representation_mode,
    )
    primary_threshold = compute_alpha_threshold(config.alpha_threshold_k, config.alpha_threshold_l)
    annotate_alpha_thresholds(metrics_rows, alpha_threshold=primary_threshold)
    write_metrics_jsonl(metrics_rows, config.raw_metrics_jsonl_path)
    write_metrics_csv(metrics_rows, config.raw_metrics_csv_path)
    write_alpha_threshold_summaries(metrics_rows, config)
    write_official_compiler_summary(metrics_rows, config.summary_json_path)
    write_official_compiler_audit_exports(metrics_rows, config)
    return metrics_rows


def compute_alpha_threshold(k: int, ell: int) -> float:
    """Compute alpha threshold 1 + log(K) / log(4L)."""
    if k <= 0:
        raise ValueError("k must be positive")
    if ell <= 0:
        raise ValueError("ell must be positive")
    return 1 + (math.log(k) / math.log(4 * ell))


def annotate_alpha_thresholds(
    rows: Sequence[DatasetMetricsRow],
    *,
    alpha_threshold: float,
) -> None:
    """Attach the primary threshold and below-threshold classification to rows."""
    for row in rows:
        row.alpha_threshold = alpha_threshold
        row.below_threshold = (
            row.alpha is not None and row.alpha_valid and row.alpha < alpha_threshold
        )


def write_alpha_threshold_summaries(
    rows: Sequence[DatasetMetricsRow],
    config: ExpansionStudyConfig,
) -> None:
    """Write aggregate alpha-threshold summaries for configured scenarios."""
    summaries = [
        build_alpha_threshold_summary(rows, scenario=scenario)
        for scenario in config.threshold_scenarios
    ]
    _write_json(config.alpha_summary_json_path, summaries)
    _write_alpha_summary_csv(summaries, config.alpha_summary_csv_path)


def build_alpha_threshold_summary(
    rows: Sequence[DatasetMetricsRow],
    *,
    scenario: ThresholdScenario,
) -> dict[str, object]:
    """Build aggregate alpha-threshold summary for one K/L scenario."""
    alpha_rows = [row for row in rows if row.alpha is not None and row.alpha_valid]
    alphas = sorted(row.alpha for row in alpha_rows if row.alpha is not None)
    alpha_threshold = compute_alpha_threshold(scenario.k, scenario.ell)
    below_count = sum(1 for alpha in alphas if alpha < alpha_threshold)
    above_count = len(alphas) - below_count

    return {
        "scenario": scenario.name,
        "k": scenario.k,
        "l": scenario.ell,
        "alpha_threshold": alpha_threshold,
        "alpha_valid_count": len(alphas),
        "mean_alpha": statistics.fmean(alphas) if alphas else None,
        "median_alpha": statistics.median(alphas) if alphas else None,
        "p90_alpha": _percentile(alphas, 0.9) if alphas else None,
        "p95_alpha": _percentile(alphas, 0.95) if alphas else None,
        "max_alpha": max(alphas) if alphas else None,
        "below_threshold_count": below_count,
        "above_threshold_count": above_count,
        "percent_below_threshold": (100 * below_count / len(alphas)) if alphas else None,
        "percent_above_threshold": (100 * above_count / len(alphas)) if alphas else None,
    }


def _write_alpha_summary_csv(summaries: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "k",
        "l",
        "alpha_threshold",
        "alpha_valid_count",
        "mean_alpha",
        "median_alpha",
        "p90_alpha",
        "p95_alpha",
        "max_alpha",
        "below_threshold_count",
        "above_threshold_count",
        "percent_below_threshold",
        "percent_above_threshold",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_official_compiler_summary(rows: Sequence[DatasetMetricsRow], path: Path) -> None:
    """Write summary statistics for the official pure EML compiler run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_official_compiler_summary(rows)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def write_official_compiler_audit_exports(
    rows: Sequence[DatasetMetricsRow],
    config: ExpansionStudyConfig,
) -> None:
    """Write Goal 2.1c official compiler audit exports."""
    alpha_rows = top_alpha_rows(rows, limit=20)
    depth_rows = top_depth_rows(rows, limit=20)
    simple_examples = build_simple_expression_audit()

    _write_json(config.top_alpha_json_path, [metric_row_to_audit_dict(row) for row in alpha_rows])
    _write_json(config.top_depth_json_path, [metric_row_to_audit_dict(row) for row in depth_rows])
    _write_json(config.simple_examples_json_path, simple_examples)


def build_official_compiler_summary(rows: Sequence[DatasetMetricsRow]) -> dict[str, object]:
    """Build alpha and support-count summary data for a raw expansion run."""
    alpha_rows = [row for row in rows if row.alpha is not None and row.alpha_valid]
    alphas = sorted(row.alpha for row in alpha_rows if row.alpha is not None)
    supported_count = sum(1 for row in rows if row.supported)

    top_alpha = top_alpha_rows(rows, limit=20)
    top_depth = top_depth_rows(rows, limit=20)

    return {
        "processed_count": len(rows),
        "official_pure_eml_supported_count": supported_count,
        "unsupported_count": len(rows) - supported_count,
        "alpha_valid_count": len(alphas),
        "mean_alpha": statistics.fmean(alphas) if alphas else None,
        "median_alpha": statistics.median(alphas) if alphas else None,
        "p90_alpha": _percentile(alphas, 0.9) if alphas else None,
        "p95_alpha": _percentile(alphas, 0.95) if alphas else None,
        "max_alpha": max(alphas) if alphas else None,
        "top_20_largest_alpha_expressions": [metric_row_to_audit_dict(row) for row in top_alpha],
        "top_20_deepest_eml_expressions": [metric_row_to_audit_dict(row) for row in top_depth],
        "simple_expression_audit": build_simple_expression_audit(),
    }


def top_alpha_rows(
    rows: Sequence[DatasetMetricsRow],
    *,
    limit: int,
) -> list[DatasetMetricsRow]:
    """Return rows with the largest valid alpha values."""
    alpha_rows = [row for row in rows if row.alpha is not None and row.alpha_valid]
    return sorted(
        alpha_rows,
        key=lambda row: row.alpha if row.alpha is not None else float("-inf"),
        reverse=True,
    )[:limit]


def top_depth_rows(
    rows: Sequence[DatasetMetricsRow],
    *,
    limit: int,
) -> list[DatasetMetricsRow]:
    """Return rows with the deepest EML trees."""
    depth_rows = [row for row in rows if row.eml_stats is not None]
    return sorted(
        depth_rows,
        key=lambda row: (
            row.eml_stats.depth if row.eml_stats is not None else -1,
            row.alpha if row.alpha is not None else float("-inf"),
        ),
        reverse=True,
    )[:limit]


def metric_row_to_audit_dict(row: DatasetMetricsRow) -> dict[str, object]:
    """Serialize a metric row for human audit outputs."""
    return {
        "index": row.index,
        "expression": row.expression,
        "srepr": row.srepr,
        "alpha": row.alpha,
        "alpha_valid": row.alpha_valid,
        "supported": row.supported,
        "ast_node_count": row.ast_stats.node_count if row.ast_stats else None,
        "ast_depth": row.ast_stats.depth if row.ast_stats else None,
        "eml_node_count": row.eml_stats.node_count if row.eml_stats else None,
        "eml_depth": row.eml_stats.depth if row.eml_stats else None,
        "eml_leaf_count": row.eml_stats.leaf_count if row.eml_stats else None,
        "eml_derived_leaf_count": row.eml_derived_leaf_count,
        "eml_hidden_compound_leaf_count": row.eml_hidden_compound_leaf_count,
    }


def build_simple_expression_audit() -> list[dict[str, object]]:
    """Build exact simple-expression audit records for the official pure compiler."""
    x, y = sp.symbols("x y")
    examples = [
        ("x+y", sp.Add(x, y, evaluate=False)),
        ("x*y", sp.Mul(x, y, evaluate=False)),
        ("log(x)", sp.log(x, evaluate=False)),
        ("exp(x)", sp.exp(x, evaluate=False)),
        ("x**2", sp.Pow(x, 2, evaluate=False)),
    ]
    records: list[dict[str, object]] = []
    for name, expr in examples:
        tree = sympy_to_eml_tree(expr, representation_mode="restricted_eml_pure")
        records.append(
            {
                "name": name,
                "expression": str(expr),
                "srepr": sp.srepr(expr),
                "ast_node_count": tree.ast_statistics.node_count,
                "eml_node_count": tree.statistics.node_count,
                "eml_depth": tree.statistics.depth,
                "alpha": tree.alpha,
                "official_eml": emit_official_eml_string(tree),
                "derived_leaf_count": tree.derived_leaf_count,
                "hidden_compound_leaf_count": tree.hidden_compound_leaf_count,
                "has_derived_leaves": tree.derived_leaf_count > 0,
            }
        )
    return records


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    index = max(0, min(len(sorted_values) - 1, math.ceil(percentile * len(sorted_values)) - 1))
    return sorted_values[index]


def load_config(path: Path) -> ExpansionStudyConfig:
    """Load a YAML Goal 2.1 expansion-study config."""
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    return ExpansionStudyConfig.model_validate(raw_config)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/expansion_v0.yaml"),
        help="Path to a YAML Goal 2.1 expansion config.",
    )
    parser.add_argument("--count", type=int, default=None, help="Optional expression count.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--representation-mode",
        choices=REPRESENTATION_MODES,
        default=None,
        help="Representation mode used for raw metrics.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 2.1 expansion-factor scale pipeline."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.count is not None:
        config.count = args.count
    if args.seed is not None:
        config.seed = args.seed
    if args.representation_mode is not None:
        config.representation_mode = args.representation_mode

    rows = run_expansion_study(config)
    supported_count = sum(1 for row in rows if row.supported)
    alpha_valid_count = sum(1 for row in rows if row.alpha_valid)
    print(f"Generated: {config.count}")
    print(f"Processed: {len(rows)}")
    print(f"Supported: {supported_count}")
    print(f"Unsupported: {len(rows) - supported_count}")
    print(f"Alpha-valid: {alpha_valid_count}")
    print(f"Inputs JSONL: {config.input_jsonl_path}")
    print(f"Raw metrics JSONL: {config.raw_metrics_jsonl_path}")
    print(f"Raw metrics CSV: {config.raw_metrics_csv_path}")
    print(f"Summary JSON: {config.summary_json_path}")
    print(f"Alpha summary CSV: {config.alpha_summary_csv_path}")
    print(f"Alpha summary JSON: {config.alpha_summary_json_path}")
    print(f"Top alpha JSON: {config.top_alpha_json_path}")
    print(f"Top depth JSON: {config.top_depth_json_path}")
    print(f"Simple examples JSON: {config.simple_examples_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
