"""Goal 2.3 stratified alpha analysis for official pure EML expansion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import sympy as sp
import yaml
from pydantic import BaseModel, model_validator

OPERATOR_ORDER = ("Add", "Mul", "Pow", "exp", "log")
BOOLEAN_FEATURES = ("contains_log", "contains_exp", "contains_Mul", "contains_Add")
AST_NODE_BUCKETS = (
    (1, 3, "1-3"),
    (4, 7, "4-7"),
    (8, 15, "8-15"),
    (16, 31, "16-31"),
    (32, None, "32+"),
)
AST_DEPTH_BUCKETS = (
    (0, 1, "0-1"),
    (2, 2, "2"),
    (3, 3, "3"),
    (4, 4, "4"),
    (5, None, "5+"),
)
EML_NODE_BUCKETS = (
    (1, 15, "1-15"),
    (16, 31, "16-31"),
    (32, 63, "32-63"),
    (64, 127, "64-127"),
    (128, 255, "128-255"),
    (256, None, "256+"),
)
ALPHA_BUCKETS = (
    (0.0, 2.0, "0-<2"),
    (2.0, 5.0, "2-<5"),
    (5.0, 10.0, "5-<10"),
    (10.0, 15.0, "10-<15"),
    (15.0, None, "15+"),
)
SUMMARY_FIELDS = [
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


class StratifiedExpansionConfig(BaseModel):
    """Configuration for Goal 2.3 stratified expansion analysis."""

    raw_metrics_csv_path: Path = Path("outputs/v0/expansion_raw_metrics.csv")
    alpha_summary_csv_path: Path = Path("outputs/v0/expansion_alpha_summary.csv")
    alpha_summary_json_path: Path = Path("outputs/v0/expansion_alpha_summary.json")
    alpha_by_ast_depth_csv_path: Path = Path("outputs/v0/alpha_by_ast_depth.csv")
    alpha_by_ast_size_bucket_csv_path: Path = Path("outputs/v0/alpha_by_ast_size_bucket.csv")
    alpha_by_operator_family_csv_path: Path = Path("outputs/v0/alpha_by_operator_family.csv")
    alpha_by_operator_signature_csv_path: Path = Path("outputs/v0/alpha_by_operator_signature.csv")
    alpha_by_boolean_features_csv_path: Path = Path("outputs/v0/alpha_by_boolean_features.csv")

    @model_validator(mode="after")
    def validate_input_paths(self) -> Self:
        if self.raw_metrics_csv_path == self.alpha_summary_csv_path:
            raise ValueError("raw metrics and alpha summary CSV paths must differ")
        return self


@dataclass(frozen=True)
class OperatorFeatures:
    """Operator-family feature counts for one expression."""

    count_Add: int = 0
    count_Mul: int = 0
    count_Pow: int = 0
    count_exp: int = 0
    count_log: int = 0
    count_symbols: int = 0
    count_constants: int = 0

    @property
    def contains_Add(self) -> bool:
        return self.count_Add > 0

    @property
    def contains_Mul(self) -> bool:
        return self.count_Mul > 0

    @property
    def contains_Pow(self) -> bool:
        return self.count_Pow > 0

    @property
    def contains_exp(self) -> bool:
        return self.count_exp > 0

    @property
    def contains_log(self) -> bool:
        return self.count_log > 0

    def count_for(self, family: str) -> int:
        return int(getattr(self, f"count_{family}"))


@dataclass(frozen=True)
class StratifiedExpressionRow:
    """Raw metrics row enriched with Goal 2.3 stratification features."""

    expression: str
    srepr: str
    ast_node_count: int
    ast_depth: int
    ast_operator_count: int
    ast_leaf_count: int
    eml_node_count: int
    eml_depth: int
    eml_operator_count: int
    eml_leaf_count: int
    alpha: float
    alpha_threshold: float
    below_threshold: bool
    features: OperatorFeatures
    ast_nodes_bucket: str
    ast_depth_bucket: str
    eml_nodes_bucket: str
    alpha_bucket: str
    dominant_operator_family: str
    operator_signature: str


@dataclass(frozen=True)
class StratifiedExpansionResult:
    """Result metadata from a stratified analysis export run."""

    input_count: int
    alpha_summary_json_count: int
    alpha_summary_csv_count: int
    output_paths: tuple[Path, ...]


def run_stratified_expansion_analysis(
    config: StratifiedExpansionConfig,
) -> StratifiedExpansionResult:
    """Load Goal 2.2 outputs and write Goal 2.3 stratified CSV summaries."""
    alpha_summary_json = load_alpha_summary_json(config.alpha_summary_json_path)
    alpha_summary_csv = load_alpha_summary_csv(config.alpha_summary_csv_path)
    rows = load_stratified_rows(config.raw_metrics_csv_path)

    write_group_summary_csv(
        group_by_ast_depth(rows),
        config.alpha_by_ast_depth_csv_path,
        group_fields=["ast_depth"],
    )
    write_group_summary_csv(
        group_by_ast_size_bucket(rows),
        config.alpha_by_ast_size_bucket_csv_path,
        group_fields=["ast_nodes_bucket"],
    )
    write_group_summary_csv(
        group_by_operator_family(rows),
        config.alpha_by_operator_family_csv_path,
        group_fields=["dominant_operator_family"],
    )
    write_group_summary_csv(
        group_by_operator_signature(rows),
        config.alpha_by_operator_signature_csv_path,
        group_fields=["operator_signature"],
    )
    write_group_summary_csv(
        group_by_boolean_features(rows),
        config.alpha_by_boolean_features_csv_path,
        group_fields=["feature", "value"],
    )

    return StratifiedExpansionResult(
        input_count=len(rows),
        alpha_summary_json_count=len(alpha_summary_json),
        alpha_summary_csv_count=len(alpha_summary_csv),
        output_paths=(
            config.alpha_by_ast_depth_csv_path,
            config.alpha_by_ast_size_bucket_csv_path,
            config.alpha_by_operator_family_csv_path,
            config.alpha_by_operator_signature_csv_path,
            config.alpha_by_boolean_features_csv_path,
        ),
    )


def load_alpha_summary_json(path: Path) -> list[dict[str, Any]]:
    """Load Goal 2.2 JSON alpha summary output."""
    with path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}")
    return data


def load_alpha_summary_csv(path: Path) -> list[dict[str, str]]:
    """Load Goal 2.2 CSV alpha summary output."""
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_stratified_rows(path: Path) -> list[StratifiedExpressionRow]:
    """Load raw metric CSV rows and enrich them with stratification features."""
    rows: list[StratifiedExpressionRow] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            if parse_bool(raw_row["supported"]) and parse_bool(raw_row["alpha_valid"]):
                rows.append(build_stratified_row(raw_row))
    return rows


def build_stratified_row(raw_row: dict[str, str]) -> StratifiedExpressionRow:
    """Build one enriched stratification row from the flattened raw metrics CSV."""
    ast_node_count = parse_int(raw_row["ast_node_count"])
    ast_depth = parse_int(raw_row["ast_depth"])
    eml_node_count = parse_int(raw_row["eml_node_count"])
    alpha = parse_float(raw_row["alpha"])
    features = count_operator_features(raw_row["srepr"])

    return StratifiedExpressionRow(
        expression=raw_row["expression"],
        srepr=raw_row["srepr"],
        ast_node_count=ast_node_count,
        ast_depth=ast_depth,
        ast_operator_count=parse_int(raw_row["ast_operator_count"]),
        ast_leaf_count=parse_int(raw_row["ast_leaf_count"]),
        eml_node_count=eml_node_count,
        eml_depth=parse_int(raw_row["eml_depth"]),
        eml_operator_count=parse_int(raw_row["eml_operator_count"]),
        eml_leaf_count=parse_int(raw_row["eml_leaf_count"]),
        alpha=alpha,
        alpha_threshold=parse_float(raw_row["alpha_threshold"]),
        below_threshold=parse_bool(raw_row["below_threshold"]),
        features=features,
        ast_nodes_bucket=bucket_ast_nodes(ast_node_count),
        ast_depth_bucket=bucket_ast_depth(ast_depth),
        eml_nodes_bucket=bucket_eml_nodes(eml_node_count),
        alpha_bucket=bucket_alpha(alpha),
        dominant_operator_family=dominant_operator_family(features),
        operator_signature=operator_signature(features),
    )


def count_operator_features(srepr: str) -> OperatorFeatures:
    """Count operator-family features from authoritative SymPy ``srepr``."""
    expr = parse_srepr(srepr)
    counts = {family: 0 for family in OPERATOR_ORDER}
    symbol_count = 0
    constant_count = 0

    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Symbol):
            symbol_count += 1
        elif node.is_Number:
            constant_count += 1
        elif isinstance(node, sp.Add):
            counts["Add"] += 1
        elif isinstance(node, sp.Mul):
            counts["Mul"] += 1
        elif isinstance(node, sp.Pow):
            counts["Pow"] += 1
        elif node.func == sp.exp:
            counts["exp"] += 1
        elif node.func == sp.log:
            counts["log"] += 1

    return OperatorFeatures(
        count_Add=counts["Add"],
        count_Mul=counts["Mul"],
        count_Pow=counts["Pow"],
        count_exp=counts["exp"],
        count_log=counts["log"],
        count_symbols=symbol_count,
        count_constants=constant_count,
    )


def parse_srepr(srepr: str) -> sp.Expr:
    """Parse generated SymPy ``srepr`` while preserving unevaluated operators."""

    def add(*args: sp.Expr, **_: object) -> sp.Expr:
        return sp.Add(*args, evaluate=False)

    def mul(*args: sp.Expr, **_: object) -> sp.Expr:
        return sp.Mul(*args, evaluate=False)

    def pow_expr(base: sp.Expr, exponent: sp.Expr, **_: object) -> sp.Expr:
        return sp.Pow(base, exponent, evaluate=False)

    def exp_expr(arg: sp.Expr, **_: object) -> sp.Expr:
        return sp.exp(arg, evaluate=False)

    def log_expr(arg: sp.Expr, **_: object) -> sp.Expr:
        return sp.log(arg, evaluate=False)

    return sp.sympify(
        srepr,
        locals={
            "Add": add,
            "Float": sp.Float,
            "Integer": sp.Integer,
            "Mul": mul,
            "Pow": pow_expr,
            "Rational": sp.Rational,
            "Symbol": sp.Symbol,
            "exp": exp_expr,
            "log": log_expr,
        },
        evaluate=False,
    )


def operator_signature(features: OperatorFeatures) -> str:
    """Return exact operator-family signature, e.g. ``Add+Mul+log``."""
    active = [family for family in OPERATOR_ORDER if features.count_for(family) > 0]
    return "+".join(active) if active else "leaf_only"


def dominant_operator_family(features: OperatorFeatures) -> str:
    """Return the most frequent operator family, with deterministic tie labels."""
    counts = {family: features.count_for(family) for family in OPERATOR_ORDER}
    max_count = max(counts.values())
    if max_count == 0:
        return "leaf_only"
    winners = [family for family in OPERATOR_ORDER if counts[family] == max_count]
    if len(winners) == 1:
        return winners[0]
    return "mixed_" + "+".join(winners)


def bucket_ast_nodes(value: int) -> str:
    """Assign AST node-count bucket."""
    return assign_int_bucket(value, AST_NODE_BUCKETS)


def bucket_ast_depth(value: int) -> str:
    """Assign AST depth bucket."""
    return assign_int_bucket(value, AST_DEPTH_BUCKETS)


def bucket_eml_nodes(value: int) -> str:
    """Assign EML node-count bucket."""
    return assign_int_bucket(value, EML_NODE_BUCKETS)


def bucket_alpha(value: float) -> str:
    """Assign alpha bucket."""
    return assign_float_bucket(value, ALPHA_BUCKETS)


def assign_int_bucket(
    value: int,
    buckets: Sequence[tuple[int, int | None, str]],
) -> str:
    """Assign an integer value to inclusive lower/upper buckets."""
    for lower, upper, label in buckets:
        if value >= lower and (upper is None or value <= upper):
            return label
    raise ValueError(f"value {value} did not match any bucket")


def assign_float_bucket(
    value: float,
    buckets: Sequence[tuple[float, float | None, str]],
) -> str:
    """Assign a float value to lower-inclusive, upper-exclusive buckets."""
    for lower, upper, label in buckets:
        if value >= lower and (upper is None or value < upper):
            return label
    raise ValueError(f"value {value} did not match any bucket")


def group_by_ast_depth(rows: Sequence[StratifiedExpressionRow]) -> list[dict[str, object]]:
    """Group alpha statistics by exact AST depth."""
    return build_group_summaries(
        rows,
        group_key=lambda row: row.ast_depth,
        group_field="ast_depth",
        sort_key=lambda key: int(key),
    )


def group_by_ast_size_bucket(
    rows: Sequence[StratifiedExpressionRow],
) -> list[dict[str, object]]:
    """Group alpha statistics by AST node-count bucket."""
    return build_group_summaries(
        rows,
        group_key=lambda row: row.ast_nodes_bucket,
        group_field="ast_nodes_bucket",
        ordered_keys=[label for _, _, label in AST_NODE_BUCKETS],
    )


def group_by_operator_family(
    rows: Sequence[StratifiedExpressionRow],
) -> list[dict[str, object]]:
    """Group alpha statistics by dominant operator family."""
    return build_group_summaries(
        rows,
        group_key=lambda row: row.dominant_operator_family,
        group_field="dominant_operator_family",
        sort_key=str,
    )


def group_by_operator_signature(
    rows: Sequence[StratifiedExpressionRow],
) -> list[dict[str, object]]:
    """Group alpha statistics by exact operator signature."""
    return build_group_summaries(
        rows,
        group_key=lambda row: row.operator_signature,
        group_field="operator_signature",
        sort_key=str,
    )


def group_by_boolean_features(
    rows: Sequence[StratifiedExpressionRow],
) -> list[dict[str, object]]:
    """Group alpha statistics by each configured boolean operator feature."""
    summaries: list[dict[str, object]] = []
    for feature in BOOLEAN_FEATURES:
        for value in (False, True):
            group_rows = [row for row in rows if bool(getattr(row.features, feature)) is value]
            summaries.append(
                {
                    "feature": feature,
                    "value": value,
                    **summarize_group(group_rows),
                }
            )
    return summaries


def build_group_summaries(
    rows: Sequence[StratifiedExpressionRow],
    *,
    group_key: Callable[[StratifiedExpressionRow], object],
    group_field: str,
    sort_key: Callable[[object], object] | None = None,
    ordered_keys: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    """Build group summary rows for one grouping key."""
    grouped: dict[object, list[StratifiedExpressionRow]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)

    if ordered_keys is not None:
        keys = [key for key in ordered_keys if key in grouped]
    else:
        key_func = sort_key if sort_key is not None else str
        keys = sorted(grouped, key=key_func)

    return [{group_field: key, **summarize_group(grouped[key])} for key in keys]


def summarize_group(rows: Sequence[StratifiedExpressionRow]) -> dict[str, object]:
    """Compute required alpha and size statistics for one group."""
    if not rows:
        return {
            "count": 0,
            "mean_alpha": None,
            "median_alpha": None,
            "p90_alpha": None,
            "p95_alpha": None,
            "max_alpha": None,
            "mean_ast_nodes": None,
            "mean_eml_nodes": None,
            "percent_below_threshold": None,
        }

    alphas = sorted(row.alpha for row in rows)
    below_count = sum(1 for row in rows if row.below_threshold)
    return {
        "count": len(rows),
        "mean_alpha": statistics.fmean(alphas),
        "median_alpha": statistics.median(alphas),
        "p90_alpha": percentile(alphas, 0.9),
        "p95_alpha": percentile(alphas, 0.95),
        "max_alpha": max(alphas),
        "mean_ast_nodes": statistics.fmean(row.ast_node_count for row in rows),
        "mean_eml_nodes": statistics.fmean(row.eml_node_count for row in rows),
        "percent_below_threshold": 100 * below_count / len(rows),
    }


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Compute upper-rank percentile for already sorted values."""
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    index = max(0, min(len(sorted_values) - 1, math.ceil(quantile * len(sorted_values)) - 1))
    return sorted_values[index]


def write_group_summary_csv(
    rows: Sequence[dict[str, object]],
    path: Path,
    *,
    group_fields: Sequence[str],
) -> None:
    """Write one grouped summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[*group_fields, *SUMMARY_FIELDS])
        writer.writeheader()
        writer.writerows(rows)


def parse_int(value: str) -> int:
    """Parse a required integer CSV field."""
    if value == "":
        raise ValueError("expected integer, got empty string")
    return int(value)


def parse_float(value: str) -> float:
    """Parse a required float CSV field."""
    if value == "":
        raise ValueError("expected float, got empty string")
    return float(value)


def parse_bool(value: str) -> bool:
    """Parse a required boolean CSV field."""
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected boolean string, got {value!r}")


def load_config(path: Path) -> StratifiedExpansionConfig:
    """Load a YAML Goal 2.3 stratified analysis config."""
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    return StratifiedExpansionConfig.model_validate(raw_config)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/expansion_v0.yaml"),
        help="Path to a YAML config with Goal 2 output paths.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 2.3 stratified expansion analysis."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    result = run_stratified_expansion_analysis(config)

    print(f"Loaded raw metric rows: {result.input_count}")
    print(f"Loaded alpha summary JSON rows: {result.alpha_summary_json_count}")
    print(f"Loaded alpha summary CSV rows: {result.alpha_summary_csv_count}")
    for path in result.output_paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
