"""Goal 4.7 stratified analysis for v1 e-graph compression metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import sympy as sp

from geml.experiments.stratified_expansion import (
    AST_NODE_BUCKETS,
    EML_NODE_BUCKETS,
    OperatorFeatures,
    bucket_ast_nodes,
    bucket_eml_nodes,
    count_operator_features,
    dominant_operator_family,
    operator_signature,
    percentile,
)
from geml.symbolic.srepr import parse_srepr

EGRAPH_SUBSET_LABELS = ("all_v1", "nontrivial_v1", "identity_heavy_v1")
TRIVIALITY_FEATURE_NAMES = (
    "has_mul_by_one",
    "has_log_one",
    "has_exp_log",
    "has_log_exp",
    "constant_only_addmul_count",
    "triviality_score",
)
EGRAPH_GROUP_SUMMARY_FIELDS = [
    "count",
    "success_count",
    "processed",
    "success",
    "timeout",
    "validation_failed",
    "extraction_failed",
    "official_compilation_failed",
    "median_goal3_dag_alpha_vs_ast_tree",
    "median_optimized_dag_alpha_vs_ast_tree",
    "median_compression_gain_vs_goal3_dag",
    "p90_compression_gain_vs_goal3_dag",
    "percent_improved",
    "percent_unchanged",
    "percent_worse",
    "percent_below_threshold_before",
    "percent_below_threshold_after",
    "success_only_after_rate",
    "all_processed_after_rate",
    "timeout_rate",
    "validation_failure_rate",
    "branch_sensitive_rule_usage_rate",
]


@dataclass(frozen=True, slots=True)
class StratifiedEgraphCompressionConfig:
    """Input and output paths for the Goal 4.7 v1 stratified analysis."""

    safe_metrics_csv_path: Path = Path("outputs/v1/egraph_compression_metrics_safe.csv")
    positive_real_metrics_csv_path: Path = Path(
        "outputs/v1/egraph_compression_metrics_positive_real.csv"
    )
    goal3_metrics_csv_path: Path = Path("outputs/v1/dag_compression_metrics.csv")
    dag_summary_json_path: Path = Path("outputs/v1/dag_compression_summary.json")
    expression_generation_summary_json_path: Path | None = Path(
        "outputs/v1/expression_generation_summary.json"
    )
    alpha_by_operator_signature_csv_path: Path = Path(
        "outputs/v1/egraph_alpha_by_operator_signature.csv"
    )
    alpha_by_operator_family_csv_path: Path = Path("outputs/v1/egraph_alpha_by_operator_family.csv")
    alpha_by_size_bucket_csv_path: Path = Path("outputs/v1/egraph_alpha_by_size_bucket.csv")
    alpha_by_rule_mode_csv_path: Path = Path("outputs/v1/egraph_alpha_by_rule_mode.csv")
    alpha_by_subset_label_csv_path: Path = Path("outputs/v1/egraph_alpha_by_subset_label.csv")
    timeout_failure_summary_csv_path: Path = Path("outputs/v1/egraph_timeout_failure_summary.csv")
    triviality_effect_summary_csv_path: Path = Path(
        "outputs/v1/egraph_triviality_effect_summary.csv"
    )

    @property
    def output_paths(self) -> tuple[Path, ...]:
        """Return all CSV output paths."""
        return (
            self.alpha_by_operator_signature_csv_path,
            self.alpha_by_operator_family_csv_path,
            self.alpha_by_size_bucket_csv_path,
            self.alpha_by_rule_mode_csv_path,
            self.alpha_by_subset_label_csv_path,
            self.timeout_failure_summary_csv_path,
            self.triviality_effect_summary_csv_path,
        )

    def validate(self) -> None:
        """Validate that Goal 4.7 writes result-bearing artifacts to v1."""
        bad_paths = [path for path in self.output_paths if "outputs/v0" in path.as_posix()]
        if bad_paths:
            joined = ", ".join(str(path) for path in bad_paths)
            raise ValueError(f"Goal 4.7 must not write serious results to outputs/v0: {joined}")


@dataclass(frozen=True, slots=True)
class TrivialityFeatures:
    """Measured triviality indicators derived from authoritative source structure."""

    has_mul_by_one: bool
    has_log_one: bool
    has_exp_log: bool
    has_log_exp: bool
    constant_only_addmul_count: int

    @property
    def triviality_score(self) -> int:
        """Return an additive score over measured triviality indicators."""
        return (
            int(self.has_mul_by_one)
            + int(self.has_log_one)
            + int(self.has_exp_log)
            + int(self.has_log_exp)
            + self.constant_only_addmul_count
        )

    @property
    def subset_label(self) -> str:
        """Return the specific non-all v1 subset label for this expression."""
        if self.triviality_score > 0:
            return "identity_heavy_v1"
        return "nontrivial_v1"

    def value_for(self, feature_name: str) -> bool | int:
        """Return one named triviality feature value."""
        if feature_name == "triviality_score":
            return self.triviality_score
        return getattr(self, feature_name)


@dataclass(frozen=True, slots=True)
class StratifiedEgraphCompressionRow:
    """One Goal 4.6 row enriched with operator, bucket, and triviality features."""

    index: int
    original_expression: str
    original_srepr: str
    rule_mode: str
    saturation_status: str
    extraction_status: str
    validation_status: str
    timeout: bool
    branch_sensitive_rules_used: bool
    original_ast_tree_nodes: int
    original_eml_dag_nodes: int
    extracted_eml_dag_nodes: int | None
    goal3_dag_alpha_vs_ast_tree: float
    optimized_dag_alpha_vs_ast_tree: float | None
    compression_gain_vs_goal3_dag: float | None
    alpha_threshold_current: float
    below_threshold_goal3_dag: bool
    below_threshold_optimized_dag: bool | None
    structural_purity_valid: bool
    features: OperatorFeatures
    triviality: TrivialityFeatures
    ast_nodes_bucket: str
    original_eml_dag_size_bucket: str
    operator_signature: str
    dominant_operator_family: str

    @property
    def is_success(self) -> bool:
        """Return whether this row has a valid extracted EML-DAG result."""
        return (
            self.extraction_status == "completed"
            and self.validation_status == "valid"
            and self.structural_purity_valid
            and self.extracted_eml_dag_nodes is not None
        )

    @property
    def subset_label(self) -> str:
        """Return the non-all subset label for this row."""
        return self.triviality.subset_label


@dataclass(frozen=True, slots=True)
class StratifiedEgraphCompressionResult:
    """Result metadata for a Goal 4.7 stratified export run."""

    input_count: int
    baseline_count: int
    baseline_srepr_mismatch_index_count: int
    dag_summary_processed_count: int | None
    generation_summary_count: int | None
    output_paths: tuple[Path, ...]


def run_stratified_egraph_compression_analysis(
    config: StratifiedEgraphCompressionConfig,
) -> StratifiedEgraphCompressionResult:
    """Load Goal 4.6 v1 metrics and write stratified comparison summaries."""
    config.validate()
    dag_summary = load_json_object(config.dag_summary_json_path)
    generation_summary = load_optional_json_object(config.expression_generation_summary_json_path)
    baselines = load_goal3_baseline_sreprs(config.goal3_metrics_csv_path)
    rows = [
        *load_stratified_egraph_rows(config.safe_metrics_csv_path, baselines),
        *load_stratified_egraph_rows(config.positive_real_metrics_csv_path, baselines),
    ]
    if not rows:
        raise ValueError("no e-graph compression rows loaded")

    write_group_summary_csv(
        group_by_operator_signature(rows),
        config.alpha_by_operator_signature_csv_path,
        group_fields=["rule_mode", "subset_label", "operator_signature"],
    )
    write_group_summary_csv(
        group_by_operator_family(rows),
        config.alpha_by_operator_family_csv_path,
        group_fields=[
            "rule_mode",
            "subset_label",
            "dominant_operator_family",
            "contains_Add",
            "contains_Mul",
            "contains_log",
            "contains_exp",
        ],
    )
    write_group_summary_csv(
        group_by_size_bucket(rows),
        config.alpha_by_size_bucket_csv_path,
        group_fields=[
            "rule_mode",
            "subset_label",
            "ast_nodes_bucket",
            "original_eml_dag_size_bucket",
        ],
    )
    write_group_summary_csv(
        group_by_rule_mode(rows),
        config.alpha_by_rule_mode_csv_path,
        group_fields=["rule_mode", "subset_label"],
    )
    write_group_summary_csv(
        group_by_subset_label(rows),
        config.alpha_by_subset_label_csv_path,
        group_fields=["subset_label", "rule_mode"],
    )
    write_group_summary_csv(
        group_by_timeout_and_validation(rows),
        config.timeout_failure_summary_csv_path,
        group_fields=["rule_mode", "subset_label", "saturation_status", "validation_status"],
    )
    write_group_summary_csv(
        group_by_triviality_feature(rows),
        config.triviality_effect_summary_csv_path,
        group_fields=["rule_mode", "subset_label", "triviality_feature", "feature_value"],
    )

    return StratifiedEgraphCompressionResult(
        input_count=len(rows),
        baseline_count=len(baselines),
        baseline_srepr_mismatch_index_count=count_baseline_srepr_mismatch_indices(
            rows,
            baselines,
        ),
        dag_summary_processed_count=_optional_int(dag_summary.get("processed_count")),
        generation_summary_count=_optional_int(generation_summary.get("generated_count")),
        output_paths=config.output_paths,
    )


def load_stratified_egraph_rows(
    path: Path,
    baseline_sreprs: Mapping[int, str],
) -> list[StratifiedEgraphCompressionRow]:
    """Load and enrich one e-graph compression metric CSV."""
    rows: list[StratifiedEgraphCompressionRow] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            row = build_stratified_egraph_row(raw_row)
            baseline_srepr = baseline_sreprs.get(row.index)
            if baseline_srepr is None:
                raise ValueError(f"missing Goal 3 baseline for e-graph row index {row.index}")
            rows.append(row)
    return rows


def count_baseline_srepr_mismatch_indices(
    rows: Sequence[StratifiedEgraphCompressionRow],
    baseline_sreprs: Mapping[int, str],
) -> int:
    """Count unique indices where Goal 3 and e-graph srepr strings differ."""
    return len(
        {
            row.index
            for row in rows
            if row.index in baseline_sreprs and row.original_srepr != baseline_sreprs[row.index]
        }
    )


def build_stratified_egraph_row(
    raw_row: Mapping[str, str],
) -> StratifiedEgraphCompressionRow:
    """Build one enriched Goal 4.7 row from a raw Goal 4.6 CSV row."""
    original_srepr = raw_row["original_srepr"]
    original_ast_tree_nodes = parse_int(raw_row["original_ast_tree_nodes"])
    original_eml_dag_nodes = parse_int(raw_row["original_eml_dag_nodes"])
    features = count_operator_features(original_srepr)
    triviality = compute_triviality_features(original_srepr)
    return StratifiedEgraphCompressionRow(
        index=parse_int(raw_row["index"]),
        original_expression=raw_row.get("original_expression", ""),
        original_srepr=original_srepr,
        rule_mode=raw_row["rule_mode"],
        saturation_status=_status_value(raw_row.get("saturation_status")),
        extraction_status=_status_value(raw_row.get("extraction_status")),
        validation_status=_status_value(raw_row.get("validation_status")),
        timeout=parse_bool(raw_row.get("timeout", "False")),
        branch_sensitive_rules_used=parse_bool(raw_row.get("branch_sensitive_rules_used", "False")),
        original_ast_tree_nodes=original_ast_tree_nodes,
        original_eml_dag_nodes=original_eml_dag_nodes,
        extracted_eml_dag_nodes=parse_optional_int(raw_row.get("extracted_eml_dag_nodes")),
        goal3_dag_alpha_vs_ast_tree=parse_float(raw_row["goal3_dag_alpha_vs_ast_tree"]),
        optimized_dag_alpha_vs_ast_tree=parse_optional_float(
            raw_row.get("optimized_dag_alpha_vs_ast_tree")
        ),
        compression_gain_vs_goal3_dag=parse_optional_float(
            raw_row.get("compression_gain_vs_goal3_dag")
        ),
        alpha_threshold_current=parse_float(raw_row["alpha_threshold_current"]),
        below_threshold_goal3_dag=parse_bool(raw_row["below_threshold_goal3_dag"]),
        below_threshold_optimized_dag=parse_optional_bool(
            raw_row.get("below_threshold_optimized_dag")
        ),
        structural_purity_valid=parse_bool(raw_row.get("structural_purity_valid", "True")),
        features=features,
        triviality=triviality,
        ast_nodes_bucket=bucket_ast_nodes(original_ast_tree_nodes),
        original_eml_dag_size_bucket=bucket_eml_nodes(original_eml_dag_nodes),
        operator_signature=operator_signature(features),
        dominant_operator_family=dominant_operator_family(features),
    )


def compute_triviality_features(srepr: str) -> TrivialityFeatures:
    """Compute measured triviality indicators from authoritative original srepr."""
    expr = parse_srepr(srepr)
    has_mul_by_one = False
    has_log_one = False
    has_exp_log = False
    has_log_exp = False
    constant_only_addmul_count = 0

    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Mul):
            if any(arg == sp.Integer(1) for arg in node.args):
                has_mul_by_one = True
            if node.args and all(arg.is_Number for arg in node.args):
                constant_only_addmul_count += 1
        elif isinstance(node, sp.Add):
            if node.args and all(arg.is_Number for arg in node.args):
                constant_only_addmul_count += 1
        elif node.func == sp.log:
            arg = node.args[0]
            if arg == sp.Integer(1):
                has_log_one = True
            if arg.func == sp.exp:
                has_log_exp = True
        elif node.func == sp.exp and node.args[0].func == sp.log:
            has_exp_log = True

    return TrivialityFeatures(
        has_mul_by_one=has_mul_by_one,
        has_log_one=has_log_one,
        has_exp_log=has_exp_log,
        has_log_exp=has_log_exp,
        constant_only_addmul_count=constant_only_addmul_count,
    )


def group_by_operator_signature(
    rows: Sequence[StratifiedEgraphCompressionRow],
) -> list[dict[str, object]]:
    """Group e-graph metrics by rule mode, subset, and operator signature."""
    return build_subset_group_summaries(
        rows,
        group_fields=["rule_mode", "operator_signature"],
        key=lambda row: (row.rule_mode, row.operator_signature),
    )


def group_by_operator_family(
    rows: Sequence[StratifiedEgraphCompressionRow],
) -> list[dict[str, object]]:
    """Group e-graph metrics by operator family plus boolean operator presence."""
    return build_subset_group_summaries(
        rows,
        group_fields=[
            "rule_mode",
            "dominant_operator_family",
            "contains_Add",
            "contains_Mul",
            "contains_log",
            "contains_exp",
        ],
        key=lambda row: (
            row.rule_mode,
            row.dominant_operator_family,
            row.features.contains_Add,
            row.features.contains_Mul,
            row.features.contains_log,
            row.features.contains_exp,
        ),
    )


def group_by_size_bucket(
    rows: Sequence[StratifiedEgraphCompressionRow],
) -> list[dict[str, object]]:
    """Group e-graph metrics by AST and original EML-DAG size buckets."""
    return build_subset_group_summaries(
        rows,
        group_fields=["rule_mode", "ast_nodes_bucket", "original_eml_dag_size_bucket"],
        key=lambda row: (row.rule_mode, row.ast_nodes_bucket, row.original_eml_dag_size_bucket),
        sort_key=lambda item: (
            _rule_mode_sort(item[0]),
            _bucket_sort(item[1], AST_NODE_BUCKETS),
            _bucket_sort(item[2], EML_NODE_BUCKETS),
        ),
    )


def group_by_rule_mode(
    rows: Sequence[StratifiedEgraphCompressionRow],
) -> list[dict[str, object]]:
    """Group e-graph metrics by rule mode and subset."""
    return build_subset_group_summaries(
        rows,
        group_fields=["rule_mode"],
        key=lambda row: (row.rule_mode,),
        sort_key=lambda item: _rule_mode_sort(item[0]),
    )


def group_by_subset_label(
    rows: Sequence[StratifiedEgraphCompressionRow],
) -> list[dict[str, object]]:
    """Group e-graph metrics by subset and rule mode."""
    summaries: list[dict[str, object]] = []
    for subset_label in EGRAPH_SUBSET_LABELS:
        subset_rows = rows_for_subset(rows, subset_label)
        grouped: dict[str, list[StratifiedEgraphCompressionRow]] = {}
        for row in subset_rows:
            grouped.setdefault(row.rule_mode, []).append(row)
        for rule_mode in sorted(grouped, key=_rule_mode_sort):
            summaries.append(
                {
                    "subset_label": subset_label,
                    "rule_mode": rule_mode,
                    **summarize_egraph_group(grouped[rule_mode]),
                }
            )
    return summaries


def group_by_timeout_and_validation(
    rows: Sequence[StratifiedEgraphCompressionRow],
) -> list[dict[str, object]]:
    """Group e-graph metrics by saturation and validation status."""
    return build_subset_group_summaries(
        rows,
        group_fields=["rule_mode", "saturation_status", "validation_status"],
        key=lambda row: (row.rule_mode, row.saturation_status, row.validation_status),
    )


def group_by_triviality_feature(
    rows: Sequence[StratifiedEgraphCompressionRow],
) -> list[dict[str, object]]:
    """Group e-graph metrics by measured triviality feature values."""
    summaries: list[dict[str, object]] = []
    for subset_label in EGRAPH_SUBSET_LABELS:
        subset_rows = rows_for_subset(rows, subset_label)
        grouped: dict[tuple[str, str, str], list[StratifiedEgraphCompressionRow]] = {}
        for row in subset_rows:
            for feature_name in TRIVIALITY_FEATURE_NAMES:
                key = (
                    row.rule_mode,
                    feature_name,
                    str(row.triviality.value_for(feature_name)),
                )
                grouped.setdefault(key, []).append(row)
        for key in sorted(grouped, key=lambda item: (_rule_mode_sort(item[0]), item[1], item[2])):
            rule_mode, feature_name, feature_value = key
            summaries.append(
                {
                    "rule_mode": rule_mode,
                    "subset_label": subset_label,
                    "triviality_feature": feature_name,
                    "feature_value": feature_value,
                    **summarize_egraph_group(grouped[key]),
                }
            )
    return summaries


def build_subset_group_summaries(
    rows: Sequence[StratifiedEgraphCompressionRow],
    *,
    group_fields: Sequence[str],
    key: Callable[[StratifiedEgraphCompressionRow], tuple[object, ...]],
    sort_key: Callable[[tuple[object, ...]], object] | None = None,
) -> list[dict[str, object]]:
    """Build grouped summaries for all, nontrivial, and identity-heavy subsets."""
    summaries: list[dict[str, object]] = []
    for subset_label in EGRAPH_SUBSET_LABELS:
        grouped: dict[tuple[object, ...], list[StratifiedEgraphCompressionRow]] = {}
        for row in rows_for_subset(rows, subset_label):
            grouped.setdefault(key(row), []).append(row)

        key_sort = sort_key if sort_key is not None else _default_key_sort
        for group_key in sorted(grouped, key=key_sort):
            base = {field: value for field, value in zip(group_fields, group_key, strict=True)}
            summaries.append(
                {
                    **base,
                    "subset_label": subset_label,
                    **summarize_egraph_group(grouped[group_key]),
                }
            )
    return summaries


def rows_for_subset(
    rows: Sequence[StratifiedEgraphCompressionRow],
    subset_label: str,
) -> list[StratifiedEgraphCompressionRow]:
    """Return rows belonging to one reported v1 subset."""
    if subset_label == "all_v1":
        return list(rows)
    if subset_label not in {"nontrivial_v1", "identity_heavy_v1"}:
        raise ValueError(f"unknown subset label: {subset_label!r}")
    return [row for row in rows if row.subset_label == subset_label]


def summarize_egraph_group(
    rows: Sequence[StratifiedEgraphCompressionRow],
) -> dict[str, object]:
    """Compute required Goal 4.7 statistics for one group."""
    if not rows:
        return {
            "count": 0,
            "success_count": 0,
            "processed": 0,
            "success": 0,
            "timeout": 0,
            "validation_failed": 0,
            "extraction_failed": 0,
            "official_compilation_failed": 0,
            "median_goal3_dag_alpha_vs_ast_tree": None,
            "median_optimized_dag_alpha_vs_ast_tree": None,
            "median_compression_gain_vs_goal3_dag": None,
            "p90_compression_gain_vs_goal3_dag": None,
            "percent_improved": None,
            "percent_unchanged": None,
            "percent_worse": None,
            "percent_below_threshold_before": None,
            "percent_below_threshold_after": None,
            "success_only_after_rate": None,
            "all_processed_after_rate": None,
            "timeout_rate": None,
            "validation_failure_rate": None,
            "branch_sensitive_rule_usage_rate": None,
        }

    success_rows = [row for row in rows if _is_successful_egraph_row(row)]
    status_counts = _egraph_status_counts(rows)
    improved, unchanged, worse = classify_improvement(success_rows)
    compression_gains = sorted(
        row.compression_gain_vs_goal3_dag
        for row in success_rows
        if row.compression_gain_vs_goal3_dag is not None
    )
    optimized_alphas = sorted(
        row.optimized_dag_alpha_vs_ast_tree
        for row in success_rows
        if row.optimized_dag_alpha_vs_ast_tree is not None
    )
    after_threshold_success_count = sum(
        row.below_threshold_optimized_dag is True for row in success_rows
    )
    success_only_after_rate = _percent(after_threshold_success_count, len(success_rows))
    all_processed_after_rate = _percent(after_threshold_success_count, len(rows))
    return {
        "count": len(rows),
        "success_count": len(success_rows),
        "processed": status_counts["processed"],
        "success": status_counts["success"],
        "timeout": status_counts["timeout"],
        "validation_failed": status_counts["validation_failed"],
        "extraction_failed": status_counts["extraction_failed"],
        "official_compilation_failed": status_counts["official_compilation_failed"],
        "median_goal3_dag_alpha_vs_ast_tree": statistics.median(
            sorted(row.goal3_dag_alpha_vs_ast_tree for row in rows)
        ),
        "median_optimized_dag_alpha_vs_ast_tree": _median_or_none(optimized_alphas),
        "median_compression_gain_vs_goal3_dag": _median_or_none(compression_gains),
        "p90_compression_gain_vs_goal3_dag": percentile(compression_gains, 0.9)
        if compression_gains
        else None,
        "percent_improved": _percent(len(improved), len(success_rows)),
        "percent_unchanged": _percent(len(unchanged), len(success_rows)),
        "percent_worse": _percent(len(worse), len(success_rows)),
        "percent_below_threshold_before": _percent(
            sum(row.below_threshold_goal3_dag for row in rows),
            len(rows),
        ),
        "percent_below_threshold_after": success_only_after_rate,
        "success_only_after_rate": success_only_after_rate,
        "all_processed_after_rate": all_processed_after_rate,
        "timeout_rate": _percent(sum(row.timeout for row in rows), len(rows)),
        "validation_failure_rate": _percent(
            sum(row.validation_status != "valid" for row in rows),
            len(rows),
        ),
        "branch_sensitive_rule_usage_rate": _percent(
            sum(row.branch_sensitive_rules_used for row in rows),
            len(rows),
        ),
    }


def _is_successful_egraph_row(row: StratifiedEgraphCompressionRow) -> bool:
    return (
        row.extraction_status == "completed"
        and row.validation_status == "valid"
        and row.structural_purity_valid
        and row.extracted_eml_dag_nodes is not None
    )


def _is_timeout_egraph_row(row: StratifiedEgraphCompressionRow) -> bool:
    return row.timeout or row.saturation_status == "timeout" or row.extraction_status == "timeout"


def _egraph_status_counts(rows: Sequence[StratifiedEgraphCompressionRow]) -> dict[str, int]:
    return {
        "processed": len(rows),
        "success": sum(_is_successful_egraph_row(row) for row in rows),
        "timeout": sum(_is_timeout_egraph_row(row) for row in rows),
        "validation_failed": sum(
            (
                not _is_timeout_egraph_row(row)
                and row.extraction_status == "completed"
                and (
                    row.validation_status not in {None, "valid"} or not row.structural_purity_valid
                )
            )
            for row in rows
        ),
        "extraction_failed": sum(
            (not _is_timeout_egraph_row(row) and row.extraction_status not in {"completed"})
            for row in rows
        ),
        "official_compilation_failed": sum(
            (
                not _is_timeout_egraph_row(row)
                and row.extraction_status == "completed"
                and row.validation_status == "valid"
                and row.extracted_eml_dag_nodes is None
            )
            for row in rows
        ),
    }


def classify_improvement(
    rows: Sequence[StratifiedEgraphCompressionRow],
) -> tuple[
    list[StratifiedEgraphCompressionRow],
    list[StratifiedEgraphCompressionRow],
    list[StratifiedEgraphCompressionRow],
]:
    """Split successful rows into improved, unchanged, and worse classes."""
    improved: list[StratifiedEgraphCompressionRow] = []
    unchanged: list[StratifiedEgraphCompressionRow] = []
    worse: list[StratifiedEgraphCompressionRow] = []
    for row in rows:
        if row.extracted_eml_dag_nodes is None:
            continue
        if row.extracted_eml_dag_nodes < row.original_eml_dag_nodes:
            improved.append(row)
        elif row.extracted_eml_dag_nodes == row.original_eml_dag_nodes:
            unchanged.append(row)
        else:
            worse.append(row)
    return improved, unchanged, worse


def load_goal3_baseline_sreprs(path: Path) -> dict[int, str]:
    """Load v1 Goal 3 baseline sreprs keyed by expression index."""
    baselines: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            baselines[parse_int(raw_row["index"])] = raw_row["srepr"]
    return baselines


def load_json_object(path: Path) -> dict[str, object]:
    """Load a required JSON object."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def load_optional_json_object(path: Path | None) -> dict[str, object]:
    """Load an optional JSON object, returning an empty dict when absent."""
    if path is None or not path.exists():
        return {}
    return load_json_object(path)


def write_group_summary_csv(
    rows: Sequence[dict[str, object]],
    path: Path,
    *,
    group_fields: Sequence[str],
) -> None:
    """Write one e-graph grouped summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[*group_fields, *EGRAPH_GROUP_SUMMARY_FIELDS],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_int(value: str) -> int:
    """Parse a required integer CSV field."""
    if value == "":
        raise ValueError("expected integer, got empty string")
    return int(value)


def parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer CSV field."""
    if value in {None, ""}:
        return None
    return int(value)


def parse_float(value: str) -> float:
    """Parse a required finite float CSV field."""
    if value == "":
        raise ValueError("expected float, got empty string")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"expected finite float, got {value!r}")
    return parsed


def parse_optional_float(value: str | None) -> float | None:
    """Parse an optional finite float CSV field."""
    if value in {None, ""}:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_bool(value: str | bool | None) -> bool:
    """Parse a required boolean CSV field."""
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected boolean string, got {value!r}")


def parse_optional_bool(value: str | bool | None) -> bool | None:
    """Parse an optional boolean CSV field."""
    if value in {None, ""}:
        return None
    return parse_bool(value)


def _status_value(value: str | None) -> str:
    return value if value not in {None, ""} else "missing"


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _bucket_sort(label: object, buckets: Sequence[tuple[int, int | None, str]]) -> int:
    labels = [bucket_label for _, _, bucket_label in buckets]
    return labels.index(str(label)) if str(label) in labels else len(labels)


def _rule_mode_sort(rule_mode: object) -> int:
    order = {"safe": 0, "positive_real_formal": 1}
    return order.get(str(rule_mode), len(order))


def _default_key_sort(key: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(str(part) for part in key)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    default_config = StratifiedEgraphCompressionConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--safe-metrics-csv",
        type=Path,
        default=default_config.safe_metrics_csv_path,
    )
    parser.add_argument(
        "--positive-real-metrics-csv",
        type=Path,
        default=default_config.positive_real_metrics_csv_path,
    )
    parser.add_argument(
        "--goal3-metrics-csv",
        type=Path,
        default=default_config.goal3_metrics_csv_path,
    )
    parser.add_argument(
        "--dag-summary-json",
        type=Path,
        default=default_config.dag_summary_json_path,
    )
    parser.add_argument(
        "--expression-generation-summary-json",
        type=Path,
        default=default_config.expression_generation_summary_json_path,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 4.7 stratified e-graph compression analysis."""
    args = build_parser().parse_args(argv)
    generation_summary_path = args.expression_generation_summary_json
    config = StratifiedEgraphCompressionConfig(
        safe_metrics_csv_path=args.safe_metrics_csv,
        positive_real_metrics_csv_path=args.positive_real_metrics_csv,
        goal3_metrics_csv_path=args.goal3_metrics_csv,
        dag_summary_json_path=args.dag_summary_json,
        expression_generation_summary_json_path=generation_summary_path,
    )
    result = run_stratified_egraph_compression_analysis(config)

    print(f"Loaded e-graph metric rows: {result.input_count}")
    print(f"Loaded Goal 3 baseline rows: {result.baseline_count}")
    if result.baseline_srepr_mismatch_index_count:
        print(
            "Goal 3/e-graph srepr mismatch indices: "
            f"{result.baseline_srepr_mismatch_index_count} "
            "(analysis used e-graph original_srepr for features)"
        )
    if result.dag_summary_processed_count is not None:
        print(f"Goal 3 summary processed count: {result.dag_summary_processed_count}")
    if result.generation_summary_count is not None:
        print(f"Generation summary count: {result.generation_summary_count}")
    for path in result.output_paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
