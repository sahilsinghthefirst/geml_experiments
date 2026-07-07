"""Goal 3.3 fixed-distribution DAG compression study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator

from geml.data.dataset import (
    GeneratedExpressionInput,
    build_symbol_locals,
    load_generated_expressions,
    parse_generated_expression,
)
from geml.data.generate_exprs import (
    DEFAULT_OPERATOR_PROBABILITIES,
    DEFAULT_POSITIVE_LOG_ARGUMENT_PROBABILITIES,
    DEFAULT_SYMBOL_NAMES,
    ExpressionGeneratorConfig,
    LogArgumentStrategy,
    generate_dataset,
)
from geml.experiments.expansion_study import compute_alpha_threshold
from geml.symbolic.dag_metrics import compute_expression_dag_analysis

type SourceSerialization = Literal["srepr"]

DAG_COMPRESSION_CSV_FIELDS = [
    "index",
    "expression",
    "srepr",
    "source_serialization",
    "supported",
    "ast_tree_node_count",
    "ast_dag_node_count",
    "ast_dag_child_ref_count",
    "ast_tree_depth",
    "ast_dag_depth",
    "ast_dag_compression",
    "eml_tree_node_count",
    "eml_dag_node_count",
    "eml_dag_child_ref_count",
    "eml_tree_depth",
    "eml_dag_depth",
    "eml_dag_compression",
    "tree_alpha",
    "dag_alpha_vs_ast_tree",
    "dag_alpha_vs_ast_dag",
    "alpha_threshold_current",
    "below_threshold_tree",
    "below_threshold_dag_vs_ast_tree",
    "below_threshold_dag_vs_ast_dag",
    "pure_eml_valid",
    "derived_leaf_count",
    "hidden_compound_leaf_count",
    "error",
]


class DagCompressionStudyConfig(BaseModel):
    """Configuration for the Goal 3.3 DAG compression study."""

    seed: int = 0
    count: int = Field(default=10_000, gt=0)
    max_depth: int = Field(default=4, ge=0)
    output_dir: Path = Path("outputs/v0")
    input_jsonl_path: Path = Path("outputs/v0/dag_compression_inputs.jsonl")
    metrics_jsonl_path: Path = Path("outputs/v0/dag_compression_metrics.jsonl")
    metrics_csv_path: Path = Path("outputs/v0/dag_compression_metrics.csv")
    summary_json_path: Path = Path("outputs/v0/dag_compression_summary.json")
    generation_summary_json_path: Path | None = None
    alpha_threshold_k: int = Field(default=4, gt=0)
    alpha_threshold_l: int = Field(default=3, gt=0)
    operator_probabilities: dict[str, float] = Field(
        default_factory=lambda: DEFAULT_OPERATOR_PROBABILITIES.copy()
    )
    target_depth_probabilities: dict[int, float] | None = None
    intermediate_leaf_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    deduplicate_srepr: bool = False
    max_generation_attempts: int | None = Field(default=None, gt=0)
    log_argument_strategy: LogArgumentStrategy = "exp_wrap"
    positive_log_argument_probabilities: dict[str, float] = Field(
        default_factory=lambda: DEFAULT_POSITIVE_LOG_ARGUMENT_PROBABILITIES.copy()
    )
    max_triviality_score: int | None = Field(default=None, ge=0)
    symbol_names: tuple[str, ...] = DEFAULT_SYMBOL_NAMES

    @model_validator(mode="after")
    def validate_symbol_names(self) -> Self:
        if not self.symbol_names:
            raise ValueError("symbol_names must not be empty")
        return self

    @model_validator(mode="after")
    def validate_output_paths(self) -> Self:
        forbidden_goal2_paths = {
            Path("outputs/v0/expansion_inputs.jsonl"),
            Path("outputs/v0/expansion_raw_metrics.jsonl"),
            Path("outputs/v0/expansion_raw_metrics.csv"),
            Path("outputs/v0/official_eml_compiler_summary.json"),
        }
        configured_paths = {
            self.input_jsonl_path,
            self.metrics_jsonl_path,
            self.metrics_csv_path,
            self.summary_json_path,
        }
        overlap = configured_paths & forbidden_goal2_paths
        if overlap:
            overlap_text = ", ".join(str(path) for path in sorted(overlap))
            raise ValueError(f"Goal 3 DAG study must not overwrite Goal 2 outputs: {overlap_text}")
        return self


class DagCompressionMetricsRow(BaseModel):
    """One per-expression DAG compression metric row."""

    index: int
    expression: str
    srepr: str
    source_serialization: SourceSerialization = "srepr"
    supported: bool
    ast_tree_node_count: int | None = None
    ast_dag_node_count: int | None = None
    ast_dag_child_ref_count: int | None = None
    ast_tree_depth: int | None = None
    ast_dag_depth: int | None = None
    ast_dag_compression: float | None = None
    eml_tree_node_count: int | None = None
    eml_dag_node_count: int | None = None
    eml_dag_child_ref_count: int | None = None
    eml_tree_depth: int | None = None
    eml_dag_depth: int | None = None
    eml_dag_compression: float | None = None
    tree_alpha: float | None = None
    dag_alpha_vs_ast_tree: float | None = None
    dag_alpha_vs_ast_dag: float | None = None
    alpha_threshold_current: float
    below_threshold_tree: bool | None = None
    below_threshold_dag_vs_ast_tree: bool | None = None
    below_threshold_dag_vs_ast_dag: bool | None = None
    pure_eml_valid: bool = False
    derived_leaf_count: int | None = None
    hidden_compound_leaf_count: int | None = None
    error: str | None = None


def run_dag_compression_study(
    config: DagCompressionStudyConfig,
) -> list[DagCompressionMetricsRow]:
    """Run the Goal 3.3 DAG compression study and write artifacts."""
    generate_dag_compression_inputs(config)
    input_rows = load_generated_expressions(config.input_jsonl_path)
    metrics_rows = compute_dag_compression_rows(input_rows, config=config)
    write_metrics_jsonl(metrics_rows, config.metrics_jsonl_path)
    write_metrics_csv(metrics_rows, config.metrics_csv_path)
    write_summary_json(metrics_rows, config.summary_json_path)
    return metrics_rows


def generate_dag_compression_inputs(config: DagCompressionStudyConfig) -> None:
    """Regenerate deterministic fixed-distribution input expressions."""
    generator_config = ExpressionGeneratorConfig(
        seed=config.seed,
        count=config.count,
        max_depth=config.max_depth,
        output_dir=config.output_dir,
        jsonl_path=config.input_jsonl_path,
        csv_path=None,
        summary_json_path=config.generation_summary_json_path,
        operator_probabilities=config.operator_probabilities,
        target_depth_probabilities=config.target_depth_probabilities,
        intermediate_leaf_probability=config.intermediate_leaf_probability,
        deduplicate_srepr=config.deduplicate_srepr,
        max_generation_attempts=config.max_generation_attempts,
        log_argument_strategy=config.log_argument_strategy,
        positive_log_argument_probabilities=config.positive_log_argument_probabilities,
        max_triviality_score=config.max_triviality_score,
        symbol_names=config.symbol_names,
    )
    generate_dataset(generator_config)


def compute_dag_compression_rows(
    input_rows: Sequence[GeneratedExpressionInput],
    *,
    config: DagCompressionStudyConfig,
) -> list[DagCompressionMetricsRow]:
    """Compute Goal 3.3 metrics for generated expression rows."""
    symbol_locals = build_symbol_locals(config.symbol_names)
    alpha_threshold = compute_alpha_threshold(config.alpha_threshold_k, config.alpha_threshold_l)
    metric_rows: list[DagCompressionMetricsRow] = []

    for fallback_index, input_row in enumerate(input_rows):
        index = input_row.index if input_row.index is not None else fallback_index
        if not input_row.srepr:
            metric_rows.append(
                _unsupported_row(
                    index=index,
                    expression=input_row.expression,
                    srepr="",
                    alpha_threshold=alpha_threshold,
                    error="ValueError: missing authoritative srepr serialization",
                )
            )
            continue

        try:
            expr, source_serialization = parse_generated_expression(
                input_row,
                symbol_locals=symbol_locals,
            )
            if source_serialization != "srepr":
                raise ValueError("DAG compression study requires srepr source serialization")
            analysis = compute_expression_dag_analysis(expr)
            metrics = analysis.metrics
            pure_eml_valid = (
                analysis.eml_tree.alpha_valid
                and analysis.eml_tree.derived_leaf_count == 0
                and analysis.eml_tree.hidden_compound_leaf_count == 0
            )
            metric_rows.append(
                DagCompressionMetricsRow(
                    index=index,
                    expression=input_row.expression,
                    srepr=analysis.srepr,
                    supported=True,
                    ast_tree_node_count=metrics.ast_tree_node_count,
                    ast_dag_node_count=metrics.ast_dag_node_count,
                    ast_dag_child_ref_count=metrics.ast_dag_child_ref_count,
                    ast_tree_depth=metrics.ast_tree_depth,
                    ast_dag_depth=metrics.ast_dag_depth,
                    ast_dag_compression=metrics.ast_dag_compression,
                    eml_tree_node_count=metrics.eml_tree_node_count,
                    eml_dag_node_count=metrics.eml_dag_node_count,
                    eml_dag_child_ref_count=metrics.eml_dag_child_ref_count,
                    eml_tree_depth=metrics.eml_tree_depth,
                    eml_dag_depth=metrics.eml_dag_depth,
                    eml_dag_compression=metrics.eml_dag_compression,
                    tree_alpha=metrics.tree_alpha,
                    dag_alpha_vs_ast_tree=metrics.dag_alpha_vs_ast_tree,
                    dag_alpha_vs_ast_dag=metrics.dag_alpha_vs_ast_dag,
                    alpha_threshold_current=alpha_threshold,
                    below_threshold_tree=metrics.tree_alpha < alpha_threshold,
                    below_threshold_dag_vs_ast_tree=(
                        metrics.dag_alpha_vs_ast_tree < alpha_threshold
                    ),
                    below_threshold_dag_vs_ast_dag=(metrics.dag_alpha_vs_ast_dag < alpha_threshold),
                    pure_eml_valid=pure_eml_valid,
                    derived_leaf_count=analysis.eml_tree.derived_leaf_count,
                    hidden_compound_leaf_count=analysis.eml_tree.hidden_compound_leaf_count,
                    error=None,
                )
            )
        except Exception as exc:
            metric_rows.append(
                _unsupported_row(
                    index=index,
                    expression=input_row.expression,
                    srepr=input_row.srepr,
                    alpha_threshold=alpha_threshold,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return metric_rows


def write_metrics_jsonl(rows: Sequence[DagCompressionMetricsRow], path: Path) -> None:
    """Write DAG compression metric rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row.model_dump(mode="json"), sort_keys=True))
            jsonl_file.write("\n")


def write_metrics_csv(rows: Sequence[DagCompressionMetricsRow], path: Path) -> None:
    """Write DAG compression metric rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DAG_COMPRESSION_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def write_summary_json(rows: Sequence[DagCompressionMetricsRow], path: Path) -> None:
    """Write aggregate DAG compression summary JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_dag_compression_summary(rows)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def build_dag_compression_summary(rows: Sequence[DagCompressionMetricsRow]) -> dict[str, object]:
    """Build aggregate summary statistics for a DAG compression run."""
    supported_rows = [row for row in rows if row.supported]
    summary: dict[str, object] = {
        "processed_count": len(rows),
        "supported_count": len(supported_rows),
        "unsupported_count": len(rows) - len(supported_rows),
    }
    summary.update(_metric_summary(rows, field_name="tree_alpha"))
    summary.update(_metric_summary(rows, field_name="dag_alpha_vs_ast_tree"))
    summary.update(_metric_summary(rows, field_name="dag_alpha_vs_ast_dag"))
    summary.update(_metric_summary(rows, field_name="eml_dag_compression"))
    summary.update(
        {
            "percent_below_threshold_tree_alpha": _percent_true(
                row.below_threshold_tree for row in supported_rows
            ),
            "percent_below_threshold_dag_alpha_vs_ast_tree": _percent_true(
                row.below_threshold_dag_vs_ast_tree for row in supported_rows
            ),
            "percent_below_threshold_dag_alpha_vs_ast_dag": _percent_true(
                row.below_threshold_dag_vs_ast_dag for row in supported_rows
            ),
        }
    )
    return summary


def load_config(path: Path) -> DagCompressionStudyConfig:
    """Load a YAML DAG compression study config."""
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    return DagCompressionStudyConfig.model_validate(raw_config)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dag_compression_v0.yaml"),
        help="Path to a YAML DAG compression study config.",
    )
    parser.add_argument("--count", type=int, default=None, help="Optional expression count.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 3.3 DAG compression study."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.count is not None:
        config.count = args.count
    if args.seed is not None:
        config.seed = args.seed

    rows = run_dag_compression_study(config)
    summary = build_dag_compression_summary(rows)
    print(f"Processed: {summary['processed_count']}")
    print(f"Supported: {summary['supported_count']}")
    print(f"Unsupported: {summary['unsupported_count']}")
    print(f"Inputs JSONL: {config.input_jsonl_path}")
    print(f"Metrics JSONL: {config.metrics_jsonl_path}")
    print(f"Metrics CSV: {config.metrics_csv_path}")
    print(f"Summary JSON: {config.summary_json_path}")
    return 0


def _unsupported_row(
    *,
    index: int,
    expression: str,
    srepr: str,
    alpha_threshold: float,
    error: str,
) -> DagCompressionMetricsRow:
    return DagCompressionMetricsRow(
        index=index,
        expression=expression,
        srepr=srepr,
        supported=False,
        alpha_threshold_current=alpha_threshold,
        error=error,
    )


def _metric_summary(
    rows: Sequence[DagCompressionMetricsRow],
    *,
    field_name: str,
) -> dict[str, float | None]:
    values = sorted(
        float(value)
        for row in rows
        if (value := getattr(row, field_name)) is not None and row.supported
    )
    return {
        f"mean_{field_name}": statistics.fmean(values) if values else None,
        f"median_{field_name}": statistics.median(values) if values else None,
        f"p90_{field_name}": _percentile(values, 0.9) if values else None,
        f"p95_{field_name}": _percentile(values, 0.95) if values else None,
        f"max_{field_name}": max(values) if values else None,
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    index = max(0, min(len(sorted_values) - 1, math.ceil(percentile * len(sorted_values)) - 1))
    return sorted_values[index]


def _percent_true(values: Sequence[bool | None]) -> float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return 100 * sum(1 for value in valid_values if value) / len(valid_values)


if __name__ == "__main__":
    raise SystemExit(main())
