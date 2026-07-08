"""Goal 5.1 official macro graph baseline for ML-facing EML compression."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import sympy as sp
import yaml

from geml.compression.macro_graph import MACRO_REPRESENTATION_MODE, build_macro_graph
from geml.compression.macro_metrics import build_macro_graph_metrics
from geml.data.dataset import (
    GeneratedExpressionInput,
    build_symbol_locals,
    load_generated_expressions,
    parse_generated_expression,
)
from geml.experiments.stratified_expansion import (
    count_operator_features,
    dominant_operator_family,
    operator_signature,
)

type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]

TRIVIALITY_FEATURES = (
    "mul_by_one_count",
    "log_one_count",
    "exp_log_count",
    "log_exp_count",
    "constant_only_add_mul_count",
)

MACRO_GRAPH_METRICS_FIELDS = [
    "index",
    "expression",
    "srepr",
    "source_serialization",
    "representation_mode",
    "success",
    "subset_label",
    "operator_signature",
    "dominant_operator_family",
    "source_ast_tree_nodes",
    "source_ast_dag_nodes",
    "goal3_eml_tree_nodes",
    "goal3_eml_dag_nodes",
    "macro_graph_nodes",
    "macro_graph_edges_or_child_refs",
    "macro_graph_depth",
    "macro_graph_alpha_vs_ast_tree",
    "macro_graph_alpha_vs_ast_dag",
    "compression_gain_vs_goal3_eml_dag",
    "expansion_valid",
    "pure_eml_equivalent",
    "error",
]


@dataclass(frozen=True, slots=True)
class MacroGraphBaselineConfig:
    """Configuration for the Goal 5.1 v1 macro graph baseline."""

    seed: int = 0
    count: int = 10_000
    max_depth: int = 4
    operator_set: tuple[str, ...] = ("add", "mul", "exp", "log")
    symbol_names: tuple[str, ...] = ("x", "y")
    source_serialization: Literal["srepr"] = "srepr"
    input_jsonl_path: Path = Path("outputs/v1/dag_compression_inputs.jsonl")
    goal3_metrics_csv_path: Path = Path("outputs/v1/dag_compression_metrics.csv")
    metrics_csv_path: Path = Path("outputs/v1/goal5_macro_graph_metrics.csv")
    metrics_jsonl_path: Path = Path("outputs/v1/goal5_macro_graph_metrics.jsonl")
    summary_json_path: Path = Path("outputs/v1/goal5_macro_graph_summary.json")

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.max_depth != 4:
            raise ValueError("Goal 5.1 v1 config expects max_depth=4")
        if tuple(self.operator_set) != ("add", "mul", "exp", "log"):
            raise ValueError("Goal 5.1 v1 operator_set must be add, mul, exp, log")
        if tuple(self.symbol_names) != ("x", "y"):
            raise ValueError("Goal 5.1 v1 symbol_names must be x, y")
        if self.source_serialization != "srepr":
            raise ValueError("Goal 5.1 requires authoritative srepr input")
        _assert_no_outputs_v0(
            [
                self.input_jsonl_path,
                self.goal3_metrics_csv_path,
                self.metrics_csv_path,
                self.metrics_jsonl_path,
                self.summary_json_path,
            ]
        )
        _assert_outputs_v1(
            [
                self.input_jsonl_path,
                self.goal3_metrics_csv_path,
                self.metrics_csv_path,
                self.metrics_jsonl_path,
                self.summary_json_path,
            ]
        )


@dataclass(frozen=True, slots=True)
class Goal3BaselineRow:
    """Goal 3 v1 DAG baseline metrics for one expression."""

    index: int
    expression: str
    srepr: str
    source_serialization: str
    ast_tree_node_count: int
    ast_dag_node_count: int
    eml_tree_node_count: int
    eml_dag_node_count: int
    tree_alpha: float
    dag_alpha_vs_ast_tree: float
    dag_alpha_vs_ast_dag: float
    pure_eml_valid: bool


@dataclass(frozen=True, slots=True)
class MacroGraphBaselineRow:
    """One per-expression Goal 5.1 macro graph metric row."""

    index: int
    expression: str
    srepr: str
    source_serialization: str
    representation_mode: str
    success: bool
    subset_label: str
    operator_signature: str
    dominant_operator_family: str
    source_ast_tree_nodes: int
    source_ast_dag_nodes: int
    goal3_eml_tree_nodes: int
    goal3_eml_dag_nodes: int
    macro_graph_nodes: int | None
    macro_graph_edges_or_child_refs: int | None
    macro_graph_depth: int | None
    macro_graph_alpha_vs_ast_tree: float | None
    macro_graph_alpha_vs_ast_dag: float | None
    compression_gain_vs_goal3_eml_dag: float | None
    expansion_valid: bool
    pure_eml_equivalent: bool
    error: str | None

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable row."""
        return {
            "index": self.index,
            "expression": self.expression,
            "srepr": self.srepr,
            "source_serialization": self.source_serialization,
            "representation_mode": self.representation_mode,
            "success": self.success,
            "subset_label": self.subset_label,
            "operator_signature": self.operator_signature,
            "dominant_operator_family": self.dominant_operator_family,
            "source_ast_tree_nodes": self.source_ast_tree_nodes,
            "source_ast_dag_nodes": self.source_ast_dag_nodes,
            "goal3_eml_tree_nodes": self.goal3_eml_tree_nodes,
            "goal3_eml_dag_nodes": self.goal3_eml_dag_nodes,
            "macro_graph_nodes": self.macro_graph_nodes,
            "macro_graph_edges_or_child_refs": self.macro_graph_edges_or_child_refs,
            "macro_graph_depth": self.macro_graph_depth,
            "macro_graph_alpha_vs_ast_tree": self.macro_graph_alpha_vs_ast_tree,
            "macro_graph_alpha_vs_ast_dag": self.macro_graph_alpha_vs_ast_dag,
            "compression_gain_vs_goal3_eml_dag": self.compression_gain_vs_goal3_eml_dag,
            "expansion_valid": self.expansion_valid,
            "pure_eml_equivalent": self.pure_eml_equivalent,
            "error": self.error,
        }

    def to_csv_dict(self) -> dict[str, object]:
        """Return a CSV-serializable row."""
        return {
            field: _csv_value(self.to_json_dict()[field]) for field in MACRO_GRAPH_METRICS_FIELDS
        }


@dataclass(frozen=True, slots=True)
class MacroGraphBaselineResult:
    """Result summary for a complete Goal 5.1 run."""

    summary: dict[str, object]
    output_paths: tuple[Path, ...]


def load_config(path: Path) -> MacroGraphBaselineConfig:
    """Load a Goal 5.1 macro graph YAML config."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return MacroGraphBaselineConfig(
        **{key: _coerce_config_value(key, value) for key, value in raw.items()}
    )


def run_goal5_macro_graph_baseline(
    config: MacroGraphBaselineConfig,
) -> MacroGraphBaselineResult:
    """Run the official macro graph baseline on the configured v1 corpus."""
    started_at = time.time()
    input_rows = load_generated_expressions(config.input_jsonl_path)[: config.count]
    if len(input_rows) != config.count:
        raise ValueError(f"expected {config.count} input rows, found {len(input_rows)}")
    baselines = load_goal3_baselines(config.goal3_metrics_csv_path)
    missing_baselines = [row.index for row in input_rows if row.index not in baselines]
    if missing_baselines:
        raise ValueError(f"missing Goal 3 baselines for indices: {missing_baselines[:10]}")
    symbol_locals = build_symbol_locals(config.symbol_names)
    rows = tuple(
        compute_macro_graph_row(
            input_row,
            baselines[input_row.index],
            symbol_locals=symbol_locals,
            config=config,
        )
        for input_row in input_rows
        if input_row.index is not None
    )
    if len(rows) != len(input_rows):
        raise ValueError("one or more input rows had missing indices")
    write_jsonl_rows(rows, config.metrics_jsonl_path)
    write_metrics_csv(rows, config.metrics_csv_path)
    summary = build_summary(rows, config, started_at=started_at, completed_at=time.time())
    write_json(config.summary_json_path, summary)
    return MacroGraphBaselineResult(
        summary=summary,
        output_paths=(
            config.metrics_csv_path,
            config.metrics_jsonl_path,
            config.summary_json_path,
        ),
    )


def compute_macro_graph_row(
    input_row: GeneratedExpressionInput,
    baseline: Goal3BaselineRow,
    *,
    symbol_locals: dict[str, sp.Symbol],
    config: MacroGraphBaselineConfig,
) -> MacroGraphBaselineRow:
    """Compute one macro graph compression metric row."""
    if input_row.index is None:
        raise ValueError("input row index must not be None")
    subset_label = subset_label_for_metadata(input_row.metadata)
    expression = input_row.expression
    srepr = input_row.srepr or baseline.srepr
    operator_metadata = _operator_metadata(srepr)
    base_kwargs = {
        "index": input_row.index,
        "expression": expression,
        "srepr": srepr,
        "source_serialization": baseline.source_serialization,
        "representation_mode": MACRO_REPRESENTATION_MODE,
        "subset_label": subset_label,
        "operator_signature": operator_metadata["operator_signature"],
        "dominant_operator_family": operator_metadata["dominant_operator_family"],
        "source_ast_tree_nodes": baseline.ast_tree_node_count,
        "source_ast_dag_nodes": baseline.ast_dag_node_count,
        "goal3_eml_tree_nodes": baseline.eml_tree_node_count,
        "goal3_eml_dag_nodes": baseline.eml_dag_node_count,
    }
    try:
        if not baseline.pure_eml_valid:
            raise ValueError("Goal 3 baseline row is not valid pure EML")
        sympy_expr, source_serialization = parse_generated_expression(
            input_row,
            symbol_locals=symbol_locals,
        )
        if source_serialization != config.source_serialization:
            raise ValueError(
                f"expected {config.source_serialization} input, got {source_serialization}"
            )
        actual_srepr = sp.srepr(sympy_expr)
        macro_graph = build_macro_graph(sympy_expr)
        metrics = build_macro_graph_metrics(
            sympy_expr,
            macro_graph=macro_graph,
            source_ast_tree_nodes=baseline.ast_tree_node_count,
            source_ast_dag_nodes=baseline.ast_dag_node_count,
            goal3_eml_tree_nodes=baseline.eml_tree_node_count,
            goal3_eml_dag_nodes=baseline.eml_dag_node_count,
        )
        success = metrics.expansion_valid and metrics.pure_eml_equivalent
        success_kwargs = {
            **base_kwargs,
            "expression": str(sympy_expr),
            "srepr": input_row.srepr or actual_srepr,
            "source_serialization": source_serialization,
        }
        return MacroGraphBaselineRow(
            **success_kwargs,
            success=success,
            macro_graph_nodes=metrics.macro_graph_nodes,
            macro_graph_edges_or_child_refs=metrics.macro_graph_edges_or_child_refs,
            macro_graph_depth=metrics.macro_graph_depth,
            macro_graph_alpha_vs_ast_tree=metrics.macro_graph_alpha_vs_ast_tree,
            macro_graph_alpha_vs_ast_dag=metrics.macro_graph_alpha_vs_ast_dag,
            compression_gain_vs_goal3_eml_dag=metrics.compression_gain_vs_goal3_eml_dag,
            expansion_valid=metrics.expansion_valid,
            pure_eml_equivalent=metrics.pure_eml_equivalent,
            error=metrics.validation_error,
        )
    except Exception as exc:
        return MacroGraphBaselineRow(
            **base_kwargs,
            success=False,
            macro_graph_nodes=None,
            macro_graph_edges_or_child_refs=None,
            macro_graph_depth=None,
            macro_graph_alpha_vs_ast_tree=None,
            macro_graph_alpha_vs_ast_dag=None,
            compression_gain_vs_goal3_eml_dag=None,
            expansion_valid=False,
            pure_eml_equivalent=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def load_goal3_baselines(path: Path) -> dict[int, Goal3BaselineRow]:
    """Load v1 Goal 3 DAG metrics keyed by expression index."""
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        rows: dict[int, Goal3BaselineRow] = {}
        for raw_row in csv.DictReader(csv_file):
            if raw_row.get("supported") != "True":
                continue
            row = Goal3BaselineRow(
                index=int(raw_row["index"]),
                expression=raw_row["expression"],
                srepr=raw_row["srepr"],
                source_serialization=raw_row["source_serialization"],
                ast_tree_node_count=int(raw_row["ast_tree_node_count"]),
                ast_dag_node_count=int(raw_row["ast_dag_node_count"]),
                eml_tree_node_count=int(raw_row["eml_tree_node_count"]),
                eml_dag_node_count=int(raw_row["eml_dag_node_count"]),
                tree_alpha=float(raw_row["tree_alpha"]),
                dag_alpha_vs_ast_tree=float(raw_row["dag_alpha_vs_ast_tree"]),
                dag_alpha_vs_ast_dag=float(raw_row["dag_alpha_vs_ast_dag"]),
                pure_eml_valid=_parse_bool(raw_row["pure_eml_valid"]),
            )
            rows[row.index] = row
        return rows


def build_summary(
    rows: Sequence[MacroGraphBaselineRow],
    config: MacroGraphBaselineConfig,
    *,
    started_at: float,
    completed_at: float,
) -> dict[str, object]:
    """Build required macro graph summary metrics."""
    success_rows = [row for row in rows if row.success]
    expansion_failures = [row for row in rows if not row.expansion_valid]
    return {
        "config": config_to_json_dict(config),
        "representation_contract": {
            "representation_mode": MACRO_REPRESENTATION_MODE,
            "is_pure_eml": False,
            "expansion_to_pure_eml_available": True,
            "uses_official_compiler_formulas": True,
            "macro_graph_size_is_not_pure_eml_alpha": True,
        },
        "processed_count": len(rows),
        "success_count": len(success_rows),
        "expansion_validation_failure_count": len(expansion_failures),
        "median_macro_graph_alpha": _median(
            row.macro_graph_alpha_vs_ast_tree for row in success_rows
        ),
        "median_macro_graph_alpha_vs_ast_tree": _median(
            row.macro_graph_alpha_vs_ast_tree for row in success_rows
        ),
        "median_macro_graph_alpha_vs_ast_dag": _median(
            row.macro_graph_alpha_vs_ast_dag for row in success_rows
        ),
        "median_compression_gain_vs_goal3_eml_dag": _median(
            row.compression_gain_vs_goal3_eml_dag for row in success_rows
        ),
        "macro_graph_alpha_vs_ast_tree": _distribution(
            row.macro_graph_alpha_vs_ast_tree for row in success_rows
        ),
        "compression_gain_vs_goal3_eml_dag": _distribution(
            row.compression_gain_vs_goal3_eml_dag for row in success_rows
        ),
        "results_by_subset_label": {
            label: summarize_subset(rows, label)
            for label in ("all_v1", "nontrivial_v1", "identity_heavy_v1")
        },
        "operator_family_summaries": {
            "by_dominant_operator_family": summarize_by_field(
                rows,
                field_name="dominant_operator_family",
            ),
            "by_operator_signature": summarize_by_field(rows, field_name="operator_signature"),
        },
        "elapsed_seconds": completed_at - started_at,
        "completed_at_unix": completed_at,
    }


def summarize_subset(rows: Sequence[MacroGraphBaselineRow], label: str) -> dict[str, object]:
    """Summarize one required v1 subset label."""
    subset_rows = (
        list(rows) if label == "all_v1" else [row for row in rows if row.subset_label == label]
    )
    success_rows = [row for row in subset_rows if row.success]
    return {
        "processed_count": len(subset_rows),
        "success_count": len(success_rows),
        "expansion_validation_failure_count": sum(not row.expansion_valid for row in subset_rows),
        "median_macro_graph_alpha": _median(
            row.macro_graph_alpha_vs_ast_tree for row in success_rows
        ),
        "median_macro_graph_alpha_vs_ast_dag": _median(
            row.macro_graph_alpha_vs_ast_dag for row in success_rows
        ),
        "median_compression_gain_vs_goal3_eml_dag": _median(
            row.compression_gain_vs_goal3_eml_dag for row in success_rows
        ),
        "macro_graph_nodes": _distribution(row.macro_graph_nodes for row in success_rows),
        "macro_graph_depth": _distribution(row.macro_graph_depth for row in success_rows),
        "macro_graph_alpha_vs_ast_tree": _distribution(
            row.macro_graph_alpha_vs_ast_tree for row in success_rows
        ),
        "compression_gain_vs_goal3_eml_dag": _distribution(
            row.compression_gain_vs_goal3_eml_dag for row in success_rows
        ),
    }


def summarize_by_field(
    rows: Sequence[MacroGraphBaselineRow],
    *,
    field_name: Literal["dominant_operator_family", "operator_signature"],
) -> list[dict[str, object]]:
    """Summarize rows by an operator family field."""
    grouped: dict[str, list[MacroGraphBaselineRow]] = {}
    for row in rows:
        grouped.setdefault(str(getattr(row, field_name)), []).append(row)
    summaries = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        success_rows = [row for row in group_rows if row.success]
        summaries.append(
            {
                field_name: key,
                "processed_count": len(group_rows),
                "success_count": len(success_rows),
                "expansion_validation_failure_count": sum(
                    not row.expansion_valid for row in group_rows
                ),
                "median_macro_graph_alpha": _median(
                    row.macro_graph_alpha_vs_ast_tree for row in success_rows
                ),
                "median_compression_gain_vs_goal3_eml_dag": _median(
                    row.compression_gain_vs_goal3_eml_dag for row in success_rows
                ),
                "macro_graph_nodes": _distribution(row.macro_graph_nodes for row in success_rows),
                "compression_gain_vs_goal3_eml_dag": _distribution(
                    row.compression_gain_vs_goal3_eml_dag for row in success_rows
                ),
            }
        )
    return summaries


def subset_label_for_metadata(metadata: dict[str, MetadataValue]) -> str:
    """Assign subset label using measured v1 nontriviality feature counters."""
    raw_features = metadata.get("nontriviality")
    if not isinstance(raw_features, dict):
        return "all_v1"
    feature_values = {
        feature: _int_value(raw_features.get(feature, 0)) for feature in TRIVIALITY_FEATURES
    }
    if any(value > 0 for value in feature_values.values()):
        return "identity_heavy_v1"
    return "nontrivial_v1"


def write_jsonl_rows(rows: Sequence[MacroGraphBaselineRow], path: Path) -> None:
    """Write per-expression macro graph metrics to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row.to_json_dict(), sort_keys=True) + "\n")


def write_metrics_csv(rows: Sequence[MacroGraphBaselineRow], path: Path) -> None:
    """Write per-expression macro graph metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MACRO_GRAPH_METRICS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def write_json(path: Path, data: dict[str, object]) -> None:
    """Write a deterministic JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def config_to_json_dict(config: MacroGraphBaselineConfig) -> dict[str, object]:
    """Return JSON-safe config values."""
    return {
        "seed": config.seed,
        "count": config.count,
        "max_depth": config.max_depth,
        "operator_set": list(config.operator_set),
        "symbol_names": list(config.symbol_names),
        "source_serialization": config.source_serialization,
        "input_jsonl_path": str(config.input_jsonl_path),
        "goal3_metrics_csv_path": str(config.goal3_metrics_csv_path),
        "metrics_csv_path": str(config.metrics_csv_path),
        "metrics_jsonl_path": str(config.metrics_jsonl_path),
        "summary_json_path": str(config.summary_json_path),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/macro_graph_v1.yaml"),
        help="Path to a YAML Goal 5.1 macro graph config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 5.1 macro graph baseline."""
    args = build_parser().parse_args(argv)
    result = run_goal5_macro_graph_baseline(load_config(args.config))
    print(f"Processed: {result.summary['processed_count']}")
    print(f"Succeeded: {result.summary['success_count']}")
    for path in result.output_paths:
        print(f"Wrote: {path}")
    return 0


def _operator_metadata(srepr: str) -> dict[str, str]:
    features = count_operator_features(srepr)
    return {
        "operator_signature": operator_signature(features),
        "dominant_operator_family": dominant_operator_family(features),
    }


def _distribution(values: Iterable[int | float | None]) -> dict[str, float | None]:
    numeric_values = [
        float(value) for value in values if value is not None and math.isfinite(float(value))
    ]
    if not numeric_values:
        return {"mean": None, "median": None, "p90": None}
    return {
        "mean": statistics.fmean(numeric_values),
        "median": statistics.median(numeric_values),
        "p90": _quantile(numeric_values, 0.9),
    }


def _median(values: Iterable[int | float | None]) -> float | None:
    numeric_values = [
        float(value) for value in values if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(numeric_values) if numeric_values else None


def _quantile(values: Sequence[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected boolean string, got {value!r}")


def _coerce_config_value(key: str, value: object) -> object:
    path_keys = {
        "input_jsonl_path",
        "goal3_metrics_csv_path",
        "metrics_csv_path",
        "metrics_jsonl_path",
        "summary_json_path",
    }
    tuple_keys = {"operator_set", "symbol_names"}
    if key in path_keys and isinstance(value, str):
        return Path(value)
    if key in tuple_keys and isinstance(value, list):
        return tuple(str(item) for item in value)
    return value


def _assert_no_outputs_v0(paths: Iterable[Path]) -> None:
    bad_paths = [path for path in paths if "outputs/v0" in path.as_posix()]
    if bad_paths:
        raise ValueError(f"Goal 5.1 macro graph baseline must not use outputs/v0: {bad_paths}")


def _assert_outputs_v1(paths: Iterable[Path]) -> None:
    bad_paths = [path for path in paths if "outputs/v1" not in path.as_posix()]
    if bad_paths:
        raise ValueError(f"Goal 5.1 macro graph baseline expects outputs/v1 paths: {bad_paths}")


def _csv_value(value: object) -> object:
    return "" if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
