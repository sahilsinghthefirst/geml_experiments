"""Dataset metrics export for generated SymPy expressions."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Self

import sympy as sp
import yaml
from pydantic import BaseModel, Field, model_validator

from geml.symbolic.ast_graph import UnsupportedExpressionError as AstUnsupportedExpressionError
from geml.symbolic.ast_graph import sympy_to_ast_tree
from geml.symbolic.eml_transpile import UnsupportedExpressionError as EmlUnsupportedExpressionError
from geml.symbolic.eml_transpile import sympy_to_eml_tree
from geml.symbolic.metrics import TreeStatistics
from geml.symbolic.representations import REPRESENTATION_MODES, RepresentationMode

type MetadataValue = str | int | float | bool | None | dict[str, Any] | list[Any]
type SourceSerialization = Literal["srepr", "expression"]


class DatasetExportConfig(BaseModel):
    """Configuration for dataset metrics export."""

    input_jsonl_path: Path = Path("outputs/v0/dataset.jsonl")
    output_jsonl_path: Path = Path("outputs/v0/dataset_metrics.jsonl")
    output_csv_path: Path = Path("outputs/v0/dataset_metrics.csv")
    representation_mode: RepresentationMode = "restricted_eml_pure"
    symbol_names: tuple[str, ...] = ("x", "y")

    @model_validator(mode="after")
    def validate_symbol_names(self) -> Self:
        if not self.symbol_names:
            raise ValueError("symbol_names must not be empty")
        return self


class GeneratedExpressionInput(BaseModel):
    """Input row produced by the expression generator."""

    index: int | None = None
    expression: str
    srepr: str | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class DatasetMetricsRow(BaseModel):
    """Integrated AST/EML metrics row for one expression."""

    index: int
    expression: str
    srepr: str
    source_serialization: SourceSerialization
    representation_mode: RepresentationMode
    ast_stats: TreeStatistics | None
    eml_stats: TreeStatistics | None
    eml_normal_leaf_count: int | None = None
    eml_derived_leaf_count: int | None = None
    eml_hidden_compound_leaf_count: int | None = None
    alpha: float | None
    alpha_valid: bool
    alpha_threshold: float | None = None
    below_threshold: bool | None = None
    supported: bool
    error: str | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


def load_generated_expressions(path: Path) -> list[GeneratedExpressionInput]:
    """Load generated expression rows from JSONL."""
    rows: list[GeneratedExpressionInput] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw_row = json.loads(stripped)
            row = GeneratedExpressionInput.model_validate(raw_row)
            if row.index is None:
                row.index = len(rows)
            row.metadata.setdefault("source_line_number", line_number)
            rows.append(row)
    return rows


def build_symbol_locals(symbol_names: Iterable[str]) -> dict[str, sp.Symbol]:
    """Build SymPy parser locals for generated symbol names."""
    return {name: sp.Symbol(name) for name in symbol_names}


def build_srepr_locals() -> dict[str, Any]:
    """Build locals that reconstruct generated srepr without evaluating operators."""

    def add(*args: sp.Expr, **_: object) -> sp.Expr:
        return sp.Add(*args, evaluate=False)

    def mul(*args: sp.Expr, **_: object) -> sp.Expr:
        return sp.Mul(*args, evaluate=False)

    def exp(arg: sp.Expr, **_: object) -> sp.Expr:
        return sp.exp(arg, evaluate=False)

    def log(arg: sp.Expr, **_: object) -> sp.Expr:
        return sp.log(arg, evaluate=False)

    return {
        "Add": add,
        "Float": sp.Float,
        "Integer": sp.Integer,
        "Mul": mul,
        "Rational": sp.Rational,
        "Symbol": sp.Symbol,
        "exp": exp,
        "log": log,
    }


def parse_generated_expression(
    row: GeneratedExpressionInput,
    *,
    symbol_locals: dict[str, sp.Symbol],
) -> tuple[sp.Expr, SourceSerialization]:
    """Parse generated input, preferring structural srepr over display strings."""
    if row.srepr:
        return sp.sympify(row.srepr, locals=build_srepr_locals(), evaluate=False), "srepr"
    return sp.sympify(row.expression, locals=symbol_locals, evaluate=False), "expression"


def compute_metrics_rows(
    rows: Sequence[GeneratedExpressionInput],
    *,
    symbol_names: Iterable[str] = ("x", "y"),
    representation_mode: RepresentationMode = "restricted_eml_pure",
) -> list[DatasetMetricsRow]:
    """Compute AST and restricted EML metrics for generated expressions."""
    if representation_mode not in REPRESENTATION_MODES:
        allowed = ", ".join(REPRESENTATION_MODES)
        raise ValueError(f"unsupported representation_mode {representation_mode!r}; use {allowed}")
    symbol_locals = build_symbol_locals(symbol_names)
    metrics_rows: list[DatasetMetricsRow] = []

    for fallback_index, row in enumerate(rows):
        index = row.index if row.index is not None else fallback_index
        expression = row.expression
        source_serialization: SourceSerialization = "srepr" if row.srepr else "expression"
        try:
            expr, source_serialization = parse_generated_expression(
                row,
                symbol_locals=symbol_locals,
            )
            srepr = sp.srepr(expr)
        except Exception as exc:
            metrics_rows.append(
                DatasetMetricsRow(
                    index=index,
                    expression=expression,
                    srepr=row.srepr if row.srepr is not None else "",
                    source_serialization=source_serialization,
                    representation_mode=representation_mode,
                    ast_stats=None,
                    eml_stats=None,
                    alpha=None,
                    alpha_valid=False,
                    supported=False,
                    error=f"{type(exc).__name__}: {exc}",
                    metadata=row.metadata,
                )
            )
            continue

        try:
            ast_tree = sympy_to_ast_tree(expr)
        except (AstUnsupportedExpressionError, ValueError) as exc:
            metrics_rows.append(
                DatasetMetricsRow(
                    index=index,
                    expression=expression,
                    srepr=srepr,
                    source_serialization=source_serialization,
                    representation_mode=representation_mode,
                    ast_stats=None,
                    eml_stats=None,
                    alpha=None,
                    alpha_valid=False,
                    supported=False,
                    error=f"{type(exc).__name__}: {exc}",
                    metadata=row.metadata,
                )
            )
            continue

        if representation_mode == "ast":
            metrics_rows.append(
                DatasetMetricsRow(
                    index=index,
                    expression=expression,
                    srepr=srepr,
                    source_serialization=source_serialization,
                    representation_mode=representation_mode,
                    ast_stats=ast_tree.statistics,
                    eml_stats=None,
                    alpha=None,
                    alpha_valid=False,
                    supported=True,
                    error=None,
                    metadata=row.metadata,
                )
            )
            continue

        try:
            eml_tree = sympy_to_eml_tree(expr, representation_mode=representation_mode)
        except (EmlUnsupportedExpressionError, ValueError) as exc:
            metrics_rows.append(
                DatasetMetricsRow(
                    index=index,
                    expression=expression,
                    srepr=srepr,
                    source_serialization=source_serialization,
                    representation_mode=representation_mode,
                    ast_stats=ast_tree.statistics,
                    eml_stats=None,
                    alpha=None,
                    alpha_valid=False,
                    supported=False,
                    error=f"{type(exc).__name__}: {exc}",
                    metadata=row.metadata,
                )
            )
            continue

        metrics_rows.append(
            DatasetMetricsRow(
                index=index,
                expression=expression,
                srepr=srepr,
                source_serialization=source_serialization,
                representation_mode=representation_mode,
                ast_stats=ast_tree.statistics,
                eml_stats=eml_tree.statistics,
                eml_normal_leaf_count=eml_tree.normal_leaf_count,
                eml_derived_leaf_count=eml_tree.derived_leaf_count,
                eml_hidden_compound_leaf_count=eml_tree.hidden_compound_leaf_count,
                alpha=eml_tree.alpha,
                alpha_valid=eml_tree.alpha_valid,
                supported=True,
                error=None,
                metadata=row.metadata,
            )
        )

    return metrics_rows


def write_metrics_jsonl(rows: Sequence[DatasetMetricsRow], path: Path) -> None:
    """Write integrated metrics rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row.model_dump(mode="json"), sort_keys=True))
            jsonl_file.write("\n")


def write_metrics_csv(rows: Sequence[DatasetMetricsRow], path: Path) -> None:
    """Write a flattened metrics summary to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
        "alpha_threshold",
        "below_threshold",
        "alpha_valid",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_flatten_metrics_row(row))


def export_dataset_metrics(config: DatasetExportConfig) -> list[DatasetMetricsRow]:
    """Load generated expressions and export AST/EML metrics."""
    input_rows = load_generated_expressions(config.input_jsonl_path)
    metrics_rows = compute_metrics_rows(
        input_rows,
        symbol_names=config.symbol_names,
        representation_mode=config.representation_mode,
    )
    write_metrics_jsonl(metrics_rows, config.output_jsonl_path)
    write_metrics_csv(metrics_rows, config.output_csv_path)
    return metrics_rows


def load_config(path: Path) -> DatasetExportConfig:
    """Load a YAML dataset export config."""
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    return DatasetExportConfig.model_validate(raw_config)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config path.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Generated expression JSONL input path.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Integrated metrics JSONL output path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Integrated metrics CSV summary output path.",
    )
    parser.add_argument(
        "--representation-mode",
        choices=REPRESENTATION_MODES,
        default=None,
        help="Representation mode used for selected metrics and alpha.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run dataset metrics export."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config is not None else DatasetExportConfig()
    if args.input_jsonl is not None:
        config.input_jsonl_path = args.input_jsonl
    if args.output_jsonl is not None:
        config.output_jsonl_path = args.output_jsonl
    if args.output_csv is not None:
        config.output_csv_path = args.output_csv
    if args.representation_mode is not None:
        config.representation_mode = args.representation_mode

    rows = export_dataset_metrics(config)
    supported_count = sum(1 for row in rows if row.supported)
    print(f"Processed {len(rows)} expressions")
    print(f"Supported: {supported_count}")
    print(f"Unsupported: {len(rows) - supported_count}")
    print(f"JSONL: {config.output_jsonl_path}")
    print(f"CSV: {config.output_csv_path}")
    return 0


def _flatten_metrics_row(row: DatasetMetricsRow) -> dict[str, str | int | float | bool | None]:
    ast_stats = row.ast_stats
    eml_stats = row.eml_stats
    return {
        "index": row.index,
        "expression": row.expression,
        "srepr": row.srepr,
        "source_serialization": row.source_serialization,
        "representation_mode": row.representation_mode,
        "supported": row.supported,
        "error": row.error,
        "ast_node_count": ast_stats.node_count if ast_stats is not None else None,
        "ast_edge_count": ast_stats.edge_count if ast_stats is not None else None,
        "ast_depth": ast_stats.depth if ast_stats is not None else None,
        "ast_leaf_count": ast_stats.leaf_count if ast_stats is not None else None,
        "ast_operator_count": ast_stats.operator_count if ast_stats is not None else None,
        "eml_node_count": eml_stats.node_count if eml_stats is not None else None,
        "eml_edge_count": eml_stats.edge_count if eml_stats is not None else None,
        "eml_depth": eml_stats.depth if eml_stats is not None else None,
        "eml_leaf_count": eml_stats.leaf_count if eml_stats is not None else None,
        "eml_normal_leaf_count": row.eml_normal_leaf_count,
        "eml_derived_leaf_count": row.eml_derived_leaf_count,
        "eml_hidden_compound_leaf_count": row.eml_hidden_compound_leaf_count,
        "eml_operator_count": eml_stats.operator_count if eml_stats is not None else None,
        "alpha": row.alpha,
        "alpha_threshold": row.alpha_threshold,
        "below_threshold": row.below_threshold,
        "alpha_valid": row.alpha_valid,
    }


if __name__ == "__main__":
    raise SystemExit(main())
