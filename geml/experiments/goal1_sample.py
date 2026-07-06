"""Small end-to-end Goal 1 sample pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator

from geml.data.dataset import (
    DatasetMetricsRow,
    GeneratedExpressionInput,
    compute_metrics_rows,
    write_metrics_csv,
    write_metrics_jsonl,
)
from geml.data.generate_exprs import (
    DEFAULT_OPERATOR_PROBABILITIES,
    DEFAULT_SYMBOL_NAMES,
    ExpressionGeneratorConfig,
    SympyExpressionGenerator,
)


class Goal1SampleConfig(BaseModel):
    """Configuration for the Goal 1 sample pipeline."""

    count: int = Field(default=100, gt=0)
    seed: int = 0
    max_depth: int = Field(default=4, ge=0)
    output_jsonl_path: Path = Path("outputs/v0/goal1_sample.jsonl")
    output_csv_path: Path = Path("outputs/v0/goal1_summary.csv")
    operator_probabilities: dict[str, float] = Field(
        default_factory=lambda: DEFAULT_OPERATOR_PROBABILITIES.copy()
    )
    symbol_names: tuple[str, ...] = DEFAULT_SYMBOL_NAMES

    @model_validator(mode="after")
    def validate_symbol_names(self) -> Self:
        if not self.symbol_names:
            raise ValueError("symbol_names must not be empty")
        return self


def run_goal1_sample(config: Goal1SampleConfig) -> list[DatasetMetricsRow]:
    """Generate expressions, compute AST/EML metrics, and write sample outputs."""
    generator_config = ExpressionGeneratorConfig(
        count=config.count,
        seed=config.seed,
        max_depth=config.max_depth,
        operator_probabilities=config.operator_probabilities,
        symbol_names=config.symbol_names,
    )
    generated_records = SympyExpressionGenerator(generator_config).generate()
    input_rows = [
        GeneratedExpressionInput(
            index=record.index,
            expression=record.expression,
            srepr=record.srepr,
            metadata=record.metadata,
        )
        for record in generated_records
    ]
    metrics_rows = compute_metrics_rows(input_rows, symbol_names=config.symbol_names)
    write_metrics_jsonl(metrics_rows, config.output_jsonl_path)
    write_metrics_csv(metrics_rows, config.output_csv_path)
    return metrics_rows


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="Number of expressions to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--max-depth", type=int, default=4, help="Maximum generated expression depth."
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("outputs/v0/goal1_sample.jsonl"),
        help="Goal 1 sample JSONL output path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/v0/goal1_summary.csv"),
        help="Goal 1 summary CSV output path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 1 sample pipeline."""
    args = build_parser().parse_args(argv)
    config = Goal1SampleConfig(
        count=args.count,
        seed=args.seed,
        max_depth=args.max_depth,
        output_jsonl_path=args.output_jsonl,
        output_csv_path=args.output_csv,
    )
    rows = run_goal1_sample(config)
    supported_count = sum(1 for row in rows if row.supported)
    print(f"Generated: {len(rows)}")
    print(f"Supported: {supported_count}")
    print(f"Unsupported: {len(rows) - supported_count}")
    print(f"JSONL: {config.output_jsonl_path}")
    print(f"CSV: {config.output_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
