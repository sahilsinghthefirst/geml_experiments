"""Bounded-depth SymPy expression generation for Goal 1."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import sympy as sp
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

type MetadataValue = str | int | float | bool | list[str] | dict[str, float]

ALLOWED_OPERATORS = frozenset({"add", "mul", "exp", "log"})
DEFAULT_OPERATOR_PROBABILITIES = {
    "add": 0.3,
    "mul": 0.3,
    "exp": 0.2,
    "log": 0.2,
}
DEFAULT_SYMBOL_NAMES = ("x", "y")


class ExpressionGeneratorConfig(BaseModel):
    """Configuration for bounded symbolic expression generation."""

    seed: int = 0
    count: int = Field(default=1000, gt=0)
    max_depth: int = Field(default=4, ge=0)
    output_dir: Path = Path("outputs/v0")
    jsonl_path: Path | None = None
    csv_path: Path | None = None
    operator_probabilities: dict[str, float] = Field(
        default_factory=lambda: DEFAULT_OPERATOR_PROBABILITIES.copy()
    )
    symbol_names: tuple[str, ...] = DEFAULT_SYMBOL_NAMES

    @field_validator("operator_probabilities")
    @classmethod
    def validate_operator_probabilities(cls, probabilities: dict[str, float]) -> dict[str, float]:
        """Require known operators with non-negative weights and at least one positive weight."""
        unknown = set(probabilities) - ALLOWED_OPERATORS
        if unknown:
            unknown_text = ", ".join(sorted(unknown))
            raise ValueError(f"unknown operators: {unknown_text}")
        if not probabilities:
            raise ValueError("operator_probabilities must not be empty")
        if any(weight < 0 for weight in probabilities.values()):
            raise ValueError("operator weights must be non-negative")
        if sum(probabilities.values()) <= 0:
            raise ValueError("at least one operator weight must be positive")
        return probabilities

    @field_validator("symbol_names")
    @classmethod
    def validate_symbol_names(cls, symbol_names: tuple[str, ...]) -> tuple[str, ...]:
        """Require at least one symbolic leaf."""
        if not symbol_names:
            raise ValueError("symbol_names must not be empty")
        return symbol_names

    @model_validator(mode="after")
    def set_default_output_paths(self) -> Self:
        """Derive output paths from output_dir when explicit paths are omitted."""
        if self.jsonl_path is None:
            self.jsonl_path = self.output_dir / "dataset.jsonl"
        if self.csv_path is None:
            self.csv_path = self.output_dir / "dataset.csv"
        return self


class GeneratedExpression(BaseModel):
    """Serializable expression record."""

    index: int
    expression: str
    srepr: str
    depth: int
    metadata: dict[str, MetadataValue]


class SympyExpressionGenerator:
    """Generate bounded-depth expressions over x, y, 1, Add, Mul, Exp, and Log."""

    def __init__(self, config: ExpressionGeneratorConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._symbols = tuple(sp.Symbol(name) for name in config.symbol_names)
        self._leaves = self._symbols + (sp.Integer(1),)

    def generate(self) -> list[GeneratedExpression]:
        """Generate configured expression records."""
        records: list[GeneratedExpression] = []
        for index in range(self.config.count):
            expr = self._generate_expr(self.config.max_depth)
            records.append(
                GeneratedExpression(
                    index=index,
                    expression=str(expr),
                    srepr=sp.srepr(expr),
                    depth=expression_depth(expr),
                    metadata={
                        "seed": self.config.seed,
                        "max_depth": self.config.max_depth,
                        "operator_probabilities": self.config.operator_probabilities,
                        "symbol_names": list(self.config.symbol_names),
                    },
                )
            )
        return records

    def _generate_expr(self, remaining_depth: int) -> sp.Expr:
        if remaining_depth <= 0:
            return self._generate_leaf()

        operator = self._choose_operator()
        next_depth = remaining_depth - 1

        if operator == "add":
            return sp.Add(
                self._generate_expr(next_depth),
                self._generate_expr(next_depth),
                evaluate=False,
            )
        if operator == "mul":
            return sp.Mul(
                self._generate_expr(next_depth),
                self._generate_expr(next_depth),
                evaluate=False,
            )
        if operator == "exp":
            return sp.exp(self._generate_expr(next_depth), evaluate=False)
        if operator == "log":
            return sp.log(self._generate_log_argument(next_depth), evaluate=False)

        raise AssertionError(f"unsupported operator selected: {operator}")

    def _generate_log_argument(self, remaining_depth: int) -> sp.Expr:
        """Prefer log arguments that are structurally valid and not obviously non-positive."""
        if remaining_depth <= 0:
            return sp.Integer(1)

        inner_depth = max(remaining_depth - 1, 0)
        return sp.exp(self._generate_expr(inner_depth), evaluate=False)

    def _generate_leaf(self) -> sp.Expr:
        return self._rng.choice(self._leaves)

    def _choose_operator(self) -> str:
        total = sum(self.config.operator_probabilities.values())
        threshold = self._rng.random() * total
        running = 0.0
        for operator, weight in self.config.operator_probabilities.items():
            running += weight
            if threshold <= running:
                return operator
        return next(reversed(self.config.operator_probabilities))


def expression_depth(expr: sp.Expr) -> int:
    """Return expression tree depth with leaves at depth 0."""
    if not expr.args:
        return 0
    return 1 + max(expression_depth(arg) for arg in expr.args)


def load_config(path: Path) -> ExpressionGeneratorConfig:
    """Load a YAML generator config."""
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    return ExpressionGeneratorConfig.model_validate(raw_config)


def write_jsonl(records: Sequence[GeneratedExpression], path: Path) -> None:
    """Write generated records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record.model_dump(mode="json"), sort_keys=True))
            jsonl_file.write("\n")


def write_csv(records: Sequence[GeneratedExpression], path: Path) -> None:
    """Write generated records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "expression", "srepr", "depth", "metadata"]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.model_dump(mode="json")
            row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
            writer.writerow(row)


def generate_dataset(config: ExpressionGeneratorConfig) -> list[GeneratedExpression]:
    """Generate expressions and save JSONL/CSV outputs."""
    generator = SympyExpressionGenerator(config)
    records = generator.generate()
    if config.jsonl_path is not None:
        write_jsonl(records, config.jsonl_path)
    if config.csv_path is not None:
        write_csv(records, config.csv_path)
    return records


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_v0.yaml"),
        help="Path to a YAML expression generator config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run expression generation from a YAML config."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    records = generate_dataset(config)
    print(f"Generated {len(records)} expressions")
    print(f"JSONL: {config.jsonl_path}")
    print(f"CSV: {config.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
