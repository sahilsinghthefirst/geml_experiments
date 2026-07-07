"""Bounded-depth SymPy expression generation for Goal 1."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import sympy as sp
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from geml.symbolic.srepr import parse_srepr

type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]
type LogArgumentStrategy = Literal["exp_wrap", "positive_domain"]

ALLOWED_OPERATORS = frozenset({"add", "mul", "exp", "log"})
ALLOWED_POSITIVE_LOG_OPERATORS = frozenset({"leaf", "add", "mul", "exp"})
DEFAULT_OPERATOR_PROBABILITIES = {
    "add": 0.3,
    "mul": 0.3,
    "exp": 0.2,
    "log": 0.2,
}
DEFAULT_POSITIVE_LOG_ARGUMENT_PROBABILITIES = {
    "leaf": 0.35,
    "add": 0.3,
    "mul": 0.25,
    "exp": 0.1,
}
DEFAULT_SYMBOL_NAMES = ("x", "y")


class NontrivialityMetrics(BaseModel):
    """Counts of corpus artifacts that can make generated expressions too trivial."""

    mul_by_one_count: int = 0
    constant_only_add_mul_count: int = 0
    log_one_count: int = 0
    exp_log_count: int = 0
    log_exp_count: int = 0

    @property
    def total_score(self) -> int:
        """Aggregate count used by optional rejection capping."""
        return (
            self.mul_by_one_count
            + self.constant_only_add_mul_count
            + self.log_one_count
            + self.exp_log_count
            + self.log_exp_count
        )


class ExpressionGenerationSummary(BaseModel):
    """Corpus-level generation quality report."""

    requested_count: int
    generated_count: int
    seed: int
    max_depth: int
    attempts: int
    unique_srepr_count: int
    output_duplicate_count: int
    output_duplicate_rate: float
    duplicate_rejection_count: int = 0
    duplicate_rejection_rate: float = 0.0
    triviality_rejection_count: int = 0
    triviality_rejection_rate: float = 0.0
    actual_depth_histogram: dict[str, int]
    target_depth_histogram: dict[str, int]
    log_argument_distribution: dict[str, int]
    nontriviality_totals: dict[str, int]
    nontriviality_rates: dict[str, float]


class ExpressionGeneratorConfig(BaseModel):
    """Configuration for bounded symbolic expression generation."""

    seed: int = 0
    count: int = Field(default=1000, gt=0)
    max_depth: int = Field(default=4, ge=0)
    output_dir: Path = Path("outputs/v0")
    jsonl_path: Path | None = None
    csv_path: Path | None = None
    summary_json_path: Path | None = None
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

    @field_validator("positive_log_argument_probabilities")
    @classmethod
    def validate_positive_log_argument_probabilities(
        cls,
        probabilities: dict[str, float],
    ) -> dict[str, float]:
        """Require known positive-log grammar operators with valid weights."""
        unknown = set(probabilities) - ALLOWED_POSITIVE_LOG_OPERATORS
        if unknown:
            unknown_text = ", ".join(sorted(unknown))
            raise ValueError(f"unknown positive log-argument operators: {unknown_text}")
        if not probabilities:
            raise ValueError("positive_log_argument_probabilities must not be empty")
        if any(weight < 0 for weight in probabilities.values()):
            raise ValueError("positive log-argument weights must be non-negative")
        if sum(probabilities.values()) <= 0:
            raise ValueError("at least one positive log-argument weight must be positive")
        return probabilities

    @field_validator("target_depth_probabilities")
    @classmethod
    def validate_target_depth_probabilities(
        cls,
        probabilities: dict[int, float] | None,
    ) -> dict[int, float] | None:
        """Require non-negative depth weights when target-depth sampling is enabled."""
        if probabilities is None:
            return None
        if not probabilities:
            raise ValueError("target_depth_probabilities must not be empty")
        if any(depth < 0 for depth in probabilities):
            raise ValueError("target depths must be non-negative")
        if any(weight < 0 for weight in probabilities.values()):
            raise ValueError("target-depth weights must be non-negative")
        if sum(probabilities.values()) <= 0:
            raise ValueError("at least one target-depth weight must be positive")
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

    @model_validator(mode="after")
    def validate_target_depth_bounds(self) -> Self:
        """Reject target-depth weights outside the configured maximum depth."""
        if self.target_depth_probabilities is None:
            return self
        too_deep = [depth for depth in self.target_depth_probabilities if depth > self.max_depth]
        if too_deep:
            depth_text = ", ".join(str(depth) for depth in sorted(too_deep))
            raise ValueError(f"target depths exceed max_depth={self.max_depth}: {depth_text}")
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
        self.summary: ExpressionGenerationSummary | None = None

    def generate(self) -> list[GeneratedExpression]:
        """Generate configured expression records."""
        records: list[GeneratedExpression] = []
        seen_sreprs: set[str] = set()
        attempts = 0
        duplicate_rejections = 0
        triviality_rejections = 0
        max_attempts = self._max_generation_attempts()

        while len(records) < self.config.count:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "failed to generate requested unique/nontrivial expression count "
                    f"after {max_attempts} attempts"
                )

            target_depth = self._sample_target_depth()
            expr = self._generate_expr(target_depth)
            srepr = sp.srepr(expr)
            nontriviality = compute_nontriviality_metrics(expr)
            if (
                self.config.max_triviality_score is not None
                and nontriviality.total_score > self.config.max_triviality_score
            ):
                triviality_rejections += 1
                continue

            if self.config.deduplicate_srepr and srepr in seen_sreprs:
                duplicate_rejections += 1
                continue

            seen_sreprs.add(srepr)
            index = len(records)
            records.append(
                GeneratedExpression(
                    index=index,
                    expression=str(expr),
                    srepr=srepr,
                    depth=expression_depth(expr),
                    metadata={
                        "seed": self.config.seed,
                        "max_depth": self.config.max_depth,
                        "target_depth": target_depth,
                        "attempt": attempts,
                        "intermediate_leaf_probability": (
                            self.config.intermediate_leaf_probability
                        ),
                        "deduplicate_srepr": self.config.deduplicate_srepr,
                        "log_argument_strategy": self.config.log_argument_strategy,
                        "max_triviality_score": self.config.max_triviality_score,
                        "nontriviality": nontriviality.model_dump(mode="json"),
                        "operator_probabilities": self.config.operator_probabilities,
                        "target_depth_probabilities": (
                            {
                                str(depth): weight
                                for depth, weight in self.config.target_depth_probabilities.items()
                            }
                            if self.config.target_depth_probabilities is not None
                            else {"max_depth": 1.0}
                        ),
                        "positive_log_argument_probabilities": (
                            self.config.positive_log_argument_probabilities
                        ),
                        "symbol_names": list(self.config.symbol_names),
                    },
                )
            )
        self.summary = summarize_generated_records(
            records,
            config=self.config,
            attempts=attempts,
            duplicate_rejections=duplicate_rejections,
            triviality_rejections=triviality_rejections,
        )
        return records

    def _generate_expr(self, remaining_depth: int) -> sp.Expr:
        if remaining_depth <= 0:
            return self._generate_leaf()

        if self._rng.random() < self.config.intermediate_leaf_probability:
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
        if self.config.log_argument_strategy == "positive_domain":
            return self._generate_positive_expr(remaining_depth)

        if remaining_depth <= 0:
            return sp.Integer(1)

        inner_depth = max(remaining_depth - 1, 0)
        return sp.exp(self._generate_expr(inner_depth), evaluate=False)

    def _generate_positive_expr(self, remaining_depth: int) -> sp.Expr:
        """Generate an expression positive on the configured positive real domain."""
        if remaining_depth <= 0:
            return self._generate_positive_leaf()

        if self._rng.random() < self.config.intermediate_leaf_probability:
            return self._generate_positive_leaf()

        operator = self._choose_positive_log_operator()
        next_depth = remaining_depth - 1
        if operator == "leaf":
            return self._generate_positive_leaf()
        if operator == "add":
            return sp.Add(
                self._generate_positive_expr(next_depth),
                self._generate_positive_expr(next_depth),
                evaluate=False,
            )
        if operator == "mul":
            return sp.Mul(
                self._generate_positive_expr(next_depth),
                self._generate_positive_expr(next_depth),
                evaluate=False,
            )
        if operator == "exp":
            return sp.exp(self._generate_expr(next_depth), evaluate=False)
        raise AssertionError(f"unsupported positive log-argument operator selected: {operator}")

    def _generate_positive_leaf(self) -> sp.Expr:
        return self._rng.choice(self._leaves)

    def _generate_leaf(self) -> sp.Expr:
        return self._rng.choice(self._leaves)

    def _choose_operator(self) -> str:
        return self._choose_weighted(self.config.operator_probabilities)

    def _choose_positive_log_operator(self) -> str:
        return self._choose_weighted(self.config.positive_log_argument_probabilities)

    def _choose_weighted(self, probabilities: dict[str, float]) -> str:
        total = sum(probabilities.values())
        threshold = self._rng.random() * total
        running = 0.0
        for operator, weight in probabilities.items():
            running += weight
            if threshold <= running:
                return operator
        return next(reversed(probabilities))

    def _sample_target_depth(self) -> int:
        if self.config.target_depth_probabilities is None:
            return self.config.max_depth
        total = sum(self.config.target_depth_probabilities.values())
        threshold = self._rng.random() * total
        running = 0.0
        for depth, weight in sorted(self.config.target_depth_probabilities.items()):
            running += weight
            if threshold <= running:
                return depth
        return max(self.config.target_depth_probabilities)

    def _max_generation_attempts(self) -> int:
        if self.config.max_generation_attempts is not None:
            return self.config.max_generation_attempts
        if self.config.deduplicate_srepr or self.config.max_triviality_score is not None:
            return max(self.config.count * 200, self.config.count + 1000)
        return self.config.count


def expression_depth(expr: sp.Expr) -> int:
    """Return expression tree depth with leaves at depth 0."""
    if not expr.args:
        return 0
    return 1 + max(expression_depth(arg) for arg in expr.args)


def compute_nontriviality_metrics(expr: sp.Expr) -> NontrivialityMetrics:
    """Count structural trivialities in a generated expression."""
    metrics = NontrivialityMetrics()
    for node in sp.preorder_traversal(expr):
        if node.func == sp.Mul and any(arg == sp.Integer(1) for arg in node.args):
            metrics.mul_by_one_count += 1
        if node.func in {sp.Add, sp.Mul} and node.args and all(arg.is_number for arg in node.args):
            metrics.constant_only_add_mul_count += 1
        if node.func == sp.log and len(node.args) == 1:
            if node.args[0] == sp.Integer(1):
                metrics.log_one_count += 1
            if node.args[0].func == sp.exp:
                metrics.log_exp_count += 1
        if node.func == sp.exp and len(node.args) == 1 and node.args[0].func == sp.log:
            metrics.exp_log_count += 1
    return metrics


def summarize_generated_records(
    records: Sequence[GeneratedExpression],
    *,
    config: ExpressionGeneratorConfig,
    attempts: int | None = None,
    duplicate_rejections: int = 0,
    triviality_rejections: int = 0,
) -> ExpressionGenerationSummary:
    """Build a corpus-level generation quality report from generated rows."""
    attempts = len(records) if attempts is None else attempts
    srepr_counts = Counter(record.srepr for record in records)
    duplicate_count = sum(count - 1 for count in srepr_counts.values() if count > 1)
    actual_depths = Counter(str(record.depth) for record in records)
    target_depths = Counter(
        str(record.metadata.get("target_depth", "unknown")) for record in records
    )
    log_arguments: Counter[str] = Counter()
    nontriviality_totals: Counter[str] = Counter()

    for record in records:
        expr = parse_srepr(record.srepr)
        log_arguments.update(classify_log_arguments(expr))
        metrics = compute_nontriviality_metrics(expr)
        for key, value in metrics.model_dump(mode="json").items():
            nontriviality_totals[key] += int(value)

    generated_count = len(records)
    nontriviality_rates = {
        key: (value / generated_count if generated_count else 0.0)
        for key, value in sorted(nontriviality_totals.items())
    }
    return ExpressionGenerationSummary(
        requested_count=config.count,
        generated_count=generated_count,
        seed=config.seed,
        max_depth=config.max_depth,
        attempts=attempts,
        unique_srepr_count=len(srepr_counts),
        output_duplicate_count=duplicate_count,
        output_duplicate_rate=duplicate_count / generated_count if generated_count else 0.0,
        duplicate_rejection_count=duplicate_rejections,
        duplicate_rejection_rate=duplicate_rejections / attempts if attempts else 0.0,
        triviality_rejection_count=triviality_rejections,
        triviality_rejection_rate=triviality_rejections / attempts if attempts else 0.0,
        actual_depth_histogram=dict(sorted(actual_depths.items(), key=lambda item: item[0])),
        target_depth_histogram=dict(sorted(target_depths.items(), key=lambda item: item[0])),
        log_argument_distribution=dict(sorted(log_arguments.items())),
        nontriviality_totals=dict(sorted(nontriviality_totals.items())),
        nontriviality_rates=nontriviality_rates,
    )


def classify_log_arguments(expr: sp.Expr) -> list[str]:
    """Return coarse structural classes for all log arguments in an expression."""
    classes: list[str] = []
    for node in sp.preorder_traversal(expr):
        if node.func != sp.log or len(node.args) != 1:
            continue
        arg = node.args[0]
        if arg == sp.Integer(1):
            classes.append("one")
        elif isinstance(arg, sp.Symbol):
            classes.append("symbol")
        elif arg.func == sp.exp:
            classes.append("exp")
        elif arg.func == sp.Add:
            classes.append("add")
        elif arg.func == sp.Mul:
            classes.append("mul")
        elif arg.func == sp.Pow:
            classes.append("pow")
        elif arg.is_number:
            classes.append("number")
        else:
            classes.append(arg.func.__name__)
    return classes


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


def write_generation_summary(summary: ExpressionGenerationSummary, path: Path) -> None:
    """Write corpus-level generation quality summary JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def generate_dataset(config: ExpressionGeneratorConfig) -> list[GeneratedExpression]:
    """Generate expressions and save JSONL/CSV outputs."""
    generator = SympyExpressionGenerator(config)
    records = generator.generate()
    if config.jsonl_path is not None:
        write_jsonl(records, config.jsonl_path)
    if config.csv_path is not None:
        write_csv(records, config.csv_path)
    if config.summary_json_path is not None and generator.summary is not None:
        write_generation_summary(generator.summary, config.summary_json_path)
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
