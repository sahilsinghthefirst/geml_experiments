"""Tests for bounded SymPy expression generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import sympy as sp
from geml.data.generate_exprs import (
    ExpressionGeneratorConfig,
    SympyExpressionGenerator,
    classify_log_arguments,
    expression_depth,
    generate_dataset,
)
from geml.symbolic.srepr import parse_srepr


def test_generation_is_reproducible() -> None:
    config = ExpressionGeneratorConfig(count=20, max_depth=3, seed=123)

    first = SympyExpressionGenerator(config).generate()
    second = SympyExpressionGenerator(config).generate()

    assert [record.expression for record in first] == [record.expression for record in second]
    assert [record.srepr for record in first] == [record.srepr for record in second]


def test_generated_depth_respects_bound() -> None:
    max_depth = 3
    config = ExpressionGeneratorConfig(count=50, max_depth=max_depth, seed=7)

    records = SympyExpressionGenerator(config).generate()

    assert records
    assert all(record.depth <= max_depth for record in records)


def test_generated_expressions_parse_with_sympy() -> None:
    config = ExpressionGeneratorConfig(count=25, max_depth=4, seed=99)
    locals_ = {name: sp.Symbol(name) for name in config.symbol_names}

    records = SympyExpressionGenerator(config).generate()

    for record in records:
        parsed = sp.sympify(record.expression, locals=locals_, evaluate=False)
        assert isinstance(parsed, sp.Expr)
        assert expression_depth(parsed) <= config.max_depth


def test_generate_dataset_writes_jsonl_and_csv(tmp_path: Path) -> None:
    config = ExpressionGeneratorConfig(
        count=5,
        max_depth=2,
        seed=3,
        output_dir=tmp_path,
        jsonl_path=tmp_path / "dataset.jsonl",
        csv_path=tmp_path / "dataset.csv",
    )

    records = generate_dataset(config)

    jsonl_rows = [
        json.loads(line)
        for line in config.jsonl_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    with config.csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))

    assert len(records) == config.count
    assert len(jsonl_rows) == config.count
    assert len(csv_rows) == config.count
    assert {"expression", "srepr", "depth", "metadata"} <= set(jsonl_rows[0])
    assert {"expression", "srepr", "depth", "metadata"} <= set(csv_rows[0])


def test_v1_style_generation_has_variable_actual_depths() -> None:
    config = ExpressionGeneratorConfig(
        count=100,
        max_depth=4,
        seed=21,
        target_depth_probabilities={0: 0.1, 1: 0.2, 2: 0.25, 3: 0.25, 4: 0.2},
        intermediate_leaf_probability=0.2,
        deduplicate_srepr=True,
        max_generation_attempts=10000,
        log_argument_strategy="positive_domain",
    )

    generator = SympyExpressionGenerator(config)
    records = generator.generate()

    depths = {record.depth for record in records}
    assert len(depths) > 1
    assert max(depths) <= config.max_depth
    assert any(depth < config.max_depth for depth in depths)
    assert generator.summary is not None
    assert len(generator.summary.actual_depth_histogram) > 1


def test_deduplicated_generation_produces_unique_sreprs() -> None:
    config = ExpressionGeneratorConfig(
        count=75,
        max_depth=3,
        seed=22,
        target_depth_probabilities={1: 0.2, 2: 0.4, 3: 0.4},
        intermediate_leaf_probability=0.1,
        deduplicate_srepr=True,
        max_generation_attempts=10000,
        log_argument_strategy="positive_domain",
    )

    generator = SympyExpressionGenerator(config)
    records = generator.generate()
    sreprs = [record.srepr for record in records]

    assert len(sreprs) == len(set(sreprs))
    assert generator.summary is not None
    assert generator.summary.output_duplicate_count == 0
    assert generator.summary.output_duplicate_rate == 0.0


def test_positive_domain_log_generation_is_not_blanket_exp_wrapped() -> None:
    config = ExpressionGeneratorConfig(
        count=20,
        max_depth=2,
        seed=23,
        operator_probabilities={"log": 1.0},
        target_depth_probabilities={1: 1.0},
        log_argument_strategy="positive_domain",
        positive_log_argument_probabilities={"leaf": 1.0},
    )

    records = SympyExpressionGenerator(config).generate()
    log_argument_classes = []
    for record in records:
        expr = parse_srepr(record.srepr)
        assert expr.func == sp.log
        assert expr.args[0].func != sp.exp
        log_argument_classes.extend(classify_log_arguments(expr))

    assert log_argument_classes
    assert set(log_argument_classes) <= {"one", "symbol"}


def test_positive_domain_log_argument_grammar() -> None:
    config = ExpressionGeneratorConfig(
        count=50,
        max_depth=3,
        seed=24,
        operator_probabilities={"log": 1.0},
        target_depth_probabilities={2: 0.5, 3: 0.5},
        log_argument_strategy="positive_domain",
        positive_log_argument_probabilities={"leaf": 0.25, "add": 0.35, "mul": 0.3, "exp": 0.1},
    )

    records = SympyExpressionGenerator(config).generate()

    for record in records:
        expr = parse_srepr(record.srepr)
        for node in sp.preorder_traversal(expr):
            if node.func == sp.log:
                assert _is_positive_domain_expr(node.args[0])


def test_generation_summary_json_reports_depth_duplicates_and_triviality(tmp_path: Path) -> None:
    config = ExpressionGeneratorConfig(
        count=30,
        max_depth=3,
        seed=25,
        output_dir=tmp_path,
        jsonl_path=tmp_path / "dataset.jsonl",
        csv_path=None,
        summary_json_path=tmp_path / "summary.json",
        target_depth_probabilities={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
        intermediate_leaf_probability=0.15,
        deduplicate_srepr=True,
        max_generation_attempts=10000,
        log_argument_strategy="positive_domain",
    )

    generate_dataset(config)
    summary = json.loads(config.summary_json_path.read_text(encoding="utf-8"))

    assert summary["generated_count"] == 30
    assert summary["unique_srepr_count"] == 30
    assert "actual_depth_histogram" in summary
    assert "duplicate_rejection_rate" in summary
    assert "nontriviality_totals" in summary
    assert "log_argument_distribution" in summary


def _is_positive_domain_expr(expr: sp.Expr) -> bool:
    if expr == sp.Integer(1) or isinstance(expr, sp.Symbol):
        return True
    if expr.func == sp.exp and len(expr.args) == 1:
        return True
    if expr.func in {sp.Add, sp.Mul}:
        return all(_is_positive_domain_expr(arg) for arg in expr.args)
    return False
