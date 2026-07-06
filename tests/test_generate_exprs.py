"""Tests for bounded SymPy expression generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import sympy as sp
from geml.data.generate_exprs import (
    ExpressionGeneratorConfig,
    SympyExpressionGenerator,
    expression_depth,
    generate_dataset,
)


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
