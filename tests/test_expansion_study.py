"""Tests for the Goal 2 expansion-factor scale pipeline."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import sympy as sp
from geml.data.dataset import GeneratedExpressionInput, compute_metrics_rows
from geml.experiments.expansion_study import (
    ExpansionStudyConfig,
    ThresholdScenario,
    annotate_alpha_thresholds,
    build_alpha_threshold_summary,
    compute_alpha_threshold,
    run_expansion_study,
)


def test_alpha_threshold_math_and_classification() -> None:
    threshold = compute_alpha_threshold(k=4, ell=3)
    assert math.isclose(threshold, 1 + (math.log(4) / math.log(12)))

    x, y = sp.symbols("x y")
    rows = compute_metrics_rows(
        [
            GeneratedExpressionInput(
                index=0,
                expression="exp(x)",
                srepr=sp.srepr(sp.exp(x, evaluate=False)),
            ),
            GeneratedExpressionInput(
                index=1,
                expression="x + y",
                srepr=sp.srepr(sp.Add(x, y, evaluate=False)),
            ),
        ],
    )
    annotate_alpha_thresholds(rows, alpha_threshold=threshold)

    assert rows[0].alpha == 1.5
    assert rows[0].alpha_threshold == threshold
    assert rows[0].below_threshold is True
    assert rows[1].alpha == 9.0
    assert rows[1].alpha_threshold == threshold
    assert rows[1].below_threshold is False

    summary = build_alpha_threshold_summary(
        rows,
        scenario=ThresholdScenario(name="current_grammar", k=4, l=3),
    )
    assert summary["below_threshold_count"] == 1
    assert summary["above_threshold_count"] == 1
    assert summary["percent_below_threshold"] == 50.0
    assert summary["percent_above_threshold"] == 50.0


def test_expansion_study_small_count_writes_inputs_and_raw_metrics(tmp_path: Path) -> None:
    config = ExpansionStudyConfig(
        seed=7,
        count=8,
        max_depth=1,
        output_dir=tmp_path,
        input_jsonl_path=tmp_path / "expansion_inputs.jsonl",
        raw_metrics_jsonl_path=tmp_path / "expansion_raw_metrics.jsonl",
        raw_metrics_csv_path=tmp_path / "expansion_raw_metrics.csv",
        summary_json_path=tmp_path / "official_eml_compiler_summary.json",
        alpha_summary_csv_path=tmp_path / "expansion_alpha_summary.csv",
        alpha_summary_json_path=tmp_path / "expansion_alpha_summary.json",
        top_alpha_json_path=tmp_path / "official_eml_top20_alpha.json",
        top_depth_json_path=tmp_path / "official_eml_top20_depth.json",
        simple_examples_json_path=tmp_path / "official_eml_simple_examples.json",
        operator_probabilities={"add": 1.0},
    )

    rows = run_expansion_study(config)

    input_rows = [
        json.loads(line)
        for line in config.input_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    raw_jsonl_rows = [
        json.loads(line)
        for line in config.raw_metrics_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    with config.raw_metrics_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        raw_csv_rows = list(csv.DictReader(csv_file))
    summary = json.loads(config.summary_json_path.read_text(encoding="utf-8"))
    alpha_summary = json.loads(config.alpha_summary_json_path.read_text(encoding="utf-8"))
    with config.alpha_summary_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        alpha_summary_csv_rows = list(csv.DictReader(csv_file))
    top_alpha = json.loads(config.top_alpha_json_path.read_text(encoding="utf-8"))
    top_depth = json.loads(config.top_depth_json_path.read_text(encoding="utf-8"))
    simple_examples = json.loads(config.simple_examples_json_path.read_text(encoding="utf-8"))
    primary_threshold = compute_alpha_threshold(config.alpha_threshold_k, config.alpha_threshold_l)

    assert len(rows) == config.count
    assert len(input_rows) == config.count
    assert len(raw_jsonl_rows) == config.count
    assert len(raw_csv_rows) == config.count
    assert {"expression", "srepr", "depth", "metadata"} <= set(input_rows[0])
    assert {"source_serialization", "representation_mode", "supported", "error"} <= set(
        raw_jsonl_rows[0]
    )
    assert {"alpha", "alpha_threshold", "below_threshold"} <= set(raw_jsonl_rows[0])
    assert {"alpha", "alpha_threshold", "below_threshold"} <= set(raw_csv_rows[0])
    assert all(row.source_serialization == "srepr" for row in rows)
    assert all(row.representation_mode == "restricted_eml_pure" for row in rows)
    assert all(row.ast_stats is not None for row in rows)
    assert all(row.eml_stats is not None for row in rows)
    assert all(row.supported is True for row in rows)
    assert all(row.alpha is not None and row.alpha_valid is True for row in rows)
    assert all(row.alpha_threshold == primary_threshold for row in rows)
    assert all(
        row.below_threshold == (row.alpha is not None and row.alpha < primary_threshold)
        for row in rows
    )
    assert all(row.error is None for row in rows)
    assert all(raw_row["alpha_threshold"] == primary_threshold for raw_row in raw_jsonl_rows)
    assert all(
        raw_row["below_threshold"]
        == (raw_row["alpha"] is not None and raw_row["alpha"] < primary_threshold)
        for raw_row in raw_jsonl_rows
    )
    assert summary["processed_count"] == config.count
    assert summary["official_pure_eml_supported_count"] == config.count
    assert summary["unsupported_count"] == 0
    assert summary["mean_alpha"] is not None
    assert len(alpha_summary) == 3
    assert len(alpha_summary_csv_rows) == 3
    assert {row["scenario"] for row in alpha_summary} == {
        "current_grammar",
        "generous_operator_vocab",
        "larger_operator_vocab",
    }
    current_summary = next(row for row in alpha_summary if row["scenario"] == "current_grammar")
    assert current_summary["k"] == 4
    assert current_summary["l"] == 3
    assert current_summary["alpha_valid_count"] == config.count
    assert current_summary["below_threshold_count"] == sum(1 for row in rows if row.below_threshold)
    assert current_summary["above_threshold_count"] == sum(
        1 for row in rows if not row.below_threshold
    )
    assert current_summary["p95_alpha"] is not None
    assert len(summary["top_20_largest_alpha_expressions"]) == config.count
    assert len(summary["top_20_deepest_eml_expressions"]) == config.count
    assert len(top_alpha) == config.count
    assert len(top_depth) == config.count
    assert {example["name"] for example in simple_examples} == {
        "x+y",
        "x*y",
        "log(x)",
        "exp(x)",
        "x**2",
    }
    assert all(example["derived_leaf_count"] == 0 for example in simple_examples)
    assert all(example["hidden_compound_leaf_count"] == 0 for example in simple_examples)
    assert all(
        example["official_eml"].startswith(("EML[", "x", "1")) for example in simple_examples
    )
