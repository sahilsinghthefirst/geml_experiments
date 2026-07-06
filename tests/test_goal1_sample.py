"""Tests for the small Goal 1 end-to-end sample pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.goal1_sample import Goal1SampleConfig, run_goal1_sample


def test_goal1_sample_pipeline_writes_expected_outputs(tmp_path: Path) -> None:
    output_jsonl_path = tmp_path / "goal1_sample.jsonl"
    output_csv_path = tmp_path / "goal1_summary.csv"

    rows = run_goal1_sample(
        Goal1SampleConfig(
            count=10,
            seed=5,
            max_depth=3,
            output_jsonl_path=output_jsonl_path,
            output_csv_path=output_csv_path,
        )
    )

    jsonl_rows = [
        json.loads(line)
        for line in output_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    with output_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))

    assert len(rows) == 10
    assert len(jsonl_rows) == 10
    assert len(csv_rows) == 10
    assert all(row.supported for row in rows)
    assert {
        "expression",
        "srepr",
        "ast_stats",
        "eml_stats",
        "alpha",
        "supported",
        "error",
    } <= set(jsonl_rows[0])
    assert {
        "expression",
        "srepr",
        "ast_node_count",
        "eml_node_count",
        "alpha",
        "supported",
        "error",
    } <= set(csv_rows[0])
