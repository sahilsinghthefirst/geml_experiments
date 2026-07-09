from __future__ import annotations

import json
from pathlib import Path

import pytest
from geml.experiments.shared import (
    build_run_metadata,
    distribution_summary,
    markdown_table,
    percent,
    safe_divide,
    write_json_object,
)


def test_shared_math_helpers_handle_missing_denominators() -> None:
    assert safe_divide(6, 3) == 2.0
    assert safe_divide(6, 0) is None
    assert percent(1, 4) == 25.0
    assert percent(1, 0) is None


def test_distribution_summary_reports_standard_fields() -> None:
    summary = distribution_summary([1, 2, 3, None])

    assert summary["mean"] == 2.0
    assert summary["median"] == 2.0
    assert summary["p90"] == pytest.approx(2.8)


def test_markdown_table_builder() -> None:
    assert markdown_table(["a", "b"], [[1, 2]]) == "| a | b |\n| --- | --- |\n| 1 | 2 |"


def test_run_metadata_contains_reproducibility_fields() -> None:
    metadata = build_run_metadata(
        config={"count": 1},
        started_at=10.0,
        completed_at=12.5,
    )

    assert metadata["config"] == {"count": 1}
    assert metadata["elapsed_seconds"] == 2.5
    assert "git_commit_hash" in metadata
    assert "python_version" in metadata
    assert "platform" in metadata
    package_versions = metadata["package_versions"]
    assert isinstance(package_versions, dict)
    assert {"sympy", "pydantic", "pyyaml"} <= set(package_versions)


def test_write_json_object_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"

    write_json_object(path, {"b": 2, "a": 1})

    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert path.read_text(encoding="utf-8").endswith("\n")
