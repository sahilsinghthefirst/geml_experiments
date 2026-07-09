"""Shared experiment helpers for reporting, IO, and reproducibility metadata."""

from __future__ import annotations

import json
import math
import platform
import statistics
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

CORE_PACKAGE_DISTRIBUTIONS: Mapping[str, str] = {
    "matplotlib": "matplotlib",
    "pydantic": "pydantic",
    "pyyaml": "PyYAML",
    "sympy": "sympy",
    "pytest": "pytest",
    "ruff": "ruff",
}


def safe_divide(numerator: int | float | None, denominator: int | float | None) -> float | None:
    """Return numerator / denominator, or None for missing/zero denominators."""
    if numerator is None or denominator in {None, 0}:
        return None
    return float(numerator) / float(denominator)


def percent(numerator: int | float | None, denominator: int | float | None) -> float | None:
    """Return 100 * numerator / denominator, or None for missing/zero denominators."""
    ratio = safe_divide(numerator, denominator)
    if ratio is None:
        return None
    return 100.0 * ratio


def percentile(values: Sequence[int | float], quantile: float) -> float | None:
    """Compute an interpolated percentile for numeric values."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    numeric_values = [float(value) for value in values if math.isfinite(float(value))]
    if not numeric_values:
        return None
    if len(numeric_values) == 1:
        return numeric_values[0]
    sorted_values = sorted(numeric_values)
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def distribution_summary(values: Iterable[int | float | None]) -> dict[str, float | None]:
    """Return the standard mean/median/p90 distribution summary."""
    numeric_values = [
        float(value) for value in values if value is not None and math.isfinite(float(value))
    ]
    if not numeric_values:
        return {"mean": None, "median": None, "p90": None}
    return {
        "mean": statistics.fmean(numeric_values),
        "median": statistics.median(numeric_values),
        "p90": percentile(numeric_values, 0.9),
    }


def markdown_table(headers: Sequence[object], rows: Sequence[Sequence[object]]) -> str:
    """Build a compact GitHub-flavored Markdown table."""
    header_line = "| " + " | ".join(str(header) for header in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def write_json_object(path: Path, payload: Mapping[str, object]) -> None:
    """Write a deterministic JSON object with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write text after creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json_object(path: Path) -> dict[str, object]:
    """Load a JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_json_list(path: Path) -> list[dict[str, object]]:
    """Load a JSON list of objects from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"expected JSON list of objects: {path}")
    return payload


def build_run_metadata(
    *,
    config: Mapping[str, object] | None = None,
    started_at: float | None = None,
    completed_at: float | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build reproducibility metadata for experiment summary artifacts."""
    payload: dict[str, object] = {
        "git_commit_hash": git_commit_hash(),
        "git_dirty": git_dirty(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
        "package_versions": package_versions(),
        "started_at_unix": started_at,
        "completed_at_unix": completed_at,
    }
    if started_at is not None and completed_at is not None:
        payload["elapsed_seconds"] = completed_at - started_at
    if config is not None:
        payload["config"] = dict(config)
    if extra:
        payload.update(dict(extra))
    return payload


def package_versions(
    distributions: Mapping[str, str] = CORE_PACKAGE_DISTRIBUTIONS,
) -> dict[str, str | None]:
    """Return installed versions for core project dependencies."""
    versions: dict[str, str | None] = {}
    for label, distribution in distributions.items():
        try:
            versions[label] = version(distribution)
        except PackageNotFoundError:
            versions[label] = None
    return versions


def git_commit_hash(repo_root: Path | None = None) -> str | None:
    """Return the current git commit hash when available."""
    result = _run_git(["rev-parse", "HEAD"], repo_root=repo_root)
    return result if result else None


def git_dirty(repo_root: Path | None = None) -> bool | None:
    """Return whether tracked or untracked files are present when git is available."""
    result = _run_git(["status", "--porcelain"], repo_root=repo_root)
    if result is None:
        return None
    return bool(result)


def _run_git(args: Sequence[str], *, repo_root: Path | None) -> str | None:
    cwd = repo_root or Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
