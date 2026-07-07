"""Goal 2.5 failure mining for raw official pure EML expansion."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator

from geml.experiments.stratified_expansion import parse_srepr
from geml.symbolic.eml_transpile import sympy_to_eml_tree
from geml.symbolic.official_eml_compiler import emit_official_eml_string

TOP_EXPRESSION_FIELDS = [
    "rank",
    "index",
    "expression",
    "srepr",
    "ast_node_count",
    "ast_depth",
    "ast_operator_count",
    "ast_leaf_count",
    "eml_node_count",
    "eml_depth",
    "eml_operator_count",
    "eml_leaf_count",
    "alpha",
    "alpha_threshold",
    "threshold_gap",
    "below_threshold",
    "official_eml_snippet",
    "official_eml_char_count",
    "official_eml_truncated",
    "official_eml_error",
]
GROUP_SUMMARY_FIELDS = [
    "count",
    "mean_alpha",
    "median_alpha",
    "p90_alpha",
    "p95_alpha",
    "max_alpha",
    "mean_ast_nodes",
    "mean_eml_nodes",
    "percent_below_threshold",
]


class FailureMiningConfig(BaseModel):
    """Configuration for Goal 2.5 expansion failure mining."""

    raw_metrics_csv_path: Path = Path("outputs/v0/expansion_raw_metrics.csv")
    alpha_by_operator_family_csv_path: Path = Path("outputs/v0/alpha_by_operator_family.csv")
    alpha_by_operator_signature_csv_path: Path = Path("outputs/v0/alpha_by_operator_signature.csv")
    alpha_by_ast_depth_csv_path: Path = Path("outputs/v0/alpha_by_ast_depth.csv")
    top_alpha_csv_path: Path = Path("outputs/v0/top_alpha_explosions.csv")
    top_eml_node_csv_path: Path = Path("outputs/v0/top_eml_node_explosions.csv")
    top_eml_depth_csv_path: Path = Path("outputs/v0/top_eml_depth_explosions.csv")
    worst_operator_signatures_csv_path: Path = Path("outputs/v0/worst_operator_signatures.csv")
    safest_operator_signatures_csv_path: Path = Path("outputs/v0/safest_operator_signatures.csv")
    depth_failure_modes_csv_path: Path = Path("outputs/v0/depth_failure_modes.csv")
    safe_eml_regime_candidates_csv_path: Path = Path("outputs/v0/safe_eml_regime_candidates.csv")
    report_md_path: Path = Path("docs/goal2/GOAL2_FAILURE_CASES.md")
    top_n: int = Field(default=20, gt=0)
    snippet_max_chars: int = Field(default=1200, gt=0)

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if self.raw_metrics_csv_path == self.top_alpha_csv_path:
            raise ValueError("input raw metrics path must differ from output paths")
        return self


@dataclass(frozen=True)
class MetricRow:
    """Raw metric row used for failure mining."""

    index: int
    expression: str
    srepr: str
    ast_node_count: int
    ast_depth: int
    ast_operator_count: int
    ast_leaf_count: int
    eml_node_count: int
    eml_depth: int
    eml_operator_count: int
    eml_leaf_count: int
    alpha: float
    alpha_threshold: float
    below_threshold: bool

    @property
    def threshold_gap(self) -> float:
        return self.alpha - self.alpha_threshold


@dataclass(frozen=True)
class GroupMetricRow:
    """Grouped metric row used for signature/depth failure mining."""

    key: str
    count: int
    mean_alpha: float
    median_alpha: float
    p90_alpha: float
    p95_alpha: float
    max_alpha: float
    mean_ast_nodes: float
    mean_eml_nodes: float
    percent_below_threshold: float

    @property
    def mean_eml_to_ast_nodes(self) -> float:
        return self.mean_eml_nodes / self.mean_ast_nodes if self.mean_ast_nodes else 0.0


@dataclass(frozen=True)
class EmlSnippet:
    """Official-style EML snippet metadata for one expression."""

    text: str
    char_count: int | None
    truncated: bool
    error: str | None = None


@dataclass(frozen=True)
class FailureMiningResult:
    """Result metadata from a Goal 2.5 failure-mining run."""

    raw_metric_count: int
    operator_family_count: int
    operator_signature_count: int
    ast_depth_group_count: int
    output_paths: tuple[Path, ...]


def run_failure_mining(config: FailureMiningConfig) -> FailureMiningResult:
    """Mine failure cases from saved Goal 2 expansion-study outputs."""
    metric_rows = load_metric_rows(config.raw_metrics_csv_path)
    operator_family_rows = load_group_metric_rows(
        config.alpha_by_operator_family_csv_path,
        key_field="dominant_operator_family",
    )
    operator_signature_rows = load_group_metric_rows(
        config.alpha_by_operator_signature_csv_path,
        key_field="operator_signature",
    )
    ast_depth_rows = load_group_metric_rows(
        config.alpha_by_ast_depth_csv_path,
        key_field="ast_depth",
    )

    top_alpha = select_top_rows(metric_rows, key=lambda row: row.alpha, limit=config.top_n)
    top_eml_nodes = select_top_rows(
        metric_rows,
        key=lambda row: row.eml_node_count,
        limit=config.top_n,
    )
    top_eml_depth = select_top_rows(
        metric_rows,
        key=lambda row: row.eml_depth,
        limit=config.top_n,
    )
    primary_threshold = metric_rows[0].alpha_threshold
    worst_signatures = rank_worst_operator_signatures(operator_signature_rows, config.top_n)
    safest_signatures = rank_safest_operator_signatures(
        operator_signature_rows,
        threshold=primary_threshold,
        limit=config.top_n,
    )
    depth_failure_modes = rank_depth_failure_modes(ast_depth_rows, config.top_n)
    safe_candidates = rank_safe_regime_candidates(
        operator_signature_rows,
        threshold=primary_threshold,
        limit=config.top_n,
    )

    write_top_expression_csv(
        top_alpha,
        config.top_alpha_csv_path,
        snippet_max_chars=config.snippet_max_chars,
    )
    write_top_expression_csv(
        top_eml_nodes,
        config.top_eml_node_csv_path,
        snippet_max_chars=config.snippet_max_chars,
    )
    write_top_expression_csv(
        top_eml_depth,
        config.top_eml_depth_csv_path,
        snippet_max_chars=config.snippet_max_chars,
    )
    write_worst_signature_csv(worst_signatures, config.worst_operator_signatures_csv_path)
    write_safe_signature_csv(safest_signatures, config.safest_operator_signatures_csv_path)
    write_depth_failure_modes_csv(depth_failure_modes, config.depth_failure_modes_csv_path)
    write_safe_regime_candidates_csv(
        safe_candidates,
        config.safe_eml_regime_candidates_csv_path,
    )
    write_failure_report(
        config.report_md_path,
        top_alpha=top_alpha,
        top_eml_depth=top_eml_depth,
        worst_signatures=worst_signatures,
        safest_signatures=safest_signatures,
        safe_candidates=safe_candidates,
        depth_failure_modes=depth_failure_modes,
        operator_family_rows=operator_family_rows,
        threshold=primary_threshold,
        output_paths=(
            config.top_alpha_csv_path,
            config.top_eml_node_csv_path,
            config.top_eml_depth_csv_path,
            config.worst_operator_signatures_csv_path,
            config.safest_operator_signatures_csv_path,
            config.depth_failure_modes_csv_path,
            config.safe_eml_regime_candidates_csv_path,
        ),
    )

    return FailureMiningResult(
        raw_metric_count=len(metric_rows),
        operator_family_count=len(operator_family_rows),
        operator_signature_count=len(operator_signature_rows),
        ast_depth_group_count=len(ast_depth_rows),
        output_paths=(
            config.top_alpha_csv_path,
            config.top_eml_node_csv_path,
            config.top_eml_depth_csv_path,
            config.worst_operator_signatures_csv_path,
            config.safest_operator_signatures_csv_path,
            config.depth_failure_modes_csv_path,
            config.safe_eml_regime_candidates_csv_path,
            config.report_md_path,
        ),
    )


def load_metric_rows(path: Path) -> list[MetricRow]:
    """Load supported alpha-valid raw metrics from CSV."""
    rows: list[MetricRow] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            if parse_bool(raw_row["supported"]) and parse_bool(raw_row["alpha_valid"]):
                rows.append(
                    MetricRow(
                        index=parse_int(raw_row["index"]),
                        expression=raw_row["expression"],
                        srepr=raw_row["srepr"],
                        ast_node_count=parse_int(raw_row["ast_node_count"]),
                        ast_depth=parse_int(raw_row["ast_depth"]),
                        ast_operator_count=parse_int(raw_row["ast_operator_count"]),
                        ast_leaf_count=parse_int(raw_row["ast_leaf_count"]),
                        eml_node_count=parse_int(raw_row["eml_node_count"]),
                        eml_depth=parse_int(raw_row["eml_depth"]),
                        eml_operator_count=parse_int(raw_row["eml_operator_count"]),
                        eml_leaf_count=parse_int(raw_row["eml_leaf_count"]),
                        alpha=parse_float(raw_row["alpha"]),
                        alpha_threshold=parse_float(raw_row["alpha_threshold"]),
                        below_threshold=parse_bool(raw_row["below_threshold"]),
                    )
                )
    if not rows:
        raise ValueError(f"no supported alpha-valid rows found in {path}")
    return rows


def load_group_metric_rows(path: Path, *, key_field: str) -> list[GroupMetricRow]:
    """Load grouped stratification metrics from CSV."""
    rows: list[GroupMetricRow] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            rows.append(
                GroupMetricRow(
                    key=raw_row[key_field],
                    count=parse_int(raw_row["count"]),
                    mean_alpha=parse_float(raw_row["mean_alpha"]),
                    median_alpha=parse_float(raw_row["median_alpha"]),
                    p90_alpha=parse_float(raw_row["p90_alpha"]),
                    p95_alpha=parse_float(raw_row["p95_alpha"]),
                    max_alpha=parse_float(raw_row["max_alpha"]),
                    mean_ast_nodes=parse_float(raw_row["mean_ast_nodes"]),
                    mean_eml_nodes=parse_float(raw_row["mean_eml_nodes"]),
                    percent_below_threshold=parse_float(raw_row["percent_below_threshold"]),
                )
            )
    if not rows:
        raise ValueError(f"no grouped metric rows found in {path}")
    return rows


def select_top_rows(
    rows: Sequence[MetricRow],
    *,
    key: Callable[[MetricRow], float | int],
    limit: int,
) -> list[MetricRow]:
    """Select top rows by a metric, using index as deterministic tie-breaker."""
    return sorted(rows, key=lambda row: (-key(row), row.index))[:limit]


def rank_worst_operator_signatures(
    rows: Sequence[GroupMetricRow],
    limit: int,
) -> list[dict[str, object]]:
    """Rank operator signatures by highest median and p90 alpha."""
    median_ranked = sorted(
        rows,
        key=lambda row: (-row.median_alpha, -row.p90_alpha, -row.count, row.key),
    )
    p90_ranked = sorted(
        rows,
        key=lambda row: (-row.p90_alpha, -row.median_alpha, -row.count, row.key),
    )
    median_ranks = {row.key: rank for rank, row in enumerate(median_ranked, start=1)}
    p90_ranks = {row.key: rank for rank, row in enumerate(p90_ranked, start=1)}
    candidates = {row.key: row for row in [*median_ranked[:limit], *p90_ranked[:limit]]}
    ranked_candidates = sorted(
        candidates.values(),
        key=lambda row: (median_ranks[row.key], p90_ranks[row.key], row.key),
    )[:limit]
    return [
        {
            "median_alpha_rank": median_ranks[row.key],
            "p90_alpha_rank": p90_ranks[row.key],
            "operator_signature": row.key,
            **group_metric_to_dict(row),
        }
        for row in ranked_candidates
    ]


def rank_safest_operator_signatures(
    rows: Sequence[GroupMetricRow],
    *,
    threshold: float,
    limit: int,
) -> list[dict[str, object]]:
    """Rank operator signatures that stay closest to the alpha threshold."""
    ranked = sorted(
        rows,
        key=lambda row: (
            -row.percent_below_threshold,
            abs(row.median_alpha - threshold),
            row.p90_alpha,
            -row.count,
            row.key,
        ),
    )[:limit]
    return [safe_signature_to_dict(row, threshold=threshold) for row in ranked]


def rank_safe_regime_candidates(
    rows: Sequence[GroupMetricRow],
    *,
    threshold: float,
    limit: int,
) -> list[dict[str, object]]:
    """Rank expression-family candidates for a raw pure EML safe regime."""
    ranked = sorted(
        rows,
        key=lambda row: (
            -row.percent_below_threshold,
            row.median_alpha - threshold,
            row.p90_alpha - threshold,
            row.mean_eml_to_ast_nodes,
            row.key,
        ),
    )[:limit]
    return [
        {
            "candidate_kind": "operator_signature",
            **safe_signature_to_dict(row, threshold=threshold),
        }
        for row in ranked
    ]


def rank_depth_failure_modes(
    rows: Sequence[GroupMetricRow],
    limit: int,
) -> list[dict[str, object]]:
    """Rank AST-depth groups where EML expansion worsens most."""
    ranked = sorted(
        rows,
        key=lambda row: (-row.p90_alpha, -row.mean_alpha, -row.mean_eml_nodes, row.key),
    )[:limit]
    return [
        {
            "ast_depth": row.key,
            "mean_eml_to_ast_nodes": row.mean_eml_to_ast_nodes,
            **group_metric_to_dict(row),
        }
        for row in ranked
    ]


def group_metric_to_dict(row: GroupMetricRow) -> dict[str, object]:
    """Serialize grouped metric columns."""
    return {
        "count": row.count,
        "mean_alpha": row.mean_alpha,
        "median_alpha": row.median_alpha,
        "p90_alpha": row.p90_alpha,
        "p95_alpha": row.p95_alpha,
        "max_alpha": row.max_alpha,
        "mean_ast_nodes": row.mean_ast_nodes,
        "mean_eml_nodes": row.mean_eml_nodes,
        "percent_below_threshold": row.percent_below_threshold,
    }


def safe_signature_to_dict(row: GroupMetricRow, *, threshold: float) -> dict[str, object]:
    """Serialize safe-signature ranking columns."""
    return {
        "operator_signature": row.key,
        "alpha_threshold": threshold,
        "median_threshold_gap": row.median_alpha - threshold,
        "p90_threshold_gap": row.p90_alpha - threshold,
        **group_metric_to_dict(row),
    }


def write_top_expression_csv(
    rows: Sequence[MetricRow],
    path: Path,
    *,
    snippet_max_chars: int,
) -> None:
    """Write top expression failures with official-style EML snippets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TOP_EXPRESSION_FIELDS)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(top_expression_to_dict(row, rank, snippet_max_chars=snippet_max_chars))


def top_expression_to_dict(
    row: MetricRow,
    rank: int,
    *,
    snippet_max_chars: int,
) -> dict[str, object]:
    """Serialize one top expression row with a truncated official EML string."""
    snippet = build_official_eml_snippet(row.srepr, max_chars=snippet_max_chars)
    return {
        "rank": rank,
        "index": row.index,
        "expression": row.expression,
        "srepr": row.srepr,
        "ast_node_count": row.ast_node_count,
        "ast_depth": row.ast_depth,
        "ast_operator_count": row.ast_operator_count,
        "ast_leaf_count": row.ast_leaf_count,
        "eml_node_count": row.eml_node_count,
        "eml_depth": row.eml_depth,
        "eml_operator_count": row.eml_operator_count,
        "eml_leaf_count": row.eml_leaf_count,
        "alpha": row.alpha,
        "alpha_threshold": row.alpha_threshold,
        "threshold_gap": row.threshold_gap,
        "below_threshold": row.below_threshold,
        "official_eml_snippet": snippet.text,
        "official_eml_char_count": snippet.char_count,
        "official_eml_truncated": snippet.truncated,
        "official_eml_error": snippet.error,
    }


def build_official_eml_snippet(srepr: str, *, max_chars: int) -> EmlSnippet:
    """Build an official-style EML string snippet for a source expression."""
    try:
        expr = parse_srepr(srepr)
        tree = sympy_to_eml_tree(expr, representation_mode="restricted_eml_pure")
        official_eml = emit_official_eml_string(tree)
    except Exception as exc:
        return EmlSnippet(
            text="",
            char_count=None,
            truncated=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    if len(official_eml) <= max_chars:
        return EmlSnippet(text=official_eml, char_count=len(official_eml), truncated=False)

    suffix = f"... [truncated; full_length={len(official_eml)}]"
    prefix_length = max(0, max_chars - len(suffix))
    return EmlSnippet(
        text=f"{official_eml[:prefix_length]}{suffix}",
        char_count=len(official_eml),
        truncated=True,
    )


def write_worst_signature_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Write worst operator-signature ranking CSV."""
    write_dict_csv(
        rows,
        path,
        fieldnames=[
            "median_alpha_rank",
            "p90_alpha_rank",
            "operator_signature",
            *GROUP_SUMMARY_FIELDS,
        ],
    )


def write_safe_signature_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Write safest operator-signature ranking CSV."""
    write_dict_csv(
        rows,
        path,
        fieldnames=[
            "operator_signature",
            "alpha_threshold",
            "median_threshold_gap",
            "p90_threshold_gap",
            *GROUP_SUMMARY_FIELDS,
        ],
    )


def write_depth_failure_modes_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Write AST-depth failure-mode ranking CSV."""
    write_dict_csv(
        rows,
        path,
        fieldnames=["ast_depth", "mean_eml_to_ast_nodes", *GROUP_SUMMARY_FIELDS],
    )


def write_safe_regime_candidates_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Write raw pure EML safe-regime candidate CSV."""
    write_dict_csv(
        rows,
        path,
        fieldnames=[
            "candidate_kind",
            "operator_signature",
            "alpha_threshold",
            "median_threshold_gap",
            "p90_threshold_gap",
            *GROUP_SUMMARY_FIELDS,
        ],
    )


def write_dict_csv(
    rows: Sequence[dict[str, object]],
    path: Path,
    *,
    fieldnames: Sequence[str],
) -> None:
    """Write dictionaries to CSV with a fixed schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_failure_report(
    path: Path,
    *,
    top_alpha: Sequence[MetricRow],
    top_eml_depth: Sequence[MetricRow],
    worst_signatures: Sequence[dict[str, object]],
    safest_signatures: Sequence[dict[str, object]],
    safe_candidates: Sequence[dict[str, object]],
    depth_failure_modes: Sequence[dict[str, object]],
    operator_family_rows: Sequence[GroupMetricRow],
    threshold: float,
    output_paths: Sequence[Path],
) -> None:
    """Write the Goal 2.5 markdown failure-case report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_failure_report_markdown(
            top_alpha=top_alpha,
            top_eml_depth=top_eml_depth,
            worst_signatures=worst_signatures,
            safest_signatures=safest_signatures,
            safe_candidates=safe_candidates,
            depth_failure_modes=depth_failure_modes,
            operator_family_rows=operator_family_rows,
            threshold=threshold,
            output_paths=output_paths,
        ),
        encoding="utf-8",
    )


def build_failure_report_markdown(
    *,
    top_alpha: Sequence[MetricRow],
    top_eml_depth: Sequence[MetricRow],
    worst_signatures: Sequence[dict[str, object]],
    safest_signatures: Sequence[dict[str, object]],
    safe_candidates: Sequence[dict[str, object]],
    depth_failure_modes: Sequence[dict[str, object]],
    operator_family_rows: Sequence[GroupMetricRow],
    threshold: float,
    output_paths: Sequence[Path],
) -> str:
    """Build the markdown body for Goal 2.5 failure mining."""
    worst_preview = list(worst_signatures[:5])
    safe_preview = list(safe_candidates[:5])
    add_mul_count = sum(
        1
        for row in worst_preview
        if "Add" in str(row["operator_signature"]) and "Mul" in str(row["operator_signature"])
    )
    log_count = sum(1 for row in worst_preview if "log" in str(row["operator_signature"]))
    exp_count = sum(1 for row in worst_preview if "exp" in str(row["operator_signature"]))
    closest = safe_preview[0] if safe_preview else None
    any_safe = any(float(row["percent_below_threshold"]) > 0 for row in safe_candidates)
    strongest_family = max(operator_family_rows, key=lambda row: row.median_alpha)

    sections = [
        "# Goal 2 Failure Cases",
        "",
        "This report mines raw official pure EML expansion failures from saved Goal 2 outputs.",
        "",
        "Important: this is structural evidence about tree expansion only. It is not "
        "model-performance evidence.",
        "",
        f"Current alpha threshold used by raw rows: `{threshold}`.",
        "",
        "## Highest-Alpha Examples",
        "",
        markdown_expression_table(top_alpha[:5], metric="alpha"),
        "",
        "## Highest-Depth Examples",
        "",
        markdown_expression_table(top_eml_depth[:5], metric="eml_depth"),
        "",
        "## Worst Operator Signatures",
        "",
        markdown_signature_table(worst_preview),
        "",
        "## Depth Failure Modes",
        "",
        markdown_depth_table(depth_failure_modes[:5]),
        "",
        "## Common Structural Causes",
        "",
        "- Add/Mul-heavy signatures dominate the worst median-alpha groups: "
        f"`{add_mul_count}` of the top `{len(worst_preview)}` worst signatures contain both.",
        f"- `log` appears in `{log_count}` of the top `{len(worst_preview)}` worst signatures.",
        f"- `exp` appears in `{exp_count}` of the top `{len(worst_preview)}` worst signatures.",
        "- The strongest dominant operator family by median alpha is "
        f"`{strongest_family.key}` with median alpha `{strongest_family.median_alpha}`.",
        "- Repeated Add/Mul macro expansion is the main structural source of large pure EML "
        "trees; log and exp wrappers add depth and amplify nested products/sums.",
        "",
        "## Safe Raw EML Regime",
        "",
    ]
    if closest is None:
        sections.append("No operator-signature candidates were available.")
    elif any_safe:
        sections.extend(
            [
                "Some signatures have expressions below the current threshold.",
                markdown_safe_table(safe_preview),
            ]
        )
    else:
        sections.extend(
            [
                "No robust safe regime appears under the current raw pure EML threshold. "
                "The closest signatures remain above threshold on median alpha.",
                markdown_safe_table(safe_preview),
            ]
        )
    sections.extend(
        [
            "",
            "## Output Files",
            "",
            *[f"- `{path}`" for path in output_paths],
        ]
    )
    return "\n".join(sections) + "\n"


def markdown_expression_table(rows: Sequence[MetricRow], *, metric: str) -> str:
    """Render a compact expression table for the report."""
    lines = [
        "| Rank | Index | Metric | Alpha | AST nodes | EML nodes | EML depth | Expression |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(rows, start=1):
        metric_value = getattr(row, metric)
        lines.append(
            "| "
            f"{rank} | {row.index} | {metric_value} | {row.alpha} | "
            f"{row.ast_node_count} | {row.eml_node_count} | {row.eml_depth} | "
            f"`{truncate_for_markdown(row.expression, 90)}` |"
        )
    return "\n".join(lines)


def markdown_signature_table(rows: Sequence[dict[str, object]]) -> str:
    """Render a compact operator-signature table."""
    lines = [
        "| Signature | Median alpha | P90 alpha | Count | Percent below threshold |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['operator_signature']}` | {row['median_alpha']} | {row['p90_alpha']} | "
            f"{row['count']} | {row['percent_below_threshold']} |"
        )
    return "\n".join(lines)


def markdown_depth_table(rows: Sequence[dict[str, object]]) -> str:
    """Render a compact depth failure-mode table."""
    lines = [
        "| AST depth | Mean alpha | P90 alpha | Mean EML/AST nodes | Count |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['ast_depth']} | {row['mean_alpha']} | {row['p90_alpha']} | "
            f"{row['mean_eml_to_ast_nodes']} | {row['count']} |"
        )
    return "\n".join(lines)


def markdown_safe_table(rows: Sequence[dict[str, object]]) -> str:
    """Render a compact safe-candidate table."""
    lines = [
        "| Signature | Median alpha | Median threshold gap | P90 alpha | Percent below threshold |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['operator_signature']}` | {row['median_alpha']} | "
            f"{row['median_threshold_gap']} | {row['p90_alpha']} | "
            f"{row['percent_below_threshold']} |"
        )
    return "\n".join(lines)


def truncate_for_markdown(text: str, max_chars: int) -> str:
    """Truncate markdown table text without introducing newlines."""
    normalized = text.replace("\n", " ")
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


def parse_int(value: str) -> int:
    """Parse a required integer CSV field."""
    if value == "":
        raise ValueError("expected integer, got empty string")
    return int(value)


def parse_float(value: str) -> float:
    """Parse a required float CSV field."""
    if value == "":
        raise ValueError("expected float, got empty string")
    return float(value)


def parse_bool(value: str) -> bool:
    """Parse a required boolean CSV field."""
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected boolean string, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=None, help="Number of rows to export.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 2.5 failure-mining export."""
    args = build_parser().parse_args(argv)
    config = FailureMiningConfig()
    if args.top_n is not None:
        config.top_n = args.top_n
    result = run_failure_mining(config)
    print(f"Loaded raw metric rows: {result.raw_metric_count}")
    print(f"Loaded operator-family groups: {result.operator_family_count}")
    print(f"Loaded operator-signature groups: {result.operator_signature_count}")
    print(f"Loaded AST-depth groups: {result.ast_depth_group_count}")
    for path in result.output_paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
