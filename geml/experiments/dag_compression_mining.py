"""Goal 3.5 success and failure mining for DAG compression."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from geml.experiments.stratified_expansion import (
    count_operator_features,
    operator_signature,
    parse_bool,
    parse_float,
    parse_int,
)

TOP_DAG_EXPRESSION_FIELDS = [
    "rank",
    "index",
    "success_score",
    "failure_score",
    "alpha_drop",
    "dag_improvement_ratio",
    "expression",
    "srepr",
    "tree_alpha",
    "dag_alpha_vs_ast_tree",
    "dag_alpha_vs_ast_dag",
    "eml_dag_compression",
    "ast_dag_compression",
    "ast_tree_node_count",
    "ast_dag_node_count",
    "eml_tree_node_count",
    "eml_dag_node_count",
    "operator_signature",
]
SIGNATURE_RANK_FIELDS = [
    "rank",
    "rank_score",
    "operator_signature",
    "count",
    "median_tree_alpha",
    "median_dag_alpha_vs_ast_tree",
    "median_dag_alpha_vs_ast_dag",
    "median_eml_dag_compression",
    "p90_eml_dag_compression",
    "percent_below_threshold_after_dag",
    "percent_below_threshold_dag_vs_ast_tree",
    "percent_below_threshold_dag_vs_ast_dag",
    "median_improvement",
]
SAFE_REGIME_FIELDS = [
    "rank",
    "candidate_kind",
    *SIGNATURE_RANK_FIELDS[2:],
]


class DagCompressionMiningConfig(BaseModel):
    """Configuration for Goal 3.5 DAG compression mining."""

    dag_metrics_csv_path: Path = Path("outputs/v0/dag_compression_metrics.csv")
    dag_operator_signature_csv_path: Path = Path("outputs/v0/dag_alpha_by_operator_signature.csv")
    dag_threshold_summary_json_path: Path = Path("outputs/v0/dag_alpha_threshold_summary.json")
    top_successes_csv_path: Path = Path("outputs/v0/top_dag_compression_successes.csv")
    top_failures_csv_path: Path = Path("outputs/v0/top_dag_compression_failures.csv")
    best_operator_signatures_csv_path: Path = Path("outputs/v0/best_dag_operator_signatures.csv")
    worst_operator_signatures_csv_path: Path = Path("outputs/v0/worst_dag_operator_signatures.csv")
    safe_regime_candidates_csv_path: Path = Path("outputs/v0/dag_safe_regime_candidates.csv")
    report_md_path: Path = Path("docs/goal3/GOAL3_DAG_COMPRESSION_FINDINGS.md")
    top_n: int = Field(default=20, gt=0)

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        output_paths = {
            self.top_successes_csv_path,
            self.top_failures_csv_path,
            self.best_operator_signatures_csv_path,
            self.worst_operator_signatures_csv_path,
            self.safe_regime_candidates_csv_path,
            self.report_md_path,
        }
        if self.dag_metrics_csv_path in output_paths:
            raise ValueError("DAG metrics input path must differ from output paths")
        if self.dag_operator_signature_csv_path in output_paths:
            raise ValueError("operator-signature input path must differ from output paths")
        return self


@dataclass(frozen=True)
class DagMetricMiningRow:
    """Per-expression row used for Goal 3.5 DAG compression mining."""

    index: int
    expression: str
    srepr: str
    tree_alpha: float
    dag_alpha_vs_ast_tree: float
    dag_alpha_vs_ast_dag: float
    eml_dag_compression: float
    ast_dag_compression: float
    ast_tree_node_count: int
    ast_dag_node_count: int
    eml_tree_node_count: int
    eml_dag_node_count: int
    operator_signature: str

    @property
    def alpha_drop(self) -> float:
        """Drop from tree alpha to DAG alpha versus AST tree size."""
        return self.tree_alpha - self.dag_alpha_vs_ast_tree

    @property
    def dag_improvement_ratio(self) -> float:
        """Tree alpha divided by DAG alpha versus AST tree size."""
        return self.tree_alpha / self.dag_alpha_vs_ast_tree

    @property
    def success_score(self) -> float:
        """High when compression is high and alpha drops substantially."""
        return self.eml_dag_compression * max(self.alpha_drop, 0.0)

    @property
    def failure_score(self) -> float:
        """High when compression is weak or DAG alpha remains high."""
        weak_compression_penalty = max(0.0, 2.0 - self.eml_dag_compression)
        return self.dag_alpha_vs_ast_tree + self.dag_alpha_vs_ast_dag + weak_compression_penalty


@dataclass(frozen=True)
class DagSignatureMiningRow:
    """Grouped operator-signature row used for Goal 3.5 mining."""

    operator_signature: str
    count: int
    median_tree_alpha: float
    median_dag_alpha_vs_ast_tree: float
    median_dag_alpha_vs_ast_dag: float
    median_eml_dag_compression: float
    p90_eml_dag_compression: float
    percent_below_threshold_after_dag: float
    percent_below_threshold_dag_vs_ast_tree: float
    percent_below_threshold_dag_vs_ast_dag: float
    median_improvement: float

    @property
    def best_score(self) -> float:
        """Score for signatures where structural DAG compression helps most."""
        return self.median_improvement * self.median_eml_dag_compression

    @property
    def worst_score(self) -> float:
        """Score for signatures where DAG alpha remains largest."""
        weak_compression_penalty = max(0.0, 2.0 - self.median_eml_dag_compression)
        return (
            self.median_dag_alpha_vs_ast_tree
            + self.median_dag_alpha_vs_ast_dag
            + weak_compression_penalty
        )

    @property
    def safe_score(self) -> float:
        """Score for candidate safe regimes under the current DAG threshold."""
        return self.percent_below_threshold_dag_vs_ast_tree


@dataclass(frozen=True)
class DagCompressionMiningResult:
    """Result metadata from a Goal 3.5 mining run."""

    dag_metric_count: int
    operator_signature_count: int
    threshold_summary_count: int
    output_paths: tuple[Path, ...]


def run_dag_compression_mining(
    config: DagCompressionMiningConfig,
) -> DagCompressionMiningResult:
    """Mine DAG compression successes and failures from saved artifacts."""
    metric_rows = load_dag_metric_mining_rows(config.dag_metrics_csv_path)
    signature_rows = load_signature_mining_rows(config.dag_operator_signature_csv_path)
    threshold_summary = load_threshold_summary(config.dag_threshold_summary_json_path)

    top_successes = select_top_successes(metric_rows, limit=config.top_n)
    top_failures = select_top_failures(metric_rows, limit=config.top_n)
    best_signatures = rank_best_operator_signatures(signature_rows, limit=config.top_n)
    worst_signatures = rank_worst_operator_signatures(signature_rows, limit=config.top_n)
    safe_candidates = rank_safe_regime_candidates(signature_rows, limit=config.top_n)

    write_top_expression_csv(top_successes, config.top_successes_csv_path)
    write_top_expression_csv(top_failures, config.top_failures_csv_path)
    write_signature_rank_csv(best_signatures, config.best_operator_signatures_csv_path)
    write_signature_rank_csv(worst_signatures, config.worst_operator_signatures_csv_path)
    write_safe_regime_candidates_csv(safe_candidates, config.safe_regime_candidates_csv_path)
    write_findings_report(
        config.report_md_path,
        top_successes=top_successes,
        top_failures=top_failures,
        best_signatures=best_signatures,
        worst_signatures=worst_signatures,
        safe_candidates=safe_candidates,
        threshold_summary=threshold_summary,
    )

    return DagCompressionMiningResult(
        dag_metric_count=len(metric_rows),
        operator_signature_count=len(signature_rows),
        threshold_summary_count=len(threshold_summary),
        output_paths=(
            config.top_successes_csv_path,
            config.top_failures_csv_path,
            config.best_operator_signatures_csv_path,
            config.worst_operator_signatures_csv_path,
            config.safe_regime_candidates_csv_path,
            config.report_md_path,
        ),
    )


def load_dag_metric_mining_rows(path: Path) -> list[DagMetricMiningRow]:
    """Load supported pure EML DAG metric rows for mining."""
    rows: list[DagMetricMiningRow] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            if parse_bool(raw_row["supported"]) and parse_bool(raw_row["pure_eml_valid"]):
                rows.append(build_dag_metric_mining_row(raw_row))
    if not rows:
        raise ValueError(f"no supported pure EML DAG rows found in {path}")
    return rows


def build_dag_metric_mining_row(raw_row: dict[str, str]) -> DagMetricMiningRow:
    """Build one mining row and derive the operator signature from srepr."""
    features = count_operator_features(raw_row["srepr"])
    return DagMetricMiningRow(
        index=parse_int(raw_row["index"]),
        expression=raw_row["expression"],
        srepr=raw_row["srepr"],
        tree_alpha=parse_float(raw_row["tree_alpha"]),
        dag_alpha_vs_ast_tree=parse_float(raw_row["dag_alpha_vs_ast_tree"]),
        dag_alpha_vs_ast_dag=parse_float(raw_row["dag_alpha_vs_ast_dag"]),
        eml_dag_compression=parse_float(raw_row["eml_dag_compression"]),
        ast_dag_compression=parse_float(raw_row["ast_dag_compression"]),
        ast_tree_node_count=parse_int(raw_row["ast_tree_node_count"]),
        ast_dag_node_count=parse_int(raw_row["ast_dag_node_count"]),
        eml_tree_node_count=parse_int(raw_row["eml_tree_node_count"]),
        eml_dag_node_count=parse_int(raw_row["eml_dag_node_count"]),
        operator_signature=operator_signature(features),
    )


def load_signature_mining_rows(path: Path) -> list[DagSignatureMiningRow]:
    """Load saved operator-signature DAG summaries for mining."""
    rows: list[DagSignatureMiningRow] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            rows.append(
                DagSignatureMiningRow(
                    operator_signature=raw_row["operator_signature"],
                    count=parse_int(raw_row["count"]),
                    median_tree_alpha=parse_float(raw_row["median_tree_alpha"]),
                    median_dag_alpha_vs_ast_tree=parse_float(
                        raw_row["median_dag_alpha_vs_ast_tree"]
                    ),
                    median_dag_alpha_vs_ast_dag=parse_float(raw_row["median_dag_alpha_vs_ast_dag"]),
                    median_eml_dag_compression=parse_float(raw_row["median_eml_dag_compression"]),
                    p90_eml_dag_compression=parse_float(raw_row["p90_eml_dag_compression"]),
                    percent_below_threshold_after_dag=parse_float(
                        raw_row["percent_below_threshold_after_dag"]
                    ),
                    percent_below_threshold_dag_vs_ast_tree=parse_float(
                        raw_row["percent_below_threshold_dag_vs_ast_tree"]
                    ),
                    percent_below_threshold_dag_vs_ast_dag=parse_float(
                        raw_row["percent_below_threshold_dag_vs_ast_dag"]
                    ),
                    median_improvement=parse_float(raw_row["median_improvement"]),
                )
            )
    if not rows:
        raise ValueError(f"no operator-signature rows found in {path}")
    return rows


def load_threshold_summary(path: Path) -> list[dict[str, Any]]:
    """Load saved Goal 3.4 threshold summaries."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list in {path}")
    return data


def select_top_successes(
    rows: Sequence[DagMetricMiningRow],
    *,
    limit: int,
) -> list[DagMetricMiningRow]:
    """Select rows where structural DAG compression helps most."""
    return sorted(
        rows,
        key=lambda row: (
            -row.success_score,
            -row.eml_dag_compression,
            -row.alpha_drop,
            row.index,
        ),
    )[:limit]


def select_top_failures(
    rows: Sequence[DagMetricMiningRow],
    *,
    limit: int,
) -> list[DagMetricMiningRow]:
    """Select rows where compression is weak or DAG alpha remains high."""
    return sorted(
        rows,
        key=lambda row: (
            -row.failure_score,
            row.eml_dag_compression,
            -row.dag_alpha_vs_ast_tree,
            row.index,
        ),
    )[:limit]


def rank_best_operator_signatures(
    rows: Sequence[DagSignatureMiningRow],
    *,
    limit: int,
) -> list[dict[str, object]]:
    """Rank operator signatures where DAG compression helps most."""
    ranked = sorted(
        rows,
        key=lambda row: (
            -row.best_score,
            -row.median_improvement,
            -row.median_eml_dag_compression,
            row.median_dag_alpha_vs_ast_tree,
            row.operator_signature,
        ),
    )[:limit]
    return [
        signature_row_to_dict(row, rank=rank, rank_score=row.best_score)
        for rank, row in enumerate(ranked, start=1)
    ]


def rank_worst_operator_signatures(
    rows: Sequence[DagSignatureMiningRow],
    *,
    limit: int,
) -> list[dict[str, object]]:
    """Rank operator signatures where DAG alpha remains highest."""
    ranked = sorted(
        rows,
        key=lambda row: (
            -row.worst_score,
            -row.median_dag_alpha_vs_ast_tree,
            row.median_eml_dag_compression,
            -row.count,
            row.operator_signature,
        ),
    )[:limit]
    return [
        signature_row_to_dict(row, rank=rank, rank_score=row.worst_score)
        for rank, row in enumerate(ranked, start=1)
    ]


def rank_safe_regime_candidates(
    rows: Sequence[DagSignatureMiningRow],
    *,
    limit: int,
) -> list[dict[str, object]]:
    """Rank operator signatures that most often cross the current DAG threshold."""
    ranked = sorted(
        rows,
        key=lambda row: (
            -row.safe_score,
            row.median_dag_alpha_vs_ast_tree,
            -row.median_improvement,
            -row.count,
            row.operator_signature,
        ),
    )[:limit]
    return [
        {
            "rank": rank,
            "candidate_kind": "operator_signature",
            **signature_row_to_dict(row, rank=None, rank_score=None),
        }
        for rank, row in enumerate(ranked, start=1)
    ]


def signature_row_to_dict(
    row: DagSignatureMiningRow,
    *,
    rank: int | None,
    rank_score: float | None,
) -> dict[str, object]:
    """Serialize one operator-signature ranking row."""
    result: dict[str, object] = {
        "operator_signature": row.operator_signature,
        "count": row.count,
        "median_tree_alpha": row.median_tree_alpha,
        "median_dag_alpha_vs_ast_tree": row.median_dag_alpha_vs_ast_tree,
        "median_dag_alpha_vs_ast_dag": row.median_dag_alpha_vs_ast_dag,
        "median_eml_dag_compression": row.median_eml_dag_compression,
        "p90_eml_dag_compression": row.p90_eml_dag_compression,
        "percent_below_threshold_after_dag": row.percent_below_threshold_after_dag,
        "percent_below_threshold_dag_vs_ast_tree": (row.percent_below_threshold_dag_vs_ast_tree),
        "percent_below_threshold_dag_vs_ast_dag": row.percent_below_threshold_dag_vs_ast_dag,
        "median_improvement": row.median_improvement,
    }
    if rank is not None:
        result = {"rank": rank, "rank_score": rank_score, **result}
    return result


def write_top_expression_csv(rows: Sequence[DagMetricMiningRow], path: Path) -> None:
    """Write ranked top-expression mining rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TOP_DAG_EXPRESSION_FIELDS)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(top_expression_to_dict(row, rank=rank))


def top_expression_to_dict(row: DagMetricMiningRow, *, rank: int) -> dict[str, object]:
    """Serialize one expression mining row."""
    return {
        "rank": rank,
        "index": row.index,
        "success_score": row.success_score,
        "failure_score": row.failure_score,
        "alpha_drop": row.alpha_drop,
        "dag_improvement_ratio": row.dag_improvement_ratio,
        "expression": row.expression,
        "srepr": row.srepr,
        "tree_alpha": row.tree_alpha,
        "dag_alpha_vs_ast_tree": row.dag_alpha_vs_ast_tree,
        "dag_alpha_vs_ast_dag": row.dag_alpha_vs_ast_dag,
        "eml_dag_compression": row.eml_dag_compression,
        "ast_dag_compression": row.ast_dag_compression,
        "ast_tree_node_count": row.ast_tree_node_count,
        "ast_dag_node_count": row.ast_dag_node_count,
        "eml_tree_node_count": row.eml_tree_node_count,
        "eml_dag_node_count": row.eml_dag_node_count,
        "operator_signature": row.operator_signature,
    }


def write_signature_rank_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Write operator-signature ranking rows."""
    write_dict_csv(rows, path, fieldnames=SIGNATURE_RANK_FIELDS)


def write_safe_regime_candidates_csv(
    rows: Sequence[dict[str, object]],
    path: Path,
) -> None:
    """Write candidate safe-regime rows."""
    write_dict_csv(rows, path, fieldnames=SAFE_REGIME_FIELDS)


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


def write_findings_report(
    path: Path,
    *,
    top_successes: Sequence[DagMetricMiningRow],
    top_failures: Sequence[DagMetricMiningRow],
    best_signatures: Sequence[dict[str, object]],
    worst_signatures: Sequence[dict[str, object]],
    safe_candidates: Sequence[dict[str, object]],
    threshold_summary: Sequence[dict[str, Any]],
) -> None:
    """Write the Goal 3.5 DAG compression findings report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_findings_report_markdown(
            top_successes=top_successes,
            top_failures=top_failures,
            best_signatures=best_signatures,
            worst_signatures=worst_signatures,
            safe_candidates=safe_candidates,
            threshold_summary=threshold_summary,
        ),
        encoding="utf-8",
    )


def build_findings_report_markdown(
    *,
    top_successes: Sequence[DagMetricMiningRow],
    top_failures: Sequence[DagMetricMiningRow],
    best_signatures: Sequence[dict[str, object]],
    worst_signatures: Sequence[dict[str, object]],
    safe_candidates: Sequence[dict[str, object]],
    threshold_summary: Sequence[dict[str, Any]],
) -> str:
    """Build the markdown body for Goal 3.5 DAG compression findings."""
    current = find_threshold_summary(threshold_summary, "current_grammar")
    current_text = (
        "No current-grammar threshold summary was available."
        if current is None
        else (
            "For `current_grammar`, percent below threshold was "
            f"`{current['percent_below_tree_alpha']}` for tree alpha, "
            f"`{current['percent_below_dag_alpha_vs_ast_tree']}` for DAG alpha vs AST tree, "
            f"and `{current['percent_below_dag_alpha_vs_ast_dag']}` for DAG alpha vs AST DAG."
        )
    )
    any_safe = any(
        float(row["percent_below_threshold_dag_vs_ast_tree"]) > 0 for row in safe_candidates
    )

    sections = [
        "# Goal 3 DAG Compression Findings",
        "",
        "This report mines exact structural DAG compression results from saved Goal 3 outputs.",
        "Compression success is ranked by high EML DAG compression and a large drop from "
        "`tree_alpha` to `dag_alpha_vs_ast_tree`.",
        "Compression failure is ranked by weak EML DAG compression or DAG alpha values that "
        "remain high after sharing.",
        "",
        "Important: these are structural representation findings only. They are not "
        "model-performance evidence.",
        "",
        "## Threshold Context",
        "",
        current_text,
        "",
        "## Top DAG Compression Successes",
        "",
        markdown_expression_table(top_successes[:5], score_field="success_score"),
        "",
        "## Top DAG Compression Failures",
        "",
        markdown_expression_table(top_failures[:5], score_field="failure_score"),
        "",
        "## Best Operator Signatures",
        "",
        markdown_signature_table(best_signatures[:5]),
        "",
        "## Worst Operator Signatures",
        "",
        markdown_signature_table(worst_signatures[:5]),
        "",
        "## Candidate Safe Regimes",
        "",
    ]
    if any_safe:
        sections.extend(
            [
                "Some operator signatures cross the current threshold after exact DAG sharing.",
                markdown_safe_table(safe_candidates[:5]),
            ]
        )
    else:
        sections.extend(
            [
                "No operator signature is a robust safe regime under the current threshold.",
                markdown_safe_table(safe_candidates[:5]),
            ]
        )
    sections.extend(
        [
            "",
            "## Interpretation",
            "",
            "- DAG compression helps most where the source tree contains repeated exact "
            "structural subtrees.",
            "- DAG compression does not rewrite algebra, commute arguments, or create hidden "
            "macro nodes; high remaining DAG alpha is therefore evidence of real structural "
            "cost in the official pure EML representation.",
            "- If DAG helps but alpha remains above threshold, the correct conclusion is that "
            "sharing reduced tree redundancy but did not by itself make the representation "
            "compact enough under that threshold.",
            "",
            "## Output Files",
            "",
            "- `outputs/v0/top_dag_compression_successes.csv`",
            "- `outputs/v0/top_dag_compression_failures.csv`",
            "- `outputs/v0/best_dag_operator_signatures.csv`",
            "- `outputs/v0/worst_dag_operator_signatures.csv`",
            "- `outputs/v0/dag_safe_regime_candidates.csv`",
            "- `outputs/v0/plots_goal3/`",
        ]
    )
    return "\n".join(sections) + "\n"


def find_threshold_summary(
    rows: Sequence[dict[str, Any]],
    scenario: str,
) -> dict[str, Any] | None:
    """Find one threshold summary by scenario name."""
    for row in rows:
        if row.get("scenario") == scenario:
            return row
    return None


def markdown_expression_table(
    rows: Sequence[DagMetricMiningRow],
    *,
    score_field: str,
) -> str:
    """Render a compact expression table for the markdown report."""
    lines = [
        "| Rank | Index | Score | Tree alpha | DAG alpha | "
        "EML DAG compression | Signature | Expression |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            f"{rank} | {row.index} | {getattr(row, score_field):.6g} | "
            f"{row.tree_alpha:.6g} | {row.dag_alpha_vs_ast_tree:.6g} | "
            f"{row.eml_dag_compression:.6g} | `{row.operator_signature}` | "
            f"`{truncate_for_markdown(row.expression, 80)}` |"
        )
    return "\n".join(lines)


def markdown_signature_table(rows: Sequence[dict[str, object]]) -> str:
    """Render a compact operator-signature table."""
    lines = [
        "| Signature | Count | Median DAG alpha | Median compression | "
        "Median improvement | Percent below DAG threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['operator_signature']}` | {row['count']} | "
            f"{row['median_dag_alpha_vs_ast_tree']} | "
            f"{row['median_eml_dag_compression']} | {row['median_improvement']} | "
            f"{row['percent_below_threshold_dag_vs_ast_tree']} |"
        )
    return "\n".join(lines)


def markdown_safe_table(rows: Sequence[dict[str, object]]) -> str:
    """Render a compact safe-regime table."""
    lines = [
        "| Signature | Count | Percent below DAG threshold | "
        "Median DAG alpha | Median compression |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['operator_signature']}` | {row['count']} | "
            f"{row['percent_below_threshold_dag_vs_ast_tree']} | "
            f"{row['median_dag_alpha_vs_ast_tree']} | "
            f"{row['median_eml_dag_compression']} |"
        )
    return "\n".join(lines)


def truncate_for_markdown(text: str, max_chars: int) -> str:
    """Truncate markdown table text without introducing newlines."""
    normalized = text.replace("\n", " ")
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dag-metrics-csv",
        type=Path,
        default=None,
        help="Saved Goal 3.3 DAG metric CSV path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of rows to export per ranking.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 3.5 DAG compression mining export."""
    args = build_parser().parse_args(argv)
    config = DagCompressionMiningConfig()
    if args.dag_metrics_csv is not None:
        config.dag_metrics_csv_path = args.dag_metrics_csv
    if args.top_n is not None:
        config.top_n = args.top_n

    result = run_dag_compression_mining(config)
    print(f"Loaded DAG metric rows: {result.dag_metric_count}")
    print(f"Loaded operator signature rows: {result.operator_signature_count}")
    print(f"Loaded threshold summary rows: {result.threshold_summary_count}")
    for path in result.output_paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
