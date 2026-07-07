"""Goal 4.8 success and failure mining for v1 e-graph compression."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TOP_EGRAPH_EXPRESSION_FIELDS = [
    "rank",
    "rank_score",
    "index",
    "rule_mode",
    "subset_label",
    "original_expression",
    "original_srepr",
    "saturation_status",
    "extraction_status",
    "validation_status",
    "timeout",
    "branch_sensitive_rules_used",
    "branch_sensitive_rule_names",
    "original_eml_dag_nodes",
    "extracted_eml_dag_nodes",
    "goal3_dag_alpha_vs_ast_tree",
    "optimized_dag_alpha_vs_ast_tree",
    "compression_gain_vs_goal3_dag",
    "below_threshold_goal3_dag",
    "below_threshold_optimized_dag",
    "threshold_status_improved",
    "eclass_count",
    "enode_count",
    "total_rules_applied",
    "validation_error",
    "error",
]
SIGNATURE_RANK_FIELDS = [
    "rank",
    "rank_score",
    "rule_mode",
    "subset_label",
    "operator_signature",
    "count",
    "success_count",
    "median_goal3_dag_alpha_vs_ast_tree",
    "median_optimized_dag_alpha_vs_ast_tree",
    "median_compression_gain_vs_goal3_dag",
    "p90_compression_gain_vs_goal3_dag",
    "percent_improved",
    "percent_unchanged",
    "percent_worse",
    "percent_below_threshold_before",
    "percent_below_threshold_after",
    "timeout_rate",
    "validation_failure_rate",
    "branch_sensitive_rule_usage_rate",
]
SAFE_REGIME_FIELDS = [
    "rank",
    "candidate_kind",
    *SIGNATURE_RANK_FIELDS[2:],
]


@dataclass(frozen=True, slots=True)
class EgraphCompressionMiningConfig:
    """Input and output paths for Goal 4.8 e-graph compression mining."""

    safe_metrics_csv_path: Path = Path("outputs/v1/egraph_compression_metrics_safe.csv")
    positive_real_metrics_csv_path: Path = Path(
        "outputs/v1/egraph_compression_metrics_positive_real.csv"
    )
    operator_signature_csv_path: Path = Path("outputs/v1/egraph_alpha_by_operator_signature.csv")
    operator_family_csv_path: Path = Path("outputs/v1/egraph_alpha_by_operator_family.csv")
    subset_label_csv_path: Path = Path("outputs/v1/egraph_alpha_by_subset_label.csv")
    top_successes_safe_csv_path: Path = Path("outputs/v1/top_egraph_compression_successes_safe.csv")
    top_successes_positive_real_csv_path: Path = Path(
        "outputs/v1/top_egraph_compression_successes_positive_real.csv"
    )
    top_failures_safe_csv_path: Path = Path("outputs/v1/top_egraph_compression_failures_safe.csv")
    top_failures_positive_real_csv_path: Path = Path(
        "outputs/v1/top_egraph_compression_failures_positive_real.csv"
    )
    best_operator_signatures_csv_path: Path = Path("outputs/v1/best_egraph_operator_signatures.csv")
    worst_operator_signatures_csv_path: Path = Path(
        "outputs/v1/worst_egraph_operator_signatures.csv"
    )
    safe_regime_candidates_csv_path: Path = Path("outputs/v1/egraph_safe_regime_candidates.csv")
    nontrivial_successes_csv_path: Path = Path("outputs/v1/egraph_nontrivial_successes.csv")
    identity_heavy_successes_csv_path: Path = Path("outputs/v1/egraph_identity_heavy_successes.csv")
    report_md_path: Path = Path("outputs/v1/GOAL4_EGRAPH_COMPRESSION_FINDINGS.md")
    top_n: int = 25

    @property
    def output_paths(self) -> tuple[Path, ...]:
        """Return all output paths."""
        return (
            self.top_successes_safe_csv_path,
            self.top_successes_positive_real_csv_path,
            self.top_failures_safe_csv_path,
            self.top_failures_positive_real_csv_path,
            self.best_operator_signatures_csv_path,
            self.worst_operator_signatures_csv_path,
            self.safe_regime_candidates_csv_path,
            self.nontrivial_successes_csv_path,
            self.identity_heavy_successes_csv_path,
            self.report_md_path,
        )

    def validate(self) -> None:
        """Validate that Goal 4.8 does not write primary outputs to v0."""
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        bad_paths = [path for path in self.output_paths if "outputs/v0" in path.as_posix()]
        if bad_paths:
            joined = ", ".join(str(path) for path in bad_paths)
            raise ValueError(f"Goal 4.8 must not write primary outputs to outputs/v0: {joined}")


@dataclass(frozen=True, slots=True)
class EgraphMetricMiningRow:
    """Per-expression row used for Goal 4.8 e-graph mining."""

    index: int
    original_expression: str
    original_srepr: str
    rule_mode: str
    saturation_status: str
    extraction_status: str
    validation_status: str
    timeout: bool
    branch_sensitive_rules_used: bool
    branch_sensitive_rule_names: str
    original_eml_dag_nodes: int
    extracted_eml_dag_nodes: int | None
    goal3_dag_alpha_vs_ast_tree: float
    optimized_dag_alpha_vs_ast_tree: float | None
    compression_gain_vs_goal3_dag: float | None
    below_threshold_goal3_dag: bool
    below_threshold_optimized_dag: bool | None
    subset_label: str
    structural_purity_valid: bool
    eclass_count: int | None
    enode_count: int | None
    total_rules_applied: int | None
    validation_error: str | None
    error: str | None

    @property
    def is_success(self) -> bool:
        """Return whether this row has a valid extracted EML-DAG result."""
        return (
            self.extraction_status == "completed"
            and self.validation_status == "valid"
            and not self.timeout
            and self.structural_purity_valid
            and self.extracted_eml_dag_nodes is not None
            and self.compression_gain_vs_goal3_dag is not None
            and self.optimized_dag_alpha_vs_ast_tree is not None
        )

    @property
    def improved_eml_dag(self) -> bool:
        """Return whether optimized EML-DAG node count improves over Goal 3."""
        return (
            self.extracted_eml_dag_nodes is not None
            and self.extracted_eml_dag_nodes < self.original_eml_dag_nodes
        )

    @property
    def no_improvement_or_worse(self) -> bool:
        """Return whether optimized EML-DAG node count fails to improve."""
        return (
            self.extracted_eml_dag_nodes is None
            or self.extracted_eml_dag_nodes >= self.original_eml_dag_nodes
        )

    @property
    def threshold_status_improved(self) -> bool:
        """Return whether e-graph extraction crossed the current alpha threshold."""
        return not self.below_threshold_goal3_dag and self.below_threshold_optimized_dag is True

    @property
    def success_score(self) -> float:
        """High when validation passes, gain is high, and optimized alpha is low."""
        if not self.is_success or not self.improved_eml_dag:
            return -1.0
        threshold_bonus = 10.0 if self.threshold_status_improved else 0.0
        alpha_penalty = self.optimized_dag_alpha_vs_ast_tree or 0.0
        return (self.compression_gain_vs_goal3_dag or 0.0) + threshold_bonus - 0.01 * alpha_penalty

    @property
    def failure_score(self) -> float:
        """High when extraction fails, times out, or produces weak compression."""
        validation_penalty = 1000.0 if self.validation_status != "valid" else 0.0
        timeout_penalty = 800.0 if self.timeout else 0.0
        no_improve_penalty = 200.0 if self.no_improvement_or_worse else 0.0
        alpha = self.optimized_dag_alpha_vs_ast_tree or self.goal3_dag_alpha_vs_ast_tree
        weak_gain_penalty = max(0.0, 1.0 - (self.compression_gain_vs_goal3_dag or 0.0)) * 100.0
        return validation_penalty + timeout_penalty + no_improve_penalty + alpha + weak_gain_penalty


@dataclass(frozen=True, slots=True)
class EgraphSignatureMiningRow:
    """Grouped operator-signature row used for Goal 4.8 mining."""

    rule_mode: str
    subset_label: str
    operator_signature: str
    count: int
    success_count: int
    median_goal3_dag_alpha_vs_ast_tree: float
    median_optimized_dag_alpha_vs_ast_tree: float | None
    median_compression_gain_vs_goal3_dag: float | None
    p90_compression_gain_vs_goal3_dag: float | None
    percent_improved: float | None
    percent_unchanged: float | None
    percent_worse: float | None
    percent_below_threshold_before: float
    percent_below_threshold_after: float | None
    timeout_rate: float
    validation_failure_rate: float
    branch_sensitive_rule_usage_rate: float

    @property
    def best_score(self) -> float:
        """Score for signatures where e-graph extraction helps most."""
        gain = self.median_compression_gain_vs_goal3_dag or 0.0
        improvement = self.percent_improved or 0.0
        threshold_after = self.percent_below_threshold_after or 0.0
        failure_penalty = self.timeout_rate + self.validation_failure_rate
        return gain * 10.0 + improvement + threshold_after - failure_penalty

    @property
    def worst_score(self) -> float:
        """Score for signatures where optimized alpha remains poor or runs fail."""
        alpha = self.median_optimized_dag_alpha_vs_ast_tree
        if alpha is None:
            alpha = self.median_goal3_dag_alpha_vs_ast_tree
        worse = self.percent_worse or 0.0
        unchanged = self.percent_unchanged or 0.0
        return (
            alpha * 10.0
            + worse
            + 0.25 * unchanged
            + self.timeout_rate
            + self.validation_failure_rate
        )

    @property
    def safe_regime_score(self) -> float:
        """Score for safe-mode candidate regimes."""
        after = self.percent_below_threshold_after or 0.0
        improved = self.percent_improved or 0.0
        return after + 0.1 * improved - self.timeout_rate - self.validation_failure_rate


@dataclass(frozen=True, slots=True)
class EgraphCompressionMiningResult:
    """Result metadata from a Goal 4.8 mining run."""

    safe_metric_count: int
    positive_real_metric_count: int
    operator_signature_count: int
    operator_family_count: int
    subset_summary_count: int
    output_paths: tuple[Path, ...]


def run_egraph_compression_mining(
    config: EgraphCompressionMiningConfig,
) -> EgraphCompressionMiningResult:
    """Mine e-graph successes and failures from saved v1 artifacts."""
    config.validate()
    safe_rows = load_egraph_metric_mining_rows(config.safe_metrics_csv_path)
    positive_rows = load_egraph_metric_mining_rows(config.positive_real_metrics_csv_path)
    all_rows = [*safe_rows, *positive_rows]
    signature_rows = load_signature_mining_rows(config.operator_signature_csv_path)
    operator_family_rows = load_csv_rows(config.operator_family_csv_path)
    subset_summary_rows = load_csv_rows(config.subset_label_csv_path)

    top_safe_successes = select_top_successes(safe_rows, limit=config.top_n)
    top_positive_successes = select_top_successes(positive_rows, limit=config.top_n)
    top_safe_failures = select_top_failures(safe_rows, limit=config.top_n)
    top_positive_failures = select_top_failures(positive_rows, limit=config.top_n)
    best_signatures = rank_best_operator_signatures(signature_rows, limit=config.top_n)
    worst_signatures = rank_worst_operator_signatures(signature_rows, limit=config.top_n)
    safe_candidates = rank_safe_regime_candidates(signature_rows, limit=config.top_n)
    nontrivial_successes = select_subset_successes(
        all_rows,
        subset_label="nontrivial_v1",
        limit=config.top_n,
    )
    identity_heavy_successes = select_subset_successes(
        all_rows,
        subset_label="identity_heavy_v1",
        limit=config.top_n,
    )

    write_top_expression_csv(top_safe_successes, config.top_successes_safe_csv_path)
    write_top_expression_csv(
        top_positive_successes,
        config.top_successes_positive_real_csv_path,
    )
    write_top_expression_csv(top_safe_failures, config.top_failures_safe_csv_path)
    write_top_expression_csv(
        top_positive_failures,
        config.top_failures_positive_real_csv_path,
    )
    write_signature_rank_csv(best_signatures, config.best_operator_signatures_csv_path)
    write_signature_rank_csv(worst_signatures, config.worst_operator_signatures_csv_path)
    write_safe_regime_candidates_csv(safe_candidates, config.safe_regime_candidates_csv_path)
    write_top_expression_csv(nontrivial_successes, config.nontrivial_successes_csv_path)
    write_top_expression_csv(identity_heavy_successes, config.identity_heavy_successes_csv_path)
    write_findings_report(
        config.report_md_path,
        rows=all_rows,
        top_safe_successes=top_safe_successes,
        top_positive_successes=top_positive_successes,
        top_safe_failures=top_safe_failures,
        top_positive_failures=top_positive_failures,
        best_signatures=best_signatures,
        worst_signatures=worst_signatures,
        safe_candidates=safe_candidates,
        nontrivial_successes=nontrivial_successes,
        identity_heavy_successes=identity_heavy_successes,
        output_paths=config.output_paths[:-1],
    )

    return EgraphCompressionMiningResult(
        safe_metric_count=len(safe_rows),
        positive_real_metric_count=len(positive_rows),
        operator_signature_count=len(signature_rows),
        operator_family_count=len(operator_family_rows),
        subset_summary_count=len(subset_summary_rows),
        output_paths=config.output_paths,
    )


def load_egraph_metric_mining_rows(path: Path) -> list[EgraphMetricMiningRow]:
    """Load one saved Goal 4.6 per-expression metrics CSV."""
    rows: list[EgraphMetricMiningRow] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            rows.append(build_egraph_metric_mining_row(raw_row))
    if not rows:
        raise ValueError(f"no e-graph metric rows found in {path}")
    return rows


def build_egraph_metric_mining_row(raw_row: dict[str, str]) -> EgraphMetricMiningRow:
    """Build one per-expression mining row."""
    return EgraphMetricMiningRow(
        index=parse_int(raw_row["index"]),
        original_expression=raw_row["original_expression"],
        original_srepr=raw_row["original_srepr"],
        rule_mode=raw_row["rule_mode"],
        saturation_status=status_value(raw_row.get("saturation_status")),
        extraction_status=status_value(raw_row.get("extraction_status")),
        validation_status=status_value(raw_row.get("validation_status")),
        timeout=parse_bool(raw_row.get("timeout", "False")),
        branch_sensitive_rules_used=parse_bool(raw_row.get("branch_sensitive_rules_used", "False")),
        branch_sensitive_rule_names=raw_row.get("branch_sensitive_rule_names", ""),
        original_eml_dag_nodes=parse_int(raw_row["original_eml_dag_nodes"]),
        extracted_eml_dag_nodes=parse_optional_int(raw_row.get("extracted_eml_dag_nodes")),
        goal3_dag_alpha_vs_ast_tree=parse_float(raw_row["goal3_dag_alpha_vs_ast_tree"]),
        optimized_dag_alpha_vs_ast_tree=parse_optional_float(
            raw_row.get("optimized_dag_alpha_vs_ast_tree")
        ),
        compression_gain_vs_goal3_dag=parse_optional_float(
            raw_row.get("compression_gain_vs_goal3_dag")
        ),
        below_threshold_goal3_dag=parse_bool(raw_row["below_threshold_goal3_dag"]),
        below_threshold_optimized_dag=parse_optional_bool(
            raw_row.get("below_threshold_optimized_dag")
        ),
        subset_label=raw_row["subset_label"],
        structural_purity_valid=parse_bool(raw_row.get("structural_purity_valid", "True")),
        eclass_count=parse_optional_int(raw_row.get("eclass_count")),
        enode_count=parse_optional_int(raw_row.get("enode_count")),
        total_rules_applied=parse_optional_int(raw_row.get("total_rules_applied")),
        validation_error=optional_str(raw_row.get("validation_error")),
        error=optional_str(raw_row.get("error")),
    )


def load_signature_mining_rows(path: Path) -> list[EgraphSignatureMiningRow]:
    """Load saved operator-signature summaries for mining."""
    rows: list[EgraphSignatureMiningRow] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            rows.append(
                EgraphSignatureMiningRow(
                    rule_mode=raw_row["rule_mode"],
                    subset_label=raw_row["subset_label"],
                    operator_signature=raw_row["operator_signature"],
                    count=parse_int(raw_row["count"]),
                    success_count=parse_int(raw_row["success_count"]),
                    median_goal3_dag_alpha_vs_ast_tree=parse_float(
                        raw_row["median_goal3_dag_alpha_vs_ast_tree"]
                    ),
                    median_optimized_dag_alpha_vs_ast_tree=parse_optional_float(
                        raw_row.get("median_optimized_dag_alpha_vs_ast_tree")
                    ),
                    median_compression_gain_vs_goal3_dag=parse_optional_float(
                        raw_row.get("median_compression_gain_vs_goal3_dag")
                    ),
                    p90_compression_gain_vs_goal3_dag=parse_optional_float(
                        raw_row.get("p90_compression_gain_vs_goal3_dag")
                    ),
                    percent_improved=parse_optional_float(raw_row.get("percent_improved")),
                    percent_unchanged=parse_optional_float(raw_row.get("percent_unchanged")),
                    percent_worse=parse_optional_float(raw_row.get("percent_worse")),
                    percent_below_threshold_before=parse_float(
                        raw_row["percent_below_threshold_before"]
                    ),
                    percent_below_threshold_after=parse_optional_float(
                        raw_row.get("percent_below_threshold_after")
                    ),
                    timeout_rate=parse_float(raw_row["timeout_rate"]),
                    validation_failure_rate=parse_float(raw_row["validation_failure_rate"]),
                    branch_sensitive_rule_usage_rate=parse_float(
                        raw_row["branch_sensitive_rule_usage_rate"]
                    ),
                )
            )
    if not rows:
        raise ValueError(f"no operator-signature rows found in {path}")
    return rows


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a saved CSV artifact."""
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def select_top_successes(
    rows: Sequence[EgraphMetricMiningRow],
    *,
    limit: int,
) -> list[EgraphMetricMiningRow]:
    """Select rows where e-graph compression helps most."""
    candidates = [row for row in rows if row.is_success and row.improved_eml_dag]
    return sorted(
        candidates,
        key=lambda row: (
            -int(row.threshold_status_improved),
            -(row.compression_gain_vs_goal3_dag or 0.0),
            row.optimized_dag_alpha_vs_ast_tree or float("inf"),
            row.index,
        ),
    )[:limit]


def select_top_failures(
    rows: Sequence[EgraphMetricMiningRow],
    *,
    limit: int,
) -> list[EgraphMetricMiningRow]:
    """Select rows where e-graph compression fails or remains expensive."""
    return sorted(
        rows,
        key=lambda row: (
            -row.failure_score,
            -(row.optimized_dag_alpha_vs_ast_tree or row.goal3_dag_alpha_vs_ast_tree),
            row.index,
        ),
    )[:limit]


def select_subset_successes(
    rows: Sequence[EgraphMetricMiningRow],
    *,
    subset_label: str,
    limit: int,
) -> list[EgraphMetricMiningRow]:
    """Select top successes within one measured v1 subset."""
    return select_top_successes(
        [row for row in rows if row.subset_label == subset_label],
        limit=limit,
    )


def rank_best_operator_signatures(
    rows: Sequence[EgraphSignatureMiningRow],
    *,
    limit: int,
) -> list[dict[str, object]]:
    """Rank operator signatures where e-graph extraction helps most."""
    ranked = sorted(
        [row for row in rows if row.subset_label == "all_v1"],
        key=lambda row: (
            -row.best_score,
            -(row.median_compression_gain_vs_goal3_dag or 0.0),
            -(row.percent_improved or 0.0),
            row.rule_mode,
            row.operator_signature,
        ),
    )[:limit]
    return [
        signature_row_to_dict(row, rank=rank, rank_score=row.best_score)
        for rank, row in enumerate(ranked, start=1)
    ]


def rank_worst_operator_signatures(
    rows: Sequence[EgraphSignatureMiningRow],
    *,
    limit: int,
) -> list[dict[str, object]]:
    """Rank operator signatures where e-graph extraction performs worst."""
    ranked = sorted(
        [row for row in rows if row.subset_label == "all_v1"],
        key=lambda row: (
            -row.worst_score,
            -(row.median_optimized_dag_alpha_vs_ast_tree or row.median_goal3_dag_alpha_vs_ast_tree),
            -row.timeout_rate,
            -row.validation_failure_rate,
            row.rule_mode,
            row.operator_signature,
        ),
    )[:limit]
    return [
        signature_row_to_dict(row, rank=rank, rank_score=row.worst_score)
        for rank, row in enumerate(ranked, start=1)
    ]


def rank_safe_regime_candidates(
    rows: Sequence[EgraphSignatureMiningRow],
    *,
    limit: int,
) -> list[dict[str, object]]:
    """Rank safe-mode signature regimes that most often pass the threshold."""
    ranked = sorted(
        [row for row in rows if row.rule_mode == "safe"],
        key=lambda row: (
            -row.safe_regime_score,
            row.subset_label != "nontrivial_v1",
            -(row.percent_below_threshold_after or 0.0),
            -row.success_count,
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
    row: EgraphSignatureMiningRow,
    *,
    rank: int | None,
    rank_score: float | None,
) -> dict[str, object]:
    """Serialize one operator-signature mining row."""
    result: dict[str, object] = {
        "rule_mode": row.rule_mode,
        "subset_label": row.subset_label,
        "operator_signature": row.operator_signature,
        "count": row.count,
        "success_count": row.success_count,
        "median_goal3_dag_alpha_vs_ast_tree": row.median_goal3_dag_alpha_vs_ast_tree,
        "median_optimized_dag_alpha_vs_ast_tree": row.median_optimized_dag_alpha_vs_ast_tree,
        "median_compression_gain_vs_goal3_dag": row.median_compression_gain_vs_goal3_dag,
        "p90_compression_gain_vs_goal3_dag": row.p90_compression_gain_vs_goal3_dag,
        "percent_improved": row.percent_improved,
        "percent_unchanged": row.percent_unchanged,
        "percent_worse": row.percent_worse,
        "percent_below_threshold_before": row.percent_below_threshold_before,
        "percent_below_threshold_after": row.percent_below_threshold_after,
        "timeout_rate": row.timeout_rate,
        "validation_failure_rate": row.validation_failure_rate,
        "branch_sensitive_rule_usage_rate": row.branch_sensitive_rule_usage_rate,
    }
    if rank is not None:
        result = {"rank": rank, "rank_score": rank_score, **result}
    return result


def write_top_expression_csv(rows: Sequence[EgraphMetricMiningRow], path: Path) -> None:
    """Write ranked per-expression mining rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TOP_EGRAPH_EXPRESSION_FIELDS)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(top_expression_to_dict(row, rank=rank))


def top_expression_to_dict(row: EgraphMetricMiningRow, *, rank: int) -> dict[str, object]:
    """Serialize one ranked per-expression mining row."""
    score = row.success_score if row.success_score >= 0 else row.failure_score
    return {
        "rank": rank,
        "rank_score": score,
        "index": row.index,
        "rule_mode": row.rule_mode,
        "subset_label": row.subset_label,
        "original_expression": row.original_expression,
        "original_srepr": row.original_srepr,
        "saturation_status": row.saturation_status,
        "extraction_status": row.extraction_status,
        "validation_status": row.validation_status,
        "timeout": row.timeout,
        "branch_sensitive_rules_used": row.branch_sensitive_rules_used,
        "branch_sensitive_rule_names": row.branch_sensitive_rule_names,
        "original_eml_dag_nodes": row.original_eml_dag_nodes,
        "extracted_eml_dag_nodes": row.extracted_eml_dag_nodes,
        "goal3_dag_alpha_vs_ast_tree": row.goal3_dag_alpha_vs_ast_tree,
        "optimized_dag_alpha_vs_ast_tree": row.optimized_dag_alpha_vs_ast_tree,
        "compression_gain_vs_goal3_dag": row.compression_gain_vs_goal3_dag,
        "below_threshold_goal3_dag": row.below_threshold_goal3_dag,
        "below_threshold_optimized_dag": row.below_threshold_optimized_dag,
        "threshold_status_improved": row.threshold_status_improved,
        "eclass_count": row.eclass_count,
        "enode_count": row.enode_count,
        "total_rules_applied": row.total_rules_applied,
        "validation_error": row.validation_error,
        "error": row.error,
    }


def write_signature_rank_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Write ranked operator-signature rows."""
    write_dict_csv(rows, path, fieldnames=SIGNATURE_RANK_FIELDS)


def write_safe_regime_candidates_csv(
    rows: Sequence[dict[str, object]],
    path: Path,
) -> None:
    """Write safe-regime candidate rows."""
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
    rows: Sequence[EgraphMetricMiningRow],
    top_safe_successes: Sequence[EgraphMetricMiningRow],
    top_positive_successes: Sequence[EgraphMetricMiningRow],
    top_safe_failures: Sequence[EgraphMetricMiningRow],
    top_positive_failures: Sequence[EgraphMetricMiningRow],
    best_signatures: Sequence[dict[str, object]],
    worst_signatures: Sequence[dict[str, object]],
    safe_candidates: Sequence[dict[str, object]],
    nontrivial_successes: Sequence[EgraphMetricMiningRow],
    identity_heavy_successes: Sequence[EgraphMetricMiningRow],
    output_paths: Sequence[Path],
) -> None:
    """Write the Goal 4.8 e-graph compression findings report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_findings_report_markdown(
            rows=rows,
            top_safe_successes=top_safe_successes,
            top_positive_successes=top_positive_successes,
            top_safe_failures=top_safe_failures,
            top_positive_failures=top_positive_failures,
            best_signatures=best_signatures,
            worst_signatures=worst_signatures,
            safe_candidates=safe_candidates,
            nontrivial_successes=nontrivial_successes,
            identity_heavy_successes=identity_heavy_successes,
            output_paths=output_paths,
        ),
        encoding="utf-8",
    )


def build_findings_report_markdown(
    *,
    rows: Sequence[EgraphMetricMiningRow],
    top_safe_successes: Sequence[EgraphMetricMiningRow],
    top_positive_successes: Sequence[EgraphMetricMiningRow],
    top_safe_failures: Sequence[EgraphMetricMiningRow],
    top_positive_failures: Sequence[EgraphMetricMiningRow],
    best_signatures: Sequence[dict[str, object]],
    worst_signatures: Sequence[dict[str, object]],
    safe_candidates: Sequence[dict[str, object]],
    nontrivial_successes: Sequence[EgraphMetricMiningRow],
    identity_heavy_successes: Sequence[EgraphMetricMiningRow],
    output_paths: Sequence[Path],
) -> str:
    """Build the markdown body for the Goal 4.8 findings report."""
    safe_rows = [row for row in rows if row.rule_mode == "safe"]
    positive_rows = [row for row in rows if row.rule_mode == "positive_real_formal"]
    safe_stats = summarize_mode_for_report(safe_rows)
    positive_stats = summarize_mode_for_report(positive_rows)

    sections = [
        "# Goal 4 E-Graph Compression Findings",
        "",
        "This report mines v1 corpus e-graph compression outputs from saved Goal 4.6 and "
        "Goal 4.7 artifacts. The v0 corpus is pilot only and is not used for these "
        "result-bearing findings.",
        "",
        "E-graphs here are non-ML compression: they search algebraic equivalences and "
        "then recompile the selected source expression through the official pure EML "
        "compiler. Improvements are structural compression results, not GNN evidence "
        "or model-performance evidence.",
        "",
        "`safe` and `positive_real_formal` are separate modes. `positive_real_formal` "
        "uses branch-sensitive positive-real assumptions and must not be mixed with "
        "safe-mode results. Successful final EML outputs remain official pure EML after "
        "extraction; rows with timeout or validation failure are kept visible instead "
        "of being silently dropped.",
        "",
        "`nontrivial_v1` and `identity_heavy_v1` are reported separately to avoid "
        "overstating easy simplifications from identities such as multiplication by "
        "one, log one, or log/exp cancellation.",
        "",
        "## Mode Summary",
        "",
        "| Rule mode | Rows | Valid non-timeout successes | Timeouts | "
        "Validation failures | Pure failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        mode_summary_line("safe", safe_stats),
        mode_summary_line("positive_real_formal", positive_stats),
        "",
        "## Top Safe Successes",
        "",
        markdown_expression_table(top_safe_successes[:5], score_field="success_score"),
        "",
        "## Top Positive-Real Successes",
        "",
        markdown_expression_table(top_positive_successes[:5], score_field="success_score"),
        "",
        "## Top Safe Failures",
        "",
        markdown_expression_table(top_safe_failures[:5], score_field="failure_score"),
        "",
        "## Top Positive-Real Failures",
        "",
        markdown_expression_table(top_positive_failures[:5], score_field="failure_score"),
        "",
        "## Best Operator Signatures",
        "",
        markdown_signature_table(best_signatures[:8]),
        "",
        "## Worst Operator Signatures",
        "",
        markdown_signature_table(worst_signatures[:8]),
        "",
        "## Safe Regime Candidates",
        "",
        markdown_safe_regime_table(safe_candidates[:8]),
        "",
        "## Subset-Specific Successes",
        "",
        "Top `nontrivial_v1` successes:",
        "",
        markdown_expression_table(nontrivial_successes[:5], score_field="success_score"),
        "",
        "Top `identity_heavy_v1` successes:",
        "",
        markdown_expression_table(identity_heavy_successes[:5], score_field="success_score"),
        "",
        "## Output Files",
        "",
        *[f"- `{path}`" for path in output_paths],
    ]
    return "\n".join(sections) + "\n"


def summarize_mode_for_report(rows: Sequence[EgraphMetricMiningRow]) -> dict[str, int]:
    """Summarize one rule mode for the markdown report."""
    return {
        "rows": len(rows),
        "successes": sum(row.is_success for row in rows),
        "timeouts": sum(row.timeout for row in rows),
        "validation_failures": sum(row.validation_status != "valid" for row in rows),
        "pure_failures": sum(
            row.extraction_status == "completed"
            and row.validation_status == "valid"
            and not row.structural_purity_valid
            for row in rows
        ),
    }


def mode_summary_line(rule_mode: str, stats: dict[str, int]) -> str:
    """Render one report summary table row."""
    return (
        f"| `{rule_mode}` | {stats['rows']} | {stats['successes']} | "
        f"{stats['timeouts']} | {stats['validation_failures']} | {stats['pure_failures']} |"
    )


def markdown_expression_table(
    rows: Sequence[EgraphMetricMiningRow],
    *,
    score_field: str,
) -> str:
    """Render a compact expression table for the markdown report."""
    lines = [
        "| Rank | Index | Mode | Score | Gain | Optimized alpha | Subset | Expression |",
        "| ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            f"{rank} | {row.index} | `{row.rule_mode}` | "
            f"{getattr(row, score_field):.6g} | "
            f"{format_optional_float(row.compression_gain_vs_goal3_dag)} | "
            f"{format_optional_float(row.optimized_dag_alpha_vs_ast_tree)} | "
            f"`{row.subset_label}` | `{truncate_for_markdown(row.original_expression, 80)}` |"
        )
    return "\n".join(lines)


def markdown_signature_table(rows: Sequence[dict[str, object]]) -> str:
    """Render a compact operator-signature table."""
    lines = [
        "| Mode | Signature | Count | Successes | Median optimized alpha | "
        "Median gain | Percent improved | Timeout rate | Validation failure rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['rule_mode']}` | `{row['operator_signature']}` | {row['count']} | "
            f"{row['success_count']} | {row['median_optimized_dag_alpha_vs_ast_tree']} | "
            f"{row['median_compression_gain_vs_goal3_dag']} | {row['percent_improved']} | "
            f"{row['timeout_rate']} | {row['validation_failure_rate']} |"
        )
    return "\n".join(lines)


def markdown_safe_regime_table(rows: Sequence[dict[str, object]]) -> str:
    """Render a compact safe-regime candidate table."""
    lines = [
        "| Subset | Signature | Count | Percent below after | Percent improved | Timeout rate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['subset_label']}` | `{row['operator_signature']}` | {row['count']} | "
            f"{row['percent_below_threshold_after']} | {row['percent_improved']} | "
            f"{row['timeout_rate']} |"
        )
    return "\n".join(lines)


def format_optional_float(value: float | None) -> str:
    """Format an optional float for compact markdown tables."""
    if value is None:
        return ""
    return f"{value:.6g}"


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


def parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer CSV field."""
    if value in {None, ""}:
        return None
    return int(value)


def parse_float(value: str) -> float:
    """Parse a required float CSV field."""
    if value == "":
        raise ValueError("expected float, got empty string")
    return float(value)


def parse_optional_float(value: str | None) -> float | None:
    """Parse an optional float CSV field."""
    if value in {None, ""}:
        return None
    return float(value)


def parse_bool(value: str | bool | None) -> bool:
    """Parse a required boolean CSV field."""
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected boolean string, got {value!r}")


def parse_optional_bool(value: str | bool | None) -> bool | None:
    """Parse an optional boolean CSV field."""
    if value in {None, ""}:
        return None
    return parse_bool(value)


def status_value(value: str | None) -> str:
    """Normalize optional status text from CSV."""
    return value if value not in {None, ""} else "missing"


def optional_str(value: str | None) -> str | None:
    """Normalize optional text from CSV."""
    if value in {None, ""}:
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 4.8 e-graph compression mining export."""
    args = build_parser().parse_args(argv)
    config = EgraphCompressionMiningConfig()
    if args.top_n is not None:
        config = EgraphCompressionMiningConfig(top_n=args.top_n)

    result = run_egraph_compression_mining(config)
    print(f"Loaded safe metric rows: {result.safe_metric_count}")
    print(f"Loaded positive-real metric rows: {result.positive_real_metric_count}")
    print(f"Loaded operator-signature summary rows: {result.operator_signature_count}")
    print(f"Loaded operator-family summary rows: {result.operator_family_count}")
    print(f"Loaded subset summary rows: {result.subset_summary_count}")
    for path in result.output_paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
