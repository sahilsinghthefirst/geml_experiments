"""Neural ranking evaluator for e-graph candidate extraction."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from geml.compression.egraph_candidate_dataset import (
    EgraphCandidateRecord,
    group_candidate_records,
    valid_labeled_records,
)
from geml.egraph.rule_sets import RuleMode
from geml.experiments.shared import percent as _percent
from geml.experiments.shared import safe_divide as _safe_divide

NEURAL_EGRAPH_METRICS_FIELDS = [
    "index",
    "split",
    "subset_label",
    "rule_mode",
    "candidate_count",
    "exact_best_candidate_expression",
    "exact_best_eml_dag_nodes",
    "neural_candidate_expression",
    "neural_eml_dag_nodes",
    "estimated_candidate_expression",
    "estimated_eml_dag_nodes",
    "ast_candidate_expression",
    "ast_eml_dag_nodes",
    "neural_regret_vs_exact_best",
    "estimated_regret_vs_exact_best",
    "ast_regret_vs_exact_best",
    "neural_matches_exact_best",
    "estimated_matches_exact_best",
    "ast_matches_exact_best",
    "neural_compression_gain_vs_goal3_dag",
    "exact_best_compression_gain_vs_goal3_dag",
    "estimated_compression_gain_vs_goal3_dag",
    "ast_compression_gain_vs_goal3_dag",
    "exact_beam_scoring_seconds",
    "neural_scoring_seconds",
    "neural_speedup_vs_exact_scoring",
    "neural_validation_status",
    "neural_same_root_eclass",
    "neural_structural_purity_valid",
    "neural_official_eml_compiled",
    "extraction_status",
    "error",
]


class NeuralScorer(Protocol):
    """Protocol for candidate feature scorers."""

    def predict_feature_dict(self, features: dict[str, float]) -> float:
        """Predict a lower-is-better ranking score."""
        ...


@dataclass(frozen=True, slots=True)
class NeuralEgraphMetricRow:
    """One per-expression/rule-mode neural extraction evaluation row."""

    index: int
    split: str
    subset_label: str
    rule_mode: RuleMode
    candidate_count: int
    exact_best_candidate_expression: str | None
    exact_best_eml_dag_nodes: int | None
    neural_candidate_expression: str | None
    neural_eml_dag_nodes: int | None
    estimated_candidate_expression: str | None
    estimated_eml_dag_nodes: int | None
    ast_candidate_expression: str | None
    ast_eml_dag_nodes: int | None
    neural_regret_vs_exact_best: int | None
    estimated_regret_vs_exact_best: int | None
    ast_regret_vs_exact_best: int | None
    neural_matches_exact_best: bool | None
    estimated_matches_exact_best: bool | None
    ast_matches_exact_best: bool | None
    neural_compression_gain_vs_goal3_dag: float | None
    exact_best_compression_gain_vs_goal3_dag: float | None
    estimated_compression_gain_vs_goal3_dag: float | None
    ast_compression_gain_vs_goal3_dag: float | None
    exact_beam_scoring_seconds: float | None
    neural_scoring_seconds: float | None
    neural_speedup_vs_exact_scoring: float | None
    neural_validation_status: str | None
    neural_same_root_eclass: bool | None
    neural_structural_purity_valid: bool | None
    neural_official_eml_compiled: bool | None
    extraction_status: str
    error: str | None

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable row."""
        return {field: getattr(self, field) for field in NEURAL_EGRAPH_METRICS_FIELDS}

    def to_csv_dict(self) -> dict[str, object]:
        """Return a CSV-serializable row."""
        return {field: _csv_value(getattr(self, field)) for field in NEURAL_EGRAPH_METRICS_FIELDS}


def evaluate_neural_egraph_extractor(
    records: Sequence[EgraphCandidateRecord],
    *,
    model: NeuralScorer,
) -> tuple[NeuralEgraphMetricRow, ...]:
    """Evaluate neural, exact, estimated, and AST ranking on candidate groups."""
    rows: list[NeuralEgraphMetricRow] = []
    groups = group_candidate_records(records)
    for group_key in sorted(groups):
        group_records = groups[group_key]
        valid_records = list(valid_labeled_records(group_records))
        if not valid_records:
            rows.append(_failed_group_row(group_records, "all candidates failed official labeling"))
            continue
        rows.append(evaluate_candidate_group(valid_records, model=model))
    return tuple(rows)


def evaluate_candidate_group(
    records: Sequence[EgraphCandidateRecord],
    *,
    model: NeuralScorer,
) -> NeuralEgraphMetricRow:
    """Evaluate one expression/rule-mode candidate group."""
    if not records:
        raise ValueError("records must not be empty")
    exact_best = min(records, key=official_tie_break_key)
    estimated_best = min(
        records,
        key=lambda record: (
            record.baseline_estimated_eml_cost,
            record.baseline_ast_node_cost,
            record.candidate_expression,
        ),
    )
    ast_best = min(
        records,
        key=lambda record: (record.baseline_ast_node_cost, record.candidate_expression),
    )
    neural_started = time.perf_counter()
    scored_records = [
        (model.predict_feature_dict(record.candidate_ir_features), record) for record in records
    ]
    neural_scoring_seconds = time.perf_counter() - neural_started
    neural_best = min(
        scored_records,
        key=lambda item: (
            item[0],
            item[1].baseline_ast_node_cost,
            item[1].candidate_expression,
        ),
    )[1]
    exact_scoring_seconds = sum(record.exact_label_runtime_seconds for record in records)
    return NeuralEgraphMetricRow(
        index=exact_best.expression_id,
        split=exact_best.split,
        subset_label=exact_best.subset_label,
        rule_mode=exact_best.rule_mode,
        candidate_count=len(records),
        exact_best_candidate_expression=exact_best.candidate_expression,
        exact_best_eml_dag_nodes=exact_best.true_official_eml_dag_nodes,
        neural_candidate_expression=neural_best.candidate_expression,
        neural_eml_dag_nodes=neural_best.true_official_eml_dag_nodes,
        estimated_candidate_expression=estimated_best.candidate_expression,
        estimated_eml_dag_nodes=estimated_best.true_official_eml_dag_nodes,
        ast_candidate_expression=ast_best.candidate_expression,
        ast_eml_dag_nodes=ast_best.true_official_eml_dag_nodes,
        neural_regret_vs_exact_best=_regret(neural_best, exact_best),
        estimated_regret_vs_exact_best=_regret(estimated_best, exact_best),
        ast_regret_vs_exact_best=_regret(ast_best, exact_best),
        neural_matches_exact_best=neural_best.candidate_id == exact_best.candidate_id,
        estimated_matches_exact_best=estimated_best.candidate_id == exact_best.candidate_id,
        ast_matches_exact_best=ast_best.candidate_id == exact_best.candidate_id,
        neural_compression_gain_vs_goal3_dag=_compression_gain(neural_best),
        exact_best_compression_gain_vs_goal3_dag=_compression_gain(exact_best),
        estimated_compression_gain_vs_goal3_dag=_compression_gain(estimated_best),
        ast_compression_gain_vs_goal3_dag=_compression_gain(ast_best),
        exact_beam_scoring_seconds=exact_scoring_seconds,
        neural_scoring_seconds=neural_scoring_seconds,
        neural_speedup_vs_exact_scoring=_safe_divide(
            exact_scoring_seconds,
            max(neural_scoring_seconds, 1e-12),
        ),
        neural_validation_status=neural_best.semantic_validation_status,
        neural_same_root_eclass=neural_best.same_root_eclass,
        neural_structural_purity_valid=neural_best.structural_purity_valid,
        neural_official_eml_compiled=neural_best.official_eml_compiled,
        extraction_status="completed",
        error=None,
    )


def build_neural_egraph_summary(
    rows: Sequence[NeuralEgraphMetricRow],
    *,
    trained_final_reasoning_gnn: bool,
) -> dict[str, object]:
    """Build a Goal 5.4 evaluation summary."""
    success_rows = [row for row in rows if _row_valid(row)]
    return {
        "processed_group_count": len(rows),
        "success_count": len(success_rows),
        "validation_failure_count": len(rows) - len(success_rows),
        "neural_vs_exact_beam": {
            "regret_vs_exact_best": _distribution(
                row.neural_regret_vs_exact_best for row in success_rows
            ),
            "percent_matching_exact_best": _percent(
                sum(row.neural_matches_exact_best is True for row in success_rows),
                len(success_rows),
            ),
            "top1_eml_dag_nodes": _distribution(row.neural_eml_dag_nodes for row in success_rows),
            "exact_best_eml_dag_nodes": _distribution(
                row.exact_best_eml_dag_nodes for row in success_rows
            ),
        },
        "neural_vs_estimated_eml_cost": {
            "neural_regret": _distribution(row.neural_regret_vs_exact_best for row in success_rows),
            "estimated_regret": _distribution(
                row.estimated_regret_vs_exact_best for row in success_rows
            ),
            "neural_minus_estimated_regret": _distribution(
                _difference(
                    row.neural_regret_vs_exact_best,
                    row.estimated_regret_vs_exact_best,
                )
                for row in success_rows
            ),
            "estimated_percent_matching_exact_best": _percent(
                sum(row.estimated_matches_exact_best is True for row in success_rows),
                len(success_rows),
            ),
        },
        "neural_vs_ast_node_cost": {
            "neural_regret": _distribution(row.neural_regret_vs_exact_best for row in success_rows),
            "ast_regret": _distribution(row.ast_regret_vs_exact_best for row in success_rows),
            "neural_minus_ast_regret": _distribution(
                _difference(row.neural_regret_vs_exact_best, row.ast_regret_vs_exact_best)
                for row in success_rows
            ),
            "ast_percent_matching_exact_best": _percent(
                sum(row.ast_matches_exact_best is True for row in success_rows),
                len(success_rows),
            ),
        },
        "compression_gain_vs_goal3_dag": {
            "neural": _distribution(
                row.neural_compression_gain_vs_goal3_dag for row in success_rows
            ),
            "exact_best": _distribution(
                row.exact_best_compression_gain_vs_goal3_dag for row in success_rows
            ),
            "estimated": _distribution(
                row.estimated_compression_gain_vs_goal3_dag for row in success_rows
            ),
            "ast": _distribution(row.ast_compression_gain_vs_goal3_dag for row in success_rows),
        },
        "runtime_tradeoff": {
            "exact_beam_scoring_seconds": _distribution(
                row.exact_beam_scoring_seconds for row in success_rows
            ),
            "neural_scoring_seconds": _distribution(
                row.neural_scoring_seconds for row in success_rows
            ),
            "neural_speedup_vs_exact_scoring": _distribution(
                row.neural_speedup_vs_exact_scoring for row in success_rows
            ),
            "scope": "candidate cost-scoring only; excludes e-graph saturation/enumeration",
        },
        "results_by_rule_mode": {
            rule_mode: summarize_neural_rows([row for row in rows if row.rule_mode == rule_mode])
            for rule_mode in ("safe", "positive_real_formal")
        },
        "results_by_subset_label": {
            label: summarize_neural_rows(
                list(rows)
                if label == "all_v1"
                else [row for row in rows if row.subset_label == label]
            )
            for label in ("all_v1", "nontrivial_v1", "identity_heavy_v1")
        },
        "results_by_split": {
            split: summarize_neural_rows([row for row in rows if row.split == split])
            for split in ("train", "validation", "test")
        },
        "integrity": {
            "ground_truth_cost": "official_pure_eml_dag_nodes",
            "neural_model_defines_mathematical_truth": False,
            "neural_selected_candidates_compile_to_official_eml": all(
                row.neural_official_eml_compiled is True for row in success_rows
            ),
            "failed_validation_rows_reported": True,
            "modified_official_eml_compiler": False,
            "trained_final_symbolic_reasoning_gnn": trained_final_reasoning_gnn,
            "model_performance_claims": False,
            "global_optimality_claimed": False,
        },
    }


def summarize_neural_rows(rows: Sequence[NeuralEgraphMetricRow]) -> dict[str, object]:
    """Summarize neural extractor rows for one split/subset/rule mode."""
    success_rows = [row for row in rows if _row_valid(row)]
    return {
        "processed_count": len(rows),
        "success_count": len(success_rows),
        "validation_failure_count": len(rows) - len(success_rows),
        "neural_top1_eml_dag_nodes": _distribution(
            row.neural_eml_dag_nodes for row in success_rows
        ),
        "exact_best_eml_dag_nodes": _distribution(
            row.exact_best_eml_dag_nodes for row in success_rows
        ),
        "neural_regret_vs_exact_best": _distribution(
            row.neural_regret_vs_exact_best for row in success_rows
        ),
        "estimated_regret_vs_exact_best": _distribution(
            row.estimated_regret_vs_exact_best for row in success_rows
        ),
        "ast_regret_vs_exact_best": _distribution(
            row.ast_regret_vs_exact_best for row in success_rows
        ),
        "percent_matching_exact_best": _percent(
            sum(row.neural_matches_exact_best is True for row in success_rows),
            len(success_rows),
        ),
        "median_compression_gain": _distribution(
            row.neural_compression_gain_vs_goal3_dag for row in success_rows
        )["median"],
        "neural_speedup_vs_exact_scoring": _distribution(
            row.neural_speedup_vs_exact_scoring for row in success_rows
        ),
    }


def official_tie_break_key(
    record: EgraphCandidateRecord,
) -> tuple[int, int, int, int, str]:
    """Tie-break candidates by official Goal 4 exact-cost order."""
    return (
        record.true_official_eml_dag_nodes
        if record.true_official_eml_dag_nodes is not None
        else 10**12,
        record.true_official_eml_tree_nodes
        if record.true_official_eml_tree_nodes is not None
        else 10**12,
        record.true_ast_dag_nodes if record.true_ast_dag_nodes is not None else 10**12,
        record.true_ast_tree_nodes if record.true_ast_tree_nodes is not None else 10**12,
        record.candidate_expression,
    )


def _failed_group_row(
    records: Sequence[EgraphCandidateRecord],
    error: str,
) -> NeuralEgraphMetricRow:
    first = records[0]
    return NeuralEgraphMetricRow(
        index=first.expression_id,
        split=first.split,
        subset_label=first.subset_label,
        rule_mode=first.rule_mode,
        candidate_count=len(records),
        exact_best_candidate_expression=None,
        exact_best_eml_dag_nodes=None,
        neural_candidate_expression=None,
        neural_eml_dag_nodes=None,
        estimated_candidate_expression=None,
        estimated_eml_dag_nodes=None,
        ast_candidate_expression=None,
        ast_eml_dag_nodes=None,
        neural_regret_vs_exact_best=None,
        estimated_regret_vs_exact_best=None,
        ast_regret_vs_exact_best=None,
        neural_matches_exact_best=None,
        estimated_matches_exact_best=None,
        ast_matches_exact_best=None,
        neural_compression_gain_vs_goal3_dag=None,
        exact_best_compression_gain_vs_goal3_dag=None,
        estimated_compression_gain_vs_goal3_dag=None,
        ast_compression_gain_vs_goal3_dag=None,
        exact_beam_scoring_seconds=sum(record.exact_label_runtime_seconds for record in records),
        neural_scoring_seconds=None,
        neural_speedup_vs_exact_scoring=None,
        neural_validation_status=None,
        neural_same_root_eclass=None,
        neural_structural_purity_valid=None,
        neural_official_eml_compiled=False,
        extraction_status="all_candidates_failed",
        error=error,
    )


def _row_valid(row: NeuralEgraphMetricRow) -> bool:
    return (
        row.extraction_status == "completed"
        and row.neural_validation_status == "valid"
        and row.neural_same_root_eclass is True
        and row.neural_structural_purity_valid is True
        and row.neural_official_eml_compiled is True
    )


def _regret(selected: EgraphCandidateRecord, exact_best: EgraphCandidateRecord) -> int | None:
    if (
        selected.true_official_eml_dag_nodes is None
        or exact_best.true_official_eml_dag_nodes is None
    ):
        return None
    return selected.true_official_eml_dag_nodes - exact_best.true_official_eml_dag_nodes


def _compression_gain(record: EgraphCandidateRecord) -> float | None:
    return _safe_divide(record.original_eml_dag_nodes, record.true_official_eml_dag_nodes)


def _difference(left: int | float | None, right: int | float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _distribution(values: Iterable[int | float | None]) -> dict[str, float | None]:
    numeric_values = [
        float(value) for value in values if value is not None and math.isfinite(float(value))
    ]
    if not numeric_values:
        return {"mean": None, "median": None, "p90": None}
    return {
        "mean": statistics.fmean(numeric_values),
        "median": statistics.median(numeric_values),
        "p90": _quantile(numeric_values, 0.9),
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def _csv_value(value: object) -> object:
    return "" if value is None else value
