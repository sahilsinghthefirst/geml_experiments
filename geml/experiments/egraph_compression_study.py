"""Goal 4.6 e-graph compression study on the repaired v1 corpus."""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import statistics
import time
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import sympy as sp
import yaml

from geml.data.dataset import (
    GeneratedExpressionInput,
    build_symbol_locals,
    load_generated_expressions,
    parse_generated_expression,
)
from geml.egraph.egraph import EGraph
from geml.egraph.extractor import ExtractionConfig, extract_expression
from geml.egraph.ir import from_sympy, to_sympy
from geml.egraph.rewrites import SaturationLimits, saturate
from geml.egraph.rule_sets import RuleMode, rules_for_mode

type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]

EGRAPH_METRICS_FIELDS = [
    "index",
    "original_expression",
    "original_srepr",
    "rule_mode",
    "assumptions",
    "saturation_status",
    "extraction_status",
    "validation_status",
    "timeout",
    "eclass_count",
    "enode_count",
    "iterations_run",
    "total_rules_applied",
    "branch_sensitive_rules_used",
    "branch_sensitive_rule_count",
    "branch_sensitive_rule_names",
    "extracted_expression",
    "extracted_srepr",
    "validation_error",
    "max_abs_error",
    "original_ast_tree_nodes",
    "original_ast_dag_nodes",
    "original_eml_tree_nodes",
    "original_eml_dag_nodes",
    "extracted_ast_tree_nodes",
    "extracted_ast_dag_nodes",
    "extracted_eml_tree_nodes",
    "extracted_eml_dag_nodes",
    "goal3_tree_alpha",
    "goal3_dag_alpha_vs_ast_tree",
    "goal3_dag_alpha_vs_ast_dag",
    "optimized_tree_alpha",
    "optimized_dag_alpha_vs_ast_tree",
    "optimized_dag_alpha_vs_ast_dag",
    "compression_gain_vs_goal3_dag",
    "alpha_threshold_current",
    "below_threshold_goal3_dag",
    "below_threshold_optimized_dag",
    "subset_label",
    "structural_purity_valid",
    "runtime_seconds",
    "error",
]

MODE_OUTPUTS = {
    "safe": ("safe_metrics_jsonl_path", "safe_metrics_csv_path"),
    "positive_real_formal": (
        "positive_real_metrics_jsonl_path",
        "positive_real_metrics_csv_path",
    ),
}

TRIVIALITY_FEATURES = (
    "mul_by_one_count",
    "log_one_count",
    "exp_log_count",
    "log_exp_count",
    "constant_only_add_mul_count",
)


@dataclass(frozen=True, slots=True)
class EgraphCompressionStudyConfig:
    """Configuration for the Goal 4.6 v1 e-graph compression study."""

    seed: int = 0
    count: int = 10_000
    max_depth: int = 4
    operator_set: tuple[str, ...] = ("add", "mul", "exp", "log")
    symbol_names: tuple[str, ...] = ("x", "y")
    source_serialization: Literal["srepr"] = "srepr"
    input_jsonl_path: Path = Path("outputs/v1/dag_compression_inputs.jsonl")
    goal3_metrics_csv_path: Path = Path("outputs/v1/dag_compression_metrics.csv")
    goal3_summary_json_path: Path = Path("outputs/v1/dag_compression_summary.json")
    v0_v1_comparison_summary_json_path: Path | None = Path(
        "outputs/v1/v0_v1_comparison_summary.json"
    )
    run_modes: tuple[RuleMode, ...] = ("safe", "positive_real_formal")
    extractor_mode: Literal["exact_eml_dag_beam_cost"] = "exact_eml_dag_beam_cost"
    max_iterations: int = 4
    max_enodes: int = 5_000
    max_eclasses: int = 5_000
    timeout_seconds: float = 0.5
    row_timeout_seconds: float = 2.0
    beam_size: int = 12
    max_candidate_depth: int = 7
    max_candidates_evaluated: int = 12
    checkpoint_interval: int = 100
    resume: bool = True
    output_dir: Path = Path("outputs/v1")
    safe_metrics_csv_path: Path = Path("outputs/v1/egraph_compression_metrics_safe.csv")
    safe_metrics_jsonl_path: Path = Path("outputs/v1/egraph_compression_metrics_safe.jsonl")
    positive_real_metrics_csv_path: Path = Path(
        "outputs/v1/egraph_compression_metrics_positive_real.csv"
    )
    positive_real_metrics_jsonl_path: Path = Path(
        "outputs/v1/egraph_compression_metrics_positive_real.jsonl"
    )
    summary_json_path: Path = Path("outputs/v1/egraph_compression_summary.json")
    run_metadata_json_path: Path = Path("outputs/v1/egraph_compression_run_metadata.json")

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.max_depth != 4:
            raise ValueError("Goal 4.6 v1 config expects max_depth=4")
        if tuple(self.operator_set) != ("add", "mul", "exp", "log"):
            raise ValueError("Goal 4.6 v1 operator_set must be add, mul, exp, log")
        if tuple(self.symbol_names) != ("x", "y"):
            raise ValueError("Goal 4.6 v1 symbol_names must be x, y")
        if self.source_serialization != "srepr":
            raise ValueError("Goal 4.6 requires authoritative srepr input")
        if not self.run_modes:
            raise ValueError("run_modes must not be empty")
        unknown_modes = set(self.run_modes) - {"safe", "positive_real_formal"}
        if unknown_modes:
            raise ValueError(f"unsupported rule modes: {sorted(unknown_modes)}")
        if self.extractor_mode != "exact_eml_dag_beam_cost":
            raise ValueError("Goal 4.6 requires exact_eml_dag_beam_cost")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.max_enodes <= 0:
            raise ValueError("max_enodes must be positive")
        if self.max_eclasses <= 0:
            raise ValueError("max_eclasses must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.row_timeout_seconds <= 0:
            raise ValueError("row_timeout_seconds must be positive")
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if self.max_candidate_depth < 0:
            raise ValueError("max_candidate_depth must be non-negative")
        if self.max_candidates_evaluated <= 0:
            raise ValueError("max_candidates_evaluated must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        _assert_no_outputs_v0(
            [
                self.output_dir,
                self.safe_metrics_csv_path,
                self.safe_metrics_jsonl_path,
                self.positive_real_metrics_csv_path,
                self.positive_real_metrics_jsonl_path,
                self.summary_json_path,
                self.run_metadata_json_path,
            ]
        )


@dataclass(frozen=True, slots=True)
class Goal3BaselineRow:
    """Goal 3 v1 DAG baseline metrics for one expression."""

    index: int
    expression: str
    srepr: str
    ast_tree_node_count: int
    ast_dag_node_count: int
    eml_tree_node_count: int
    eml_dag_node_count: int
    tree_alpha: float
    dag_alpha_vs_ast_tree: float
    dag_alpha_vs_ast_dag: float
    alpha_threshold_current: float
    below_threshold_dag_vs_ast_tree: bool


@dataclass(frozen=True, slots=True)
class EgraphCompressionRow:
    """One Goal 4.6 per-expression e-graph metric row."""

    index: int
    original_expression: str
    original_srepr: str
    rule_mode: RuleMode
    assumptions: str | None
    saturation_status: str | None
    extraction_status: str | None
    validation_status: str | None
    timeout: bool
    eclass_count: int | None
    enode_count: int | None
    iterations_run: int | None
    total_rules_applied: int | None
    branch_sensitive_rules_used: bool
    branch_sensitive_rule_count: int
    branch_sensitive_rule_names: tuple[str, ...]
    extracted_expression: str | None
    extracted_srepr: str | None
    validation_error: str | None
    max_abs_error: float | None
    original_ast_tree_nodes: int
    original_ast_dag_nodes: int
    original_eml_tree_nodes: int
    original_eml_dag_nodes: int
    extracted_ast_tree_nodes: int | None
    extracted_ast_dag_nodes: int | None
    extracted_eml_tree_nodes: int | None
    extracted_eml_dag_nodes: int | None
    goal3_tree_alpha: float
    goal3_dag_alpha_vs_ast_tree: float
    goal3_dag_alpha_vs_ast_dag: float
    optimized_tree_alpha: float | None
    optimized_dag_alpha_vs_ast_tree: float | None
    optimized_dag_alpha_vs_ast_dag: float | None
    compression_gain_vs_goal3_dag: float | None
    alpha_threshold_current: float
    below_threshold_goal3_dag: bool
    below_threshold_optimized_dag: bool | None
    subset_label: str
    structural_purity_valid: bool
    runtime_seconds: float
    error: str | None

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable row."""
        return {
            "index": self.index,
            "original_expression": self.original_expression,
            "original_srepr": self.original_srepr,
            "rule_mode": self.rule_mode,
            "assumptions": self.assumptions,
            "saturation_status": self.saturation_status,
            "extraction_status": self.extraction_status,
            "validation_status": self.validation_status,
            "timeout": self.timeout,
            "eclass_count": self.eclass_count,
            "enode_count": self.enode_count,
            "iterations_run": self.iterations_run,
            "total_rules_applied": self.total_rules_applied,
            "branch_sensitive_rules_used": self.branch_sensitive_rules_used,
            "branch_sensitive_rule_count": self.branch_sensitive_rule_count,
            "branch_sensitive_rule_names": list(self.branch_sensitive_rule_names),
            "extracted_expression": self.extracted_expression,
            "extracted_srepr": self.extracted_srepr,
            "validation_error": self.validation_error,
            "max_abs_error": self.max_abs_error,
            "original_ast_tree_nodes": self.original_ast_tree_nodes,
            "original_ast_dag_nodes": self.original_ast_dag_nodes,
            "original_eml_tree_nodes": self.original_eml_tree_nodes,
            "original_eml_dag_nodes": self.original_eml_dag_nodes,
            "extracted_ast_tree_nodes": self.extracted_ast_tree_nodes,
            "extracted_ast_dag_nodes": self.extracted_ast_dag_nodes,
            "extracted_eml_tree_nodes": self.extracted_eml_tree_nodes,
            "extracted_eml_dag_nodes": self.extracted_eml_dag_nodes,
            "goal3_tree_alpha": self.goal3_tree_alpha,
            "goal3_dag_alpha_vs_ast_tree": self.goal3_dag_alpha_vs_ast_tree,
            "goal3_dag_alpha_vs_ast_dag": self.goal3_dag_alpha_vs_ast_dag,
            "optimized_tree_alpha": self.optimized_tree_alpha,
            "optimized_dag_alpha_vs_ast_tree": self.optimized_dag_alpha_vs_ast_tree,
            "optimized_dag_alpha_vs_ast_dag": self.optimized_dag_alpha_vs_ast_dag,
            "compression_gain_vs_goal3_dag": self.compression_gain_vs_goal3_dag,
            "alpha_threshold_current": self.alpha_threshold_current,
            "below_threshold_goal3_dag": self.below_threshold_goal3_dag,
            "below_threshold_optimized_dag": self.below_threshold_optimized_dag,
            "subset_label": self.subset_label,
            "structural_purity_valid": self.structural_purity_valid,
            "runtime_seconds": self.runtime_seconds,
            "error": self.error,
        }

    def to_csv_dict(self) -> dict[str, object]:
        """Return a CSV-serializable row."""
        row = self.to_json_dict()
        row["branch_sensitive_rule_names"] = json.dumps(
            row["branch_sensitive_rule_names"],
            sort_keys=True,
        )
        return row


@dataclass(frozen=True, slots=True)
class EgraphCompressionStudyResult:
    """Result summary for a complete Goal 4.6 run."""

    summary: dict[str, object]
    output_paths: tuple[Path, ...]


class RowTimeoutError(TimeoutError):
    """Raised when one expression exceeds the outer wall-clock row timeout."""


def load_config(path: Path) -> EgraphCompressionStudyConfig:
    """Load an e-graph compression YAML config."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return EgraphCompressionStudyConfig(
        **{key: _coerce_config_value(key, value) for key, value in raw.items()}
    )


def run_egraph_compression_study(
    config: EgraphCompressionStudyConfig,
) -> EgraphCompressionStudyResult:
    """Run safe and positive-real e-graph compression on the configured v1 corpus."""
    started_at = time.time()
    input_rows = load_generated_expressions(config.input_jsonl_path)[: config.count]
    if len(input_rows) != config.count:
        raise ValueError(f"expected {config.count} input rows, found {len(input_rows)}")
    baselines = load_goal3_baselines(config.goal3_metrics_csv_path)
    missing_baselines = [row.index for row in input_rows if row.index not in baselines]
    if missing_baselines:
        raise ValueError(f"missing Goal 3 baselines for indices: {missing_baselines[:10]}")

    all_rows_by_mode: dict[RuleMode, tuple[EgraphCompressionRow, ...]] = {}
    for rule_mode in config.run_modes:
        rows = run_rule_mode(rule_mode, input_rows, baselines, config)
        all_rows_by_mode[rule_mode] = rows
        write_metrics_csv(rows, _csv_path_for_mode(rule_mode, config))

    summary = build_summary(all_rows_by_mode, config)
    write_json(config.summary_json_path, summary)
    metadata = build_run_metadata(config, started_at=started_at, completed_at=time.time())
    write_json(config.run_metadata_json_path, metadata)
    return EgraphCompressionStudyResult(
        summary=summary,
        output_paths=tuple(
            dict.fromkeys(
                [
                    config.safe_metrics_csv_path,
                    config.safe_metrics_jsonl_path,
                    config.positive_real_metrics_csv_path,
                    config.positive_real_metrics_jsonl_path,
                    config.summary_json_path,
                    config.run_metadata_json_path,
                ]
            )
        ),
    )


def run_rule_mode(
    rule_mode: RuleMode,
    input_rows: Sequence[GeneratedExpressionInput],
    baselines: dict[int, Goal3BaselineRow],
    config: EgraphCompressionStudyConfig,
) -> tuple[EgraphCompressionRow, ...]:
    """Run or resume one rule mode."""
    jsonl_path = _jsonl_path_for_mode(rule_mode, config)
    if not config.resume and jsonl_path.exists():
        jsonl_path.unlink()
    if not config.resume:
        csv_path = _csv_path_for_mode(rule_mode, config)
        if csv_path.exists():
            csv_path.unlink()

    existing_rows = load_existing_rows(jsonl_path) if config.resume else {}
    pending_rows: list[EgraphCompressionRow] = []
    rows_by_index: dict[int, EgraphCompressionRow] = dict(existing_rows)
    symbol_locals = build_symbol_locals(config.symbol_names)

    for position, input_row in enumerate(input_rows, start=1):
        if input_row.index is None:
            raise ValueError("input row index must not be None")
        if input_row.index in rows_by_index:
            continue
        row = compute_egraph_row(
            input_row,
            baselines[input_row.index],
            symbol_locals=symbol_locals,
            rule_mode=rule_mode,
            config=config,
        )
        pending_rows.append(row)
        rows_by_index[row.index] = row
        if len(pending_rows) >= config.checkpoint_interval:
            append_jsonl_rows(jsonl_path, pending_rows)
            pending_rows.clear()
            write_checkpoint_metadata(rule_mode, rows_by_index, len(input_rows), config)

        if position % config.checkpoint_interval == 0:
            write_checkpoint_metadata(rule_mode, rows_by_index, len(input_rows), config)

    if pending_rows:
        append_jsonl_rows(jsonl_path, pending_rows)
        pending_rows.clear()
    write_checkpoint_metadata(rule_mode, rows_by_index, len(input_rows), config)

    missing_indices = sorted(
        index
        for index in (row.index for row in input_rows)
        if index is not None and index not in rows_by_index
    )
    if missing_indices:
        raise ValueError(f"mode {rule_mode} dropped rows: {missing_indices[:10]}")
    return tuple(rows_by_index[index] for index in sorted(rows_by_index))


def compute_egraph_row(
    input_row: GeneratedExpressionInput,
    baseline: Goal3BaselineRow,
    *,
    symbol_locals: dict[str, sp.Symbol],
    rule_mode: RuleMode,
    config: EgraphCompressionStudyConfig,
) -> EgraphCompressionRow:
    """Compute one e-graph compression metric row."""
    if input_row.index is None:
        raise ValueError("input row index must not be None")
    started_at = time.monotonic()
    subset_label = subset_label_for_metadata(input_row.metadata)
    base_kwargs = _base_row_kwargs(input_row, baseline, rule_mode, subset_label)
    try:
        with row_timeout(config.row_timeout_seconds):
            sympy_expr, source_serialization = parse_generated_expression(
                input_row,
                symbol_locals=symbol_locals,
            )
            if source_serialization != "srepr":
                raise ValueError("Goal 4.6 requires srepr source serialization")
            ir_expr = from_sympy(sympy_expr)
            egraph = EGraph()
            root_id = egraph.add_expr(ir_expr)
            saturation_result = saturate(
                egraph,
                rules_for_mode(rule_mode),
                limits=SaturationLimits(
                    max_iterations=config.max_iterations,
                    max_enodes=config.max_enodes,
                    max_eclasses=config.max_eclasses,
                    timeout_seconds=config.timeout_seconds,
                ),
            )
            extraction_result = extract_expression(
                egraph,
                root_id,
                original_expression=ir_expr,
                config=ExtractionConfig(
                    extractor_mode=config.extractor_mode,
                    beam_size=config.beam_size,
                    max_candidate_depth=config.max_candidate_depth,
                    max_candidates_evaluated=config.max_candidates_evaluated,
                    timeout_seconds=config.timeout_seconds,
                    allow_positive_real_rules=rule_mode == "positive_real_formal",
                    rule_mode=rule_mode,
                ),
            )
        extracted_srepr = (
            sp.srepr(to_sympy(extraction_result.expression))
            if extraction_result.expression is not None
            else None
        )
        extracted_eml_dag_nodes = extraction_result.extracted_eml_dag_nodes
        optimized_dag_alpha_vs_ast_tree = _safe_divide(
            extracted_eml_dag_nodes,
            baseline.ast_tree_node_count,
        )
        runtime_seconds = time.monotonic() - started_at
        timeout = saturation_result.status == "timeout" or extraction_result.extraction_timeout
        return EgraphCompressionRow(
            **base_kwargs,
            assumptions=extraction_result.assumptions,
            saturation_status=saturation_result.status,
            extraction_status=extraction_result.extraction_status,
            validation_status=extraction_result.validation_status,
            timeout=timeout,
            eclass_count=saturation_result.eclass_count,
            enode_count=saturation_result.enode_count,
            iterations_run=saturation_result.iterations_completed,
            total_rules_applied=saturation_result.total_applications,
            branch_sensitive_rules_used=extraction_result.branch_sensitive_rules_used,
            branch_sensitive_rule_count=extraction_result.branch_sensitive_rule_count,
            branch_sensitive_rule_names=extraction_result.branch_sensitive_rule_names,
            extracted_expression=extraction_result.extracted_expression,
            extracted_srepr=extracted_srepr,
            validation_error=_validation_error(extraction_result),
            max_abs_error=extraction_result.positive_real_max_abs_error,
            extracted_ast_tree_nodes=extraction_result.extracted_ast_tree_nodes,
            extracted_ast_dag_nodes=extraction_result.extracted_ast_dag_nodes,
            extracted_eml_tree_nodes=extraction_result.extracted_eml_tree_nodes,
            extracted_eml_dag_nodes=extracted_eml_dag_nodes,
            optimized_tree_alpha=_safe_divide(
                extraction_result.extracted_eml_tree_nodes,
                baseline.ast_tree_node_count,
            ),
            optimized_dag_alpha_vs_ast_tree=optimized_dag_alpha_vs_ast_tree,
            optimized_dag_alpha_vs_ast_dag=_safe_divide(
                extracted_eml_dag_nodes,
                baseline.ast_dag_node_count,
            ),
            compression_gain_vs_goal3_dag=_safe_divide(
                baseline.eml_dag_node_count,
                extracted_eml_dag_nodes,
            ),
            below_threshold_optimized_dag=_below_threshold(
                optimized_dag_alpha_vs_ast_tree,
                baseline.alpha_threshold_current,
            ),
            structural_purity_valid=bool(extraction_result.integrity_valid),
            runtime_seconds=runtime_seconds,
            error="; ".join(extraction_result.errors) if extraction_result.errors else None,
        )
    except RowTimeoutError as exc:
        runtime_seconds = time.monotonic() - started_at
        branch_sensitive_names = _branch_sensitive_names(rule_mode)
        return EgraphCompressionRow(
            **base_kwargs,
            assumptions="positive_real_formal" if rule_mode == "positive_real_formal" else None,
            saturation_status="timeout",
            extraction_status="timeout",
            validation_status="error",
            timeout=True,
            eclass_count=None,
            enode_count=None,
            iterations_run=None,
            total_rules_applied=None,
            branch_sensitive_rules_used=bool(branch_sensitive_names),
            branch_sensitive_rule_count=len(branch_sensitive_names),
            branch_sensitive_rule_names=branch_sensitive_names,
            extracted_expression=None,
            extracted_srepr=None,
            validation_error=f"{type(exc).__name__}: {exc}",
            max_abs_error=None,
            extracted_ast_tree_nodes=None,
            extracted_ast_dag_nodes=None,
            extracted_eml_tree_nodes=None,
            extracted_eml_dag_nodes=None,
            optimized_tree_alpha=None,
            optimized_dag_alpha_vs_ast_tree=None,
            optimized_dag_alpha_vs_ast_dag=None,
            compression_gain_vs_goal3_dag=None,
            below_threshold_optimized_dag=None,
            structural_purity_valid=True,
            runtime_seconds=runtime_seconds,
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:
        runtime_seconds = time.monotonic() - started_at
        return EgraphCompressionRow(
            **base_kwargs,
            assumptions="positive_real_formal" if rule_mode == "positive_real_formal" else None,
            saturation_status=None,
            extraction_status="failed",
            validation_status="error",
            timeout=False,
            eclass_count=None,
            enode_count=None,
            iterations_run=None,
            total_rules_applied=None,
            branch_sensitive_rules_used=rule_mode == "positive_real_formal",
            branch_sensitive_rule_count=0,
            branch_sensitive_rule_names=(),
            extracted_expression=None,
            extracted_srepr=None,
            validation_error=f"{type(exc).__name__}: {exc}",
            max_abs_error=None,
            extracted_ast_tree_nodes=None,
            extracted_ast_dag_nodes=None,
            extracted_eml_tree_nodes=None,
            extracted_eml_dag_nodes=None,
            optimized_tree_alpha=None,
            optimized_dag_alpha_vs_ast_tree=None,
            optimized_dag_alpha_vs_ast_dag=None,
            compression_gain_vs_goal3_dag=None,
            below_threshold_optimized_dag=None,
            structural_purity_valid=False,
            runtime_seconds=runtime_seconds,
            error=f"{type(exc).__name__}: {exc}",
        )


def subset_label_for_metadata(metadata: dict[str, MetadataValue]) -> str:
    """Assign subset label using measured v1 nontriviality feature counters."""
    raw_features = metadata.get("nontriviality")
    if not isinstance(raw_features, dict):
        return "all_v1"
    feature_values = {
        feature: _int_value(raw_features.get(feature, 0)) for feature in TRIVIALITY_FEATURES
    }
    if any(value > 0 for value in feature_values.values()):
        return "identity_heavy_v1"
    return "nontrivial_v1"


def load_goal3_baselines(path: Path) -> dict[int, Goal3BaselineRow]:
    """Load v1 Goal 3 DAG metrics keyed by expression index."""
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = {}
        for raw_row in csv.DictReader(csv_file):
            if raw_row.get("supported") != "True":
                continue
            row = Goal3BaselineRow(
                index=int(raw_row["index"]),
                expression=raw_row["expression"],
                srepr=raw_row["srepr"],
                ast_tree_node_count=int(raw_row["ast_tree_node_count"]),
                ast_dag_node_count=int(raw_row["ast_dag_node_count"]),
                eml_tree_node_count=int(raw_row["eml_tree_node_count"]),
                eml_dag_node_count=int(raw_row["eml_dag_node_count"]),
                tree_alpha=float(raw_row["tree_alpha"]),
                dag_alpha_vs_ast_tree=float(raw_row["dag_alpha_vs_ast_tree"]),
                dag_alpha_vs_ast_dag=float(raw_row["dag_alpha_vs_ast_dag"]),
                alpha_threshold_current=float(raw_row["alpha_threshold_current"]),
                below_threshold_dag_vs_ast_tree=_parse_bool(
                    raw_row["below_threshold_dag_vs_ast_tree"]
                ),
            )
            rows[row.index] = row
        return rows


def load_existing_rows(path: Path) -> dict[int, EgraphCompressionRow]:
    """Load existing JSONL rows for resume."""
    if not path.exists():
        return {}
    rows: dict[int, EgraphCompressionRow] = {}
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw_row = json.loads(stripped)
            if not isinstance(raw_row, dict):
                raise ValueError(f"invalid JSONL row at {path}:{line_number}")
            row = row_from_json_dict(raw_row)
            rows[row.index] = row
    return rows


def row_from_json_dict(raw_row: dict[str, object]) -> EgraphCompressionRow:
    """Reconstruct an e-graph metric row from JSON."""
    return EgraphCompressionRow(
        index=int(raw_row["index"]),
        original_expression=str(raw_row["original_expression"]),
        original_srepr=str(raw_row["original_srepr"]),
        rule_mode=_rule_mode(str(raw_row["rule_mode"])),
        assumptions=_optional_str(raw_row.get("assumptions")),
        saturation_status=_optional_str(raw_row.get("saturation_status")),
        extraction_status=_optional_str(raw_row.get("extraction_status")),
        validation_status=_optional_str(raw_row.get("validation_status")),
        timeout=bool(raw_row["timeout"]),
        eclass_count=_optional_int(raw_row.get("eclass_count")),
        enode_count=_optional_int(raw_row.get("enode_count")),
        iterations_run=_optional_int(raw_row.get("iterations_run")),
        total_rules_applied=_optional_int(raw_row.get("total_rules_applied")),
        branch_sensitive_rules_used=bool(raw_row["branch_sensitive_rules_used"]),
        branch_sensitive_rule_count=int(raw_row["branch_sensitive_rule_count"]),
        branch_sensitive_rule_names=tuple(
            str(name) for name in raw_row.get("branch_sensitive_rule_names", [])
        ),
        extracted_expression=_optional_str(raw_row.get("extracted_expression")),
        extracted_srepr=_optional_str(raw_row.get("extracted_srepr")),
        validation_error=_optional_str(raw_row.get("validation_error")),
        max_abs_error=_optional_float(raw_row.get("max_abs_error")),
        original_ast_tree_nodes=int(raw_row["original_ast_tree_nodes"]),
        original_ast_dag_nodes=int(raw_row["original_ast_dag_nodes"]),
        original_eml_tree_nodes=int(raw_row["original_eml_tree_nodes"]),
        original_eml_dag_nodes=int(raw_row["original_eml_dag_nodes"]),
        extracted_ast_tree_nodes=_optional_int(raw_row.get("extracted_ast_tree_nodes")),
        extracted_ast_dag_nodes=_optional_int(raw_row.get("extracted_ast_dag_nodes")),
        extracted_eml_tree_nodes=_optional_int(raw_row.get("extracted_eml_tree_nodes")),
        extracted_eml_dag_nodes=_optional_int(raw_row.get("extracted_eml_dag_nodes")),
        goal3_tree_alpha=float(raw_row["goal3_tree_alpha"]),
        goal3_dag_alpha_vs_ast_tree=float(raw_row["goal3_dag_alpha_vs_ast_tree"]),
        goal3_dag_alpha_vs_ast_dag=float(raw_row["goal3_dag_alpha_vs_ast_dag"]),
        optimized_tree_alpha=_optional_float(raw_row.get("optimized_tree_alpha")),
        optimized_dag_alpha_vs_ast_tree=_optional_float(
            raw_row.get("optimized_dag_alpha_vs_ast_tree")
        ),
        optimized_dag_alpha_vs_ast_dag=_optional_float(
            raw_row.get("optimized_dag_alpha_vs_ast_dag")
        ),
        compression_gain_vs_goal3_dag=_optional_float(raw_row.get("compression_gain_vs_goal3_dag")),
        alpha_threshold_current=float(raw_row["alpha_threshold_current"]),
        below_threshold_goal3_dag=bool(raw_row["below_threshold_goal3_dag"]),
        below_threshold_optimized_dag=_optional_bool(raw_row.get("below_threshold_optimized_dag")),
        subset_label=str(raw_row["subset_label"]),
        structural_purity_valid=bool(raw_row["structural_purity_valid"]),
        runtime_seconds=float(raw_row["runtime_seconds"]),
        error=_optional_str(raw_row.get("error")),
    )


def append_jsonl_rows(path: Path, rows: Sequence[EgraphCompressionRow]) -> None:
    """Append completed rows to a JSONL output file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row.to_json_dict(), sort_keys=True) + "\n")


def write_metrics_csv(rows: Sequence[EgraphCompressionRow], path: Path) -> None:
    """Write materialized CSV metrics for one rule mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=EGRAPH_METRICS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def write_checkpoint_metadata(
    rule_mode: RuleMode,
    rows_by_index: dict[int, EgraphCompressionRow],
    expected_count: int,
    config: EgraphCompressionStudyConfig,
) -> None:
    """Write resumable checkpoint metadata after each chunk."""
    mode_checkpoint = {
        "rule_mode": rule_mode,
        "processed_count": len(rows_by_index),
        "expected_count": expected_count,
        "complete": len(rows_by_index) == expected_count,
        "last_index": max(rows_by_index) if rows_by_index else None,
        "jsonl_path": str(_jsonl_path_for_mode(rule_mode, config)),
        "updated_at_unix": time.time(),
    }
    write_json(_checkpoint_path_for_mode(rule_mode, config), mode_checkpoint)

    existing = _load_json_object_if_exists(config.run_metadata_json_path)
    checkpoints = existing.get("checkpoints", {}) if isinstance(existing, dict) else {}
    if not isinstance(checkpoints, dict):
        checkpoints = {}
    checkpoints[rule_mode] = mode_checkpoint
    metadata = build_run_metadata(config, started_at=None, completed_at=None)
    metadata["checkpoints"] = checkpoints
    write_json(config.run_metadata_json_path, metadata)


def build_summary(
    rows_by_mode: dict[RuleMode, Sequence[EgraphCompressionRow]],
    config: EgraphCompressionStudyConfig,
) -> dict[str, object]:
    """Build required summary metrics by rule mode."""
    return {
        "config": config_to_json_dict(config),
        "rule_modes": {rule_mode: summarize_rows(rows) for rule_mode, rows in rows_by_mode.items()},
    }


def summarize_rows(rows: Sequence[EgraphCompressionRow]) -> dict[str, object]:
    """Summarize one rule mode."""
    success_rows = [
        row
        for row in rows
        if row.extraction_status == "completed"
        and row.validation_status == "valid"
        and row.structural_purity_valid
    ]
    timeout_rows = [row for row in rows if row.timeout]
    validation_failures = [row for row in rows if row.validation_status not in {None, "valid"}]
    improved, unchanged, worse = _improvement_groups(success_rows)
    return {
        "processed_count": len(rows),
        "success_count": len(success_rows),
        "timeout_count": len(timeout_rows),
        "validation_failure_count": len(validation_failures),
        "original_eml_dag_nodes": _distribution(row.original_eml_dag_nodes for row in rows),
        "extracted_eml_dag_nodes": _distribution(
            row.extracted_eml_dag_nodes for row in success_rows
        ),
        "goal3_dag_alpha": _distribution(row.goal3_dag_alpha_vs_ast_tree for row in rows),
        "optimized_dag_alpha": _distribution(
            row.optimized_dag_alpha_vs_ast_tree for row in success_rows
        ),
        "compression_gain_vs_goal3_dag": _distribution(
            row.compression_gain_vs_goal3_dag for row in success_rows
        ),
        "percent_improved": _percent(len(improved), len(success_rows)),
        "percent_unchanged": _percent(len(unchanged), len(success_rows)),
        "percent_worse": _percent(len(worse), len(success_rows)),
        "percent_below_threshold_before_egraph": _percent(
            sum(row.below_threshold_goal3_dag for row in rows),
            len(rows),
        ),
        "percent_below_threshold_after_egraph": _percent(
            sum(row.below_threshold_optimized_dag is True for row in success_rows),
            len(success_rows),
        ),
        "runtime_seconds": _runtime_distribution(row.runtime_seconds for row in rows),
        "results_by_subset_label": {
            label: summarize_subset(rows, label)
            for label in ("all_v1", "nontrivial_v1", "identity_heavy_v1")
        },
    }


def summarize_subset(rows: Sequence[EgraphCompressionRow], label: str) -> dict[str, object]:
    """Summarize one subset label."""
    subset_rows = (
        list(rows) if label == "all_v1" else [row for row in rows if row.subset_label == label]
    )
    if not subset_rows:
        return {
            "processed_count": 0,
            "success_count": 0,
            "percent_improved": None,
            "optimized_dag_alpha": _distribution(()),
            "compression_gain_vs_goal3_dag": _distribution(()),
        }
    success_rows = [
        row
        for row in subset_rows
        if row.extraction_status == "completed"
        and row.validation_status == "valid"
        and row.structural_purity_valid
    ]
    improved, unchanged, worse = _improvement_groups(success_rows)
    return {
        "processed_count": len(subset_rows),
        "success_count": len(success_rows),
        "timeout_count": sum(row.timeout for row in subset_rows),
        "percent_improved": _percent(len(improved), len(success_rows)),
        "percent_unchanged": _percent(len(unchanged), len(success_rows)),
        "percent_worse": _percent(len(worse), len(success_rows)),
        "goal3_dag_alpha": _distribution(row.goal3_dag_alpha_vs_ast_tree for row in subset_rows),
        "optimized_dag_alpha": _distribution(
            row.optimized_dag_alpha_vs_ast_tree for row in success_rows
        ),
        "compression_gain_vs_goal3_dag": _distribution(
            row.compression_gain_vs_goal3_dag for row in success_rows
        ),
    }


def build_run_metadata(
    config: EgraphCompressionStudyConfig,
    *,
    started_at: float | None,
    completed_at: float | None,
) -> dict[str, object]:
    """Build run metadata and resource limits."""
    metadata: dict[str, object] = {
        "config": config_to_json_dict(config),
        "resource_limits": {
            "max_iterations": config.max_iterations,
            "max_enodes": config.max_enodes,
            "max_eclasses": config.max_eclasses,
            "timeout_seconds": config.timeout_seconds,
            "row_timeout_seconds": config.row_timeout_seconds,
            "beam_size": config.beam_size,
            "max_candidate_depth": config.max_candidate_depth,
            "max_candidates_evaluated": config.max_candidates_evaluated,
        },
        "input_contract": {
            "count": config.count,
            "seed": config.seed,
            "max_depth": config.max_depth,
            "operators": list(config.operator_set),
            "symbols": list(config.symbol_names),
            "source_serialization": config.source_serialization,
            "source": "v1 generator outputs",
        },
        "started_at_unix": started_at,
        "completed_at_unix": completed_at,
    }
    if completed_at is not None and started_at is not None:
        metadata["elapsed_seconds"] = completed_at - started_at
    return metadata


def config_to_json_dict(config: EgraphCompressionStudyConfig) -> dict[str, object]:
    """Return JSON-safe config values."""
    return {
        "seed": config.seed,
        "count": config.count,
        "max_depth": config.max_depth,
        "operator_set": list(config.operator_set),
        "symbol_names": list(config.symbol_names),
        "source_serialization": config.source_serialization,
        "input_jsonl_path": str(config.input_jsonl_path),
        "goal3_metrics_csv_path": str(config.goal3_metrics_csv_path),
        "goal3_summary_json_path": str(config.goal3_summary_json_path),
        "v0_v1_comparison_summary_json_path": str(config.v0_v1_comparison_summary_json_path)
        if config.v0_v1_comparison_summary_json_path
        else None,
        "run_modes": list(config.run_modes),
        "extractor_mode": config.extractor_mode,
        "max_iterations": config.max_iterations,
        "max_enodes": config.max_enodes,
        "max_eclasses": config.max_eclasses,
        "timeout_seconds": config.timeout_seconds,
        "row_timeout_seconds": config.row_timeout_seconds,
        "beam_size": config.beam_size,
        "max_candidate_depth": config.max_candidate_depth,
        "max_candidates_evaluated": config.max_candidates_evaluated,
        "checkpoint_interval": config.checkpoint_interval,
        "resume": config.resume,
        "output_dir": str(config.output_dir),
        "safe_metrics_csv_path": str(config.safe_metrics_csv_path),
        "safe_metrics_jsonl_path": str(config.safe_metrics_jsonl_path),
        "positive_real_metrics_csv_path": str(config.positive_real_metrics_csv_path),
        "positive_real_metrics_jsonl_path": str(config.positive_real_metrics_jsonl_path),
        "summary_json_path": str(config.summary_json_path),
        "run_metadata_json_path": str(config.run_metadata_json_path),
    }


def _base_row_kwargs(
    input_row: GeneratedExpressionInput,
    baseline: Goal3BaselineRow,
    rule_mode: RuleMode,
    subset_label: str,
) -> dict[str, object]:
    if input_row.index is None:
        raise ValueError("input row index must not be None")
    return {
        "index": input_row.index,
        "original_expression": input_row.expression,
        "original_srepr": input_row.srepr or baseline.srepr,
        "rule_mode": rule_mode,
        "original_ast_tree_nodes": baseline.ast_tree_node_count,
        "original_ast_dag_nodes": baseline.ast_dag_node_count,
        "original_eml_tree_nodes": baseline.eml_tree_node_count,
        "original_eml_dag_nodes": baseline.eml_dag_node_count,
        "goal3_tree_alpha": baseline.tree_alpha,
        "goal3_dag_alpha_vs_ast_tree": baseline.dag_alpha_vs_ast_tree,
        "goal3_dag_alpha_vs_ast_dag": baseline.dag_alpha_vs_ast_dag,
        "alpha_threshold_current": baseline.alpha_threshold_current,
        "below_threshold_goal3_dag": baseline.below_threshold_dag_vs_ast_tree,
        "subset_label": subset_label,
    }


def _validation_error(extraction_result: object) -> str | None:
    errors = getattr(extraction_result, "errors", ())
    validation_status = getattr(extraction_result, "validation_status", None)
    if errors:
        return "; ".join(str(error) for error in errors)
    if validation_status not in {None, "valid"}:
        return f"validation_status={validation_status}"
    return None


@contextmanager
def row_timeout(timeout_seconds: float) -> Iterable[None]:
    """Interrupt one row if it exceeds the outer wall-clock timeout."""
    previous_handler = signal.getsignal(signal.SIGALRM)

    def handle_timeout(_signum: int, _frame: object) -> None:
        raise RowTimeoutError(f"row exceeded {timeout_seconds} seconds")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _branch_sensitive_names(rule_mode: RuleMode) -> tuple[str, ...]:
    return tuple(rule.name for rule in rules_for_mode(rule_mode) if rule.branch_sensitive)


def _jsonl_path_for_mode(rule_mode: RuleMode, config: EgraphCompressionStudyConfig) -> Path:
    attr, _ = MODE_OUTPUTS[rule_mode]
    return getattr(config, attr)


def _csv_path_for_mode(rule_mode: RuleMode, config: EgraphCompressionStudyConfig) -> Path:
    _, attr = MODE_OUTPUTS[rule_mode]
    return getattr(config, attr)


def _checkpoint_path_for_mode(rule_mode: RuleMode, config: EgraphCompressionStudyConfig) -> Path:
    return config.output_dir / f"egraph_compression_checkpoint_{_mode_slug(rule_mode)}.json"


def _mode_slug(rule_mode: RuleMode) -> str:
    return "positive_real" if rule_mode == "positive_real_formal" else "safe"


def _safe_divide(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return float(numerator) / float(denominator)


def _below_threshold(value: float | None, threshold: float) -> bool | None:
    if value is None:
        return None
    return value < threshold


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


def _runtime_distribution(values: Iterable[float]) -> dict[str, float | None]:
    distribution = _distribution(values)
    return {
        "mean_per_expression": distribution["mean"],
        "median_per_expression": distribution["median"],
        "p90_per_expression": distribution["p90"],
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
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _improvement_groups(
    rows: Sequence[EgraphCompressionRow],
) -> tuple[list[EgraphCompressionRow], list[EgraphCompressionRow], list[EgraphCompressionRow]]:
    improved: list[EgraphCompressionRow] = []
    unchanged: list[EgraphCompressionRow] = []
    worse: list[EgraphCompressionRow] = []
    for row in rows:
        if row.extracted_eml_dag_nodes is None:
            continue
        if row.extracted_eml_dag_nodes < row.original_eml_dag_nodes:
            improved.append(row)
        elif row.extracted_eml_dag_nodes == row.original_eml_dag_nodes:
            unchanged.append(row)
        else:
            worse.append(row)
    return improved, unchanged, worse


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) == "True":
        return True
    if str(value) == "False":
        return False
    raise ValueError(f"cannot parse bool: {value!r}")


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    return _parse_bool(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _int_value(value: object) -> int:
    if value is None:
        return 0
    return int(value)


def _rule_mode(value: str) -> RuleMode:
    if value not in {"safe", "positive_real_formal"}:
        raise ValueError(f"invalid rule mode: {value!r}")
    return value


def _coerce_config_value(key: str, value: object) -> object:
    path_keys = {
        "input_jsonl_path",
        "goal3_metrics_csv_path",
        "goal3_summary_json_path",
        "v0_v1_comparison_summary_json_path",
        "output_dir",
        "safe_metrics_csv_path",
        "safe_metrics_jsonl_path",
        "positive_real_metrics_csv_path",
        "positive_real_metrics_jsonl_path",
        "summary_json_path",
        "run_metadata_json_path",
    }
    tuple_keys = {"operator_set", "symbol_names", "run_modes"}
    if key in path_keys and value is not None:
        return Path(str(value))
    if key in tuple_keys:
        if not isinstance(value, list):
            raise ValueError(f"{key} must be a list")
        return tuple(value)
    return value


def _assert_no_outputs_v0(paths: Sequence[Path]) -> None:
    bad_paths = [path for path in paths if "outputs/v0" in path.as_posix()]
    if bad_paths:
        joined = ", ".join(str(path) for path in bad_paths)
        raise ValueError(f"Goal 4.6 must not write serious results to outputs/v0: {joined}")


def _load_json_object_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict[str, object]) -> None:
    """Write a JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the Goal 4.6 e-graph study."""
    parser = argparse.ArgumentParser(description="Run Goal 4.6 e-graph compression study.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/egraph_compression_v1.yaml"),
        help="Path to Goal 4.6 YAML config.",
    )
    parser.add_argument("--count", type=int, default=None, help="Optional count override.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite existing e-graph outputs instead of resuming.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Goal 4.6 from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.count is not None:
        config = replace(config, count=args.count)
    if args.no_resume:
        config = replace(config, resume=False)
    result = run_egraph_compression_study(config)
    print(f"Summary: {config.summary_json_path}")
    print(f"Run metadata: {config.run_metadata_json_path}")
    print(f"Output paths: {len(result.output_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
