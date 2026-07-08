"""Candidate dataset generation for neural e-graph extraction cost models."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sympy as sp
from pydantic import BaseModel, Field

from geml.compression.motif_dataset import SplitConfig, SplitName, assign_split
from geml.data.dataset import (
    GeneratedExpressionInput,
    build_symbol_locals,
    load_generated_expressions,
    parse_generated_expression,
)
from geml.egraph.costs import ast_node_cost, estimated_eml_cost, exact_eml_dag_cost
from geml.egraph.egraph import EGraph
from geml.egraph.extractor import ExtractionConfig, enumerate_candidates
from geml.egraph.ir import (
    Add,
    Const,
    Div,
    Exp,
    Expr,
    Log,
    Mul,
    Neg,
    Pow,
    Sub,
    Var,
    display,
    to_sympy,
)
from geml.egraph.rewrites import SaturationLimits, saturate
from geml.egraph.rule_sets import RuleMode, rules_for_mode
from geml.egraph.validation import positive_real_numeric_validation, validate_same_eclass
from geml.experiments.egraph_compression_study import (
    Goal3BaselineRow,
    RowTimeoutError,
    load_goal3_baselines,
    row_timeout,
    subset_label_for_metadata,
)

type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]

DEFAULT_FEATURE_NAMES = (
    "candidate_ast_node_cost",
    "candidate_estimated_eml_cost",
    "candidate_depth",
    "candidate_leaf_count",
    "candidate_variable_count",
    "candidate_constant_count",
    "candidate_add_count",
    "candidate_mul_count",
    "candidate_sub_count",
    "candidate_div_count",
    "candidate_pow_count",
    "candidate_neg_count",
    "candidate_exp_count",
    "candidate_log_count",
    "candidate_max_constant_abs",
    "candidate_sum_constant_abs",
    "source_ast_node_count",
    "source_ast_dag_node_count",
    "source_eml_dag_node_count",
    "candidate_rank",
    "egraph_eclass_count",
    "egraph_enode_count",
    "rule_mode_is_positive_real",
)


@dataclass(frozen=True, slots=True)
class CandidateGenerationConfig:
    """Configuration for bounded e-graph candidate dataset generation."""

    seed: int = 0
    count: int = 10_000
    symbol_names: tuple[str, ...] = ("x", "y")
    input_jsonl_path: Path = Path("outputs/v1/dag_compression_inputs.jsonl")
    goal3_metrics_csv_path: Path = Path("outputs/v1/dag_compression_metrics.csv")
    run_modes: tuple[RuleMode, ...] = ("safe", "positive_real_formal")
    max_iterations: int = 4
    max_enodes: int = 5_000
    max_eclasses: int = 5_000
    saturation_timeout_seconds: float = 0.5
    row_timeout_seconds: float = 2.0
    beam_size: int = 12
    max_candidate_depth: int = 7
    max_candidates_evaluated: int = 12

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be positive")
        if not self.run_modes:
            raise ValueError("run_modes must not be empty")
        unknown_modes = set(self.run_modes) - {"safe", "positive_real_formal"}
        if unknown_modes:
            raise ValueError(f"unsupported rule modes: {sorted(unknown_modes)}")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.max_enodes <= 0:
            raise ValueError("max_enodes must be positive")
        if self.max_eclasses <= 0:
            raise ValueError("max_eclasses must be positive")
        if self.saturation_timeout_seconds <= 0:
            raise ValueError("saturation_timeout_seconds must be positive")
        if self.row_timeout_seconds <= 0:
            raise ValueError("row_timeout_seconds must be positive")
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if self.max_candidate_depth < 0:
            raise ValueError("max_candidate_depth must be non-negative")
        if self.max_candidates_evaluated <= 0:
            raise ValueError("max_candidates_evaluated must be positive")


class EgraphCandidateRecord(BaseModel):
    """One labeled root-candidate expression from a saturated e-graph."""

    expression_id: int
    candidate_id: str
    candidate_rank: int = Field(ge=1)
    original_expression: str
    original_srepr: str
    candidate_expression: str
    candidate_srepr: str
    candidate_ir_features: dict[str, float]
    true_official_eml_dag_nodes: int | None
    true_official_eml_tree_nodes: int | None
    true_ast_tree_nodes: int | None
    true_ast_dag_nodes: int | None
    source_ast_node_count: int
    source_ast_dag_node_count: int
    original_eml_dag_nodes: int
    compression_gain_vs_goal3_dag: float | None
    rule_mode: RuleMode
    subset_label: str
    split: SplitName
    saturation_status: str
    eclass_count: int
    enode_count: int
    exact_label_runtime_seconds: float
    baseline_estimated_eml_cost: int
    baseline_ast_node_cost: int
    same_root_eclass: bool
    semantic_validation_status: str | None
    structural_purity_valid: bool
    official_eml_compiled: bool
    error: str | None = None

    @property
    def group_key(self) -> tuple[int, RuleMode]:
        """Return the within-root ranking group."""
        return (self.expression_id, self.rule_mode)


def load_or_build_candidate_records(
    config: CandidateGenerationConfig,
    *,
    split_config: SplitConfig,
    output_jsonl_path: Path,
    reuse_existing: bool,
) -> tuple[EgraphCandidateRecord, ...]:
    """Load an existing candidate dataset or regenerate it from v1 expressions."""
    if reuse_existing and output_jsonl_path.exists():
        records = load_candidate_records_jsonl(output_jsonl_path)
        if _dataset_matches_config(records, config):
            return records
    rows = load_generated_expressions(config.input_jsonl_path)[: config.count]
    if len(rows) != config.count:
        raise ValueError(f"expected {config.count} input rows, found {len(rows)}")
    baselines = load_goal3_baselines(config.goal3_metrics_csv_path)
    records = build_candidate_records(rows, baselines, config=config, split_config=split_config)
    write_candidate_records_jsonl(records, output_jsonl_path)
    return records


def build_candidate_records(
    input_rows: Sequence[GeneratedExpressionInput],
    baselines: dict[int, Goal3BaselineRow],
    *,
    config: CandidateGenerationConfig,
    split_config: SplitConfig,
) -> tuple[EgraphCandidateRecord, ...]:
    """Build official-cost-labeled candidate records for input rows."""
    symbol_locals = build_symbol_locals(config.symbol_names)
    records: list[EgraphCandidateRecord] = []
    for position, input_row in enumerate(input_rows, start=1):
        if input_row.index is None:
            raise ValueError("input row index must not be None")
        baseline = baselines[input_row.index]
        sympy_expr, source_serialization = parse_generated_expression(
            input_row,
            symbol_locals=symbol_locals,
        )
        if source_serialization != "srepr":
            raise ValueError("Goal 5.4 requires srepr source serialization")
        ir_expr = from_sympy_without_simplification(sympy_expr)
        for rule_mode in config.run_modes:
            records.extend(
                build_candidate_records_for_ir(
                    ir_expr,
                    input_row=input_row,
                    baseline=baseline,
                    rule_mode=rule_mode,
                    config=config,
                    split=assign_split(input_row.index, split_config),
                )
            )
        if position % 500 == 0:
            print(
                f"Generated neural e-graph candidate labels: {position}/{len(input_rows)}",
                flush=True,
            )
    return tuple(records)


def build_candidate_records_for_ir(
    ir_expr: Expr,
    *,
    input_row: GeneratedExpressionInput,
    baseline: Goal3BaselineRow,
    rule_mode: RuleMode,
    config: CandidateGenerationConfig,
    split: SplitName,
) -> tuple[EgraphCandidateRecord, ...]:
    """Build candidate records for a single already-parsed IR expression."""
    if input_row.index is None:
        raise ValueError("input row index must not be None")
    egraph = EGraph()
    root_id = egraph.add_expr(ir_expr)
    candidate_generation_error = None
    try:
        with row_timeout(config.row_timeout_seconds):
            saturation_result = saturate(
                egraph,
                rules_for_mode(rule_mode),
                limits=SaturationLimits(
                    max_iterations=config.max_iterations,
                    max_enodes=config.max_enodes,
                    max_eclasses=config.max_eclasses,
                    timeout_seconds=config.saturation_timeout_seconds,
                ),
            )
            extraction_config = ExtractionConfig(
                extractor_mode="exact_eml_dag_beam_cost",
                beam_size=config.beam_size,
                max_candidate_depth=config.max_candidate_depth,
                max_candidates_evaluated=config.max_candidates_evaluated,
                timeout_seconds=config.saturation_timeout_seconds,
                allow_positive_real_rules=rule_mode == "positive_real_formal",
                rule_mode=rule_mode,
            )
            candidate_exprs = [
                candidate.expression
                for candidate in enumerate_candidates(
                    egraph,
                    root_id,
                    beam_size=config.beam_size,
                    max_depth=config.max_candidate_depth,
                    mode="exact_eml_dag_beam_cost",
                    config=extraction_config,
                )[: config.max_candidates_evaluated]
            ]
            saturation_status = saturation_result.status
            eclass_count = saturation_result.eclass_count
            enode_count = saturation_result.enode_count
    except RowTimeoutError as exc:
        candidate_generation_error = f"candidate_generation_timeout: {exc}"
        candidate_exprs = [ir_expr]
        saturation_status = "timeout"
        eclass_count = egraph.eclass_count
        enode_count = egraph.enode_count
    except Exception as exc:
        candidate_generation_error = f"candidate_enumeration_failed: {type(exc).__name__}: {exc}"
        candidate_exprs = [ir_expr]
        saturation_status = "failed"
        eclass_count = egraph.eclass_count
        enode_count = egraph.enode_count
    if not candidate_exprs:
        candidate_generation_error = "candidate_enumeration_returned_no_candidates"
        candidate_exprs = [ir_expr]
    subset_label = subset_label_for_metadata(input_row.metadata)
    return tuple(
        _candidate_record(
            candidate_expr,
            candidate_rank=rank,
            input_row=input_row,
            baseline=baseline,
            original_ir_expr=ir_expr,
            egraph=egraph,
            root_id=root_id,
            rule_mode=rule_mode,
            split=split,
            subset_label=subset_label,
            saturation_status=saturation_status,
            eclass_count=eclass_count,
            enode_count=enode_count,
            candidate_generation_error=candidate_generation_error,
        )
        for rank, candidate_expr in enumerate(candidate_exprs, start=1)
    )


def candidate_feature_names(records: Iterable[EgraphCandidateRecord]) -> tuple[str, ...]:
    """Return a stable feature-name tuple for a candidate collection."""
    names: set[str] = set(DEFAULT_FEATURE_NAMES)
    for record in records:
        names.update(record.candidate_ir_features)
    return tuple(sorted(names))


def write_candidate_records_jsonl(
    records: Sequence[EgraphCandidateRecord],
    path: Path,
) -> None:
    """Write candidate records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")


def load_candidate_records_jsonl(path: Path) -> tuple[EgraphCandidateRecord, ...]:
    """Load candidate records from JSONL."""
    records: list[EgraphCandidateRecord] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(EgraphCandidateRecord.model_validate_json(stripped))
            except Exception as exc:
                raise ValueError(f"invalid candidate row at {path}:{line_number}") from exc
    return tuple(records)


def from_sympy_without_simplification(sympy_expr: sp.Expr) -> Expr:
    """Convert SymPy into the e-graph IR through the existing structural converter."""
    from geml.egraph.ir import from_sympy

    return from_sympy(sympy_expr)


def extract_candidate_features(
    expr: Expr,
    *,
    candidate_rank: int,
    baseline: Goal3BaselineRow,
    rule_mode: RuleMode,
    eclass_count: int,
    enode_count: int,
) -> dict[str, float]:
    """Extract graph-neutral numeric features from an IR candidate."""
    counts: dict[str, int] = {
        "variable": 0,
        "constant": 0,
        "add": 0,
        "mul": 0,
        "sub": 0,
        "div": 0,
        "pow": 0,
        "neg": 0,
        "exp": 0,
        "log": 0,
    }
    constants: list[float] = []
    _collect_expr_stats(expr, counts=counts, constants=constants)
    feature_values = {
        "candidate_ast_node_cost": float(ast_node_cost(expr)),
        "candidate_estimated_eml_cost": float(estimated_eml_cost(expr)),
        "candidate_depth": float(_expr_depth(expr)),
        "candidate_leaf_count": float(counts["variable"] + counts["constant"]),
        "candidate_variable_count": float(counts["variable"]),
        "candidate_constant_count": float(counts["constant"]),
        "candidate_add_count": float(counts["add"]),
        "candidate_mul_count": float(counts["mul"]),
        "candidate_sub_count": float(counts["sub"]),
        "candidate_div_count": float(counts["div"]),
        "candidate_pow_count": float(counts["pow"]),
        "candidate_neg_count": float(counts["neg"]),
        "candidate_exp_count": float(counts["exp"]),
        "candidate_log_count": float(counts["log"]),
        "candidate_max_constant_abs": max(constants, default=0.0),
        "candidate_sum_constant_abs": sum(constants),
        "source_ast_node_count": float(baseline.ast_tree_node_count),
        "source_ast_dag_node_count": float(baseline.ast_dag_node_count),
        "source_eml_dag_node_count": float(baseline.eml_dag_node_count),
        "candidate_rank": float(candidate_rank),
        "egraph_eclass_count": float(eclass_count),
        "egraph_enode_count": float(enode_count),
        "rule_mode_is_positive_real": float(rule_mode == "positive_real_formal"),
    }
    return {name: feature_values.get(name, 0.0) for name in DEFAULT_FEATURE_NAMES}


def valid_labeled_records(
    records: Iterable[EgraphCandidateRecord],
) -> tuple[EgraphCandidateRecord, ...]:
    """Return records with official EML labels available."""
    return tuple(
        record
        for record in records
        if record.official_eml_compiled
        and record.true_official_eml_dag_nodes is not None
        and record.true_official_eml_tree_nodes is not None
    )


def group_candidate_records(
    records: Iterable[EgraphCandidateRecord],
) -> dict[tuple[int, RuleMode], list[EgraphCandidateRecord]]:
    """Group candidate rows by expression id and rule mode."""
    groups: dict[tuple[int, RuleMode], list[EgraphCandidateRecord]] = {}
    for record in records:
        groups.setdefault(record.group_key, []).append(record)
    return groups


def summarize_candidate_dataset(records: Sequence[EgraphCandidateRecord]) -> dict[str, object]:
    """Return lightweight dataset integrity statistics."""
    groups = group_candidate_records(records)
    valid_records = valid_labeled_records(records)
    return {
        "candidate_count": len(records),
        "valid_labeled_candidate_count": len(valid_records),
        "group_count": len(groups),
        "official_label_failure_count": len(records) - len(valid_records),
        "rule_mode_counts": {
            rule_mode: sum(record.rule_mode == rule_mode for record in records)
            for rule_mode in ("safe", "positive_real_formal")
        },
        "split_counts": {
            split: len({record.group_key for record in records if record.split == split})
            for split in ("train", "validation", "test")
        },
    }


def _candidate_record(
    candidate_expr: Expr,
    *,
    candidate_rank: int,
    input_row: GeneratedExpressionInput,
    baseline: Goal3BaselineRow,
    original_ir_expr: Expr,
    egraph: EGraph,
    root_id: int,
    rule_mode: RuleMode,
    split: SplitName,
    subset_label: str,
    saturation_status: str,
    eclass_count: int,
    enode_count: int,
    candidate_generation_error: str | None,
) -> EgraphCandidateRecord:
    if input_row.index is None:
        raise ValueError("input row index must not be None")
    started_at = time.perf_counter()
    exact_cost = None
    exact_error = None
    try:
        exact_cost = exact_eml_dag_cost(candidate_expr)
    except Exception as exc:
        exact_error = f"{type(exc).__name__}: {exc}"
    exact_runtime = time.perf_counter() - started_at

    candidate_id = f"{input_row.index}:{rule_mode}:{candidate_rank}"
    candidate_srepr = sp.srepr(to_sympy(candidate_expr))
    same_eclass = validate_same_eclass(egraph, root_id, egraph.add_expr(candidate_expr))
    semantic_validation = positive_real_numeric_validation(original_ir_expr, candidate_expr)
    official_eml_compiled = exact_cost is not None
    true_dag_nodes = exact_cost.eml_dag_nodes if exact_cost is not None else None
    true_tree_nodes = exact_cost.eml_tree_nodes if exact_cost is not None else None
    features = extract_candidate_features(
        candidate_expr,
        candidate_rank=candidate_rank,
        baseline=baseline,
        rule_mode=rule_mode,
        eclass_count=eclass_count,
        enode_count=enode_count,
    )
    return EgraphCandidateRecord(
        expression_id=input_row.index,
        candidate_id=candidate_id,
        candidate_rank=candidate_rank,
        original_expression=input_row.expression,
        original_srepr=input_row.srepr or baseline.srepr,
        candidate_expression=display(candidate_expr),
        candidate_srepr=candidate_srepr,
        candidate_ir_features=features,
        true_official_eml_dag_nodes=true_dag_nodes,
        true_official_eml_tree_nodes=true_tree_nodes,
        true_ast_tree_nodes=exact_cost.ast_tree_nodes if exact_cost is not None else None,
        true_ast_dag_nodes=exact_cost.ast_dag_nodes if exact_cost is not None else None,
        source_ast_node_count=baseline.ast_tree_node_count,
        source_ast_dag_node_count=baseline.ast_dag_node_count,
        original_eml_dag_nodes=baseline.eml_dag_node_count,
        compression_gain_vs_goal3_dag=_safe_divide(baseline.eml_dag_node_count, true_dag_nodes),
        rule_mode=rule_mode,
        subset_label=subset_label,
        split=split,
        saturation_status=saturation_status,
        eclass_count=eclass_count,
        enode_count=enode_count,
        exact_label_runtime_seconds=exact_runtime,
        baseline_estimated_eml_cost=estimated_eml_cost(candidate_expr),
        baseline_ast_node_cost=ast_node_cost(candidate_expr),
        same_root_eclass=same_eclass.validation_status == "valid",
        semantic_validation_status=semantic_validation.validation_status,
        structural_purity_valid=bool(exact_cost.integrity_valid)
        if exact_cost is not None
        else False,
        official_eml_compiled=official_eml_compiled,
        error=_join_errors(candidate_generation_error, exact_error),
    )


def _collect_expr_stats(expr: Expr, *, counts: dict[str, int], constants: list[float]) -> None:
    if isinstance(expr, Var):
        counts["variable"] += 1
        return
    if isinstance(expr, Const):
        counts["constant"] += 1
        constants.append(abs(float(expr.value)))
        return
    if isinstance(expr, Add):
        counts["add"] += 1
        _collect_expr_stats(expr.left, counts=counts, constants=constants)
        _collect_expr_stats(expr.right, counts=counts, constants=constants)
        return
    if isinstance(expr, Mul):
        counts["mul"] += 1
        _collect_expr_stats(expr.left, counts=counts, constants=constants)
        _collect_expr_stats(expr.right, counts=counts, constants=constants)
        return
    if isinstance(expr, Sub):
        counts["sub"] += 1
        _collect_expr_stats(expr.left, counts=counts, constants=constants)
        _collect_expr_stats(expr.right, counts=counts, constants=constants)
        return
    if isinstance(expr, Div):
        counts["div"] += 1
        _collect_expr_stats(expr.left, counts=counts, constants=constants)
        _collect_expr_stats(expr.right, counts=counts, constants=constants)
        return
    if isinstance(expr, Pow):
        counts["pow"] += 1
        _collect_expr_stats(expr.base, counts=counts, constants=constants)
        _collect_expr_stats(expr.exponent, counts=counts, constants=constants)
        return
    if isinstance(expr, Neg):
        counts["neg"] += 1
        _collect_expr_stats(expr.value, counts=counts, constants=constants)
        return
    if isinstance(expr, Exp):
        counts["exp"] += 1
        _collect_expr_stats(expr.value, counts=counts, constants=constants)
        return
    if isinstance(expr, Log):
        counts["log"] += 1
        _collect_expr_stats(expr.value, counts=counts, constants=constants)
        return
    raise TypeError(f"unsupported expression type: {type(expr).__name__}")


def _expr_depth(expr: Expr) -> int:
    if isinstance(expr, Var | Const):
        return 1
    if isinstance(expr, Neg | Exp | Log):
        return 1 + _expr_depth(expr.value)
    if isinstance(expr, Add | Mul | Sub | Div):
        return 1 + max(_expr_depth(expr.left), _expr_depth(expr.right))
    if isinstance(expr, Pow):
        return 1 + max(_expr_depth(expr.base), _expr_depth(expr.exponent))
    raise TypeError(f"unsupported expression type: {type(expr).__name__}")


def _dataset_matches_config(
    records: Sequence[EgraphCandidateRecord],
    config: CandidateGenerationConfig,
) -> bool:
    if not records:
        return False
    expression_ids = {record.expression_id for record in records}
    if min(expression_ids, default=-1) != 0 or max(expression_ids, default=-1) != config.count - 1:
        return False
    return set(record.rule_mode for record in records) == set(config.run_modes)


def _join_errors(*errors: str | None) -> str | None:
    joined = "; ".join(error for error in errors if error)
    return joined or None


def _safe_divide(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return float(numerator) / float(denominator)
