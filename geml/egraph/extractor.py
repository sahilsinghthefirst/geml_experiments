"""Extraction utilities for Goal 4 e-graphs."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Literal

from geml.egraph.costs import (
    ExactEmlDagCost,
    ExtractorMode,
    ast_node_cost,
    estimated_eml_cost,
    exact_eml_dag_cost,
)
from geml.egraph.egraph import EGraph, ENode
from geml.egraph.ir import Add, Const, Div, Exp, Expr, Log, Mul, Neg, Pow, Sub, Var, display
from geml.egraph.rule_sets import DEFAULT_RULE_MODE, RuleMode, rule_mode_summary
from geml.egraph.validation import (
    PositiveRealNumericValidationResult,
    SameEClassValidationResult,
    positive_real_numeric_validation,
    validate_same_eclass,
)

type BaselineExtractionStatus = Literal["completed", "failed"]
type ExtractionStatus = Literal[
    "completed",
    "failed",
    "timeout",
    "no_candidates",
    "all_candidates_failed",
]


@dataclass(frozen=True, slots=True)
class ExtractedExpression:
    """A baseline extracted expression."""

    status: BaselineExtractionStatus
    root_eclass_id: int
    expression: Expr | None
    cost: int | None
    cost_model: str


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Configuration for Goal 4.4 extraction."""

    extractor_mode: ExtractorMode = "ast_node_cost"
    beam_size: int = 16
    max_candidate_depth: int = 8
    max_candidates_evaluated: int = 64
    timeout_seconds: float = 10.0
    allow_positive_real_rules: bool = False
    rule_mode: RuleMode = DEFAULT_RULE_MODE

    def __post_init__(self) -> None:
        if self.extractor_mode not in {
            "ast_node_cost",
            "estimated_eml_cost",
            "exact_eml_dag_beam_cost",
        }:
            raise ValueError(f"unknown extractor mode: {self.extractor_mode!r}")
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if self.max_candidate_depth < 0:
            raise ValueError("max_candidate_depth must be non-negative")
        if self.max_candidates_evaluated <= 0:
            raise ValueError("max_candidates_evaluated must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.rule_mode == "positive_real_formal" and not self.allow_positive_real_rules:
            raise ValueError(
                "positive_real_formal extraction requires allow_positive_real_rules=True"
            )


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Output row for a Goal 4.4 extraction run."""

    original_expression: str
    extracted_expression: str | None
    rule_mode: RuleMode
    extractor_mode: ExtractorMode
    beam_size: int
    max_candidate_depth: int
    max_candidates_evaluated: int
    timeout_seconds: float
    allow_positive_real_rules: bool
    candidate_count: int
    selected_candidate_rank: int | None
    extracted_ast_tree_nodes: int | None
    extracted_ast_dag_nodes: int | None
    extracted_eml_tree_nodes: int | None
    extracted_eml_dag_nodes: int | None
    extraction_status: ExtractionStatus
    extraction_timeout: bool
    tie_break_info: dict[str, object]
    validation_status: str | None
    same_root_eclass: bool | None
    positive_real_max_abs_error: float | None
    integrity_valid: bool | None
    integrity_errors: tuple[str, ...]
    assumptions: str | None
    branch_sensitive_rules_used: bool
    branch_sensitive_rule_count: int
    branch_sensitive_rule_names: tuple[str, ...]
    expression: Expr | None = None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Candidate:
    expression: Expr
    priority_cost: int


class _ExtractionTimeout(Exception):
    pass


def extract_min_ast_size(egraph: EGraph, root_eclass_id: int) -> ExtractedExpression:
    """Extract the smallest acyclic IR tree by ordinary AST node count."""
    root_id = egraph.find(root_eclass_id)
    best = _best_expr(egraph, root_id)
    if best is None:
        return ExtractedExpression(
            status="failed",
            root_eclass_id=root_id,
            expression=None,
            cost=None,
            cost_model="ast_node_count",
        )
    cost, expr = best
    return ExtractedExpression(
        status="completed",
        root_eclass_id=root_id,
        expression=expr,
        cost=cost,
        cost_model="ast_node_count",
    )


def extract_expression(
    egraph: EGraph,
    root_eclass_id: int,
    *,
    original_expression: Expr | None = None,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    """Extract one expression from a root e-class using a configured objective."""
    extraction_config = config or ExtractionConfig()
    root_id = egraph.find(root_eclass_id)
    original = original_expression or _fallback_original_expression(egraph, root_id)
    original_string = display(original) if original is not None else ""
    started_at = time.monotonic()

    try:
        _check_timeout(started_at, extraction_config)
        candidates = enumerate_candidates(
            egraph,
            root_id,
            beam_size=extraction_config.beam_size,
            max_depth=extraction_config.max_candidate_depth,
            mode=extraction_config.extractor_mode,
            started_at=started_at,
            config=extraction_config,
        )
        if not candidates:
            return _empty_result(
                original_string=original_string,
                config=extraction_config,
                status="no_candidates",
            )

        if extraction_config.extractor_mode == "exact_eml_dag_beam_cost":
            return _extract_exact(
                egraph=egraph,
                root_id=root_id,
                original=original,
                original_string=original_string,
                candidates=candidates,
                started_at=started_at,
                config=extraction_config,
            )

        selected = _select_estimated_candidate(candidates, extraction_config.extractor_mode)
        exact_cost = exact_eml_dag_cost(selected.expression)
        validation = _validate_extraction(egraph, root_id, original, selected.expression)
        return _result_from_exact_cost(
            original_string=original_string,
            exact_cost=exact_cost,
            config=extraction_config,
            candidate_count=len(candidates),
            selected_candidate_rank=_candidate_rank(candidates, selected),
            status="completed",
            validation=validation,
            same_eclass=validate_same_eclass(
                egraph,
                root_id,
                egraph.add_expr(selected.expression),
            ),
            tie_break_info=_baseline_tie_break_info(selected, exact_cost),
        )
    except _ExtractionTimeout:
        return _empty_result(
            original_string=original_string,
            config=extraction_config,
            status="timeout",
            extraction_timeout=True,
        )
    except Exception as exc:
        return _empty_result(
            original_string=original_string,
            config=extraction_config,
            status="failed",
            errors=(f"{type(exc).__name__}: {exc}",),
        )


def extract(
    egraph: EGraph,
    root_eclass_id: int,
    *,
    original_expression: Expr | None = None,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    """Compatibility alias for mode-aware extraction."""
    return extract_expression(
        egraph,
        root_eclass_id,
        original_expression=original_expression,
        config=config,
    )


def enumerate_candidates(
    egraph: EGraph,
    root_eclass_id: int,
    *,
    beam_size: int,
    max_depth: int,
    mode: ExtractorMode,
    started_at: float | None = None,
    config: ExtractionConfig | None = None,
) -> tuple[_Candidate, ...]:
    """Enumerate bounded top-K candidate expressions from an e-class."""
    extraction_config = config or ExtractionConfig(extractor_mode=mode)
    started = time.monotonic() if started_at is None else started_at
    root_id = egraph.find(root_eclass_id)
    eclass_ids = egraph.eclass_ids()
    candidates_by_depth: dict[tuple[int, int], tuple[_Candidate, ...]] = {}

    for depth in range(max_depth + 1):
        for eclass_id in eclass_ids:
            _check_timeout(started, extraction_config)
            candidates: list[_Candidate] = []
            for node in egraph.get_eclass_nodes(eclass_id):
                if not node.children:
                    candidates.append(_make_candidate(_expr_from_node(node, []), mode))
                    continue
                if depth == 0:
                    continue

                child_lists: list[tuple[_Candidate, ...]] = []
                for child_id in node.children:
                    child_candidates = candidates_by_depth.get(
                        (egraph.find(child_id), depth - 1),
                        (),
                    )
                    if not child_candidates:
                        break
                    child_lists.append(child_candidates)
                else:
                    for child_tuple in product(*child_lists):
                        _check_timeout(started, extraction_config)
                        child_exprs = [candidate.expression for candidate in child_tuple]
                        expr = _expr_from_node(node, child_exprs)
                        candidates.append(_make_candidate(expr, mode))
                        if len(candidates) > beam_size * 8:
                            candidates = list(_top_unique_candidates(candidates, beam_size))

            candidates_by_depth[(eclass_id, depth)] = _top_unique_candidates(
                candidates,
                beam_size,
            )

    return candidates_by_depth.get((root_id, max_depth), ())


def _extract_exact(
    *,
    egraph: EGraph,
    root_id: int,
    original: Expr | None,
    original_string: str,
    candidates: tuple[_Candidate, ...],
    started_at: float,
    config: ExtractionConfig,
) -> ExtractionResult:
    successful_costs: list[ExactEmlDagCost] = []
    errors: list[str] = []
    evaluated = 0

    for candidate in candidates[: config.max_candidates_evaluated]:
        _check_timeout(started_at, config)
        evaluated += 1
        try:
            successful_costs.append(exact_eml_dag_cost(candidate.expression))
        except Exception as exc:
            errors.append(f"{display(candidate.expression)}: {type(exc).__name__}: {exc}")

    if not successful_costs:
        return _empty_result(
            original_string=original_string,
            config=config,
            status="all_candidates_failed",
            candidate_count=evaluated,
            errors=tuple(errors),
        )

    ranked_costs = sorted(successful_costs, key=lambda cost: cost.tie_break_key)
    selected = ranked_costs[0]
    validation = _validate_extraction(egraph, root_id, original, selected.expression)
    same_eclass = validate_same_eclass(egraph, root_id, egraph.add_expr(selected.expression))
    return _result_from_exact_cost(
        original_string=original_string,
        exact_cost=selected,
        config=config,
        candidate_count=evaluated,
        selected_candidate_rank=1,
        status="completed",
        validation=validation,
        same_eclass=same_eclass,
        tie_break_info={
            "tie_break_order": (
                "extracted_eml_dag_nodes",
                "extracted_eml_tree_nodes",
                "extracted_ast_dag_nodes",
                "extracted_ast_tree_nodes",
                "stable_expression_string",
            ),
            "tie_break_key": selected.tie_break_key,
            "successful_candidate_count": len(successful_costs),
        },
        errors=tuple(errors),
    )


def _select_estimated_candidate(
    candidates: tuple[_Candidate, ...],
    mode: ExtractorMode,
) -> _Candidate:
    if mode == "ast_node_cost":
        return min(
            candidates,
            key=lambda candidate: (
                ast_node_cost(candidate.expression),
                display(candidate.expression),
            ),
        )
    if mode == "estimated_eml_cost":
        return min(
            candidates,
            key=lambda candidate: (
                estimated_eml_cost(candidate.expression),
                ast_node_cost(candidate.expression),
                display(candidate.expression),
            ),
        )
    raise ValueError(f"unsupported non-exact extractor mode: {mode!r}")


def _validate_extraction(
    egraph: EGraph,
    root_id: int,
    original: Expr | None,
    extracted: Expr,
) -> PositiveRealNumericValidationResult | None:
    if original is None:
        return None
    extracted_id = egraph.add_expr(extracted)
    same_eclass = validate_same_eclass(egraph, root_id, extracted_id)
    if same_eclass.validation_status != "valid":
        return PositiveRealNumericValidationResult(
            validation_status="invalid",
            max_abs_error=None,
            sample_count=0,
            detail=same_eclass.detail,
        )
    return positive_real_numeric_validation(original, extracted)


def _result_from_exact_cost(
    *,
    original_string: str,
    exact_cost: ExactEmlDagCost,
    config: ExtractionConfig,
    candidate_count: int,
    selected_candidate_rank: int,
    status: ExtractionStatus,
    validation: PositiveRealNumericValidationResult | None,
    same_eclass: SameEClassValidationResult,
    tie_break_info: dict[str, object],
    errors: tuple[str, ...] = (),
) -> ExtractionResult:
    mode_summary = rule_mode_summary(config.rule_mode)
    validation_status = validation.validation_status if validation is not None else None
    if same_eclass.validation_status != "valid":
        validation_status = same_eclass.validation_status
    return ExtractionResult(
        original_expression=original_string,
        extracted_expression=exact_cost.expression_string,
        rule_mode=config.rule_mode,
        extractor_mode=config.extractor_mode,
        beam_size=config.beam_size,
        max_candidate_depth=config.max_candidate_depth,
        max_candidates_evaluated=config.max_candidates_evaluated,
        timeout_seconds=config.timeout_seconds,
        allow_positive_real_rules=config.allow_positive_real_rules,
        candidate_count=candidate_count,
        selected_candidate_rank=selected_candidate_rank,
        extracted_ast_tree_nodes=exact_cost.ast_tree_nodes,
        extracted_ast_dag_nodes=exact_cost.ast_dag_nodes,
        extracted_eml_tree_nodes=exact_cost.eml_tree_nodes,
        extracted_eml_dag_nodes=exact_cost.eml_dag_nodes,
        extraction_status=status,
        extraction_timeout=False,
        tie_break_info=tie_break_info,
        validation_status=validation_status,
        same_root_eclass=same_eclass.validation_status == "valid",
        positive_real_max_abs_error=validation.max_abs_error if validation is not None else None,
        integrity_valid=exact_cost.integrity_valid,
        integrity_errors=exact_cost.integrity_errors,
        assumptions=mode_summary.assumptions,
        branch_sensitive_rules_used=mode_summary.branch_sensitive_rules_used,
        branch_sensitive_rule_count=mode_summary.branch_sensitive_rule_count,
        branch_sensitive_rule_names=mode_summary.branch_sensitive_rule_names,
        expression=exact_cost.expression,
        errors=errors,
    )


def _empty_result(
    *,
    original_string: str,
    config: ExtractionConfig,
    status: ExtractionStatus,
    extraction_timeout: bool = False,
    candidate_count: int = 0,
    errors: tuple[str, ...] = (),
) -> ExtractionResult:
    mode_summary = rule_mode_summary(config.rule_mode)
    return ExtractionResult(
        original_expression=original_string,
        extracted_expression=None,
        rule_mode=config.rule_mode,
        extractor_mode=config.extractor_mode,
        beam_size=config.beam_size,
        max_candidate_depth=config.max_candidate_depth,
        max_candidates_evaluated=config.max_candidates_evaluated,
        timeout_seconds=config.timeout_seconds,
        allow_positive_real_rules=config.allow_positive_real_rules,
        candidate_count=candidate_count,
        selected_candidate_rank=None,
        extracted_ast_tree_nodes=None,
        extracted_ast_dag_nodes=None,
        extracted_eml_tree_nodes=None,
        extracted_eml_dag_nodes=None,
        extraction_status=status,
        extraction_timeout=extraction_timeout,
        tie_break_info={},
        validation_status=None,
        same_root_eclass=None,
        positive_real_max_abs_error=None,
        integrity_valid=None,
        integrity_errors=(),
        assumptions=mode_summary.assumptions,
        branch_sensitive_rules_used=mode_summary.branch_sensitive_rules_used,
        branch_sensitive_rule_count=mode_summary.branch_sensitive_rule_count,
        branch_sensitive_rule_names=mode_summary.branch_sensitive_rule_names,
        expression=None,
        errors=errors,
    )


def _make_candidate(expr: Expr, mode: ExtractorMode) -> _Candidate:
    if mode == "ast_node_cost":
        cost = ast_node_cost(expr)
    else:
        cost = estimated_eml_cost(expr)
    return _Candidate(expression=expr, priority_cost=cost)


def _top_unique_candidates(candidates: list[_Candidate], beam_size: int) -> tuple[_Candidate, ...]:
    unique: dict[str, _Candidate] = {}
    for candidate in candidates:
        key = display(candidate.expression)
        existing = unique.get(key)
        if existing is None or candidate.priority_cost < existing.priority_cost:
            unique[key] = candidate
    ranked = sorted(
        unique.values(),
        key=lambda candidate: (
            candidate.priority_cost,
            ast_node_cost(candidate.expression),
            display(candidate.expression),
        ),
    )
    return tuple(ranked[:beam_size])


def _candidate_rank(candidates: tuple[_Candidate, ...], selected: _Candidate) -> int:
    for index, candidate in enumerate(candidates, start=1):
        if display(candidate.expression) == display(selected.expression):
            return index
    return 1


def _baseline_tie_break_info(
    selected: _Candidate,
    exact_cost: ExactEmlDagCost,
) -> dict[str, object]:
    return {
        "baseline_priority_cost": selected.priority_cost,
        "exact_eml_dag_nodes_diagnostic": exact_cost.eml_dag_nodes,
        "warning": "AST and estimated extractors are baselines, not EML-optimal objectives.",
    }


def _fallback_original_expression(egraph: EGraph, root_id: int) -> Expr | None:
    baseline = extract_min_ast_size(egraph, root_id)
    return baseline.expression


def _check_timeout(started_at: float, config: ExtractionConfig) -> None:
    if time.monotonic() - started_at >= config.timeout_seconds:
        raise _ExtractionTimeout


def _best_expr(
    egraph: EGraph,
    eclass_id: int,
) -> tuple[int, Expr] | None:
    """Return the least AST-size expression by monotone fixpoint extraction."""
    root_id = egraph.find(eclass_id)
    best_by_eclass: dict[int, tuple[int, Expr]] = {}
    eclass_ids = egraph.eclass_ids()

    while True:
        changed = False
        for candidate_eclass_id in eclass_ids:
            for node in egraph.get_eclass_nodes(candidate_eclass_id):
                candidate = _node_candidate(egraph, node, best_by_eclass)
                if candidate is None:
                    continue
                current = best_by_eclass.get(candidate_eclass_id)
                if current is None or _best_tuple_key(candidate) < _best_tuple_key(current):
                    best_by_eclass[candidate_eclass_id] = candidate
                    changed = True
        if not changed:
            return best_by_eclass.get(root_id)


def _node_candidate(
    egraph: EGraph,
    node: ENode,
    best_by_eclass: Mapping[int, tuple[int, Expr]],
) -> tuple[int, Expr] | None:
    child_results: list[tuple[int, Expr]] = []
    for child_id in node.children:
        child_result = best_by_eclass.get(egraph.find(child_id))
        if child_result is None:
            return None
        child_results.append(child_result)

    child_cost = sum(cost for cost, _ in child_results)
    child_exprs = [expr for _, expr in child_results]
    expr = _expr_from_node(node, child_exprs)
    return 1 + child_cost, expr


def _best_tuple_key(item: tuple[int, Expr]) -> tuple[int, str]:
    cost, expr = item
    return cost, display(expr)


def _expr_from_node(node: ENode, children: list[Expr]) -> Expr:
    if node.op == "var" and isinstance(node.value, str):
        return Var(node.value)
    if node.op == "const" and node.value is not None and not isinstance(node.value, str):
        return Const(node.value)
    if node.op == "add" and len(children) == 2:
        return Add(children[0], children[1])
    if node.op == "mul" and len(children) == 2:
        return Mul(children[0], children[1])
    if node.op == "neg" and len(children) == 1:
        return Neg(children[0])
    if node.op == "sub" and len(children) == 2:
        return Sub(children[0], children[1])
    if node.op == "div" and len(children) == 2:
        return Div(children[0], children[1])
    if node.op == "pow" and len(children) == 2:
        return Pow(children[0], children[1])
    if node.op == "exp" and len(children) == 1:
        return Exp(children[0])
    if node.op == "log" and len(children) == 1:
        return Log(children[0])
    raise ValueError(f"invalid e-node for extraction: {node}")
