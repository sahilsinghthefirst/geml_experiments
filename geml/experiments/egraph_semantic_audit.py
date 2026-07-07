"""Goal 4.9 semantic, purity, and provenance audit for v1 e-graph compression."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import sympy as sp

from geml.egraph.costs import ExactEmlDagCost, exact_eml_dag_cost, validate_pure_eml_integrity
from geml.egraph.egraph import EGraph
from geml.egraph.extractor import ExtractionConfig, enumerate_candidates, extract_expression
from geml.egraph.ir import Add, Const, Exp, Expr, Log, Mul, Pow, Sub, Var, display, to_sympy
from geml.egraph.rewrites import RewriteRule, SaturationLimits, apply_rewrite
from geml.egraph.rule_sets import rules_for_mode
from geml.experiments.dag_semantic_audit import audit_eml_dag_structure, evaluate_eml_dag
from geml.symbolic.dag_graph import DagGraph
from geml.symbolic.eml_nodes import EmlTree

type SemanticAuditRuleMode = Literal["safe", "positive_real_formal"]

RUN_MODES: tuple[SemanticAuditRuleMode, ...] = ("safe", "positive_real_formal")
POSITIVE_REAL_SAMPLES: tuple[dict[str, float], ...] = (
    {"x": 1.3, "y": 2.1},
    {"x": 1.7, "y": 3.4},
    {"x": 4.2, "y": 1.6},
)
REWRITE_PATH_FILES = (
    Path("geml/egraph/egraph.py"),
    Path("geml/egraph/ir.py"),
    Path("geml/egraph/patterns.py"),
    Path("geml/egraph/rewrites.py"),
    Path("geml/egraph/rule_sets.py"),
)

SEMANTIC_AUDIT_CSV_FIELDS = [
    "original_expression",
    "extracted_expression",
    "rule_mode",
    "assumptions",
    "rules_applied_by_name",
    "rules_applied_by_tier",
    "branch_sensitive_rules_applied",
    "branch_sensitive_rule_names_applied",
    "saturation_status",
    "extraction_status",
    "validation_status",
    "original_value_samples",
    "extracted_value_samples",
    "max_abs_error",
    "original_eml_dag_nodes",
    "extracted_eml_dag_nodes",
    "compression_gain",
    "structural_purity_valid",
    "purity_failure_reason",
    "eclass_count",
    "enode_count",
    "candidate_eml_dag_metrics",
    "selected_candidate_rank",
    "eml_dag_value_samples",
    "eml_dag_max_abs_error",
    "eml_dag_validation_status",
    "provenance_validation_status",
    "sympy_simplify_rewrite_path_free",
]


@dataclass(frozen=True, slots=True)
class AuditExpression:
    """One fixed expression for the Goal 4.9 audit."""

    expression: str
    ir: Expr


@dataclass(frozen=True, slots=True)
class EgraphSemanticAuditConfig:
    """Configuration for the Goal 4.9 audit."""

    output_dir: Path = Path("outputs/v1")
    json_path: Path = Path("outputs/v1/goal4_egraph_semantic_audit.json")
    csv_path: Path = Path("outputs/v1/goal4_egraph_semantic_audit.csv")
    report_path: Path = Path("docs/goal4/GOAL4_EGRAPH_SEMANTIC_AUDIT.md")
    saturation_max_iterations: int = 6
    saturation_max_enodes: int = 20_000
    saturation_max_eclasses: int = 20_000
    saturation_timeout_seconds: float = 5.0
    beam_size: int = 32
    max_candidate_depth: int = 8
    max_candidates_evaluated: int = 32
    extraction_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if any(
            "outputs/v0" in path.as_posix()
            for path in (self.output_dir, self.json_path, self.csv_path)
        ):
            raise ValueError("Goal 4.9 audit must not write primary outputs to outputs/v0")
        if self.saturation_max_iterations < 0:
            raise ValueError("saturation_max_iterations must be non-negative")
        if self.saturation_max_enodes <= 0:
            raise ValueError("saturation_max_enodes must be positive")
        if self.saturation_max_eclasses <= 0:
            raise ValueError("saturation_max_eclasses must be positive")
        if self.saturation_timeout_seconds <= 0:
            raise ValueError("saturation_timeout_seconds must be positive")
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if self.max_candidate_depth < 0:
            raise ValueError("max_candidate_depth must be non-negative")
        if self.max_candidates_evaluated <= 0:
            raise ValueError("max_candidates_evaluated must be positive")
        if self.extraction_timeout_seconds <= 0:
            raise ValueError("extraction_timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class ProvenanceSaturationResult:
    """Saturation summary with explicit rewrite provenance."""

    status: str
    iterations_completed: int
    total_applications: int
    enode_count: int
    eclass_count: int
    elapsed_seconds: float
    rules_applied_by_name: dict[str, int]
    rules_applied_by_tier: dict[str, int]
    branch_sensitive_rules_applied: bool
    branch_sensitive_rule_names_applied: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PurityAudit:
    """Combined EML tree and EML-DAG purity result."""

    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NumericAudit:
    """Positive-real numeric audit samples."""

    original_values: tuple[float, ...]
    extracted_values: tuple[float, ...]
    eml_dag_values: tuple[float, ...]
    max_abs_error: float | None
    eml_dag_max_abs_error: float | None
    validation_status: str
    eml_dag_validation_status: str


@dataclass(frozen=True, slots=True)
class EgraphSemanticAuditRow:
    """One expression/mode row from the Goal 4.9 audit."""

    original_expression: str
    extracted_expression: str | None
    rule_mode: SemanticAuditRuleMode
    assumptions: str | None
    rules_applied_by_name: dict[str, int]
    rules_applied_by_tier: dict[str, int]
    branch_sensitive_rules_applied: bool
    branch_sensitive_rule_names_applied: tuple[str, ...]
    saturation_status: str
    extraction_status: str
    validation_status: str
    original_value_samples: tuple[float, ...]
    extracted_value_samples: tuple[float, ...]
    max_abs_error: float | None
    original_eml_dag_nodes: int
    extracted_eml_dag_nodes: int | None
    compression_gain: float | None
    structural_purity_valid: bool
    purity_failure_reason: str | None
    eclass_count: int
    enode_count: int
    candidate_eml_dag_metrics: tuple[dict[str, object], ...]
    selected_candidate_rank: int | None
    eml_dag_value_samples: tuple[float, ...]
    eml_dag_max_abs_error: float | None
    eml_dag_validation_status: str
    provenance_validation_status: str
    sympy_simplify_rewrite_path_free: bool

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable row."""
        return {
            "original_expression": self.original_expression,
            "extracted_expression": self.extracted_expression,
            "rule_mode": self.rule_mode,
            "assumptions": self.assumptions,
            "rules_applied_by_name": self.rules_applied_by_name,
            "rules_applied_by_tier": self.rules_applied_by_tier,
            "branch_sensitive_rules_applied": self.branch_sensitive_rules_applied,
            "branch_sensitive_rule_names_applied": list(self.branch_sensitive_rule_names_applied),
            "saturation_status": self.saturation_status,
            "extraction_status": self.extraction_status,
            "validation_status": self.validation_status,
            "original_value_samples": list(self.original_value_samples),
            "extracted_value_samples": list(self.extracted_value_samples),
            "max_abs_error": self.max_abs_error,
            "original_eml_dag_nodes": self.original_eml_dag_nodes,
            "extracted_eml_dag_nodes": self.extracted_eml_dag_nodes,
            "compression_gain": self.compression_gain,
            "structural_purity_valid": self.structural_purity_valid,
            "purity_failure_reason": self.purity_failure_reason,
            "eclass_count": self.eclass_count,
            "enode_count": self.enode_count,
            "candidate_eml_dag_metrics": list(self.candidate_eml_dag_metrics),
            "selected_candidate_rank": self.selected_candidate_rank,
            "eml_dag_value_samples": list(self.eml_dag_value_samples),
            "eml_dag_max_abs_error": self.eml_dag_max_abs_error,
            "eml_dag_validation_status": self.eml_dag_validation_status,
            "provenance_validation_status": self.provenance_validation_status,
            "sympy_simplify_rewrite_path_free": self.sympy_simplify_rewrite_path_free,
        }

    def to_csv_dict(self) -> dict[str, object]:
        """Return a CSV-serializable row."""
        row = self.to_json_dict()
        for key in (
            "rules_applied_by_name",
            "rules_applied_by_tier",
            "branch_sensitive_rule_names_applied",
            "original_value_samples",
            "extracted_value_samples",
            "candidate_eml_dag_metrics",
            "eml_dag_value_samples",
        ):
            row[key] = json.dumps(row[key], sort_keys=True)
        return row


@dataclass(frozen=True, slots=True)
class EgraphSemanticAuditResult:
    """Complete Goal 4.9 audit result bundle."""

    rows: tuple[EgraphSemanticAuditRow, ...]
    summary: dict[str, object]
    output_paths: tuple[Path, ...]


def audit_expressions() -> tuple[AuditExpression, ...]:
    """Return the selected Goal 4.9 audit expressions as explicit IR."""
    x = Var("x")
    y = Var("y")
    x_plus_one = Add(x, Const(1))
    x_times_x = Mul(x, x)
    y_times_y = Mul(y, y)
    return (
        AuditExpression("x+y", Add(x, y)),
        AuditExpression("y+x", Add(y, x)),
        AuditExpression("x+1", x_plus_one),
        AuditExpression("x+2-1", Sub(Add(x, Const(2)), Const(1))),
        AuditExpression("x*1", Mul(x, Const(1))),
        AuditExpression("x*x", x_times_x),
        AuditExpression("x**2", Pow(x, Const(2))),
        AuditExpression("(x+1)*(x+1)", Mul(x_plus_one, x_plus_one)),
        AuditExpression("log(x)+log(x)", Add(Log(x), Log(x))),
        AuditExpression("log(exp(x))", Log(Exp(x))),
        AuditExpression("exp(log(x))", Exp(Log(x))),
        AuditExpression("log(x*y)", Log(Mul(x, y))),
        AuditExpression("exp(x+y)", Exp(Add(x, y))),
        AuditExpression(
            "((x*x)*(y*y))*((x*x)*(x + 1))",
            Mul(Mul(x_times_x, y_times_y), Mul(x_times_x, x_plus_one)),
        ),
    )


def run_egraph_semantic_audit(
    config: EgraphSemanticAuditConfig | None = None,
) -> EgraphSemanticAuditResult:
    """Run the Goal 4.9 semantic/provenance audit and write artifacts."""
    audit_config = config or EgraphSemanticAuditConfig()
    rows = tuple(
        run_semantic_audit_row(audit_expression, rule_mode, audit_config)
        for audit_expression in audit_expressions()
        for rule_mode in RUN_MODES
    )
    summary = build_summary(rows)
    write_audit_csv(rows, audit_config.csv_path)
    write_audit_json(rows, summary, audit_config.json_path)
    write_audit_report(rows, summary, audit_config.report_path, audit_config)
    return EgraphSemanticAuditResult(
        rows=rows,
        summary=summary,
        output_paths=(audit_config.json_path, audit_config.csv_path, audit_config.report_path),
    )


def run_semantic_audit_row(
    audit_expression: AuditExpression,
    rule_mode: SemanticAuditRuleMode,
    config: EgraphSemanticAuditConfig,
) -> EgraphSemanticAuditRow:
    """Run saturation, extraction, validation, EML compilation, and purity checks."""
    original_cost = exact_eml_dag_cost(audit_expression.ir)
    egraph = EGraph()
    root_id = egraph.add_expr(audit_expression.ir)
    rules = rules_for_mode(rule_mode)
    saturation_result = saturate_with_provenance(
        egraph,
        rules,
        limits=SaturationLimits(
            max_iterations=config.saturation_max_iterations,
            max_enodes=config.saturation_max_enodes,
            max_eclasses=config.saturation_max_eclasses,
            timeout_seconds=config.saturation_timeout_seconds,
        ),
    )
    extraction_config = build_extraction_config(rule_mode, config)
    candidate_metrics = trace_candidate_eml_dag_metrics(egraph, root_id, extraction_config)
    extraction_result = extract_expression(
        egraph,
        root_id,
        original_expression=audit_expression.ir,
        config=extraction_config,
    )
    extracted_cost = (
        exact_eml_dag_cost(extraction_result.expression)
        if extraction_result.expression is not None
        else None
    )
    purity = (
        audit_pure_eml_structural_integrity(extracted_cost.eml_tree, extracted_cost.eml_dag)
        if extracted_cost is not None
        else PurityAudit(valid=False, errors=("no extracted EML output",))
    )
    numeric_audit = run_numeric_audit(
        audit_expression.ir,
        extraction_result.expression,
        extracted_cost,
    )
    provenance_status = validate_provenance(
        rule_mode=rule_mode,
        saturation=saturation_result,
        candidate_metrics=candidate_metrics,
        selected_expression=extraction_result.extracted_expression,
        rewrite_path_free=rewrite_path_is_sympy_simplify_free(),
    )

    validation_status = numeric_audit.validation_status
    if extraction_result.validation_status not in {None, "valid"}:
        validation_status = extraction_result.validation_status

    return EgraphSemanticAuditRow(
        original_expression=audit_expression.expression,
        extracted_expression=extraction_result.extracted_expression,
        rule_mode=rule_mode,
        assumptions=extraction_result.assumptions,
        rules_applied_by_name=saturation_result.rules_applied_by_name,
        rules_applied_by_tier=saturation_result.rules_applied_by_tier,
        branch_sensitive_rules_applied=saturation_result.branch_sensitive_rules_applied,
        branch_sensitive_rule_names_applied=saturation_result.branch_sensitive_rule_names_applied,
        saturation_status=saturation_result.status,
        extraction_status=extraction_result.extraction_status,
        validation_status=validation_status,
        original_value_samples=numeric_audit.original_values,
        extracted_value_samples=numeric_audit.extracted_values,
        max_abs_error=numeric_audit.max_abs_error,
        original_eml_dag_nodes=original_cost.eml_dag_nodes,
        extracted_eml_dag_nodes=extraction_result.extracted_eml_dag_nodes,
        compression_gain=safe_divide(
            original_cost.eml_dag_nodes,
            extraction_result.extracted_eml_dag_nodes,
        ),
        structural_purity_valid=purity.valid,
        purity_failure_reason="; ".join(purity.errors) if purity.errors else None,
        eclass_count=saturation_result.eclass_count,
        enode_count=saturation_result.enode_count,
        candidate_eml_dag_metrics=candidate_metrics,
        selected_candidate_rank=selected_candidate_rank(
            candidate_metrics,
            extraction_result.extracted_expression,
        ),
        eml_dag_value_samples=numeric_audit.eml_dag_values,
        eml_dag_max_abs_error=numeric_audit.eml_dag_max_abs_error,
        eml_dag_validation_status=numeric_audit.eml_dag_validation_status,
        provenance_validation_status=provenance_status,
        sympy_simplify_rewrite_path_free=rewrite_path_is_sympy_simplify_free(),
    )


def saturate_with_provenance(
    egraph: EGraph,
    rules: Sequence[RewriteRule],
    *,
    limits: SaturationLimits,
) -> ProvenanceSaturationResult:
    """Run equality saturation while recording actual applied rule provenance."""
    started_at = time.monotonic()
    rules_by_name = {rule.name: rule for rule in rules}
    by_name: Counter[str] = Counter()
    by_tier: Counter[str] = Counter()
    total_applications = 0

    if limits.max_iterations == 0:
        return provenance_result(
            "iteration_limit",
            0,
            total_applications,
            egraph,
            started_at,
            by_name,
            by_tier,
            rules_by_name,
        )

    for iteration in range(limits.max_iterations):
        status = limit_status(egraph, limits, started_at)
        if status is not None:
            return provenance_result(
                status,
                iteration,
                total_applications,
                egraph,
                started_at,
                by_name,
                by_tier,
                rules_by_name,
            )

        iteration_changed = False
        for rule in rules:
            if (status := limit_status(egraph, limits, started_at)) is not None:
                return provenance_result(
                    status,
                    iteration,
                    total_applications,
                    egraph,
                    started_at,
                    by_name,
                    by_tier,
                    rules_by_name,
                )
            result = apply_rewrite(egraph, rule)
            if result.applications:
                by_name[rule.name] += result.applications
                by_tier[rule.tier] += result.applications
            total_applications += result.applications
            if result.changed:
                iteration_changed = True
            egraph.rebuild()

        if not iteration_changed:
            return provenance_result(
                "saturated",
                iteration + 1,
                total_applications,
                egraph,
                started_at,
                by_name,
                by_tier,
                rules_by_name,
            )

    return provenance_result(
        "iteration_limit",
        limits.max_iterations,
        total_applications,
        egraph,
        started_at,
        by_name,
        by_tier,
        rules_by_name,
    )


def provenance_result(
    status: str,
    iterations_completed: int,
    total_applications: int,
    egraph: EGraph,
    started_at: float,
    by_name: Counter[str],
    by_tier: Counter[str],
    rules_by_name: dict[str, RewriteRule],
) -> ProvenanceSaturationResult:
    """Build a provenance saturation result."""
    branch_sensitive_rule_names = tuple(
        sorted(
            name
            for name, count in by_name.items()
            if count > 0 and rules_by_name[name].branch_sensitive
        )
    )
    return ProvenanceSaturationResult(
        status=status,
        iterations_completed=iterations_completed,
        total_applications=total_applications,
        enode_count=egraph.enode_count,
        eclass_count=egraph.eclass_count,
        elapsed_seconds=time.monotonic() - started_at,
        rules_applied_by_name=dict(sorted(by_name.items())),
        rules_applied_by_tier=dict(sorted(by_tier.items())),
        branch_sensitive_rules_applied=bool(branch_sensitive_rule_names),
        branch_sensitive_rule_names_applied=branch_sensitive_rule_names,
    )


def limit_status(
    egraph: EGraph,
    limits: SaturationLimits,
    started_at: float,
) -> str | None:
    """Return the first reached saturation limit, if any."""
    if egraph.enode_count >= limits.max_enodes:
        return "enode_limit"
    if egraph.eclass_count >= limits.max_eclasses:
        return "eclass_limit"
    if time.monotonic() - started_at >= limits.timeout_seconds:
        return "timeout"
    return None


def trace_candidate_eml_dag_metrics(
    egraph: EGraph,
    root_id: int,
    config: ExtractionConfig,
) -> tuple[dict[str, object], ...]:
    """Record candidate exact EML-DAG metrics used by the extractor objective."""
    candidates = enumerate_candidates(
        egraph,
        root_id,
        beam_size=config.beam_size,
        max_depth=config.max_candidate_depth,
        mode=config.extractor_mode,
        config=config,
    )
    rows: list[dict[str, object]] = []
    for candidate in candidates[: config.max_candidates_evaluated]:
        try:
            cost = exact_eml_dag_cost(candidate.expression)
            rows.append(
                {
                    "expression": cost.expression_string,
                    "ast_tree_nodes": cost.ast_tree_nodes,
                    "ast_dag_nodes": cost.ast_dag_nodes,
                    "eml_tree_nodes": cost.eml_tree_nodes,
                    "eml_dag_nodes": cost.eml_dag_nodes,
                    "tie_break_key": list(cost.tie_break_key),
                    "integrity_valid": cost.integrity_valid,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "expression": display(candidate.expression),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("eml_dag_nodes", math.inf),
            row.get("eml_tree_nodes", math.inf),
            row.get("ast_dag_nodes", math.inf),
            row.get("ast_tree_nodes", math.inf),
            str(row.get("expression", "")),
        ),
    )
    return tuple({**row, "rank": rank} for rank, row in enumerate(ranked, start=1))


def run_numeric_audit(
    original: Expr,
    extracted: Expr | None,
    extracted_cost: ExactEmlDagCost | None,
    *,
    tolerance: float = 1e-8,
) -> NumericAudit:
    """Compare original, extracted, and extracted EML-DAG values on positive probes."""
    if extracted is None or extracted_cost is None:
        return NumericAudit(
            original_values=(),
            extracted_values=(),
            eml_dag_values=(),
            max_abs_error=None,
            eml_dag_max_abs_error=None,
            validation_status="error",
            eml_dag_validation_status="error",
        )

    original_values: list[float] = []
    extracted_values: list[float] = []
    eml_dag_values: list[float] = []
    try:
        for sample in POSITIVE_REAL_SAMPLES:
            original_value = evaluate_ir_positive_real(original, sample)
            extracted_value = evaluate_ir_positive_real(extracted, sample)
            eml_dag_value = evaluate_eml_dag(extracted_cost.eml_dag, sample)
            original_values.append(original_value)
            extracted_values.append(extracted_value)
            eml_dag_values.append(eml_dag_value)
    except Exception:
        return NumericAudit(
            original_values=tuple(original_values),
            extracted_values=tuple(extracted_values),
            eml_dag_values=tuple(eml_dag_values),
            max_abs_error=None,
            eml_dag_max_abs_error=None,
            validation_status="error",
            eml_dag_validation_status="error",
        )

    errors = [
        abs(left - right) for left, right in zip(original_values, extracted_values, strict=True)
    ]
    eml_dag_errors = [
        abs(left - right) for left, right in zip(extracted_values, eml_dag_values, strict=True)
    ]
    max_abs_error = max(errors) if errors else 0.0
    eml_dag_max_abs_error = max(eml_dag_errors) if eml_dag_errors else 0.0
    return NumericAudit(
        original_values=tuple(original_values),
        extracted_values=tuple(extracted_values),
        eml_dag_values=tuple(eml_dag_values),
        max_abs_error=max_abs_error,
        eml_dag_max_abs_error=eml_dag_max_abs_error,
        validation_status="valid" if max_abs_error <= tolerance else "invalid",
        eml_dag_validation_status="valid" if eml_dag_max_abs_error <= tolerance else "invalid",
    )


def evaluate_ir_positive_real(expr: Expr, values: dict[str, float]) -> float:
    """Evaluate an IR expression on positive real probes."""
    substitutions = {sp.Symbol(name): value for name, value in values.items()}
    return float(to_sympy(expr).evalf(subs=substitutions))


def audit_pure_eml_structural_integrity(eml_tree: EmlTree, eml_dag: DagGraph) -> PurityAudit:
    """Check pure EML tree and DAG integrity, including child slots and explicit refs."""
    errors = list(validate_pure_eml_integrity(eml_tree, eml_dag))
    dag_audit = audit_eml_dag_structure(eml_dag)
    errors.extend(dag_audit.errors)
    return PurityAudit(valid=not errors, errors=tuple(dict.fromkeys(errors)))


def validate_provenance(
    *,
    rule_mode: SemanticAuditRuleMode,
    saturation: ProvenanceSaturationResult,
    candidate_metrics: Sequence[dict[str, object]],
    selected_expression: str | None,
    rewrite_path_free: bool,
) -> str:
    """Validate that rewrites and extraction are auditable."""
    if rule_mode == "safe" and saturation.branch_sensitive_rules_applied:
        return "invalid"
    if not rewrite_path_free:
        return "invalid"
    if selected_expression is None:
        return "error"
    if not candidate_metrics:
        return "invalid"
    if not all(name for name in saturation.rules_applied_by_name):
        return "invalid"
    return "valid"


def rewrite_path_is_sympy_simplify_free() -> bool:
    """Return whether the rewrite/e-graph path source files avoid SymPy simplify."""
    repo_root = Path(__file__).resolve().parents[2]
    for relative_path in REWRITE_PATH_FILES:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        if "simplify" in text:
            return False
    return True


def selected_candidate_rank(
    candidate_metrics: Sequence[dict[str, object]],
    selected_expression: str | None,
) -> int | None:
    """Return the selected candidate's rank in the exact EML-DAG trace."""
    if selected_expression is None:
        return None
    for row in candidate_metrics:
        if row.get("expression") == selected_expression:
            return int(row["rank"])
    return None


def build_summary(rows: Sequence[EgraphSemanticAuditRow]) -> dict[str, object]:
    """Build a compact semantic/provenance audit summary."""
    return {
        "row_count": len(rows),
        "all_outputs_under_outputs_v1": True,
        "all_structural_purity_valid": all(row.structural_purity_valid for row in rows),
        "all_semantic_validation_valid": all(row.validation_status == "valid" for row in rows),
        "all_eml_dag_validation_valid": all(
            row.eml_dag_validation_status == "valid" for row in rows
        ),
        "safe_branch_sensitive_application_count": sum(
            row.rule_mode == "safe" and row.branch_sensitive_rules_applied for row in rows
        ),
        "positive_real_branch_sensitive_application_count": sum(
            row.rule_mode == "positive_real_formal" and row.branch_sensitive_rules_applied
            for row in rows
        ),
        "provenance_invalid_count": sum(
            row.provenance_validation_status != "valid" for row in rows
        ),
        "sympy_simplify_rewrite_path_free": all(
            row.sympy_simplify_rewrite_path_free for row in rows
        ),
        "best_compression": [
            {
                "original_expression": row.original_expression,
                "rule_mode": row.rule_mode,
                "extracted_expression": row.extracted_expression,
                "compression_gain": row.compression_gain,
            }
            for row in sorted(rows, key=lambda item: item.compression_gain or 0.0, reverse=True)[:5]
        ],
    }


def write_audit_csv(rows: Sequence[EgraphSemanticAuditRow], path: Path) -> None:
    """Write the audit CSV artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SEMANTIC_AUDIT_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def write_audit_json(
    rows: Sequence[EgraphSemanticAuditRow],
    summary: dict[str, object],
    path: Path,
) -> None:
    """Write the audit JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "samples": list(POSITIVE_REAL_SAMPLES),
        "complex_domain_claim": False,
        "rows": [row.to_json_dict() for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_audit_report(
    rows: Sequence[EgraphSemanticAuditRow],
    summary: dict[str, object],
    path: Path,
    config: EgraphSemanticAuditConfig,
) -> None:
    """Write the Goal 4.9 markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_audit_report_markdown(rows, summary, config), encoding="utf-8")


def build_audit_report_markdown(
    rows: Sequence[EgraphSemanticAuditRow],
    summary: dict[str, object],
    config: EgraphSemanticAuditConfig,
) -> str:
    """Build the markdown report body."""
    lookup = {(row.original_expression, row.rule_mode): row for row in rows}
    lines = [
        "# Goal 4.9 E-Graph Semantic Audit",
        "",
        "This audit checks the fixed Goal 4 selected expressions on the v1 artifact path.",
        "It is a semantic, structural-purity, and provenance audit for non-ML e-graph",
        "compression. It does not use v0 as the primary corpus and does not make neural",
        "model claims.",
        "",
        "## Scope",
        "",
        "- run modes: `safe`, `positive_real_formal`",
        "- extractor: `exact_eml_dag_beam_cost`",
        "- semantic checks: positive-real numeric probes only; no complex-domain validity is "
        "claimed",
        "- final metrics: official pure EML compilation followed by exact structural EML-DAG "
        "conversion",
        "- rewrite provenance: actual applied rewrite names and tiers are recorded",
        "- extraction provenance: candidate exact EML-DAG metrics are recorded",
        "- shortcut check: rewrite/e-graph source files are scanned for `simplify` usage",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{config.json_path}`",
        f"- CSV: `{config.csv_path}`",
        f"- report: `{config.report_path}`",
        "",
        "## Summary",
        "",
        f"- rows: `{summary['row_count']}`",
        f"- all structural purity checks valid: `{summary['all_structural_purity_valid']}`",
        f"- all positive-real semantic checks valid: `{summary['all_semantic_validation_valid']}`",
        "- all extracted EML-DAG evaluator checks valid: "
        f"`{summary['all_eml_dag_validation_valid']}`",
        "- branch-sensitive rule applications in safe mode: "
        f"`{summary['safe_branch_sensitive_application_count']}`",
        "- branch-sensitive rule applications in positive-real mode: "
        f"`{summary['positive_real_branch_sensitive_application_count']}`",
        "- rewrite path free of SymPy simplify shortcut: "
        f"`{summary['sympy_simplify_rewrite_path_free']}`",
        "",
        "## Required Cases",
        "",
        "- `log(exp(x))` safe extraction: "
        f"`{lookup[('log(exp(x))', 'safe')].extracted_expression}`; "
        "branch-sensitive applied: "
        f"`{lookup[('log(exp(x))', 'safe')].branch_sensitive_rules_applied}`",
        "- `log(exp(x))` positive-real extraction: "
        f"`{lookup[('log(exp(x))', 'positive_real_formal')].extracted_expression}`; "
        "branch-sensitive applied: "
        f"`{lookup[('log(exp(x))', 'positive_real_formal')].branch_sensitive_rules_applied}`",
        "- `exp(log(x))` safe extraction: "
        f"`{lookup[('exp(log(x))', 'safe')].extracted_expression}`; "
        "branch-sensitive applied: "
        f"`{lookup[('exp(log(x))', 'safe')].branch_sensitive_rules_applied}`",
        "- `exp(log(x))` positive-real extraction: "
        f"`{lookup[('exp(log(x))', 'positive_real_formal')].extracted_expression}`; "
        "branch-sensitive applied: "
        f"`{lookup[('exp(log(x))', 'positive_real_formal')].branch_sensitive_rules_applied}`",
        "- `x+2-1` safe EML-DAG nodes: "
        f"`{lookup[('x+2-1', 'safe')].original_eml_dag_nodes} -> "
        f"{lookup[('x+2-1', 'safe')].extracted_eml_dag_nodes}`",
        "",
        "## Audit Table",
        "",
        "| Expression | Mode | Extracted | Rules applied | Branch-sensitive | "
        "Original D_EML | Extracted D_EML | Gain | Semantic | Pure |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        *render_audit_table(rows),
        "",
        "## Integrity Boundary",
        "",
        "Every successful extracted expression is compiled through the official pure EML",
        "compiler before metrics are recorded. The final EML tree/DAG must contain only",
        "`eml` internal nodes and variable or constant-`1` leaves. Derived leaves, hidden",
        "compound leaves, macro/template nodes, invalid child slots, and collapsed duplicate",
        "child references are audit failures.",
        "",
        "The positive-real mode is branch-sensitive formal algebra. It is reported separately",
        "from safe mode and is not universal complex-domain algebra.",
        "",
    ]
    return "\n".join(lines)


def render_audit_table(rows: Sequence[EgraphSemanticAuditRow]) -> list[str]:
    """Render compact markdown rows."""
    rendered: list[str] = []
    for row in rows:
        rules = ", ".join(row.rules_applied_by_name) if row.rules_applied_by_name else "none"
        rendered.append(
            "| "
            f"`{row.original_expression}` | `{row.rule_mode}` | "
            f"`{row.extracted_expression}` | `{rules}` | "
            f"`{row.branch_sensitive_rules_applied}` | {row.original_eml_dag_nodes} | "
            f"{row.extracted_eml_dag_nodes} | {format_optional_float(row.compression_gain)} | "
            f"`{row.validation_status}` | `{row.structural_purity_valid}` |"
        )
    return rendered


def build_extraction_config(
    rule_mode: SemanticAuditRuleMode,
    config: EgraphSemanticAuditConfig,
) -> ExtractionConfig:
    """Build the exact EML-DAG extraction config."""
    return ExtractionConfig(
        extractor_mode="exact_eml_dag_beam_cost",
        beam_size=config.beam_size,
        max_candidate_depth=config.max_candidate_depth,
        max_candidates_evaluated=config.max_candidates_evaluated,
        timeout_seconds=config.extraction_timeout_seconds,
        allow_positive_real_rules=rule_mode == "positive_real_formal",
        rule_mode=rule_mode,
    )


def safe_divide(numerator: int | None, denominator: int | None) -> float | None:
    """Return a safe float ratio."""
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def format_optional_float(value: float | None) -> str:
    """Format an optional float."""
    if value is None:
        return ""
    return f"{value:.6g}"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the Goal 4.9 audit CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v1"))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/goal4/GOAL4_EGRAPH_SEMANTIC_AUDIT.md"),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> EgraphSemanticAuditConfig:
    """Build config from parsed CLI arguments."""
    output_dir = args.output_dir
    return EgraphSemanticAuditConfig(
        output_dir=output_dir,
        json_path=args.json or output_dir / "goal4_egraph_semantic_audit.json",
        csv_path=args.csv or output_dir / "goal4_egraph_semantic_audit.csv",
        report_path=args.report,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 4.9 audit from the command line."""
    config = config_from_args(build_arg_parser().parse_args(argv))
    result = run_egraph_semantic_audit(config)
    print(f"Rows: {len(result.rows)}")
    for path in result.output_paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
