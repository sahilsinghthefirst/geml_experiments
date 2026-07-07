"""Goal 4.5 selected-expression e-graph compression audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from geml.egraph.costs import exact_eml_dag_cost
from geml.egraph.egraph import EGraph
from geml.egraph.extractor import ExtractionConfig, extract_expression
from geml.egraph.ir import Add, Const, Exp, Expr, Log, Mul, Pow, Sub, Var
from geml.egraph.rewrites import SaturationLimits, saturate
from geml.egraph.rule_sets import rules_for_mode

type AuditRuleMode = Literal["safe", "positive_real_formal"]

AUDIT_CSV_FIELDS = [
    "expression",
    "rule_mode",
    "assumptions",
    "saturation_status",
    "extraction_status",
    "validation_status",
    "eclass_count",
    "enode_count",
    "iterations_run",
    "rules_applied_count",
    "branch_sensitive_rules_used",
    "branch_sensitive_rule_names",
    "extracted_expression",
    "original_ast_tree_nodes",
    "original_ast_dag_nodes",
    "original_eml_tree_nodes",
    "original_eml_dag_nodes",
    "extracted_ast_tree_nodes",
    "extracted_ast_dag_nodes",
    "extracted_eml_tree_nodes",
    "extracted_eml_dag_nodes",
    "compression_gain_vs_original_eml_dag",
    "optimized_dag_alpha_vs_ast_tree",
    "optimized_dag_alpha_vs_ast_dag",
    "structural_purity_valid",
    "max_abs_error",
]

RUN_MODES: tuple[AuditRuleMode, ...] = ("safe", "positive_real_formal")


@dataclass(frozen=True, slots=True)
class AuditExpression:
    """One selected expression for the Goal 4.5 audit."""

    expression: str
    ir: Expr


@dataclass(frozen=True, slots=True)
class Goal4EgraphAuditConfig:
    """Configuration for the selected-expression Goal 4 e-graph audit."""

    output_dir: Path = Path("outputs/v1")
    csv_path: Path = Path("outputs/v1/goal4_egraph_audit.csv")
    json_path: Path = Path("outputs/v1/goal4_egraph_audit.json")
    report_path: Path = Path("docs/goal4/GOAL4_EGRAPH_AUDIT.md")
    saturation_max_iterations: int = 6
    saturation_max_enodes: int = 20_000
    saturation_max_eclasses: int = 20_000
    saturation_timeout_seconds: float = 5.0
    beam_size: int = 32
    max_candidate_depth: int = 8
    max_candidates_evaluated: int = 32
    extraction_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if _is_outputs_v0(self.output_dir) or _is_outputs_v0(self.csv_path):
            raise ValueError("Goal 4.5 audit must not write outputs/v0")
        if _is_outputs_v0(self.json_path):
            raise ValueError("Goal 4.5 audit must not write outputs/v0")
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
class Goal4EgraphAuditRow:
    """One expression/mode row from the Goal 4.5 e-graph audit."""

    expression: str
    rule_mode: AuditRuleMode
    assumptions: str | None
    saturation_status: str
    extraction_status: str
    validation_status: str | None
    eclass_count: int
    enode_count: int
    iterations_run: int
    rules_applied_count: int
    branch_sensitive_rules_used: bool
    branch_sensitive_rule_names: tuple[str, ...]
    extracted_expression: str | None
    original_ast_tree_nodes: int
    original_ast_dag_nodes: int
    original_eml_tree_nodes: int
    original_eml_dag_nodes: int
    extracted_ast_tree_nodes: int | None
    extracted_ast_dag_nodes: int | None
    extracted_eml_tree_nodes: int | None
    extracted_eml_dag_nodes: int | None
    compression_gain_vs_original_eml_dag: float | None
    optimized_dag_alpha_vs_ast_tree: float | None
    optimized_dag_alpha_vs_ast_dag: float | None
    structural_purity_valid: bool
    max_abs_error: float | None

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable row."""
        return {
            "expression": self.expression,
            "rule_mode": self.rule_mode,
            "assumptions": self.assumptions,
            "saturation_status": self.saturation_status,
            "extraction_status": self.extraction_status,
            "validation_status": self.validation_status,
            "eclass_count": self.eclass_count,
            "enode_count": self.enode_count,
            "iterations_run": self.iterations_run,
            "rules_applied_count": self.rules_applied_count,
            "branch_sensitive_rules_used": self.branch_sensitive_rules_used,
            "branch_sensitive_rule_names": list(self.branch_sensitive_rule_names),
            "extracted_expression": self.extracted_expression,
            "original_ast_tree_nodes": self.original_ast_tree_nodes,
            "original_ast_dag_nodes": self.original_ast_dag_nodes,
            "original_eml_tree_nodes": self.original_eml_tree_nodes,
            "original_eml_dag_nodes": self.original_eml_dag_nodes,
            "extracted_ast_tree_nodes": self.extracted_ast_tree_nodes,
            "extracted_ast_dag_nodes": self.extracted_ast_dag_nodes,
            "extracted_eml_tree_nodes": self.extracted_eml_tree_nodes,
            "extracted_eml_dag_nodes": self.extracted_eml_dag_nodes,
            "compression_gain_vs_original_eml_dag": self.compression_gain_vs_original_eml_dag,
            "optimized_dag_alpha_vs_ast_tree": self.optimized_dag_alpha_vs_ast_tree,
            "optimized_dag_alpha_vs_ast_dag": self.optimized_dag_alpha_vs_ast_dag,
            "structural_purity_valid": self.structural_purity_valid,
            "max_abs_error": self.max_abs_error,
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
class Goal4EgraphAuditResult:
    """Complete Goal 4.5 audit result bundle."""

    rows: tuple[Goal4EgraphAuditRow, ...]
    equivalence_checks: tuple[dict[str, object], ...]
    summary: dict[str, object]
    output_paths: tuple[Path, ...]


def audit_expressions() -> tuple[AuditExpression, ...]:
    """Return the selected Goal 4.5 audit expressions as explicit IR."""
    x = Var("x")
    y = Var("y")
    x_plus_one = Add(x, Const(1))
    x_times_x = Mul(x, x)
    y_times_y = Mul(y, y)

    return (
        AuditExpression("x + y", Add(x, y)),
        AuditExpression("y + x", Add(y, x)),
        AuditExpression("x + 1", x_plus_one),
        AuditExpression("x + 2 - 1", Sub(Add(x, Const(2)), Const(1))),
        AuditExpression("x * 1", Mul(x, Const(1))),
        AuditExpression("x * x", x_times_x),
        AuditExpression("x**2", Pow(x, Const(2))),
        AuditExpression("(x + 1) * (x + 1)", Mul(x_plus_one, x_plus_one)),
        AuditExpression("log(x) + log(x)", Add(Log(x), Log(x))),
        AuditExpression("log(exp(x))", Log(Exp(x))),
        AuditExpression("exp(log(x))", Exp(Log(x))),
        AuditExpression("log(x*y)", Log(Mul(x, y))),
        AuditExpression("exp(x+y)", Exp(Add(x, y))),
        AuditExpression(
            "((x*x)*(y*y))*((x*x)*(x + 1))",
            Mul(Mul(x_times_x, y_times_y), Mul(x_times_x, x_plus_one)),
        ),
    )


def run_goal4_egraph_audit(
    config: Goal4EgraphAuditConfig | None = None,
) -> Goal4EgraphAuditResult:
    """Run the selected-expression e-graph audit and write artifacts."""
    audit_config = config or Goal4EgraphAuditConfig()
    rows = tuple(
        run_audit_row(audit_expression, rule_mode, audit_config)
        for audit_expression in audit_expressions()
        for rule_mode in RUN_MODES
    )
    equivalence_checks = run_equivalence_checks(audit_config)
    summary = build_summary(rows, equivalence_checks)

    write_audit_csv(rows, audit_config.csv_path)
    write_audit_json(rows, equivalence_checks, summary, audit_config.json_path)
    write_audit_report(rows, equivalence_checks, summary, audit_config.report_path, audit_config)

    return Goal4EgraphAuditResult(
        rows=rows,
        equivalence_checks=equivalence_checks,
        summary=summary,
        output_paths=(
            audit_config.csv_path,
            audit_config.json_path,
            audit_config.report_path,
        ),
    )


def run_audit_row(
    audit_expression: AuditExpression,
    rule_mode: AuditRuleMode,
    config: Goal4EgraphAuditConfig,
) -> Goal4EgraphAuditRow:
    """Run saturation, extraction, validation, and metrics for one row."""
    original_cost = exact_eml_dag_cost(audit_expression.ir)
    egraph = EGraph()
    root_id = egraph.add_expr(audit_expression.ir)
    saturation_result = saturate(
        egraph,
        rules_for_mode(rule_mode),
        limits=SaturationLimits(
            max_iterations=config.saturation_max_iterations,
            max_enodes=config.saturation_max_enodes,
            max_eclasses=config.saturation_max_eclasses,
            timeout_seconds=config.saturation_timeout_seconds,
        ),
    )
    extraction_result = extract_expression(
        egraph,
        root_id,
        original_expression=audit_expression.ir,
        config=_extraction_config(rule_mode, config),
    )
    extracted_eml_dag_nodes = extraction_result.extracted_eml_dag_nodes
    extracted_ast_tree_nodes = extraction_result.extracted_ast_tree_nodes
    extracted_ast_dag_nodes = extraction_result.extracted_ast_dag_nodes

    return Goal4EgraphAuditRow(
        expression=audit_expression.expression,
        rule_mode=rule_mode,
        assumptions=extraction_result.assumptions,
        saturation_status=saturation_result.status,
        extraction_status=extraction_result.extraction_status,
        validation_status=extraction_result.validation_status,
        eclass_count=saturation_result.eclass_count,
        enode_count=saturation_result.enode_count,
        iterations_run=saturation_result.iterations_completed,
        rules_applied_count=saturation_result.total_applications,
        branch_sensitive_rules_used=extraction_result.branch_sensitive_rules_used,
        branch_sensitive_rule_names=extraction_result.branch_sensitive_rule_names,
        extracted_expression=extraction_result.extracted_expression,
        original_ast_tree_nodes=original_cost.ast_tree_nodes,
        original_ast_dag_nodes=original_cost.ast_dag_nodes,
        original_eml_tree_nodes=original_cost.eml_tree_nodes,
        original_eml_dag_nodes=original_cost.eml_dag_nodes,
        extracted_ast_tree_nodes=extracted_ast_tree_nodes,
        extracted_ast_dag_nodes=extracted_ast_dag_nodes,
        extracted_eml_tree_nodes=extraction_result.extracted_eml_tree_nodes,
        extracted_eml_dag_nodes=extracted_eml_dag_nodes,
        compression_gain_vs_original_eml_dag=_safe_divide(
            original_cost.eml_dag_nodes,
            extracted_eml_dag_nodes,
        ),
        optimized_dag_alpha_vs_ast_tree=_safe_divide(
            extracted_eml_dag_nodes,
            original_cost.ast_tree_nodes,
        ),
        optimized_dag_alpha_vs_ast_dag=_safe_divide(
            extracted_eml_dag_nodes,
            original_cost.ast_dag_nodes,
        ),
        structural_purity_valid=bool(extraction_result.integrity_valid),
        max_abs_error=extraction_result.positive_real_max_abs_error,
    )


def run_equivalence_checks(config: Goal4EgraphAuditConfig) -> tuple[dict[str, object], ...]:
    """Run explicit same-e-class checks discussed by the audit report."""
    x = Var("x")
    y = Var("y")
    checks = (
        ("x+y_vs_y+x", Add(x, y), Add(y, x)),
        ("x+1_vs_x+2-1", Add(x, Const(1)), Sub(Add(x, Const(2)), Const(1))),
        ("log_exp_x_vs_x", Log(Exp(x)), x),
        ("exp_log_x_vs_x", Exp(Log(x)), x),
        ("x_pow_2_vs_x_times_x", Pow(x, Const(2)), Mul(x, x)),
    )
    return tuple(
        {
            "check": name,
            "rule_mode": rule_mode,
            "same_eclass": same_eclass(left, right, rule_mode, config),
        }
        for name, left, right in checks
        for rule_mode in RUN_MODES
    )


def same_eclass(
    left: Expr,
    right: Expr,
    rule_mode: AuditRuleMode,
    config: Goal4EgraphAuditConfig,
) -> bool:
    """Return whether two expressions end in the same e-class under one mode."""
    egraph = EGraph()
    left_id = egraph.add_expr(left)
    right_id = egraph.add_expr(right)
    saturate(
        egraph,
        rules_for_mode(rule_mode),
        limits=SaturationLimits(
            max_iterations=config.saturation_max_iterations,
            max_enodes=config.saturation_max_enodes,
            max_eclasses=config.saturation_max_eclasses,
            timeout_seconds=config.saturation_timeout_seconds,
        ),
    )
    return egraph.find(left_id) == egraph.find(right_id)


def build_summary(
    rows: Sequence[Goal4EgraphAuditRow],
    equivalence_checks: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Build a compact JSON/report summary."""
    worsened = [
        row
        for row in rows
        if row.extracted_eml_dag_nodes is not None
        and row.extracted_eml_dag_nodes > row.original_eml_dag_nodes
    ]
    improved = [
        row
        for row in rows
        if row.extracted_eml_dag_nodes is not None
        and row.extracted_eml_dag_nodes < row.original_eml_dag_nodes
    ]
    completed = [row for row in rows if row.extraction_status == "completed"]
    return {
        "row_count": len(rows),
        "completed_count": len(completed),
        "all_final_eml_outputs_pure": all(row.structural_purity_valid for row in rows),
        "worsened_count": len(worsened),
        "worsened_rows": [
            {
                "expression": row.expression,
                "rule_mode": row.rule_mode,
                "original_eml_dag_nodes": row.original_eml_dag_nodes,
                "extracted_eml_dag_nodes": row.extracted_eml_dag_nodes,
            }
            for row in worsened
        ],
        "improved_count": len(improved),
        "best_improvements": [
            {
                "expression": row.expression,
                "rule_mode": row.rule_mode,
                "extracted_expression": row.extracted_expression,
                "compression_gain_vs_original_eml_dag": row.compression_gain_vs_original_eml_dag,
            }
            for row in sorted(
                improved,
                key=lambda item: item.compression_gain_vs_original_eml_dag or 0.0,
                reverse=True,
            )[:5]
        ],
        "equivalence_checks": list(equivalence_checks),
    }


def write_audit_csv(rows: Sequence[Goal4EgraphAuditRow], path: Path) -> None:
    """Write per-row audit CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=AUDIT_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def write_audit_json(
    rows: Sequence[Goal4EgraphAuditRow],
    equivalence_checks: Sequence[dict[str, object]],
    summary: dict[str, object],
    path: Path,
) -> None:
    """Write audit JSON bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "equivalence_checks": list(equivalence_checks),
        "rows": [row.to_json_dict() for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_audit_report(
    rows: Sequence[Goal4EgraphAuditRow],
    equivalence_checks: Sequence[dict[str, object]],
    summary: dict[str, object],
    path: Path,
    config: Goal4EgraphAuditConfig,
) -> None:
    """Write the Goal 4.5 markdown audit report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_audit_report_markdown(rows, equivalence_checks, summary, config),
        encoding="utf-8",
    )


def build_audit_report_markdown(
    rows: Sequence[Goal4EgraphAuditRow],
    equivalence_checks: Sequence[dict[str, object]],
    summary: dict[str, object],
    config: Goal4EgraphAuditConfig,
) -> str:
    """Build the Goal 4.5 markdown audit report."""
    lines = [
        "# Goal 4.5 E-Graph Compression Audit",
        "",
        "This report audits selected expressions before any 10k v1 Goal 4 run. It is a",
        "semantic and compression sanity check for algebraic e-graph rewriting, not a",
        "large-scale corpus result.",
        "",
        "## Artifacts",
        "",
        f"- CSV: `{config.csv_path}`",
        f"- JSON: `{config.json_path}`",
        f"- report: `{config.report_path}`",
        "",
        "## Method",
        "",
        "- run modes: `safe`, `positive_real_formal`",
        "- extractor: `exact_eml_dag_beam_cost`",
        "- final EML metrics: official pure EML compiler, then exact Goal 3 EML-DAG",
        "- positive-real results are branch-sensitive and reported separately",
        "- no 10k pipeline, neural model, or visualization was run",
        "",
        "## Key Checks",
        "",
        *_render_equivalence_findings(equivalence_checks),
        *_render_targeted_findings(rows),
        "",
        "## Purity And Worsening",
        "",
        f"- all final EML outputs pure: `{summary['all_final_eml_outputs_pure']}`",
        f"- rows with worse extracted EML-DAG size: `{summary['worsened_count']}`",
    ]
    if summary["worsened_rows"]:
        lines.extend(_render_worsened_rows(summary["worsened_rows"]))
    else:
        lines.append("- no audited extraction worsened official pure EML-DAG size")

    lines.extend(
        [
            "",
            "## Audit Table",
            "",
            "| Expression | Mode | Extracted | Original D_EML | "
            "Extracted D_EML | Gain | Status | Pure |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
            *_render_row_table(rows),
            "",
            "## Interpretation Boundary",
            "",
            "This is a selected-expression audit only. It confirms that the current Goal 4",
            "infrastructure preserves pure EML outputs on these cases and that",
            "positive-real log/exp simplifications remain separated from safe mode.",
            "It does not prove global minimal EML form and does not replace the future",
            "v1 corpus-scale Goal 4 run.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_equivalence_findings(
    equivalence_checks: Sequence[dict[str, object]],
) -> list[str]:
    check = _check_lookup(equivalence_checks)
    return [
        "- `x+y` and `y+x` end in the same e-class in safe mode: "
        f"`{check[('x+y_vs_y+x', 'safe')]}`",
        "- `x+y` and `y+x` end in the same e-class in positive-real mode: "
        f"`{check[('x+y_vs_y+x', 'positive_real_formal')]}`",
        "- `x+1` and `x+2-1` end in the same e-class in safe mode: "
        f"`{check[('x+1_vs_x+2-1', 'safe')]}`",
        "- `x+1` and `x+2-1` end in the same e-class in positive-real mode: "
        f"`{check[('x+1_vs_x+2-1', 'positive_real_formal')]}`",
        f"- `log(exp(x))` is equivalent to `x` in safe mode: `{check[('log_exp_x_vs_x', 'safe')]}`",
        "- `log(exp(x))` is equivalent to `x` in positive-real mode: "
        f"`{check[('log_exp_x_vs_x', 'positive_real_formal')]}`",
        f"- `exp(log(x))` is equivalent to `x` in safe mode: `{check[('exp_log_x_vs_x', 'safe')]}`",
        "- `exp(log(x))` is equivalent to `x` in positive-real mode: "
        f"`{check[('exp_log_x_vs_x', 'positive_real_formal')]}`",
    ]


def _render_targeted_findings(rows: Sequence[Goal4EgraphAuditRow]) -> list[str]:
    lookup = _row_lookup(rows)
    x_pow_safe = lookup[("x**2", "safe")]
    x_pow_positive = lookup[("x**2", "positive_real_formal")]
    log_exp_safe = lookup[("log(exp(x))", "safe")]
    log_exp_positive = lookup[("log(exp(x))", "positive_real_formal")]
    exp_log_safe = lookup[("exp(log(x))", "safe")]
    exp_log_positive = lookup[("exp(log(x))", "positive_real_formal")]
    return [
        f"- `log(exp(x))` extracted expression in safe mode: `{log_exp_safe.extracted_expression}`",
        "- `log(exp(x))` extracted expression in positive-real mode: "
        f"`{log_exp_positive.extracted_expression}`",
        f"- `exp(log(x))` extracted expression in safe mode: `{exp_log_safe.extracted_expression}`",
        "- `exp(log(x))` extracted expression in positive-real mode: "
        f"`{exp_log_positive.extracted_expression}`",
        "- `x**2` safe-mode EML-DAG nodes: "
        f"`{x_pow_safe.original_eml_dag_nodes} -> {x_pow_safe.extracted_eml_dag_nodes}`",
        "- `x**2` positive-real EML-DAG nodes: "
        f"`{x_pow_positive.original_eml_dag_nodes} -> {x_pow_positive.extracted_eml_dag_nodes}`",
    ]


def _render_worsened_rows(rows: object) -> list[str]:
    if not isinstance(rows, list):
        return []
    rendered = ["", "Worsened rows:"]
    for row in rows:
        if isinstance(row, dict):
            rendered.append(
                "- "
                f"`{row['expression']}` / `{row['rule_mode']}`: "
                f"`{row['original_eml_dag_nodes']} -> {row['extracted_eml_dag_nodes']}`"
            )
    return rendered


def _render_row_table(rows: Sequence[Goal4EgraphAuditRow]) -> list[str]:
    return [
        " | ".join(
            [
                f"| `{row.expression}`",
                f"`{row.rule_mode}`",
                f"`{row.extracted_expression}`",
                str(row.original_eml_dag_nodes),
                str(row.extracted_eml_dag_nodes),
                _format_optional_float(row.compression_gain_vs_original_eml_dag),
                f"`{row.extraction_status}`",
                f"`{row.structural_purity_valid}` |",
            ]
        )
        for row in rows
    ]


def _row_lookup(
    rows: Sequence[Goal4EgraphAuditRow],
) -> dict[tuple[str, AuditRuleMode], Goal4EgraphAuditRow]:
    return {(row.expression, row.rule_mode): row for row in rows}


def _check_lookup(
    equivalence_checks: Sequence[dict[str, object]],
) -> dict[tuple[str, str], object]:
    return {
        (str(row["check"]), str(row["rule_mode"])): row["same_eclass"] for row in equivalence_checks
    }


def _extraction_config(
    rule_mode: AuditRuleMode,
    config: Goal4EgraphAuditConfig,
) -> ExtractionConfig:
    return ExtractionConfig(
        extractor_mode="exact_eml_dag_beam_cost",
        beam_size=config.beam_size,
        max_candidate_depth=config.max_candidate_depth,
        max_candidates_evaluated=config.max_candidates_evaluated,
        timeout_seconds=config.extraction_timeout_seconds,
        allow_positive_real_rules=rule_mode == "positive_real_formal",
        rule_mode=rule_mode,
    )


def _safe_divide(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def _is_outputs_v0(path: Path) -> bool:
    return "outputs/v0" in path.as_posix()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the Goal 4.5 audit CLI parser."""
    parser = argparse.ArgumentParser(description="Run the Goal 4.5 e-graph audit.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v1"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=Path("docs/goal4/GOAL4_EGRAPH_AUDIT.md"))
    parser.add_argument("--saturation-max-iterations", type=int, default=6)
    parser.add_argument("--saturation-max-enodes", type=int, default=20_000)
    parser.add_argument("--saturation-max-eclasses", type=int, default=20_000)
    parser.add_argument("--saturation-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--beam-size", type=int, default=32)
    parser.add_argument("--max-candidate-depth", type=int, default=8)
    parser.add_argument("--max-candidates-evaluated", type=int, default=32)
    parser.add_argument("--extraction-timeout-seconds", type=float, default=5.0)
    return parser


def config_from_args(args: argparse.Namespace) -> Goal4EgraphAuditConfig:
    """Build audit config from parsed CLI args."""
    output_dir = args.output_dir
    csv_path = args.csv or output_dir / "goal4_egraph_audit.csv"
    json_path = args.json or output_dir / "goal4_egraph_audit.json"
    return Goal4EgraphAuditConfig(
        output_dir=output_dir,
        csv_path=csv_path,
        json_path=json_path,
        report_path=args.report,
        saturation_max_iterations=args.saturation_max_iterations,
        saturation_max_enodes=args.saturation_max_enodes,
        saturation_max_eclasses=args.saturation_max_eclasses,
        saturation_timeout_seconds=args.saturation_timeout_seconds,
        beam_size=args.beam_size,
        max_candidate_depth=args.max_candidate_depth,
        max_candidates_evaluated=args.max_candidates_evaluated,
        extraction_timeout_seconds=args.extraction_timeout_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 4.5 audit from the command line."""
    parser = build_arg_parser()
    config = config_from_args(parser.parse_args(argv))
    result = run_goal4_egraph_audit(config)
    print(f"Rows: {len(result.rows)}")
    print(f"CSV: {config.csv_path}")
    print(f"JSON: {config.json_path}")
    print(f"Report: {config.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
