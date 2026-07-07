"""Goal 3.6 semantic and structural audit for exact DAG compression."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import sympy as sp
from pydantic import BaseModel, Field, model_validator

from geml.symbolic.dag_graph import (
    FORBIDDEN_DAG_KINDS,
    FORBIDDEN_DAG_LABELS,
    DagChildRef,
    DagGraph,
    DagNode,
    StructuralSignature,
    validate_dag_graph,
)
from geml.symbolic.dag_metrics import ExpressionDagAnalysis, compute_expression_dag_analysis
from geml.symbolic.official_eml_compiler import evaluate_official_eml_tree

UNSUPPORTED_FINAL_EML_LABELS = (
    frozenset(
        {
            "Add",
            "Mul",
            "Sub",
            "Div",
            "Pow",
            "Exp",
            "Log",
            "Derived",
            "add",
            "mul",
            "pow",
            "exp",
            "log",
            "macro",
            "template",
        }
    )
    | FORBIDDEN_DAG_LABELS
)
DEFAULT_SAFE_INPUTS = (
    {"x": 1.2, "y": 1.4, "z": 1.6},
    {"x": 1.3, "y": 2.1, "z": 3.2},
    {"x": 2.0, "y": 1.7, "z": 1.5},
)
SEMANTIC_TOLERANCE = 1e-8
AUDIT_CSV_FIELDS = [
    "name",
    "expression",
    "srepr",
    "ast_tree_node_count",
    "ast_dag_node_count",
    "eml_tree_node_count",
    "eml_dag_node_count",
    "tree_alpha",
    "dag_alpha_vs_ast_tree",
    "dag_alpha_vs_ast_dag",
    "eml_dag_compression",
    "top_shared_subtree_signatures",
    "structural_valid",
    "semantic_numeric_valid",
    "max_abs_error_tree",
    "max_abs_error_dag",
    "derived_leaf_count",
    "hidden_compound_leaf_count",
    "macro_template_node_count",
    "unsupported_final_label_count",
    "duplicate_child_ref_parent_count",
    "error",
]


class DagSemanticAuditConfig(BaseModel):
    """Configuration for Goal 3.6 semantic DAG audit exports."""

    json_path: Path = Path("outputs/v0/goal3_dag_semantic_audit.json")
    csv_path: Path = Path("outputs/v0/goal3_dag_semantic_audit.csv")
    docs_path: Path = Path("docs/goal3/GOAL3_DAG_SEMANTIC_AUDIT.md")
    top_shared_limit: int = Field(default=10, gt=0)
    semantic_tolerance: float = Field(default=SEMANTIC_TOLERANCE, gt=0)

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if self.json_path == self.csv_path:
            raise ValueError("JSON and CSV audit output paths must differ")
        return self


@dataclass(frozen=True)
class AuditExpression:
    """One fixed expression in the Goal 3.6 audit suite."""

    name: str
    expr: sp.Expr


class SharedSubtreeRecord(BaseModel):
    """Shared subtree signature with reuse metadata."""

    rank: int
    node_id: int
    label: str
    kind: str
    reuse_count: int
    incoming_child_ref_count: int
    signature: str


class StructuralAuditResult(BaseModel):
    """Structural purity checks for a pure EML DAG."""

    valid: bool
    errors: list[str]
    derived_leaf_count: int
    hidden_compound_leaf_count: int
    macro_template_node_count: int
    unsupported_final_label_count: int
    duplicate_child_ref_parent_count: int


class SemanticCheckResult(BaseModel):
    """Numeric semantic equivalence checks for one expression."""

    valid: bool
    max_abs_error_tree: float
    max_abs_error_dag: float
    evaluations: list[dict[str, object]]


class DagSemanticAuditRow(BaseModel):
    """One exported Goal 3.6 semantic audit row."""

    name: str
    expression: str
    srepr: str
    ast_tree_node_count: int | None = None
    ast_dag_node_count: int | None = None
    eml_tree_node_count: int | None = None
    eml_dag_node_count: int | None = None
    tree_alpha: float | None = None
    dag_alpha_vs_ast_tree: float | None = None
    dag_alpha_vs_ast_dag: float | None = None
    eml_dag_compression: float | None = None
    top_shared_subtree_signatures: list[SharedSubtreeRecord] = Field(default_factory=list)
    structural_valid: bool = False
    structural_errors: list[str] = Field(default_factory=list)
    semantic_numeric_valid: bool = False
    semantic_evaluations: list[dict[str, object]] = Field(default_factory=list)
    max_abs_error_tree: float | None = None
    max_abs_error_dag: float | None = None
    derived_leaf_count: int | None = None
    hidden_compound_leaf_count: int | None = None
    macro_template_node_count: int | None = None
    unsupported_final_label_count: int | None = None
    duplicate_child_ref_parent_count: int | None = None
    error: str | None = None


class DagSemanticAuditResult(BaseModel):
    """Full Goal 3.6 audit export payload."""

    audit_name: str
    expression_count: int
    structural_valid_count: int
    semantic_numeric_valid_count: int
    safe_positive_inputs: list[dict[str, float]]
    rows: list[DagSemanticAuditRow]


def run_dag_semantic_audit(config: DagSemanticAuditConfig) -> DagSemanticAuditResult:
    """Run the fixed Goal 3.6 audit suite and write JSON, CSV, and markdown docs."""
    rows = [
        audit_expression(
            audit_expr,
            safe_inputs=DEFAULT_SAFE_INPUTS,
            top_shared_limit=config.top_shared_limit,
            semantic_tolerance=config.semantic_tolerance,
        )
        for audit_expr in build_audit_suite()
    ]
    result = DagSemanticAuditResult(
        audit_name="goal3_dag_semantic_audit",
        expression_count=len(rows),
        structural_valid_count=sum(1 for row in rows if row.structural_valid),
        semantic_numeric_valid_count=sum(1 for row in rows if row.semantic_numeric_valid),
        safe_positive_inputs=[dict(input_row) for input_row in DEFAULT_SAFE_INPUTS],
        rows=rows,
    )
    write_audit_json(result, config.json_path)
    write_audit_csv(rows, config.csv_path)
    write_audit_docs(result, config.docs_path, csv_path=config.csv_path, json_path=config.json_path)
    return result


def audit_expression(
    audit_expr: AuditExpression,
    *,
    safe_inputs: Sequence[dict[str, float]],
    top_shared_limit: int,
    semantic_tolerance: float,
) -> DagSemanticAuditRow:
    """Audit one expression for structural purity and numeric semantics."""
    try:
        analysis = compute_expression_dag_analysis(audit_expr.expr)
        structural = audit_eml_dag_structure(analysis.eml_dag)
        semantic = check_numeric_semantics(
            audit_expr.expr,
            analysis=analysis,
            safe_inputs=safe_inputs,
            tolerance=semantic_tolerance,
        )
        metrics = analysis.metrics
        return DagSemanticAuditRow(
            name=audit_expr.name,
            expression=metrics.expression,
            srepr=metrics.srepr,
            ast_tree_node_count=metrics.ast_tree_node_count,
            ast_dag_node_count=metrics.ast_dag_node_count,
            eml_tree_node_count=metrics.eml_tree_node_count,
            eml_dag_node_count=metrics.eml_dag_node_count,
            tree_alpha=metrics.tree_alpha,
            dag_alpha_vs_ast_tree=metrics.dag_alpha_vs_ast_tree,
            dag_alpha_vs_ast_dag=metrics.dag_alpha_vs_ast_dag,
            eml_dag_compression=metrics.eml_dag_compression,
            top_shared_subtree_signatures=top_shared_subtree_signatures(
                analysis.eml_dag,
                limit=top_shared_limit,
            ),
            structural_valid=structural.valid,
            structural_errors=structural.errors,
            semantic_numeric_valid=semantic.valid,
            semantic_evaluations=semantic.evaluations,
            max_abs_error_tree=semantic.max_abs_error_tree,
            max_abs_error_dag=semantic.max_abs_error_dag,
            derived_leaf_count=structural.derived_leaf_count,
            hidden_compound_leaf_count=structural.hidden_compound_leaf_count,
            macro_template_node_count=structural.macro_template_node_count,
            unsupported_final_label_count=structural.unsupported_final_label_count,
            duplicate_child_ref_parent_count=structural.duplicate_child_ref_parent_count,
            error=None,
        )
    except Exception as exc:
        return DagSemanticAuditRow(
            name=audit_expr.name,
            expression=str(audit_expr.expr),
            srepr=sp.srepr(audit_expr.expr),
            error=f"{type(exc).__name__}: {exc}",
        )


def build_audit_suite() -> list[AuditExpression]:
    """Return the fixed Goal 3.6 audit expression suite."""
    x, y = sp.symbols("x y")
    repeated_sum_left = sp.Add(x, 1, evaluate=False)
    repeated_sum_right = sp.Add(x, 1, evaluate=False)
    repeated_product_left = sp.Mul(x, x, evaluate=False)
    repeated_product_right = sp.Mul(x, x, evaluate=False)
    complex_left = sp.Mul(
        sp.Mul(x, x, evaluate=False),
        sp.Mul(y, y, evaluate=False),
        evaluate=False,
    )
    complex_right = sp.Mul(
        sp.Mul(x, x, evaluate=False),
        sp.Add(x, 1, evaluate=False),
        evaluate=False,
    )
    return [
        AuditExpression("x+y", sp.Add(x, y, evaluate=False)),
        AuditExpression("x*y", sp.Mul(x, y, evaluate=False)),
        AuditExpression("log(x)", sp.log(x, evaluate=False)),
        AuditExpression("exp(x)", sp.exp(x, evaluate=False)),
        AuditExpression("x**2", sp.Pow(x, 2, evaluate=False)),
        AuditExpression("x+x", sp.Add(x, x, evaluate=False)),
        AuditExpression("x*x", sp.Mul(x, x, evaluate=False)),
        AuditExpression(
            "(x+1)*(x+1)",
            sp.Mul(repeated_sum_left, repeated_sum_right, evaluate=False),
        ),
        AuditExpression(
            "(x*x)*(x*x)",
            sp.Mul(repeated_product_left, repeated_product_right, evaluate=False),
        ),
        AuditExpression(
            "log(x)+log(x)",
            sp.Add(sp.log(x, evaluate=False), sp.log(x, evaluate=False), evaluate=False),
        ),
        AuditExpression(
            "exp(x)+exp(x)",
            sp.Add(sp.exp(x, evaluate=False), sp.exp(x, evaluate=False), evaluate=False),
        ),
        AuditExpression(
            "((x*x)*(y*y))*((x*x)*(x+1))",
            sp.Mul(complex_left, complex_right, evaluate=False),
        ),
    ]


def audit_eml_dag_structure(dag: DagGraph) -> StructuralAuditResult:
    """Run structural purity checks for a pure EML DAG."""
    errors: list[str] = []
    try:
        validate_dag_graph(dag)
    except ValueError as exc:
        errors.append(str(exc))

    refs_by_parent = child_refs_by_parent(dag.child_refs)
    derived_leaf_count = 0
    hidden_compound_leaf_count = 0
    macro_template_node_count = 0
    unsupported_final_label_count = 0
    duplicate_child_ref_parent_count = 0

    if dag.metadata.get("dag_mode") != "restricted_eml_pure_dag":
        errors.append(f"expected restricted_eml_pure_dag, got {dag.metadata.get('dag_mode')!r}")

    for parent_id, refs in refs_by_parent.items():
        child_ids = [ref.child_id for ref in refs]
        if len(child_ids) != len(set(child_ids)):
            duplicate_child_ref_parent_count += 1
            sorted_refs = sorted(refs, key=lambda ref: ref.slot_index)
            if [ref.child_slot for ref in sorted_refs] != ["left", "right"]:
                errors.append(f"duplicate child refs for parent {parent_id} lost left/right slots")

    for node in dag.nodes:
        refs = refs_by_parent.get(node.id, [])
        if node.kind == "derived":
            derived_leaf_count += 1
        if node.metadata.get("contains_hidden_compound") is True:
            hidden_compound_leaf_count += 1
        if node.kind in FORBIDDEN_DAG_KINDS or node.label in FORBIDDEN_DAG_LABELS:
            macro_template_node_count += 1
        if node.label in UNSUPPORTED_FINAL_EML_LABELS or node.label.startswith("eml_"):
            unsupported_final_label_count += 1

        if refs:
            if node.kind != "eml" or node.label != "eml":
                errors.append(
                    f"internal EML DAG node {node.id} must be kind='eml', label='eml'; "
                    f"got kind={node.kind!r}, label={node.label!r}"
                )
            if len(refs) != 2:
                errors.append(f"EML DAG internal node {node.id} has {len(refs)} child refs")
            sorted_refs = sorted(refs, key=lambda ref: ref.slot_index)
            if [ref.child_slot for ref in sorted_refs] != ["left", "right"]:
                errors.append(
                    f"EML DAG internal node {node.id} child slots are "
                    f"{[ref.child_slot for ref in sorted_refs]!r}, expected ['left', 'right']"
                )
        else:
            if node.kind == "constant":
                if node.label != "1":
                    errors.append(f"constant EML DAG leaf {node.id} has label {node.label!r}")
            elif node.kind != "variable":
                errors.append(
                    f"EML DAG leaf {node.id} must be variable or constant 1; "
                    f"got kind={node.kind!r}, label={node.label!r}"
                )

    if derived_leaf_count:
        errors.append(f"derived leaves present: {derived_leaf_count}")
    if hidden_compound_leaf_count:
        errors.append(f"hidden compound leaves present: {hidden_compound_leaf_count}")
    if macro_template_node_count:
        errors.append(f"macro/template/forbidden nodes present: {macro_template_node_count}")
    if unsupported_final_label_count:
        errors.append(f"unsupported final EML labels present: {unsupported_final_label_count}")

    deduped_errors = list(dict.fromkeys(errors))
    return StructuralAuditResult(
        valid=not deduped_errors,
        errors=deduped_errors,
        derived_leaf_count=derived_leaf_count,
        hidden_compound_leaf_count=hidden_compound_leaf_count,
        macro_template_node_count=macro_template_node_count,
        unsupported_final_label_count=unsupported_final_label_count,
        duplicate_child_ref_parent_count=duplicate_child_ref_parent_count,
    )


def assert_eml_dag_structurally_pure(dag: DagGraph) -> None:
    """Raise if a pure EML DAG leaks hidden complexity or invalid structure."""
    result = audit_eml_dag_structure(dag)
    if not result.valid:
        raise ValueError("; ".join(result.errors))


def check_numeric_semantics(
    expr: sp.Expr,
    *,
    analysis: ExpressionDagAnalysis,
    safe_inputs: Sequence[dict[str, float]],
    tolerance: float,
) -> SemanticCheckResult:
    """Compare original SymPy, EML tree, and EML DAG numeric values."""
    evaluations: list[dict[str, object]] = []
    max_tree_error = 0.0
    max_dag_error = 0.0

    for values in safe_inputs:
        sympy_value = evaluate_sympy_expression(expr, values)
        tree_value = evaluate_official_eml_tree(analysis.eml_tree, values)
        dag_value = evaluate_eml_dag(analysis.eml_dag, values)
        tree_error = abs(sympy_value - tree_value)
        dag_error = abs(sympy_value - dag_value)
        tree_matches = math.isclose(sympy_value, tree_value, rel_tol=tolerance, abs_tol=tolerance)
        dag_matches = math.isclose(sympy_value, dag_value, rel_tol=tolerance, abs_tol=tolerance)
        tree_dag_matches = math.isclose(tree_value, dag_value, rel_tol=tolerance, abs_tol=tolerance)
        max_tree_error = max(max_tree_error, tree_error)
        max_dag_error = max(max_dag_error, dag_error)
        evaluations.append(
            {
                "inputs": dict(values),
                "sympy_value": sympy_value,
                "eml_tree_value": tree_value,
                "eml_dag_value": dag_value,
                "tree_abs_error": tree_error,
                "dag_abs_error": dag_error,
                "tree_matches_sympy": tree_matches,
                "dag_matches_sympy": dag_matches,
                "dag_matches_tree": tree_dag_matches,
            }
        )

    valid = all(
        bool(row["tree_matches_sympy"])
        and bool(row["dag_matches_sympy"])
        and bool(row["dag_matches_tree"])
        for row in evaluations
    )
    return SemanticCheckResult(
        valid=valid,
        max_abs_error_tree=max_tree_error,
        max_abs_error_dag=max_dag_error,
        evaluations=evaluations,
    )


def evaluate_sympy_expression(expr: sp.Expr, values: dict[str, float]) -> float:
    """Evaluate a SymPy expression numerically without changing structure."""
    substitutions = {sp.Symbol(name): value for name, value in values.items()}
    return float(expr.evalf(subs=substitutions))


def evaluate_eml_dag(dag: DagGraph, values: dict[str, float]) -> float:
    """Numerically evaluate a pure EML DAG using explicit child references."""
    refs_by_parent = child_refs_by_parent(dag.child_refs)
    nodes_by_id = {node.id: node for node in dag.nodes}
    cache: dict[int, float] = {}

    def evaluate_node(node_id: int) -> float:
        if node_id in cache:
            return cache[node_id]
        node = nodes_by_id[node_id]
        refs = sorted(refs_by_parent.get(node_id, []), key=lambda ref: ref.slot_index)
        if not refs:
            value = evaluate_eml_leaf(node, values)
        else:
            if node.kind != "eml" or node.label != "eml":
                raise ValueError(f"internal EML DAG node {node_id} must be kind='eml', label='eml'")
            if len(refs) != 2:
                raise ValueError(f"eml DAG node {node_id} must have exactly two child refs")
            left = evaluate_node(refs[0].child_id)
            right = evaluate_node(refs[1].child_id)
            value = math.exp(left) - real_log_with_formal_zero(right)
        cache[node_id] = value
        return value

    return evaluate_node(dag.root_id)


def evaluate_eml_leaf(node: DagNode, values: dict[str, float]) -> float:
    """Evaluate one pure EML leaf node."""
    if node.kind == "constant" and node.label == "1":
        return 1.0
    if node.kind == "variable":
        if node.label not in values:
            raise ValueError(f"missing numeric value for variable {node.label!r}")
        return values[node.label]
    raise ValueError(f"unsupported EML DAG leaf kind={node.kind!r}, label={node.label!r}")


def real_log_with_formal_zero(value: float) -> float:
    """Match the official pure EML evaluator's formal log(0) convention."""
    if value == 0:
        return -math.inf
    return math.log(value)


def top_shared_subtree_signatures(
    dag: DagGraph,
    *,
    limit: int,
) -> list[SharedSubtreeRecord]:
    """Return shared DAG subtree signatures ordered by source-tree reuse count."""
    incoming_counts = Counter(ref.child_id for ref in dag.child_refs)
    records: list[SharedSubtreeRecord] = []
    for node in dag.nodes:
        reuse_count = source_tree_reuse_count(node)
        incoming_count = incoming_counts[node.id]
        if reuse_count <= 1 and incoming_count <= 1:
            continue
        records.append(
            SharedSubtreeRecord(
                rank=0,
                node_id=node.id,
                label=node.label,
                kind=node.kind,
                reuse_count=reuse_count,
                incoming_child_ref_count=incoming_count,
                signature=repr(dag_structural_signature(dag, node.id)),
            )
        )

    ranked = sorted(
        records,
        key=lambda record: (
            -record.reuse_count,
            -record.incoming_child_ref_count,
            record.kind,
            record.label,
            record.node_id,
        ),
    )[:limit]
    return [record.model_copy(update={"rank": rank}) for rank, record in enumerate(ranked, start=1)]


def source_tree_reuse_count(node: DagNode) -> int:
    """Return how many tree nodes were represented by this DAG node."""
    source_ids = node.metadata.get("source_tree_node_ids", [])
    if isinstance(source_ids, list):
        return len(source_ids)
    return 1


def dag_structural_signature(dag: DagGraph, node_id: int) -> StructuralSignature:
    """Build a canonical structural signature directly from a DAG node."""
    refs_by_parent = child_refs_by_parent(dag.child_refs)
    nodes_by_id = {node.id: node for node in dag.nodes}
    cache: dict[int, StructuralSignature] = {}

    def build(current_node_id: int) -> StructuralSignature:
        if current_node_id in cache:
            return cache[current_node_id]
        node = nodes_by_id[current_node_id]
        refs = sorted(refs_by_parent.get(current_node_id, []), key=lambda ref: ref.slot_index)
        child_signatures = tuple((ref.slot_index, build(ref.child_id)) for ref in refs)
        if not child_signatures:
            signature = (
                dag.metadata.get("source_representation_mode", "unknown"),
                "leaf",
                node.kind,
                node.label,
                leaf_structural_value(node),
            )
        elif len(child_signatures) == 1:
            signature = (
                dag.metadata.get("source_representation_mode", "unknown"),
                "unary",
                node.kind,
                node.label,
                child_signatures[0],
            )
        elif len(child_signatures) == 2:
            signature = (
                dag.metadata.get("source_representation_mode", "unknown"),
                "binary",
                node.kind,
                node.label,
                child_signatures[0],
                child_signatures[1],
            )
        else:
            raise ValueError(f"n-ary DAG node {current_node_id} has {len(child_signatures)} refs")
        cache[current_node_id] = signature
        return signature

    return build(node_id)


def leaf_structural_value(node: DagNode) -> tuple[tuple[str, object], ...]:
    """Return the structural metadata portion of a leaf signature."""
    if node.kind != "constant":
        return ()
    structural_keys = ("denominator", "numerator", "value")
    return tuple(
        (key, freeze_value(node.metadata[key])) for key in structural_keys if key in node.metadata
    )


def freeze_value(value: object) -> object:
    """Freeze metadata for stable structural signatures."""
    if isinstance(value, dict):
        return tuple(sorted((str(key), freeze_value(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(freeze_value(item) for item in value)
    return value


def child_refs_by_parent(
    child_refs: Sequence[DagChildRef],
) -> dict[int, list[DagChildRef]]:
    """Group child refs by parent id."""
    refs_by_parent: dict[int, list[DagChildRef]] = defaultdict(list)
    for ref in child_refs:
        refs_by_parent[ref.parent_id].append(ref)
    return refs_by_parent


def write_audit_json(result: DagSemanticAuditResult, path: Path) -> None:
    """Write structured audit JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_audit_csv(rows: Sequence[DagSemanticAuditRow], path: Path) -> None:
    """Write flattened audit CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=AUDIT_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(audit_row_to_csv_dict(row))


def audit_row_to_csv_dict(row: DagSemanticAuditRow) -> dict[str, object]:
    """Serialize one audit row for CSV export."""
    row_dict = row.model_dump(mode="json")
    row_dict["top_shared_subtree_signatures"] = json.dumps(
        row_dict["top_shared_subtree_signatures"],
        sort_keys=True,
    )
    return {field: row_dict.get(field) for field in AUDIT_CSV_FIELDS}


def write_audit_docs(
    result: DagSemanticAuditResult,
    path: Path,
    *,
    csv_path: Path,
    json_path: Path,
) -> None:
    """Write the Goal 3.6 markdown audit report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_audit_docs_markdown(result, csv_path=csv_path, json_path=json_path),
        encoding="utf-8",
    )


def build_audit_docs_markdown(
    result: DagSemanticAuditResult,
    *,
    csv_path: Path,
    json_path: Path,
) -> str:
    """Build markdown documentation for the semantic audit."""
    invalid_rows = [
        row
        for row in result.rows
        if not row.structural_valid or not row.semantic_numeric_valid or row.error is not None
    ]
    sections = [
        "# Goal 3 DAG Semantic Audit",
        "",
        "This audit checks that exact structural DAG compression does not reintroduce "
        "hidden complexity into official pure EML representations.",
        "",
        "The audit is structural first: it verifies that DAG sharing is exact subtree "
        "sharing, not macro creation, derived leaves, parameterized templates, or algebraic "
        "simplification.",
        "",
        "## Scope",
        "",
        "- AST and EML trees are built with the existing converters.",
        "- AST and EML DAGs are built with exact structural hashing.",
        "- Numeric checks compare the original SymPy expression, the official pure EML tree, "
        "and the pure EML DAG on safe positive real inputs.",
        "- No official compiler formulas are changed by this audit.",
        "",
        "## Summary",
        "",
        f"- Expressions audited: `{result.expression_count}`",
        f"- Structurally valid EML DAGs: `{result.structural_valid_count}`",
        f"- Numerically valid EML DAGs: `{result.semantic_numeric_valid_count}`",
        f"- JSON output: `{json_path}`",
        f"- CSV output: `{csv_path}`",
        "",
        "## Audit Table",
        "",
        markdown_audit_table(result.rows),
        "",
        "## Hidden-Complexity Checks",
        "",
        "- No derived leaves are allowed.",
        "- No hidden compound leaves are allowed.",
        "- No macro or template DAG nodes are allowed.",
        "- Pure EML DAG internal nodes must be exactly `eml`.",
        "- Pure EML DAG leaves must be variables or constant `1`.",
        "- Child slots must be preserved as `left` and `right` for binary EML nodes.",
        "- Duplicate child references must remain explicit references.",
        "",
        "## Result",
        "",
    ]
    if invalid_rows:
        sections.extend(
            [
                "One or more audit rows failed. See the JSON output for full errors.",
                markdown_failure_table(invalid_rows),
            ]
        )
    else:
        sections.append(
            "All audited expressions passed structural purity and numeric semantic checks."
        )
    sections.extend(
        [
            "",
            "This remains a representation-level audit. It does not claim neural model "
            "performance improvement.",
        ]
    )
    return "\n".join(sections) + "\n"


def markdown_audit_table(rows: Sequence[DagSemanticAuditRow]) -> str:
    """Render compact audit metrics table."""
    lines = [
        "| Expression | AST tree | AST DAG | EML tree | EML DAG | Tree alpha | "
        "DAG alpha | EML DAG compression | Shared records |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.name}` | {row.ast_tree_node_count} | {row.ast_dag_node_count} | "
            f"{row.eml_tree_node_count} | {row.eml_dag_node_count} | "
            f"{format_optional_float(row.tree_alpha)} | "
            f"{format_optional_float(row.dag_alpha_vs_ast_tree)} | "
            f"{format_optional_float(row.eml_dag_compression)} | "
            f"{len(row.top_shared_subtree_signatures)} |"
        )
    return "\n".join(lines)


def markdown_failure_table(rows: Sequence[DagSemanticAuditRow]) -> str:
    """Render compact failure table."""
    lines = [
        "| Expression | Structural valid | Semantic valid | Error |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        error = row.error or "; ".join(row.structural_errors) or "semantic mismatch"
        lines.append(
            "| "
            f"`{row.name}` | {row.structural_valid} | {row.semantic_numeric_valid} | "
            f"`{truncate_for_markdown(error, 120)}` |"
        )
    return "\n".join(lines)


def format_optional_float(value: float | None) -> str:
    """Format an optional float for markdown."""
    if value is None:
        return ""
    return f"{value:.6g}"


def truncate_for_markdown(text: str, max_chars: int) -> str:
    """Truncate markdown table text without introducing newlines."""
    normalized = text.replace("\n", " ")
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="Audit JSON output path.")
    parser.add_argument("--csv", type=Path, default=None, help="Audit CSV output path.")
    parser.add_argument("--docs", type=Path, default=None, help="Audit markdown docs path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 3.6 DAG semantic audit."""
    args = build_parser().parse_args(argv)
    config = DagSemanticAuditConfig()
    if args.json is not None:
        config.json_path = args.json
    if args.csv is not None:
        config.csv_path = args.csv
    if args.docs is not None:
        config.docs_path = args.docs

    result = run_dag_semantic_audit(config)
    print(f"Audited expressions: {result.expression_count}")
    print(f"Structurally valid: {result.structural_valid_count}")
    print(f"Numerically valid: {result.semantic_numeric_valid_count}")
    print(f"JSON: {config.json_path}")
    print(f"CSV: {config.csv_path}")
    print(f"Docs: {config.docs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
