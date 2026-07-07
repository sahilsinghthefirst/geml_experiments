"""Tests for Goal 4.9 e-graph semantic/provenance audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.egraph.costs import exact_eml_dag_cost
from geml.egraph.ir import Mul, Var
from geml.experiments.egraph_semantic_audit import (
    EgraphSemanticAuditConfig,
    audit_pure_eml_structural_integrity,
    main,
    run_egraph_semantic_audit,
)


def test_egraph_semantic_audit_runs_and_writes_v1_outputs(tmp_path: Path) -> None:
    config = audit_config(tmp_path)

    result = run_egraph_semantic_audit(config)

    assert len(result.rows) == 28
    assert config.json_path.exists()
    assert config.csv_path.exists()
    assert config.report_path.exists()
    assert "outputs/v1" in config.json_path.as_posix()
    assert "outputs/v1" in config.csv_path.as_posix()

    with config.csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    assert len(csv_rows) == 28

    payload = json.loads(config.json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["all_structural_purity_valid"] is True
    assert payload["summary"]["all_semantic_validation_valid"] is True
    assert payload["summary"]["all_eml_dag_validation_valid"] is True


def test_log_exp_requires_positive_real_branch_sensitive_provenance(tmp_path: Path) -> None:
    payload = run_audit_payload(tmp_path)
    rows = row_lookup(payload)

    safe = rows[("log(exp(x))", "safe")]
    positive = rows[("log(exp(x))", "positive_real_formal")]

    assert safe["extracted_expression"] != "x"
    assert safe["branch_sensitive_rules_applied"] is False
    assert safe["branch_sensitive_rule_names_applied"] == []
    assert safe["provenance_validation_status"] == "valid"

    assert positive["extracted_expression"] == "x"
    assert positive["branch_sensitive_rules_applied"] is True
    assert "log_exp_inverse" in positive["branch_sensitive_rule_names_applied"]
    assert positive["rules_applied_by_name"]["log_exp_inverse"] > 0


def test_exp_log_requires_positive_real_branch_sensitive_provenance(tmp_path: Path) -> None:
    payload = run_audit_payload(tmp_path)
    rows = row_lookup(payload)

    safe = rows[("exp(log(x))", "safe")]
    positive = rows[("exp(log(x))", "positive_real_formal")]

    assert safe["extracted_expression"] != "x"
    assert safe["branch_sensitive_rules_applied"] is False
    assert safe["provenance_validation_status"] == "valid"

    assert positive["extracted_expression"] == "x"
    assert positive["branch_sensitive_rules_applied"] is True
    assert "exp_log_inverse" in positive["branch_sensitive_rule_names_applied"]
    assert positive["rules_applied_by_name"]["exp_log_inverse"] > 0


def test_x_plus_two_minus_one_compresses_under_safe_rules(tmp_path: Path) -> None:
    payload = run_audit_payload(tmp_path)
    row = row_lookup(payload)[("x+2-1", "safe")]

    assert row["extracted_eml_dag_nodes"] < row["original_eml_dag_nodes"]
    assert row["compression_gain"] > 1.0
    assert row["branch_sensitive_rules_applied"] is False
    assert "sub_lowering" in row["rules_applied_by_name"]
    assert row["candidate_eml_dag_metrics"]
    assert row["selected_candidate_rank"] is not None


def test_structural_purity_catches_injected_fake_derived_leaf() -> None:
    cost = exact_eml_dag_cost(Var("x"))
    bad_nodes = [
        node.model_copy(
            update={
                "kind": "derived",
                "label": "log(expr)",
                "metadata": {"contains_hidden_compound": True},
            }
        )
        if node.id == cost.eml_dag.root_id
        else node
        for node in cost.eml_dag.nodes
    ]
    bad_dag = cost.eml_dag.model_copy(
        update={
            "nodes": bad_nodes,
            "node_kinds": {node.id: node.kind for node in bad_nodes},
            "node_labels": {node.id: node.label for node in bad_nodes},
        }
    )

    result = audit_pure_eml_structural_integrity(cost.eml_tree, bad_dag)

    assert result.valid is False
    assert any("derived" in error for error in result.errors)
    assert any("hidden compound" in error for error in result.errors)


def test_eml_dag_evaluator_and_sympy_evaluator_agree_on_positive_probes(
    tmp_path: Path,
) -> None:
    payload = run_audit_payload(tmp_path)
    row = row_lookup(payload)[("x*x", "safe")]

    assert row["validation_status"] == "valid"
    assert row["eml_dag_validation_status"] == "valid"
    assert row["eml_dag_max_abs_error"] < 1e-8

    direct_cost = exact_eml_dag_cost(Mul(Var("x"), Var("x")))
    direct_result = audit_pure_eml_structural_integrity(direct_cost.eml_tree, direct_cost.eml_dag)
    assert direct_result.valid is True


def test_egraph_semantic_audit_cli_runs_with_custom_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "v1"
    report_path = tmp_path / "docs" / "goal4" / "GOAL4_EGRAPH_SEMANTIC_AUDIT.md"

    exit_code = main(["--output-dir", str(output_dir), "--report", str(report_path)])

    assert exit_code == 0
    assert (output_dir / "goal4_egraph_semantic_audit.json").exists()
    assert (output_dir / "goal4_egraph_semantic_audit.csv").exists()
    assert report_path.exists()


def audit_config(tmp_path: Path) -> EgraphSemanticAuditConfig:
    """Build a temp v1 audit config."""
    output_dir = tmp_path / "outputs" / "v1"
    return EgraphSemanticAuditConfig(
        output_dir=output_dir,
        json_path=output_dir / "goal4_egraph_semantic_audit.json",
        csv_path=output_dir / "goal4_egraph_semantic_audit.csv",
        report_path=tmp_path / "docs" / "goal4" / "GOAL4_EGRAPH_SEMANTIC_AUDIT.md",
    )


def run_audit_payload(tmp_path: Path) -> dict[str, object]:
    """Run the audit and return its JSON payload."""
    config = audit_config(tmp_path)
    run_egraph_semantic_audit(config)
    payload = json.loads(config.json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("audit payload must be a JSON object")
    return payload


def row_lookup(payload: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    """Return audit rows keyed by expression and mode."""
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise TypeError("payload rows must be a list")
    return {
        (str(row["original_expression"]), str(row["rule_mode"])): row
        for row in rows
        if isinstance(row, dict)
    }
