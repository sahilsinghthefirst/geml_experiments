from __future__ import annotations

import csv
import json
from pathlib import Path

from geml.experiments.goal4_egraph_audit import (
    Goal4EgraphAuditConfig,
    main,
    run_goal4_egraph_audit,
)


def test_goal4_egraph_audit_runs_and_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "v1"
    report_path = tmp_path / "docs" / "goal4" / "GOAL4_EGRAPH_AUDIT.md"
    config = Goal4EgraphAuditConfig(
        output_dir=output_dir,
        csv_path=output_dir / "goal4_egraph_audit.csv",
        json_path=output_dir / "goal4_egraph_audit.json",
        report_path=report_path,
    )

    result = run_goal4_egraph_audit(config)

    assert len(result.rows) == 28
    assert config.csv_path.exists()
    assert config.json_path.exists()
    assert config.report_path.exists()

    with config.csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    assert len(csv_rows) == 28

    payload = json.loads(config.json_path.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 28
    assert payload["summary"]["all_final_eml_outputs_pure"] is True
    assert "Goal 4.5 E-Graph Compression Audit" in config.report_path.read_text(encoding="utf-8")


def test_goal4_egraph_audit_safe_and_positive_modes_differ_as_expected(
    tmp_path: Path,
) -> None:
    payload = run_audit_payload(tmp_path)
    rows = {(row["expression"], row["rule_mode"]): row for row in payload["rows"]}

    assert rows[("log(exp(x))", "safe")]["extracted_expression"] != "x"
    assert rows[("log(exp(x))", "positive_real_formal")]["extracted_expression"] == "x"
    assert rows[("exp(log(x))", "safe")]["extracted_expression"] != "x"
    assert rows[("exp(log(x))", "positive_real_formal")]["extracted_expression"] == "x"
    assert (
        rows[("log(x*y)", "positive_real_formal")]["extracted_eml_dag_nodes"]
        < rows[("log(x*y)", "safe")]["extracted_eml_dag_nodes"]
    )


def test_goal4_egraph_audit_branch_sensitive_markers_by_mode(tmp_path: Path) -> None:
    payload = run_audit_payload(tmp_path)
    safe_rows = [row for row in payload["rows"] if row["rule_mode"] == "safe"]
    positive_rows = [row for row in payload["rows"] if row["rule_mode"] == "positive_real_formal"]

    assert safe_rows
    assert positive_rows
    assert all(row["branch_sensitive_rules_used"] is False for row in safe_rows)
    assert all(row["branch_sensitive_rule_names"] == [] for row in safe_rows)
    assert all(row["branch_sensitive_rules_used"] is True for row in positive_rows)
    assert all(row["branch_sensitive_rule_names"] for row in positive_rows)


def test_goal4_egraph_audit_no_derived_eml_leaves_after_extraction(tmp_path: Path) -> None:
    payload = run_audit_payload(tmp_path)

    assert all(row["structural_purity_valid"] is True for row in payload["rows"])


def test_goal4_egraph_audit_cli_runs_with_custom_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "v1"
    report_path = tmp_path / "docs" / "goal4" / "GOAL4_EGRAPH_AUDIT.md"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "goal4_egraph_audit.csv").exists()
    assert (output_dir / "goal4_egraph_audit.json").exists()
    assert report_path.exists()


def run_audit_payload(tmp_path: Path) -> dict[str, object]:
    output_dir = tmp_path / "outputs" / "v1"
    config = Goal4EgraphAuditConfig(
        output_dir=output_dir,
        csv_path=output_dir / "goal4_egraph_audit.csv",
        json_path=output_dir / "goal4_egraph_audit.json",
        report_path=tmp_path / "docs" / "goal4" / "GOAL4_EGRAPH_AUDIT.md",
    )
    run_goal4_egraph_audit(config)
    payload = json.loads(config.json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("audit payload must be a JSON object")
    return payload
