"""Tests for the complete Goal 3 DAG compression pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from geml.experiments.run_goal3_dag_pipeline import (
    GOAL3_PLOT_FILENAMES,
    Goal3DagPipelineConfig,
    run_goal3_dag_pipeline,
)


def test_goal3_dag_pipeline_small_end_to_end(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "v0"
    docs_dir = tmp_path / "docs"
    config = Goal3DagPipelineConfig(
        count=25,
        seed=0,
        max_depth=3,
        output_dir=output_dir,
        input_jsonl_path=output_dir / "dag_compression_inputs.jsonl",
        metrics_jsonl_path=output_dir / "dag_compression_metrics.jsonl",
        metrics_csv_path=output_dir / "dag_compression_metrics.csv",
        summary_json_path=output_dir / "dag_compression_summary.json",
        dag_alpha_threshold_summary_csv_path=output_dir / "dag_alpha_threshold_summary.csv",
        dag_alpha_threshold_summary_json_path=output_dir / "dag_alpha_threshold_summary.json",
        dag_alpha_by_ast_size_bucket_csv_path=output_dir / "dag_alpha_by_ast_size_bucket.csv",
        dag_alpha_by_ast_depth_csv_path=output_dir / "dag_alpha_by_ast_depth.csv",
        dag_alpha_by_operator_family_csv_path=output_dir / "dag_alpha_by_operator_family.csv",
        dag_alpha_by_operator_signature_csv_path=(
            output_dir / "dag_alpha_by_operator_signature.csv"
        ),
        dag_alpha_by_boolean_features_csv_path=output_dir / "dag_alpha_by_boolean_features.csv",
        plots_dir=output_dir / "plots_goal3",
        top_successes_csv_path=output_dir / "top_dag_compression_successes.csv",
        top_failures_csv_path=output_dir / "top_dag_compression_failures.csv",
        best_operator_signatures_csv_path=output_dir / "best_dag_operator_signatures.csv",
        worst_operator_signatures_csv_path=output_dir / "worst_dag_operator_signatures.csv",
        safe_regime_candidates_csv_path=output_dir / "dag_safe_regime_candidates.csv",
        findings_report_md_path=output_dir / "GOAL3_DAG_COMPRESSION_FINDINGS.md",
        semantic_audit_json_path=output_dir / "goal3_dag_semantic_audit.json",
        semantic_audit_csv_path=output_dir / "goal3_dag_semantic_audit.csv",
        semantic_audit_docs_path=docs_dir / "GOAL3_DAG_SEMANTIC_AUDIT.md",
        final_report_path=docs_dir / "GOAL3_DAG_COMPRESSION_STUDY.md",
        summary_doc_path=docs_dir / "GOAL3_SUMMARY.md",
        mining_top_n=5,
    )

    result = run_goal3_dag_pipeline(config)

    assert result.processed_count == 25
    assert result.supported_count == 25
    assert result.unsupported_count == 0
    assert result.final_report_path.exists()
    assert result.summary_doc_path.exists()
    assert config.semantic_audit_json_path.exists()
    assert config.semantic_audit_csv_path.exists()
    assert config.semantic_audit_docs_path.exists()
    assert {path.name for path in result.plot_result.plot_paths} == set(GOAL3_PLOT_FILENAMES)
    for output_path in result.generated_files:
        assert output_path.exists()

    final_report = config.final_report_path.read_text(encoding="utf-8")
    assert "Goal 3 Question" in final_report
    assert "Recommendation For Goal 4" in final_report
    assert "Do not introduce equivalence-pair generation or neural models" in final_report

    audit_payload = json.loads(config.semantic_audit_json_path.read_text(encoding="utf-8"))
    assert audit_payload["expression_count"] == 12
    assert audit_payload["structural_valid_count"] == 12
    assert audit_payload["semantic_numeric_valid_count"] == 12
