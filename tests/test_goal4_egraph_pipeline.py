"""Tests for the complete Goal 4 non-ML e-graph pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from geml.experiments.egraph_compression_study import EgraphCompressionStudyConfig
from geml.experiments.plot_egraph_compression import GOAL4_PLOT_FILENAMES
from geml.experiments.run_goal4_egraph_pipeline import (
    Goal4EgraphPipelineConfig,
    load_pipeline_config,
    run_goal4_egraph_pipeline,
)


def test_goal4_egraph_pipeline_small_end_to_end(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "v1"
    docs_dir = tmp_path / "docs" / "goal4"
    egraph_config = EgraphCompressionStudyConfig(
        count=25,
        output_dir=output_dir,
        safe_metrics_csv_path=output_dir / "egraph_compression_metrics_safe.csv",
        safe_metrics_jsonl_path=output_dir / "egraph_compression_metrics_safe.jsonl",
        positive_real_metrics_csv_path=(
            output_dir / "egraph_compression_metrics_positive_real.csv"
        ),
        positive_real_metrics_jsonl_path=(
            output_dir / "egraph_compression_metrics_positive_real.jsonl"
        ),
        summary_json_path=output_dir / "egraph_compression_summary.json",
        run_metadata_json_path=output_dir / "egraph_compression_run_metadata.json",
        timeout_seconds=0.25,
        beam_size=8,
        max_candidate_depth=7,
        max_candidates_evaluated=8,
        checkpoint_interval=10,
        resume=False,
    )
    config = Goal4EgraphPipelineConfig(
        egraph_config=egraph_config,
        expression_generation_summary_json_path=None,
        final_report_path=docs_dir / "GOAL4_NONML_COMPRESSION_STUDY.md",
        summary_doc_path=docs_dir / "GOAL4_SUMMARY.md",
        semantic_audit_json_path=output_dir / "goal4_egraph_semantic_audit.json",
        semantic_audit_csv_path=output_dir / "goal4_egraph_semantic_audit.csv",
        semantic_audit_docs_path=docs_dir / "GOAL4_EGRAPH_SEMANTIC_AUDIT.md",
        mining_top_n=5,
    )

    result = run_goal4_egraph_pipeline(config)

    assert result.processed_count_by_mode == {"safe": 25, "positive_real_formal": 25}
    assert result.success_count_by_mode["safe"] > 0
    assert result.success_count_by_mode["positive_real_formal"] > 0
    assert result.final_report_path.exists()
    assert result.summary_doc_path.exists()
    assert config.semantic_audit_json_path.exists()
    assert config.semantic_audit_csv_path.exists()
    assert config.semantic_audit_docs_path.exists()
    assert {path.name for path in result.plot_result.plot_paths} == set(GOAL4_PLOT_FILENAMES)
    for generated_path in result.generated_files:
        assert generated_path.exists()

    output_files = [path for path in result.generated_files if "outputs" in path.parts]
    assert output_files
    assert all("outputs/v1" in path.as_posix() for path in output_files)

    safe_rows = read_jsonl(config.egraph_config.safe_metrics_jsonl_path)
    positive_rows = read_jsonl(config.egraph_config.positive_real_metrics_jsonl_path)
    assert len(safe_rows) == 25
    assert len(positive_rows) == 25
    assert {row["index"] for row in safe_rows} == set(range(25))
    assert {row["index"] for row in positive_rows} == set(range(25))

    summary = json.loads(config.egraph_config.summary_json_path.read_text(encoding="utf-8"))
    assert summary["rule_modes"]["safe"]["processed_count"] == 25
    assert summary["rule_modes"]["positive_real_formal"]["processed_count"] == 25

    audit_payload = json.loads(config.semantic_audit_json_path.read_text(encoding="utf-8"))
    assert audit_payload["summary"]["all_structural_purity_valid"] is True
    assert audit_payload["summary"]["safe_branch_sensitive_application_count"] == 0

    final_report = config.final_report_path.read_text(encoding="utf-8")
    assert "Goal 4 Question" in final_report
    assert "positive_real_formal" in final_report
    assert "not GNN or neural-model evidence" in final_report

    forbidden_file_terms = ("gnn", "neural", "/models/")
    assert not any(
        any(term in path.as_posix().lower() for term in forbidden_file_terms)
        for path in result.generated_files
    )


def test_goal4_pipeline_loads_v1_yaml_config() -> None:
    config = load_pipeline_config(Path("configs/egraph_compression_v1.yaml"))

    assert config.egraph_config.count == 10_000
    assert config.egraph_config.output_dir.as_posix() == "outputs/v1"
    assert set(config.egraph_config.run_modes) == {"safe", "positive_real_formal"}
    assert config.final_report_path.as_posix() == "docs/goal4/GOAL4_NONML_COMPRESSION_STUDY.md"
    assert config.summary_doc_path.as_posix() == "docs/goal4/GOAL4_SUMMARY.md"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw_row = json.loads(line)
            if not isinstance(raw_row, dict):
                raise TypeError("JSONL row must be an object")
            rows.append(raw_row)
    return rows
