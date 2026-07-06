"""Tests for the end-to-end Goal 2 expansion pipeline."""

from __future__ import annotations

from pathlib import Path

from geml.experiments.run_goal2_expansion_pipeline import (
    Goal2ExpansionPipelineConfig,
    run_goal2_expansion_pipeline,
)


def test_goal2_pipeline_small_count_writes_outputs_and_final_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "v0"
    report_path = tmp_path / "docs" / "GOAL2_EXPANSION_STUDY.md"
    config = Goal2ExpansionPipelineConfig(
        seed=11,
        count=25,
        max_depth=2,
        output_dir=output_dir,
        input_jsonl_path=output_dir / "expansion_inputs.jsonl",
        raw_metrics_jsonl_path=output_dir / "expansion_raw_metrics.jsonl",
        raw_metrics_csv_path=output_dir / "expansion_raw_metrics.csv",
        summary_json_path=output_dir / "official_eml_compiler_summary.json",
        alpha_summary_csv_path=output_dir / "expansion_alpha_summary.csv",
        alpha_summary_json_path=output_dir / "expansion_alpha_summary.json",
        alpha_by_ast_depth_csv_path=output_dir / "alpha_by_ast_depth.csv",
        alpha_by_ast_size_bucket_csv_path=output_dir / "alpha_by_ast_size_bucket.csv",
        alpha_by_operator_family_csv_path=output_dir / "alpha_by_operator_family.csv",
        alpha_by_operator_signature_csv_path=output_dir / "alpha_by_operator_signature.csv",
        alpha_by_boolean_features_csv_path=output_dir / "alpha_by_boolean_features.csv",
        plots_dir=output_dir / "plots",
        top_20_alpha_csv_path=output_dir / "top_20_alpha_expressions.csv",
        top_20_eml_node_csv_path=output_dir / "top_20_eml_node_expressions.csv",
        top_20_eml_depth_csv_path=output_dir / "top_20_eml_depth_expressions.csv",
        top_alpha_explosions_csv_path=output_dir / "top_alpha_explosions.csv",
        top_eml_node_explosions_csv_path=output_dir / "top_eml_node_explosions.csv",
        top_eml_depth_explosions_csv_path=output_dir / "top_eml_depth_explosions.csv",
        worst_operator_signatures_csv_path=output_dir / "worst_operator_signatures.csv",
        safest_operator_signatures_csv_path=output_dir / "safest_operator_signatures.csv",
        depth_failure_modes_csv_path=output_dir / "depth_failure_modes.csv",
        safe_eml_regime_candidates_csv_path=output_dir / "safe_eml_regime_candidates.csv",
        failure_report_md_path=output_dir / "GOAL2_FAILURE_CASES.md",
        final_report_path=report_path,
        top_alpha_json_path=output_dir / "official_eml_top20_alpha.json",
        top_depth_json_path=output_dir / "official_eml_top20_depth.json",
        simple_examples_json_path=output_dir / "official_eml_simple_examples.json",
        failure_top_n=5,
        snippet_max_chars=200,
    )

    result = run_goal2_expansion_pipeline(config)

    assert result.processed_count == 25
    assert result.supported_count == 25
    assert result.unsupported_count == 0
    assert len(result.alpha_summaries) == 3
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Goal And Scientific Question" in report
    assert "Recommendation For Goal 3" in report
    assert "DAG compression" in report
    for path in result.generated_files:
        assert path.exists(), path
    assert len(list((output_dir / "plots").glob("*.png"))) == 10
