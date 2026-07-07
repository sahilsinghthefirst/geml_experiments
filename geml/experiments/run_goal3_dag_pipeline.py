"""End-to-end Goal 3 DAG compression pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator

from geml.experiments.dag_compression_mining import (
    DagCompressionMiningConfig,
    DagCompressionMiningResult,
    run_dag_compression_mining,
)
from geml.experiments.dag_compression_study import (
    DagCompressionStudyConfig,
    run_dag_compression_study,
)
from geml.experiments.dag_semantic_audit import (
    SEMANTIC_TOLERANCE,
    DagSemanticAuditConfig,
    DagSemanticAuditResult,
    run_dag_semantic_audit,
)
from geml.experiments.plot_dag_compression import (
    GOAL3_PLOT_FILENAMES,
    DagCompressionPlotConfig,
    DagCompressionPlotResult,
    run_dag_compression_plots,
)
from geml.experiments.stratified_dag_compression import (
    StratifiedDagCompressionConfig,
    StratifiedDagCompressionResult,
    run_stratified_dag_compression_analysis,
)


class Goal3DagPipelineConfig(DagCompressionStudyConfig):
    """Configuration for the complete Goal 3 DAG compression pipeline."""

    dag_alpha_threshold_summary_csv_path: Path = Path("outputs/v0/dag_alpha_threshold_summary.csv")
    dag_alpha_threshold_summary_json_path: Path = Path(
        "outputs/v0/dag_alpha_threshold_summary.json"
    )
    dag_alpha_by_ast_size_bucket_csv_path: Path = Path(
        "outputs/v0/dag_alpha_by_ast_size_bucket.csv"
    )
    dag_alpha_by_ast_depth_csv_path: Path = Path("outputs/v0/dag_alpha_by_ast_depth.csv")
    dag_alpha_by_operator_family_csv_path: Path = Path(
        "outputs/v0/dag_alpha_by_operator_family.csv"
    )
    dag_alpha_by_operator_signature_csv_path: Path = Path(
        "outputs/v0/dag_alpha_by_operator_signature.csv"
    )
    dag_alpha_by_boolean_features_csv_path: Path = Path(
        "outputs/v0/dag_alpha_by_boolean_features.csv"
    )
    plots_dir: Path = Path("outputs/v0/plots_goal3")
    top_successes_csv_path: Path = Path("outputs/v0/top_dag_compression_successes.csv")
    top_failures_csv_path: Path = Path("outputs/v0/top_dag_compression_failures.csv")
    best_operator_signatures_csv_path: Path = Path("outputs/v0/best_dag_operator_signatures.csv")
    worst_operator_signatures_csv_path: Path = Path("outputs/v0/worst_dag_operator_signatures.csv")
    safe_regime_candidates_csv_path: Path = Path("outputs/v0/dag_safe_regime_candidates.csv")
    findings_report_md_path: Path = Path("docs/goal3/GOAL3_DAG_COMPRESSION_FINDINGS.md")
    semantic_audit_json_path: Path = Path("outputs/v0/goal3_dag_semantic_audit.json")
    semantic_audit_csv_path: Path = Path("outputs/v0/goal3_dag_semantic_audit.csv")
    semantic_audit_docs_path: Path = Path("docs/goal3/GOAL3_DAG_SEMANTIC_AUDIT.md")
    final_report_path: Path = Path("docs/goal3/GOAL3_DAG_COMPRESSION_STUDY.md")
    summary_doc_path: Path = Path("docs/goal3/GOAL3_SUMMARY.md")
    mining_top_n: int = Field(default=20, gt=0)
    semantic_top_shared_limit: int = Field(default=10, gt=0)
    semantic_tolerance: float = Field(default=SEMANTIC_TOLERANCE, gt=0)

    @model_validator(mode="after")
    def validate_goal3_paths(self) -> Self:
        output_paths = {
            self.dag_alpha_threshold_summary_csv_path,
            self.dag_alpha_threshold_summary_json_path,
            self.dag_alpha_by_ast_size_bucket_csv_path,
            self.dag_alpha_by_ast_depth_csv_path,
            self.dag_alpha_by_operator_family_csv_path,
            self.dag_alpha_by_operator_signature_csv_path,
            self.dag_alpha_by_boolean_features_csv_path,
            self.top_successes_csv_path,
            self.top_failures_csv_path,
            self.best_operator_signatures_csv_path,
            self.worst_operator_signatures_csv_path,
            self.safe_regime_candidates_csv_path,
            self.findings_report_md_path,
            self.semantic_audit_json_path,
            self.semantic_audit_csv_path,
            self.final_report_path,
            self.summary_doc_path,
        }
        input_paths = {self.input_jsonl_path, self.metrics_jsonl_path, self.metrics_csv_path}
        overlap = input_paths & output_paths
        if overlap:
            overlap_text = ", ".join(str(path) for path in sorted(overlap))
            raise ValueError(f"Goal 3 input and output paths must differ: {overlap_text}")
        return self

    def to_stratified_config(self) -> StratifiedDagCompressionConfig:
        """Build Goal 3.4 stratified config from the shared pipeline config."""
        return StratifiedDagCompressionConfig(
            dag_metrics_csv_path=self.metrics_csv_path,
            dag_summary_json_path=self.summary_json_path,
            dag_alpha_threshold_summary_csv_path=self.dag_alpha_threshold_summary_csv_path,
            dag_alpha_threshold_summary_json_path=self.dag_alpha_threshold_summary_json_path,
            dag_alpha_by_ast_size_bucket_csv_path=self.dag_alpha_by_ast_size_bucket_csv_path,
            dag_alpha_by_ast_depth_csv_path=self.dag_alpha_by_ast_depth_csv_path,
            dag_alpha_by_operator_family_csv_path=self.dag_alpha_by_operator_family_csv_path,
            dag_alpha_by_operator_signature_csv_path=(
                self.dag_alpha_by_operator_signature_csv_path
            ),
            dag_alpha_by_boolean_features_csv_path=self.dag_alpha_by_boolean_features_csv_path,
        )

    def to_plot_config(self) -> DagCompressionPlotConfig:
        """Build Goal 3.5 plotting config from the shared pipeline config."""
        return DagCompressionPlotConfig(
            dag_metrics_csv_path=self.metrics_csv_path,
            dag_threshold_summary_json_path=self.dag_alpha_threshold_summary_json_path,
            dag_operator_family_csv_path=self.dag_alpha_by_operator_family_csv_path,
            dag_ast_size_bucket_csv_path=self.dag_alpha_by_ast_size_bucket_csv_path,
            plots_dir=self.plots_dir,
        )

    def to_mining_config(self) -> DagCompressionMiningConfig:
        """Build Goal 3.5 mining config from the shared pipeline config."""
        return DagCompressionMiningConfig(
            dag_metrics_csv_path=self.metrics_csv_path,
            dag_operator_signature_csv_path=self.dag_alpha_by_operator_signature_csv_path,
            dag_threshold_summary_json_path=self.dag_alpha_threshold_summary_json_path,
            top_successes_csv_path=self.top_successes_csv_path,
            top_failures_csv_path=self.top_failures_csv_path,
            best_operator_signatures_csv_path=self.best_operator_signatures_csv_path,
            worst_operator_signatures_csv_path=self.worst_operator_signatures_csv_path,
            safe_regime_candidates_csv_path=self.safe_regime_candidates_csv_path,
            report_md_path=self.findings_report_md_path,
            top_n=self.mining_top_n,
        )

    def to_audit_config(self) -> DagSemanticAuditConfig:
        """Build Goal 3.6 semantic audit config from the shared pipeline config."""
        return DagSemanticAuditConfig(
            json_path=self.semantic_audit_json_path,
            csv_path=self.semantic_audit_csv_path,
            docs_path=self.semantic_audit_docs_path,
            top_shared_limit=self.semantic_top_shared_limit,
            semantic_tolerance=self.semantic_tolerance,
        )


@dataclass(frozen=True)
class Goal3DagPipelineResult:
    """Result summary from the complete Goal 3 DAG compression pipeline."""

    processed_count: int
    supported_count: int
    unsupported_count: int
    mean_tree_alpha: float | None
    median_tree_alpha: float | None
    p90_tree_alpha: float | None
    mean_dag_alpha_vs_ast_tree: float | None
    median_dag_alpha_vs_ast_tree: float | None
    p90_dag_alpha_vs_ast_tree: float | None
    mean_dag_alpha_vs_ast_dag: float | None
    median_dag_alpha_vs_ast_dag: float | None
    p90_dag_alpha_vs_ast_dag: float | None
    mean_eml_dag_compression: float | None
    median_eml_dag_compression: float | None
    p90_eml_dag_compression: float | None
    percent_below_threshold_before_dag: float | None
    percent_below_threshold_after_dag_vs_ast_tree: float | None
    percent_below_threshold_after_dag_vs_ast_dag: float | None
    top_compression_success_family: str | None
    top_compression_failure_family: str | None
    stratified_result: StratifiedDagCompressionResult
    plot_result: DagCompressionPlotResult
    mining_result: DagCompressionMiningResult
    semantic_audit_result: DagSemanticAuditResult
    generated_files: tuple[Path, ...]
    final_report_path: Path
    summary_doc_path: Path


def run_goal3_dag_pipeline(config: Goal3DagPipelineConfig) -> Goal3DagPipelineResult:
    """Run the complete Goal 3 DAG compression study in dependency order."""
    run_dag_compression_study(config)
    stratified_result = run_stratified_dag_compression_analysis(config.to_stratified_config())
    plot_result = run_dag_compression_plots(config.to_plot_config())
    mining_result = run_dag_compression_mining(config.to_mining_config())
    semantic_audit_result = run_dag_semantic_audit(config.to_audit_config())

    final_report = build_final_goal3_report(config)
    summary_doc = build_goal3_summary_doc(config)
    write_text(config.final_report_path, final_report)
    write_text(config.summary_doc_path, summary_doc)

    summary = load_json_object(config.summary_json_path)
    best_signatures = load_csv_rows(config.best_operator_signatures_csv_path)
    worst_signatures = load_csv_rows(config.worst_operator_signatures_csv_path)
    generated_files = tuple(
        dict.fromkeys(
            [
                config.input_jsonl_path,
                config.metrics_jsonl_path,
                config.metrics_csv_path,
                config.summary_json_path,
                *stratified_result.output_paths,
                *plot_result.plot_paths,
                *mining_result.output_paths,
                config.semantic_audit_json_path,
                config.semantic_audit_csv_path,
                config.semantic_audit_docs_path,
                config.final_report_path,
                config.summary_doc_path,
            ]
        )
    )

    return Goal3DagPipelineResult(
        processed_count=int(summary["processed_count"]),
        supported_count=int(summary["supported_count"]),
        unsupported_count=int(summary["unsupported_count"]),
        mean_tree_alpha=optional_float(summary.get("mean_tree_alpha")),
        median_tree_alpha=optional_float(summary.get("median_tree_alpha")),
        p90_tree_alpha=optional_float(summary.get("p90_tree_alpha")),
        mean_dag_alpha_vs_ast_tree=optional_float(summary.get("mean_dag_alpha_vs_ast_tree")),
        median_dag_alpha_vs_ast_tree=optional_float(summary.get("median_dag_alpha_vs_ast_tree")),
        p90_dag_alpha_vs_ast_tree=optional_float(summary.get("p90_dag_alpha_vs_ast_tree")),
        mean_dag_alpha_vs_ast_dag=optional_float(summary.get("mean_dag_alpha_vs_ast_dag")),
        median_dag_alpha_vs_ast_dag=optional_float(summary.get("median_dag_alpha_vs_ast_dag")),
        p90_dag_alpha_vs_ast_dag=optional_float(summary.get("p90_dag_alpha_vs_ast_dag")),
        mean_eml_dag_compression=optional_float(summary.get("mean_eml_dag_compression")),
        median_eml_dag_compression=optional_float(summary.get("median_eml_dag_compression")),
        p90_eml_dag_compression=optional_float(summary.get("p90_eml_dag_compression")),
        percent_below_threshold_before_dag=optional_float(
            summary.get("percent_below_threshold_tree_alpha")
        ),
        percent_below_threshold_after_dag_vs_ast_tree=optional_float(
            summary.get("percent_below_threshold_dag_alpha_vs_ast_tree")
        ),
        percent_below_threshold_after_dag_vs_ast_dag=optional_float(
            summary.get("percent_below_threshold_dag_alpha_vs_ast_dag")
        ),
        top_compression_success_family=best_signatures[0]["operator_signature"]
        if best_signatures
        else None,
        top_compression_failure_family=worst_signatures[0]["operator_signature"]
        if worst_signatures
        else None,
        stratified_result=stratified_result,
        plot_result=plot_result,
        mining_result=mining_result,
        semantic_audit_result=semantic_audit_result,
        generated_files=generated_files,
        final_report_path=config.final_report_path,
        summary_doc_path=config.summary_doc_path,
    )


def build_final_goal3_report(config: Goal3DagPipelineConfig) -> str:
    """Build the final Goal 3 DAG compression study report from saved artifacts."""
    summary = load_json_object(config.summary_json_path)
    threshold_rows = load_json_list(config.dag_alpha_threshold_summary_json_path)
    operator_family_rows = load_csv_rows(config.dag_alpha_by_operator_family_csv_path)
    ast_size_rows = load_csv_rows(config.dag_alpha_by_ast_size_bucket_csv_path)
    boolean_rows = load_csv_rows(config.dag_alpha_by_boolean_features_csv_path)
    best_signatures = load_csv_rows(config.best_operator_signatures_csv_path)
    worst_signatures = load_csv_rows(config.worst_operator_signatures_csv_path)
    safe_candidates = load_csv_rows(config.safe_regime_candidates_csv_path)
    top_successes = load_csv_rows(config.top_successes_csv_path)
    top_failures = load_csv_rows(config.top_failures_csv_path)
    semantic_audit = load_json_object(config.semantic_audit_json_path)
    current_threshold = find_row(threshold_rows, "scenario", "current_grammar")
    top_success_family = best_signatures[0]["operator_signature"]
    top_failure_family = worst_signatures[0]["operator_signature"]
    strongest_compression_family = max(
        operator_family_rows,
        key=lambda row: parse_float(row["median_eml_dag_compression"]),
    )
    highest_dag_alpha_family = max(
        operator_family_rows,
        key=lambda row: parse_float(row["median_dag_alpha_vs_ast_tree"]),
    )
    conclusion = build_rescue_conclusion(summary)

    sections = [
        "# Goal 3 DAG Compression Study",
        "",
        "## Goal 3 Question",
        "",
        "Goal 3 asks whether exact structural DAG compression can make official pure EML "
        "structurally competitive with AST representations after Goal 2 showed raw EML "
        "trees are representation-complete but expensive.",
        "",
        "The comparison is structural only. It does not make a neural model performance claim.",
        "",
        "## Relation To Goal 2",
        "",
        "Goal 2 measured raw official pure EML trees on the fixed-seed 10k expression "
        "distribution. It found every expression could compile to pure EML, but tree alpha "
        "was far above threshold. Goal 3 keeps the same distribution first so the DAG "
        "result is directly comparable.",
        "",
        f"- processed expressions: `{summary['processed_count']}`",
        f"- supported expressions: `{summary['supported_count']}`",
        f"- unsupported expressions: `{summary['unsupported_count']}`",
        "",
        "## Exact Structural DAG Definition",
        "",
        "A Goal 3 DAG node represents one unique structural subtree. Two tree subtrees may "
        "share a DAG node only when their full canonical structural signatures are identical.",
        "",
        "Allowed sharing:",
        "",
        "- identical leaf signatures: kind plus label/value",
        "- identical unary signatures: kind, label, and child signature",
        "- identical binary signatures: kind, label, ordered left/right child signatures",
        "- repeated child references such as `EML(a, a)`, with both references kept explicit",
        "",
        "Forbidden sharing:",
        "",
        "- derived leaves",
        "- hidden compound-expression leaves",
        "- macro or template nodes",
        "- parameterized macro sharing",
        "- algebraic simplification for compression",
        "- pattern sharing with holes such as `EML(1, z)`",
        "- treating `x + y` and `y + x` as identical unless upstream AST normalization already "
        "made them structurally identical",
        "- treating `x*x` and `x**2` as identical unless the source converter represents them "
        "identically",
        "",
        "## AST DAG vs EML DAG",
        "",
        "AST DAG compression shares repeated source AST subtrees. EML DAG compression shares "
        "repeated official pure EML subtrees after macro expansion. EML has more repeated "
        "structural material, especially the constant `1` and repeated macro-expansion "
        "subtrees, so EML DAG compression is real. The key question is whether that sharing "
        "is enough to cross alpha thresholds.",
        "",
        "## Aggregate Metrics",
        "",
        markdown_aggregate_table(summary),
        "",
        "Tree alpha falls substantially after EML DAG sharing, but the DAG alpha is still "
        "measured both against the AST tree and AST DAG baselines:",
        "",
        "- `dag_alpha_vs_ast_tree = D_EML_nodes / T_AST_nodes`",
        "- `dag_alpha_vs_ast_dag = D_EML_nodes / D_AST_nodes`",
        "",
        "## Threshold Scenarios",
        "",
        markdown_threshold_table(threshold_rows),
        "",
        "Current-grammar threshold result:",
        "",
        f"- tree alpha below threshold: `{current_threshold['percent_below_tree_alpha']}%`",
        "- DAG alpha vs AST tree below threshold: "
        f"`{current_threshold['percent_below_dag_alpha_vs_ast_tree']}%`",
        "- DAG alpha vs AST DAG below threshold: "
        f"`{current_threshold['percent_below_dag_alpha_vs_ast_dag']}%`",
        "",
        "## Stratified Findings",
        "",
        "- strongest median EML DAG compression family: "
        f"`{strongest_compression_family['dominant_operator_family']}` with median "
        f"compression `{strongest_compression_family['median_eml_dag_compression']}`",
        "- highest median DAG-alpha family: "
        f"`{highest_dag_alpha_family['dominant_operator_family']}` with median "
        f"DAG alpha `{highest_dag_alpha_family['median_dag_alpha_vs_ast_tree']}`",
        f"- top compression-success signature: `{top_success_family}`",
        f"- top compression-failure signature: `{top_failure_family}`",
        "",
        "AST-size bucket summary:",
        "",
        markdown_ast_size_table(ast_size_rows),
        "",
        "Operator-family summary:",
        "",
        markdown_operator_family_table(operator_family_rows),
        "",
        "Selected boolean-feature summary:",
        "",
        markdown_boolean_feature_table(boolean_rows),
        "",
        "## Plots",
        "",
        *[f"- `{config.plots_dir / filename}`" for filename in GOAL3_PLOT_FILENAMES],
        "",
        "## Success And Failure Cases",
        "",
        "Top compression successes are ranked by high EML DAG compression and large drop "
        "from tree alpha to DAG alpha. Top failures are ranked by weak compression or high "
        "remaining DAG alpha.",
        "",
        "Top success examples:",
        "",
        markdown_expression_rank_table(top_successes[:5], score_field="success_score"),
        "",
        "Top failure examples:",
        "",
        markdown_expression_rank_table(top_failures[:5], score_field="failure_score"),
        "",
        "Best operator-signature compression groups:",
        "",
        markdown_signature_table(best_signatures[:5]),
        "",
        "Worst remaining DAG-alpha groups:",
        "",
        markdown_signature_table(worst_signatures[:5]),
        "",
        "Candidate safe regimes:",
        "",
        markdown_safe_candidate_table(safe_candidates[:5]),
        "",
        "## Semantic Audit Results",
        "",
        f"- audit expressions: `{semantic_audit['expression_count']}`",
        f"- structurally valid EML DAGs: `{semantic_audit['structural_valid_count']}`",
        f"- numerically valid EML DAGs: `{semantic_audit['semantic_numeric_valid_count']}`",
        f"- audit JSON: `{config.semantic_audit_json_path}`",
        f"- audit CSV: `{config.semantic_audit_csv_path}`",
        f"- audit docs: `{config.semantic_audit_docs_path}`",
        "",
        "The audit verifies no derived leaves, hidden compound leaves, macro/template nodes, "
        "unsupported final EML labels, invalid child slots, or collapsed duplicate child "
        "references. It also compares original SymPy, EML tree, and EML DAG numeric values "
        "on safe positive real inputs.",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "DAG compression helps, but under the current fixed-seed distribution it does not "
        "rescue raw official pure EML structurally as a general representation. EML DAGs are "
        "much smaller than raw EML trees, yet the median DAG alpha remains above the current "
        "threshold and only a small slice of operator families crosses it.",
        "",
        "## Recommendation For Goal 4",
        "",
        "Goal 4 should move from size-only analysis to fair graph-representation baselines: "
        "AST-tree/AST-DAG baselines, EML-DAG baselines, and eventually AST-GNN versus "
        "EML-DAG-GNN comparisons. The EML side must keep the Goal 3 contract: exact "
        "structural DAG sharing only, no hidden macro nodes, no derived leaves, and no "
        "algebraic simplification used as compression.",
        "",
        "Do not introduce equivalence-pair generation or neural models until the graph "
        "baseline task is explicitly scoped.",
        "",
        "## Reproducible Commands",
        "",
        "```bash",
        ".venv/bin/python -m geml.experiments.run_goal3_dag_pipeline "
        "--config configs/dag_compression_v0.yaml",
        ".venv/bin/python -m pytest",
        ".venv/bin/python -m ruff check .",
        ".venv/bin/python -m ruff format . --check",
        "```",
    ]
    return "\n".join(sections) + "\n"


def build_goal3_summary_doc(config: Goal3DagPipelineConfig) -> str:
    """Build a concise Goal 3 summary from saved pipeline artifacts."""
    summary = load_json_object(config.summary_json_path)
    threshold_rows = load_json_list(config.dag_alpha_threshold_summary_json_path)
    best_signatures = load_csv_rows(config.best_operator_signatures_csv_path)
    worst_signatures = load_csv_rows(config.worst_operator_signatures_csv_path)
    semantic_audit = load_json_object(config.semantic_audit_json_path)
    current_threshold = find_row(threshold_rows, "scenario", "current_grammar")

    sections = [
        "# Goal 3 Summary",
        "",
        "Goal 3 implemented exact structural DAG compression for AST and official pure EML "
        "trees, measured it on the fixed-seed Goal 2 distribution, mined where it helps and "
        "fails, and audited that compression does not hide complexity.",
        "",
        "## Main Result",
        "",
        "- raw tree alpha remains the Goal 2 baseline",
        "- EML DAG compression substantially reduces pure EML size",
        "- current-threshold pass rate improves but remains low",
        "- no derived leaves, hidden compound leaves, macro nodes, or semantic simplification "
        "are used",
        "",
        "## Headline Numbers",
        "",
        f"- processed: `{summary['processed_count']}`",
        f"- supported: `{summary['supported_count']}`",
        f"- mean tree alpha: `{summary['mean_tree_alpha']}`",
        f"- median tree alpha: `{summary['median_tree_alpha']}`",
        f"- p90 tree alpha: `{summary['p90_tree_alpha']}`",
        f"- mean DAG alpha vs AST tree: `{summary['mean_dag_alpha_vs_ast_tree']}`",
        f"- median DAG alpha vs AST tree: `{summary['median_dag_alpha_vs_ast_tree']}`",
        f"- p90 DAG alpha vs AST tree: `{summary['p90_dag_alpha_vs_ast_tree']}`",
        f"- mean EML DAG compression: `{summary['mean_eml_dag_compression']}`",
        f"- median EML DAG compression: `{summary['median_eml_dag_compression']}`",
        f"- p90 EML DAG compression: `{summary['p90_eml_dag_compression']}`",
        f"- current threshold below before DAG: `{current_threshold['percent_below_tree_alpha']}%`",
        "- current threshold below after DAG vs AST tree: "
        f"`{current_threshold['percent_below_dag_alpha_vs_ast_tree']}%`",
        "",
        "## Best And Worst Families",
        "",
        f"- top compression-success signature: `{best_signatures[0]['operator_signature']}`",
        f"- top compression-failure signature: `{worst_signatures[0]['operator_signature']}`",
        "",
        "## Semantic Audit",
        "",
        f"- audit expressions: `{semantic_audit['expression_count']}`",
        f"- structurally valid: `{semantic_audit['structural_valid_count']}`",
        f"- numerically valid: `{semantic_audit['semantic_numeric_valid_count']}`",
        "",
        "## Conclusion",
        "",
        build_rescue_conclusion(summary),
        "",
        "## Primary Artifacts",
        "",
        f"- `{config.final_report_path}`",
        f"- `{config.summary_json_path}`",
        f"- `{config.metrics_csv_path}`",
        f"- `{config.dag_alpha_threshold_summary_json_path}`",
        f"- `{config.findings_report_md_path}`",
        f"- `{config.semantic_audit_docs_path}`",
    ]
    return "\n".join(sections) + "\n"


def build_rescue_conclusion(summary: dict[str, object]) -> str:
    """Build the compact structural conclusion for the final report."""
    before = float(summary["percent_below_threshold_tree_alpha"])
    after = float(summary["percent_below_threshold_dag_alpha_vs_ast_tree"])
    median_tree = float(summary["median_tree_alpha"])
    median_dag = float(summary["median_dag_alpha_vs_ast_tree"])
    median_compression = float(summary["median_eml_dag_compression"])
    return (
        "Exact structural DAG sharing reduces the median alpha from "
        f"`{median_tree}` to `{median_dag}` versus AST tree size, with median EML DAG "
        f"compression `{median_compression}`. The current-threshold pass rate improves "
        f"from `{before}%` before DAG sharing to `{after}%` after DAG sharing. This helps "
        "materially, but it does not broadly rescue raw official pure EML under the "
        "current structural threshold."
    )


def markdown_aggregate_table(summary: dict[str, object]) -> str:
    """Render aggregate Goal 3 metric table."""
    rows = [
        ("tree_alpha", "T_EML nodes / T_AST nodes"),
        ("dag_alpha_vs_ast_tree", "D_EML nodes / T_AST nodes"),
        ("dag_alpha_vs_ast_dag", "D_EML nodes / D_AST nodes"),
        ("eml_dag_compression", "T_EML nodes / D_EML nodes"),
    ]
    lines = [
        "| Metric | Definition | Mean | Median | P90 | P95 | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field, definition in rows:
        lines.append(
            "| "
            f"`{field}` | {definition} | {summary[f'mean_{field}']} | "
            f"{summary[f'median_{field}']} | {summary[f'p90_{field}']} | "
            f"{summary[f'p95_{field}']} | {summary[f'max_{field}']} |"
        )
    return "\n".join(lines)


def markdown_threshold_table(rows: list[dict[str, object]]) -> str:
    """Render Goal 3 threshold scenario table."""
    lines = [
        "| Scenario | K | L | Threshold | Tree below | "
        "DAG vs AST tree below | DAG vs AST DAG below |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['scenario']}` | {row['k']} | {row['l']} | "
            f"{row['alpha_threshold']} | {row['percent_below_tree_alpha']} | "
            f"{row['percent_below_dag_alpha_vs_ast_tree']} | "
            f"{row['percent_below_dag_alpha_vs_ast_dag']} |"
        )
    return "\n".join(lines)


def markdown_ast_size_table(rows: list[dict[str, str]]) -> str:
    """Render AST size bucket table."""
    lines = [
        "| AST node bucket | Count | Median tree alpha | Median DAG alpha | Median compression |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['ast_nodes_bucket']}` | {row['count']} | {row['median_tree_alpha']} | "
            f"{row['median_dag_alpha_vs_ast_tree']} | {row['median_eml_dag_compression']} |"
        )
    return "\n".join(lines)


def markdown_operator_family_table(rows: list[dict[str, str]]) -> str:
    """Render dominant-operator-family table."""
    ranked = sorted(
        rows,
        key=lambda row: parse_float(row["median_dag_alpha_vs_ast_tree"]),
        reverse=True,
    )
    lines = [
        "| Family | Count | Median DAG alpha | Median compression | "
        "Median improvement | Below threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(
            "| "
            f"`{row['dominant_operator_family']}` | {row['count']} | "
            f"{row['median_dag_alpha_vs_ast_tree']} | "
            f"{row['median_eml_dag_compression']} | {row['median_improvement']} | "
            f"{row['percent_below_threshold_dag_vs_ast_tree']} |"
        )
    return "\n".join(lines)


def markdown_boolean_feature_table(rows: list[dict[str, str]]) -> str:
    """Render selected boolean feature rows."""
    selected = [
        row
        for row in rows
        if row["value"] == "True"
        and row["feature"] in {"contains_Add", "contains_Mul", "contains_log", "contains_exp"}
    ]
    lines = [
        "| Feature | Count | Median DAG alpha | Median compression | Percent below threshold |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            "| "
            f"`{row['feature']}` | {row['count']} | "
            f"{row['median_dag_alpha_vs_ast_tree']} | "
            f"{row['median_eml_dag_compression']} | "
            f"{row['percent_below_threshold_dag_vs_ast_tree']} |"
        )
    return "\n".join(lines)


def markdown_expression_rank_table(rows: list[dict[str, str]], *, score_field: str) -> str:
    """Render ranked success/failure expression rows."""
    lines = [
        "| Rank | Index | Score | Tree alpha | DAG alpha | "
        "EML DAG compression | Signature | Expression |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['rank']} | {row['index']} | {row[score_field]} | {row['tree_alpha']} | "
            f"{row['dag_alpha_vs_ast_tree']} | {row['eml_dag_compression']} | "
            f"`{row['operator_signature']}` | `{truncate_for_markdown(row['expression'], 80)}` |"
        )
    return "\n".join(lines)


def markdown_signature_table(rows: list[dict[str, str]]) -> str:
    """Render operator-signature ranking rows."""
    lines = [
        "| Signature | Count | Median DAG alpha | Median compression | "
        "Median improvement | Below threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['operator_signature']}` | {row['count']} | "
            f"{row['median_dag_alpha_vs_ast_tree']} | "
            f"{row['median_eml_dag_compression']} | {row['median_improvement']} | "
            f"{row['percent_below_threshold_dag_vs_ast_tree']} |"
        )
    return "\n".join(lines)


def markdown_safe_candidate_table(rows: list[dict[str, str]]) -> str:
    """Render DAG safe-regime candidates."""
    lines = [
        "| Signature | Count | Percent below threshold | Median DAG alpha | Median compression |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['operator_signature']}` | {row['count']} | "
            f"{row['percent_below_threshold_dag_vs_ast_tree']} | "
            f"{row['median_dag_alpha_vs_ast_tree']} | "
            f"{row['median_eml_dag_compression']} |"
        )
    return "\n".join(lines)


def load_config(path: Path) -> Goal3DagPipelineConfig:
    """Load a YAML Goal 3 pipeline config."""
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    return Goal3DagPipelineConfig.model_validate(raw_config)


def load_json_object(path: Path) -> dict[str, object]:
    """Load a JSON object."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def load_json_list(path: Path) -> list[dict[str, object]]:
    """Load a JSON list of objects."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list in {path}")
    return data


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV rows."""
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def write_text(path: Path, text: str) -> None:
    """Write text to a path, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def find_row(rows: list[dict[str, object]], key: str, value: object) -> dict[str, object]:
    """Find a row by exact key/value match."""
    for row in rows:
        if row.get(key) == value:
            return row
    raise ValueError(f"no row found with {key}={value!r}")


def optional_float(value: object) -> float | None:
    """Convert optional numeric JSON value to float."""
    if value is None:
        return None
    return float(value)


def parse_float(value: str) -> float:
    """Parse a required float string."""
    if value == "":
        raise ValueError("expected float, got empty string")
    return float(value)


def truncate_for_markdown(text: str, max_chars: int) -> str:
    """Truncate markdown table text without introducing newlines."""
    normalized = text.replace("\n", " ")
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dag_compression_v0.yaml"),
        help="Path to the Goal 3 DAG pipeline YAML config.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the complete Goal 3 DAG compression pipeline."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    result = run_goal3_dag_pipeline(config)

    print(f"Processed: {result.processed_count}")
    print(f"Supported: {result.supported_count}")
    print(f"Unsupported: {result.unsupported_count}")
    print(
        "Tree alpha mean/median/p90: "
        f"{result.mean_tree_alpha} / {result.median_tree_alpha} / {result.p90_tree_alpha}"
    )
    print(
        "DAG alpha vs AST tree mean/median/p90: "
        f"{result.mean_dag_alpha_vs_ast_tree} / {result.median_dag_alpha_vs_ast_tree} / "
        f"{result.p90_dag_alpha_vs_ast_tree}"
    )
    print(
        "DAG alpha vs AST DAG mean/median/p90: "
        f"{result.mean_dag_alpha_vs_ast_dag} / {result.median_dag_alpha_vs_ast_dag} / "
        f"{result.p90_dag_alpha_vs_ast_dag}"
    )
    print(
        "EML DAG compression mean/median/p90: "
        f"{result.mean_eml_dag_compression} / {result.median_eml_dag_compression} / "
        f"{result.p90_eml_dag_compression}"
    )
    print(f"Percent below threshold before DAG: {result.percent_below_threshold_before_dag}")
    print(
        "Percent below threshold after DAG vs AST tree: "
        f"{result.percent_below_threshold_after_dag_vs_ast_tree}"
    )
    print(
        "Percent below threshold after DAG vs AST DAG: "
        f"{result.percent_below_threshold_after_dag_vs_ast_dag}"
    )
    print(f"Top compression success family: {result.top_compression_success_family}")
    print(f"Top compression failure family: {result.top_compression_failure_family}")
    print("Generated files:")
    for path in result.generated_files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
