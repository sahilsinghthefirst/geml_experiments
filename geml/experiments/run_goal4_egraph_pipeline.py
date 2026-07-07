"""End-to-end Goal 4 non-ML e-graph compression pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from geml.experiments.egraph_compression_mining import (
    EgraphCompressionMiningConfig,
    EgraphCompressionMiningResult,
    run_egraph_compression_mining,
)
from geml.experiments.egraph_compression_study import (
    EgraphCompressionStudyConfig,
    EgraphCompressionStudyResult,
    run_egraph_compression_study,
)
from geml.experiments.egraph_compression_study import (
    load_config as load_egraph_config,
)
from geml.experiments.egraph_semantic_audit import (
    EgraphSemanticAuditConfig,
    EgraphSemanticAuditResult,
    run_egraph_semantic_audit,
)
from geml.experiments.plot_egraph_compression import (
    EgraphCompressionPlotConfig,
    EgraphCompressionPlotResult,
    run_egraph_compression_plots,
)
from geml.experiments.stratified_egraph_compression import (
    StratifiedEgraphCompressionConfig,
    StratifiedEgraphCompressionResult,
    run_stratified_egraph_compression_analysis,
)

GOAL4_MODE_ORDER = ("safe", "positive_real_formal")
GOAL4_SUBSET_ORDER = ("all_v1", "nontrivial_v1", "identity_heavy_v1")


@dataclass(frozen=True, slots=True)
class Goal4EgraphPipelineConfig:
    """Configuration for the complete Goal 4 e-graph pipeline."""

    egraph_config: EgraphCompressionStudyConfig
    expression_generation_summary_json_path: Path | None = Path(
        "outputs/v1/expression_generation_summary.json"
    )
    final_report_path: Path = Path("docs/goal4/GOAL4_NONML_COMPRESSION_STUDY.md")
    summary_doc_path: Path = Path("docs/goal4/GOAL4_SUMMARY.md")
    semantic_audit_json_path: Path = Path("outputs/v1/goal4_egraph_semantic_audit.json")
    semantic_audit_csv_path: Path = Path("outputs/v1/goal4_egraph_semantic_audit.csv")
    semantic_audit_docs_path: Path = Path("docs/goal4/GOAL4_EGRAPH_SEMANTIC_AUDIT.md")
    mining_top_n: int = 25

    @property
    def output_dir(self) -> Path:
        """Return the v1 output directory for derived Goal 4 artifacts."""
        return self.egraph_config.output_dir

    @property
    def alpha_by_operator_signature_csv_path(self) -> Path:
        """Return the operator-signature stratified output path."""
        return self.output_dir / "egraph_alpha_by_operator_signature.csv"

    @property
    def alpha_by_operator_family_csv_path(self) -> Path:
        """Return the operator-family stratified output path."""
        return self.output_dir / "egraph_alpha_by_operator_family.csv"

    @property
    def alpha_by_size_bucket_csv_path(self) -> Path:
        """Return the size-bucket stratified output path."""
        return self.output_dir / "egraph_alpha_by_size_bucket.csv"

    @property
    def alpha_by_rule_mode_csv_path(self) -> Path:
        """Return the rule-mode stratified output path."""
        return self.output_dir / "egraph_alpha_by_rule_mode.csv"

    @property
    def alpha_by_subset_label_csv_path(self) -> Path:
        """Return the subset-label stratified output path."""
        return self.output_dir / "egraph_alpha_by_subset_label.csv"

    @property
    def timeout_failure_summary_csv_path(self) -> Path:
        """Return the timeout/failure stratified output path."""
        return self.output_dir / "egraph_timeout_failure_summary.csv"

    @property
    def triviality_effect_summary_csv_path(self) -> Path:
        """Return the triviality-effect stratified output path."""
        return self.output_dir / "egraph_triviality_effect_summary.csv"

    @property
    def plots_dir(self) -> Path:
        """Return the Goal 4 plot directory."""
        return self.output_dir / "plots_goal4"

    @property
    def top_successes_safe_csv_path(self) -> Path:
        """Return the safe-mode top successes path."""
        return self.output_dir / "top_egraph_compression_successes_safe.csv"

    @property
    def top_successes_positive_real_csv_path(self) -> Path:
        """Return the positive-real top successes path."""
        return self.output_dir / "top_egraph_compression_successes_positive_real.csv"

    @property
    def top_failures_safe_csv_path(self) -> Path:
        """Return the safe-mode top failures path."""
        return self.output_dir / "top_egraph_compression_failures_safe.csv"

    @property
    def top_failures_positive_real_csv_path(self) -> Path:
        """Return the positive-real top failures path."""
        return self.output_dir / "top_egraph_compression_failures_positive_real.csv"

    @property
    def best_operator_signatures_csv_path(self) -> Path:
        """Return the best operator-signature mining path."""
        return self.output_dir / "best_egraph_operator_signatures.csv"

    @property
    def worst_operator_signatures_csv_path(self) -> Path:
        """Return the worst operator-signature mining path."""
        return self.output_dir / "worst_egraph_operator_signatures.csv"

    @property
    def safe_regime_candidates_csv_path(self) -> Path:
        """Return the safe-regime mining path."""
        return self.output_dir / "egraph_safe_regime_candidates.csv"

    @property
    def nontrivial_successes_csv_path(self) -> Path:
        """Return the nontrivial-v1 top successes path."""
        return self.output_dir / "egraph_nontrivial_successes.csv"

    @property
    def identity_heavy_successes_csv_path(self) -> Path:
        """Return the identity-heavy-v1 top successes path."""
        return self.output_dir / "egraph_identity_heavy_successes.csv"

    @property
    def findings_report_md_path(self) -> Path:
        """Return the mined findings report path under outputs/v1."""
        return self.output_dir / "GOAL4_EGRAPH_COMPRESSION_FINDINGS.md"

    def validate(self) -> None:
        """Validate that this is a v1, two-mode, non-ML Goal 4 pipeline."""
        missing_modes = set(GOAL4_MODE_ORDER) - set(self.egraph_config.run_modes)
        if missing_modes:
            raise ValueError(f"Goal 4.10 requires both rule modes; missing {sorted(missing_modes)}")
        if self.mining_top_n <= 0:
            raise ValueError("mining_top_n must be positive")
        checked_paths = [
            self.egraph_config.input_jsonl_path,
            self.egraph_config.goal3_metrics_csv_path,
            self.egraph_config.goal3_summary_json_path,
            self.egraph_config.output_dir,
            self.egraph_config.safe_metrics_csv_path,
            self.egraph_config.safe_metrics_jsonl_path,
            self.egraph_config.positive_real_metrics_csv_path,
            self.egraph_config.positive_real_metrics_jsonl_path,
            self.egraph_config.summary_json_path,
            self.egraph_config.run_metadata_json_path,
            self.semantic_audit_json_path,
            self.semantic_audit_csv_path,
        ]
        bad_paths = [path for path in checked_paths if "outputs/v0" in path.as_posix()]
        if bad_paths:
            joined = ", ".join(str(path) for path in bad_paths)
            raise ValueError(f"Goal 4.10 result-bearing paths must use outputs/v1: {joined}")

    def to_stratified_config(self) -> StratifiedEgraphCompressionConfig:
        """Build the Goal 4.7 stratified-analysis config."""
        return StratifiedEgraphCompressionConfig(
            safe_metrics_csv_path=self.egraph_config.safe_metrics_csv_path,
            positive_real_metrics_csv_path=self.egraph_config.positive_real_metrics_csv_path,
            goal3_metrics_csv_path=self.egraph_config.goal3_metrics_csv_path,
            dag_summary_json_path=self.egraph_config.goal3_summary_json_path,
            expression_generation_summary_json_path=self.expression_generation_summary_json_path,
            alpha_by_operator_signature_csv_path=self.alpha_by_operator_signature_csv_path,
            alpha_by_operator_family_csv_path=self.alpha_by_operator_family_csv_path,
            alpha_by_size_bucket_csv_path=self.alpha_by_size_bucket_csv_path,
            alpha_by_rule_mode_csv_path=self.alpha_by_rule_mode_csv_path,
            alpha_by_subset_label_csv_path=self.alpha_by_subset_label_csv_path,
            timeout_failure_summary_csv_path=self.timeout_failure_summary_csv_path,
            triviality_effect_summary_csv_path=self.triviality_effect_summary_csv_path,
        )

    def to_plot_config(self) -> EgraphCompressionPlotConfig:
        """Build the Goal 4.8 plotting config."""
        return EgraphCompressionPlotConfig(
            safe_metrics_csv_path=self.egraph_config.safe_metrics_csv_path,
            positive_real_metrics_csv_path=self.egraph_config.positive_real_metrics_csv_path,
            operator_signature_csv_path=self.alpha_by_operator_signature_csv_path,
            operator_family_csv_path=self.alpha_by_operator_family_csv_path,
            subset_label_csv_path=self.alpha_by_subset_label_csv_path,
            plots_dir=self.plots_dir,
        )

    def to_mining_config(self) -> EgraphCompressionMiningConfig:
        """Build the Goal 4.8 mining config."""
        return EgraphCompressionMiningConfig(
            safe_metrics_csv_path=self.egraph_config.safe_metrics_csv_path,
            positive_real_metrics_csv_path=self.egraph_config.positive_real_metrics_csv_path,
            operator_signature_csv_path=self.alpha_by_operator_signature_csv_path,
            operator_family_csv_path=self.alpha_by_operator_family_csv_path,
            subset_label_csv_path=self.alpha_by_subset_label_csv_path,
            top_successes_safe_csv_path=self.top_successes_safe_csv_path,
            top_successes_positive_real_csv_path=self.top_successes_positive_real_csv_path,
            top_failures_safe_csv_path=self.top_failures_safe_csv_path,
            top_failures_positive_real_csv_path=self.top_failures_positive_real_csv_path,
            best_operator_signatures_csv_path=self.best_operator_signatures_csv_path,
            worst_operator_signatures_csv_path=self.worst_operator_signatures_csv_path,
            safe_regime_candidates_csv_path=self.safe_regime_candidates_csv_path,
            nontrivial_successes_csv_path=self.nontrivial_successes_csv_path,
            identity_heavy_successes_csv_path=self.identity_heavy_successes_csv_path,
            report_md_path=self.findings_report_md_path,
            top_n=self.mining_top_n,
        )

    def to_audit_config(self) -> EgraphSemanticAuditConfig:
        """Build the Goal 4.9 semantic/provenance audit config."""
        return EgraphSemanticAuditConfig(
            output_dir=self.output_dir,
            json_path=self.semantic_audit_json_path,
            csv_path=self.semantic_audit_csv_path,
            report_path=self.semantic_audit_docs_path,
        )


@dataclass(frozen=True, slots=True)
class Goal4EgraphPipelineResult:
    """Result summary from the complete Goal 4 e-graph pipeline."""

    processed_count_by_mode: dict[str, int]
    success_count_by_mode: dict[str, int]
    timeout_count_by_mode: dict[str, int]
    validation_failure_count_by_mode: dict[str, int]
    median_goal3_alpha_by_mode: dict[str, float | None]
    median_optimized_alpha_by_mode: dict[str, float | None]
    median_compression_gain_by_mode: dict[str, float | None]
    percent_below_threshold_before_by_mode: dict[str, float | None]
    percent_below_threshold_after_by_mode: dict[str, float | None]
    top_success_family: str | None
    top_failure_family: str | None
    compression_result: EgraphCompressionStudyResult
    stratified_result: StratifiedEgraphCompressionResult
    plot_result: EgraphCompressionPlotResult
    mining_result: EgraphCompressionMiningResult
    semantic_audit_result: EgraphSemanticAuditResult
    generated_files: tuple[Path, ...]
    final_report_path: Path
    summary_doc_path: Path


def load_pipeline_config(path: Path) -> Goal4EgraphPipelineConfig:
    """Load the Goal 4.10 pipeline from the Goal 4.6 e-graph YAML config."""
    return Goal4EgraphPipelineConfig(egraph_config=load_egraph_config(path))


def run_goal4_egraph_pipeline(
    config: Goal4EgraphPipelineConfig,
    *,
    rerun_egraph: bool = False,
) -> Goal4EgraphPipelineResult:
    """Run the complete Goal 4 non-ML e-graph compression study."""
    config.validate()
    ensure_v1_goal3_baseline(config)
    compression_result = run_or_load_egraph_compression(config.egraph_config, rerun=rerun_egraph)
    stratified_result = run_stratified_egraph_compression_analysis(config.to_stratified_config())
    plot_result = run_egraph_compression_plots(config.to_plot_config())
    mining_result = run_egraph_compression_mining(config.to_mining_config())
    semantic_audit_result = run_egraph_semantic_audit(config.to_audit_config())

    final_report = build_goal4_nonml_compression_study(config)
    summary_doc = build_goal4_summary_doc(config)
    write_text(config.final_report_path, final_report)
    write_text(config.summary_doc_path, summary_doc)

    summary = load_json_object(config.egraph_config.summary_json_path)
    mode_stats = summary_mode_stats(summary)
    operator_family_rows = load_csv_rows(config.alpha_by_operator_family_csv_path)
    top_success_family = format_family_label(select_top_success_family(operator_family_rows))
    top_failure_family = format_family_label(select_top_failure_family(operator_family_rows))
    generated_files = tuple(
        dict.fromkeys(
            [
                *compression_result.output_paths,
                *stratified_result.output_paths,
                *plot_result.plot_paths,
                *mining_result.output_paths,
                *semantic_audit_result.output_paths,
                config.final_report_path,
                config.summary_doc_path,
            ]
        )
    )

    return Goal4EgraphPipelineResult(
        processed_count_by_mode={
            mode: int(mode_stats.get(mode, {}).get("processed_count", 0))
            for mode in GOAL4_MODE_ORDER
        },
        success_count_by_mode={
            mode: int(mode_stats.get(mode, {}).get("success_count", 0)) for mode in GOAL4_MODE_ORDER
        },
        timeout_count_by_mode={
            mode: int(mode_stats.get(mode, {}).get("timeout_count", 0)) for mode in GOAL4_MODE_ORDER
        },
        validation_failure_count_by_mode={
            mode: int(mode_stats.get(mode, {}).get("validation_failure_count", 0))
            for mode in GOAL4_MODE_ORDER
        },
        median_goal3_alpha_by_mode={
            mode: optional_float(nested_get(mode_stats, mode, "goal3_dag_alpha", "median"))
            for mode in GOAL4_MODE_ORDER
        },
        median_optimized_alpha_by_mode={
            mode: optional_float(nested_get(mode_stats, mode, "optimized_dag_alpha", "median"))
            for mode in GOAL4_MODE_ORDER
        },
        median_compression_gain_by_mode={
            mode: optional_float(
                nested_get(mode_stats, mode, "compression_gain_vs_goal3_dag", "median")
            )
            for mode in GOAL4_MODE_ORDER
        },
        percent_below_threshold_before_by_mode={
            mode: optional_float(
                mode_stats.get(mode, {}).get("percent_below_threshold_before_egraph")
            )
            for mode in GOAL4_MODE_ORDER
        },
        percent_below_threshold_after_by_mode={
            mode: optional_float(
                mode_stats.get(mode, {}).get("percent_below_threshold_after_egraph")
            )
            for mode in GOAL4_MODE_ORDER
        },
        top_success_family=top_success_family,
        top_failure_family=top_failure_family,
        compression_result=compression_result,
        stratified_result=stratified_result,
        plot_result=plot_result,
        mining_result=mining_result,
        semantic_audit_result=semantic_audit_result,
        generated_files=generated_files,
        final_report_path=config.final_report_path,
        summary_doc_path=config.summary_doc_path,
    )


def ensure_v1_goal3_baseline(config: Goal4EgraphPipelineConfig) -> None:
    """Validate that the v1 Goal 3 baseline inputs are present and large enough."""
    egraph_config = config.egraph_config
    if not egraph_config.goal3_metrics_csv_path.exists():
        raise FileNotFoundError(egraph_config.goal3_metrics_csv_path)
    if not egraph_config.goal3_summary_json_path.exists():
        raise FileNotFoundError(egraph_config.goal3_summary_json_path)
    if not egraph_config.input_jsonl_path.exists():
        raise FileNotFoundError(egraph_config.input_jsonl_path)
    baseline_count = count_csv_data_rows(egraph_config.goal3_metrics_csv_path)
    if baseline_count < egraph_config.count:
        raise ValueError(
            "Goal 4.10 requires v1 Goal 3 DAG baselines for every input row: "
            f"expected at least {egraph_config.count}, found {baseline_count}"
        )
    input_count = count_jsonl_rows(egraph_config.input_jsonl_path)
    if input_count < egraph_config.count:
        raise ValueError(
            "Goal 4.10 requires v1 generator inputs for every row: "
            f"expected at least {egraph_config.count}, found {input_count}"
        )


def run_or_load_egraph_compression(
    config: EgraphCompressionStudyConfig,
    *,
    rerun: bool,
) -> EgraphCompressionStudyResult:
    """Run Goal 4.6 or load complete existing v1 outputs."""
    output_paths = egraph_compression_output_paths(config)
    if not rerun and egraph_outputs_complete(config):
        return EgraphCompressionStudyResult(
            summary=load_json_object(config.summary_json_path),
            output_paths=output_paths,
        )
    validate_egraph_resume_counts(config)
    return run_egraph_compression_study(config)


def egraph_outputs_complete(config: EgraphCompressionStudyConfig) -> bool:
    """Return whether saved Goal 4.6 outputs exactly match the requested count."""
    required_paths = egraph_compression_output_paths(config)
    if any(not path.exists() for path in required_paths):
        return False
    try:
        summary = load_json_object(config.summary_json_path)
        mode_stats = summary_mode_stats(summary)
        for mode in GOAL4_MODE_ORDER:
            stats = mode_stats.get(mode)
            if (
                not isinstance(stats, Mapping)
                or int(stats.get("processed_count", -1)) != config.count
            ):
                return False
            csv_path, jsonl_path = egraph_mode_paths(config, mode)
            if count_csv_data_rows(csv_path) != config.count:
                return False
            if count_jsonl_rows(jsonl_path) != config.count:
                return False
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


def validate_egraph_resume_counts(config: EgraphCompressionStudyConfig) -> None:
    """Avoid resuming from output files that already contain too many rows."""
    for mode in GOAL4_MODE_ORDER:
        _csv_path, jsonl_path = egraph_mode_paths(config, mode)
        if jsonl_path.exists():
            existing_count = count_jsonl_rows(jsonl_path)
            if existing_count > config.count:
                raise ValueError(
                    f"{jsonl_path} has {existing_count} rows but config count is {config.count}; "
                    "use a fresh output directory or disable resume for this smaller run"
                )


def egraph_mode_paths(
    config: EgraphCompressionStudyConfig,
    mode: str,
) -> tuple[Path, Path]:
    """Return CSV and JSONL output paths for one rule mode."""
    if mode == "safe":
        return config.safe_metrics_csv_path, config.safe_metrics_jsonl_path
    if mode == "positive_real_formal":
        return config.positive_real_metrics_csv_path, config.positive_real_metrics_jsonl_path
    raise ValueError(f"unsupported rule mode: {mode}")


def egraph_compression_output_paths(config: EgraphCompressionStudyConfig) -> tuple[Path, ...]:
    """Return the core Goal 4.6 output paths."""
    return (
        config.safe_metrics_csv_path,
        config.safe_metrics_jsonl_path,
        config.positive_real_metrics_csv_path,
        config.positive_real_metrics_jsonl_path,
        config.summary_json_path,
        config.run_metadata_json_path,
    )


def build_goal4_nonml_compression_study(config: Goal4EgraphPipelineConfig) -> str:
    """Build the final Goal 4 non-ML compression report from saved v1 artifacts."""
    summary = load_json_object(config.egraph_config.summary_json_path)
    metadata = load_optional_json_object(config.egraph_config.run_metadata_json_path)
    goal3_summary = load_optional_json_object(config.egraph_config.goal3_summary_json_path)
    subset_rows = load_csv_rows(config.alpha_by_subset_label_csv_path)
    operator_family_rows = load_csv_rows(config.alpha_by_operator_family_csv_path)
    best_signatures = load_csv_rows(config.best_operator_signatures_csv_path)
    worst_signatures = load_csv_rows(config.worst_operator_signatures_csv_path)
    safe_successes = load_csv_rows(config.top_successes_safe_csv_path)
    positive_successes = load_csv_rows(config.top_successes_positive_real_csv_path)
    safe_failures = load_csv_rows(config.top_failures_safe_csv_path)
    positive_failures = load_csv_rows(config.top_failures_positive_real_csv_path)
    nontrivial_successes = load_csv_rows(config.nontrivial_successes_csv_path)
    identity_successes = load_csv_rows(config.identity_heavy_successes_csv_path)
    semantic_audit = load_json_object(config.semantic_audit_json_path)
    resource_limits = metadata.get("resource_limits", {}) if isinstance(metadata, Mapping) else {}
    top_success_family = select_top_success_family(operator_family_rows)
    top_failure_family = select_top_failure_family(operator_family_rows)

    sections = [
        "# Goal 4 Non-ML Compression Study",
        "",
        "## Goal 4 Question",
        "",
        "Can e-graph equality saturation and EML-aware extraction reduce official pure "
        "EML-DAG size beyond Goal 3 exact structural DAG sharing?",
        "",
        "Goal 4 is non-ML compression. It uses algebraic rewrite rules before official "
        "pure EML compilation, then scores the extracted expression by official pure "
        "EML-DAG size.",
        "",
        "## Relation To Goals 2, 3, And 3R",
        "",
        "- Goal 2 measured raw official pure EML expansion and showed the pure trees are "
        "representation-complete but structurally expensive.",
        "- Goal 3 added exact structural DAG compression after official pure EML compilation.",
        "- Goal 3R repaired the corpus and established v1 as the serious baseline.",
        "- Goal 4 performs non-ML algebraic compression before EML compilation, then "
        "compares against the Goal 3 v1 exact EML-DAG baseline.",
        "",
        "## Why V1 Is Used",
        "",
        "`outputs/v1` is the default result-bearing corpus. `outputs/v0` is pilot and "
        "diagnostic only. V1 fixes depth, duplicate, log-argument, and triviality artifacts "
        "that would otherwise overstate or distort e-graph compression.",
        "",
        f"- configured count: `{config.egraph_config.count}`",
        f"- v1 input JSONL: `{config.egraph_config.input_jsonl_path}`",
        f"- v1 Goal 3 metrics: `{config.egraph_config.goal3_metrics_csv_path}`",
        f"- v1 Goal 3 summary processed count: `{goal3_summary.get('processed_count', 'n/a')}`",
        "",
        "## Rule Modes And Assumptions",
        "",
        "- `safe`: commutativity, associativity, identities, safe inverse forms, sub lowering, "
        "double negation, and exact bounded constant folding. It excludes branch-sensitive "
        "log/exp identities.",
        "- `positive_real_formal`: includes safe rules plus positive-real formal log/exp "
        "rules. This mode is branch-sensitive, relies on the v1 positive-real domain "
        "convention, and makes no universal complex-domain validity claim.",
        "",
        "The two modes are reported separately. Goal 4 metrics must not be mixed with Goal 3 "
        "exact-DAG metrics without naming the mode.",
        "",
        "## Extractor Objective",
        "",
        "The headline extractor is `exact_eml_dag_beam_cost`. It enumerates source "
        "candidates from the root e-class, converts each candidate to SymPy without "
        "simplification, compiles with the official pure EML compiler, converts the result "
        "to an exact structural EML DAG, and selects the candidate with minimum official "
        "pure EML-DAG node count.",
        "",
        "Ordinary AST node count is only a baseline. It is not an EML-optimal objective.",
        "",
        "Tie-breaking order:",
        "",
        "1. extracted official pure EML-DAG nodes",
        "2. extracted official pure EML-tree nodes",
        "3. extracted source AST-DAG nodes",
        "4. extracted source AST-tree nodes",
        "5. stable expression string",
        "",
        "## Resource Limits",
        "",
        markdown_resource_limits(resource_limits),
        "",
        "## 10k V1 Results: Safe Mode",
        "",
        markdown_mode_detail(summary, "safe"),
        "",
        "## 10k V1 Results: Positive-Real Formal Mode",
        "",
        markdown_mode_detail(summary, "positive_real_formal"),
        "",
        "Positive-real rows are labeled with branch-sensitive assumptions and branch-sensitive "
        "rule usage. They are not complex-domain algebra claims.",
        "",
        "## Subset Analysis",
        "",
        "The corpus is split by measured triviality features, not guesses. `identity_heavy_v1` "
        "contains measured identity or trivial simplification opportunities. "
        "`nontrivial_v1` excludes those measured features.",
        "",
        markdown_subset_table(subset_rows),
        "",
        "The median nontrivial compression gain remains much closer to `1.0` than the "
        "identity-heavy gain. Goal 4 therefore helps, but easy identity simplifications "
        "explain a large share of the aggregate improvement.",
        "",
        "## Operator-Family Analysis",
        "",
        f"- top success family: `{format_family_label(top_success_family)}`",
        f"- top failure family: `{format_family_label(top_failure_family)}`",
        "",
        markdown_operator_family_table(operator_family_rows),
        "",
        "Add/Mul-heavy and mixed-operator expressions still dominate much of the remaining "
        "difficulty because algebraic source simplification does not remove the recursive "
        "official pure EML expansion cost in the general case. Pure `exp` groups are small "
        "but validation-heavy in the mined failure ranking; larger mixed families such as "
        "`Mul+exp`, `Add+exp`, and Add/Mul/log/exp mixtures remain important failure regimes.",
        "",
        "## Runtime And Timeout Analysis",
        "",
        markdown_runtime_table(summary, subset_rows),
        "",
        "Timeout rows are retained in the CSV/JSONL artifacts. They are included in processed "
        "counts, timeout rates, and failure summaries rather than silently dropped.",
        "",
        "## Success And Failure Case Studies",
        "",
        "Top safe-mode successes:",
        "",
        markdown_expression_table(safe_successes[:5]),
        "",
        "Top positive-real successes:",
        "",
        markdown_expression_table(positive_successes[:5]),
        "",
        "Top nontrivial successes:",
        "",
        markdown_expression_table(nontrivial_successes[:5]),
        "",
        "Top identity-heavy successes:",
        "",
        markdown_expression_table(identity_successes[:5]),
        "",
        "Top safe-mode failures:",
        "",
        markdown_expression_table(safe_failures[:5]),
        "",
        "Top positive-real failures:",
        "",
        markdown_expression_table(positive_failures[:5]),
        "",
        "Best operator-signature groups:",
        "",
        markdown_signature_table(best_signatures[:5]),
        "",
        "Worst operator-signature groups:",
        "",
        markdown_signature_table(worst_signatures[:5]),
        "",
        "## Semantic And Provenance Audit",
        "",
        markdown_semantic_audit_summary(semantic_audit, config),
        "",
        "The audit checks selected expressions in both modes, records rewrite provenance by "
        "rule name and tier, confirms safe mode does not apply branch-sensitive rules, and "
        "verifies the EML-DAG evaluator agrees with positive-real numeric probes. SymPy "
        "`simplify` is allowed only as an optional diagnostic outside the rewrite path.",
        "",
        "## Integrity Statement",
        "",
        "Final EML outputs remain official pure EML after extraction. The pipeline checks for:",
        "",
        "- no derived leaves",
        "- no hidden compound-expression leaves",
        "- no fake macro leaves",
        "- no macro/template EML nodes",
        "- no modified official EML compiler formulas",
        "- internal EML nodes labeled only `eml`",
        "- EML leaves restricted to variables or constant `1`",
        "- no SymPy.simplify rewrite shortcut",
        "",
        "Improvements are structural non-ML compression results, not GNN or neural-model evidence.",
        "",
        "## Recommendation For Goal 5",
        "",
        goal5_recommendation(summary, subset_rows),
        "",
        "Prepare ML-facing graph representations for later experiments, but do not start "
        "Goal 5 here. The next graph surfaces should keep separate views for source AST "
        "trees/DAGs, official pure EML DAGs, e-graph-optimized source ASTs, and "
        "e-graph-optimized official pure EML DAGs, with rule mode, assumptions, subset "
        "labels, validation status, and rewrite provenance carried as metadata.",
        "",
        "## Reproducible Command",
        "",
        "```bash",
        ".venv/bin/python -m geml.experiments.run_goal4_egraph_pipeline "
        "--config configs/egraph_compression_v1.yaml",
        ".venv/bin/python -m pytest",
        ".venv/bin/python -m ruff check .",
        ".venv/bin/python -m ruff format . --check",
        "```",
    ]
    return "\n".join(sections) + "\n"


def build_goal4_summary_doc(config: Goal4EgraphPipelineConfig) -> str:
    """Build a concise Goal 4 summary from saved v1 artifacts."""
    summary = load_json_object(config.egraph_config.summary_json_path)
    subset_rows = load_csv_rows(config.alpha_by_subset_label_csv_path)
    operator_family_rows = load_csv_rows(config.alpha_by_operator_family_csv_path)
    best_signatures = load_csv_rows(config.best_operator_signatures_csv_path)
    worst_signatures = load_csv_rows(config.worst_operator_signatures_csv_path)
    semantic_audit = load_json_object(config.semantic_audit_json_path)
    top_success_family = select_top_success_family(operator_family_rows)
    top_failure_family = select_top_failure_family(operator_family_rows)

    sections = [
        "# Goal 4 Summary",
        "",
        "Goal 4 implemented non-ML e-graph compression for the repaired v1 corpus. It runs "
        "safe and positive-real formal rewrite modes, uses EML-aware extraction, compares "
        "against the Goal 3 exact EML-DAG baseline, mines successes/failures, plots summary "
        "artifacts, and audits semantics, structural purity, and rewrite provenance.",
        "",
        "## Headline Result",
        "",
        markdown_mode_summary_table(summary),
        "",
        "The safe mode improves many rows but only modestly changes the threshold pass rate. "
        "The positive-real formal mode improves more rows, but its branch-sensitive "
        "assumptions must be reported separately. Neither mode broadly rescues official "
        "pure EML-DAG size under the current threshold.",
        "",
        "## Subsets",
        "",
        markdown_subset_table(subset_rows),
        "",
        "Identity-heavy rows drive the largest gains. Nontrivial rows remain difficult.",
        "",
        "## Best And Worst Families",
        "",
        f"- top success family: `{format_family_label(top_success_family)}`",
        f"- top failure family: `{format_family_label(top_failure_family)}`",
        f"- top success signature: `{best_signatures[0]['operator_signature']}`",
        f"- top failure signature: `{worst_signatures[0]['operator_signature']}`",
        "",
        "## Semantic Audit",
        "",
        markdown_semantic_audit_summary(semantic_audit, config),
        "",
        "## Primary Artifacts",
        "",
        f"- `{config.egraph_config.summary_json_path}`",
        f"- `{config.findings_report_md_path}`",
        f"- `{config.final_report_path}`",
        f"- `{config.summary_doc_path}`",
        f"- `{config.semantic_audit_docs_path}`",
        f"- `{config.plots_dir}`",
        "",
        "## Goal 5 Recommendation",
        "",
        goal5_recommendation(summary, subset_rows),
    ]
    return "\n".join(sections) + "\n"


def goal5_recommendation(
    summary: Mapping[str, object],
    subset_rows: Sequence[Mapping[str, str]],
) -> str:
    """Return the Goal 5 recommendation text."""
    positive = summary_mode_stats(summary).get("positive_real_formal", {})
    safe = summary_mode_stats(summary).get("safe", {})
    safe_after = optional_float(safe.get("percent_below_threshold_after_egraph"))
    positive_after = optional_float(positive.get("percent_below_threshold_after_egraph"))
    nontrivial_positive = find_subset_row(subset_rows, "positive_real_formal", "nontrivial_v1")
    nontrivial_gain = (
        nontrivial_positive.get("median_compression_gain_vs_goal3_dag")
        if nontrivial_positive
        else "n/a"
    )
    return (
        "Non-ML e-graph compression is a useful baseline and should remain in the "
        "evaluation stack, but it is not enough by itself. The threshold pass rate remains "
        f"low after safe mode (`{format_percent(safe_after)}`) and positive-real mode "
        f"(`{format_percent(positive_after)}`), while nontrivial positive-real median gain is "
        f"`{nontrivial_gain}`. Goal 5 should therefore still investigate ML-facing "
        "motif or macro compression, with honest separation between structural "
        "compression results and future model-performance claims."
    )


def markdown_mode_detail(summary: Mapping[str, object], mode: str) -> str:
    """Render the detailed metrics for one rule mode."""
    stats = summary_mode_stats(summary)[mode]
    return "\n".join(
        [
            f"- processed: `{stats['processed_count']}`",
            f"- success: `{stats['success_count']}`",
            f"- timeout: `{stats['timeout_count']}`",
            f"- validation failures: `{stats['validation_failure_count']}`",
            f"- median Goal 3 DAG alpha: `{nested_get(stats, 'goal3_dag_alpha', 'median')}`",
            f"- median optimized DAG alpha: `{nested_get(stats, 'optimized_dag_alpha', 'median')}`",
            "- median compression gain vs Goal 3 DAG: "
            f"`{nested_get(stats, 'compression_gain_vs_goal3_dag', 'median')}`",
            f"- percent improved: `{format_percent(stats.get('percent_improved'))}`",
            f"- percent unchanged: `{format_percent(stats.get('percent_unchanged'))}`",
            f"- percent worse: `{format_percent(stats.get('percent_worse'))}`",
            "- below threshold before e-graph: "
            f"`{format_percent(stats.get('percent_below_threshold_before_egraph'))}`",
            "- below threshold after e-graph: "
            f"`{format_percent(stats.get('percent_below_threshold_after_egraph'))}`",
            "- median runtime per expression: "
            f"`{nested_get(stats, 'runtime_seconds', 'median_per_expression')}` seconds",
        ]
    )


def markdown_mode_summary_table(summary: Mapping[str, object]) -> str:
    """Render the two-mode headline metric table."""
    lines = [
        "| Mode | Processed | Success | Timeout | Validation failures | "
        "Median Goal 3 alpha | Median optimized alpha | Median gain | Below before | Below after |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    mode_stats = summary_mode_stats(summary)
    for mode in GOAL4_MODE_ORDER:
        stats = mode_stats.get(mode, {})
        lines.append(
            "| "
            f"`{mode}` | {stats.get('processed_count', 'n/a')} | "
            f"{stats.get('success_count', 'n/a')} | {stats.get('timeout_count', 'n/a')} | "
            f"{stats.get('validation_failure_count', 'n/a')} | "
            f"{nested_get(stats, 'goal3_dag_alpha', 'median')} | "
            f"{nested_get(stats, 'optimized_dag_alpha', 'median')} | "
            f"{nested_get(stats, 'compression_gain_vs_goal3_dag', 'median')} | "
            f"{format_percent(stats.get('percent_below_threshold_before_egraph'))} | "
            f"{format_percent(stats.get('percent_below_threshold_after_egraph'))} |"
        )
    return "\n".join(lines)


def markdown_subset_table(rows: Sequence[Mapping[str, str]]) -> str:
    """Render subset-specific e-graph summary rows."""
    lines = [
        "| Subset | Mode | Count | Success | Median Goal 3 alpha | Median optimized alpha | "
        "Median gain | Improved | Unchanged | Worse | Below before | Below after | Timeout | "
        "Validation failure | Branch-sensitive usage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            GOAL4_SUBSET_ORDER.index(row["subset_label"])
            if row["subset_label"] in GOAL4_SUBSET_ORDER
            else len(GOAL4_SUBSET_ORDER),
            GOAL4_MODE_ORDER.index(row["rule_mode"])
            if row["rule_mode"] in GOAL4_MODE_ORDER
            else len(GOAL4_MODE_ORDER),
        ),
    )
    for row in ordered_rows:
        lines.append(
            "| "
            f"`{row['subset_label']}` | `{row['rule_mode']}` | {row['count']} | "
            f"{row['success_count']} | {row['median_goal3_dag_alpha_vs_ast_tree']} | "
            f"{row['median_optimized_dag_alpha_vs_ast_tree']} | "
            f"{row['median_compression_gain_vs_goal3_dag']} | "
            f"{format_percent(row['percent_improved'])} | "
            f"{format_percent(row['percent_unchanged'])} | "
            f"{format_percent(row['percent_worse'])} | "
            f"{format_percent(row['percent_below_threshold_before'])} | "
            f"{format_percent(row['percent_below_threshold_after'])} | "
            f"{format_percent(row['timeout_rate'])} | "
            f"{format_percent(row['validation_failure_rate'])} | "
            f"{format_percent(row['branch_sensitive_rule_usage_rate'])} |"
        )
    return "\n".join(lines)


def markdown_operator_family_table(rows: Sequence[Mapping[str, str]], *, limit: int = 12) -> str:
    """Render the highest-signal operator-family rows."""
    all_rows = [row for row in rows if row.get("subset_label") == "all_v1"]
    ranked = sorted(
        all_rows,
        key=lambda row: (
            -parse_optional_float(row.get("median_compression_gain_vs_goal3_dag"), 0.0),
            -parse_optional_float(row.get("percent_improved"), 0.0),
            row.get("rule_mode", ""),
            row.get("dominant_operator_family", ""),
        ),
    )[:limit]
    lines = [
        "| Mode | Family | Contains | Count | Median optimized alpha | Median gain | "
        "Improved | Timeout | Validation failure |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(
            "| "
            f"`{row['rule_mode']}` | `{row['dominant_operator_family']}` | "
            f"`{contains_signature(row)}` | {row['count']} | "
            f"{row['median_optimized_dag_alpha_vs_ast_tree']} | "
            f"{row['median_compression_gain_vs_goal3_dag']} | "
            f"{format_percent(row['percent_improved'])} | "
            f"{format_percent(row['timeout_rate'])} | "
            f"{format_percent(row['validation_failure_rate'])} |"
        )
    return "\n".join(lines)


def markdown_runtime_table(
    summary: Mapping[str, object],
    subset_rows: Sequence[Mapping[str, str]],
) -> str:
    """Render runtime and timeout metrics."""
    lines = [
        "| Mode | Median runtime seconds | Mean runtime seconds | Timeout count | "
        "All-v1 timeout rate | Nontrivial timeout rate | Identity-heavy timeout rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    mode_stats = summary_mode_stats(summary)
    for mode in GOAL4_MODE_ORDER:
        stats = mode_stats.get(mode, {})
        all_row = find_subset_row(subset_rows, mode, "all_v1")
        nontrivial_row = find_subset_row(subset_rows, mode, "nontrivial_v1")
        identity_row = find_subset_row(subset_rows, mode, "identity_heavy_v1")
        lines.append(
            "| "
            f"`{mode}` | {nested_get(stats, 'runtime_seconds', 'median_per_expression')} | "
            f"{nested_get(stats, 'runtime_seconds', 'mean_per_expression')} | "
            f"{stats.get('timeout_count', 'n/a')} | "
            f"{format_percent(all_row.get('timeout_rate') if all_row else None)} | "
            f"{format_percent(nontrivial_row.get('timeout_rate') if nontrivial_row else None)} | "
            f"{format_percent(identity_row.get('timeout_rate') if identity_row else None)} |"
        )
    return "\n".join(lines)


def markdown_expression_table(rows: Sequence[Mapping[str, str]]) -> str:
    """Render ranked expression success/failure rows."""
    if not rows:
        return "_No rows._"
    lines = [
        "| Rank | Index | Mode | Subset | Original EML-DAG | Extracted EML-DAG | Gain | "
        "Optimized alpha | Threshold improved | Expression |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('rank', '')} | {row.get('index', '')} | `{row.get('rule_mode', '')}` | "
            f"`{row.get('subset_label', '')}` | {row.get('original_eml_dag_nodes', '')} | "
            f"{row.get('extracted_eml_dag_nodes', '')} | "
            f"{row.get('compression_gain_vs_goal3_dag', '')} | "
            f"{row.get('optimized_dag_alpha_vs_ast_tree', '')} | "
            f"{row.get('threshold_status_improved', '')} | "
            f"`{truncate_for_markdown(row.get('original_expression', ''), 60)}` |"
        )
    return "\n".join(lines)


def markdown_signature_table(rows: Sequence[Mapping[str, str]]) -> str:
    """Render operator-signature ranking rows."""
    lines = [
        "| Rank | Mode | Signature | Count | Success | Median optimized alpha | Median gain | "
        "Improved | Timeout | Validation failure |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['rank']} | `{row['rule_mode']}` | `{row['operator_signature']}` | "
            f"{row['count']} | {row['success_count']} | "
            f"{row['median_optimized_dag_alpha_vs_ast_tree']} | "
            f"{row['median_compression_gain_vs_goal3_dag']} | "
            f"{format_percent(row['percent_improved'])} | "
            f"{format_percent(row['timeout_rate'])} | "
            f"{format_percent(row['validation_failure_rate'])} |"
        )
    return "\n".join(lines)


def markdown_resource_limits(resource_limits: Mapping[str, object]) -> str:
    """Render configured equality saturation and extraction limits."""
    fields = [
        "max_iterations",
        "max_enodes",
        "max_eclasses",
        "timeout_seconds",
        "row_timeout_seconds",
        "beam_size",
        "max_candidate_depth",
        "max_candidates_evaluated",
    ]
    return "\n".join(f"- {field}: `{resource_limits.get(field, 'n/a')}`" for field in fields)


def markdown_semantic_audit_summary(
    semantic_audit: Mapping[str, object],
    config: Goal4EgraphPipelineConfig,
) -> str:
    """Render semantic/provenance audit status."""
    audit_summary = semantic_audit.get("summary", {})
    if not isinstance(audit_summary, Mapping):
        audit_summary = {}
    return "\n".join(
        [
            f"- audit rows: `{audit_summary.get('row_count', 'n/a')}`",
            "- all semantic validation valid: "
            f"`{audit_summary.get('all_semantic_validation_valid', 'n/a')}`",
            "- all EML-DAG validation valid: "
            f"`{audit_summary.get('all_eml_dag_validation_valid', 'n/a')}`",
            "- all structural purity valid: "
            f"`{audit_summary.get('all_structural_purity_valid', 'n/a')}`",
            "- safe branch-sensitive applications: "
            f"`{audit_summary.get('safe_branch_sensitive_application_count', 'n/a')}`",
            "- positive-real branch-sensitive applications: "
            f"`{audit_summary.get('positive_real_branch_sensitive_application_count', 'n/a')}`",
            f"- provenance invalid count: `{audit_summary.get('provenance_invalid_count', 'n/a')}`",
            "- SymPy simplify rewrite path free: "
            f"`{audit_summary.get('sympy_simplify_rewrite_path_free', 'n/a')}`",
            f"- audit JSON: `{config.semantic_audit_json_path}`",
            f"- audit CSV: `{config.semantic_audit_csv_path}`",
            f"- audit docs: `{config.semantic_audit_docs_path}`",
        ]
    )


def select_top_success_family(rows: Sequence[Mapping[str, str]]) -> Mapping[str, str] | None:
    """Select the strongest operator-family success row."""
    candidates = [row for row in rows if row.get("subset_label") == "all_v1"]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            parse_optional_float(row.get("median_compression_gain_vs_goal3_dag"), 0.0) * 10.0
            + parse_optional_float(row.get("percent_improved"), 0.0)
            + parse_optional_float(row.get("percent_below_threshold_after"), 0.0)
            - parse_optional_float(row.get("timeout_rate"), 0.0)
            - parse_optional_float(row.get("validation_failure_rate"), 0.0),
            parse_optional_float(row.get("count"), 0.0),
        ),
    )


def select_top_failure_family(rows: Sequence[Mapping[str, str]]) -> Mapping[str, str] | None:
    """Select the weakest operator-family row."""
    candidates = [row for row in rows if row.get("subset_label") == "all_v1"]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            parse_optional_float(row.get("median_optimized_dag_alpha_vs_ast_tree"), 0.0) * 10.0
            + parse_optional_float(row.get("percent_worse"), 0.0)
            + 0.25 * parse_optional_float(row.get("percent_unchanged"), 0.0)
            + parse_optional_float(row.get("timeout_rate"), 0.0)
            + parse_optional_float(row.get("validation_failure_rate"), 0.0),
            parse_optional_float(row.get("count"), 0.0),
        ),
    )


def format_family_label(row: Mapping[str, str] | None) -> str | None:
    """Format an operator-family row as a compact label."""
    if row is None:
        return None
    contains = contains_signature(row)
    return f"{row['rule_mode']}:{row['dominant_operator_family']}[{contains}]"


def contains_signature(row: Mapping[str, str]) -> str:
    """Return a compact contains-feature signature for a family row."""
    flags = [
        name.removeprefix("contains_")
        for name in ("contains_Add", "contains_Mul", "contains_log", "contains_exp")
        if row.get(name) == "True"
    ]
    return "+".join(flags) if flags else "leaf_only"


def find_subset_row(
    rows: Sequence[Mapping[str, str]],
    rule_mode: str,
    subset_label: str,
) -> Mapping[str, str] | None:
    """Find one subset summary row."""
    for row in rows:
        if row.get("rule_mode") == rule_mode and row.get("subset_label") == subset_label:
            return row
    return None


def summary_mode_stats(summary: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    """Return rule-mode stats from the summary JSON."""
    raw_modes = summary.get("rule_modes", {})
    if not isinstance(raw_modes, Mapping):
        raise TypeError("egraph summary rule_modes must be a mapping")
    return {str(mode): stats for mode, stats in raw_modes.items() if isinstance(stats, Mapping)}


def nested_get(mapping: Mapping[str, object], *keys: str) -> object:
    """Return a nested mapping value or n/a."""
    value: object = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return "n/a"
        value = value[key]
    return value


def optional_float(value: object) -> float | None:
    """Return a finite float or None."""
    if value in {None, "", "n/a"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_optional_float(value: object, default: float) -> float:
    """Parse a finite float-like value with a default."""
    parsed = optional_float(value)
    return default if parsed is None else parsed


def format_percent(value: object) -> str:
    """Format a percent value that is already on the 0-100 scale."""
    parsed = optional_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.3f}%"


def truncate_for_markdown(value: object, limit: int) -> str:
    """Return a compact single-cell markdown value."""
    text = str(value).replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_json_object(path: Path) -> dict[str, object]:
    """Load a JSON object."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


def load_optional_json_object(path: Path | None) -> dict[str, object]:
    """Load an optional JSON object."""
    if path is None or not path.exists():
        return {}
    return load_json_object(path)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a CSV artifact."""
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def count_csv_data_rows(path: Path) -> int:
    """Count data rows in a CSV file."""
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return sum(1 for _row in csv.DictReader(csv_file))


def count_jsonl_rows(path: Path) -> int:
    """Count non-empty JSONL rows."""
    with path.open("r", encoding="utf-8") as jsonl_file:
        return sum(1 for line in jsonl_file if line.strip())


def write_text(path: Path, text: str) -> None:
    """Write text after ensuring the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/egraph_compression_v1.yaml"),
        help="Goal 4.6 e-graph compression YAML config.",
    )
    parser.add_argument(
        "--rerun-egraph",
        action="store_true",
        help="Force rerunning Goal 4.6 compression instead of loading complete outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 4.10 pipeline from the command line."""
    args = build_parser().parse_args(argv)
    config = load_pipeline_config(args.config)
    result = run_goal4_egraph_pipeline(config, rerun_egraph=args.rerun_egraph)

    print("Goal 4 e-graph pipeline complete")
    for mode in GOAL4_MODE_ORDER:
        print(
            f"{mode}: processed={result.processed_count_by_mode[mode]}, "
            f"success={result.success_count_by_mode[mode]}, "
            f"timeout={result.timeout_count_by_mode[mode]}, "
            f"validation_failures={result.validation_failure_count_by_mode[mode]}"
        )
    print(f"Top success family: {result.top_success_family}")
    print(f"Top failure family: {result.top_failure_family}")
    print(f"Final report: {result.final_report_path}")
    print(f"Summary doc: {result.summary_doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
