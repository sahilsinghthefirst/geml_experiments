"""End-to-end Goal 2 expansion-factor pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator

from geml.experiments.expansion_failure_mining import (
    FailureMiningConfig,
    FailureMiningResult,
    run_failure_mining,
)
from geml.experiments.expansion_study import ExpansionStudyConfig, run_expansion_study
from geml.experiments.plot_expansion_study import (
    PLOT_FILENAMES,
    ExpansionPlotConfig,
    ExpansionPlotResult,
    run_expansion_plots,
)
from geml.experiments.stratified_expansion import (
    StratifiedExpansionConfig,
    StratifiedExpansionResult,
    run_stratified_expansion_analysis,
)


class Goal2ExpansionPipelineConfig(ExpansionStudyConfig):
    """Configuration for the complete Goal 2 expansion-factor pipeline."""

    alpha_by_ast_depth_csv_path: Path = Path("outputs/v0/alpha_by_ast_depth.csv")
    alpha_by_ast_size_bucket_csv_path: Path = Path("outputs/v0/alpha_by_ast_size_bucket.csv")
    alpha_by_operator_family_csv_path: Path = Path("outputs/v0/alpha_by_operator_family.csv")
    alpha_by_operator_signature_csv_path: Path = Path("outputs/v0/alpha_by_operator_signature.csv")
    alpha_by_boolean_features_csv_path: Path = Path("outputs/v0/alpha_by_boolean_features.csv")
    plots_dir: Path = Path("outputs/v0/plots")
    top_20_alpha_csv_path: Path = Path("outputs/v0/top_20_alpha_expressions.csv")
    top_20_eml_node_csv_path: Path = Path("outputs/v0/top_20_eml_node_expressions.csv")
    top_20_eml_depth_csv_path: Path = Path("outputs/v0/top_20_eml_depth_expressions.csv")
    top_alpha_explosions_csv_path: Path = Path("outputs/v0/top_alpha_explosions.csv")
    top_eml_node_explosions_csv_path: Path = Path("outputs/v0/top_eml_node_explosions.csv")
    top_eml_depth_explosions_csv_path: Path = Path("outputs/v0/top_eml_depth_explosions.csv")
    worst_operator_signatures_csv_path: Path = Path("outputs/v0/worst_operator_signatures.csv")
    safest_operator_signatures_csv_path: Path = Path("outputs/v0/safest_operator_signatures.csv")
    depth_failure_modes_csv_path: Path = Path("outputs/v0/depth_failure_modes.csv")
    safe_eml_regime_candidates_csv_path: Path = Path("outputs/v0/safe_eml_regime_candidates.csv")
    failure_report_md_path: Path = Path("docs/goal2/GOAL2_FAILURE_CASES.md")
    final_report_path: Path = Path("docs/goal2/GOAL2_EXPANSION_STUDY.md")
    failure_top_n: int = Field(default=20, gt=0)
    snippet_max_chars: int = Field(default=1200, gt=0)

    @model_validator(mode="after")
    def validate_goal2_mode(self) -> Self:
        if self.representation_mode != "restricted_eml_pure":
            raise ValueError("Goal 2 pipeline requires representation_mode=restricted_eml_pure")
        return self

    def to_stratified_config(self) -> StratifiedExpansionConfig:
        """Build Goal 2.3 config from the shared pipeline config."""
        return StratifiedExpansionConfig(
            raw_metrics_csv_path=self.raw_metrics_csv_path,
            alpha_summary_csv_path=self.alpha_summary_csv_path,
            alpha_summary_json_path=self.alpha_summary_json_path,
            alpha_by_ast_depth_csv_path=self.alpha_by_ast_depth_csv_path,
            alpha_by_ast_size_bucket_csv_path=self.alpha_by_ast_size_bucket_csv_path,
            alpha_by_operator_family_csv_path=self.alpha_by_operator_family_csv_path,
            alpha_by_operator_signature_csv_path=self.alpha_by_operator_signature_csv_path,
            alpha_by_boolean_features_csv_path=self.alpha_by_boolean_features_csv_path,
        )

    def to_plot_config(self) -> ExpansionPlotConfig:
        """Build Goal 2.4 config from the shared pipeline config."""
        return ExpansionPlotConfig(
            raw_metrics_csv_path=self.raw_metrics_csv_path,
            alpha_summary_json_path=self.alpha_summary_json_path,
            alpha_by_ast_depth_csv_path=self.alpha_by_ast_depth_csv_path,
            alpha_by_operator_family_csv_path=self.alpha_by_operator_family_csv_path,
            plots_dir=self.plots_dir,
            top_alpha_csv_path=self.top_20_alpha_csv_path,
            top_eml_node_csv_path=self.top_20_eml_node_csv_path,
            top_eml_depth_csv_path=self.top_20_eml_depth_csv_path,
        )

    def to_failure_config(self) -> FailureMiningConfig:
        """Build Goal 2.5 config from the shared pipeline config."""
        return FailureMiningConfig(
            raw_metrics_csv_path=self.raw_metrics_csv_path,
            alpha_by_operator_family_csv_path=self.alpha_by_operator_family_csv_path,
            alpha_by_operator_signature_csv_path=self.alpha_by_operator_signature_csv_path,
            alpha_by_ast_depth_csv_path=self.alpha_by_ast_depth_csv_path,
            top_alpha_csv_path=self.top_alpha_explosions_csv_path,
            top_eml_node_csv_path=self.top_eml_node_explosions_csv_path,
            top_eml_depth_csv_path=self.top_eml_depth_explosions_csv_path,
            worst_operator_signatures_csv_path=self.worst_operator_signatures_csv_path,
            safest_operator_signatures_csv_path=self.safest_operator_signatures_csv_path,
            depth_failure_modes_csv_path=self.depth_failure_modes_csv_path,
            safe_eml_regime_candidates_csv_path=self.safe_eml_regime_candidates_csv_path,
            report_md_path=self.failure_report_md_path,
            top_n=self.failure_top_n,
            snippet_max_chars=self.snippet_max_chars,
        )


@dataclass(frozen=True)
class Goal2PipelineResult:
    """Result summary from the complete Goal 2 pipeline."""

    processed_count: int
    supported_count: int
    unsupported_count: int
    mean_alpha: float | None
    median_alpha: float | None
    p90_alpha: float | None
    p95_alpha: float | None
    max_alpha: float | None
    alpha_summaries: tuple[dict[str, object], ...]
    stratified_result: StratifiedExpansionResult
    plot_result: ExpansionPlotResult
    failure_result: FailureMiningResult
    generated_files: tuple[Path, ...]
    final_report_path: Path


def run_goal2_expansion_pipeline(
    config: Goal2ExpansionPipelineConfig,
) -> Goal2PipelineResult:
    """Run the complete Goal 2 expansion-factor study in dependency order."""
    run_expansion_study(config)
    stratified_result = run_stratified_expansion_analysis(config.to_stratified_config())
    plot_result = run_expansion_plots(config.to_plot_config())
    failure_result = run_failure_mining(config.to_failure_config())

    official_summary = load_json_object(config.summary_json_path)
    alpha_summaries = tuple(load_json_list(config.alpha_summary_json_path))
    final_report = build_final_goal2_report(config)
    write_text(config.final_report_path, final_report)

    generated_files = tuple(
        dict.fromkeys(
            [
                config.input_jsonl_path,
                config.raw_metrics_jsonl_path,
                config.raw_metrics_csv_path,
                config.alpha_summary_csv_path,
                config.alpha_summary_json_path,
                config.summary_json_path,
                config.top_alpha_json_path,
                config.top_depth_json_path,
                config.simple_examples_json_path,
                *stratified_result.output_paths,
                *plot_result.plot_paths,
                *plot_result.table_paths,
                *failure_result.output_paths,
                config.final_report_path,
            ]
        )
    )

    return Goal2PipelineResult(
        processed_count=int(official_summary["processed_count"]),
        supported_count=int(official_summary["official_pure_eml_supported_count"]),
        unsupported_count=int(official_summary["unsupported_count"]),
        mean_alpha=optional_float(official_summary.get("mean_alpha")),
        median_alpha=optional_float(official_summary.get("median_alpha")),
        p90_alpha=optional_float(official_summary.get("p90_alpha")),
        p95_alpha=optional_float(official_summary.get("p95_alpha")),
        max_alpha=optional_float(official_summary.get("max_alpha")),
        alpha_summaries=alpha_summaries,
        stratified_result=stratified_result,
        plot_result=plot_result,
        failure_result=failure_result,
        generated_files=generated_files,
        final_report_path=config.final_report_path,
    )


def build_final_goal2_report(config: Goal2ExpansionPipelineConfig) -> str:
    """Build the final Goal 2 expansion-study report from saved artifacts."""
    official_summary = load_json_object(config.summary_json_path)
    alpha_summaries = load_json_list(config.alpha_summary_json_path)
    operator_family_rows = load_csv_rows(config.alpha_by_operator_family_csv_path)
    ast_size_rows = load_csv_rows(config.alpha_by_ast_size_bucket_csv_path)
    boolean_rows = load_csv_rows(config.alpha_by_boolean_features_csv_path)
    worst_signatures = load_csv_rows(config.worst_operator_signatures_csv_path)
    safe_candidates = load_csv_rows(config.safe_eml_regime_candidates_csv_path)
    depth_modes = load_csv_rows(config.depth_failure_modes_csv_path)

    strongest_family = max(operator_family_rows, key=lambda row: parse_float(row["median_alpha"]))
    worst_signature = worst_signatures[0]
    closest_candidate = safe_candidates[0]
    current_threshold = alpha_summaries[0]
    current_percent_below = parse_float_like(current_threshold["percent_below_threshold"])

    sections = [
        "# Goal 2 Expansion Study",
        "",
        "## Goal And Scientific Question",
        "",
        "Goal 2 asks whether the official pure recursive EML representation is "
        "structurally smaller than a standard expression AST before any compression or "
        "modeling. The measured quantity is alpha:",
        "",
        "```text",
        "alpha = |T_EML| / |T_AST|",
        "```",
        "",
        "If raw pure EML alpha is usually far above the theoretical threshold, then "
        "uncompressed EML trees are unlikely to be computationally smaller than ASTs.",
        "",
        "## Official Compiler And Representation",
        "",
        "The pure compiler ports macro definitions from:",
        "",
        "- Repository: `VA00/SymbolicRegressionPackage`",
        "- File: `EML_toolkit/EmL_compiler/eml_compiler_v4.py`",
        "",
        "Core primitive:",
        "",
        "```text",
        "EML(a, b) = exp(a) - log(b)",
        "```",
        "",
        "Goal 2 pure EML grammar:",
        "",
        "```text",
        "P ::= variable | 1 | eml(P, P)",
        "```",
        "",
        "Every internal node must be `eml`; leaves may only be variables or constant "
        "`1`. Derived leaves are invalid for alpha because they can hide compound "
        "source expressions inside a single leaf and artificially reduce expansion.",
        "",
        "## Dataset And Run Configuration",
        "",
        f"- expression count: `{config.count}`",
        f"- seed: `{config.seed}`",
        f"- max source depth: `{config.max_depth}`",
        f"- representation mode: `{config.representation_mode}`",
        "- supported official pure EML count: "
        f"`{official_summary['official_pure_eml_supported_count']}`",
        f"- unsupported count: `{official_summary['unsupported_count']}`",
        "",
        "## Threshold Model",
        "",
        "Threshold formula:",
        "",
        "```text",
        "alpha_threshold = 1 + log(K) / log(4L)",
        "```",
        "",
        f"Primary row-level K/L values: `K={config.alpha_threshold_k}`, "
        f"`L={config.alpha_threshold_l}`.",
        "",
        markdown_threshold_table(alpha_summaries),
        "",
        "## Aggregate Alpha Results",
        "",
        f"- mean alpha: `{official_summary['mean_alpha']}`",
        f"- median alpha: `{official_summary['median_alpha']}`",
        f"- p90 alpha: `{official_summary['p90_alpha']}`",
        f"- p95 alpha: `{official_summary['p95_alpha']}`",
        f"- max alpha: `{official_summary['max_alpha']}`",
        f"- current-threshold percent below: `{current_percent_below}`",
        "",
        "## Stratified Findings",
        "",
        f"- highest median-alpha dominant family: `{strongest_family['dominant_operator_family']}` "
        f"with median alpha `{strongest_family['median_alpha']}`",
        f"- worst operator signature by failure mining: `{worst_signature['operator_signature']}` "
        f"with median alpha `{worst_signature['median_alpha']}`",
        "- AST-size bucket summaries show alpha rising as source trees get larger:",
        "",
        markdown_ast_size_table(ast_size_rows),
        "",
        "- Boolean feature summaries show Add/Mul participation is the main structural risk:",
        "",
        markdown_boolean_feature_table(boolean_rows),
        "",
        "## Plots",
        "",
        *[f"- `{config.plots_dir / filename}`" for filename in PLOT_FILENAMES],
        "",
        "## Failure Modes",
        "",
        "The top failure-mode tables are:",
        "",
        "- `outputs/v0/top_alpha_explosions.csv`",
        "- `outputs/v0/top_eml_node_explosions.csv`",
        "- `outputs/v0/top_eml_depth_explosions.csv`",
        "- `outputs/v0/worst_operator_signatures.csv`",
        "- `outputs/v0/depth_failure_modes.csv`",
        "",
        "Worst signature preview:",
        "",
        markdown_worst_signature_table(worst_signatures[:5]),
        "",
        "Depth failure-mode preview:",
        "",
        markdown_depth_mode_table(depth_modes[:5]),
        "",
        "## Safe-Regime Candidates",
        "",
        f"Closest raw pure EML candidate: `{closest_candidate['operator_signature']}` "
        f"with median alpha `{closest_candidate['median_alpha']}` and median threshold "
        f"gap `{closest_candidate['median_threshold_gap']}`.",
        "",
        markdown_safe_candidate_table(safe_candidates[:5]),
        "",
        "No robust raw pure EML safe regime appears under the current threshold when "
        "using these generated expressions.",
        "",
        "## Conclusion",
        "",
        "Raw official pure EML expansion is scientifically valid but structurally "
        "expensive. The representation removes operator vocabulary but expands common "
        "Add/Mul/Pow/log/exp source patterns into much larger trees. The 10k run shows "
        "alpha far above all tested thresholds for almost every expression, so raw "
        "pure EML trees are unlikely to be computationally smaller than ASTs without "
        "a separate compression mechanism.",
        "",
        "This is structural evidence only. It is not model-performance evidence.",
        "",
        "## Recommendation For Goal 3",
        "",
        "Goal 3 should study DAG compression or shared-subexpression compression for "
        "pure EML before introducing GNNs, neural models, or equivalence-pair "
        "generation. Compression should be measured against the same AST baseline, "
        "threshold scenarios, and failure strata from Goal 2.",
        "",
        "## Reproducible Commands",
        "",
        "```bash",
        ".venv/bin/python -m geml.experiments.run_goal2_expansion_pipeline "
        "--config configs/expansion_v0.yaml",
        ".venv/bin/python -m pytest",
        ".venv/bin/python -m ruff check .",
        ".venv/bin/python -m ruff format . --check",
        "```",
    ]
    return "\n".join(sections) + "\n"


def markdown_threshold_table(rows: list[dict[str, object]]) -> str:
    """Render threshold scenario table."""
    lines = [
        "| Scenario | K | L | Alpha threshold | Percent below | Percent above |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['scenario']}` | {row['k']} | {row['l']} | "
            f"{row['alpha_threshold']} | {row['percent_below_threshold']} | "
            f"{row['percent_above_threshold']} |"
        )
    return "\n".join(lines)


def markdown_ast_size_table(rows: list[dict[str, str]]) -> str:
    """Render AST-size bucket table."""
    lines = [
        "| AST node bucket | Count | Median alpha | P90 alpha | Mean EML nodes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['ast_nodes_bucket']}` | {row['count']} | {row['median_alpha']} | "
            f"{row['p90_alpha']} | {row['mean_eml_nodes']} |"
        )
    return "\n".join(lines)


def markdown_boolean_feature_table(rows: list[dict[str, str]]) -> str:
    """Render selected boolean feature rows."""
    selected = [
        row
        for row in rows
        if row["value"] == "True" and row["feature"] in {"contains_Add", "contains_Mul"}
    ]
    lines = [
        "| Feature | Count | Median alpha | P90 alpha | Percent below threshold |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            "| "
            f"`{row['feature']}` | {row['count']} | {row['median_alpha']} | "
            f"{row['p90_alpha']} | {row['percent_below_threshold']} |"
        )
    return "\n".join(lines)


def markdown_worst_signature_table(rows: list[dict[str, str]]) -> str:
    """Render worst-signature preview table."""
    lines = [
        "| Signature | Median alpha | P90 alpha | Count | Percent below threshold |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['operator_signature']}` | {row['median_alpha']} | {row['p90_alpha']} | "
            f"{row['count']} | {row['percent_below_threshold']} |"
        )
    return "\n".join(lines)


def markdown_depth_mode_table(rows: list[dict[str, str]]) -> str:
    """Render depth failure-mode preview table."""
    lines = [
        "| AST depth | Mean alpha | P90 alpha | Mean EML/AST nodes | Count |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['ast_depth']} | {row['mean_alpha']} | {row['p90_alpha']} | "
            f"{row['mean_eml_to_ast_nodes']} | {row['count']} |"
        )
    return "\n".join(lines)


def markdown_safe_candidate_table(rows: list[dict[str, str]]) -> str:
    """Render safe-regime candidate preview table."""
    lines = [
        "| Signature | Median alpha | Median threshold gap | P90 alpha | Percent below threshold |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['operator_signature']}` | {row['median_alpha']} | "
            f"{row['median_threshold_gap']} | {row['p90_alpha']} | "
            f"{row['percent_below_threshold']} |"
        )
    return "\n".join(lines)


def load_config(path: Path) -> Goal2ExpansionPipelineConfig:
    """Load a YAML Goal 2 pipeline config."""
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    return Goal2ExpansionPipelineConfig.model_validate(raw_config)


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


def parse_float_like(value: object) -> float:
    """Parse a JSON numeric value."""
    if value is None:
        raise ValueError("expected numeric JSON value")
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/expansion_v0.yaml"),
        help="Path to the Goal 2 expansion pipeline YAML config.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the complete Goal 2 expansion-factor pipeline."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    result = run_goal2_expansion_pipeline(config)

    print(f"Processed: {result.processed_count}")
    print(f"Supported official pure EML: {result.supported_count}")
    print(f"Unsupported: {result.unsupported_count}")
    print(f"Mean alpha: {result.mean_alpha}")
    print(f"Median alpha: {result.median_alpha}")
    print(f"P90 alpha: {result.p90_alpha}")
    print(f"P95 alpha: {result.p95_alpha}")
    print(f"Max alpha: {result.max_alpha}")
    print("Threshold scenarios:")
    for row in result.alpha_summaries:
        print(f"  {row['scenario']}: {row['percent_below_threshold']}% below")
    print("Generated files:")
    for path in result.generated_files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
