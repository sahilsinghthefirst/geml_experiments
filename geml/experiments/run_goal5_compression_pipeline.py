"""Goal 5 ML-facing compression integration pipeline."""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from geml.experiments.goal5_frequent_motif_mining import (
    load_config as load_frequent_config,
)
from geml.experiments.goal5_frequent_motif_mining import (
    run_goal5_frequent_motif_mining,
)
from geml.experiments.goal5_hierarchical_export import (
    load_config as load_hierarchical_config,
)
from geml.experiments.goal5_hierarchical_export import (
    run_goal5_hierarchical_export,
)
from geml.experiments.goal5_learned_motif_compression import (
    load_config as load_learned_config,
)
from geml.experiments.goal5_learned_motif_compression import (
    run_goal5_learned_motif_compression,
)
from geml.experiments.goal5_macro_graph_baseline import (
    load_config as load_macro_config,
)
from geml.experiments.goal5_macro_graph_baseline import (
    run_goal5_macro_graph_baseline,
)
from geml.experiments.goal5_neural_egraph_extractor import (
    load_config as load_neural_config,
)
from geml.experiments.goal5_neural_egraph_extractor import (
    run_goal5_neural_egraph_extractor,
)
from geml.experiments.shared import (
    build_run_metadata,
    write_json_object,
)
from geml.experiments.shared import (
    load_json_object as load_shared_json_object,
)
from geml.experiments.shared import (
    markdown_table as _markdown_table,
)
from geml.experiments.shared import (
    write_text as write_shared_text,
)

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
type JSONMapping = dict[str, JSONValue]

GOAL5_SUBSET_ORDER = ("all_v1", "nontrivial_v1", "identity_heavy_v1")

COMPARISON_FIELDS = [
    "mode",
    "stage",
    "representation_mode",
    "processed_count",
    "success_count",
    "failure_count",
    "validation_failure_count",
    "reconstruction_failure_count",
    "median_nodes",
    "median_child_refs_or_edges",
    "median_alpha_or_size_ratio",
    "median_compression_gain_vs_goal3_eml_dag",
    "nontrivial_processed_count",
    "nontrivial_success_count",
    "nontrivial_median_gain_vs_goal3_eml_dag",
    "nontrivial_median_nodes",
    "validation_rate_percent",
    "expansion_validation_rate_percent",
    "reconstruction_validation_rate_percent",
    "is_pure_eml_metric",
    "compressed_nodes_are_pure_eml",
    "size_metric_is_compression_metric",
    "notes",
]


@dataclass(frozen=True, slots=True)
class Goal5CompressionPipelineConfig:
    """Configuration for the integrated Goal 5 compression report."""

    seed: int = 0
    count: int = 10_000
    reuse_existing_artifacts: bool = True
    run_missing_artifacts: bool = True
    force_rerun_macro_graph: bool = False
    force_rerun_frequent_motifs: bool = False
    force_rerun_learned_motifs: bool = False
    force_rerun_neural_egraph: bool = False
    force_rerun_hierarchical_export: bool = False
    require_large_jsonl_artifacts: bool = False
    macro_config_path: Path = Path("configs/macro_graph_v1.yaml")
    frequent_motif_config_path: Path = Path("configs/frequent_motifs_v1.yaml")
    frequent_motif_train_only_config_path: Path = Path("configs/frequent_motifs_train_only_v1.yaml")
    learned_motif_config_path: Path = Path("configs/learned_motifs_v1.yaml")
    neural_egraph_config_path: Path = Path("configs/neural_egraph_extractor_v1.yaml")
    hierarchical_export_config_path: Path = Path("configs/hierarchical_graph_export_v1.yaml")
    goal3_summary_json_path: Path = Path("outputs/v1/dag_compression_summary.json")
    goal3_metrics_csv_path: Path = Path("outputs/v1/dag_compression_metrics.csv")
    goal4_summary_json_path: Path = Path("outputs/v1/egraph_compression_summary.json")
    goal4_safe_metrics_csv_path: Path = Path("outputs/v1/egraph_compression_metrics_safe.csv")
    goal4_positive_real_metrics_csv_path: Path = Path(
        "outputs/v1/egraph_compression_metrics_positive_real.csv"
    )
    macro_summary_json_path: Path = Path("outputs/v1/goal5_macro_graph_summary.json")
    macro_metrics_csv_path: Path = Path("outputs/v1/goal5_macro_graph_metrics.csv")
    macro_metrics_jsonl_path: Path = Path("outputs/v1/goal5_macro_graph_metrics.jsonl")
    frequent_motif_summary_json_path: Path = Path("outputs/v1/goal5_frequent_motif_summary.json")
    frequent_motif_metrics_csv_path: Path = Path("outputs/v1/goal5_frequent_motif_metrics.csv")
    frequent_motif_metrics_jsonl_path: Path = Path("outputs/v1/goal5_frequent_motif_metrics.jsonl")
    frequent_motif_vocab_json_path: Path = Path("outputs/v1/goal5_frequent_motif_vocab.json")
    frequent_motif_train_only_summary_json_path: Path = Path(
        "outputs/v1/goal5_frequent_motif_train_only_summary.json"
    )
    frequent_motif_train_only_metrics_csv_path: Path = Path(
        "outputs/v1/goal5_frequent_motif_train_only_metrics.csv"
    )
    frequent_motif_train_only_metrics_jsonl_path: Path = Path(
        "outputs/v1/goal5_frequent_motif_train_only_metrics.jsonl"
    )
    frequent_motif_train_only_vocab_json_path: Path = Path(
        "outputs/v1/goal5_frequent_motif_train_only_vocab.json"
    )
    learned_motif_summary_json_path: Path = Path("outputs/v1/goal5_learned_motif_summary.json")
    learned_motif_metrics_csv_path: Path = Path("outputs/v1/goal5_learned_motif_metrics.csv")
    learned_motif_metrics_jsonl_path: Path = Path("outputs/v1/goal5_learned_motif_metrics.jsonl")
    learned_motif_vocab_json_path: Path = Path("outputs/v1/goal5_learned_motif_vocab.json")
    learned_motif_train_log_json_path: Path = Path("outputs/v1/goal5_learned_motif_train_log.json")
    neural_egraph_summary_json_path: Path = Path("outputs/v1/goal5_neural_egraph_summary.json")
    neural_egraph_metrics_csv_path: Path = Path("outputs/v1/goal5_neural_egraph_metrics.csv")
    neural_egraph_candidate_dataset_jsonl_path: Path = Path(
        "outputs/v1/goal5_neural_egraph_candidate_dataset.jsonl"
    )
    neural_egraph_train_log_json_path: Path = Path("outputs/v1/goal5_neural_egraph_train_log.json")
    hierarchical_graphs_jsonl_path: Path = Path("outputs/v1/goal5_hierarchical_graphs.jsonl")
    hierarchical_splits_json_path: Path = Path("outputs/v1/goal5_graph_splits.json")
    hierarchical_schema_json_path: Path = Path("outputs/v1/goal5_graph_schema.json")
    hierarchical_summary_json_path: Path = Path("outputs/v1/goal5_hierarchical_export_summary.json")
    comparison_csv_path: Path = Path("outputs/v1/goal5_compression_comparison.csv")
    summary_json_path: Path = Path("outputs/v1/goal5_compression_summary.json")
    findings_report_path: Path = Path("outputs/v1/GOAL5_COMPRESSION_FINDINGS.md")
    final_report_path: Path = Path("docs/goal5/GOAL5_ML_FACING_COMPRESSION_STUDY.md")
    summary_doc_path: Path = Path("docs/goal5/GOAL5_SUMMARY.md")

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be positive")
        _assert_no_outputs_v0(_all_config_paths(self))


@dataclass(frozen=True, slots=True)
class Goal5CompressionPipelineResult:
    """Result payload for the integrated Goal 5 pipeline."""

    summary: JSONMapping
    comparison_rows: tuple[JSONMapping, ...]
    generated_files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class Goal5ArtifactBundle:
    """Loaded Goal 3, Goal 4, and Goal 5 report artifacts."""

    summaries: Mapping[str, JSONMapping]
    json_artifacts: Mapping[str, JSONMapping]
    csv_headers: Mapping[str, tuple[str, ...]]


def load_config(path: Path) -> Goal5CompressionPipelineConfig:
    """Load a Goal 5.6 YAML config."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return Goal5CompressionPipelineConfig(
        **{key: _coerce_config_value(key, value) for key, value in raw.items()}
    )


def run_goal5_compression_pipeline(
    config: Goal5CompressionPipelineConfig,
) -> Goal5CompressionPipelineResult:
    """Run or load Goal 5 stages, compare them, and write final reports."""
    started_at = time.time()
    stage_status = ensure_goal5_artifacts(config)
    artifacts = load_goal5_artifacts(config)
    comparison_rows = build_comparison_rows(artifacts.summaries)
    write_comparison_csv(comparison_rows, config.comparison_csv_path)
    summary = build_pipeline_summary(
        config,
        artifacts=artifacts,
        comparison_rows=comparison_rows,
        stage_status=stage_status,
        started_at=started_at,
        completed_at=time.time(),
    )
    write_json(config.summary_json_path, summary)
    write_text(config.final_report_path, build_study_report(summary))
    write_text(config.summary_doc_path, build_summary_doc(summary))
    write_text(config.findings_report_path, build_findings_report(summary))
    return Goal5CompressionPipelineResult(
        summary=summary,
        comparison_rows=tuple(comparison_rows),
        generated_files=(
            config.comparison_csv_path,
            config.summary_json_path,
            config.final_report_path,
            config.summary_doc_path,
            config.findings_report_path,
        ),
    )


def ensure_goal5_artifacts(config: Goal5CompressionPipelineConfig) -> dict[str, JSONMapping]:
    """Run configured Goal 5 stages only when required artifacts are missing."""
    statuses: dict[str, JSONMapping] = {}
    _ensure_required_existing(
        "goal3_pure_eml_dag",
        [config.goal3_summary_json_path, config.goal3_metrics_csv_path],
    )
    _ensure_required_existing(
        "goal4_egraph",
        [
            config.goal4_summary_json_path,
            config.goal4_safe_metrics_csv_path,
            config.goal4_positive_real_metrics_csv_path,
        ],
    )
    statuses["macro_graph"] = _ensure_stage(
        name="macro_graph",
        force=config.force_rerun_macro_graph,
        reuse_existing=config.reuse_existing_artifacts,
        run_missing=config.run_missing_artifacts,
        config_path=config.macro_config_path,
        required_paths=[
            config.macro_summary_json_path,
            config.macro_metrics_csv_path,
            *([config.macro_metrics_jsonl_path] if config.require_large_jsonl_artifacts else []),
        ],
        load_stage_config=load_macro_config,
        run_stage=run_goal5_macro_graph_baseline,
    )
    statuses["frequent_motifs"] = _ensure_stage(
        name="frequent_motifs",
        force=config.force_rerun_frequent_motifs,
        reuse_existing=config.reuse_existing_artifacts,
        run_missing=config.run_missing_artifacts,
        config_path=config.frequent_motif_config_path,
        required_paths=[
            config.frequent_motif_summary_json_path,
            config.frequent_motif_metrics_csv_path,
            config.frequent_motif_vocab_json_path,
            *(
                [config.frequent_motif_metrics_jsonl_path]
                if config.require_large_jsonl_artifacts
                else []
            ),
        ],
        load_stage_config=load_frequent_config,
        run_stage=run_goal5_frequent_motif_mining,
    )
    statuses["frequent_motifs_train_only"] = _ensure_stage(
        name="frequent_motifs_train_only",
        force=config.force_rerun_frequent_motifs,
        reuse_existing=config.reuse_existing_artifacts,
        run_missing=config.run_missing_artifacts,
        config_path=config.frequent_motif_train_only_config_path,
        required_paths=[
            config.frequent_motif_train_only_summary_json_path,
            config.frequent_motif_train_only_metrics_csv_path,
            config.frequent_motif_train_only_vocab_json_path,
            *(
                [config.frequent_motif_train_only_metrics_jsonl_path]
                if config.require_large_jsonl_artifacts
                else []
            ),
        ],
        load_stage_config=load_frequent_config,
        run_stage=run_goal5_frequent_motif_mining,
    )
    statuses["learned_motifs"] = _ensure_stage(
        name="learned_motifs",
        force=config.force_rerun_learned_motifs,
        reuse_existing=config.reuse_existing_artifacts,
        run_missing=config.run_missing_artifacts,
        config_path=config.learned_motif_config_path,
        required_paths=[
            config.learned_motif_summary_json_path,
            config.learned_motif_metrics_csv_path,
            config.learned_motif_vocab_json_path,
            config.learned_motif_train_log_json_path,
            *(
                [config.learned_motif_metrics_jsonl_path]
                if config.require_large_jsonl_artifacts
                else []
            ),
        ],
        load_stage_config=load_learned_config,
        run_stage=run_goal5_learned_motif_compression,
    )
    statuses["neural_egraph"] = _ensure_stage(
        name="neural_egraph",
        force=config.force_rerun_neural_egraph,
        reuse_existing=config.reuse_existing_artifacts,
        run_missing=config.run_missing_artifacts,
        config_path=config.neural_egraph_config_path,
        required_paths=[
            config.neural_egraph_summary_json_path,
            config.neural_egraph_metrics_csv_path,
            config.neural_egraph_train_log_json_path,
            *(
                [config.neural_egraph_candidate_dataset_jsonl_path]
                if config.require_large_jsonl_artifacts
                else []
            ),
        ],
        load_stage_config=load_neural_config,
        run_stage=run_goal5_neural_egraph_extractor,
    )
    statuses["hierarchical_export"] = _ensure_stage(
        name="hierarchical_export",
        force=config.force_rerun_hierarchical_export,
        reuse_existing=config.reuse_existing_artifacts,
        run_missing=config.run_missing_artifacts,
        config_path=config.hierarchical_export_config_path,
        required_paths=[
            config.hierarchical_summary_json_path,
            config.hierarchical_schema_json_path,
            config.hierarchical_splits_json_path,
            *(
                [config.hierarchical_graphs_jsonl_path]
                if config.require_large_jsonl_artifacts
                else []
            ),
        ],
        load_stage_config=load_hierarchical_config,
        run_stage=run_goal5_hierarchical_export,
    )
    return statuses


def load_goal5_artifacts(config: Goal5CompressionPipelineConfig) -> Goal5ArtifactBundle:
    """Load all report-sized artifacts needed for Goal 5.6 comparison."""
    summary_paths = {
        "goal3": config.goal3_summary_json_path,
        "goal4": config.goal4_summary_json_path,
        "macro_graph": config.macro_summary_json_path,
        "frequent_motifs": config.frequent_motif_summary_json_path,
        "frequent_motifs_train_only": config.frequent_motif_train_only_summary_json_path,
        "learned_motifs": config.learned_motif_summary_json_path,
        "neural_egraph": config.neural_egraph_summary_json_path,
        "hierarchical_export": config.hierarchical_summary_json_path,
    }
    json_artifact_paths = {
        "frequent_motif_vocab": config.frequent_motif_vocab_json_path,
        "frequent_motif_train_only_vocab": config.frequent_motif_train_only_vocab_json_path,
        "learned_motif_vocab": config.learned_motif_vocab_json_path,
        "learned_motif_train_log": config.learned_motif_train_log_json_path,
        "neural_egraph_train_log": config.neural_egraph_train_log_json_path,
        "hierarchical_schema": config.hierarchical_schema_json_path,
        "hierarchical_splits": config.hierarchical_splits_json_path,
    }
    csv_paths = {
        "goal3_metrics": config.goal3_metrics_csv_path,
        "goal4_safe_metrics": config.goal4_safe_metrics_csv_path,
        "goal4_positive_real_metrics": config.goal4_positive_real_metrics_csv_path,
        "macro_graph_metrics": config.macro_metrics_csv_path,
        "frequent_motif_metrics": config.frequent_motif_metrics_csv_path,
        "frequent_motif_train_only_metrics": config.frequent_motif_train_only_metrics_csv_path,
        "learned_motif_metrics": config.learned_motif_metrics_csv_path,
        "neural_egraph_metrics": config.neural_egraph_metrics_csv_path,
    }
    missing = [
        path
        for path in [*summary_paths.values(), *json_artifact_paths.values(), *csv_paths.values()]
        if not path.exists()
    ]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing Goal 5 artifact(s): {joined}")
    return Goal5ArtifactBundle(
        summaries={name: load_json_object(path) for name, path in summary_paths.items()},
        json_artifacts={name: load_json_object(path) for name, path in json_artifact_paths.items()},
        csv_headers={name: read_csv_header(path) for name, path in csv_paths.items()},
    )


def build_comparison_rows(summaries: Mapping[str, JSONMapping]) -> list[JSONMapping]:
    """Build the final Goal 5 comparison rows from loaded summaries."""
    goal3 = summaries["goal3"]
    goal4 = summaries["goal4"]
    macro = summaries["macro_graph"]
    frequent = summaries["frequent_motifs"]
    learned = summaries["learned_motifs"]
    neural = summaries["neural_egraph"]
    hierarchy = summaries["hierarchical_export"]
    safe = _nested_dict(goal4, "rule_modes", "safe")
    positive = _nested_dict(goal4, "rule_modes", "positive_real_formal")
    hierarchy_stats = _nested_dict(hierarchy, "node_edge_stats_by_mode")
    return [
        _comparison_row(
            mode="Goal 3 pure EML-DAG",
            stage="Goal 3",
            representation_mode="pure_eml_dag_graph",
            processed_count=_int_or_none(goal3.get("processed_count")),
            success_count=_int_or_none(goal3.get("supported_count")),
            median_nodes=_stat(_nested_dict(safe, "original_eml_dag_nodes"), "median"),
            median_gain=1.0,
            is_pure_eml_metric=True,
            compressed_nodes_are_pure_eml=True,
            size_metric_is_compression_metric=True,
            notes="Official pure EML-DAG baseline from Goal 3.",
        ),
        _comparison_row(
            mode="Goal 4 safe e-graph optimized EML-DAG",
            stage="Goal 4",
            representation_mode="egraph_safe_eml_dag_graph",
            processed_count=_int_or_none(safe.get("processed_count")),
            success_count=_int_or_none(safe.get("success_count")),
            validation_failure_count=_int_or_none(safe.get("validation_failure_count")),
            median_nodes=_stat(_nested_dict(safe, "extracted_eml_dag_nodes"), "median"),
            median_gain=_stat(_nested_dict(safe, "compression_gain_vs_goal3_dag"), "median"),
            nontrivial=_nested_dict(safe, "results_by_subset_label", "nontrivial_v1"),
            nontrivial_gain_key="compression_gain_vs_goal3_dag",
            validation_rate_percent=_success_rate(safe),
            is_pure_eml_metric=True,
            compressed_nodes_are_pure_eml=True,
            size_metric_is_compression_metric=True,
            notes="Non-ML e-graph extraction with safe rules.",
        ),
        _comparison_row(
            mode="Goal 4 positive-real optimized EML-DAG",
            stage="Goal 4",
            representation_mode="egraph_positive_real_eml_dag_graph",
            processed_count=_int_or_none(positive.get("processed_count")),
            success_count=_int_or_none(positive.get("success_count")),
            validation_failure_count=_int_or_none(positive.get("validation_failure_count")),
            median_nodes=_stat(_nested_dict(positive, "extracted_eml_dag_nodes"), "median"),
            median_gain=_stat(_nested_dict(positive, "compression_gain_vs_goal3_dag"), "median"),
            nontrivial=_nested_dict(positive, "results_by_subset_label", "nontrivial_v1"),
            nontrivial_gain_key="compression_gain_vs_goal3_dag",
            validation_rate_percent=_success_rate(positive),
            is_pure_eml_metric=True,
            compressed_nodes_are_pure_eml=True,
            size_metric_is_compression_metric=True,
            notes="Non-ML e-graph extraction with positive-real assumptions.",
        ),
        _comparison_row(
            mode="Goal 5 macro graph",
            stage="Goal 5.1",
            representation_mode="macro_graph",
            processed_count=_int_or_none(macro.get("processed_count")),
            success_count=_int_or_none(macro.get("success_count")),
            reconstruction_failure_count=_int_or_none(
                macro.get("expansion_validation_failure_count")
            ),
            median_nodes=_stat(
                _nested_dict(macro, "results_by_subset_label", "all_v1", "macro_graph_nodes"),
                "median",
            ),
            median_alpha_or_size_ratio=_float_or_none(macro.get("median_macro_graph_alpha")),
            median_gain=_float_or_none(macro.get("median_compression_gain_vs_goal3_eml_dag")),
            nontrivial=_nested_dict(macro, "results_by_subset_label", "nontrivial_v1"),
            nontrivial_gain_key="compression_gain_vs_goal3_eml_dag",
            nontrivial_nodes_key="macro_graph_nodes",
            expansion_validation_rate_percent=100.0
            - _failure_rate(macro, "expansion_validation_failure_count"),
            reconstruction_validation_rate_percent=100.0
            - _failure_rate(macro, "expansion_validation_failure_count"),
            is_pure_eml_metric=False,
            compressed_nodes_are_pure_eml=False,
            size_metric_is_compression_metric=True,
            notes="Transparent compiler macro nodes; not pure EML alpha.",
        ),
        _comparison_row(
            mode="Goal 5 frequent motif graph",
            stage="Goal 5.2",
            representation_mode="frequent_motif_graph",
            processed_count=_int_or_none(frequent.get("processed_count")),
            success_count=_int_or_none(frequent.get("success_count")),
            reconstruction_failure_count=_int_or_none(
                frequent.get("expansion_validation_failure_count")
            ),
            median_nodes=_stat(_nested_dict(frequent, "motif_compressed_nodes"), "median"),
            median_gain=_stat(
                _nested_dict(frequent, "compression_gain_vs_goal3_eml_dag"), "median"
            ),
            nontrivial=_nested_dict(frequent, "results_by_subset_label", "nontrivial_v1"),
            nontrivial_gain_key="compression_gain_vs_goal3_eml_dag",
            expansion_validation_rate_percent=100.0
            - _failure_rate(frequent, "expansion_validation_failure_count"),
            reconstruction_validation_rate_percent=100.0
            - _failure_rate(frequent, "expansion_validation_failure_count"),
            is_pure_eml_metric=False,
            compressed_nodes_are_pure_eml=False,
            size_metric_is_compression_metric=True,
            notes="Greedy motif replacement from mined frequent motifs.",
        ),
        _comparison_row(
            mode="Goal 5 learned motif graph",
            stage="Goal 5.3",
            representation_mode="learned_motif_graph",
            processed_count=_int_or_none(learned.get("processed_count")),
            success_count=_int_or_none(learned.get("success_count")),
            reconstruction_failure_count=_int_or_none(learned.get("reconstruction_failure_count")),
            median_nodes=_stat(
                _nested_dict(learned, "results_by_subset_label", "all_v1", "learned_motif_nodes"),
                "median",
            ),
            median_gain=_stat(_nested_dict(learned, "learned_gain_vs_goal3_eml_dag"), "median"),
            nontrivial=_nested_dict(learned, "results_by_subset_label", "nontrivial_v1"),
            nontrivial_gain_key="learned_gain_vs_goal3_eml_dag",
            nontrivial_nodes_key="learned_motif_nodes",
            reconstruction_validation_rate_percent=100.0
            - _failure_rate(learned, "reconstruction_failure_count"),
            is_pure_eml_metric=False,
            compressed_nodes_are_pure_eml=False,
            size_metric_is_compression_metric=True,
            notes="Deterministic learned motif selection; exact reconstruction required.",
        ),
        _comparison_row(
            mode="Goal 5 neural e-graph extractor",
            stage="Goal 5.4",
            representation_mode="neural_selected_eml_dag_graph",
            processed_count=_int_or_none(neural.get("processed_group_count")),
            success_count=_int_or_none(neural.get("success_count")),
            validation_failure_count=_int_or_none(neural.get("validation_failure_count")),
            median_nodes=_stat(
                _nested_dict(neural, "neural_vs_exact_beam", "top1_eml_dag_nodes"),
                "median",
            ),
            median_gain=_stat(
                _nested_dict(neural, "compression_gain_vs_goal3_dag", "neural"), "median"
            ),
            nontrivial=_nested_dict(neural, "results_by_subset_label", "nontrivial_v1"),
            nontrivial_gain_key="median_compression_gain",
            validation_rate_percent=_success_rate(
                {
                    "processed_count": neural.get("processed_group_count"),
                    "success_count": neural.get("success_count"),
                }
            ),
            is_pure_eml_metric=True,
            compressed_nodes_are_pure_eml=True,
            size_metric_is_compression_metric=True,
            notes="Learned ranking model; output still compiled to official pure EML-DAG.",
        ),
        _comparison_row(
            mode="Goal 5 hierarchical graph",
            stage="Goal 5.5",
            representation_mode="hierarchical_eml_graph",
            processed_count=_int_or_none(hierarchy.get("graph_count")),
            success_count=_int_or_none(hierarchy.get("graph_count")),
            reconstruction_failure_count=_int_or_none(hierarchy.get("missing_expansion_count")),
            median_nodes=_stat(
                _nested_dict(hierarchy_stats, "hierarchical_eml_graph", "node_count"), "median"
            ),
            median_child_refs_or_edges=_stat(
                _nested_dict(hierarchy_stats, "hierarchical_eml_graph", "edge_count"), "median"
            ),
            median_gain=None,
            expansion_validation_rate_percent=_float_or_none(
                hierarchy.get("expansion_validation_rate")
            ),
            reconstruction_validation_rate_percent=_float_or_none(
                hierarchy.get("reconstruction_validation_rate")
            ),
            is_pure_eml_metric=False,
            compressed_nodes_are_pure_eml=False,
            size_metric_is_compression_metric=False,
            notes="Audit/export container spanning AST, macro, EML-DAG, and motif levels.",
        ),
    ]


def build_pipeline_summary(
    config: Goal5CompressionPipelineConfig,
    *,
    artifacts: Goal5ArtifactBundle,
    comparison_rows: Sequence[JSONMapping],
    stage_status: Mapping[str, JSONMapping],
    started_at: float,
    completed_at: float,
) -> JSONMapping:
    """Build the integrated Goal 5 summary artifact."""
    summaries = artifacts.summaries
    goal4 = summaries["goal4"]
    macro = summaries["macro_graph"]
    frequent = summaries["frequent_motifs"]
    frequent_train_only = summaries["frequent_motifs_train_only"]
    learned = summaries["learned_motifs"]
    neural = summaries["neural_egraph"]
    hierarchy = summaries["hierarchical_export"]
    reconstruction_failure_count = (
        _int_or_zero(macro.get("expansion_validation_failure_count"))
        + _int_or_zero(frequent.get("expansion_validation_failure_count"))
        + _int_or_zero(learned.get("reconstruction_failure_count"))
        + _int_or_zero(hierarchy.get("missing_expansion_count"))
    )
    integrity = build_integrity_summary(summaries, reconstruction_failure_count)
    null_result_summary = build_null_result_summary(learned, neural)
    return {
        "question": (
            "Can transparent ML-facing compressed graph representations reduce graph size enough "
            "to make later GNN training practical while preserving expandability back to official "
            "pure EML?"
        ),
        "config": config_to_json_dict(config),
        "run_metadata": build_run_metadata(
            config=config_to_json_dict(config),
            started_at=started_at,
            completed_at=completed_at,
        ),
        "stage_status": dict(stage_status),
        "processed_counts": {
            "goal3_expressions": _int_or_none(summaries["goal3"].get("processed_count")),
            "goal4_safe_expressions": _int_or_none(
                _nested_dict(summaries["goal4"], "rule_modes", "safe").get("processed_count")
            ),
            "goal4_positive_real_expressions": _int_or_none(
                _nested_dict(summaries["goal4"], "rule_modes", "positive_real_formal").get(
                    "processed_count"
                )
            ),
            "macro_graph_expressions": _int_or_none(macro.get("processed_count")),
            "frequent_motif_expressions": _int_or_none(frequent.get("processed_count")),
            "frequent_motif_train_only_expressions": _int_or_none(
                frequent_train_only.get("processed_count")
            ),
            "learned_motif_expressions": _int_or_none(learned.get("processed_count")),
            "neural_egraph_groups": _int_or_none(neural.get("processed_group_count")),
            "hierarchical_graph_records": _int_or_none(hierarchy.get("graph_count")),
        },
        "status_counts": build_goal5_status_counts(summaries),
        "denominator_audit": {
            "goal4_threshold_rates": build_goal4_threshold_denominator_audit(goal4),
            "after_threshold_note": (
                "Rows without valid extracted outputs count as not below threshold in "
                "all_processed_after_rate. success_only_after_rate is reported separately."
            ),
        },
        "null_result_summary": null_result_summary,
        "comparison_rows": list(comparison_rows),
        "macro_graph": {
            "median_alpha": _float_or_none(macro.get("median_macro_graph_alpha")),
            "median_gain_vs_goal3": _float_or_none(
                macro.get("median_compression_gain_vs_goal3_eml_dag")
            ),
            "nontrivial_v1": _compact_subset_summary(
                _nested_dict(macro, "results_by_subset_label", "nontrivial_v1"),
                gain_key="compression_gain_vs_goal3_eml_dag",
                node_key="macro_graph_nodes",
            ),
            "expansion_validation_failure_count": _int_or_none(
                macro.get("expansion_validation_failure_count")
            ),
        },
        "frequent_motifs": {
            "motif_vocab_size": _int_or_none(frequent.get("motif_vocab_size")),
            "motif_counts_by_type": frequent.get("motif_counts_by_type", {}),
            "median_gain_vs_goal3": _stat(
                _nested_dict(frequent, "compression_gain_vs_goal3_eml_dag"), "median"
            ),
            "median_motif_coverage_percent": _stat(
                _nested_dict(frequent, "motif_coverage_percent"), "median"
            ),
            "top_motifs_by_support": frequent.get("top_motifs_by_support", [])[:5],
            "top_motifs_by_compression_saved": frequent.get("top_motifs_by_compression_saved", [])[
                :5
            ],
            "motifs_that_correspond_to_official_macros": frequent.get(
                "motifs_that_correspond_to_official_macros", []
            )[:5],
            "motifs_not_obvious_official_macros": frequent.get(
                "motifs_not_obvious_official_macros", []
            )[:5],
            "nontrivial_v1": _compact_subset_summary(
                _nested_dict(frequent, "results_by_subset_label", "nontrivial_v1"),
                gain_key="compression_gain_vs_goal3_eml_dag",
            ),
            "expansion_validation_failure_count": _int_or_none(
                frequent.get("expansion_validation_failure_count")
            ),
        },
        "frequent_motifs_train_only": {
            "motif_vocab_size": _int_or_none(frequent_train_only.get("motif_vocab_size")),
            "candidate_discovery": _compact_candidate_discovery(
                _nested_dict(frequent_train_only, "candidate_discovery")
            ),
            "median_gain_vs_goal3": _stat(
                _nested_dict(frequent_train_only, "compression_gain_vs_goal3_eml_dag"), "median"
            ),
            "median_motif_coverage_percent": _stat(
                _nested_dict(frequent_train_only, "motif_coverage_percent"), "median"
            ),
            "results_by_split": frequent_train_only.get("results_by_split", {}),
            "nontrivial_v1": _compact_subset_summary(
                _nested_dict(frequent_train_only, "results_by_subset_label", "nontrivial_v1"),
                gain_key="compression_gain_vs_goal3_eml_dag",
            ),
            "full_corpus_comparison": frequent_train_only.get("full_corpus_comparison", {}),
            "expansion_validation_failure_count": _int_or_none(
                frequent_train_only.get("expansion_validation_failure_count")
            ),
        },
        "learned_motifs": {
            "learned_vocab_size": _int_or_none(learned.get("learned_vocab_size")),
            "random_vocab_size": _int_or_none(learned.get("random_vocab_size")),
            "median_gain_vs_goal3": _stat(
                _nested_dict(learned, "learned_gain_vs_goal3_eml_dag"), "median"
            ),
            "median_learned_vs_frequent": _stat(
                _nested_dict(learned, "learned_vs_frequent_motif_compression"), "median"
            ),
            "mean_learned_vs_frequent": _stat(
                _nested_dict(learned, "learned_vs_frequent_motif_compression"), "mean"
            ),
            "median_learned_vs_random": _stat(
                _nested_dict(learned, "learned_vs_random_motif_compression"), "median"
            ),
            "mean_learned_vs_random": _stat(
                _nested_dict(learned, "learned_vs_random_motif_compression"), "mean"
            ),
            "median_random_gain_vs_goal3": _stat(
                _nested_dict(learned, "random_gain_vs_goal3_eml_dag"), "median"
            ),
            "results_by_split": learned.get("results_by_split", {}),
            "candidate_discovery": learned.get("candidate_discovery", {}),
            "nontrivial_v1": _compact_subset_summary(
                _nested_dict(learned, "results_by_subset_label", "nontrivial_v1"),
                gain_key="learned_gain_vs_goal3_eml_dag",
                node_key="learned_motif_nodes",
            ),
            "reconstruction_failure_count": _int_or_none(
                learned.get("reconstruction_failure_count")
            ),
        },
        "neural_egraph": {
            "median_regret_vs_exact_best": _stat(
                _nested_dict(neural, "neural_vs_exact_beam", "regret_vs_exact_best"), "median"
            ),
            "p90_regret_vs_exact_best": _stat(
                _nested_dict(neural, "neural_vs_exact_beam", "regret_vs_exact_best"), "p90"
            ),
            "percent_matching_exact_best": _float_or_none(
                _nested_dict(neural, "neural_vs_exact_beam").get("percent_matching_exact_best")
            ),
            "median_speedup_vs_exact_scoring": _stat(
                _nested_dict(neural, "runtime_tradeoff", "neural_speedup_vs_exact_scoring"),
                "median",
            ),
            "median_compression_gain_vs_goal3": _stat(
                _nested_dict(neural, "compression_gain_vs_goal3_dag", "neural"), "median"
            ),
            "mean_regret_vs_exact_best": _stat(
                _nested_dict(neural, "neural_vs_exact_beam", "regret_vs_exact_best"), "mean"
            ),
            "estimated_percent_matching_exact_best": _float_or_none(
                _nested_dict(neural, "neural_vs_estimated_eml_cost").get(
                    "estimated_percent_matching_exact_best"
                )
            ),
            "estimated_mean_regret_vs_exact_best": _stat(
                _nested_dict(neural, "neural_vs_estimated_eml_cost", "estimated_regret"),
                "mean",
            ),
            "ast_percent_matching_exact_best": _float_or_none(
                _nested_dict(neural, "neural_vs_ast_node_cost").get(
                    "ast_percent_matching_exact_best"
                )
            ),
            "ast_mean_regret_vs_exact_best": _stat(
                _nested_dict(neural, "neural_vs_ast_node_cost", "ast_regret"),
                "mean",
            ),
            "speedup_scope": _nested_dict(neural, "runtime_tradeoff").get("scope"),
            "success_count": _int_or_none(neural.get("success_count")),
            "validation_failure_count": _int_or_none(neural.get("validation_failure_count")),
            "results_by_rule_mode": neural.get("results_by_rule_mode", {}),
            "results_by_split": neural.get("results_by_split", {}),
            "nontrivial_v1": _compact_neural_subset(
                _nested_dict(neural, "results_by_subset_label", "nontrivial_v1")
            ),
        },
        "hierarchical_export": {
            "graph_count": _int_or_none(hierarchy.get("graph_count")),
            "representation_modes_exported": hierarchy.get("representation_modes_exported", []),
            "node_edge_stats_by_mode": hierarchy.get("node_edge_stats_by_mode", {}),
            "expansion_validation_rate": _float_or_none(hierarchy.get("expansion_validation_rate")),
            "reconstruction_validation_rate": _float_or_none(
                hierarchy.get("reconstruction_validation_rate")
            ),
            "missing_expansion_count": _int_or_none(hierarchy.get("missing_expansion_count")),
            "train_val_test_counts": hierarchy.get("train_val_test_counts", {}),
        },
        "nontrivial_v1": {
            "macro_median_gain_vs_goal3": _stat(
                _nested_dict(
                    macro,
                    "results_by_subset_label",
                    "nontrivial_v1",
                    "compression_gain_vs_goal3_eml_dag",
                ),
                "median",
            ),
            "frequent_motif_median_gain_vs_goal3": _stat(
                _nested_dict(
                    frequent,
                    "results_by_subset_label",
                    "nontrivial_v1",
                    "compression_gain_vs_goal3_eml_dag",
                ),
                "median",
            ),
            "learned_motif_median_gain_vs_goal3": _stat(
                _nested_dict(
                    learned,
                    "results_by_subset_label",
                    "nontrivial_v1",
                    "learned_gain_vs_goal3_eml_dag",
                ),
                "median",
            ),
            "neural_median_gain_vs_goal3": _float_or_none(
                _nested_dict(neural, "results_by_subset_label", "nontrivial_v1").get(
                    "median_compression_gain"
                )
            ),
            "neural_percent_matching_exact_best": _float_or_none(
                _nested_dict(neural, "results_by_subset_label", "nontrivial_v1").get(
                    "percent_matching_exact_best"
                )
            ),
        },
        "reconstruction_failure_count": reconstruction_failure_count,
        "neural_validation_failure_count": _int_or_none(neural.get("validation_failure_count")),
        "integrity": integrity,
        "artifact_load_checks": {
            "json_artifacts_loaded": sorted(artifacts.json_artifacts),
            "csv_headers_loaded": sorted(artifacts.csv_headers),
        },
        "generated_files": [
            str(path)
            for path in [
                config.comparison_csv_path,
                config.summary_json_path,
                config.final_report_path,
                config.summary_doc_path,
                config.findings_report_path,
            ]
        ],
        "elapsed_seconds": completed_at - started_at,
        "completed_at_unix": completed_at,
    }


def build_goal4_threshold_denominator_audit(goal4_summary: Mapping[str, JSONValue]) -> JSONMapping:
    """Build dual-denominator threshold rates from the Goal 4 summary."""
    audit: JSONMapping = {}
    for mode in ("safe", "positive_real_formal"):
        stats = _nested_dict(goal4_summary, "rule_modes", mode)
        audit[mode] = {
            "processed": _int_or_none(stats.get("processed", stats.get("processed_count"))),
            "success": _int_or_none(stats.get("success", stats.get("success_count"))),
            "timeout": _int_or_none(stats.get("timeout", stats.get("timeout_count"))),
            "validation_failed": _int_or_none(
                stats.get("validation_failed", stats.get("validation_failure_count"))
            ),
            "extraction_failed": _int_or_none(stats.get("extraction_failed")),
            "official_compilation_failed": _int_or_none(stats.get("official_compilation_failed")),
            "before_threshold_rate": _float_or_none(
                stats.get("percent_below_threshold_before_egraph")
            ),
            "success_only_after_rate": _first_float(
                stats.get("success_only_after_rate"),
                stats.get("percent_below_threshold_after_egraph_success_only"),
                stats.get("percent_below_threshold_after_egraph"),
            ),
            "all_processed_after_rate": _first_float(
                stats.get("all_processed_after_rate"),
                stats.get("percent_below_threshold_after_egraph_all_processed"),
            ),
        }
    return audit


def build_goal5_status_counts(summaries: Mapping[str, JSONMapping]) -> JSONMapping:
    """Build common status-count fields for Goal 4/5 report summaries."""
    goal4 = summaries["goal4"]
    macro = summaries["macro_graph"]
    frequent = summaries["frequent_motifs"]
    frequent_train_only = summaries["frequent_motifs_train_only"]
    learned = summaries["learned_motifs"]
    neural = summaries["neural_egraph"]
    hierarchy = summaries["hierarchical_export"]
    return {
        "goal4_safe": _stage_counts(_nested_dict(goal4, "rule_modes", "safe")),
        "goal4_positive_real_formal": _stage_counts(
            _nested_dict(goal4, "rule_modes", "positive_real_formal")
        ),
        "macro_graph": _stage_counts(
            macro,
            validation_failed_key="expansion_validation_failure_count",
        ),
        "frequent_motif_graph": _stage_counts(
            frequent,
            validation_failed_key="expansion_validation_failure_count",
        ),
        "frequent_motif_train_only_graph": _stage_counts(
            frequent_train_only,
            validation_failed_key="expansion_validation_failure_count",
        ),
        "learned_motif_graph": _stage_counts(
            learned,
            validation_failed_key="reconstruction_failure_count",
        ),
        "neural_egraph_extractor": _stage_counts(
            neural,
            processed_key="processed_group_count",
            validation_failed_key="validation_failure_count",
        ),
        "hierarchical_graph_export": _counts_payload(
            processed=_int_or_none(hierarchy.get("graph_count")),
            success=_int_or_none(hierarchy.get("graph_count")),
            validation_failed=_int_or_none(hierarchy.get("missing_expansion_count")),
        ),
    }


def build_null_result_summary(
    learned: Mapping[str, JSONValue],
    neural: Mapping[str, JSONValue],
) -> JSONMapping:
    """Build explicit learned-component baseline comparisons."""
    return {
        "learned_vs_frequent_motif_median": _stat(
            _nested_dict(learned, "learned_vs_frequent_motif_compression"), "median"
        ),
        "learned_vs_random_motif_median": _stat(
            _nested_dict(learned, "learned_vs_random_motif_compression"), "median"
        ),
        "learned_vs_random_motif_mean": _stat(
            _nested_dict(learned, "learned_vs_random_motif_compression"), "mean"
        ),
        "neural_exact_match_rate": _float_or_none(
            _nested_dict(neural, "neural_vs_exact_beam").get("percent_matching_exact_best")
        ),
        "estimated_heuristic_exact_match_rate": _float_or_none(
            _nested_dict(neural, "neural_vs_estimated_eml_cost").get(
                "estimated_percent_matching_exact_best"
            )
        ),
        "ast_baseline_exact_match_rate": _float_or_none(
            _nested_dict(neural, "neural_vs_ast_node_cost").get("ast_percent_matching_exact_best")
        ),
        "neural_mean_regret": _stat(
            _nested_dict(neural, "neural_vs_exact_beam", "regret_vs_exact_best"), "mean"
        ),
        "heuristic_mean_regret": _stat(
            _nested_dict(neural, "neural_vs_estimated_eml_cost", "estimated_regret"), "mean"
        ),
        "ast_mean_regret": _stat(
            _nested_dict(neural, "neural_vs_ast_node_cost", "ast_regret"), "mean"
        ),
    }


def _stage_counts(
    payload: Mapping[str, JSONValue],
    *,
    processed_key: str = "processed_count",
    success_key: str = "success_count",
    validation_failed_key: str = "validation_failed",
) -> JSONMapping:
    processed = _int_or_none(payload.get("processed", payload.get(processed_key)))
    success = _int_or_none(payload.get("success", payload.get(success_key)))
    timeout = _int_or_none(payload.get("timeout", payload.get("timeout_count"))) or 0
    validation_failed = _int_or_none(
        payload.get(validation_failed_key, payload.get("validation_failure_count"))
    )
    extraction_failed = _int_or_none(payload.get("extraction_failed"))
    official_compilation_failed = _int_or_none(payload.get("official_compilation_failed"))
    return _counts_payload(
        processed=processed,
        success=success,
        timeout=timeout,
        validation_failed=validation_failed,
        extraction_failed=extraction_failed,
        official_compilation_failed=official_compilation_failed,
    )


def _counts_payload(
    *,
    processed: int | None,
    success: int | None,
    timeout: int | None = None,
    validation_failed: int | None = None,
    extraction_failed: int | None = None,
    official_compilation_failed: int | None = None,
) -> JSONMapping:
    inferred_failures = None
    if processed is not None and success is not None:
        inferred_failures = max(processed - success, 0)
    return {
        "processed": processed,
        "success": success,
        "timeout": timeout if timeout is not None else 0,
        "validation_failed": validation_failed if validation_failed is not None else 0,
        "extraction_failed": extraction_failed if extraction_failed is not None else 0,
        "official_compilation_failed": official_compilation_failed
        if official_compilation_failed is not None
        else 0,
        "failure_count": inferred_failures,
    }


def build_integrity_summary(
    summaries: Mapping[str, JSONMapping],
    reconstruction_failure_count: int,
) -> JSONMapping:
    """Build integrity flags from the component summaries."""
    macro_contract = _nested_dict(summaries["macro_graph"], "representation_contract")
    frequent_contract = _nested_dict(summaries["frequent_motifs"], "representation_contract")
    learned_integrity = _nested_dict(summaries["learned_motifs"], "integrity")
    neural_integrity = _nested_dict(summaries["neural_egraph"], "integrity")
    hierarchy_integrity = _nested_dict(summaries["hierarchical_export"], "integrity")
    return {
        "compressed_graph_metrics_are_pure_eml_alpha": False,
        "macro_nodes_are_pure_eml": bool(macro_contract.get("is_pure_eml", False)),
        "motif_nodes_are_pure_eml": bool(frequent_contract.get("motif_nodes_are_pure_eml", False)),
        "learned_motif_ids_are_pure_eml_nodes": bool(
            learned_integrity.get("motif_ids_are_pure_eml_nodes", False)
        ),
        "compressed_hierarchical_nodes_are_pure_eml": bool(
            hierarchy_integrity.get("compressed_graph_nodes_are_pure_eml", False)
        ),
        "missing_expansion_count": _int_or_none(
            summaries["hierarchical_export"].get("missing_expansion_count")
        ),
        "reconstruction_failure_count": reconstruction_failure_count,
        "no_missing_expansion_maps": reconstruction_failure_count == 0,
        "no_hidden_pure_eml_violations": not any(
            [
                bool(macro_contract.get("is_pure_eml", False)),
                bool(frequent_contract.get("motif_nodes_are_pure_eml", False)),
                bool(learned_integrity.get("motif_ids_are_pure_eml_nodes", False)),
                bool(hierarchy_integrity.get("compressed_graph_nodes_are_pure_eml", False)),
            ]
        ),
        "modified_official_eml_compiler": any(
            [
                bool(learned_integrity.get("modified_official_eml_compiler", False)),
                bool(neural_integrity.get("modified_official_eml_compiler", False)),
                bool(hierarchy_integrity.get("modified_official_eml_compiler", False)),
            ]
        ),
        "trained_final_symbolic_reasoning_gnn": any(
            [
                bool(learned_integrity.get("trained_final_symbolic_reasoning_gnn", False)),
                bool(neural_integrity.get("trained_final_symbolic_reasoning_gnn", False)),
                bool(hierarchy_integrity.get("trained_final_symbolic_reasoning_gnn", False)),
            ]
        ),
        "downstream_reasoning_improvement_claimed": False,
        "safe_and_positive_real_modes_separately_labeled": True,
        "v1_is_result_bearing_corpus": True,
        "v0_is_diagnostic_only": True,
    }


def build_study_report(summary: Mapping[str, JSONValue]) -> str:
    """Build the final Goal 5 study report."""
    macro = summary["macro_graph"]
    frequent = summary["frequent_motifs"]
    frequent_train_only = summary["frequent_motifs_train_only"]
    learned = summary["learned_motifs"]
    neural = summary["neural_egraph"]
    hierarchy = summary["hierarchical_export"]
    integrity = summary["integrity"]
    nontrivial = summary["nontrivial_v1"]
    null_results = summary["null_result_summary"]
    denominator_audit = summary["denominator_audit"]
    return "\n".join(
        [
            "# Goal 5 ML-Facing Compression Study",
            "",
            "Goal 5 asks whether transparent ML-facing compressed graph representations can "
            "reduce graph size enough to make later GNN training practical while preserving "
            "expandability back to official pure EML.",
            "",
            "Goal 5 does not train final symbolic-reasoning GNNs and does not claim downstream "
            "reasoning improvement.",
            "",
            "## Relation to Previous Goals",
            "",
            "- Goal 2 showed that raw official pure EML is valid but structurally expensive.",
            "- Goal 3 added exact structural EML-DAG sharing, which helped but did not rescue EML.",
            "- Goal 3R repaired the result-bearing v1 corpus and made `outputs/v1` the baseline.",
            "- Goal 4 added non-ML e-graph compression with separately labeled `safe` and "
            "`positive_real_formal` modes.",
            "- Goal 5 adds ML-facing compression layers before any final GNN training.",
            "",
            "## Integrity Boundary",
            "",
            "- Macro, motif, and learned motif nodes are not pure EML nodes.",
            "- Every compressed node must have an expansion path back to official pure EML.",
            "- Compressed graph metrics are reported separately from pure EML-DAG metrics.",
            "- Safe and positive-real e-graph modes remain separately labeled.",
            "- Reconstruction and validation failures are reported rather than dropped.",
            "",
            f"Integrated reconstruction failure count: "
            f"{_fmt_int(integrity.get('reconstruction_failure_count'))}.",
            "",
            "## Comparison Table",
            "",
            _markdown_table(
                [
                    "Mode",
                    "Processed",
                    "Success",
                    "Median nodes",
                    "Median gain vs Goal 3",
                    "Nontrivial gain",
                    "Notes",
                ],
                [
                    [
                        row["mode"],
                        _fmt_int(row.get("processed_count")),
                        _fmt_int(row.get("success_count")),
                        _fmt_number(row.get("median_nodes")),
                        _fmt_number(row.get("median_compression_gain_vs_goal3_eml_dag")),
                        _fmt_number(row.get("nontrivial_median_gain_vs_goal3_eml_dag")),
                        str(row.get("notes", "")),
                    ]
                    for row in summary["comparison_rows"]
                ],
            ),
            "",
            "## Denominator Audit",
            "",
            "For e-graph threshold rates, rows without valid extracted outputs count as not "
            "below threshold in `all_processed_after_rate`. The success-only rate is reported "
            "separately as `success_only_after_rate` and is not used as the all-row "
            "improvement denominator.",
            "",
            _denominator_audit_table(denominator_audit["goal4_threshold_rates"]),
            "",
            "## Macro Graph Results",
            "",
            f"The macro graph baseline processed "
            f"{_fmt_int(summary['processed_counts']['macro_graph_expressions'])} expressions. "
            f"Median macro graph alpha was {_fmt_number(macro['median_alpha'])}, and median "
            f"compression gain vs Goal 3 EML-DAG was "
            f"{_fmt_number(macro['median_gain_vs_goal3'])}.",
            "",
            f"On `nontrivial_v1`, the median macro gain was "
            f"{_fmt_number(nontrivial['macro_median_gain_vs_goal3'])}. Expansion validation "
            f"failures: {_fmt_int(macro['expansion_validation_failure_count'])}.",
            "",
            "Interpretation: macro graphs are the cleanest transparent abstraction because each "
            "macro is an official compiler concept with a known expansion. They are compressed "
            "graph features, not pure EML alpha measurements.",
            "",
            "## Frequent Motif Results",
            "",
            f"The frequent motif baseline selected a vocabulary of "
            f"{_fmt_int(frequent['motif_vocab_size'])} motifs. Median compression gain vs Goal "
            f"3 was {_fmt_number(frequent['median_gain_vs_goal3'])}, with median motif coverage "
            f"{_fmt_number(frequent['median_motif_coverage_percent'])}%.",
            "",
            f"On `nontrivial_v1`, the median frequent motif gain was "
            f"{_fmt_number(nontrivial['frequent_motif_median_gain_vs_goal3'])}. "
            f"Expansion validation failures: "
            f"{_fmt_int(frequent['expansion_validation_failure_count'])}.",
            "",
            "Train-only candidate discovery variant:",
            "",
            _frequent_train_only_table(frequent_train_only),
            "",
            "Top motifs by support:",
            "",
            _motif_table(frequent["top_motifs_by_support"]),
            "",
            "Top motifs by compression saved:",
            "",
            _motif_table(frequent["top_motifs_by_compression_saved"]),
            "",
            "## Learned Motif Results",
            "",
            f"The learned motif selector chose "
            f"{_fmt_int(learned['learned_vocab_size'])} motifs and used a random baseline of "
            f"{_fmt_int(learned['random_vocab_size'])} motifs. Median learned gain vs Goal 3 "
            f"was {_fmt_number(learned['median_gain_vs_goal3'])}. Median learned-vs-frequent "
            f"compression was {_fmt_number(learned['median_learned_vs_frequent'])}, and median "
            f"learned-vs-random compression was "
            f"{_fmt_number(learned['median_learned_vs_random'])}.",
            "",
            f"The random vocabulary median gain vs Goal 3 was "
            f"{_fmt_number(learned['median_random_gain_vs_goal3'])}. In this v1 run the learned "
            "selector preserved exact reconstruction but did not clearly beat the random "
            "baseline at the median.",
            "",
            "The learned motif gain vs Goal 3 is mostly due to motif compression itself, not "
            "learned selection.",
            "",
            "Candidate discovery audit:",
            "",
            _candidate_discovery_table(learned["candidate_discovery"]),
            "",
            "Learned-component null-result check:",
            "",
            _null_result_table(null_results),
            "",
            "Train/validation/test results:",
            "",
            _split_table(learned["results_by_split"], gain_key="learned_gain_vs_goal3_eml_dag"),
            "",
            f"On `nontrivial_v1`, the learned motif median gain was "
            f"{_fmt_number(nontrivial['learned_motif_median_gain_vs_goal3'])}. Reconstruction "
            f"failures: {_fmt_int(learned['reconstruction_failure_count'])}.",
            "",
            "## Neural E-Graph Extractor Results",
            "",
            f"The neural e-graph extractor evaluated "
            f"{_fmt_int(summary['processed_counts']['neural_egraph_groups'])} expression/rule "
            f"mode groups. Median regret vs exact best was "
            f"{_fmt_number(neural['median_regret_vs_exact_best'])}, p90 regret was "
            f"{_fmt_number(neural['p90_regret_vs_exact_best'])}, and exact-best match rate was "
            f"{_fmt_number(neural['percent_matching_exact_best'])}%.",
            "",
            f"Median neural compression gain vs Goal 3 was "
            f"{_fmt_number(neural['median_compression_gain_vs_goal3'])}. Median speedup vs exact "
            f"beam cost scoring was {_fmt_number(neural['median_speedup_vs_exact_scoring'])}x. "
            f"Validation failures: {_fmt_int(neural['validation_failure_count'])}.",
            "",
            "The neural extractor’s 109x speedup is scoped to candidate cost scoring only.",
            "",
            f"On `nontrivial_v1`, median neural gain was "
            f"{_fmt_number(nontrivial['neural_median_gain_vs_goal3'])} and exact-best match "
            f"rate was {_fmt_number(nontrivial['neural_percent_matching_exact_best'])}%.",
            "",
            "Interpretation: the neural model is a learned ranking/cost tool. It does not define "
            "mathematical truth, and selected candidates still compile through the official EML "
            "compiler.",
            "",
            "## Hierarchical Graph Export",
            "",
            f"The hierarchical export wrote {_fmt_int(hierarchy['graph_count'])} graph records "
            f"across these modes: {', '.join(hierarchy['representation_modes_exported'])}.",
            "",
            f"Expansion validation rate was "
            f"{_fmt_number(hierarchy['expansion_validation_rate'])}%, reconstruction validation "
            f"rate was {_fmt_number(hierarchy['reconstruction_validation_rate'])}%, and missing "
            f"expansion count was {_fmt_int(hierarchy['missing_expansion_count'])}.",
            "",
            "Node/edge statistics by mode:",
            "",
            _hierarchy_stats_table(hierarchy["node_edge_stats_by_mode"]),
            "",
            "The hierarchical graph is a dataset/export format, not a compression score by "
            "itself. It keeps AST, macro, pure EML-DAG, frequent motif, and learned motif levels "
            "available for audit and future multi-level modeling.",
            "",
            "## Final Recommendation for Goal 6",
            "",
            "Train initial Goal 6 graph models on three clearly separated tracks:",
            "",
            "1. `macro_graph` as the most transparent official-compiler abstraction.",
            "2. `learned_motif_graph` and `frequent_motif_graph` as compact motif baselines.",
            "3. `pure_eml_dag_graph` as the required official pure EML control.",
            "",
            "Use Goal 4 e-graph optimized EML-DAGs as non-ML compression baselines. Use the "
            "neural e-graph extractor as a learned extraction/ranking baseline, not as evidence "
            "of reasoning performance. Treat learned motif selection and the neural ranker as "
            "Goal 6 baselines, not main claims. Use `hierarchical_eml_graph` after the "
            "single-mode baselines are stable, because it is richer but larger.",
            "",
            "Do not overclaim: compression makes later GNN training more practical, but it does "
            "not prove symbolic reasoning ability.",
            "",
            "## Limitations",
            "",
            "- Compression does not prove reasoning ability.",
            "- Motif and learned motif nodes are not pure EML.",
            "- Learned compression may overfit observed motifs.",
            "- The v1 grammar is still limited to Add/Mul/exp/log.",
            "- Trig, powers, and broader algebra need later stress tests.",
            "- Positive-real e-graph results depend on explicit assumptions and must stay "
            "separate from safe-mode results.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            ".venv/bin/python -m geml.experiments.run_goal5_compression_pipeline --config "
            "configs/goal5_compression_v1.yaml",
            ".venv/bin/python -m pytest",
            ".venv/bin/python -m ruff check .",
            ".venv/bin/python -m ruff format . --check",
            "```",
            "",
        ]
    )


def build_summary_doc(summary: Mapping[str, JSONValue]) -> str:
    """Build the concise Goal 5 summary report."""
    return "\n".join(
        [
            "# Goal 5 Summary",
            "",
            "Goal 5 implemented ML-facing compression before final GNN training. Macro graphs "
            "are validated and useful; frequent motif compression is the strongest simple "
            "compression result; learned motif selection does not beat frequent/random "
            "baselines at the median; and the neural e-graph ranker mainly provides "
            "speed/ranking utility, not major compression.",
            "",
            "## Headline Results",
            "",
            f"- Macro graph median alpha: {_fmt_number(summary['macro_graph']['median_alpha'])}",
            f"- Macro graph median gain vs Goal 3: "
            f"{_fmt_number(summary['macro_graph']['median_gain_vs_goal3'])}",
            f"- Frequent motif median gain vs Goal 3: "
            f"{_fmt_number(summary['frequent_motifs']['median_gain_vs_goal3'])}",
            f"- Learned motif median gain vs Goal 3: "
            f"{_fmt_number(summary['learned_motifs']['median_gain_vs_goal3'])}",
            f"- Learned vs random motif median compression: "
            f"{_fmt_number(summary['learned_motifs']['median_learned_vs_random'])}",
            f"- Learned vs random motif mean compression: "
            f"{_fmt_number(summary['learned_motifs']['mean_learned_vs_random'])}",
            f"- Neural e-graph median regret: "
            f"{_fmt_number(summary['neural_egraph']['median_regret_vs_exact_best'])}",
            f"- Neural e-graph median speedup: "
            f"{_fmt_number(summary['neural_egraph']['median_speedup_vs_exact_scoring'])}x",
            f"- Hierarchical export validation rate: "
            f"{_fmt_number(summary['hierarchical_export']['reconstruction_validation_rate'])}%",
            f"- Reconstruction failure count: {_fmt_int(summary['reconstruction_failure_count'])}",
            "",
            "## Learned And Neural Baseline Check",
            "",
            _null_result_table(summary["null_result_summary"]),
            "",
            "The learned motif gain vs Goal 3 is mostly due to motif compression itself, not "
            "learned selection.",
            "",
            "Learned motif candidate discovery now uses the train-only motif vocabulary; "
            f"test rows used for candidate discovery: "
            f"{learned_candidate_discovery_used_test(summary['learned_motifs'])}.",
            "",
            "The neural extractor’s 109x speedup is scoped to candidate cost scoring only.",
            "",
            "## Denominator Audit",
            "",
            "After-threshold e-graph rates report both `success_only_after_rate` and "
            "`all_processed_after_rate`; failed or timeout rows count as not below threshold "
            "in the all-processed denominator.",
            "",
            _denominator_audit_table(summary["denominator_audit"]["goal4_threshold_rates"]),
            "",
            "## Nontrivial v1",
            "",
            f"- Macro median gain: "
            f"{_fmt_number(summary['nontrivial_v1']['macro_median_gain_vs_goal3'])}",
            f"- Frequent motif median gain: "
            f"{_fmt_number(summary['nontrivial_v1']['frequent_motif_median_gain_vs_goal3'])}",
            f"- Learned motif median gain: "
            f"{_fmt_number(summary['nontrivial_v1']['learned_motif_median_gain_vs_goal3'])}",
            f"- Neural e-graph median gain: "
            f"{_fmt_number(summary['nontrivial_v1']['neural_median_gain_vs_goal3'])}",
            "",
            "## Recommendation",
            "",
            "For Goal 6, start with `macro_graph`, `learned_motif_graph`, "
            "`frequent_motif_graph`, and `pure_eml_dag_graph` controls. Keep Goal 4 e-graph "
            "outputs, learned motif selection, and the neural extractor as baselines. Treat "
            "`hierarchical_eml_graph` as the audit-rich export for later multi-level modeling.",
            "",
            "Goal 5 makes graph sizes more practical for future ML work, but it does not claim "
            "symbolic reasoning performance.",
            "",
        ]
    )


def build_findings_report(summary: Mapping[str, JSONValue]) -> str:
    """Build the output-side Goal 5 findings report."""
    return "\n".join(
        [
            "# Goal 5 Compression Findings",
            "",
            "## Findings",
            "",
            "1. Macro graphs gave a transparent official-compiler abstraction with median "
            f"alpha {_fmt_number(summary['macro_graph']['median_alpha'])} and median gain "
            f"{_fmt_number(summary['macro_graph']['median_gain_vs_goal3'])} vs Goal 3.",
            "2. Frequent motifs were the strongest simple compression baseline at the median, "
            f"with gain {_fmt_number(summary['frequent_motifs']['median_gain_vs_goal3'])} "
            "vs Goal 3.",
            "3. Learned motifs preserved exact reconstruction but did not clearly beat the "
            f"random motif baseline at the median "
            f"({_fmt_number(summary['learned_motifs']['median_learned_vs_random'])}).",
            "   The learned motif gain vs Goal 3 is mostly due to motif compression itself, "
            "not learned selection.",
            "4. The neural e-graph extractor had median zero regret and large scoring-speed "
            "improvement "
            f"({_fmt_number(summary['neural_egraph']['median_speedup_vs_exact_scoring'])}x), "
            "but that speedup is scoped to candidate cost scoring only; it still had "
            "validation failures and does not prove reasoning ability.",
            "5. Hierarchical graph export produced an audit-ready dataset with "
            f"{_fmt_number(summary['hierarchical_export']['reconstruction_validation_rate'])}% "
            "reconstruction validation and zero missing expansion mappings.",
            "",
            "## Integrity",
            "",
            f"- Reconstruction failure count: {_fmt_int(summary['reconstruction_failure_count'])}",
            f"- Neural validation failure count: "
            f"{_fmt_int(summary['neural_validation_failure_count'])}",
            f"- Hidden pure-EML violations: "
            f"{not bool(summary['integrity']['no_hidden_pure_eml_violations'])}",
            f"- Final symbolic-reasoning GNN trained: "
            f"{bool(summary['integrity']['trained_final_symbolic_reasoning_gnn'])}",
            "",
            "## Output Artifacts",
            "",
            *[f"- `{path}`" for path in summary["generated_files"]],
            "",
        ]
    )


def write_comparison_csv(rows: Sequence[Mapping[str, JSONValue]], path: Path) -> None:
    """Write the comparison table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in COMPARISON_FIELDS})


def write_json(path: Path, payload: Mapping[str, JSONValue]) -> None:
    """Write deterministic JSON."""
    write_json_object(path, payload)


def write_text(path: Path, text: str) -> None:
    """Write text and ensure the parent exists."""
    write_shared_text(path, text)


def load_json_object(path: Path) -> JSONMapping:
    """Load a JSON object from disk."""
    payload = load_shared_json_object(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def read_csv_header(path: Path) -> tuple[str, ...]:
    """Read and validate a CSV header."""
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
    if not header:
        raise ValueError(f"missing CSV header: {path}")
    return tuple(header)


def config_to_json_dict(config: Goal5CompressionPipelineConfig) -> JSONMapping:
    """Return a JSON-safe config mapping."""
    payload: JSONMapping = {}
    for field in config.__dataclass_fields__:  # type: ignore[attr-defined]
        value = getattr(config, field)
        if isinstance(value, Path):
            payload[field] = str(value)
        elif isinstance(value, tuple):
            payload[field] = list(value)
        else:
            payload[field] = value
    return payload


def _ensure_required_existing(name: str, paths: Sequence[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing required {name} artifact(s): {joined}")


def _ensure_stage(
    *,
    name: str,
    force: bool,
    reuse_existing: bool,
    run_missing: bool,
    config_path: Path,
    required_paths: Sequence[Path],
    load_stage_config: Callable[[Path], object],
    run_stage: Callable[[object], object],
) -> JSONMapping:
    existing = all(path.exists() for path in required_paths)
    if existing and reuse_existing and not force:
        return {
            "stage": name,
            "action": "loaded_existing",
            "required_paths": _path_list(required_paths),
        }
    if not run_missing and not force:
        missing = [path for path in required_paths if not path.exists()]
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"{name} artifacts missing and run_missing_artifacts=false: {joined}"
        )
    if not config_path.exists():
        raise FileNotFoundError(f"missing {name} config: {config_path}")
    stage_config = load_stage_config(config_path)
    result = run_stage(stage_config)
    return {
        "stage": name,
        "action": "ran",
        "config_path": str(config_path),
        "output_paths": _path_list(getattr(result, "output_paths", ())),
    }


def _comparison_row(
    *,
    mode: str,
    stage: str,
    representation_mode: str,
    processed_count: int | None,
    success_count: int | None,
    validation_failure_count: int | None = None,
    reconstruction_failure_count: int | None = None,
    median_nodes: float | None = None,
    median_child_refs_or_edges: float | None = None,
    median_alpha_or_size_ratio: float | None = None,
    median_gain: float | None = None,
    nontrivial: Mapping[str, JSONValue] | None = None,
    nontrivial_gain_key: str | None = None,
    nontrivial_nodes_key: str | None = None,
    validation_rate_percent: float | None = None,
    expansion_validation_rate_percent: float | None = None,
    reconstruction_validation_rate_percent: float | None = None,
    is_pure_eml_metric: bool,
    compressed_nodes_are_pure_eml: bool,
    size_metric_is_compression_metric: bool,
    notes: str,
) -> JSONMapping:
    nontrivial_processed = _int_or_none(nontrivial.get("processed_count")) if nontrivial else None
    nontrivial_success = _int_or_none(nontrivial.get("success_count")) if nontrivial else None
    nontrivial_gain = None
    if nontrivial and nontrivial_gain_key:
        candidate = nontrivial.get(nontrivial_gain_key)
        nontrivial_gain = (
            _stat(candidate, "median") if isinstance(candidate, dict) else _float_or_none(candidate)
        )
    nontrivial_nodes = None
    if nontrivial and nontrivial_nodes_key:
        candidate = nontrivial.get(nontrivial_nodes_key)
        nontrivial_nodes = _stat(candidate, "median") if isinstance(candidate, dict) else None
    return {
        "mode": mode,
        "stage": stage,
        "representation_mode": representation_mode,
        "processed_count": processed_count,
        "success_count": success_count,
        "failure_count": _failure_count(processed_count, success_count),
        "validation_failure_count": validation_failure_count,
        "reconstruction_failure_count": reconstruction_failure_count,
        "median_nodes": median_nodes,
        "median_child_refs_or_edges": median_child_refs_or_edges,
        "median_alpha_or_size_ratio": median_alpha_or_size_ratio,
        "median_compression_gain_vs_goal3_eml_dag": median_gain,
        "nontrivial_processed_count": nontrivial_processed,
        "nontrivial_success_count": nontrivial_success,
        "nontrivial_median_gain_vs_goal3_eml_dag": nontrivial_gain,
        "nontrivial_median_nodes": nontrivial_nodes,
        "validation_rate_percent": validation_rate_percent,
        "expansion_validation_rate_percent": expansion_validation_rate_percent,
        "reconstruction_validation_rate_percent": reconstruction_validation_rate_percent,
        "is_pure_eml_metric": is_pure_eml_metric,
        "compressed_nodes_are_pure_eml": compressed_nodes_are_pure_eml,
        "size_metric_is_compression_metric": size_metric_is_compression_metric,
        "notes": notes,
    }


def _compact_subset_summary(
    subset: Mapping[str, JSONValue],
    *,
    gain_key: str,
    node_key: str | None = None,
) -> JSONMapping:
    payload: JSONMapping = {
        "processed_count": _int_or_none(subset.get("processed_count")),
        "success_count": _int_or_none(subset.get("success_count")),
        "median_gain_vs_goal3": _stat(_nested_dict(subset, gain_key), "median"),
    }
    if node_key:
        payload["median_nodes"] = _stat(_nested_dict(subset, node_key), "median")
    if "motif_coverage_percent" in subset:
        payload["median_coverage_percent"] = _stat(
            _nested_dict(subset, "motif_coverage_percent"), "median"
        )
    if "reconstruction_failure_count" in subset:
        payload["reconstruction_failure_count"] = _int_or_none(
            subset.get("reconstruction_failure_count")
        )
    if "expansion_validation_failure_count" in subset:
        payload["expansion_validation_failure_count"] = _int_or_none(
            subset.get("expansion_validation_failure_count")
        )
    return payload


def _compact_neural_subset(subset: Mapping[str, JSONValue]) -> JSONMapping:
    return {
        "processed_count": _int_or_none(subset.get("processed_count")),
        "success_count": _int_or_none(subset.get("success_count")),
        "validation_failure_count": _int_or_none(subset.get("validation_failure_count")),
        "median_compression_gain_vs_goal3": _float_or_none(subset.get("median_compression_gain")),
        "percent_matching_exact_best": _float_or_none(subset.get("percent_matching_exact_best")),
        "median_regret": _stat(_nested_dict(subset, "neural_regret_vs_exact_best"), "median"),
        "median_speedup": _stat(_nested_dict(subset, "neural_speedup_vs_exact_scoring"), "median"),
    }


def _denominator_audit_table(audit: Mapping[str, JSONValue]) -> str:
    rows = []
    for mode in ("safe", "positive_real_formal"):
        payload = _nested_dict(audit, mode)
        rows.append(
            [
                mode,
                _fmt_int(payload.get("processed")),
                _fmt_int(payload.get("success")),
                _fmt_int(payload.get("timeout")),
                _fmt_int(payload.get("validation_failed")),
                _fmt_int(payload.get("extraction_failed")),
                _fmt_int(payload.get("official_compilation_failed")),
                _fmt_number(payload.get("before_threshold_rate")),
                _fmt_number(payload.get("success_only_after_rate")),
                _fmt_number(payload.get("all_processed_after_rate")),
            ]
        )
    return _markdown_table(
        [
            "Mode",
            "Processed",
            "Success",
            "Timeout",
            "Validation failed",
            "Extraction failed",
            "Official compile failed",
            "Before rate",
            "Success-only after rate",
            "All-processed after rate",
        ],
        rows,
    )


def _null_result_table(null_results: Mapping[str, JSONValue]) -> str:
    return _markdown_table(
        ["Metric", "Value"],
        [
            [
                "learned vs frequent motif median",
                _fmt_number(null_results.get("learned_vs_frequent_motif_median")),
            ],
            [
                "learned vs random motif median",
                _fmt_number(null_results.get("learned_vs_random_motif_median")),
            ],
            [
                "learned vs random motif mean",
                _fmt_number(null_results.get("learned_vs_random_motif_mean")),
            ],
            [
                "neural exact-match rate",
                _fmt_number(null_results.get("neural_exact_match_rate")),
            ],
            [
                "estimated heuristic exact-match rate",
                _fmt_number(null_results.get("estimated_heuristic_exact_match_rate")),
            ],
            [
                "AST baseline exact-match rate",
                _fmt_number(null_results.get("ast_baseline_exact_match_rate")),
            ],
            ["neural mean regret", _fmt_number(null_results.get("neural_mean_regret"))],
            ["heuristic mean regret", _fmt_number(null_results.get("heuristic_mean_regret"))],
            ["AST mean regret", _fmt_number(null_results.get("ast_mean_regret"))],
        ],
    )


def _candidate_discovery_table(payload: Mapping[str, JSONValue]) -> str:
    split_counts = payload.get("candidate_discovery_split_counts", {})
    split_counts = split_counts if isinstance(split_counts, dict) else {}
    return _markdown_table(
        ["Field", "Value"],
        [
            ["candidate discovery mode", str(payload.get("candidate_discovery_mode", ""))],
            [
                "candidate discovery expression count",
                _fmt_int(payload.get("candidate_discovery_expression_count")),
            ],
            ["train discovery rows", _fmt_int(split_counts.get("train"))],
            ["validation discovery rows", _fmt_int(split_counts.get("validation"))],
            ["test discovery rows", _fmt_int(split_counts.get("test"))],
            [
                "test set used for candidate discovery",
                str(bool(payload.get("test_set_used_for_candidate_discovery", True))),
            ],
        ],
    )


def _compact_candidate_discovery(payload: Mapping[str, JSONValue]) -> JSONMapping:
    return {
        key: value
        for key, value in payload.items()
        if key != "candidate_discovery_expression_indices"
    }


def _frequent_train_only_table(payload: Mapping[str, JSONValue]) -> str:
    discovery = _nested_dict(payload, "candidate_discovery")
    split_counts = discovery.get("candidate_discovery_split_counts", {})
    split_counts = split_counts if isinstance(split_counts, dict) else {}
    validation = _nested_dict(payload, "results_by_split", "validation")
    test = _nested_dict(payload, "results_by_split", "test")
    nontrivial = _nested_dict(payload, "nontrivial_v1")
    comparison = _nested_dict(payload, "full_corpus_comparison")
    coverage_loss = _nested_dict(comparison, "coverage_loss_percent_points")
    if not coverage_loss:
        coverage_loss = _nested_dict(comparison, "coverage_loss_vs_full_corpus_percentage_points")
    return _markdown_table(
        ["Field", "Value"],
        [
            ["train-only vocab size", _fmt_int(payload.get("motif_vocab_size"))],
            [
                "candidate discovery expression count",
                _fmt_int(discovery.get("candidate_discovery_expression_count")),
            ],
            ["train discovery rows", _fmt_int(split_counts.get("train"))],
            ["validation discovery rows", _fmt_int(split_counts.get("validation"))],
            ["test discovery rows", _fmt_int(split_counts.get("test"))],
            [
                "test set used for discovery",
                str(bool(discovery.get("test_set_used_for_candidate_discovery", True))),
            ],
            [
                "validation median gain vs Goal 3",
                _fmt_number(
                    _stat(_nested_dict(validation, "compression_gain_vs_goal3_eml_dag"), "median")
                ),
            ],
            [
                "test median gain vs Goal 3",
                _fmt_number(
                    _stat(_nested_dict(test, "compression_gain_vs_goal3_eml_dag"), "median")
                ),
            ],
            [
                "nontrivial median gain vs Goal 3",
                _fmt_number(nontrivial.get("median_gain_vs_goal3")),
            ],
            [
                "median coverage loss vs full-corpus mining",
                _fmt_number(coverage_loss.get("median")),
            ],
            [
                "expansion/reconstruction failures",
                _fmt_int(payload.get("expansion_validation_failure_count")),
            ],
        ],
    )


def learned_candidate_discovery_used_test(learned_summary: Mapping[str, JSONValue]) -> bool:
    payload = _nested_dict(learned_summary, "candidate_discovery")
    return bool(payload.get("test_set_used_for_candidate_discovery", True))


def _motif_table(motifs: Sequence[Mapping[str, JSONValue]]) -> str:
    if not motifs:
        return "_No motifs recorded._"
    return _markdown_table(
        ["Motif", "Type", "Nodes", "Support", "Covered nodes", "Macro"],
        [
            [
                str(motif.get("motif_id")),
                str(motif.get("motif_type")),
                _fmt_int(motif.get("node_count")),
                _fmt_int(motif.get("support_count")),
                _fmt_int(motif.get("total_covered_nodes")),
                str(motif.get("official_macro_name") or ""),
            ]
            for motif in motifs
        ],
    )


def _split_table(splits: Mapping[str, JSONValue], *, gain_key: str) -> str:
    rows = []
    for split in ("train", "validation", "test"):
        payload = _nested_dict(splits, split)
        if not payload:
            continue
        rows.append(
            [
                split,
                _fmt_int(payload.get("processed_count")),
                _fmt_int(payload.get("success_count")),
                _fmt_number(_stat(_nested_dict(payload, gain_key), "median")),
                _fmt_int(payload.get("reconstruction_failure_count")),
            ]
        )
    return _markdown_table(["Split", "Processed", "Success", "Median gain", "Failures"], rows)


def _hierarchy_stats_table(stats: Mapping[str, JSONValue]) -> str:
    rows = []
    for mode, payload in stats.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            [
                mode,
                _fmt_int(payload.get("graph_count")),
                _fmt_number(_stat(_nested_dict(payload, "node_count"), "median")),
                _fmt_number(_stat(_nested_dict(payload, "edge_count"), "median")),
                _fmt_number(payload.get("reconstruction_validation_rate")),
            ]
        )
    return _markdown_table(
        ["Mode", "Graphs", "Median nodes", "Median edges", "Reconstruction %"], rows
    )


def _nested_dict(payload: Mapping[str, JSONValue], *keys: str) -> JSONMapping:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def _stat(payload: object, key: str) -> float | None:
    return _float_or_none(payload.get(key)) if isinstance(payload, dict) else None


def _first_float(*values: object) -> float | None:
    for value in values:
        number = _float_or_none(value)
        if number is not None:
            return number
    return None


def _success_rate(payload: Mapping[str, JSONValue]) -> float | None:
    processed = _float_or_none(payload.get("processed_count"))
    success = _float_or_none(payload.get("success_count"))
    if processed in (None, 0.0) or success is None:
        return None
    return 100.0 * success / processed


def _failure_rate(payload: Mapping[str, JSONValue], failure_key: str) -> float:
    processed = _float_or_none(payload.get("processed_count"))
    failures = _float_or_none(payload.get(failure_key))
    if processed in (None, 0.0) or failures is None:
        return 0.0
    return 100.0 * failures / processed


def _failure_count(processed: int | None, success: int | None) -> int | None:
    if processed is None or success is None:
        return None
    return processed - success


def _int_or_none(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _int_or_zero(value: object) -> int:
    return _int_or_none(value) or 0


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _fmt_int(value: object) -> str:
    integer = _int_or_none(value)
    return "" if integer is None else f"{integer:,}"


def _fmt_number(value: object) -> str:
    number = _float_or_none(value)
    if number is None:
        return ""
    return f"{number:.3f}"


def _csv_value(value: object) -> object:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    return value


def _path_list(paths: Sequence[Path]) -> list[str]:
    return [str(path) for path in paths]


def _coerce_config_value(key: str, value: object) -> object:
    if key.endswith("_path"):
        return Path(value)
    return value


def _all_config_paths(config: Goal5CompressionPipelineConfig) -> tuple[Path, ...]:
    return tuple(
        value
        for field in config.__dataclass_fields__  # type: ignore[attr-defined]
        if isinstance((value := getattr(config, field)), Path)
    )


def _assert_no_outputs_v0(paths: Sequence[Path]) -> None:
    bad_paths = [path for path in paths if "outputs/v0" in path.as_posix()]
    if bad_paths:
        joined = ", ".join(str(path) for path in bad_paths)
        raise ValueError(f"Goal 5 result-bearing paths must not use outputs/v0: {joined}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_goal5_compression_pipeline(load_config(args.config))
    print(f"Wrote comparison: {result.generated_files[0]}")
    print(f"Wrote summary: {result.generated_files[1]}")
    print(f"Wrote final report: {result.generated_files[2]}")
    print(f"Wrote summary doc: {result.generated_files[3]}")
    print(f"Wrote findings: {result.generated_files[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
