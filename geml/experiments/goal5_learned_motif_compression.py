"""Goal 5.3 learned motif vocabulary and discrete compression baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from geml.compression.learned_motifs import (
    build_learned_motif_vocabulary,
    select_random_motif_ids,
    write_learned_motif_vocabulary,
)
from geml.compression.motif_dataset import (
    SplitConfig,
    assign_split,
    build_split_rows,
    load_frequent_motif_baseline_rows,
    load_macro_graph_baseline_rows,
    summarize_split_counts,
)
from geml.compression.motif_mining import enumerate_motif_occurrences
from geml.compression.motif_rewrite import (
    MotifCompressionSummary,
    greedy_motif_compress_occurrences_summary,
)
from geml.compression.motif_selection_model import (
    MotifScoringWeights,
    MotifSelectionResult,
    MotifSplitFeature,
    build_weight_grid,
    optimize_motif_selection,
    score_motif,
)
from geml.compression.motif_vocab import MotifRecord, MotifVocabulary, load_motif_vocabulary
from geml.experiments.goal5_frequent_motif_mining import (
    FrequentMotifMiningConfig,
    GraphBundle,
    build_graph_bundles,
)
from geml.experiments.shared import (
    build_run_metadata,
)
from geml.experiments.shared import (
    safe_divide as _safe_divide,
)
from geml.experiments.shared import (
    write_json_object as write_json,
)

type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]

LEARNED_MOTIF_METRICS_FIELDS = [
    "index",
    "split",
    "subset_label",
    "original_eml_dag_nodes",
    "frequent_motif_nodes",
    "learned_motif_nodes",
    "random_motif_nodes",
    "macro_graph_nodes",
    "learned_gain_vs_goal3_eml_dag",
    "learned_gain_vs_frequent_motif",
    "learned_gain_vs_macro_graph",
    "random_gain_vs_goal3_eml_dag",
    "motif_coverage_percent",
    "reconstruction_valid",
    "learned_selected_graph_type",
    "random_selected_graph_type",
    "error",
]


@dataclass(frozen=True, slots=True)
class LearnedMotifCompressionConfig:
    """Configuration for Goal 5.3 learned motif compression."""

    seed: int = 0
    count: int = 10_000
    max_depth: int = 4
    operator_set: tuple[str, ...] = ("add", "mul", "exp", "log")
    symbol_names: tuple[str, ...] = ("x", "y")
    source_serialization: str = "srepr"
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    min_motif_nodes: int = 1
    max_motif_nodes: int = 2
    learned_vocab_sizes: tuple[int, ...] = (20, 30)
    coverage_bonuses: tuple[float, ...] = (0.0, 0.01)
    nontrivial_coverage_bonuses: tuple[float, ...] = (0.0, 0.02)
    vocab_complexity_penalties: tuple[float, ...] = (0.0, 0.1)
    expansion_complexity_penalties: tuple[float, ...] = (0.0, 0.05)
    input_jsonl_path: Path = Path("outputs/v1/dag_compression_inputs.jsonl")
    goal3_metrics_csv_path: Path = Path("outputs/v1/dag_compression_metrics.csv")
    macro_graph_metrics_csv_path: Path = Path("outputs/v1/goal5_macro_graph_metrics.csv")
    frequent_motif_vocab_json_path: Path = Path("outputs/v1/goal5_frequent_motif_vocab.json")
    frequent_motif_metrics_csv_path: Path = Path("outputs/v1/goal5_frequent_motif_metrics.csv")
    learned_vocab_json_path: Path = Path("outputs/v1/goal5_learned_motif_vocab.json")
    metrics_csv_path: Path = Path("outputs/v1/goal5_learned_motif_metrics.csv")
    metrics_jsonl_path: Path = Path("outputs/v1/goal5_learned_motif_metrics.jsonl")
    summary_json_path: Path = Path("outputs/v1/goal5_learned_motif_summary.json")
    train_log_json_path: Path = Path("outputs/v1/goal5_learned_motif_train_log.json")

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.max_depth != 4:
            raise ValueError("Goal 5.3 v1 config expects max_depth=4")
        if tuple(self.operator_set) != ("add", "mul", "exp", "log"):
            raise ValueError("Goal 5.3 v1 operator_set must be add, mul, exp, log")
        if tuple(self.symbol_names) != ("x", "y"):
            raise ValueError("Goal 5.3 v1 symbol_names must be x, y")
        if self.source_serialization != "srepr":
            raise ValueError("Goal 5.3 requires authoritative srepr input")
        if self.min_motif_nodes <= 0:
            raise ValueError("min_motif_nodes must be positive")
        if self.max_motif_nodes < self.min_motif_nodes:
            raise ValueError("max_motif_nodes must be >= min_motif_nodes")
        if not self.learned_vocab_sizes:
            raise ValueError("learned_vocab_sizes must not be empty")
        _assert_no_outputs_v0(
            [
                self.input_jsonl_path,
                self.goal3_metrics_csv_path,
                self.macro_graph_metrics_csv_path,
                self.frequent_motif_vocab_json_path,
                self.frequent_motif_metrics_csv_path,
                self.learned_vocab_json_path,
                self.metrics_csv_path,
                self.metrics_jsonl_path,
                self.summary_json_path,
                self.train_log_json_path,
            ]
        )


@dataclass(frozen=True, slots=True)
class CandidateStats:
    """Mutable split statistics while scanning candidate motif occurrences."""

    motif: MotifRecord
    support_by_split: dict[str, int]
    coverage_by_split: dict[str, int]
    nontrivial_coverage_by_split: dict[str, int]


@dataclass(frozen=True, slots=True)
class LearnedMotifMetricRow:
    """One per-expression learned motif metric row."""

    index: int
    split: str
    subset_label: str
    original_eml_dag_nodes: int
    frequent_motif_nodes: int
    learned_motif_nodes: int | None
    random_motif_nodes: int | None
    macro_graph_nodes: int
    learned_gain_vs_goal3_eml_dag: float | None
    learned_gain_vs_frequent_motif: float | None
    learned_gain_vs_macro_graph: float | None
    random_gain_vs_goal3_eml_dag: float | None
    motif_coverage_percent: float | None
    reconstruction_valid: bool
    learned_selected_graph_type: str | None
    random_selected_graph_type: str | None
    error: str | None

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable row."""
        return {field: getattr(self, field) for field in LEARNED_MOTIF_METRICS_FIELDS}

    def to_csv_dict(self) -> dict[str, object]:
        """Return a CSV-serializable row."""
        return {field: _csv_value(getattr(self, field)) for field in LEARNED_MOTIF_METRICS_FIELDS}


@dataclass(frozen=True, slots=True)
class LearnedMotifCompressionResult:
    """Result summary for Goal 5.3."""

    summary: dict[str, object]
    output_paths: tuple[Path, ...]


def load_config(path: Path) -> LearnedMotifCompressionConfig:
    """Load a Goal 5.3 YAML config."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return LearnedMotifCompressionConfig(
        **{key: _coerce_config_value(key, value) for key, value in raw.items()}
    )


def run_goal5_learned_motif_compression(
    config: LearnedMotifCompressionConfig,
) -> LearnedMotifCompressionResult:
    """Run learned motif selection and discrete compression on v1."""
    started_at = time.time()
    split_config = SplitConfig(
        seed=config.seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    frequent_rows = load_frequent_motif_baseline_rows(config.frequent_motif_metrics_csv_path)
    macro_rows = load_macro_graph_baseline_rows(config.macro_graph_metrics_csv_path)
    bundles = build_graph_bundles(_frequent_config(config))
    print(f"Built graph bundles: {len(bundles)}", flush=True)

    split_by_index = {
        bundle.baseline.index: assign_split(bundle.baseline.index, split_config)
        for bundle in bundles
    }
    candidate_vocabulary = load_candidate_motif_vocabulary(config.frequent_motif_vocab_json_path)
    candidate_motifs = filter_candidate_motifs(candidate_vocabulary)
    candidate_features = compute_candidate_features(
        bundles,
        candidate_motifs,
        split_by_index=split_by_index,
        config=config,
    )
    print(f"Computed candidate features: {len(candidate_features)}", flush=True)
    selection_result = optimize_motif_selection(
        candidate_features,
        vocab_sizes=config.learned_vocab_sizes,
        weight_grid=build_weight_grid(
            coverage_bonuses=config.coverage_bonuses,
            nontrivial_coverage_bonuses=config.nontrivial_coverage_bonuses,
            vocab_complexity_penalties=config.vocab_complexity_penalties,
            expansion_complexity_penalties=config.expansion_complexity_penalties,
        ),
    )
    candidate_features_by_id = {feature.motif_id: feature for feature in candidate_features}
    candidate_motifs_by_id = {motif.motif_id: motif for motif in candidate_motifs}
    selected_weights = MotifScoringWeights(
        coverage_bonus=selection_result.selected_weights["coverage_bonus"],
        nontrivial_coverage_bonus=selection_result.selected_weights["nontrivial_coverage_bonus"],
        vocab_complexity_penalty=selection_result.selected_weights["vocab_complexity_penalty"],
        expansion_complexity_penalty=selection_result.selected_weights[
            "expansion_complexity_penalty"
        ],
    )
    selected_scores = {
        motif_id: score_motif(candidate_features_by_id[motif_id], selected_weights, split="train")
        for motif_id in selection_result.selected_motif_ids
    }
    random_motif_ids = select_random_motif_ids(
        sorted(candidate_motifs_by_id),
        vocab_size=selection_result.selected_vocab_size,
        seed=config.seed,
    )
    learned_vocabulary = build_learned_motif_vocabulary(
        selected_motif_ids=selection_result.selected_motif_ids,
        selected_scores=selected_scores,
        selected_weights=selection_result.selected_weights,
        candidate_features_by_id=candidate_features_by_id,
        candidate_motifs_by_id=candidate_motifs_by_id,
        random_baseline_motif_ids=random_motif_ids,
    )
    learned_vocabulary = learned_vocabulary.model_copy(
        update={
            "metadata": {
                **dict(learned_vocabulary.metadata),
                "candidate_discovery": candidate_discovery_payload(candidate_vocabulary),
            }
        }
    )
    write_learned_motif_vocabulary(learned_vocabulary, config.learned_vocab_json_path)
    print(
        f"Selected learned motif vocabulary: {learned_vocabulary.selected_vocab_size}",
        flush=True,
    )

    learned_motifs = [
        candidate_motifs_by_id[motif_id] for motif_id in selection_result.selected_motif_ids
    ]
    random_motifs = [candidate_motifs_by_id[motif_id] for motif_id in random_motif_ids]
    rows = tuple(
        compute_learned_metric_row(
            bundle,
            split=split_by_index[bundle.baseline.index],
            frequent_row=frequent_rows[bundle.baseline.index],
            macro_row=macro_rows[bundle.baseline.index],
            learned_motifs=learned_motifs,
            random_motifs=random_motifs,
            config=config,
        )
        for bundle in bundles
    )
    print(f"Computed learned motif rows: {len(rows)}", flush=True)
    write_metrics_jsonl(rows, config.metrics_jsonl_path)
    write_metrics_csv(rows, config.metrics_csv_path)
    completed_at = time.time()
    train_log = build_train_log(
        config,
        selection_result,
        split_rows=build_split_rows(split_by_index, split_config),
        candidate_features=candidate_features,
        candidate_vocabulary=candidate_vocabulary,
        random_motif_ids=random_motif_ids,
        started_at=started_at,
        completed_at=completed_at,
    )
    write_json(config.train_log_json_path, train_log)
    summary = build_summary(
        rows,
        config,
        selection_result,
        candidate_vocabulary=candidate_vocabulary,
        learned_vocab_size=learned_vocabulary.selected_vocab_size,
        random_vocab_size=len(random_motif_ids),
        started_at=started_at,
        completed_at=completed_at,
    )
    write_json(config.summary_json_path, summary)
    return LearnedMotifCompressionResult(
        summary=summary,
        output_paths=(
            config.learned_vocab_json_path,
            config.metrics_csv_path,
            config.metrics_jsonl_path,
            config.summary_json_path,
            config.train_log_json_path,
        ),
    )


def load_candidate_motifs(path: Path) -> tuple[MotifRecord, ...]:
    """Load graph-applicable candidate motifs from Goal 5.2."""
    return filter_candidate_motifs(load_candidate_motif_vocabulary(path))


def load_candidate_motif_vocabulary(path: Path) -> MotifVocabulary:
    """Load the candidate motif vocabulary with discovery metadata."""
    return load_motif_vocabulary(path)


def filter_candidate_motifs(vocabulary: MotifVocabulary) -> tuple[MotifRecord, ...]:
    """Return graph-applicable candidate motifs from one vocabulary."""
    candidates = tuple(
        motif
        for motif in vocabulary.motifs
        if motif.motif_type in {"pure_eml_dag", "macro_graph"}
        and motif.node_count > 1
        and motif.expansion_map_to_pure_eml_available
    )
    if not candidates:
        raise ValueError("candidate motif pool is empty")
    return candidates


def compute_candidate_features(
    bundles: Sequence[GraphBundle],
    candidate_motifs: Sequence[MotifRecord],
    *,
    split_by_index: dict[int, str],
    config: LearnedMotifCompressionConfig,
) -> tuple[MotifSplitFeature, ...]:
    """Compute train/validation/test support features for candidate motifs."""
    stats = {
        motif.motif_id: CandidateStats(
            motif=motif,
            support_by_split=defaultdict(int),
            coverage_by_split=defaultdict(int),
            nontrivial_coverage_by_split=defaultdict(int),
        )
        for motif in candidate_motifs
    }
    motif_by_key = {(motif.motif_type, motif.signature): motif for motif in candidate_motifs}
    for bundle in bundles:
        split = split_by_index[bundle.baseline.index]
        for graph in (bundle.pure_graph, bundle.macro_graph):
            occurrences = enumerate_motif_occurrences(
                graph,
                min_motif_nodes=config.min_motif_nodes,
                max_motif_nodes=config.max_motif_nodes,
                expression_index=bundle.baseline.index,
                subset_label=bundle.subset_label,
            )
            for occurrence in occurrences:
                motif = motif_by_key.get((occurrence.graph_type, occurrence.signature))
                if motif is None:
                    continue
                item = stats[motif.motif_id]
                item.support_by_split[split] += 1
                item.coverage_by_split[split] += occurrence.node_count
                if bundle.subset_label == "nontrivial_v1":
                    item.nontrivial_coverage_by_split[split] += occurrence.node_count

    return tuple(_stats_to_feature(item) for item in stats.values())


def compute_learned_metric_row(
    bundle: GraphBundle,
    *,
    split: str,
    frequent_row: object,
    macro_row: object,
    learned_motifs: Sequence[MotifRecord],
    random_motifs: Sequence[MotifRecord],
    config: LearnedMotifCompressionConfig,
) -> LearnedMotifMetricRow:
    """Compute one learned/random/frequent/macro comparison row."""
    try:
        pure_occurrences = enumerate_motif_occurrences(
            bundle.pure_graph,
            min_motif_nodes=config.min_motif_nodes,
            max_motif_nodes=config.max_motif_nodes,
            expression_index=bundle.baseline.index,
            subset_label=bundle.subset_label,
        )
        macro_occurrences = enumerate_motif_occurrences(
            bundle.macro_graph,
            min_motif_nodes=config.min_motif_nodes,
            max_motif_nodes=config.max_motif_nodes,
            expression_index=bundle.baseline.index,
            subset_label=bundle.subset_label,
        )
        learned_pure = greedy_motif_compress_occurrences_summary(
            bundle.pure_graph,
            pure_occurrences,
            learned_motifs,
        )
        learned_macro = greedy_motif_compress_occurrences_summary(
            bundle.macro_graph,
            macro_occurrences,
            learned_motifs,
        )
        random_pure = greedy_motif_compress_occurrences_summary(
            bundle.pure_graph,
            pure_occurrences,
            random_motifs,
        )
        random_macro = greedy_motif_compress_occurrences_summary(
            bundle.macro_graph,
            macro_occurrences,
            random_motifs,
        )
        learned_result, learned_graph_type = _select_best_result(learned_pure, learned_macro)
        random_result, random_graph_type = _select_best_result(random_pure, random_macro)
        learned_nodes = learned_result.compressed_node_count
        random_nodes = random_result.compressed_node_count
        return LearnedMotifMetricRow(
            index=bundle.baseline.index,
            split=split,
            subset_label=bundle.subset_label,
            original_eml_dag_nodes=bundle.baseline.eml_dag_node_count,
            frequent_motif_nodes=frequent_row.frequent_motif_nodes,
            learned_motif_nodes=learned_nodes,
            random_motif_nodes=random_nodes,
            macro_graph_nodes=macro_row.macro_graph_nodes,
            learned_gain_vs_goal3_eml_dag=_safe_divide(
                bundle.baseline.eml_dag_node_count,
                learned_nodes,
            ),
            learned_gain_vs_frequent_motif=_safe_divide(
                frequent_row.frequent_motif_nodes,
                learned_nodes,
            ),
            learned_gain_vs_macro_graph=_safe_divide(
                macro_row.macro_graph_nodes,
                learned_nodes,
            ),
            random_gain_vs_goal3_eml_dag=_safe_divide(
                bundle.baseline.eml_dag_node_count,
                random_nodes,
            ),
            motif_coverage_percent=learned_result.motif_coverage_percent,
            reconstruction_valid=learned_result.expansion_valid,
            learned_selected_graph_type=learned_graph_type,
            random_selected_graph_type=random_graph_type,
            error=None,
        )
    except Exception as exc:
        return LearnedMotifMetricRow(
            index=bundle.baseline.index,
            split=split,
            subset_label=bundle.subset_label,
            original_eml_dag_nodes=bundle.baseline.eml_dag_node_count,
            frequent_motif_nodes=frequent_row.frequent_motif_nodes,
            learned_motif_nodes=None,
            random_motif_nodes=None,
            macro_graph_nodes=macro_row.macro_graph_nodes,
            learned_gain_vs_goal3_eml_dag=None,
            learned_gain_vs_frequent_motif=None,
            learned_gain_vs_macro_graph=None,
            random_gain_vs_goal3_eml_dag=None,
            motif_coverage_percent=None,
            reconstruction_valid=False,
            learned_selected_graph_type=None,
            random_selected_graph_type=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def build_train_log(
    config: LearnedMotifCompressionConfig,
    selection_result: MotifSelectionResult,
    *,
    split_rows: Sequence[object],
    candidate_features: Sequence[MotifSplitFeature],
    candidate_vocabulary: MotifVocabulary,
    random_motif_ids: Sequence[str],
    started_at: float,
    completed_at: float,
) -> dict[str, object]:
    """Build the Goal 5.3 train log artifact."""
    return {
        "config": config_to_json_dict(config),
        "model_type": selection_result.model_type,
        "trained_final_reasoning_gnn": selection_result.trained_final_reasoning_gnn,
        "split_counts": summarize_split_counts(split_rows),
        "candidate_pool_size": len(candidate_features),
        "candidate_discovery": candidate_discovery_payload(candidate_vocabulary),
        "selected_vocab_size": selection_result.selected_vocab_size,
        "selected_weights": selection_result.selected_weights,
        "selected_motif_ids": selection_result.selected_motif_ids,
        "random_motif_ids": list(random_motif_ids),
        "hyperparameter_trials": [
            trial.model_dump(mode="json") for trial in selection_result.trials
        ],
        "selection_policy": (
            "train scores choose motifs; validation objective chooses hyperparameters; "
            "candidate discovery should use train-only motif mining for leakage-free results"
        ),
        "test_set_used_for_selection": False,
        "test_set_used_for_candidate_discovery": bool(
            candidate_discovery_payload(candidate_vocabulary).get(
                "test_set_used_for_candidate_discovery",
                True,
            )
        ),
        "run_metadata": build_run_metadata(
            config=config_to_json_dict(config),
            started_at=started_at,
            completed_at=completed_at,
        ),
    }


def build_summary(
    rows: Sequence[LearnedMotifMetricRow],
    config: LearnedMotifCompressionConfig,
    selection_result: MotifSelectionResult,
    *,
    candidate_vocabulary: MotifVocabulary,
    learned_vocab_size: int,
    random_vocab_size: int,
    started_at: float,
    completed_at: float,
) -> dict[str, object]:
    """Build required Goal 5.3 summary."""
    success_rows = [row for row in rows if row.reconstruction_valid and row.error is None]
    return {
        "config": config_to_json_dict(config),
        "processed_count": len(rows),
        "success_count": len(success_rows),
        "reconstruction_failure_count": sum(not row.reconstruction_valid for row in rows),
        "learned_vocab_size": learned_vocab_size,
        "random_vocab_size": random_vocab_size,
        "selected_weights": selection_result.selected_weights,
        "candidate_discovery": candidate_discovery_payload(candidate_vocabulary),
        "learned_vs_frequent_motif_compression": _distribution(
            row.learned_gain_vs_frequent_motif for row in success_rows
        ),
        "learned_vs_random_motif_compression": _learned_vs_random_distribution(success_rows),
        "learned_vs_macro_graph_baseline": _distribution(
            row.learned_gain_vs_macro_graph for row in success_rows
        ),
        "learned_gain_vs_goal3_eml_dag": _distribution(
            row.learned_gain_vs_goal3_eml_dag for row in success_rows
        ),
        "random_gain_vs_goal3_eml_dag": _distribution(
            row.random_gain_vs_goal3_eml_dag for row in success_rows
        ),
        "motif_coverage_percent": _distribution(row.motif_coverage_percent for row in success_rows),
        "results_by_split": {
            split: summarize_group([row for row in rows if row.split == split])
            for split in ("train", "validation", "test")
        },
        "results_by_subset_label": {
            label: summarize_group(
                list(rows)
                if label == "all_v1"
                else [row for row in rows if row.subset_label == label]
            )
            for label in ("all_v1", "nontrivial_v1", "identity_heavy_v1")
        },
        "heldout_generalization": {
            "validation": summarize_group([row for row in rows if row.split == "validation"]),
            "test": summarize_group([row for row in rows if row.split == "test"]),
        },
        "integrity": {
            "motif_ids_are_pure_eml_nodes": False,
            "reconstruction_validity_required": True,
            "test_set_used_for_selection": False,
            "test_set_used_for_candidate_discovery": bool(
                candidate_discovery_payload(candidate_vocabulary).get(
                    "test_set_used_for_candidate_discovery",
                    True,
                )
            ),
            "modified_official_eml_compiler": False,
            "trained_final_symbolic_reasoning_gnn": False,
            "model_performance_claims": False,
        },
        "elapsed_seconds": completed_at - started_at,
        "completed_at_unix": completed_at,
        "run_metadata": build_run_metadata(
            config=config_to_json_dict(config),
            started_at=started_at,
            completed_at=completed_at,
        ),
    }


def summarize_group(rows: Sequence[LearnedMotifMetricRow]) -> dict[str, object]:
    """Summarize one split or subset group."""
    success_rows = [row for row in rows if row.reconstruction_valid and row.error is None]
    return {
        "processed_count": len(rows),
        "success_count": len(success_rows),
        "reconstruction_failure_count": sum(not row.reconstruction_valid for row in rows),
        "learned_motif_nodes": _distribution(row.learned_motif_nodes for row in success_rows),
        "random_motif_nodes": _distribution(row.random_motif_nodes for row in success_rows),
        "learned_gain_vs_goal3_eml_dag": _distribution(
            row.learned_gain_vs_goal3_eml_dag for row in success_rows
        ),
        "random_gain_vs_goal3_eml_dag": _distribution(
            row.random_gain_vs_goal3_eml_dag for row in success_rows
        ),
        "learned_gain_vs_frequent_motif": _distribution(
            row.learned_gain_vs_frequent_motif for row in success_rows
        ),
        "learned_gain_vs_macro_graph": _distribution(
            row.learned_gain_vs_macro_graph for row in success_rows
        ),
        "learned_vs_random_motif_compression": _learned_vs_random_distribution(success_rows),
        "motif_coverage_percent": _distribution(row.motif_coverage_percent for row in success_rows),
    }


def candidate_discovery_payload(vocabulary: MotifVocabulary) -> dict[str, object]:
    """Return candidate discovery metadata used by learned motif selection."""
    metadata = dict(vocabulary.metadata)
    return {
        "candidate_discovery_mode": metadata.get("candidate_discovery_mode", "unknown"),
        "candidate_discovery_split": metadata.get("candidate_discovery_split", "unknown"),
        "candidate_discovery_expression_count": metadata.get(
            "candidate_discovery_expression_count"
        ),
        "candidate_discovery_split_counts": metadata.get(
            "candidate_discovery_split_counts",
            {},
        ),
        "test_set_used_for_candidate_discovery": bool(
            metadata.get("test_set_used_for_candidate_discovery", True)
        ),
        "validation_set_used_for_candidate_discovery": bool(
            metadata.get("validation_set_used_for_candidate_discovery", True)
        ),
    }


def write_metrics_jsonl(rows: Sequence[LearnedMotifMetricRow], path: Path) -> None:
    """Write per-expression learned motif metrics to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row.to_json_dict(), sort_keys=True) + "\n")


def write_metrics_csv(rows: Sequence[LearnedMotifMetricRow], path: Path) -> None:
    """Write per-expression learned motif metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LEARNED_MOTIF_METRICS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def config_to_json_dict(config: LearnedMotifCompressionConfig) -> dict[str, object]:
    """Return JSON-safe config values."""
    return {
        "seed": config.seed,
        "count": config.count,
        "max_depth": config.max_depth,
        "operator_set": list(config.operator_set),
        "symbol_names": list(config.symbol_names),
        "source_serialization": config.source_serialization,
        "train_fraction": config.train_fraction,
        "validation_fraction": config.validation_fraction,
        "min_motif_nodes": config.min_motif_nodes,
        "max_motif_nodes": config.max_motif_nodes,
        "learned_vocab_sizes": list(config.learned_vocab_sizes),
        "coverage_bonuses": list(config.coverage_bonuses),
        "nontrivial_coverage_bonuses": list(config.nontrivial_coverage_bonuses),
        "vocab_complexity_penalties": list(config.vocab_complexity_penalties),
        "expansion_complexity_penalties": list(config.expansion_complexity_penalties),
        "input_jsonl_path": str(config.input_jsonl_path),
        "goal3_metrics_csv_path": str(config.goal3_metrics_csv_path),
        "macro_graph_metrics_csv_path": str(config.macro_graph_metrics_csv_path),
        "frequent_motif_vocab_json_path": str(config.frequent_motif_vocab_json_path),
        "frequent_motif_metrics_csv_path": str(config.frequent_motif_metrics_csv_path),
        "learned_vocab_json_path": str(config.learned_vocab_json_path),
        "metrics_csv_path": str(config.metrics_csv_path),
        "metrics_jsonl_path": str(config.metrics_jsonl_path),
        "summary_json_path": str(config.summary_json_path),
        "train_log_json_path": str(config.train_log_json_path),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/learned_motifs_v1.yaml"),
        help="Path to a YAML Goal 5.3 learned motif config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Goal 5.3 learned motif compression."""
    args = build_parser().parse_args(argv)
    result = run_goal5_learned_motif_compression(load_config(args.config))
    print(f"Processed: {result.summary['processed_count']}")
    print(f"Succeeded: {result.summary['success_count']}")
    print(f"Learned vocab size: {result.summary['learned_vocab_size']}")
    for path in result.output_paths:
        print(f"Wrote: {path}")
    return 0


def _stats_to_feature(item: CandidateStats) -> MotifSplitFeature:
    support_count = sum(item.support_by_split.values())
    coverage = sum(item.coverage_by_split.values())
    node_savings = support_count * max(item.motif.node_count - 1, 0)
    return MotifSplitFeature(
        motif_id=item.motif.motif_id,
        motif_type=item.motif.motif_type,
        node_count=item.motif.node_count,
        edge_count=item.motif.edge_count,
        support_count=support_count,
        train_support=item.support_by_split["train"],
        validation_support=item.support_by_split["validation"],
        test_support=item.support_by_split["test"],
        node_savings=node_savings,
        train_node_savings=item.support_by_split["train"] * max(item.motif.node_count - 1, 0),
        validation_node_savings=item.support_by_split["validation"]
        * max(item.motif.node_count - 1, 0),
        test_node_savings=item.support_by_split["test"] * max(item.motif.node_count - 1, 0),
        coverage=coverage,
        train_coverage=item.coverage_by_split["train"],
        validation_coverage=item.coverage_by_split["validation"],
        test_coverage=item.coverage_by_split["test"],
        nontrivial_coverage=sum(item.nontrivial_coverage_by_split.values()),
        train_nontrivial_coverage=item.nontrivial_coverage_by_split["train"],
        validation_nontrivial_coverage=item.nontrivial_coverage_by_split["validation"],
        test_nontrivial_coverage=item.nontrivial_coverage_by_split["test"],
        expansion_complexity=item.motif.edge_count + len(item.motif.boundary_child_refs),
    )


def _frequent_config(config: LearnedMotifCompressionConfig) -> FrequentMotifMiningConfig:
    return FrequentMotifMiningConfig(
        seed=config.seed,
        count=config.count,
        max_depth=config.max_depth,
        operator_set=config.operator_set,
        symbol_names=config.symbol_names,
        source_serialization="srepr",
        min_motif_nodes=config.min_motif_nodes,
        max_motif_nodes=config.max_motif_nodes,
        input_jsonl_path=config.input_jsonl_path,
        goal3_metrics_csv_path=config.goal3_metrics_csv_path,
        macro_graph_metrics_csv_path=config.macro_graph_metrics_csv_path,
        vocab_json_path=config.frequent_motif_vocab_json_path,
        metrics_csv_path=config.frequent_motif_metrics_csv_path,
        metrics_jsonl_path=Path("outputs/v1/_unused_goal5_learned_frequent_metrics.jsonl"),
        summary_json_path=Path("outputs/v1/_unused_goal5_learned_frequent_summary.json"),
    )


def _select_best_result(
    pure_result: MotifCompressionSummary,
    macro_result: MotifCompressionSummary,
) -> tuple[MotifCompressionSummary, str]:
    if macro_result.compressed_node_count <= pure_result.compressed_node_count:
        return macro_result, "macro_graph"
    return pure_result, "pure_eml_dag"


def _distribution(values: Iterable[int | float | None]) -> dict[str, float | None]:
    numeric_values = [
        float(value) for value in values if value is not None and math.isfinite(float(value))
    ]
    if not numeric_values:
        return {"mean": None, "median": None, "p90": None}
    return {
        "mean": statistics.fmean(numeric_values),
        "median": statistics.median(numeric_values),
        "p90": _quantile(numeric_values, 0.9),
    }


def _learned_vs_random_distribution(
    rows: Sequence[LearnedMotifMetricRow],
) -> dict[str, float | None]:
    return _distribution(
        _safe_divide(row.random_motif_nodes, row.learned_motif_nodes) for row in rows
    )


def _quantile(values: Sequence[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def _coerce_config_value(key: str, value: object) -> object:
    path_keys = {
        "input_jsonl_path",
        "goal3_metrics_csv_path",
        "macro_graph_metrics_csv_path",
        "frequent_motif_vocab_json_path",
        "frequent_motif_metrics_csv_path",
        "learned_vocab_json_path",
        "metrics_csv_path",
        "metrics_jsonl_path",
        "summary_json_path",
        "train_log_json_path",
    }
    tuple_keys = {
        "operator_set",
        "symbol_names",
        "learned_vocab_sizes",
        "coverage_bonuses",
        "nontrivial_coverage_bonuses",
        "vocab_complexity_penalties",
        "expansion_complexity_penalties",
    }
    if key in path_keys and isinstance(value, str):
        return Path(value)
    if key in tuple_keys and isinstance(value, list):
        return tuple(value)
    return value


def _assert_no_outputs_v0(paths: Iterable[Path]) -> None:
    bad_paths = [path for path in paths if "outputs/v0" in path.as_posix()]
    if bad_paths:
        raise ValueError(f"Goal 5.3 learned motifs must not use outputs/v0: {bad_paths}")


def _csv_value(value: object) -> object:
    return "" if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
