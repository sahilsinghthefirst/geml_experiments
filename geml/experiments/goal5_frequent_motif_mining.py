"""Goal 5.2 frequent motif mining over v1 pure EML-DAGs and macro graphs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import sympy as sp
import yaml

from geml.compression.macro_graph import build_macro_graph
from geml.compression.motif_mining import (
    MiningGraph,
    build_motif_vocabulary,
    mine_frequent_motifs,
    mining_graph_from_dag,
    mining_graph_from_macro_graph,
)
from geml.compression.motif_rewrite import (
    MotifCompressionSummary,
    greedy_motif_compress_graph_summary,
)
from geml.compression.motif_vocab import (
    MotifVocabulary,
    motif_to_summary_dict,
    write_motif_vocabulary,
)
from geml.data.dataset import (
    GeneratedExpressionInput,
    build_symbol_locals,
    load_generated_expressions,
    parse_generated_expression,
)
from geml.experiments.goal5_macro_graph_baseline import subset_label_for_metadata
from geml.symbolic.dag_graph import tree_to_dag
from geml.symbolic.official_eml_compiler import sympy_to_official_eml_tree

type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]

MOTIF_METRICS_FIELDS = [
    "index",
    "expression",
    "srepr",
    "subset_label",
    "original_eml_dag_nodes",
    "macro_graph_nodes",
    "eml_motif_compressed_nodes",
    "macro_motif_compressed_nodes",
    "motif_compressed_nodes",
    "motif_compressed_child_refs",
    "motif_vocab_size",
    "motif_coverage_percent",
    "compression_gain_vs_goal3_eml_dag",
    "compression_gain_vs_macro_graph",
    "eml_expansion_valid",
    "macro_expansion_valid",
    "expansion_valid",
    "selected_motif_count",
    "selected_graph_type",
    "error",
]


@dataclass(frozen=True, slots=True)
class FrequentMotifMiningConfig:
    """Configuration for the Goal 5.2 v1 frequent motif mining baseline."""

    seed: int = 0
    count: int = 10_000
    max_depth: int = 4
    operator_set: tuple[str, ...] = ("add", "mul", "exp", "log")
    symbol_names: tuple[str, ...] = ("x", "y")
    source_serialization: Literal["srepr"] = "srepr"
    min_motif_nodes: int = 1
    max_motif_nodes: int = 2
    min_support: int = 50
    max_vocab_size: int = 90
    input_jsonl_path: Path = Path("outputs/v1/dag_compression_inputs.jsonl")
    goal3_metrics_csv_path: Path = Path("outputs/v1/dag_compression_metrics.csv")
    macro_graph_metrics_csv_path: Path = Path("outputs/v1/goal5_macro_graph_metrics.csv")
    vocab_json_path: Path = Path("outputs/v1/goal5_frequent_motif_vocab.json")
    metrics_csv_path: Path = Path("outputs/v1/goal5_frequent_motif_metrics.csv")
    metrics_jsonl_path: Path = Path("outputs/v1/goal5_frequent_motif_metrics.jsonl")
    summary_json_path: Path = Path("outputs/v1/goal5_frequent_motif_summary.json")

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.max_depth != 4:
            raise ValueError("Goal 5.2 v1 config expects max_depth=4")
        if tuple(self.operator_set) != ("add", "mul", "exp", "log"):
            raise ValueError("Goal 5.2 v1 operator_set must be add, mul, exp, log")
        if tuple(self.symbol_names) != ("x", "y"):
            raise ValueError("Goal 5.2 v1 symbol_names must be x, y")
        if self.source_serialization != "srepr":
            raise ValueError("Goal 5.2 requires authoritative srepr input")
        if self.min_motif_nodes <= 0:
            raise ValueError("min_motif_nodes must be positive")
        if self.max_motif_nodes < self.min_motif_nodes:
            raise ValueError("max_motif_nodes must be >= min_motif_nodes")
        if self.min_support <= 0:
            raise ValueError("min_support must be positive")
        if self.max_vocab_size <= 0:
            raise ValueError("max_vocab_size must be positive")
        _assert_no_outputs_v0(
            [
                self.input_jsonl_path,
                self.goal3_metrics_csv_path,
                self.macro_graph_metrics_csv_path,
                self.vocab_json_path,
                self.metrics_csv_path,
                self.metrics_jsonl_path,
                self.summary_json_path,
            ]
        )


@dataclass(frozen=True, slots=True)
class Goal3BaselineRow:
    """Goal 3 v1 baseline row."""

    index: int
    expression: str
    srepr: str
    ast_tree_node_count: int
    ast_dag_node_count: int
    eml_tree_node_count: int
    eml_dag_node_count: int
    pure_eml_valid: bool


@dataclass(frozen=True, slots=True)
class MacroGraphMetricRow:
    """Goal 5.1 macro graph metric row."""

    index: int
    macro_graph_nodes: int
    macro_graph_edges_or_child_refs: int
    expansion_valid: bool


@dataclass(frozen=True, slots=True)
class GraphBundle:
    """Both graph families for one expression."""

    input_row: GeneratedExpressionInput
    baseline: Goal3BaselineRow
    macro_metrics: MacroGraphMetricRow
    sympy_expr: sp.Expr
    subset_label: str
    pure_graph: MiningGraph
    macro_graph: MiningGraph


@dataclass(frozen=True, slots=True)
class FrequentMotifMetricRow:
    """One per-expression Goal 5.2 motif compression metric row."""

    index: int
    expression: str
    srepr: str
    subset_label: str
    original_eml_dag_nodes: int
    macro_graph_nodes: int
    eml_motif_compressed_nodes: int | None
    macro_motif_compressed_nodes: int | None
    motif_compressed_nodes: int | None
    motif_compressed_child_refs: int | None
    motif_vocab_size: int
    motif_coverage_percent: float | None
    compression_gain_vs_goal3_eml_dag: float | None
    compression_gain_vs_macro_graph: float | None
    eml_expansion_valid: bool
    macro_expansion_valid: bool
    expansion_valid: bool
    selected_motif_count: int | None
    selected_graph_type: str | None
    error: str | None

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable row."""
        return {field: getattr(self, field) for field in MOTIF_METRICS_FIELDS}

    def to_csv_dict(self) -> dict[str, object]:
        """Return a CSV-serializable row."""
        return {field: _csv_value(getattr(self, field)) for field in MOTIF_METRICS_FIELDS}


@dataclass(frozen=True, slots=True)
class FrequentMotifMiningResult:
    """Result summary for Goal 5.2 motif mining."""

    summary: dict[str, object]
    output_paths: tuple[Path, ...]


def load_config(path: Path) -> FrequentMotifMiningConfig:
    """Load a Goal 5.2 frequent motif YAML config."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return FrequentMotifMiningConfig(
        **{key: _coerce_config_value(key, value) for key, value in raw.items()}
    )


def run_goal5_frequent_motif_mining(
    config: FrequentMotifMiningConfig,
) -> FrequentMotifMiningResult:
    """Run frequent motif mining and greedy replacement on the configured v1 corpus."""
    started_at = time.time()
    bundles = build_graph_bundles(config)
    print(f"Built graph bundles: {len(bundles)}", flush=True)
    subset_labels_by_graph_id = {
        bundle.pure_graph.graph_id: bundle.subset_label for bundle in bundles
    } | {bundle.macro_graph.graph_id: bundle.subset_label for bundle in bundles}
    expression_indices_by_graph_id = {
        bundle.pure_graph.graph_id: bundle.baseline.index for bundle in bundles
    } | {bundle.macro_graph.graph_id: bundle.baseline.index for bundle in bundles}

    pure_records = mine_frequent_motifs(
        [bundle.pure_graph for bundle in bundles],
        min_motif_nodes=config.min_motif_nodes,
        max_motif_nodes=config.max_motif_nodes,
        min_support=config.min_support,
        max_vocab_size=config.max_vocab_size,
        subset_labels_by_graph_id=subset_labels_by_graph_id,
        expression_indices_by_graph_id=expression_indices_by_graph_id,
    )
    print(f"Mined pure EML-DAG motifs: {len(pure_records)}", flush=True)
    macro_records = mine_frequent_motifs(
        [bundle.macro_graph for bundle in bundles],
        min_motif_nodes=config.min_motif_nodes,
        max_motif_nodes=config.max_motif_nodes,
        min_support=config.min_support,
        max_vocab_size=config.max_vocab_size,
        subset_labels_by_graph_id=subset_labels_by_graph_id,
        expression_indices_by_graph_id=expression_indices_by_graph_id,
    )
    print(f"Mined macro graph motifs: {len(macro_records)}", flush=True)
    vocabulary = build_motif_vocabulary(
        pure_records=pure_records,
        macro_records=macro_records,
        max_vocab_size=config.max_vocab_size,
        config=config_to_json_dict(config),
    )
    write_motif_vocabulary(vocabulary, config.vocab_json_path)
    print(f"Wrote motif vocabulary: {vocabulary.motif_count}", flush=True)

    rows = tuple(compute_motif_metric_row(bundle, vocabulary, config) for bundle in bundles)
    print(f"Computed motif compression rows: {len(rows)}", flush=True)
    write_metrics_jsonl(rows, config.metrics_jsonl_path)
    write_metrics_csv(rows, config.metrics_csv_path)
    summary = build_summary(
        rows,
        vocabulary,
        config,
        started_at=started_at,
        completed_at=time.time(),
    )
    write_json(config.summary_json_path, summary)
    return FrequentMotifMiningResult(
        summary=summary,
        output_paths=(
            config.vocab_json_path,
            config.metrics_csv_path,
            config.metrics_jsonl_path,
            config.summary_json_path,
        ),
    )


def build_graph_bundles(config: FrequentMotifMiningConfig) -> tuple[GraphBundle, ...]:
    """Load v1 inputs and reconstruct pure EML-DAG and macro graph families."""
    input_rows = load_generated_expressions(config.input_jsonl_path)[: config.count]
    if len(input_rows) != config.count:
        raise ValueError(f"expected {config.count} input rows, found {len(input_rows)}")
    baselines = load_goal3_baselines(config.goal3_metrics_csv_path)
    macro_metrics = load_macro_graph_metrics(config.macro_graph_metrics_csv_path)
    symbol_locals = build_symbol_locals(config.symbol_names)
    bundles: list[GraphBundle] = []
    for input_row in input_rows:
        if input_row.index is None:
            raise ValueError("input row index must not be None")
        baseline = baselines[input_row.index]
        macro_metric = macro_metrics[input_row.index]
        sympy_expr, source_serialization = parse_generated_expression(
            input_row,
            symbol_locals=symbol_locals,
        )
        if source_serialization != config.source_serialization:
            raise ValueError(f"expected srepr input, got {source_serialization}")
        eml_tree = sympy_to_official_eml_tree(sympy_expr)
        pure_dag = tree_to_dag(eml_tree)
        macro_graph = build_macro_graph(sympy_expr)
        bundles.append(
            GraphBundle(
                input_row=input_row,
                baseline=baseline,
                macro_metrics=macro_metric,
                sympy_expr=sympy_expr,
                subset_label=subset_label_for_metadata(input_row.metadata),
                pure_graph=mining_graph_from_dag(
                    pure_dag,
                    graph_id=f"pure:{input_row.index}",
                    expression_index=input_row.index,
                ),
                macro_graph=mining_graph_from_macro_graph(
                    macro_graph,
                    graph_id=f"macro:{input_row.index}",
                    expression_index=input_row.index,
                ),
            )
        )
    return tuple(bundles)


def compute_motif_metric_row(
    bundle: GraphBundle,
    vocabulary: MotifVocabulary,
    config: FrequentMotifMiningConfig,
) -> FrequentMotifMetricRow:
    """Compute one per-expression motif compression metric row."""
    try:
        pure_result = greedy_motif_compress_graph_summary(
            bundle.pure_graph,
            vocabulary.motifs_by_type("pure_eml_dag"),
            min_motif_nodes=config.min_motif_nodes,
            max_motif_nodes=config.max_motif_nodes,
            expression_index=bundle.baseline.index,
            subset_label=bundle.subset_label,
        )
        macro_result = greedy_motif_compress_graph_summary(
            bundle.macro_graph,
            vocabulary.motifs_by_type("macro_graph"),
            min_motif_nodes=config.min_motif_nodes,
            max_motif_nodes=config.max_motif_nodes,
            expression_index=bundle.baseline.index,
            subset_label=bundle.subset_label,
        )
        selected_result, selected_graph_type = _select_best_result(pure_result, macro_result)
        motif_compressed_nodes = selected_result.compressed_node_count
        return FrequentMotifMetricRow(
            index=bundle.baseline.index,
            expression=str(bundle.sympy_expr),
            srepr=bundle.input_row.srepr or sp.srepr(bundle.sympy_expr),
            subset_label=bundle.subset_label,
            original_eml_dag_nodes=bundle.baseline.eml_dag_node_count,
            macro_graph_nodes=bundle.macro_metrics.macro_graph_nodes,
            eml_motif_compressed_nodes=pure_result.compressed_node_count,
            macro_motif_compressed_nodes=macro_result.compressed_node_count,
            motif_compressed_nodes=motif_compressed_nodes,
            motif_compressed_child_refs=selected_result.compressed_child_ref_count,
            motif_vocab_size=vocabulary.motif_count,
            motif_coverage_percent=selected_result.motif_coverage_percent,
            compression_gain_vs_goal3_eml_dag=_safe_divide(
                bundle.baseline.eml_dag_node_count,
                motif_compressed_nodes,
            ),
            compression_gain_vs_macro_graph=_safe_divide(
                bundle.macro_metrics.macro_graph_nodes,
                motif_compressed_nodes,
            ),
            eml_expansion_valid=pure_result.expansion_valid,
            macro_expansion_valid=macro_result.expansion_valid,
            expansion_valid=pure_result.expansion_valid and macro_result.expansion_valid,
            selected_motif_count=selected_result.selected_replacement_count,
            selected_graph_type=selected_graph_type,
            error=None,
        )
    except Exception as exc:
        return FrequentMotifMetricRow(
            index=bundle.baseline.index,
            expression=str(bundle.sympy_expr),
            srepr=bundle.input_row.srepr or sp.srepr(bundle.sympy_expr),
            subset_label=bundle.subset_label,
            original_eml_dag_nodes=bundle.baseline.eml_dag_node_count,
            macro_graph_nodes=bundle.macro_metrics.macro_graph_nodes,
            eml_motif_compressed_nodes=None,
            macro_motif_compressed_nodes=None,
            motif_compressed_nodes=None,
            motif_compressed_child_refs=None,
            motif_vocab_size=vocabulary.motif_count,
            motif_coverage_percent=None,
            compression_gain_vs_goal3_eml_dag=None,
            compression_gain_vs_macro_graph=None,
            eml_expansion_valid=False,
            macro_expansion_valid=False,
            expansion_valid=False,
            selected_motif_count=None,
            selected_graph_type=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def build_summary(
    rows: Sequence[FrequentMotifMetricRow],
    vocabulary: MotifVocabulary,
    config: FrequentMotifMiningConfig,
    *,
    started_at: float,
    completed_at: float,
) -> dict[str, object]:
    """Build required Goal 5.2 summary metrics."""
    success_rows = [row for row in rows if row.expansion_valid and row.error is None]
    return {
        "config": config_to_json_dict(config),
        "processed_count": len(rows),
        "success_count": len(success_rows),
        "expansion_validation_failure_count": sum(not row.expansion_valid for row in rows),
        "motif_vocab_size": vocabulary.motif_count,
        "motif_counts_by_type": {
            motif_type: len(vocabulary.motifs_by_type(motif_type))
            for motif_type in ("pure_eml_dag", "macro_graph", "mixed_macro_expansion")
        },
        "motif_compressed_nodes": _distribution(row.motif_compressed_nodes for row in success_rows),
        "motif_coverage_percent": _distribution(row.motif_coverage_percent for row in success_rows),
        "compression_gain_vs_goal3_eml_dag": _distribution(
            row.compression_gain_vs_goal3_eml_dag for row in success_rows
        ),
        "compression_gain_vs_macro_graph": _distribution(
            row.compression_gain_vs_macro_graph for row in success_rows
        ),
        "results_by_subset_label": {
            label: summarize_subset(rows, label)
            for label in ("all_v1", "nontrivial_v1", "identity_heavy_v1")
        },
        "top_motifs_by_support": [
            motif_to_summary_dict(motif)
            for motif in sorted(vocabulary.motifs, key=lambda item: -item.support_count)[:20]
        ],
        "top_motifs_by_compression_saved": [
            motif_to_summary_dict(motif)
            for motif in sorted(vocabulary.motifs, key=lambda item: -item.compression_score)[:20]
        ],
        "top_motifs_by_nontrivial_v1_coverage": [
            motif_to_summary_dict(motif)
            for motif in sorted(
                vocabulary.motifs,
                key=lambda item: -item.covered_nodes_by_subset_label.get("nontrivial_v1", 0),
            )[:20]
        ],
        "motifs_that_correspond_to_official_macros": [
            motif_to_summary_dict(motif)
            for motif in vocabulary.motifs
            if motif.is_obvious_official_macro
        ][:50],
        "motifs_not_obvious_official_macros": [
            motif_to_summary_dict(motif)
            for motif in vocabulary.motifs
            if not motif.is_obvious_official_macro
        ][:50],
        "representation_contract": {
            "motif_nodes_are_pure_eml": False,
            "every_motif_has_expansion_map_to_original_graph": all(
                bool(motif.expansion_map_to_original_graph) for motif in vocabulary.motifs
            ),
            "reports_motif_metrics_separately": True,
            "trained_neural_models": False,
        },
        "elapsed_seconds": completed_at - started_at,
        "completed_at_unix": completed_at,
    }


def summarize_subset(rows: Sequence[FrequentMotifMetricRow], label: str) -> dict[str, object]:
    """Summarize one required v1 subset label."""
    subset_rows = (
        list(rows) if label == "all_v1" else [row for row in rows if row.subset_label == label]
    )
    success_rows = [row for row in subset_rows if row.expansion_valid and row.error is None]
    return {
        "processed_count": len(subset_rows),
        "success_count": len(success_rows),
        "expansion_validation_failure_count": sum(not row.expansion_valid for row in subset_rows),
        "motif_coverage_percent": _distribution(row.motif_coverage_percent for row in success_rows),
        "compression_gain_vs_goal3_eml_dag": _distribution(
            row.compression_gain_vs_goal3_eml_dag for row in success_rows
        ),
        "compression_gain_vs_macro_graph": _distribution(
            row.compression_gain_vs_macro_graph for row in success_rows
        ),
    }


def load_goal3_baselines(path: Path) -> dict[int, Goal3BaselineRow]:
    """Load Goal 3 v1 DAG baseline rows."""
    rows: dict[int, Goal3BaselineRow] = {}
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            if raw_row.get("supported") != "True":
                continue
            row = Goal3BaselineRow(
                index=int(raw_row["index"]),
                expression=raw_row["expression"],
                srepr=raw_row["srepr"],
                ast_tree_node_count=int(raw_row["ast_tree_node_count"]),
                ast_dag_node_count=int(raw_row["ast_dag_node_count"]),
                eml_tree_node_count=int(raw_row["eml_tree_node_count"]),
                eml_dag_node_count=int(raw_row["eml_dag_node_count"]),
                pure_eml_valid=_parse_bool(raw_row["pure_eml_valid"]),
            )
            rows[row.index] = row
    return rows


def load_macro_graph_metrics(path: Path) -> dict[int, MacroGraphMetricRow]:
    """Load Goal 5.1 macro graph metric rows."""
    rows: dict[int, MacroGraphMetricRow] = {}
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            if raw_row.get("success") != "True":
                continue
            row = MacroGraphMetricRow(
                index=int(raw_row["index"]),
                macro_graph_nodes=int(raw_row["macro_graph_nodes"]),
                macro_graph_edges_or_child_refs=int(raw_row["macro_graph_edges_or_child_refs"]),
                expansion_valid=_parse_bool(raw_row["expansion_valid"]),
            )
            rows[row.index] = row
    return rows


def write_metrics_jsonl(rows: Sequence[FrequentMotifMetricRow], path: Path) -> None:
    """Write per-expression motif metrics to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row.to_json_dict(), sort_keys=True) + "\n")


def write_metrics_csv(rows: Sequence[FrequentMotifMetricRow], path: Path) -> None:
    """Write per-expression motif metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MOTIF_METRICS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def write_json(path: Path, data: dict[str, object]) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def config_to_json_dict(config: FrequentMotifMiningConfig) -> dict[str, object]:
    """Return JSON-safe config values."""
    return {
        "seed": config.seed,
        "count": config.count,
        "max_depth": config.max_depth,
        "operator_set": list(config.operator_set),
        "symbol_names": list(config.symbol_names),
        "source_serialization": config.source_serialization,
        "min_motif_nodes": config.min_motif_nodes,
        "max_motif_nodes": config.max_motif_nodes,
        "min_support": config.min_support,
        "max_vocab_size": config.max_vocab_size,
        "input_jsonl_path": str(config.input_jsonl_path),
        "goal3_metrics_csv_path": str(config.goal3_metrics_csv_path),
        "macro_graph_metrics_csv_path": str(config.macro_graph_metrics_csv_path),
        "vocab_json_path": str(config.vocab_json_path),
        "metrics_csv_path": str(config.metrics_csv_path),
        "metrics_jsonl_path": str(config.metrics_jsonl_path),
        "summary_json_path": str(config.summary_json_path),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/frequent_motifs_v1.yaml"),
        help="Path to a YAML Goal 5.2 frequent motif config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 5.2 frequent motif mining baseline."""
    args = build_parser().parse_args(argv)
    result = run_goal5_frequent_motif_mining(load_config(args.config))
    print(f"Processed: {result.summary['processed_count']}")
    print(f"Succeeded: {result.summary['success_count']}")
    print(f"Motif vocab size: {result.summary['motif_vocab_size']}")
    for path in result.output_paths:
        print(f"Wrote: {path}")
    return 0


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


def _safe_divide(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return float(numerator) / float(denominator)


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected boolean string, got {value!r}")


def _coerce_config_value(key: str, value: object) -> object:
    path_keys = {
        "input_jsonl_path",
        "goal3_metrics_csv_path",
        "macro_graph_metrics_csv_path",
        "vocab_json_path",
        "metrics_csv_path",
        "metrics_jsonl_path",
        "summary_json_path",
    }
    tuple_keys = {"operator_set", "symbol_names"}
    if key in path_keys and isinstance(value, str):
        return Path(value)
    if key in tuple_keys and isinstance(value, list):
        return tuple(str(item) for item in value)
    return value


def _assert_no_outputs_v0(paths: Iterable[Path]) -> None:
    bad_paths = [path for path in paths if "outputs/v0" in path.as_posix()]
    if bad_paths:
        raise ValueError(f"Goal 5.2 frequent motifs must not use outputs/v0: {bad_paths}")


def _csv_value(value: object) -> object:
    return "" if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
