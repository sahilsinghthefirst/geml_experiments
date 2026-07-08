"""Goal 5.5 hierarchical graph exports for future GNN dataset preparation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sympy as sp
import yaml

from geml.compression.graph_export import (
    graph_can_reconstruct_pure_eml_dag,
    graph_record_from_ast_tree,
    graph_record_from_dag,
    graph_record_from_macro_graph,
    graph_record_from_motif_compressed_graph,
    write_splits_json,
)
from geml.compression.graph_schema import (
    REPRESENTATION_MODES,
    GraphExportRecord,
    write_graph_schema,
)
from geml.compression.hierarchical_graph import build_hierarchical_eml_graph
from geml.compression.learned_motifs import load_learned_motif_vocabulary
from geml.compression.macro_expansions import validate_expansion_against_official
from geml.compression.macro_graph import build_macro_graph
from geml.compression.motif_dataset import SplitConfig, assign_split
from geml.compression.motif_mining import mining_graph_from_dag, mining_graph_from_macro_graph
from geml.compression.motif_rewrite import MotifCompressionResult, greedy_motif_compress_graph
from geml.compression.motif_vocab import MotifRecord, load_motif_vocabulary
from geml.data.dataset import (
    GeneratedExpressionInput,
    build_symbol_locals,
    load_generated_expressions,
    parse_generated_expression,
)
from geml.experiments.goal5_macro_graph_baseline import subset_label_for_metadata
from geml.symbolic.ast_graph import sympy_to_ast_tree
from geml.symbolic.dag_graph import tree_to_dag
from geml.symbolic.official_eml_compiler import sympy_to_official_eml_tree
from geml.symbolic.srepr import parse_srepr

type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]


@dataclass(frozen=True, slots=True)
class HierarchicalGraphExportConfig:
    """Configuration for Goal 5.5 v1 hierarchical graph export."""

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
    input_jsonl_path: Path = Path("outputs/v1/dag_compression_inputs.jsonl")
    frequent_motif_vocab_json_path: Path = Path("outputs/v1/goal5_frequent_motif_vocab.json")
    learned_motif_vocab_json_path: Path = Path("outputs/v1/goal5_learned_motif_vocab.json")
    egraph_safe_metrics_csv_path: Path = Path("outputs/v1/egraph_compression_metrics_safe.csv")
    egraph_positive_real_metrics_csv_path: Path = Path(
        "outputs/v1/egraph_compression_metrics_positive_real.csv"
    )
    graphs_jsonl_path: Path = Path("outputs/v1/goal5_hierarchical_graphs.jsonl")
    splits_json_path: Path = Path("outputs/v1/goal5_graph_splits.json")
    schema_json_path: Path = Path("outputs/v1/goal5_graph_schema.json")
    summary_json_path: Path = Path("outputs/v1/goal5_hierarchical_export_summary.json")

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.max_depth != 4:
            raise ValueError("Goal 5.5 v1 config expects max_depth=4")
        if tuple(self.operator_set) != ("add", "mul", "exp", "log"):
            raise ValueError("Goal 5.5 v1 operator_set must be add, mul, exp, log")
        if tuple(self.symbol_names) != ("x", "y"):
            raise ValueError("Goal 5.5 v1 symbol_names must be x, y")
        if self.source_serialization != "srepr":
            raise ValueError("Goal 5.5 requires authoritative srepr input")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be < 1")
        if self.min_motif_nodes <= 0:
            raise ValueError("min_motif_nodes must be positive")
        if self.max_motif_nodes < self.min_motif_nodes:
            raise ValueError("max_motif_nodes must be >= min_motif_nodes")
        _assert_no_outputs_v0(
            [
                self.input_jsonl_path,
                self.frequent_motif_vocab_json_path,
                self.learned_motif_vocab_json_path,
                self.egraph_safe_metrics_csv_path,
                self.egraph_positive_real_metrics_csv_path,
                self.graphs_jsonl_path,
                self.splits_json_path,
                self.schema_json_path,
                self.summary_json_path,
            ]
        )


@dataclass(frozen=True, slots=True)
class EgraphExportRow:
    """One valid e-graph extracted expression row."""

    index: int
    extracted_srepr: str
    subset_label: str
    rule_mode: str


@dataclass(slots=True)
class ExportAccumulator:
    """Streaming summary state for exported graph records."""

    graph_count: int = 0
    graph_ids: set[str] = field(default_factory=set)
    representation_modes: set[str] = field(default_factory=set)
    node_counts_by_mode: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    edge_counts_by_mode: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    expansion_valid_by_mode: dict[str, list[bool]] = field(
        default_factory=lambda: defaultdict(list)
    )
    reconstruction_valid_by_mode: dict[str, list[bool]] = field(
        default_factory=lambda: defaultdict(list)
    )
    missing_expansion_by_mode: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    train_val_test_graph_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    train_val_test_expression_ids: dict[str, set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    reconstruction_failures: int = 0
    expansion_failures: int = 0

    def add(self, record: GraphExportRecord) -> None:
        """Add one graph record to streaming stats."""
        self.graph_count += 1
        self.graph_ids.add(record.graph_id)
        self.representation_modes.add(record.representation_mode)
        self.node_counts_by_mode[record.representation_mode].append(len(record.nodes))
        self.edge_counts_by_mode[record.representation_mode].append(len(record.edges))
        self.expansion_valid_by_mode[record.representation_mode].append(
            record.validation.expansion_valid
        )
        self.reconstruction_valid_by_mode[record.representation_mode].append(
            graph_can_reconstruct_pure_eml_dag(record)
        )
        self.missing_expansion_by_mode[record.representation_mode] += (
            record.validation.missing_expansion_count
        )
        self.train_val_test_graph_counts[record.split] += 1
        self.train_val_test_expression_ids[record.split].add(record.source_expression_id)
        if not record.validation.expansion_valid:
            self.expansion_failures += 1
        if not graph_can_reconstruct_pure_eml_dag(record):
            self.reconstruction_failures += 1

    def to_summary(self) -> dict[str, object]:
        """Return JSON-safe summary state."""
        return {
            "graph_count": self.graph_count,
            "unique_graph_id_count": len(self.graph_ids),
            "all_graph_ids_unique": len(self.graph_ids) == self.graph_count,
            "representation_modes_exported": sorted(self.representation_modes),
            "node_edge_stats_by_mode": {
                mode: {
                    "graph_count": len(self.node_counts_by_mode[mode]),
                    "node_count": _distribution(self.node_counts_by_mode[mode]),
                    "edge_count": _distribution(self.edge_counts_by_mode[mode]),
                    "expansion_validation_rate": _percent(
                        sum(self.expansion_valid_by_mode[mode]),
                        len(self.expansion_valid_by_mode[mode]),
                    ),
                    "reconstruction_validation_rate": _percent(
                        sum(self.reconstruction_valid_by_mode[mode]),
                        len(self.reconstruction_valid_by_mode[mode]),
                    ),
                    "missing_expansion_count": self.missing_expansion_by_mode[mode],
                }
                for mode in sorted(self.node_counts_by_mode)
            },
            "expansion_validation_rate": _percent(
                self.graph_count - self.expansion_failures,
                self.graph_count,
            ),
            "reconstruction_validation_rate": _percent(
                self.graph_count - self.reconstruction_failures,
                self.graph_count,
            ),
            "missing_expansion_count": sum(self.missing_expansion_by_mode.values()),
            "train_val_test_counts": {
                split: {
                    "graph_count": self.train_val_test_graph_counts[split],
                    "expression_count": len(self.train_val_test_expression_ids[split]),
                }
                for split in ("train", "validation", "test")
            },
        }


@dataclass(frozen=True, slots=True)
class HierarchicalGraphExportResult:
    """Result summary for Goal 5.5."""

    summary: dict[str, object]
    output_paths: tuple[Path, ...]


def load_config(path: Path) -> HierarchicalGraphExportConfig:
    """Load a Goal 5.5 YAML config."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return HierarchicalGraphExportConfig(
        **{key: _coerce_config_value(key, value) for key, value in raw.items()}
    )


def run_goal5_hierarchical_export(
    config: HierarchicalGraphExportConfig,
) -> HierarchicalGraphExportResult:
    """Run the Goal 5.5 hierarchical export pipeline."""
    started_at = time.time()
    split_config = SplitConfig(
        seed=config.seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    input_rows = load_generated_expressions(config.input_jsonl_path)[: config.count]
    if len(input_rows) != config.count:
        raise ValueError(f"expected {config.count} input rows, found {len(input_rows)}")
    frequent_vocab = load_motif_vocabulary(config.frequent_motif_vocab_json_path)
    learned_vocab = load_learned_motif_vocabulary(config.learned_motif_vocab_json_path)
    frequent_motifs_by_id = {motif.motif_id: motif for motif in frequent_vocab.motifs}
    learned_source_motifs = [
        frequent_motifs_by_id[motif.source_motif_id]
        for motif in learned_vocab.motifs
        if motif.source_motif_id in frequent_motifs_by_id
    ]
    safe_rows = load_egraph_export_rows(config.egraph_safe_metrics_csv_path, rule_mode="safe")
    positive_rows = load_egraph_export_rows(
        config.egraph_positive_real_metrics_csv_path,
        rule_mode="positive_real_formal",
    )
    graph_ids_by_split: dict[str, list[str]] = defaultdict(list)
    expression_ids_by_split: dict[str, list[int]] = defaultdict(list)
    accumulator = ExportAccumulator()
    skipped_egraph_counts = {
        "safe": config.count - len(safe_rows),
        "positive_real_formal": config.count - len(positive_rows),
    }
    write_graph_schema(config.schema_json_path)
    config.graphs_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    symbol_locals = build_symbol_locals(config.symbol_names)
    with config.graphs_jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for position, input_row in enumerate(input_rows, start=1):
            if input_row.index is None:
                raise ValueError("input row index must not be None")
            split = assign_split(input_row.index, split_config)
            expression_ids_by_split[split].append(input_row.index)
            records = build_records_for_expression(
                input_row,
                symbol_locals=symbol_locals,
                split=split,
                frequent_motifs=frequent_vocab.motifs,
                learned_motifs=learned_source_motifs,
                safe_egraph_row=safe_rows.get(input_row.index),
                positive_egraph_row=positive_rows.get(input_row.index),
                config=config,
            )
            for record in records:
                jsonl_file.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
                accumulator.add(record)
                graph_ids_by_split[split].append(record.graph_id)
            if position % 500 == 0:
                print(f"Exported hierarchical graph rows: {position}/{len(input_rows)}", flush=True)

    write_splits_json(
        graph_ids_by_split=graph_ids_by_split,
        expression_ids_by_split=expression_ids_by_split,
        path=config.splits_json_path,
    )
    summary = {
        "config": config_to_json_dict(config),
        **accumulator.to_summary(),
        "expected_representation_modes": list(REPRESENTATION_MODES),
        "egraph_skipped_invalid_or_missing_counts": skipped_egraph_counts,
        "integrity": {
            "export_implies_model_performance": False,
            "compressed_graph_nodes_are_pure_eml": False,
            "compressed_nodes_expand_to_official_pure_eml": True,
            "schema_preserves_audit_reconstruction_links": True,
            "modified_official_eml_compiler": False,
            "trained_final_symbolic_reasoning_gnn": False,
            "hidden_target_labels_in_graph_records": False,
            "safe_and_positive_real_modes_mixed_without_labels": False,
        },
        "elapsed_seconds": time.time() - started_at,
        "completed_at_unix": time.time(),
    }
    write_json(config.summary_json_path, summary)
    return HierarchicalGraphExportResult(
        summary=summary,
        output_paths=(
            config.graphs_jsonl_path,
            config.splits_json_path,
            config.schema_json_path,
            config.summary_json_path,
        ),
    )


def build_records_for_expression(
    input_row: GeneratedExpressionInput,
    *,
    symbol_locals: dict[str, sp.Symbol],
    split: str,
    frequent_motifs: Sequence[MotifRecord],
    learned_motifs: Sequence[MotifRecord],
    safe_egraph_row: EgraphExportRow | None,
    positive_egraph_row: EgraphExportRow | None,
    config: HierarchicalGraphExportConfig,
) -> tuple[GraphExportRecord, ...]:
    """Build all graph records for one source expression."""
    if input_row.index is None:
        raise ValueError("input row index must not be None")
    sympy_expr, source_serialization = parse_generated_expression(
        input_row, symbol_locals=symbol_locals
    )
    if source_serialization != config.source_serialization:
        raise ValueError(f"expected srepr input, got {source_serialization}")
    subset_label = subset_label_for_metadata(input_row.metadata)
    ast_tree = sympy_to_ast_tree(sympy_expr)
    ast_dag = tree_to_dag(ast_tree)
    eml_tree = sympy_to_official_eml_tree(sympy_expr)
    pure_dag = tree_to_dag(eml_tree)
    macro_graph = build_macro_graph(sympy_expr)
    macro_validation = validate_expansion_against_official(macro_graph, sympy_expr)

    ast_record = graph_record_from_ast_tree(
        ast_tree,
        graph_id=_graph_id(input_row.index, "ast_tree_graph"),
        source_expression_id=input_row.index,
        subset_label=subset_label,
        split=split,
    )
    ast_dag_record = graph_record_from_dag(
        ast_dag,
        graph_id=_graph_id(input_row.index, "ast_dag_graph"),
        representation_mode="ast_dag_graph",
        source_expression_id=input_row.index,
        subset_label=subset_label,
        split=split,
        source_label="source_ast_dag",
    )
    pure_record = graph_record_from_dag(
        pure_dag,
        graph_id=_graph_id(input_row.index, "pure_eml_dag_graph"),
        representation_mode="pure_eml_dag_graph",
        source_expression_id=input_row.index,
        subset_label=subset_label,
        split=split,
        source_label="official_pure_eml_dag",
    )
    macro_record = graph_record_from_macro_graph(
        macro_graph,
        graph_id=_graph_id(input_row.index, "macro_graph"),
        source_expression_id=input_row.index,
        subset_label=subset_label,
        split=split,
        pure_eml_target_ids=[pure_record.root_node_id],
    )
    macro_record = macro_record.model_copy(
        update={
            "metadata": {
                **macro_record.metadata,
                "reconstruction_valid": macro_validation.expansion_valid,
            }
        }
    )
    pure_mining = mining_graph_from_dag(
        pure_dag,
        graph_id=f"pure:{input_row.index}",
        expression_index=input_row.index,
    )
    macro_mining = mining_graph_from_macro_graph(
        macro_graph,
        graph_id=f"macro:{input_row.index}",
        expression_index=input_row.index,
    )
    motif_result = _best_compressed_graph(
        pure_mining,
        macro_mining,
        motifs=frequent_motifs,
        expression_index=input_row.index,
        subset_label=subset_label,
        config=config,
    )
    motif_record = graph_record_from_motif_compressed_graph(
        motif_result.compressed_graph,
        graph_id=_graph_id(input_row.index, "frequent_motif_graph"),
        representation_mode="frequent_motif_graph",
        source_expression_id=input_row.index,
        subset_label=subset_label,
        split=split,
        pure_eml_target_ids=[pure_record.root_node_id],
    )
    learned_result = _best_compressed_graph(
        pure_mining,
        macro_mining,
        motifs=learned_motifs,
        expression_index=input_row.index,
        subset_label=subset_label,
        config=config,
    )
    learned_record = graph_record_from_motif_compressed_graph(
        learned_result.compressed_graph,
        graph_id=_graph_id(input_row.index, "learned_motif_graph"),
        representation_mode="learned_motif_graph",
        source_expression_id=input_row.index,
        subset_label=subset_label,
        split=split,
        pure_eml_target_ids=[pure_record.root_node_id],
        learned=True,
    )
    hierarchical_record = build_hierarchical_eml_graph(
        graph_id=_graph_id(input_row.index, "hierarchical_eml_graph"),
        source_expression_id=input_row.index,
        subset_label=subset_label,
        split=split,
        ast_graph=ast_record,
        macro_graph=macro_record,
        pure_eml_graph=pure_record,
        motif_graph=motif_record,
        learned_motif_graph=learned_record,
    )
    records = [
        ast_record,
        ast_dag_record,
        pure_record,
        macro_record,
        motif_record,
        learned_record,
        hierarchical_record,
    ]
    for egraph_row, mode in (
        (safe_egraph_row, "egraph_safe_eml_dag_graph"),
        (positive_egraph_row, "egraph_positive_real_eml_dag_graph"),
    ):
        if egraph_row is None:
            continue
        egraph_expr = parse_srepr(egraph_row.extracted_srepr)
        egraph_dag = tree_to_dag(sympy_to_official_eml_tree(egraph_expr))
        records.append(
            graph_record_from_dag(
                egraph_dag,
                graph_id=_graph_id(input_row.index, mode),
                representation_mode=mode,  # type: ignore[arg-type]
                source_expression_id=input_row.index,
                subset_label=egraph_row.subset_label,
                split=split,
                source_label=f"goal4_{egraph_row.rule_mode}_extracted_eml_dag",
            )
        )
    return tuple(records)


def load_egraph_export_rows(path: Path, *, rule_mode: str) -> dict[int, EgraphExportRow]:
    """Load valid Goal 4 e-graph extraction rows for graph export."""
    rows: dict[int, EgraphExportRow] = {}
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw in csv.DictReader(csv_file):
            if raw.get("extraction_status") != "completed":
                continue
            if raw.get("validation_status") != "valid":
                continue
            if raw.get("structural_purity_valid") != "True":
                continue
            extracted_srepr = raw.get("extracted_srepr")
            if not extracted_srepr:
                continue
            rows[int(raw["index"])] = EgraphExportRow(
                index=int(raw["index"]),
                extracted_srepr=extracted_srepr,
                subset_label=raw["subset_label"],
                rule_mode=rule_mode,
            )
    return rows


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def config_to_json_dict(config: HierarchicalGraphExportConfig) -> dict[str, object]:
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
        "input_jsonl_path": str(config.input_jsonl_path),
        "frequent_motif_vocab_json_path": str(config.frequent_motif_vocab_json_path),
        "learned_motif_vocab_json_path": str(config.learned_motif_vocab_json_path),
        "egraph_safe_metrics_csv_path": str(config.egraph_safe_metrics_csv_path),
        "egraph_positive_real_metrics_csv_path": str(config.egraph_positive_real_metrics_csv_path),
        "graphs_jsonl_path": str(config.graphs_jsonl_path),
        "splits_json_path": str(config.splits_json_path),
        "schema_json_path": str(config.schema_json_path),
        "summary_json_path": str(config.summary_json_path),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hierarchical_graph_export_v1.yaml"),
        help="Path to a YAML Goal 5.5 graph export config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Goal 5.5 hierarchical graph export."""
    args = build_parser().parse_args(argv)
    result = run_goal5_hierarchical_export(load_config(args.config))
    print(f"Exported graphs: {result.summary['graph_count']}")
    for path in result.output_paths:
        print(f"Wrote: {path}")
    return 0


def _best_compressed_graph(
    pure_graph: object,
    macro_graph: object,
    *,
    motifs: Sequence[MotifRecord],
    expression_index: int,
    subset_label: str,
    config: HierarchicalGraphExportConfig,
) -> MotifCompressionResult:
    pure_result = greedy_motif_compress_graph(
        pure_graph,  # type: ignore[arg-type]
        [motif for motif in motifs if motif.motif_type == "pure_eml_dag"],
        min_motif_nodes=config.min_motif_nodes,
        max_motif_nodes=config.max_motif_nodes,
        expression_index=expression_index,
        subset_label=subset_label,
    )
    macro_result = greedy_motif_compress_graph(
        macro_graph,  # type: ignore[arg-type]
        [motif for motif in motifs if motif.motif_type == "macro_graph"],
        min_motif_nodes=config.min_motif_nodes,
        max_motif_nodes=config.max_motif_nodes,
        expression_index=expression_index,
        subset_label=subset_label,
    )
    if macro_result.compressed_node_count <= pure_result.compressed_node_count:
        return macro_result
    return pure_result


def _graph_id(index: int, representation_mode: str) -> str:
    return f"v1:{index}:{representation_mode}"


def _distribution(values: Iterable[int | float]) -> dict[str, float | None]:
    numeric_values = [float(value) for value in values if math.isfinite(float(value))]
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
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * float(numerator) / float(denominator)


def _coerce_config_value(key: str, value: object) -> object:
    path_keys = {
        "input_jsonl_path",
        "frequent_motif_vocab_json_path",
        "learned_motif_vocab_json_path",
        "egraph_safe_metrics_csv_path",
        "egraph_positive_real_metrics_csv_path",
        "graphs_jsonl_path",
        "splits_json_path",
        "schema_json_path",
        "summary_json_path",
    }
    tuple_keys = {"operator_set", "symbol_names"}
    if key in path_keys and isinstance(value, str):
        return Path(value)
    if key in tuple_keys and isinstance(value, list):
        return tuple(value)
    return value


def _assert_no_outputs_v0(paths: Sequence[Path]) -> None:
    bad_paths = [path for path in paths if "outputs/v0" in path.as_posix()]
    if bad_paths:
        raise ValueError(f"Goal 5.5 graph export must not use outputs/v0: {bad_paths}")


if __name__ == "__main__":
    raise SystemExit(main())
