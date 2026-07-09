"""Goal 5.4 neural cost model for e-graph extraction ranking."""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from geml.compression.egraph_candidate_dataset import (
    CandidateGenerationConfig,
    load_or_build_candidate_records,
    summarize_candidate_dataset,
)
from geml.compression.motif_dataset import SplitConfig
from geml.compression.neural_cost_model import (
    NeuralCostModelConfig,
    train_neural_cost_model,
)
from geml.compression.neural_egraph_extractor import (
    NEURAL_EGRAPH_METRICS_FIELDS,
    NeuralEgraphMetricRow,
    build_neural_egraph_summary,
    evaluate_neural_egraph_extractor,
    summarize_neural_rows,
)
from geml.egraph.rule_sets import RuleMode
from geml.experiments.shared import build_run_metadata, write_json_object

type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]


@dataclass(frozen=True, slots=True)
class NeuralEgraphExtractorConfig:
    """Configuration for Goal 5.4 neural e-graph extraction."""

    seed: int = 0
    count: int = 10_000
    max_depth: int = 4
    operator_set: tuple[str, ...] = ("add", "mul", "exp", "log")
    symbol_names: tuple[str, ...] = ("x", "y")
    source_serialization: str = "srepr"
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    run_modes: tuple[RuleMode, ...] = ("safe", "positive_real_formal")
    max_iterations: int = 4
    max_enodes: int = 5_000
    max_eclasses: int = 5_000
    saturation_timeout_seconds: float = 0.5
    row_timeout_seconds: float = 2.0
    beam_size: int = 12
    max_candidate_depth: int = 7
    max_candidates_evaluated: int = 12
    hidden_size: int = 12
    epochs: int = 12
    learning_rate: float = 0.01
    max_pairs_per_group: int = 8
    l2_penalty: float = 1e-5
    reuse_existing_candidate_dataset: bool = True
    input_jsonl_path: Path = Path("outputs/v1/dag_compression_inputs.jsonl")
    goal3_metrics_csv_path: Path = Path("outputs/v1/dag_compression_metrics.csv")
    candidate_dataset_jsonl_path: Path = Path(
        "outputs/v1/goal5_neural_egraph_candidate_dataset.jsonl"
    )
    metrics_csv_path: Path = Path("outputs/v1/goal5_neural_egraph_metrics.csv")
    summary_json_path: Path = Path("outputs/v1/goal5_neural_egraph_summary.json")
    train_log_json_path: Path = Path("outputs/v1/goal5_neural_egraph_train_log.json")

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.max_depth != 4:
            raise ValueError("Goal 5.4 v1 config expects max_depth=4")
        if tuple(self.operator_set) != ("add", "mul", "exp", "log"):
            raise ValueError("Goal 5.4 v1 operator_set must be add, mul, exp, log")
        if tuple(self.symbol_names) != ("x", "y"):
            raise ValueError("Goal 5.4 v1 symbol_names must be x, y")
        if self.source_serialization != "srepr":
            raise ValueError("Goal 5.4 requires authoritative srepr input")
        unknown_modes = set(self.run_modes) - {"safe", "positive_real_formal"}
        if unknown_modes:
            raise ValueError(f"unsupported rule modes: {sorted(unknown_modes)}")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be < 1")
        _assert_no_outputs_v0(
            [
                self.input_jsonl_path,
                self.goal3_metrics_csv_path,
                self.candidate_dataset_jsonl_path,
                self.metrics_csv_path,
                self.summary_json_path,
                self.train_log_json_path,
            ]
        )


@dataclass(frozen=True, slots=True)
class NeuralEgraphExtractorResult:
    """Result payload for a Goal 5.4 run."""

    summary: dict[str, object]
    output_paths: tuple[Path, ...]


def load_config(path: Path) -> NeuralEgraphExtractorConfig:
    """Load a Goal 5.4 YAML config."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return NeuralEgraphExtractorConfig(
        **{key: _coerce_config_value(key, value) for key, value in raw.items()}
    )


def run_goal5_neural_egraph_extractor(
    config: NeuralEgraphExtractorConfig,
) -> NeuralEgraphExtractorResult:
    """Run candidate generation, neural cost training, and extraction evaluation."""
    started_at = time.time()
    split_config = SplitConfig(
        seed=config.seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    candidate_config = CandidateGenerationConfig(
        seed=config.seed,
        count=config.count,
        symbol_names=config.symbol_names,
        input_jsonl_path=config.input_jsonl_path,
        goal3_metrics_csv_path=config.goal3_metrics_csv_path,
        run_modes=config.run_modes,
        max_iterations=config.max_iterations,
        max_enodes=config.max_enodes,
        max_eclasses=config.max_eclasses,
        saturation_timeout_seconds=config.saturation_timeout_seconds,
        row_timeout_seconds=config.row_timeout_seconds,
        beam_size=config.beam_size,
        max_candidate_depth=config.max_candidate_depth,
        max_candidates_evaluated=config.max_candidates_evaluated,
    )
    records = load_or_build_candidate_records(
        candidate_config,
        split_config=split_config,
        output_jsonl_path=config.candidate_dataset_jsonl_path,
        reuse_existing=config.reuse_existing_candidate_dataset,
    )
    print(f"Loaded candidate records: {len(records)}", flush=True)
    train_result = train_neural_cost_model(
        records,
        config=NeuralCostModelConfig(
            seed=config.seed,
            hidden_size=config.hidden_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            max_pairs_per_group=config.max_pairs_per_group,
            l2_penalty=config.l2_penalty,
        ),
    )
    print("Trained neural cost model", flush=True)
    metric_rows = evaluate_neural_egraph_extractor(records, model=train_result.model)
    print(f"Evaluated candidate groups: {len(metric_rows)}", flush=True)
    write_metrics_csv(metric_rows, config.metrics_csv_path)
    completed_at = time.time()
    summary = build_summary(
        metric_rows,
        config=config,
        candidate_dataset_summary=summarize_candidate_dataset(records),
        trained_final_reasoning_gnn=train_result.model.trained_final_reasoning_gnn,
        started_at=started_at,
        completed_at=completed_at,
    )
    write_json(config.summary_json_path, summary)
    train_log = build_train_log(
        config,
        candidate_dataset_summary=summarize_candidate_dataset(records),
        model_payload=train_result.model.model_dump(mode="json"),
        train_payload=train_result.train_log,
        metric_rows=metric_rows,
        started_at=started_at,
        completed_at=completed_at,
    )
    write_json(config.train_log_json_path, train_log)
    return NeuralEgraphExtractorResult(
        summary=summary,
        output_paths=(
            config.candidate_dataset_jsonl_path,
            config.metrics_csv_path,
            config.summary_json_path,
            config.train_log_json_path,
        ),
    )


def build_summary(
    metric_rows: Sequence[NeuralEgraphMetricRow],
    *,
    config: NeuralEgraphExtractorConfig,
    candidate_dataset_summary: dict[str, object],
    trained_final_reasoning_gnn: bool,
    started_at: float,
    completed_at: float,
) -> dict[str, object]:
    """Build the final Goal 5.4 summary artifact."""
    summary = build_neural_egraph_summary(
        metric_rows,
        trained_final_reasoning_gnn=trained_final_reasoning_gnn,
    )
    summary["config"] = config_to_json_dict(config)
    summary["candidate_dataset"] = candidate_dataset_summary
    summary["heldout_test"] = summarize_neural_rows(
        [row for row in metric_rows if row.split == "test"]
    )
    summary["validation_split"] = summarize_neural_rows(
        [row for row in metric_rows if row.split == "validation"]
    )
    summary["elapsed_seconds"] = completed_at - started_at
    summary["completed_at_unix"] = completed_at
    summary["run_metadata"] = build_run_metadata(
        config=config_to_json_dict(config),
        started_at=started_at,
        completed_at=completed_at,
    )
    return summary


def build_train_log(
    config: NeuralEgraphExtractorConfig,
    *,
    candidate_dataset_summary: dict[str, object],
    model_payload: dict[str, object],
    train_payload: dict[str, MetadataValue],
    metric_rows: Sequence[NeuralEgraphMetricRow],
    started_at: float,
    completed_at: float,
) -> dict[str, object]:
    """Build a train log with model, split, and validation evidence."""
    return {
        "config": config_to_json_dict(config),
        "candidate_dataset": candidate_dataset_summary,
        "model": model_payload,
        "training": train_payload,
        "validation_summary": summarize_neural_rows(
            [row for row in metric_rows if row.split == "validation"]
        ),
        "test_summary": summarize_neural_rows([row for row in metric_rows if row.split == "test"]),
        "test_set_used_for_training": False,
        "trained_final_reasoning_gnn": False,
        "run_metadata": build_run_metadata(
            config=config_to_json_dict(config),
            started_at=started_at,
            completed_at=completed_at,
        ),
    }


def write_metrics_csv(rows: Sequence[NeuralEgraphMetricRow], path: Path) -> None:
    """Write per-group neural e-graph metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=NEURAL_EGRAPH_METRICS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a deterministic JSON artifact."""
    write_json_object(path, payload)


def config_to_json_dict(config: NeuralEgraphExtractorConfig) -> dict[str, object]:
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
        "run_modes": list(config.run_modes),
        "max_iterations": config.max_iterations,
        "max_enodes": config.max_enodes,
        "max_eclasses": config.max_eclasses,
        "saturation_timeout_seconds": config.saturation_timeout_seconds,
        "row_timeout_seconds": config.row_timeout_seconds,
        "beam_size": config.beam_size,
        "max_candidate_depth": config.max_candidate_depth,
        "max_candidates_evaluated": config.max_candidates_evaluated,
        "hidden_size": config.hidden_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "max_pairs_per_group": config.max_pairs_per_group,
        "l2_penalty": config.l2_penalty,
        "reuse_existing_candidate_dataset": config.reuse_existing_candidate_dataset,
        "input_jsonl_path": str(config.input_jsonl_path),
        "goal3_metrics_csv_path": str(config.goal3_metrics_csv_path),
        "candidate_dataset_jsonl_path": str(config.candidate_dataset_jsonl_path),
        "metrics_csv_path": str(config.metrics_csv_path),
        "summary_json_path": str(config.summary_json_path),
        "train_log_json_path": str(config.train_log_json_path),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/neural_egraph_extractor_v1.yaml"),
        help="Path to a YAML Goal 5.4 neural e-graph extractor config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Goal 5.4 neural e-graph extraction."""
    args = build_parser().parse_args(argv)
    result = run_goal5_neural_egraph_extractor(load_config(args.config))
    print(f"Processed groups: {result.summary['processed_group_count']}")
    print(f"Succeeded: {result.summary['success_count']}")
    for path in result.output_paths:
        print(f"Wrote: {path}")
    return 0


def _coerce_config_value(key: str, value: object) -> object:
    path_keys = {
        "input_jsonl_path",
        "goal3_metrics_csv_path",
        "candidate_dataset_jsonl_path",
        "metrics_csv_path",
        "summary_json_path",
        "train_log_json_path",
    }
    tuple_keys = {"operator_set", "symbol_names", "run_modes"}
    if key in path_keys and isinstance(value, str):
        return Path(value)
    if key in tuple_keys and isinstance(value, list):
        return tuple(value)
    return value


def _assert_no_outputs_v0(paths: Sequence[Path]) -> None:
    bad_paths = [path for path in paths if "outputs/v0" in path.as_posix()]
    if bad_paths:
        raise ValueError(f"Goal 5.4 neural extractor must not use outputs/v0: {bad_paths}")


if __name__ == "__main__":
    raise SystemExit(main())
