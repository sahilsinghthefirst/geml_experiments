"""Goal 3.4 stratified analysis for DAG compression metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator

from geml.experiments.expansion_study import compute_alpha_threshold
from geml.experiments.stratified_expansion import (
    AST_NODE_BUCKETS,
    BOOLEAN_FEATURES,
    OperatorFeatures,
    bucket_ast_nodes,
    count_operator_features,
    dominant_operator_family,
    operator_signature,
    parse_bool,
    parse_float,
    parse_int,
    percentile,
)

DAG_BOOLEAN_FEATURES = ("contains_Add", "contains_Mul", "contains_log", "contains_exp")
DAG_GROUP_SUMMARY_FIELDS = [
    "count",
    "median_tree_alpha",
    "median_dag_alpha_vs_ast_tree",
    "median_dag_alpha_vs_ast_dag",
    "median_eml_dag_compression",
    "p90_eml_dag_compression",
    "percent_below_threshold_after_dag",
    "percent_below_threshold_dag_vs_ast_tree",
    "percent_below_threshold_dag_vs_ast_dag",
    "median_improvement",
]
DAG_THRESHOLD_SUMMARY_FIELDS = [
    "scenario",
    "k",
    "l",
    "alpha_threshold",
    "row_count",
    "percent_below_tree_alpha",
    "percent_below_dag_alpha_vs_ast_tree",
    "percent_below_dag_alpha_vs_ast_dag",
]


class DagThresholdScenario(BaseModel):
    """Alpha-threshold scenario parameters for DAG metrics."""

    name: str
    k: int = Field(gt=0)
    ell: int = Field(gt=0, alias="l")


class StratifiedDagCompressionConfig(BaseModel):
    """Configuration for the Goal 3.4 stratified DAG analysis."""

    dag_metrics_csv_path: Path = Path("outputs/v0/dag_compression_metrics.csv")
    dag_summary_json_path: Path = Path("outputs/v0/dag_compression_summary.json")
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
    threshold_scenarios: tuple[DagThresholdScenario, ...] = Field(
        default_factory=lambda: (
            DagThresholdScenario(name="current_grammar", k=4, l=3),
            DagThresholdScenario(name="generous_operator_vocab", k=20, l=3),
            DagThresholdScenario(name="larger_operator_vocab", k=50, l=3),
        )
    )

    @model_validator(mode="after")
    def validate_output_paths(self) -> Self:
        output_paths = {
            self.dag_alpha_threshold_summary_csv_path,
            self.dag_alpha_threshold_summary_json_path,
            self.dag_alpha_by_ast_size_bucket_csv_path,
            self.dag_alpha_by_ast_depth_csv_path,
            self.dag_alpha_by_operator_family_csv_path,
            self.dag_alpha_by_operator_signature_csv_path,
            self.dag_alpha_by_boolean_features_csv_path,
        }
        if self.dag_metrics_csv_path in output_paths or self.dag_summary_json_path in output_paths:
            raise ValueError("input DAG metric paths must differ from output paths")
        return self


@dataclass(frozen=True)
class StratifiedDagCompressionRow:
    """One DAG metric row enriched with operator and bucket features."""

    expression: str
    srepr: str
    ast_tree_node_count: int
    ast_dag_node_count: int
    ast_tree_depth: int
    ast_dag_depth: int
    eml_tree_node_count: int
    eml_dag_node_count: int
    tree_alpha: float
    dag_alpha_vs_ast_tree: float
    dag_alpha_vs_ast_dag: float
    eml_dag_compression: float
    features: OperatorFeatures
    ast_nodes_bucket: str
    operator_signature: str
    dominant_operator_family: str

    @property
    def improvement(self) -> float:
        """Tree alpha divided by DAG alpha versus AST tree."""
        return self.tree_alpha / self.dag_alpha_vs_ast_tree


@dataclass(frozen=True)
class StratifiedDagCompressionResult:
    """Result metadata from a Goal 3.4 analysis export run."""

    input_count: int
    summary_processed_count: int
    threshold_summary_count: int
    output_paths: tuple[Path, ...]


def run_stratified_dag_compression_analysis(
    config: StratifiedDagCompressionConfig,
) -> StratifiedDagCompressionResult:
    """Load Goal 3.3 metrics and write Goal 3.4 stratified summaries."""
    dag_summary = load_dag_summary(config.dag_summary_json_path)
    rows = load_stratified_dag_rows(config.dag_metrics_csv_path)
    threshold_summaries = build_threshold_summaries(rows, config.threshold_scenarios)

    write_json(config.dag_alpha_threshold_summary_json_path, threshold_summaries)
    write_threshold_summary_csv(threshold_summaries, config.dag_alpha_threshold_summary_csv_path)
    write_group_summary_csv(
        group_by_ast_size_bucket(rows),
        config.dag_alpha_by_ast_size_bucket_csv_path,
        group_fields=["ast_nodes_bucket"],
    )
    write_group_summary_csv(
        group_by_ast_depth(rows),
        config.dag_alpha_by_ast_depth_csv_path,
        group_fields=["ast_tree_depth"],
    )
    write_group_summary_csv(
        group_by_operator_family(rows),
        config.dag_alpha_by_operator_family_csv_path,
        group_fields=["dominant_operator_family"],
    )
    write_group_summary_csv(
        group_by_operator_signature(rows),
        config.dag_alpha_by_operator_signature_csv_path,
        group_fields=["operator_signature"],
    )
    write_group_summary_csv(
        group_by_boolean_features(rows),
        config.dag_alpha_by_boolean_features_csv_path,
        group_fields=["feature", "value"],
    )

    return StratifiedDagCompressionResult(
        input_count=len(rows),
        summary_processed_count=int(dag_summary["processed_count"]),
        threshold_summary_count=len(threshold_summaries),
        output_paths=(
            config.dag_alpha_threshold_summary_csv_path,
            config.dag_alpha_threshold_summary_json_path,
            config.dag_alpha_by_ast_size_bucket_csv_path,
            config.dag_alpha_by_ast_depth_csv_path,
            config.dag_alpha_by_operator_family_csv_path,
            config.dag_alpha_by_operator_signature_csv_path,
            config.dag_alpha_by_boolean_features_csv_path,
        ),
    )


def load_dag_summary(path: Path) -> dict[str, object]:
    """Load Goal 3.3 DAG compression summary JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def load_stratified_dag_rows(path: Path) -> list[StratifiedDagCompressionRow]:
    """Load DAG compression metric CSV rows and enrich them with features."""
    rows: list[StratifiedDagCompressionRow] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            if parse_bool(raw_row["supported"]) and parse_bool(raw_row["pure_eml_valid"]):
                rows.append(build_stratified_dag_row(raw_row))
    if not rows:
        raise ValueError(f"no supported pure EML DAG rows found in {path}")
    return rows


def build_stratified_dag_row(raw_row: dict[str, str]) -> StratifiedDagCompressionRow:
    """Build one enriched Goal 3.4 row from a raw DAG metrics CSV row."""
    ast_tree_node_count = parse_int(raw_row["ast_tree_node_count"])
    features = count_operator_features(raw_row["srepr"])
    return StratifiedDagCompressionRow(
        expression=raw_row["expression"],
        srepr=raw_row["srepr"],
        ast_tree_node_count=ast_tree_node_count,
        ast_dag_node_count=parse_int(raw_row["ast_dag_node_count"]),
        ast_tree_depth=parse_int(raw_row["ast_tree_depth"]),
        ast_dag_depth=parse_int(raw_row["ast_dag_depth"]),
        eml_tree_node_count=parse_int(raw_row["eml_tree_node_count"]),
        eml_dag_node_count=parse_int(raw_row["eml_dag_node_count"]),
        tree_alpha=parse_float(raw_row["tree_alpha"]),
        dag_alpha_vs_ast_tree=parse_float(raw_row["dag_alpha_vs_ast_tree"]),
        dag_alpha_vs_ast_dag=parse_float(raw_row["dag_alpha_vs_ast_dag"]),
        eml_dag_compression=parse_float(raw_row["eml_dag_compression"]),
        features=features,
        ast_nodes_bucket=bucket_ast_nodes(ast_tree_node_count),
        operator_signature=operator_signature(features),
        dominant_operator_family=dominant_operator_family(features),
    )


def build_threshold_summaries(
    rows: Sequence[StratifiedDagCompressionRow],
    scenarios: Sequence[DagThresholdScenario],
) -> list[dict[str, object]]:
    """Build threshold summaries for tree and DAG alpha metrics."""
    summaries: list[dict[str, object]] = []
    for scenario in scenarios:
        threshold = compute_alpha_threshold(scenario.k, scenario.ell)
        summaries.append(
            {
                "scenario": scenario.name,
                "k": scenario.k,
                "l": scenario.ell,
                "alpha_threshold": threshold,
                "row_count": len(rows),
                "percent_below_tree_alpha": percent_below(
                    [row.tree_alpha for row in rows],
                    threshold=threshold,
                ),
                "percent_below_dag_alpha_vs_ast_tree": percent_below(
                    [row.dag_alpha_vs_ast_tree for row in rows],
                    threshold=threshold,
                ),
                "percent_below_dag_alpha_vs_ast_dag": percent_below(
                    [row.dag_alpha_vs_ast_dag for row in rows],
                    threshold=threshold,
                ),
            }
        )
    return summaries


def percent_below(values: Sequence[float], *, threshold: float) -> float:
    """Return the percentage of values strictly below the threshold."""
    if not values:
        return 0.0
    return 100 * sum(1 for value in values if value < threshold) / len(values)


def median_improvement(rows: Sequence[StratifiedDagCompressionRow]) -> float | None:
    """Return median tree-alpha to DAG-alpha improvement."""
    if not rows:
        return None
    return statistics.median(sorted(row.improvement for row in rows))


def group_by_ast_size_bucket(
    rows: Sequence[StratifiedDagCompressionRow],
) -> list[dict[str, object]]:
    """Group DAG alpha statistics by AST node-count bucket."""
    return build_group_summaries(
        rows,
        group_key=lambda row: row.ast_nodes_bucket,
        group_field="ast_nodes_bucket",
        ordered_keys=[label for _, _, label in AST_NODE_BUCKETS],
    )


def group_by_ast_depth(rows: Sequence[StratifiedDagCompressionRow]) -> list[dict[str, object]]:
    """Group DAG alpha statistics by exact AST tree depth."""
    return build_group_summaries(
        rows,
        group_key=lambda row: row.ast_tree_depth,
        group_field="ast_tree_depth",
        sort_key=lambda key: int(key),
    )


def group_by_operator_family(
    rows: Sequence[StratifiedDagCompressionRow],
) -> list[dict[str, object]]:
    """Group DAG alpha statistics by dominant operator family."""
    return build_group_summaries(
        rows,
        group_key=lambda row: row.dominant_operator_family,
        group_field="dominant_operator_family",
        sort_key=str,
    )


def group_by_operator_signature(
    rows: Sequence[StratifiedDagCompressionRow],
) -> list[dict[str, object]]:
    """Group DAG alpha statistics by exact operator signature."""
    return build_group_summaries(
        rows,
        group_key=lambda row: row.operator_signature,
        group_field="operator_signature",
        sort_key=str,
    )


def group_by_boolean_features(
    rows: Sequence[StratifiedDagCompressionRow],
) -> list[dict[str, object]]:
    """Group DAG alpha statistics by selected boolean operator features."""
    summaries: list[dict[str, object]] = []
    for feature in DAG_BOOLEAN_FEATURES:
        if feature not in BOOLEAN_FEATURES:
            raise ValueError(f"unsupported boolean feature {feature!r}")
        for value in (False, True):
            group_rows = [row for row in rows if bool(getattr(row.features, feature)) is value]
            summaries.append(
                {
                    "feature": feature,
                    "value": value,
                    **summarize_dag_group(group_rows),
                }
            )
    return summaries


def build_group_summaries(
    rows: Sequence[StratifiedDagCompressionRow],
    *,
    group_key: Callable[[StratifiedDagCompressionRow], object],
    group_field: str,
    sort_key: Callable[[object], object] | None = None,
    ordered_keys: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    """Build group summary rows for one grouping key."""
    grouped: dict[object, list[StratifiedDagCompressionRow]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)

    if ordered_keys is not None:
        keys = [key for key in ordered_keys if key in grouped]
    else:
        key_func = sort_key if sort_key is not None else str
        keys = sorted(grouped, key=key_func)

    return [{group_field: key, **summarize_dag_group(grouped[key])} for key in keys]


def summarize_dag_group(rows: Sequence[StratifiedDagCompressionRow]) -> dict[str, object]:
    """Compute required DAG alpha statistics for one group."""
    if not rows:
        return {
            "count": 0,
            "median_tree_alpha": None,
            "median_dag_alpha_vs_ast_tree": None,
            "median_dag_alpha_vs_ast_dag": None,
            "median_eml_dag_compression": None,
            "p90_eml_dag_compression": None,
            "percent_below_threshold_after_dag": None,
            "percent_below_threshold_dag_vs_ast_tree": None,
            "percent_below_threshold_dag_vs_ast_dag": None,
            "median_improvement": None,
        }

    current_threshold = compute_alpha_threshold(4, 3)
    eml_compressions = sorted(row.eml_dag_compression for row in rows)
    dag_alpha_vs_ast_tree_values = [row.dag_alpha_vs_ast_tree for row in rows]
    dag_alpha_vs_ast_dag_values = [row.dag_alpha_vs_ast_dag for row in rows]
    percent_after_dag = percent_below(
        dag_alpha_vs_ast_tree_values,
        threshold=current_threshold,
    )
    return {
        "count": len(rows),
        "median_tree_alpha": statistics.median(sorted(row.tree_alpha for row in rows)),
        "median_dag_alpha_vs_ast_tree": statistics.median(sorted(dag_alpha_vs_ast_tree_values)),
        "median_dag_alpha_vs_ast_dag": statistics.median(sorted(dag_alpha_vs_ast_dag_values)),
        "median_eml_dag_compression": statistics.median(eml_compressions),
        "p90_eml_dag_compression": percentile(eml_compressions, 0.9),
        "percent_below_threshold_after_dag": percent_after_dag,
        "percent_below_threshold_dag_vs_ast_tree": percent_after_dag,
        "percent_below_threshold_dag_vs_ast_dag": percent_below(
            dag_alpha_vs_ast_dag_values,
            threshold=current_threshold,
        ),
        "median_improvement": median_improvement(rows),
    }


def write_json(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write JSON rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), indent=2, sort_keys=True), encoding="utf-8")


def write_threshold_summary_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Write threshold scenario summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DAG_THRESHOLD_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_group_summary_csv(
    rows: Sequence[dict[str, object]],
    path: Path,
    *,
    group_fields: Sequence[str],
) -> None:
    """Write one grouped DAG summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[*group_fields, *DAG_GROUP_SUMMARY_FIELDS],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> StratifiedDagCompressionConfig:
    """Load a YAML config that may include Goal 3.4 output paths."""
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    allowed_keys = set(StratifiedDagCompressionConfig.model_fields)
    filtered_config = {key: value for key, value in raw_config.items() if key in allowed_keys}
    return StratifiedDagCompressionConfig.model_validate(filtered_config)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dag_compression_v0.yaml"),
        help="Optional YAML config with Goal 3.4 paths.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Goal 3.4 stratified DAG compression analysis."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    result = run_stratified_dag_compression_analysis(config)

    print(f"Loaded DAG metric rows: {result.input_count}")
    print(f"Summary processed count: {result.summary_processed_count}")
    print(f"Threshold scenarios: {result.threshold_summary_count}")
    for path in result.output_paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
