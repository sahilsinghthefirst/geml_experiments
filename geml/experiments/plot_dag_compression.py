"""Goal 3.5 reproducible plots for DAG compression metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Self

import matplotlib
from matplotlib.figure import Figure
from pydantic import BaseModel, model_validator

from geml.experiments.stratified_expansion import parse_bool, parse_float, parse_int

GOAL3_PLOT_FILENAMES = (
    "tree_alpha_vs_dag_alpha.png",
    "eml_tree_nodes_vs_eml_dag_nodes.png",
    "eml_dag_compression_histogram.png",
    "dag_alpha_vs_ast_tree_histogram.png",
    "dag_alpha_vs_ast_dag_histogram.png",
    "median_dag_alpha_by_operator_family.png",
    "median_eml_dag_compression_by_operator_family.png",
    "percent_below_threshold_tree_vs_dag.png",
    "dag_improvement_by_ast_size_bucket.png",
)


class DagCompressionPlotConfig(BaseModel):
    """Configuration for Goal 3.5 DAG compression plotting."""

    dag_metrics_csv_path: Path = Path("outputs/v0/dag_compression_metrics.csv")
    dag_threshold_summary_json_path: Path = Path("outputs/v0/dag_alpha_threshold_summary.json")
    dag_operator_family_csv_path: Path = Path("outputs/v0/dag_alpha_by_operator_family.csv")
    dag_ast_size_bucket_csv_path: Path = Path("outputs/v0/dag_alpha_by_ast_size_bucket.csv")
    plots_dir: Path = Path("outputs/v0/plots_goal3")

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if self.plots_dir in {
            self.dag_metrics_csv_path,
            self.dag_threshold_summary_json_path,
            self.dag_operator_family_csv_path,
            self.dag_ast_size_bucket_csv_path,
        }:
            raise ValueError("plots_dir must differ from input artifact paths")
        return self


@dataclass(frozen=True)
class DagCompressionPlotRow:
    """Per-expression metric fields needed by Goal 3.5 plots."""

    ast_tree_node_count: int
    ast_dag_node_count: int
    eml_tree_node_count: int
    eml_dag_node_count: int
    tree_alpha: float
    dag_alpha_vs_ast_tree: float
    dag_alpha_vs_ast_dag: float
    eml_dag_compression: float


@dataclass(frozen=True)
class DagCompressionPlotResult:
    """Result metadata from a Goal 3.5 plotting run."""

    dag_metric_count: int
    threshold_summary_count: int
    operator_family_count: int
    ast_size_bucket_count: int
    plot_paths: tuple[Path, ...]


def run_dag_compression_plots(
    config: DagCompressionPlotConfig,
) -> DagCompressionPlotResult:
    """Load saved Goal 3 artifacts and write DAG compression plots."""
    metric_rows = load_dag_metric_rows(config.dag_metrics_csv_path)
    threshold_rows = load_json_rows(config.dag_threshold_summary_json_path)
    operator_family_rows = load_csv_rows(config.dag_operator_family_csv_path)
    ast_size_bucket_rows = load_csv_rows(config.dag_ast_size_bucket_csv_path)

    config.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = (
        plot_tree_alpha_vs_dag_alpha(
            metric_rows,
            config.plots_dir / "tree_alpha_vs_dag_alpha.png",
        ),
        plot_eml_tree_nodes_vs_eml_dag_nodes(
            metric_rows,
            config.plots_dir / "eml_tree_nodes_vs_eml_dag_nodes.png",
        ),
        plot_eml_dag_compression_histogram(
            metric_rows,
            config.plots_dir / "eml_dag_compression_histogram.png",
        ),
        plot_dag_alpha_vs_ast_tree_histogram(
            metric_rows,
            config.plots_dir / "dag_alpha_vs_ast_tree_histogram.png",
        ),
        plot_dag_alpha_vs_ast_dag_histogram(
            metric_rows,
            config.plots_dir / "dag_alpha_vs_ast_dag_histogram.png",
        ),
        plot_median_dag_alpha_by_operator_family(
            operator_family_rows,
            config.plots_dir / "median_dag_alpha_by_operator_family.png",
        ),
        plot_median_eml_dag_compression_by_operator_family(
            operator_family_rows,
            config.plots_dir / "median_eml_dag_compression_by_operator_family.png",
        ),
        plot_percent_below_threshold_tree_vs_dag(
            threshold_rows,
            config.plots_dir / "percent_below_threshold_tree_vs_dag.png",
        ),
        plot_dag_improvement_by_ast_size_bucket(
            ast_size_bucket_rows,
            config.plots_dir / "dag_improvement_by_ast_size_bucket.png",
        ),
    )

    return DagCompressionPlotResult(
        dag_metric_count=len(metric_rows),
        threshold_summary_count=len(threshold_rows),
        operator_family_count=len(operator_family_rows),
        ast_size_bucket_count=len(ast_size_bucket_rows),
        plot_paths=plot_paths,
    )


def load_dag_metric_rows(path: Path) -> list[DagCompressionPlotRow]:
    """Load supported pure EML DAG metric rows from saved CSV."""
    rows: list[DagCompressionPlotRow] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            if parse_bool(raw_row["supported"]) and parse_bool(raw_row["pure_eml_valid"]):
                rows.append(
                    DagCompressionPlotRow(
                        ast_tree_node_count=parse_int(raw_row["ast_tree_node_count"]),
                        ast_dag_node_count=parse_int(raw_row["ast_dag_node_count"]),
                        eml_tree_node_count=parse_int(raw_row["eml_tree_node_count"]),
                        eml_dag_node_count=parse_int(raw_row["eml_dag_node_count"]),
                        tree_alpha=parse_float(raw_row["tree_alpha"]),
                        dag_alpha_vs_ast_tree=parse_float(raw_row["dag_alpha_vs_ast_tree"]),
                        dag_alpha_vs_ast_dag=parse_float(raw_row["dag_alpha_vs_ast_dag"]),
                        eml_dag_compression=parse_float(raw_row["eml_dag_compression"]),
                    )
                )
    if not rows:
        raise ValueError(f"no supported pure EML DAG rows found in {path}")
    return rows


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    """Load a saved JSON list artifact."""
    with path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list in {path}")
    if not data:
        raise ValueError(f"no rows found in {path}")
    return data


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a saved CSV artifact."""
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def plot_tree_alpha_vs_dag_alpha(
    rows: Sequence[DagCompressionPlotRow],
    path: Path,
) -> Path:
    """Plot raw tree alpha against DAG alpha versus AST tree size."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    tree_values = [row.tree_alpha for row in rows]
    dag_values = [row.dag_alpha_vs_ast_tree for row in rows]
    ax.scatter(tree_values, dag_values, s=8, alpha=0.35)
    lower = min(min(tree_values), min(dag_values))
    upper = max(max(tree_values), max(dag_values))
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1)
    ax.set_title("Tree Alpha vs DAG Alpha")
    ax.set_xlabel("Tree alpha = EML tree nodes / AST tree nodes")
    ax.set_ylabel("DAG alpha = EML DAG nodes / AST tree nodes")
    return save_figure(fig, path)


def plot_eml_tree_nodes_vs_eml_dag_nodes(
    rows: Sequence[DagCompressionPlotRow],
    path: Path,
) -> Path:
    """Plot EML tree node count against EML DAG unique node count."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    tree_nodes = [row.eml_tree_node_count for row in rows]
    dag_nodes = [row.eml_dag_node_count for row in rows]
    ax.scatter(tree_nodes, dag_nodes, s=8, alpha=0.35)
    lower = min(min(tree_nodes), min(dag_nodes))
    upper = max(max(tree_nodes), max(dag_nodes))
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1)
    ax.set_title("EML Tree Nodes vs EML DAG Nodes")
    ax.set_xlabel("EML tree nodes")
    ax.set_ylabel("EML DAG unique nodes")
    return save_figure(fig, path)


def plot_eml_dag_compression_histogram(
    rows: Sequence[DagCompressionPlotRow],
    path: Path,
) -> Path:
    """Plot the EML DAG compression-ratio distribution."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([row.eml_dag_compression for row in rows], bins=40)
    ax.set_title("EML DAG Compression Distribution")
    ax.set_xlabel("EML tree nodes / EML DAG nodes")
    ax.set_ylabel("Expression count")
    return save_figure(fig, path)


def plot_dag_alpha_vs_ast_tree_histogram(
    rows: Sequence[DagCompressionPlotRow],
    path: Path,
) -> Path:
    """Plot DAG alpha versus AST tree distribution."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([row.dag_alpha_vs_ast_tree for row in rows], bins=40)
    ax.set_title("DAG Alpha vs AST Tree Distribution")
    ax.set_xlabel("EML DAG nodes / AST tree nodes")
    ax.set_ylabel("Expression count")
    return save_figure(fig, path)


def plot_dag_alpha_vs_ast_dag_histogram(
    rows: Sequence[DagCompressionPlotRow],
    path: Path,
) -> Path:
    """Plot DAG alpha versus AST DAG distribution."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([row.dag_alpha_vs_ast_dag for row in rows], bins=40)
    ax.set_title("DAG Alpha vs AST DAG Distribution")
    ax.set_xlabel("EML DAG nodes / AST DAG nodes")
    ax.set_ylabel("Expression count")
    return save_figure(fig, path)


def plot_median_dag_alpha_by_operator_family(
    rows: Sequence[dict[str, str]],
    path: Path,
) -> Path:
    """Plot median DAG alpha by dominant operator family."""
    sorted_rows = sorted(
        rows,
        key=lambda row: parse_float(row["median_dag_alpha_vs_ast_tree"]),
        reverse=True,
    )
    labels = [row["dominant_operator_family"] for row in sorted_rows]
    values = [parse_float(row["median_dag_alpha_vs_ast_tree"]) for row in sorted_rows]

    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values)
    ax.set_title("Median DAG Alpha by Operator Family")
    ax.set_xlabel("Dominant operator family")
    ax.set_ylabel("Median EML DAG nodes / AST tree nodes")
    ax.tick_params(axis="x", rotation=60)
    return save_figure(fig, path)


def plot_median_eml_dag_compression_by_operator_family(
    rows: Sequence[dict[str, str]],
    path: Path,
) -> Path:
    """Plot median EML DAG compression by dominant operator family."""
    sorted_rows = sorted(
        rows,
        key=lambda row: parse_float(row["median_eml_dag_compression"]),
        reverse=True,
    )
    labels = [row["dominant_operator_family"] for row in sorted_rows]
    values = [parse_float(row["median_eml_dag_compression"]) for row in sorted_rows]

    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values)
    ax.set_title("Median EML DAG Compression by Operator Family")
    ax.set_xlabel("Dominant operator family")
    ax.set_ylabel("Median EML tree nodes / EML DAG nodes")
    ax.tick_params(axis="x", rotation=60)
    return save_figure(fig, path)


def plot_percent_below_threshold_tree_vs_dag(
    rows: Sequence[dict[str, Any]],
    path: Path,
) -> Path:
    """Plot threshold-pass percentages for tree alpha and DAG alpha variants."""
    labels = [str(row["scenario"]) for row in rows]
    x_positions = list(range(len(rows)))
    width = 0.25
    series = (
        ("percent_below_tree_alpha", "Tree alpha", -width),
        ("percent_below_dag_alpha_vs_ast_tree", "DAG alpha vs AST tree", 0.0),
        ("percent_below_dag_alpha_vs_ast_dag", "DAG alpha vs AST DAG", width),
    )

    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 6))
    for field, label, offset in series:
        ax.bar(
            [position + offset for position in x_positions],
            [float(row[field]) for row in rows],
            width=width,
            label=label,
        )
    ax.set_title("Percent Below Threshold: Tree vs DAG")
    ax.set_xlabel("Threshold scenario")
    ax.set_ylabel("Percent below threshold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    return save_figure(fig, path)


def plot_dag_improvement_by_ast_size_bucket(
    rows: Sequence[dict[str, str]],
    path: Path,
) -> Path:
    """Plot median tree-alpha to DAG-alpha improvement by AST node bucket."""
    labels = [row["ast_nodes_bucket"] for row in rows]
    values = [parse_float(row["median_improvement"]) for row in rows]

    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_title("DAG Improvement by AST Size Bucket")
    ax.set_xlabel("AST node-count bucket")
    ax.set_ylabel("Median tree alpha / DAG alpha")
    return save_figure(fig, path)


def save_figure(fig: Figure, path: Path) -> Path:
    """Save and close a Matplotlib figure."""
    plt = get_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def get_pyplot() -> ModuleType:
    """Import pyplot after selecting the non-interactive backend."""
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as pyplot

    return pyplot


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dag-metrics-csv",
        type=Path,
        default=None,
        help="Saved Goal 3.3 DAG metric CSV path.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for generated Goal 3 PNG plots.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 3.5 DAG compression plot export."""
    args = build_parser().parse_args(argv)
    config = DagCompressionPlotConfig()
    if args.dag_metrics_csv is not None:
        config.dag_metrics_csv_path = args.dag_metrics_csv
    if args.plots_dir is not None:
        config.plots_dir = args.plots_dir

    result = run_dag_compression_plots(config)
    print(f"Loaded DAG metric rows: {result.dag_metric_count}")
    print(f"Loaded threshold summary rows: {result.threshold_summary_count}")
    print(f"Loaded operator family rows: {result.operator_family_count}")
    print(f"Loaded AST size bucket rows: {result.ast_size_bucket_count}")
    for path in result.plot_paths:
        print(f"Wrote plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
