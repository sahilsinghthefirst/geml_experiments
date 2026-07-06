"""Goal 2.4 reproducible plots for official pure EML expansion."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Self

import matplotlib
from matplotlib.figure import Figure
from pydantic import BaseModel, model_validator

PLOT_FILENAMES = (
    "alpha_histogram.png",
    "alpha_histogram_log_scale.png",
    "ast_nodes_vs_eml_nodes.png",
    "ast_depth_vs_alpha.png",
    "eml_depth_vs_alpha.png",
    "alpha_by_ast_depth.png",
    "alpha_by_operator_family.png",
    "percent_below_threshold_by_ast_depth.png",
    "percent_below_threshold_by_operator_family.png",
    "eml_nodes_by_ast_nodes.png",
)
TOP_TABLE_FIELDS = [
    "rank",
    "index",
    "expression",
    "srepr",
    "ast_node_count",
    "ast_depth",
    "ast_operator_count",
    "ast_leaf_count",
    "eml_node_count",
    "eml_depth",
    "eml_operator_count",
    "eml_leaf_count",
    "alpha",
    "alpha_threshold",
    "below_threshold",
]


class ExpansionPlotConfig(BaseModel):
    """Configuration for Goal 2.4 expansion-factor plotting."""

    raw_metrics_csv_path: Path = Path("outputs/v0/expansion_raw_metrics.csv")
    alpha_summary_json_path: Path = Path("outputs/v0/expansion_alpha_summary.json")
    alpha_by_ast_depth_csv_path: Path = Path("outputs/v0/alpha_by_ast_depth.csv")
    alpha_by_operator_family_csv_path: Path = Path("outputs/v0/alpha_by_operator_family.csv")
    plots_dir: Path = Path("outputs/v0/plots")
    top_alpha_csv_path: Path = Path("outputs/v0/top_20_alpha_expressions.csv")
    top_eml_node_csv_path: Path = Path("outputs/v0/top_20_eml_node_expressions.csv")
    top_eml_depth_csv_path: Path = Path("outputs/v0/top_20_eml_depth_expressions.csv")

    @model_validator(mode="after")
    def validate_output_paths(self) -> Self:
        if self.plots_dir == self.raw_metrics_csv_path:
            raise ValueError("plots_dir must not equal raw_metrics_csv_path")
        return self


@dataclass(frozen=True)
class RawMetricRow:
    """Raw metric row fields needed by Goal 2.4 plots."""

    index: int
    expression: str
    srepr: str
    ast_node_count: int
    ast_depth: int
    ast_operator_count: int
    ast_leaf_count: int
    eml_node_count: int
    eml_depth: int
    eml_operator_count: int
    eml_leaf_count: int
    alpha: float
    alpha_threshold: float
    below_threshold: bool


@dataclass(frozen=True)
class ExpansionPlotResult:
    """Result metadata from a Goal 2.4 plotting run."""

    raw_metric_count: int
    alpha_summary_count: int
    plot_paths: tuple[Path, ...]
    table_paths: tuple[Path, ...]


def run_expansion_plots(config: ExpansionPlotConfig) -> ExpansionPlotResult:
    """Read saved Goal 2 CSV/JSON outputs and write reproducible plots/tables."""
    raw_rows = load_raw_metric_rows(config.raw_metrics_csv_path)
    alpha_summary = load_alpha_summary(config.alpha_summary_json_path)
    ast_depth_rows = load_group_rows(config.alpha_by_ast_depth_csv_path)
    operator_family_rows = load_group_rows(config.alpha_by_operator_family_csv_path)

    config.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = (
        plot_alpha_histogram(raw_rows, config.plots_dir / "alpha_histogram.png"),
        plot_alpha_histogram_log_scale(
            raw_rows,
            config.plots_dir / "alpha_histogram_log_scale.png",
        ),
        plot_ast_nodes_vs_eml_nodes(raw_rows, config.plots_dir / "ast_nodes_vs_eml_nodes.png"),
        plot_ast_depth_vs_alpha(raw_rows, config.plots_dir / "ast_depth_vs_alpha.png"),
        plot_eml_depth_vs_alpha(raw_rows, config.plots_dir / "eml_depth_vs_alpha.png"),
        plot_alpha_by_ast_depth(ast_depth_rows, config.plots_dir / "alpha_by_ast_depth.png"),
        plot_alpha_by_operator_family(
            operator_family_rows,
            config.plots_dir / "alpha_by_operator_family.png",
        ),
        plot_percent_below_threshold_by_ast_depth(
            ast_depth_rows,
            config.plots_dir / "percent_below_threshold_by_ast_depth.png",
        ),
        plot_percent_below_threshold_by_operator_family(
            operator_family_rows,
            config.plots_dir / "percent_below_threshold_by_operator_family.png",
        ),
        plot_eml_nodes_by_ast_nodes(raw_rows, config.plots_dir / "eml_nodes_by_ast_nodes.png"),
    )
    table_paths = write_top_expression_tables(raw_rows, config)

    return ExpansionPlotResult(
        raw_metric_count=len(raw_rows),
        alpha_summary_count=len(alpha_summary),
        plot_paths=plot_paths,
        table_paths=table_paths,
    )


def load_raw_metric_rows(path: Path) -> list[RawMetricRow]:
    """Load supported alpha-valid raw metric rows from the saved CSV."""
    rows: list[RawMetricRow] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            if parse_bool(raw_row["supported"]) and parse_bool(raw_row["alpha_valid"]):
                rows.append(
                    RawMetricRow(
                        index=parse_int(raw_row["index"]),
                        expression=raw_row["expression"],
                        srepr=raw_row["srepr"],
                        ast_node_count=parse_int(raw_row["ast_node_count"]),
                        ast_depth=parse_int(raw_row["ast_depth"]),
                        ast_operator_count=parse_int(raw_row["ast_operator_count"]),
                        ast_leaf_count=parse_int(raw_row["ast_leaf_count"]),
                        eml_node_count=parse_int(raw_row["eml_node_count"]),
                        eml_depth=parse_int(raw_row["eml_depth"]),
                        eml_operator_count=parse_int(raw_row["eml_operator_count"]),
                        eml_leaf_count=parse_int(raw_row["eml_leaf_count"]),
                        alpha=parse_float(raw_row["alpha"]),
                        alpha_threshold=parse_float(raw_row["alpha_threshold"]),
                        below_threshold=parse_bool(raw_row["below_threshold"]),
                    )
                )
    if not rows:
        raise ValueError(f"no supported alpha-valid rows found in {path}")
    return rows


def load_alpha_summary(path: Path) -> list[dict[str, Any]]:
    """Load saved Goal 2.2 alpha summary JSON."""
    with path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}")
    return data


def load_group_rows(path: Path) -> list[dict[str, str]]:
    """Load a saved Goal 2.3 grouped CSV file."""
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def plot_alpha_histogram(rows: Sequence[RawMetricRow], path: Path) -> Path:
    """Plot alpha distribution as a count histogram."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([row.alpha for row in rows], bins=40)
    ax.set_title("Official Pure EML Alpha Distribution")
    ax.set_xlabel("Alpha = EML nodes / AST nodes")
    ax.set_ylabel("Expression count")
    return save_figure(fig, path)


def plot_alpha_histogram_log_scale(rows: Sequence[RawMetricRow], path: Path) -> Path:
    """Plot alpha distribution with log-scaled expression counts."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([row.alpha for row in rows], bins=40)
    ax.set_yscale("log")
    ax.set_title("Official Pure EML Alpha Distribution (Log Count Scale)")
    ax.set_xlabel("Alpha = EML nodes / AST nodes")
    ax.set_ylabel("Expression count (log scale)")
    return save_figure(fig, path)


def plot_ast_nodes_vs_eml_nodes(rows: Sequence[RawMetricRow], path: Path) -> Path:
    """Plot raw AST node count against pure EML node count."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        [row.ast_node_count for row in rows],
        [row.eml_node_count for row in rows],
        s=8,
        alpha=0.35,
    )
    ax.set_title("AST Nodes vs Official Pure EML Nodes")
    ax.set_xlabel("AST node count")
    ax.set_ylabel("EML node count")
    return save_figure(fig, path)


def plot_ast_depth_vs_alpha(rows: Sequence[RawMetricRow], path: Path) -> Path:
    """Plot AST depth against alpha."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        [row.ast_depth for row in rows],
        [row.alpha for row in rows],
        s=8,
        alpha=0.35,
    )
    ax.set_title("AST Depth vs Official Pure EML Alpha")
    ax.set_xlabel("AST depth")
    ax.set_ylabel("Alpha")
    return save_figure(fig, path)


def plot_eml_depth_vs_alpha(rows: Sequence[RawMetricRow], path: Path) -> Path:
    """Plot EML depth against alpha."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        [row.eml_depth for row in rows],
        [row.alpha for row in rows],
        s=8,
        alpha=0.35,
    )
    ax.set_title("EML Depth vs Official Pure EML Alpha")
    ax.set_xlabel("EML depth")
    ax.set_ylabel("Alpha")
    return save_figure(fig, path)


def plot_alpha_by_ast_depth(rows: Sequence[dict[str, str]], path: Path) -> Path:
    """Plot grouped alpha statistics by AST depth."""
    plt = get_pyplot()
    sorted_rows = sorted(rows, key=lambda row: parse_int(row["ast_depth"]))
    depths = [parse_int(row["ast_depth"]) for row in sorted_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, [parse_float(row["mean_alpha"]) for row in sorted_rows], marker="o")
    ax.plot(depths, [parse_float(row["median_alpha"]) for row in sorted_rows], marker="o")
    ax.plot(depths, [parse_float(row["p90_alpha"]) for row in sorted_rows], marker="o")
    ax.set_title("Alpha by AST Depth")
    ax.set_xlabel("AST depth")
    ax.set_ylabel("Alpha")
    ax.legend(["Mean alpha", "Median alpha", "P90 alpha"])
    return save_figure(fig, path)


def plot_alpha_by_operator_family(rows: Sequence[dict[str, str]], path: Path) -> Path:
    """Plot mean alpha by dominant operator family."""
    plt = get_pyplot()
    sorted_rows = sorted(rows, key=lambda row: parse_float(row["mean_alpha"]), reverse=True)
    labels = [row["dominant_operator_family"] for row in sorted_rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, [parse_float(row["mean_alpha"]) for row in sorted_rows])
    ax.set_title("Mean Alpha by Dominant Operator Family")
    ax.set_xlabel("Dominant operator family")
    ax.set_ylabel("Mean alpha")
    ax.tick_params(axis="x", rotation=60)
    return save_figure(fig, path)


def plot_percent_below_threshold_by_ast_depth(
    rows: Sequence[dict[str, str]],
    path: Path,
) -> Path:
    """Plot percent below the Goal 2.2 threshold by AST depth."""
    plt = get_pyplot()
    sorted_rows = sorted(rows, key=lambda row: parse_int(row["ast_depth"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [parse_int(row["ast_depth"]) for row in sorted_rows],
        [parse_float(row["percent_below_threshold"]) for row in sorted_rows],
    )
    ax.set_title("Percent Below Alpha Threshold by AST Depth")
    ax.set_xlabel("AST depth")
    ax.set_ylabel("Percent below threshold")
    return save_figure(fig, path)


def plot_percent_below_threshold_by_operator_family(
    rows: Sequence[dict[str, str]],
    path: Path,
) -> Path:
    """Plot percent below the Goal 2.2 threshold by dominant operator family."""
    plt = get_pyplot()
    sorted_rows = sorted(
        rows,
        key=lambda row: parse_float(row["percent_below_threshold"]),
        reverse=True,
    )
    labels = [row["dominant_operator_family"] for row in sorted_rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, [parse_float(row["percent_below_threshold"]) for row in sorted_rows])
    ax.set_title("Percent Below Alpha Threshold by Dominant Operator Family")
    ax.set_xlabel("Dominant operator family")
    ax.set_ylabel("Percent below threshold")
    ax.tick_params(axis="x", rotation=60)
    return save_figure(fig, path)


def plot_eml_nodes_by_ast_nodes(rows: Sequence[RawMetricRow], path: Path) -> Path:
    """Plot mean EML node count by exact AST node count."""
    grouped: dict[int, list[int]] = {}
    for row in rows:
        grouped.setdefault(row.ast_node_count, []).append(row.eml_node_count)

    ast_nodes = sorted(grouped)
    mean_eml_nodes = [
        sum(grouped[node_count]) / len(grouped[node_count]) for node_count in ast_nodes
    ]

    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ast_nodes, mean_eml_nodes, marker="o")
    ax.set_title("Mean EML Nodes by AST Node Count")
    ax.set_xlabel("AST node count")
    ax.set_ylabel("Mean EML node count")
    return save_figure(fig, path)


def write_top_expression_tables(
    rows: Sequence[RawMetricRow],
    config: ExpansionPlotConfig,
) -> tuple[Path, ...]:
    """Write top-20 expression tables by alpha, EML nodes, and EML depth."""
    table_specs: tuple[tuple[Path, Callable[[RawMetricRow], float | int]], ...] = (
        (config.top_alpha_csv_path, lambda row: row.alpha),
        (config.top_eml_node_csv_path, lambda row: row.eml_node_count),
        (config.top_eml_depth_csv_path, lambda row: row.eml_depth),
    )
    paths: list[Path] = []
    for path, key in table_specs:
        write_top_table(sorted(rows, key=key, reverse=True)[:20], path)
        paths.append(path)
    return tuple(paths)


def write_top_table(rows: Sequence[RawMetricRow], path: Path) -> None:
    """Write one top-expression CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TOP_TABLE_FIELDS)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "index": row.index,
                    "expression": row.expression,
                    "srepr": row.srepr,
                    "ast_node_count": row.ast_node_count,
                    "ast_depth": row.ast_depth,
                    "ast_operator_count": row.ast_operator_count,
                    "ast_leaf_count": row.ast_leaf_count,
                    "eml_node_count": row.eml_node_count,
                    "eml_depth": row.eml_depth,
                    "eml_operator_count": row.eml_operator_count,
                    "eml_leaf_count": row.eml_leaf_count,
                    "alpha": row.alpha,
                    "alpha_threshold": row.alpha_threshold,
                    "below_threshold": row.below_threshold,
                }
            )


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


def parse_int(value: str) -> int:
    """Parse a required integer CSV field."""
    if value == "":
        raise ValueError("expected integer, got empty string")
    return int(value)


def parse_float(value: str) -> float:
    """Parse a required float CSV field."""
    if value == "":
        raise ValueError("expected float, got empty string")
    return float(value)


def parse_bool(value: str) -> bool:
    """Parse a required boolean CSV field."""
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected boolean string, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-metrics-csv",
        type=Path,
        default=None,
        help="Saved Goal 2 raw metrics CSV path.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for generated PNG plots.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 2.4 plot export."""
    args = build_parser().parse_args(argv)
    config = ExpansionPlotConfig()
    if args.raw_metrics_csv is not None:
        config.raw_metrics_csv_path = args.raw_metrics_csv
    if args.plots_dir is not None:
        config.plots_dir = args.plots_dir

    result = run_expansion_plots(config)
    print(f"Loaded raw metric rows: {result.raw_metric_count}")
    print(f"Loaded alpha summary rows: {result.alpha_summary_count}")
    for path in result.plot_paths:
        print(f"Wrote plot: {path}")
    for path in result.table_paths:
        print(f"Wrote table: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
