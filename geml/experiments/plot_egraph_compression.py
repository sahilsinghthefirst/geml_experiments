"""Goal 4.8 reproducible plots for v1 e-graph compression metrics."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import matplotlib
from matplotlib.figure import Figure

GOAL4_PLOT_FILENAMES = (
    "goal3_dag_alpha_vs_egraph_optimized_alpha_safe.png",
    "goal3_dag_alpha_vs_egraph_optimized_alpha_positive_real.png",
    "egraph_compression_gain_histogram_safe.png",
    "egraph_compression_gain_histogram_positive_real.png",
    "percent_below_threshold_before_after_egraph.png",
    "optimized_alpha_by_operator_family.png",
    "compression_gain_by_operator_family.png",
    "compression_gain_by_subset_label.png",
    "timeout_rate_by_operator_family.png",
    "egraph_nodes_vs_compression_gain.png",
    "nontrivial_vs_identity_heavy_improvement.png",
)


@dataclass(frozen=True, slots=True)
class EgraphCompressionPlotConfig:
    """Input and output paths for Goal 4.8 e-graph compression plots."""

    safe_metrics_csv_path: Path = Path("outputs/v1/egraph_compression_metrics_safe.csv")
    positive_real_metrics_csv_path: Path = Path(
        "outputs/v1/egraph_compression_metrics_positive_real.csv"
    )
    operator_signature_csv_path: Path = Path("outputs/v1/egraph_alpha_by_operator_signature.csv")
    operator_family_csv_path: Path = Path("outputs/v1/egraph_alpha_by_operator_family.csv")
    subset_label_csv_path: Path = Path("outputs/v1/egraph_alpha_by_subset_label.csv")
    plots_dir: Path = Path("outputs/v1/plots_goal4")

    def validate(self) -> None:
        """Validate that Goal 4.8 does not write primary outputs to v0."""
        if "outputs/v0" in self.plots_dir.as_posix():
            raise ValueError("Goal 4.8 plots must not be written to outputs/v0")


@dataclass(frozen=True, slots=True)
class EgraphCompressionPlotRow:
    """Per-expression fields needed by Goal 4.8 plots."""

    index: int
    rule_mode: str
    validation_status: str
    extraction_status: str
    timeout: bool
    enode_count: int | None
    goal3_dag_alpha_vs_ast_tree: float
    optimized_dag_alpha_vs_ast_tree: float | None
    compression_gain_vs_goal3_dag: float | None
    subset_label: str
    structural_purity_valid: bool

    @property
    def is_success(self) -> bool:
        """Return whether this row has a valid optimized EML-DAG result."""
        return (
            self.extraction_status == "completed"
            and self.validation_status == "valid"
            and self.structural_purity_valid
            and self.optimized_dag_alpha_vs_ast_tree is not None
            and self.compression_gain_vs_goal3_dag is not None
        )


@dataclass(frozen=True, slots=True)
class EgraphCompressionPlotResult:
    """Result metadata from a Goal 4.8 plotting run."""

    safe_metric_count: int
    positive_real_metric_count: int
    operator_signature_count: int
    operator_family_count: int
    subset_summary_count: int
    plot_paths: tuple[Path, ...]


def run_egraph_compression_plots(
    config: EgraphCompressionPlotConfig,
) -> EgraphCompressionPlotResult:
    """Load saved v1 Goal 4 artifacts and write all requested PNG plots."""
    config.validate()
    safe_rows = load_egraph_plot_rows(config.safe_metrics_csv_path)
    positive_rows = load_egraph_plot_rows(config.positive_real_metrics_csv_path)
    operator_signature_rows = load_csv_rows(config.operator_signature_csv_path)
    operator_family_rows = load_csv_rows(config.operator_family_csv_path)
    subset_rows = load_csv_rows(config.subset_label_csv_path)

    config.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = (
        plot_goal3_vs_optimized_alpha(
            safe_rows,
            config.plots_dir / "goal3_dag_alpha_vs_egraph_optimized_alpha_safe.png",
            title="Goal 3 DAG Alpha vs Safe E-Graph Optimized Alpha",
            source_path=config.safe_metrics_csv_path,
        ),
        plot_goal3_vs_optimized_alpha(
            positive_rows,
            config.plots_dir / "goal3_dag_alpha_vs_egraph_optimized_alpha_positive_real.png",
            title="Goal 3 DAG Alpha vs Positive-Real E-Graph Optimized Alpha",
            source_path=config.positive_real_metrics_csv_path,
        ),
        plot_compression_gain_histogram(
            safe_rows,
            config.plots_dir / "egraph_compression_gain_histogram_safe.png",
            title="Safe E-Graph Compression Gain Distribution",
            source_path=config.safe_metrics_csv_path,
        ),
        plot_compression_gain_histogram(
            positive_rows,
            config.plots_dir / "egraph_compression_gain_histogram_positive_real.png",
            title="Positive-Real E-Graph Compression Gain Distribution",
            source_path=config.positive_real_metrics_csv_path,
        ),
        plot_percent_below_threshold_before_after(
            subset_rows,
            config.plots_dir / "percent_below_threshold_before_after_egraph.png",
            source_path=config.subset_label_csv_path,
        ),
        plot_operator_family_metric(
            operator_family_rows,
            config.plots_dir / "optimized_alpha_by_operator_family.png",
            metric_field="median_optimized_dag_alpha_vs_ast_tree",
            title="Optimized DAG Alpha by Operator Family",
            ylabel="Median optimized EML-DAG nodes / AST tree nodes",
            source_path=config.operator_family_csv_path,
        ),
        plot_operator_family_metric(
            operator_family_rows,
            config.plots_dir / "compression_gain_by_operator_family.png",
            metric_field="median_compression_gain_vs_goal3_dag",
            title="Compression Gain by Operator Family",
            ylabel="Median Goal 3 EML-DAG nodes / optimized EML-DAG nodes",
            source_path=config.operator_family_csv_path,
        ),
        plot_compression_gain_by_subset_label(
            subset_rows,
            config.plots_dir / "compression_gain_by_subset_label.png",
            source_path=config.subset_label_csv_path,
        ),
        plot_operator_family_metric(
            operator_family_rows,
            config.plots_dir / "timeout_rate_by_operator_family.png",
            metric_field="timeout_rate",
            title="Timeout Rate by Operator Family",
            ylabel="Timeout rate (%)",
            source_path=config.operator_family_csv_path,
            weight_field="count",
        ),
        plot_egraph_nodes_vs_compression_gain(
            [*safe_rows, *positive_rows],
            config.plots_dir / "egraph_nodes_vs_compression_gain.png",
            source_paths=(config.safe_metrics_csv_path, config.positive_real_metrics_csv_path),
        ),
        plot_nontrivial_vs_identity_heavy_improvement(
            subset_rows,
            config.plots_dir / "nontrivial_vs_identity_heavy_improvement.png",
            source_path=config.subset_label_csv_path,
        ),
    )

    return EgraphCompressionPlotResult(
        safe_metric_count=len(safe_rows),
        positive_real_metric_count=len(positive_rows),
        operator_signature_count=len(operator_signature_rows),
        operator_family_count=len(operator_family_rows),
        subset_summary_count=len(subset_rows),
        plot_paths=plot_paths,
    )


def load_egraph_plot_rows(path: Path) -> list[EgraphCompressionPlotRow]:
    """Load one saved Goal 4.6 per-expression metrics CSV."""
    rows: list[EgraphCompressionPlotRow] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            rows.append(
                EgraphCompressionPlotRow(
                    index=parse_int(raw_row["index"]),
                    rule_mode=raw_row["rule_mode"],
                    validation_status=status_value(raw_row.get("validation_status")),
                    extraction_status=status_value(raw_row.get("extraction_status")),
                    timeout=parse_bool(raw_row.get("timeout", "False")),
                    enode_count=parse_optional_int(raw_row.get("enode_count")),
                    goal3_dag_alpha_vs_ast_tree=parse_float(raw_row["goal3_dag_alpha_vs_ast_tree"]),
                    optimized_dag_alpha_vs_ast_tree=parse_optional_float(
                        raw_row.get("optimized_dag_alpha_vs_ast_tree")
                    ),
                    compression_gain_vs_goal3_dag=parse_optional_float(
                        raw_row.get("compression_gain_vs_goal3_dag")
                    ),
                    subset_label=raw_row["subset_label"],
                    structural_purity_valid=parse_bool(
                        raw_row.get("structural_purity_valid", "True")
                    ),
                )
            )
    if not rows:
        raise ValueError(f"no e-graph metric rows found in {path}")
    return rows


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a saved CSV artifact."""
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def plot_goal3_vs_optimized_alpha(
    rows: Sequence[EgraphCompressionPlotRow],
    path: Path,
    *,
    title: str,
    source_path: Path,
) -> Path:
    """Scatter Goal 3 DAG alpha against optimized e-graph DAG alpha."""
    success_rows = successful_rows(rows)
    x_values = [row.goal3_dag_alpha_vs_ast_tree for row in success_rows]
    y_values = [float(row.optimized_dag_alpha_vs_ast_tree) for row in success_rows]

    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_values, y_values, s=8, alpha=0.35)
    lower = min([*x_values, *y_values])
    upper = max([*x_values, *y_values])
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Goal 3 exact EML-DAG alpha vs AST tree")
    ax.set_ylabel("E-graph optimized EML-DAG alpha vs AST tree")
    return save_figure(fig, path, source_paths=(source_path,))


def plot_compression_gain_histogram(
    rows: Sequence[EgraphCompressionPlotRow],
    path: Path,
    *,
    title: str,
    source_path: Path,
) -> Path:
    """Plot the compression gain distribution for one rule mode."""
    gains = [float(row.compression_gain_vs_goal3_dag) for row in successful_rows(rows)]
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gains, bins=40)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Goal 3 EML-DAG nodes / optimized EML-DAG nodes")
    ax.set_ylabel("Expression count")
    return save_figure(fig, path, source_paths=(source_path,))


def plot_percent_below_threshold_before_after(
    rows: Sequence[dict[str, str]],
    path: Path,
    *,
    source_path: Path,
) -> Path:
    """Plot before/after threshold-pass percentages by rule mode."""
    all_rows = [
        row
        for row in rows
        if row["subset_label"] == "all_v1" and row["rule_mode"] in {"safe", "positive_real_formal"}
    ]
    all_rows = sorted(all_rows, key=lambda row: rule_mode_sort(row["rule_mode"]))
    labels = [display_mode(row["rule_mode"]) for row in all_rows]
    before_values = [parse_float(row["percent_below_threshold_before"]) for row in all_rows]
    after_values = [parse_float(row["percent_below_threshold_after"]) for row in all_rows]

    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = list(range(len(all_rows)))
    width = 0.35
    ax.bar(
        [position - width / 2 for position in positions],
        before_values,
        width=width,
        label="Before",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        after_values,
        width=width,
        label="After",
    )
    ax.set_title("Percent Below Threshold Before vs After E-Graph Extraction")
    ax.set_xlabel("Rule mode")
    ax.set_ylabel("Percent below current alpha threshold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.legend()
    return save_figure(fig, path, source_paths=(source_path,))


def plot_operator_family_metric(
    rows: Sequence[dict[str, str]],
    path: Path,
    *,
    metric_field: str,
    title: str,
    ylabel: str,
    source_path: Path,
    weight_field: str = "success_count",
    max_families: int = 12,
) -> Path:
    """Plot a weighted operator-family summary for safe and positive-real modes."""
    points = aggregate_operator_family_metric(
        rows,
        metric_field=metric_field,
        weight_field=weight_field,
        max_families=max_families,
    )
    labels = sorted(
        {family for _, family in points},
        key=lambda family: (
            -sum(point.count for key, point in points.items() if key[1] == family),
            family,
        ),
    )
    modes = ["safe", "positive_real_formal"]
    width = 0.35
    positions = list(range(len(labels)))

    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(12, 6))
    for offset_index, mode in enumerate(modes):
        offset = (offset_index - 0.5) * width
        values = [points.get((mode, label), FamilyPlotPoint(0, 0.0)).value for label in labels]
        ax.bar(
            [position + offset for position in positions],
            values,
            width=width,
            label=display_mode(mode),
        )
    ax.set_title(title)
    ax.set_xlabel("Dominant operator family")
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.legend()
    return save_figure(fig, path, source_paths=(source_path,))


def plot_compression_gain_by_subset_label(
    rows: Sequence[dict[str, str]],
    path: Path,
    *,
    source_path: Path,
) -> Path:
    """Plot median compression gain by v1 subset label and rule mode."""
    return plot_subset_metric(
        rows,
        path,
        metric_field="median_compression_gain_vs_goal3_dag",
        title="Compression Gain by v1 Subset",
        ylabel="Median Goal 3 EML-DAG nodes / optimized EML-DAG nodes",
        source_path=source_path,
    )


def plot_nontrivial_vs_identity_heavy_improvement(
    rows: Sequence[dict[str, str]],
    path: Path,
    *,
    source_path: Path,
) -> Path:
    """Plot percent-improved split between nontrivial and identity-heavy subsets."""
    selected_rows = [row for row in rows if row["subset_label"] != "all_v1"]
    return plot_subset_metric(
        selected_rows,
        path,
        metric_field="percent_improved",
        title="Nontrivial vs Identity-Heavy Improvement",
        ylabel="Percent improved among successful rows",
        source_path=source_path,
    )


def plot_subset_metric(
    rows: Sequence[dict[str, str]],
    path: Path,
    *,
    metric_field: str,
    title: str,
    ylabel: str,
    source_path: Path,
) -> Path:
    """Plot one subset-label metric with grouped rule-mode bars."""
    subset_order = ["all_v1", "nontrivial_v1", "identity_heavy_v1"]
    labels = [label for label in subset_order if any(row["subset_label"] == label for row in rows)]
    modes = ["safe", "positive_real_formal"]
    values_by_key = {
        (row["rule_mode"], row["subset_label"]): parse_optional_float(row.get(metric_field))
        for row in rows
    }
    width = 0.35
    positions = list(range(len(labels)))

    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset_index, mode in enumerate(modes):
        offset = (offset_index - 0.5) * width
        values = [values_by_key.get((mode, label), 0.0) or 0.0 for label in labels]
        ax.bar(
            [position + offset for position in positions],
            values,
            width=width,
            label=display_mode(mode),
        )
    ax.set_title(title)
    ax.set_xlabel("v1 subset label")
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    return save_figure(fig, path, source_paths=(source_path,))


def plot_egraph_nodes_vs_compression_gain(
    rows: Sequence[EgraphCompressionPlotRow],
    path: Path,
    *,
    source_paths: Sequence[Path],
) -> Path:
    """Scatter e-node count against compression gain for both rule modes."""
    mode_rows = {
        "safe": [
            row
            for row in rows
            if row.rule_mode == "safe"
            and row.is_success
            and row.enode_count is not None
            and row.compression_gain_vs_goal3_dag is not None
        ],
        "positive_real_formal": [
            row
            for row in rows
            if row.rule_mode == "positive_real_formal"
            and row.is_success
            and row.enode_count is not None
            and row.compression_gain_vs_goal3_dag is not None
        ],
    }
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, mode_values in mode_rows.items():
        ax.scatter(
            [int(row.enode_count) for row in mode_values],
            [float(row.compression_gain_vs_goal3_dag) for row in mode_values],
            s=8,
            alpha=0.3,
            label=display_mode(mode),
        )
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_title("E-Graph Nodes vs Compression Gain")
    ax.set_xlabel("E-node count after saturation")
    ax.set_ylabel("Goal 3 EML-DAG nodes / optimized EML-DAG nodes")
    ax.legend()
    return save_figure(fig, path, source_paths=source_paths)


@dataclass(frozen=True, slots=True)
class FamilyPlotPoint:
    """Weighted aggregate used by operator-family plots."""

    count: int
    value: float


def aggregate_operator_family_metric(
    rows: Sequence[dict[str, str]],
    *,
    metric_field: str,
    weight_field: str,
    max_families: int,
) -> dict[tuple[str, str], FamilyPlotPoint]:
    """Aggregate stratified operator-family rows into plot-ready points."""
    all_rows = [row for row in rows if row["subset_label"] == "all_v1"]
    family_counts: dict[str, int] = {}
    for row in all_rows:
        family = row["dominant_operator_family"]
        family_counts[family] = family_counts.get(family, 0) + parse_int(row["count"])
    selected_families = {
        family
        for family, _ in sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))[
            :max_families
        ]
    }

    weighted: dict[tuple[str, str], tuple[float, int]] = {}
    for row in all_rows:
        family = row["dominant_operator_family"]
        if family not in selected_families:
            continue
        value = parse_optional_float(row.get(metric_field))
        if value is None:
            continue
        weight = parse_int(row[weight_field])
        if weight <= 0:
            continue
        key = (row["rule_mode"], family)
        total, count = weighted.get(key, (0.0, 0))
        weighted[key] = (total + value * weight, count + weight)
    return {
        key: FamilyPlotPoint(count=count, value=total / count)
        for key, (total, count) in weighted.items()
        if count > 0
    }


def successful_rows(rows: Sequence[EgraphCompressionPlotRow]) -> list[EgraphCompressionPlotRow]:
    """Return valid rows with optimized metrics."""
    success_rows = [row for row in rows if row.is_success]
    if not success_rows:
        raise ValueError("no successful e-graph rows available for plot")
    return success_rows


def save_figure(fig: Figure, path: Path, *, source_paths: Sequence[Path]) -> Path:
    """Save and close a Matplotlib figure with source metadata."""
    plt = get_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    source_text = ", ".join(str(path) for path in source_paths)
    fig.savefig(
        path,
        dpi=160,
        metadata={
            "Title": fig.axes[0].get_title() if fig.axes else path.name,
            "Description": f"Source filename(s): {source_text}",
        },
    )
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


def parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer CSV field."""
    if value in {None, ""}:
        return None
    return int(value)


def parse_float(value: str) -> float:
    """Parse a required float CSV field."""
    if value == "":
        raise ValueError("expected float, got empty string")
    return float(value)


def parse_optional_float(value: str | None) -> float | None:
    """Parse an optional float CSV field."""
    if value in {None, ""}:
        return None
    return float(value)


def parse_bool(value: str | bool | None) -> bool:
    """Parse a required boolean CSV field."""
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected boolean string, got {value!r}")


def status_value(value: str | None) -> str:
    """Normalize optional status text from CSV."""
    return value if value not in {None, ""} else "missing"


def display_mode(rule_mode: str) -> str:
    """Return a compact display label for a rule mode."""
    if rule_mode == "positive_real_formal":
        return "positive_real"
    return rule_mode


def rule_mode_sort(rule_mode: str) -> int:
    """Sort safe before positive-real mode."""
    return {"safe": 0, "positive_real_formal": 1}.get(rule_mode, 2)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--safe-metrics-csv", type=Path, default=None)
    parser.add_argument("--positive-real-metrics-csv", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Goal 4.8 e-graph compression plot export."""
    args = build_parser().parse_args(argv)
    config = EgraphCompressionPlotConfig()
    if args.plots_dir is not None:
        config = EgraphCompressionPlotConfig(
            safe_metrics_csv_path=config.safe_metrics_csv_path,
            positive_real_metrics_csv_path=config.positive_real_metrics_csv_path,
            operator_signature_csv_path=config.operator_signature_csv_path,
            operator_family_csv_path=config.operator_family_csv_path,
            subset_label_csv_path=config.subset_label_csv_path,
            plots_dir=args.plots_dir,
        )
    if args.safe_metrics_csv is not None or args.positive_real_metrics_csv is not None:
        config = EgraphCompressionPlotConfig(
            safe_metrics_csv_path=args.safe_metrics_csv or config.safe_metrics_csv_path,
            positive_real_metrics_csv_path=args.positive_real_metrics_csv
            or config.positive_real_metrics_csv_path,
            operator_signature_csv_path=config.operator_signature_csv_path,
            operator_family_csv_path=config.operator_family_csv_path,
            subset_label_csv_path=config.subset_label_csv_path,
            plots_dir=config.plots_dir,
        )

    result = run_egraph_compression_plots(config)
    print(f"Loaded safe metric rows: {result.safe_metric_count}")
    print(f"Loaded positive-real metric rows: {result.positive_real_metric_count}")
    print(f"Loaded operator-signature summary rows: {result.operator_signature_count}")
    print(f"Loaded operator-family summary rows: {result.operator_family_count}")
    print(f"Loaded subset summary rows: {result.subset_summary_count}")
    for path in result.plot_paths:
        print(f"Wrote plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
