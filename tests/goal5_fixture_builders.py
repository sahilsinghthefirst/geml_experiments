"""Small v1 fixture builders for Goal 4/5 tests.

These helpers intentionally generate test-local artifacts under ``tmp_path``.
They prevent the fast test suite from depending on gitignored production files
such as ``outputs/v1/dag_compression_inputs.jsonl`` or Goal 5 metrics CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geml.experiments.dag_compression_study import (
    DagCompressionStudyConfig,
    run_dag_compression_study,
)
from geml.experiments.egraph_compression_study import (
    EgraphCompressionStudyConfig,
    run_egraph_compression_study,
)
from geml.experiments.goal5_frequent_motif_mining import (
    FrequentMotifMiningConfig,
    run_goal5_frequent_motif_mining,
)
from geml.experiments.goal5_learned_motif_compression import (
    LearnedMotifCompressionConfig,
    run_goal5_learned_motif_compression,
)
from geml.experiments.goal5_macro_graph_baseline import (
    MacroGraphBaselineConfig,
    run_goal5_macro_graph_baseline,
)

SMALL_FIXTURE_COUNT = 25


@dataclass(frozen=True, slots=True)
class SmallV1FixturePaths:
    """Paths for a small generated v1-shaped fixture tree."""

    output_dir: Path
    input_jsonl_path: Path
    goal3_metrics_csv_path: Path
    goal3_metrics_jsonl_path: Path
    goal3_summary_json_path: Path
    macro_metrics_csv_path: Path
    macro_metrics_jsonl_path: Path
    macro_summary_json_path: Path
    frequent_vocab_json_path: Path
    frequent_metrics_csv_path: Path
    frequent_metrics_jsonl_path: Path
    frequent_summary_json_path: Path
    learned_vocab_json_path: Path
    learned_metrics_csv_path: Path
    learned_metrics_jsonl_path: Path
    learned_summary_json_path: Path
    learned_train_log_json_path: Path
    egraph_safe_metrics_csv_path: Path
    egraph_safe_metrics_jsonl_path: Path
    egraph_positive_metrics_csv_path: Path
    egraph_positive_metrics_jsonl_path: Path
    egraph_summary_json_path: Path
    egraph_run_metadata_json_path: Path


def small_v1_paths(tmp_path: Path) -> SmallV1FixturePaths:
    """Return deterministic v1-shaped paths under ``tmp_path``."""
    output_dir = tmp_path / "outputs" / "v1"
    return SmallV1FixturePaths(
        output_dir=output_dir,
        input_jsonl_path=output_dir / "dag_compression_inputs.jsonl",
        goal3_metrics_csv_path=output_dir / "dag_compression_metrics.csv",
        goal3_metrics_jsonl_path=output_dir / "dag_compression_metrics.jsonl",
        goal3_summary_json_path=output_dir / "dag_compression_summary.json",
        macro_metrics_csv_path=output_dir / "goal5_macro_graph_metrics.csv",
        macro_metrics_jsonl_path=output_dir / "goal5_macro_graph_metrics.jsonl",
        macro_summary_json_path=output_dir / "goal5_macro_graph_summary.json",
        frequent_vocab_json_path=output_dir / "goal5_frequent_motif_vocab.json",
        frequent_metrics_csv_path=output_dir / "goal5_frequent_motif_metrics.csv",
        frequent_metrics_jsonl_path=output_dir / "goal5_frequent_motif_metrics.jsonl",
        frequent_summary_json_path=output_dir / "goal5_frequent_motif_summary.json",
        learned_vocab_json_path=output_dir / "goal5_learned_motif_vocab.json",
        learned_metrics_csv_path=output_dir / "goal5_learned_motif_metrics.csv",
        learned_metrics_jsonl_path=output_dir / "goal5_learned_motif_metrics.jsonl",
        learned_summary_json_path=output_dir / "goal5_learned_motif_summary.json",
        learned_train_log_json_path=output_dir / "goal5_learned_motif_train_log.json",
        egraph_safe_metrics_csv_path=output_dir / "egraph_compression_metrics_safe.csv",
        egraph_safe_metrics_jsonl_path=output_dir / "egraph_compression_metrics_safe.jsonl",
        egraph_positive_metrics_csv_path=(
            output_dir / "egraph_compression_metrics_positive_real.csv"
        ),
        egraph_positive_metrics_jsonl_path=(
            output_dir / "egraph_compression_metrics_positive_real.jsonl"
        ),
        egraph_summary_json_path=output_dir / "egraph_compression_summary.json",
        egraph_run_metadata_json_path=output_dir / "egraph_compression_run_metadata.json",
    )


def ensure_goal3_fixture(
    tmp_path: Path, *, count: int = SMALL_FIXTURE_COUNT
) -> SmallV1FixturePaths:
    """Generate small Goal 3 DAG fixture inputs and metrics if needed."""
    paths = small_v1_paths(tmp_path)
    if paths.input_jsonl_path.exists() and paths.goal3_metrics_csv_path.exists():
        return paths
    run_dag_compression_study(
        DagCompressionStudyConfig(
            seed=0,
            count=count,
            max_depth=4,
            output_dir=paths.output_dir,
            input_jsonl_path=paths.input_jsonl_path,
            metrics_jsonl_path=paths.goal3_metrics_jsonl_path,
            metrics_csv_path=paths.goal3_metrics_csv_path,
            summary_json_path=paths.goal3_summary_json_path,
        )
    )
    return paths


def ensure_macro_fixture(
    tmp_path: Path, *, count: int = SMALL_FIXTURE_COUNT
) -> SmallV1FixturePaths:
    """Generate small Goal 5.1 macro graph fixture metrics if needed."""
    paths = ensure_goal3_fixture(tmp_path, count=count)
    if paths.macro_metrics_csv_path.exists():
        return paths
    run_goal5_macro_graph_baseline(
        MacroGraphBaselineConfig(
            count=count,
            input_jsonl_path=paths.input_jsonl_path,
            goal3_metrics_csv_path=paths.goal3_metrics_csv_path,
            metrics_csv_path=paths.macro_metrics_csv_path,
            metrics_jsonl_path=paths.macro_metrics_jsonl_path,
            summary_json_path=paths.macro_summary_json_path,
        )
    )
    return paths


def ensure_frequent_fixture(
    tmp_path: Path,
    *,
    count: int = SMALL_FIXTURE_COUNT,
) -> SmallV1FixturePaths:
    """Generate small Goal 5.2 frequent motif fixture artifacts if needed."""
    paths = ensure_macro_fixture(tmp_path, count=count)
    if paths.frequent_metrics_csv_path.exists() and paths.frequent_vocab_json_path.exists():
        return paths
    run_goal5_frequent_motif_mining(
        FrequentMotifMiningConfig(
            count=count,
            min_support=2,
            max_vocab_size=30,
            full_corpus_metrics_csv_path=None,
            input_jsonl_path=paths.input_jsonl_path,
            goal3_metrics_csv_path=paths.goal3_metrics_csv_path,
            macro_graph_metrics_csv_path=paths.macro_metrics_csv_path,
            vocab_json_path=paths.frequent_vocab_json_path,
            metrics_csv_path=paths.frequent_metrics_csv_path,
            metrics_jsonl_path=paths.frequent_metrics_jsonl_path,
            summary_json_path=paths.frequent_summary_json_path,
        )
    )
    return paths


def ensure_learned_fixture(
    tmp_path: Path,
    *,
    count: int = SMALL_FIXTURE_COUNT,
) -> SmallV1FixturePaths:
    """Generate small Goal 5.3 learned motif fixture artifacts if needed."""
    paths = ensure_frequent_fixture(tmp_path, count=count)
    if paths.learned_metrics_csv_path.exists() and paths.learned_vocab_json_path.exists():
        return paths
    run_goal5_learned_motif_compression(
        LearnedMotifCompressionConfig(
            count=count,
            learned_vocab_sizes=(5, 8),
            coverage_bonuses=(0.0, 0.01),
            nontrivial_coverage_bonuses=(0.0,),
            vocab_complexity_penalties=(0.0,),
            expansion_complexity_penalties=(0.0,),
            input_jsonl_path=paths.input_jsonl_path,
            goal3_metrics_csv_path=paths.goal3_metrics_csv_path,
            macro_graph_metrics_csv_path=paths.macro_metrics_csv_path,
            frequent_motif_vocab_json_path=paths.frequent_vocab_json_path,
            frequent_motif_metrics_csv_path=paths.frequent_metrics_csv_path,
            learned_vocab_json_path=paths.learned_vocab_json_path,
            metrics_csv_path=paths.learned_metrics_csv_path,
            metrics_jsonl_path=paths.learned_metrics_jsonl_path,
            summary_json_path=paths.learned_summary_json_path,
            train_log_json_path=paths.learned_train_log_json_path,
        )
    )
    return paths


def ensure_egraph_fixture(
    tmp_path: Path,
    *,
    count: int = SMALL_FIXTURE_COUNT,
) -> SmallV1FixturePaths:
    """Generate small Goal 4 e-graph fixture artifacts if needed."""
    paths = ensure_goal3_fixture(tmp_path, count=count)
    if (
        paths.egraph_safe_metrics_csv_path.exists()
        and paths.egraph_positive_metrics_csv_path.exists()
    ):
        return paths
    run_egraph_compression_study(
        EgraphCompressionStudyConfig(
            count=count,
            input_jsonl_path=paths.input_jsonl_path,
            goal3_metrics_csv_path=paths.goal3_metrics_csv_path,
            goal3_summary_json_path=paths.goal3_summary_json_path,
            v0_v1_comparison_summary_json_path=None,
            output_dir=paths.output_dir,
            safe_metrics_csv_path=paths.egraph_safe_metrics_csv_path,
            safe_metrics_jsonl_path=paths.egraph_safe_metrics_jsonl_path,
            positive_real_metrics_csv_path=paths.egraph_positive_metrics_csv_path,
            positive_real_metrics_jsonl_path=paths.egraph_positive_metrics_jsonl_path,
            summary_json_path=paths.egraph_summary_json_path,
            run_metadata_json_path=paths.egraph_run_metadata_json_path,
            timeout_seconds=0.25,
            beam_size=8,
            max_candidate_depth=7,
            max_candidates_evaluated=8,
            checkpoint_interval=10,
            resume=False,
        )
    )
    return paths


def ensure_hierarchical_fixture(
    tmp_path: Path,
    *,
    count: int = SMALL_FIXTURE_COUNT,
) -> SmallV1FixturePaths:
    """Generate all small upstream artifacts needed by hierarchical export tests."""
    ensure_learned_fixture(tmp_path, count=count)
    return ensure_egraph_fixture(tmp_path, count=count)
