"""Dataset and split helpers for learned motif selection."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from geml.data.dataset import GeneratedExpressionInput, load_generated_expressions

type SplitName = Literal["train", "validation", "test"]
type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Deterministic expression-index split config."""

    seed: int = 0
    train_fraction: float = 0.7
    validation_fraction: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be < 1")


@dataclass(frozen=True, slots=True)
class ExpressionSplitRow:
    """One expression's deterministic split assignment."""

    index: int
    split: SplitName


@dataclass(frozen=True, slots=True)
class FrequentMotifBaselineRow:
    """Goal 5.2 per-row baseline metrics used by Goal 5.3."""

    index: int
    subset_label: str
    original_eml_dag_nodes: int
    frequent_motif_nodes: int
    macro_graph_nodes: int
    expansion_valid: bool


@dataclass(frozen=True, slots=True)
class MacroGraphBaselineRow:
    """Goal 5.1 macro graph baseline node counts."""

    index: int
    macro_graph_nodes: int
    expansion_valid: bool


def assign_split(index: int, config: SplitConfig) -> SplitName:
    """Assign one expression index to train/validation/test deterministically."""
    digest = hashlib.sha256(f"{config.seed}:{index}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    if bucket < config.train_fraction:
        return "train"
    if bucket < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def build_split_rows(indices: Iterable[int], config: SplitConfig) -> tuple[ExpressionSplitRow, ...]:
    """Build deterministic split rows for expression indices."""
    return tuple(
        ExpressionSplitRow(index=index, split=assign_split(index, config)) for index in indices
    )


def load_v1_input_rows(path: Path, *, count: int) -> tuple[GeneratedExpressionInput, ...]:
    """Load v1 generated expression rows."""
    rows = tuple(load_generated_expressions(path)[:count])
    if len(rows) != count:
        raise ValueError(f"expected {count} input rows, found {len(rows)}")
    return rows


def load_frequent_motif_baseline_rows(path: Path) -> dict[int, FrequentMotifBaselineRow]:
    """Load Goal 5.2 per-row motif baseline metrics."""
    rows: dict[int, FrequentMotifBaselineRow] = {}
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            row = FrequentMotifBaselineRow(
                index=int(raw_row["index"]),
                subset_label=raw_row["subset_label"],
                original_eml_dag_nodes=int(raw_row["original_eml_dag_nodes"]),
                frequent_motif_nodes=int(raw_row["motif_compressed_nodes"]),
                macro_graph_nodes=int(raw_row["macro_graph_nodes"]),
                expansion_valid=_parse_bool(raw_row["expansion_valid"]),
            )
            rows[row.index] = row
    return rows


def load_macro_graph_baseline_rows(path: Path) -> dict[int, MacroGraphBaselineRow]:
    """Load Goal 5.1 macro graph baseline metrics."""
    rows: dict[int, MacroGraphBaselineRow] = {}
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            row = MacroGraphBaselineRow(
                index=int(raw_row["index"]),
                macro_graph_nodes=int(raw_row["macro_graph_nodes"]),
                expansion_valid=_parse_bool(raw_row["expansion_valid"]),
            )
            rows[row.index] = row
    return rows


def summarize_split_counts(split_rows: Sequence[ExpressionSplitRow]) -> dict[str, int]:
    """Return expression counts by split."""
    counts = {"train": 0, "validation": 0, "test": 0}
    for row in split_rows:
        counts[row.split] += 1
    return counts


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected boolean string, got {value!r}")
