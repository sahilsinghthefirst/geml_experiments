"""Frequent motif vocabulary records for Goal 5 compression baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

type MotifType = Literal["pure_eml_dag", "macro_graph", "mixed_macro_expansion"]
type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]


class MotifNodeTemplate(BaseModel):
    """A motif-internal node with a local id."""

    local_id: int = Field(ge=0)
    label: str
    kind: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class MotifChildRefTemplate(BaseModel):
    """A motif-internal ordered child reference."""

    parent_local_id: int = Field(ge=0)
    child_local_id: int = Field(ge=0)
    child_slot: str
    slot_index: int = Field(ge=0)


class MotifBoundaryRefTemplate(BaseModel):
    """A motif boundary child reference that must be supplied by an occurrence."""

    parent_local_id: int = Field(ge=0)
    boundary_slot_index: int = Field(ge=0)
    boundary_slot: str
    child_slot: str
    slot_index: int = Field(ge=0)


class MotifRecord(BaseModel):
    """One frequent motif vocabulary entry."""

    motif_id: str
    motif_type: MotifType
    signature: str
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    support_count: int = Field(ge=0)
    support_graph_count: int = Field(ge=0)
    total_covered_nodes: int = Field(ge=0)
    compression_score: float
    expansion_map_to_original_graph: dict[str, MetadataValue]
    expansion_map_to_pure_eml_available: bool
    internal_nodes: list[MotifNodeTemplate]
    internal_child_refs: list[MotifChildRefTemplate] = Field(default_factory=list)
    boundary_child_refs: list[MotifBoundaryRefTemplate] = Field(default_factory=list)
    support_by_subset_label: dict[str, int] = Field(default_factory=dict)
    covered_nodes_by_subset_label: dict[str, int] = Field(default_factory=dict)
    official_macro_name: str | None = None
    is_obvious_official_macro: bool = False
    sample_occurrences: list[dict[str, MetadataValue]] = Field(default_factory=list)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class MotifVocabulary(BaseModel):
    """Serializable frequent motif vocabulary."""

    vocabulary_version: str = "frequent_motifs_v1"
    motifs: list[MotifRecord]
    config: dict[str, MetadataValue] = Field(default_factory=dict)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @property
    def motif_count(self) -> int:
        """Return number of motifs in the vocabulary."""
        return len(self.motifs)

    def motifs_by_type(self, motif_type: MotifType) -> list[MotifRecord]:
        """Return motifs of one type."""
        return [motif for motif in self.motifs if motif.motif_type == motif_type]


def write_motif_vocabulary(vocabulary: MotifVocabulary, path: Path) -> None:
    """Write a deterministic motif vocabulary JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(vocabulary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_motif_vocabulary(path: Path) -> MotifVocabulary:
    """Load a motif vocabulary JSON artifact."""
    return MotifVocabulary.model_validate_json(path.read_text(encoding="utf-8"))


def motif_to_summary_dict(motif: MotifRecord) -> dict[str, object]:
    """Return a compact summary for reports."""
    return {
        "motif_id": motif.motif_id,
        "motif_type": motif.motif_type,
        "node_count": motif.node_count,
        "edge_count": motif.edge_count,
        "support_count": motif.support_count,
        "support_graph_count": motif.support_graph_count,
        "total_covered_nodes": motif.total_covered_nodes,
        "compression_score": motif.compression_score,
        "official_macro_name": motif.official_macro_name,
        "is_obvious_official_macro": motif.is_obvious_official_macro,
        "support_by_subset_label": motif.support_by_subset_label,
        "covered_nodes_by_subset_label": motif.covered_nodes_by_subset_label,
    }
