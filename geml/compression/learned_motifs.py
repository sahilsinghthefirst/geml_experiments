"""Learned motif vocabulary records and deterministic baselines."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from geml.compression.motif_selection_model import MotifSplitFeature
from geml.compression.motif_vocab import MotifRecord

type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]


class LearnedMotifRecord(BaseModel):
    """One selected learned motif with split-aware statistics."""

    motif_id: str
    source_motif_id: str
    motif_type: str
    learned_score: float
    support_count: int = Field(ge=0)
    train_support: int = Field(ge=0)
    val_support: int = Field(ge=0)
    test_support: int = Field(ge=0)
    node_savings: int = Field(ge=0)
    coverage: int = Field(ge=0)
    expansion_map: dict[str, MetadataValue]
    expansion_valid: bool
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class LearnedMotifVocabulary(BaseModel):
    """Serializable learned motif vocabulary."""

    vocabulary_version: str = "learned_motifs_v1"
    model_type: str = "deterministic_linear_motif_scorer"
    trained_final_reasoning_gnn: bool = False
    motifs: list[LearnedMotifRecord]
    selected_weights: dict[str, float]
    selected_vocab_size: int = Field(ge=0)
    candidate_pool_size: int = Field(ge=0)
    random_baseline_motif_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


def build_learned_motif_vocabulary(
    *,
    selected_motif_ids: list[str],
    selected_scores: dict[str, float],
    selected_weights: dict[str, float],
    candidate_features_by_id: dict[str, MotifSplitFeature],
    candidate_motifs_by_id: dict[str, MotifRecord],
    random_baseline_motif_ids: list[str],
) -> LearnedMotifVocabulary:
    """Build a learned motif vocabulary artifact from selected source motifs."""
    motifs: list[LearnedMotifRecord] = []
    for rank, source_motif_id in enumerate(selected_motif_ids):
        feature = candidate_features_by_id[source_motif_id]
        motif = candidate_motifs_by_id[source_motif_id]
        motifs.append(
            LearnedMotifRecord(
                motif_id=f"learned_motif_{rank:04d}",
                source_motif_id=source_motif_id,
                motif_type=motif.motif_type,
                learned_score=selected_scores[source_motif_id],
                support_count=feature.support_count,
                train_support=feature.train_support,
                val_support=feature.validation_support,
                test_support=feature.test_support,
                node_savings=feature.node_savings,
                coverage=feature.coverage,
                expansion_map=motif.expansion_map_to_original_graph,
                expansion_valid=bool(motif.expansion_map_to_pure_eml_available),
                node_count=motif.node_count,
                edge_count=motif.edge_count,
                metadata={
                    "source_signature": motif.signature,
                    "is_pure_eml": False,
                    "selection_rank": rank,
                },
            )
        )
    return LearnedMotifVocabulary(
        motifs=motifs,
        selected_weights=selected_weights,
        selected_vocab_size=len(motifs),
        candidate_pool_size=len(candidate_motifs_by_id),
        random_baseline_motif_ids=random_baseline_motif_ids,
        metadata={
            "motif_nodes_are_pure_eml": False,
            "graph_format_neutral": True,
            "selection_uses_test_set": False,
        },
    )


def select_random_motif_ids(
    candidate_motif_ids: list[str],
    *,
    vocab_size: int,
    seed: int,
) -> list[str]:
    """Select a deterministic random motif baseline with the requested size."""
    if vocab_size > len(candidate_motif_ids):
        raise ValueError("vocab_size must be <= candidate pool size")
    rng = random.Random(seed)
    shuffled = list(candidate_motif_ids)
    rng.shuffle(shuffled)
    return sorted(shuffled[:vocab_size])


def write_learned_motif_vocabulary(vocabulary: LearnedMotifVocabulary, path: Path) -> None:
    """Write a deterministic learned motif vocabulary JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(vocabulary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_learned_motif_vocabulary(path: Path) -> LearnedMotifVocabulary:
    """Load a learned motif vocabulary JSON artifact."""
    return LearnedMotifVocabulary.model_validate_json(path.read_text(encoding="utf-8"))
