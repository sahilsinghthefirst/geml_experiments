"""Deterministic learned motif scoring and vocabulary selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from geml.compression.motif_dataset import SplitName


class MotifSplitFeature(BaseModel):
    """Train/validation/test feature statistics for one candidate motif."""

    motif_id: str
    motif_type: str
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    support_count: int = Field(ge=0)
    train_support: int = Field(ge=0)
    validation_support: int = Field(ge=0)
    test_support: int = Field(ge=0)
    node_savings: int = Field(ge=0)
    train_node_savings: int = Field(ge=0)
    validation_node_savings: int = Field(ge=0)
    test_node_savings: int = Field(ge=0)
    coverage: int = Field(ge=0)
    train_coverage: int = Field(ge=0)
    validation_coverage: int = Field(ge=0)
    test_coverage: int = Field(ge=0)
    nontrivial_coverage: int = Field(ge=0)
    train_nontrivial_coverage: int = Field(ge=0)
    validation_nontrivial_coverage: int = Field(ge=0)
    test_nontrivial_coverage: int = Field(ge=0)
    expansion_complexity: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class MotifScoringWeights:
    """Linear motif scoring weights."""

    coverage_bonus: float = 0.0
    nontrivial_coverage_bonus: float = 0.0
    vocab_complexity_penalty: float = 0.0
    expansion_complexity_penalty: float = 0.0

    def to_json_dict(self) -> dict[str, float]:
        """Return JSON-safe weights."""
        return {
            "coverage_bonus": self.coverage_bonus,
            "nontrivial_coverage_bonus": self.nontrivial_coverage_bonus,
            "vocab_complexity_penalty": self.vocab_complexity_penalty,
            "expansion_complexity_penalty": self.expansion_complexity_penalty,
        }


class MotifSelectionTrial(BaseModel):
    """One hyperparameter trial in deterministic motif selection."""

    weights: dict[str, float]
    vocab_size: int
    train_objective: float
    validation_objective: float
    selected_motif_ids: list[str]


class MotifSelectionResult(BaseModel):
    """Selected motif ids and training log."""

    selected_motif_ids: list[str]
    selected_weights: dict[str, float]
    selected_vocab_size: int
    trials: list[MotifSelectionTrial]
    model_type: str = "deterministic_linear_motif_scorer"
    trained_final_reasoning_gnn: bool = False


def score_motif(
    feature: MotifSplitFeature,
    weights: MotifScoringWeights,
    *,
    split: SplitName,
) -> float:
    """Score a motif for one split."""
    node_savings = _split_value(feature, split, "node_savings")
    coverage = _split_value(feature, split, "coverage")
    nontrivial_coverage = _split_value(feature, split, "nontrivial_coverage")
    return (
        float(node_savings)
        + weights.coverage_bonus * float(coverage)
        + weights.nontrivial_coverage_bonus * float(nontrivial_coverage)
        - weights.vocab_complexity_penalty * float(feature.node_count)
        - weights.expansion_complexity_penalty * float(feature.expansion_complexity)
    )


def optimize_motif_selection(
    features: Sequence[MotifSplitFeature],
    *,
    vocab_sizes: Sequence[int],
    weight_grid: Sequence[MotifScoringWeights],
) -> MotifSelectionResult:
    """Select motif scorer hyperparameters using validation objective only."""
    if not features:
        raise ValueError("features must not be empty")
    if not vocab_sizes:
        raise ValueError("vocab_sizes must not be empty")
    if not weight_grid:
        raise ValueError("weight_grid must not be empty")

    trials: list[MotifSelectionTrial] = []
    for weights in weight_grid:
        train_scored = sorted(
            ((score_motif(feature, weights, split="train"), feature) for feature in features),
            key=lambda item: (-item[0], item[1].motif_type, item[1].motif_id),
        )
        for vocab_size in vocab_sizes:
            selected_features = [
                feature for _, feature in train_scored[: min(vocab_size, len(train_scored))]
            ]
            train_objective = sum(
                score_motif(feature, weights, split="train") for feature in selected_features
            )
            validation_objective = sum(
                score_motif(feature, weights, split="validation") for feature in selected_features
            )
            trials.append(
                MotifSelectionTrial(
                    weights=weights.to_json_dict(),
                    vocab_size=len(selected_features),
                    train_objective=train_objective,
                    validation_objective=validation_objective,
                    selected_motif_ids=[feature.motif_id for feature in selected_features],
                )
            )

    best = max(
        trials,
        key=lambda trial: (
            trial.validation_objective,
            trial.train_objective,
            -trial.vocab_size,
            trial.selected_motif_ids,
        ),
    )
    return MotifSelectionResult(
        selected_motif_ids=best.selected_motif_ids,
        selected_weights=best.weights,
        selected_vocab_size=best.vocab_size,
        trials=trials,
    )


def build_weight_grid(
    *,
    coverage_bonuses: Sequence[float],
    nontrivial_coverage_bonuses: Sequence[float],
    vocab_complexity_penalties: Sequence[float],
    expansion_complexity_penalties: Sequence[float],
) -> tuple[MotifScoringWeights, ...]:
    """Build a deterministic hyperparameter grid."""
    return tuple(
        MotifScoringWeights(
            coverage_bonus=coverage_bonus,
            nontrivial_coverage_bonus=nontrivial_bonus,
            vocab_complexity_penalty=vocab_penalty,
            expansion_complexity_penalty=expansion_penalty,
        )
        for coverage_bonus in coverage_bonuses
        for nontrivial_bonus in nontrivial_coverage_bonuses
        for vocab_penalty in vocab_complexity_penalties
        for expansion_penalty in expansion_complexity_penalties
    )


def _split_value(feature: MotifSplitFeature, split: SplitName, metric: str) -> int:
    if split == "train":
        return int(getattr(feature, f"train_{metric}"))
    if split == "validation":
        return int(getattr(feature, f"validation_{metric}"))
    if split == "test":
        return int(getattr(feature, f"test_{metric}"))
    raise ValueError(f"unknown split {split!r}")
