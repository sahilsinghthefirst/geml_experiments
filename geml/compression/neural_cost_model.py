"""Deterministic lightweight neural cost model for e-graph candidates."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from geml.compression.egraph_candidate_dataset import (
    EgraphCandidateRecord,
    candidate_feature_names,
    valid_labeled_records,
)
from geml.egraph.rule_sets import RuleMode

type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]


@dataclass(frozen=True, slots=True)
class NeuralCostModelConfig:
    """Training configuration for the feature-based MLP ranker."""

    seed: int = 0
    hidden_size: int = 12
    epochs: int = 12
    learning_rate: float = 0.01
    max_pairs_per_group: int = 8
    l2_penalty: float = 1e-5

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_pairs_per_group <= 0:
            raise ValueError("max_pairs_per_group must be positive")
        if self.l2_penalty < 0:
            raise ValueError("l2_penalty must be non-negative")


class NeuralCostModel(BaseModel):
    """Serializable one-hidden-layer MLP cost scorer.

    Lower predicted scores are ranked as cheaper candidates.
    """

    model_type: str = "feature_mlp_pairwise_ranker"
    trained_final_reasoning_gnn: bool = False
    feature_names: list[str]
    feature_means: list[float]
    feature_stds: list[float]
    input_hidden_weights: list[list[float]]
    hidden_bias: list[float]
    hidden_output_weights: list[float]
    output_bias: float
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    def predict_feature_dict(self, features: dict[str, float]) -> float:
        """Predict a ranking score from a feature dictionary."""
        vector = normalize_feature_vector(
            feature_dict_to_vector(features, self.feature_names),
            means=self.feature_means,
            stds=self.feature_stds,
        )
        return self.predict_vector(vector)

    def predict_vector(self, vector: Sequence[float]) -> float:
        """Predict a ranking score from an already-normalized vector."""
        hidden = []
        for row, bias in zip(self.input_hidden_weights, self.hidden_bias, strict=True):
            activation = bias + sum(
                weight * value for weight, value in zip(row, vector, strict=True)
            )
            hidden.append(max(0.0, activation))
        return self.output_bias + sum(
            weight * value for weight, value in zip(self.hidden_output_weights, hidden, strict=True)
        )


class NeuralCostTrainResult(BaseModel):
    """Trained cost model and train log payload."""

    model: NeuralCostModel
    train_log: dict[str, MetadataValue]


def train_neural_cost_model(
    records: Sequence[EgraphCandidateRecord],
    *,
    config: NeuralCostModelConfig,
) -> NeuralCostTrainResult:
    """Train a deterministic pairwise MLP ranker on training candidate groups."""
    train_records = [record for record in valid_labeled_records(records) if record.split == "train"]
    feature_names = list(candidate_feature_names(train_records))
    feature_matrix = [
        feature_dict_to_vector(record.candidate_ir_features, feature_names)
        for record in train_records
    ]
    if feature_matrix:
        means, stds = fit_normalizer(feature_matrix)
    else:
        means = [0.0 for _ in feature_names]
        stds = [1.0 for _ in feature_names]
    normalized_by_id = {
        record.candidate_id: normalize_feature_vector(vector, means=means, stds=stds)
        for record, vector in zip(train_records, feature_matrix, strict=True)
    }

    rng = random.Random(config.seed)
    input_hidden_weights = [
        [rng.uniform(-0.05, 0.05) for _ in feature_names] for _ in range(config.hidden_size)
    ]
    hidden_bias = [0.0 for _ in range(config.hidden_size)]
    hidden_output_weights = [rng.uniform(-0.05, 0.05) for _ in range(config.hidden_size)]
    output_bias = 0.0

    pairs = build_pairwise_training_pairs(
        train_records, max_pairs_per_group=config.max_pairs_per_group
    )
    epoch_logs: list[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        epoch_rng = random.Random(config.seed + epoch + 1)
        shuffled_pairs = list(pairs)
        epoch_rng.shuffle(shuffled_pairs)
        total_loss = 0.0
        for better_id, worse_id in shuffled_pairs:
            loss, output_bias = _train_pair(
                normalized_by_id[better_id],
                normalized_by_id[worse_id],
                input_hidden_weights=input_hidden_weights,
                hidden_bias=hidden_bias,
                hidden_output_weights=hidden_output_weights,
                output_bias=output_bias,
                learning_rate=config.learning_rate,
                l2_penalty=config.l2_penalty,
            )
            total_loss += loss
        mean_loss = total_loss / len(shuffled_pairs) if shuffled_pairs else 0.0
        epoch_logs.append(
            {
                "epoch": epoch,
                "pair_count": len(shuffled_pairs),
                "mean_pairwise_logistic_loss": mean_loss,
            }
        )

    model = NeuralCostModel(
        feature_names=feature_names,
        feature_means=means,
        feature_stds=stds,
        input_hidden_weights=input_hidden_weights,
        hidden_bias=hidden_bias,
        hidden_output_weights=hidden_output_weights,
        output_bias=output_bias,
        metadata={
            "seed": config.seed,
            "hidden_size": config.hidden_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "max_pairs_per_group": config.max_pairs_per_group,
            "l2_penalty": config.l2_penalty,
            "training_objective": "pairwise_within_root_logistic_ranking",
            "test_set_used_for_training": False,
        },
    )
    return NeuralCostTrainResult(
        model=model,
        train_log={
            "model_type": model.model_type,
            "trained_final_reasoning_gnn": False,
            "training_objective": "pairwise candidate ranking within expression/rule_mode groups",
            "seed": config.seed,
            "feature_names": feature_names,
            "train_candidate_count": len(train_records),
            "train_pair_count": len(pairs),
            "epoch_logs": epoch_logs,
            "test_set_used_for_training": False,
        },
    )


def build_pairwise_training_pairs(
    records: Sequence[EgraphCandidateRecord],
    *,
    max_pairs_per_group: int,
) -> tuple[tuple[str, str], ...]:
    """Build deterministic lower-cost/higher-cost pairs within each root group."""
    groups: dict[tuple[int, RuleMode], list[EgraphCandidateRecord]] = defaultdict(list)
    for record in records:
        if record.true_official_eml_dag_nodes is not None:
            groups[record.group_key].append(record)

    pairs: list[tuple[str, str]] = []
    for group_key in sorted(groups):
        ranked = sorted(groups[group_key], key=_official_tie_break_key)
        if len(ranked) < 2:
            continue
        best = ranked[0]
        added = 0
        for candidate in ranked[1:]:
            if candidate.true_official_eml_dag_nodes == best.true_official_eml_dag_nodes:
                continue
            pairs.append((best.candidate_id, candidate.candidate_id))
            added += 1
            if added >= max_pairs_per_group:
                break
    return tuple(pairs)


def feature_dict_to_vector(features: dict[str, float], feature_names: Sequence[str]) -> list[float]:
    """Project a feature dictionary into a stable dense vector."""
    return [float(features.get(name, 0.0)) for name in feature_names]


def fit_normalizer(feature_matrix: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    """Fit mean/std normalization parameters."""
    if not feature_matrix:
        return [], []
    width = len(feature_matrix[0])
    means: list[float] = []
    stds: list[float] = []
    for column in range(width):
        values = [row[column] for row in feature_matrix]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        means.append(mean)
        stds.append(std if std > 1e-12 else 1.0)
    return means, stds


def normalize_feature_vector(
    vector: Sequence[float],
    *,
    means: Sequence[float],
    stds: Sequence[float],
) -> list[float]:
    """Normalize a feature vector with fitted parameters."""
    return [(value - mean) / std for value, mean, std in zip(vector, means, stds, strict=True)]


def predict_records(
    model: NeuralCostModel,
    records: Iterable[EgraphCandidateRecord],
) -> dict[str, float]:
    """Predict scores keyed by candidate id."""
    return {
        record.candidate_id: model.predict_feature_dict(record.candidate_ir_features)
        for record in records
    }


def _official_tie_break_key(
    record: EgraphCandidateRecord,
) -> tuple[int, int, int, int, str]:
    return (
        record.true_official_eml_dag_nodes
        if record.true_official_eml_dag_nodes is not None
        else 10**12,
        record.true_official_eml_tree_nodes
        if record.true_official_eml_tree_nodes is not None
        else 10**12,
        record.true_ast_dag_nodes if record.true_ast_dag_nodes is not None else 10**12,
        record.true_ast_tree_nodes if record.true_ast_tree_nodes is not None else 10**12,
        record.candidate_expression,
    )


def _forward(
    vector: Sequence[float],
    *,
    input_hidden_weights: Sequence[Sequence[float]],
    hidden_bias: Sequence[float],
    hidden_output_weights: Sequence[float],
    output_bias: float,
) -> tuple[float, list[float], list[float]]:
    pre_activations = []
    hidden = []
    for row, bias in zip(input_hidden_weights, hidden_bias, strict=True):
        pre_activation = bias + sum(
            weight * value for weight, value in zip(row, vector, strict=True)
        )
        pre_activations.append(pre_activation)
        hidden.append(max(0.0, pre_activation))
    output = output_bias + sum(
        weight * value for weight, value in zip(hidden_output_weights, hidden, strict=True)
    )
    return output, pre_activations, hidden


def _train_pair(
    better: Sequence[float],
    worse: Sequence[float],
    *,
    input_hidden_weights: list[list[float]],
    hidden_bias: list[float],
    hidden_output_weights: list[float],
    output_bias: float,
    learning_rate: float,
    l2_penalty: float,
) -> tuple[float, float]:
    better_output, better_pre, better_hidden = _forward(
        better,
        input_hidden_weights=input_hidden_weights,
        hidden_bias=hidden_bias,
        hidden_output_weights=hidden_output_weights,
        output_bias=output_bias,
    )
    worse_output, worse_pre, worse_hidden = _forward(
        worse,
        input_hidden_weights=input_hidden_weights,
        hidden_bias=hidden_bias,
        hidden_output_weights=hidden_output_weights,
        output_bias=output_bias,
    )
    margin = better_output - worse_output
    loss = _softplus(margin)
    grad_margin = _sigmoid(margin)

    grad_hidden_output = [
        grad_margin * (better_value - worse_value)
        for better_value, worse_value in zip(better_hidden, worse_hidden, strict=True)
    ]
    grad_output_bias = 0.0
    old_hidden_output = list(hidden_output_weights)
    grad_input_hidden = [
        [0.0 for _ in range(len(better))] for _ in range(len(input_hidden_weights))
    ]
    grad_hidden_bias = [0.0 for _ in range(len(hidden_bias))]

    for output_grad, vector, pre_activations in (
        (grad_margin, better, better_pre),
        (-grad_margin, worse, worse_pre),
    ):
        grad_output_bias += output_grad
        for hidden_index, pre_activation in enumerate(pre_activations):
            if pre_activation <= 0.0:
                continue
            hidden_grad = output_grad * old_hidden_output[hidden_index]
            grad_hidden_bias[hidden_index] += hidden_grad
            for feature_index, value in enumerate(vector):
                grad_input_hidden[hidden_index][feature_index] += hidden_grad * value

    for hidden_index, row in enumerate(input_hidden_weights):
        for feature_index, weight in enumerate(row):
            row[feature_index] = weight - learning_rate * (
                grad_input_hidden[hidden_index][feature_index] + l2_penalty * weight
            )
        hidden_bias[hidden_index] -= learning_rate * grad_hidden_bias[hidden_index]
        hidden_output_weights[hidden_index] -= learning_rate * (
            grad_hidden_output[hidden_index] + l2_penalty * hidden_output_weights[hidden_index]
        )
    output_bias -= learning_rate * grad_output_bias
    return loss, output_bias


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _softplus(value: float) -> float:
    if value > 30:
        return value
    if value < -30:
        return math.exp(value)
    return math.log1p(math.exp(value))
