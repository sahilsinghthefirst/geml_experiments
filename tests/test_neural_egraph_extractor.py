"""Tests for Goal 5.4 neural e-graph extraction components."""

from __future__ import annotations

from pathlib import Path

import geml.compression.egraph_candidate_dataset as candidate_dataset
from geml.compression.egraph_candidate_dataset import (
    CandidateGenerationConfig,
    EgraphCandidateRecord,
    build_candidate_records_for_ir,
)
from geml.compression.neural_cost_model import (
    NeuralCostModelConfig,
    train_neural_cost_model,
)
from geml.compression.neural_egraph_extractor import evaluate_candidate_group
from geml.data.dataset import GeneratedExpressionInput
from geml.egraph.ir import Const, Mul, Var
from geml.experiments.egraph_compression_study import Goal3BaselineRow
from pytest import MonkeyPatch


def test_candidate_labels_come_from_official_eml_dag_metrics(
    monkeypatch: MonkeyPatch,
) -> None:
    calls = {"exact_cost": 0}
    original = candidate_dataset.exact_eml_dag_cost

    def wrapped_exact_cost(expr: object) -> object:
        calls["exact_cost"] += 1
        return original(expr)

    monkeypatch.setattr(candidate_dataset, "exact_eml_dag_cost", wrapped_exact_cost)
    records = _small_candidate_records(split="train")

    assert records
    assert calls["exact_cost"] == len(records)
    assert all(record.official_eml_compiled for record in records)
    assert all(record.true_official_eml_dag_nodes is not None for record in records)


def test_no_sympy_simplify_shortcut_is_used() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in [
        repo_root / "geml/compression/egraph_candidate_dataset.py",
        repo_root / "geml/compression/neural_cost_model.py",
        repo_root / "geml/compression/neural_egraph_extractor.py",
    ]:
        source = path.read_text(encoding="utf-8")
        assert ".simplify" not in source
        assert "simplify(" not in source


def test_model_training_is_reproducible_with_seed() -> None:
    records = _synthetic_records()
    config = NeuralCostModelConfig(seed=7, hidden_size=4, epochs=3, learning_rate=0.02)

    left = train_neural_cost_model(records, config=config)
    right = train_neural_cost_model(records, config=config)

    assert left.model.model_dump(mode="json") == right.model.model_dump(mode="json")
    assert left.train_log["train_pair_count"] == right.train_log["train_pair_count"]


def test_neural_selected_candidate_still_validates() -> None:
    records = _small_candidate_records(split="train")
    train_result = train_neural_cost_model(
        records,
        config=NeuralCostModelConfig(seed=0, hidden_size=4, epochs=1),
    )

    row = evaluate_candidate_group(records, model=train_result.model)

    assert row.extraction_status == "completed"
    assert row.neural_validation_status == "valid"
    assert row.neural_same_root_eclass is True
    assert row.neural_structural_purity_valid is True
    assert row.neural_official_eml_compiled is True


def test_evaluation_computes_regret_correctly() -> None:
    records = _synthetic_records()

    class SelectSecond:
        def predict_feature_dict(self, features: dict[str, float]) -> float:
            return features["choose_second"]

    row = evaluate_candidate_group(records[:2], model=SelectSecond())

    assert row.exact_best_eml_dag_nodes == 5
    assert row.neural_eml_dag_nodes == 8
    assert row.neural_regret_vs_exact_best == 3
    assert row.neural_matches_exact_best is False


def test_no_final_symbolic_reasoning_gnn_is_trained() -> None:
    result = train_neural_cost_model(
        _synthetic_records(),
        config=NeuralCostModelConfig(seed=1, hidden_size=4, epochs=1),
    )

    assert result.model.trained_final_reasoning_gnn is False
    assert result.train_log["trained_final_reasoning_gnn"] is False


def _small_candidate_records(*, split: str) -> tuple[EgraphCandidateRecord, ...]:
    original = Mul(Var("x"), Const(1))
    return build_candidate_records_for_ir(
        original,
        input_row=GeneratedExpressionInput(
            index=0,
            expression="x*1",
            srepr="Mul(Symbol('x'), Integer(1))",
            metadata={"nontriviality": {"mul_by_one_count": 1}},
        ),
        baseline=_baseline(),
        rule_mode="safe",
        config=CandidateGenerationConfig(
            count=1,
            run_modes=("safe",),
            max_iterations=3,
            beam_size=6,
            max_candidate_depth=5,
            max_candidates_evaluated=6,
            saturation_timeout_seconds=5,
        ),
        split=split,  # type: ignore[arg-type]
    )


def _baseline() -> Goal3BaselineRow:
    return Goal3BaselineRow(
        index=0,
        expression="x*1",
        srepr="Mul(Symbol('x'), Integer(1))",
        ast_tree_node_count=3,
        ast_dag_node_count=3,
        eml_tree_node_count=19,
        eml_dag_node_count=19,
        tree_alpha=19 / 3,
        dag_alpha_vs_ast_tree=19 / 3,
        dag_alpha_vs_ast_dag=19 / 3,
        alpha_threshold_current=4.0,
        below_threshold_dag_vs_ast_tree=False,
    )


def _synthetic_records() -> tuple[EgraphCandidateRecord, ...]:
    return (
        _record("a", cost=5, choose_second=1.0, split="train"),
        _record("b", cost=8, choose_second=0.0, split="train"),
        _record("c", expression_id=1, cost=4, choose_second=1.0, split="train"),
        _record("d", expression_id=1, cost=9, choose_second=0.0, split="train"),
    )


def _record(
    suffix: str,
    *,
    cost: int,
    choose_second: float,
    split: str,
    expression_id: int = 0,
) -> EgraphCandidateRecord:
    return EgraphCandidateRecord(
        expression_id=expression_id,
        candidate_id=f"{expression_id}:safe:{suffix}",
        candidate_rank=1,
        original_expression="x",
        original_srepr="Symbol('x')",
        candidate_expression=suffix,
        candidate_srepr="Symbol('x')",
        candidate_ir_features={
            "choose_second": choose_second,
            "candidate_ast_node_cost": float(cost),
            "candidate_estimated_eml_cost": float(cost),
        },
        true_official_eml_dag_nodes=cost,
        true_official_eml_tree_nodes=cost,
        true_ast_tree_nodes=1,
        true_ast_dag_nodes=1,
        source_ast_node_count=1,
        source_ast_dag_node_count=1,
        original_eml_dag_nodes=10,
        compression_gain_vs_goal3_dag=10 / cost,
        rule_mode="safe",
        subset_label="nontrivial_v1",
        split=split,  # type: ignore[arg-type]
        saturation_status="saturated",
        eclass_count=1,
        enode_count=1,
        exact_label_runtime_seconds=0.01,
        baseline_estimated_eml_cost=cost,
        baseline_ast_node_cost=cost,
        same_root_eclass=True,
        semantic_validation_status="valid",
        structural_purity_valid=True,
        official_eml_compiled=True,
        error=None,
    )
