from __future__ import annotations

from pathlib import Path

import geml.egraph.costs as costs
from geml.egraph.egraph import EGraph
from geml.egraph.extractor import (
    ExtractionConfig,
    ExtractionResult,
    enumerate_candidates,
    extract_expression,
    extract_min_ast_size,
)
from geml.egraph.ir import Add, Const, Exp, Expr, Log, Mul, Sub, Var, display
from geml.egraph.rewrites import SaturationLimits, saturate
from geml.egraph.rule_sets import rules_for_mode
from geml.egraph.validation import positive_real_numeric_validation
from pytest import MonkeyPatch


def test_x_plus_two_minus_one_extracts_to_x_plus_one_equivalent_in_safe_mode() -> None:
    x = Var("x")
    original = Add(x, Sub(Const(2), Const(1)))
    egraph = EGraph()
    root_id = egraph.add_expr(original)
    saturate_safe(egraph)

    result = extract_expression(
        egraph,
        root_id,
        original_expression=original,
        config=exact_config(),
    )

    assert result.extraction_status == "completed"
    assert result.validation_status == "valid"
    assert result.expression is not None
    equivalent = positive_real_numeric_validation(result.expression, Add(x, Const(1)))
    assert equivalent.validation_status == "valid"


def test_x_times_one_extracts_to_x_in_safe_mode() -> None:
    original = Mul(Var("x"), Const(1))
    egraph = EGraph()
    root_id = egraph.add_expr(original)
    saturate_safe(egraph)

    result = extract_expression(
        egraph,
        root_id,
        original_expression=original,
        config=exact_config(),
    )

    assert result.extraction_status == "completed"
    assert result.extracted_expression == "x"
    assert result.validation_status == "valid"


def test_commutative_forms_extract_to_same_cost_minimal_form() -> None:
    x = Var("x")
    y = Var("y")
    egraph = EGraph()
    left_id = egraph.add_expr(Add(x, y))
    right_id = egraph.add_expr(Add(y, x))
    saturate_safe(egraph)

    left = extract_expression(egraph, left_id, original_expression=Add(x, y), config=exact_config())
    right = extract_expression(
        egraph,
        right_id,
        original_expression=Add(y, x),
        config=exact_config(),
    )

    assert left.extraction_status == "completed"
    assert right.extraction_status == "completed"
    assert left.extracted_expression == right.extracted_expression == "Add(x,y)"


def test_log_exp_extracts_to_x_only_in_positive_real_formal_mode() -> None:
    x = Var("x")
    original = Log(Exp(x))

    safe = extract_after_saturation(original, exact_config())
    positive = extract_after_saturation(original, positive_exact_config())

    assert safe.extraction_status == "completed"
    assert safe.extracted_expression != "x"
    assert positive.extraction_status == "completed"
    assert positive.extracted_expression == "x"
    assert positive.assumptions == "positive_real_formal"


def test_exp_log_extracts_to_x_only_in_positive_real_formal_mode() -> None:
    x = Var("x")
    original = Exp(Log(x))

    safe = extract_after_saturation(original, exact_config())
    positive = extract_after_saturation(original, positive_exact_config())

    assert safe.extraction_status == "completed"
    assert safe.extracted_expression != "x"
    assert positive.extraction_status == "completed"
    assert positive.extracted_expression == "x"


def test_extract_min_ast_size_survives_log_exp_inverse_cycle() -> None:
    egraph, root_id, _original = log_exp_inverse_cycle_egraph()

    result = extract_min_ast_size(egraph, root_id)

    assert result.status == "completed"
    assert result.cost == 4
    assert result.expression is not None
    assert display(result.expression) == "Add(x,Exp(x))"


def test_cyclic_egraph_candidate_enumeration_does_not_fail() -> None:
    egraph, root_id, _original = log_exp_inverse_cycle_egraph()

    candidates = enumerate_candidates(
        egraph,
        root_id,
        beam_size=16,
        max_depth=6,
        mode="exact_eml_dag_beam_cost",
        config=positive_exact_config(beam_size=16, max_candidate_depth=6),
    )

    assert candidates
    assert display(candidates[0].expression) == "Add(x,Exp(x))"


def test_cyclic_candidate_count_is_stable_across_repeated_calls() -> None:
    egraph, root_id, _original = log_exp_inverse_cycle_egraph()
    config = positive_exact_config(beam_size=16, max_candidate_depth=6)

    first = enumerate_candidates(
        egraph,
        root_id,
        beam_size=16,
        max_depth=6,
        mode="exact_eml_dag_beam_cost",
        config=config,
    )
    second = enumerate_candidates(
        egraph,
        root_id,
        beam_size=16,
        max_depth=6,
        mode="exact_eml_dag_beam_cost",
        config=config,
    )

    assert [display(candidate.expression) for candidate in first] == [
        display(candidate.expression) for candidate in second
    ]


def test_positive_real_formal_cycle_enumeration_is_not_silently_undercounted() -> None:
    egraph, root_id, _original = log_exp_inverse_cycle_egraph()

    candidates = enumerate_candidates(
        egraph,
        root_id,
        beam_size=16,
        max_depth=6,
        mode="exact_eml_dag_beam_cost",
        config=positive_exact_config(beam_size=16, max_candidate_depth=6),
    )
    expressions = {display(candidate.expression) for candidate in candidates}

    assert len(candidates) > 1
    assert "Add(Log(Exp(x)),Exp(x))" in expressions


def test_ast_node_cost_and_exact_eml_dag_cost_can_choose_different_candidates() -> None:
    x = Var("x")
    y = Var("y")
    original = Log(Mul(x, y))
    egraph = EGraph()
    root_id = egraph.add_expr(original)
    saturate_positive(egraph)

    ast_result = extract_expression(
        egraph,
        root_id,
        original_expression=original,
        config=ExtractionConfig(
            extractor_mode="ast_node_cost",
            beam_size=24,
            max_candidate_depth=6,
            max_candidates_evaluated=24,
            timeout_seconds=5,
            allow_positive_real_rules=True,
            rule_mode="positive_real_formal",
        ),
    )
    exact_result = extract_expression(
        egraph,
        root_id,
        original_expression=original,
        config=positive_exact_config(beam_size=24, max_candidate_depth=6),
    )

    assert ast_result.extraction_status == "completed"
    assert exact_result.extraction_status == "completed"
    assert ast_result.extracted_expression == "Log(Mul(x,y))"
    assert exact_result.extracted_expression in {"Add(Log(x),Log(y))", "Add(Log(y),Log(x))"}
    assert exact_result.extracted_eml_dag_nodes is not None
    assert ast_result.extracted_eml_dag_nodes is not None
    assert exact_result.extracted_eml_dag_nodes < ast_result.extracted_eml_dag_nodes


def test_exact_eml_dag_beam_cost_calls_official_compiler(monkeypatch: MonkeyPatch) -> None:
    calls = {"compiler": 0}
    original_compiler = costs.official_eml_compiler.sympy_to_official_eml_tree

    def wrapped_compiler(expr: object) -> object:
        calls["compiler"] += 1
        return original_compiler(expr)

    monkeypatch.setattr(
        costs.official_eml_compiler,
        "sympy_to_official_eml_tree",
        wrapped_compiler,
    )

    result = extract_after_saturation(Mul(Var("x"), Const(1)), exact_config())

    assert result.extraction_status == "completed"
    assert calls["compiler"] > 0


def test_exact_eml_dag_beam_cost_calls_goal3_dag_metrics(monkeypatch: MonkeyPatch) -> None:
    calls = {"tree_to_dag": 0}
    original_tree_to_dag = costs.dag_graph.tree_to_dag

    def wrapped_tree_to_dag(tree: object) -> object:
        calls["tree_to_dag"] += 1
        return original_tree_to_dag(tree)

    monkeypatch.setattr(costs.dag_graph, "tree_to_dag", wrapped_tree_to_dag)

    result = extract_after_saturation(Mul(Var("x"), Const(1)), exact_config())

    assert result.extraction_status == "completed"
    assert calls["tree_to_dag"] > 0


def test_extracted_exact_result_preserves_pure_eml_integrity() -> None:
    result = extract_after_saturation(Add(Var("x"), Const(0)), exact_config())

    assert result.extraction_status == "completed"
    assert result.integrity_valid is True
    assert result.integrity_errors == ()


def test_timeout_returns_status_row_not_crash() -> None:
    original = Add(Var("x"), Var("y"))
    egraph = EGraph()
    root_id = egraph.add_expr(original)

    result = extract_expression(
        egraph,
        root_id,
        original_expression=original,
        config=ExtractionConfig(
            extractor_mode="exact_eml_dag_beam_cost",
            beam_size=8,
            max_candidate_depth=4,
            max_candidates_evaluated=8,
            timeout_seconds=1e-12,
        ),
    )

    assert result.extraction_status == "timeout"
    assert result.extraction_timeout is True


def test_cyclic_timeout_returns_status_row_not_crash() -> None:
    egraph, root_id, original = log_exp_inverse_cycle_egraph()

    result = extract_expression(
        egraph,
        root_id,
        original_expression=original,
        config=ExtractionConfig(
            extractor_mode="exact_eml_dag_beam_cost",
            beam_size=16,
            max_candidate_depth=8,
            max_candidates_evaluated=16,
            timeout_seconds=1e-12,
            allow_positive_real_rules=True,
            rule_mode="positive_real_formal",
        ),
    )

    assert result.extraction_status == "timeout"
    assert result.extraction_timeout is True


def test_extractor_and_costs_do_not_use_sympy_simplify_as_extractor() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in [
        repo_root / "geml/egraph/extractor.py",
        repo_root / "geml/egraph/costs.py",
    ]:
        source = path.read_text(encoding="utf-8")
        assert ".simplify" not in source
        assert "simplify(" not in source


def extract_after_saturation(original: Expr, config: ExtractionConfig) -> ExtractionResult:
    egraph = EGraph()
    root_id = egraph.add_expr(original)
    if config.rule_mode == "positive_real_formal":
        saturate_positive(egraph)
    else:
        saturate_safe(egraph)
    return extract_expression(egraph, root_id, original_expression=original, config=config)


def log_exp_inverse_cycle_egraph() -> tuple[EGraph, int, Expr]:
    x = Var("x")
    original = Add(x, Exp(x))
    egraph = EGraph()
    root_id = egraph.add_expr(original)
    x_id = egraph.add_expr(x)
    log_exp_x_id = egraph.add_expr(Log(Exp(x)))
    egraph.union(x_id, log_exp_x_id)
    egraph.rebuild()
    return egraph, root_id, original


def saturate_safe(egraph: EGraph) -> None:
    saturate(
        egraph,
        rules_for_mode("safe"),
        limits=SaturationLimits(
            max_iterations=10,
            max_enodes=10_000,
            max_eclasses=10_000,
            timeout_seconds=5,
        ),
    )


def saturate_positive(egraph: EGraph) -> None:
    saturate(
        egraph,
        rules_for_mode("positive_real_formal"),
        limits=SaturationLimits(
            max_iterations=5,
            max_enodes=20_000,
            max_eclasses=20_000,
            timeout_seconds=5,
        ),
    )


def exact_config() -> ExtractionConfig:
    return ExtractionConfig(
        extractor_mode="exact_eml_dag_beam_cost",
        beam_size=16,
        max_candidate_depth=6,
        max_candidates_evaluated=16,
        timeout_seconds=5,
    )


def positive_exact_config(
    *,
    beam_size: int = 16,
    max_candidate_depth: int = 6,
) -> ExtractionConfig:
    return ExtractionConfig(
        extractor_mode="exact_eml_dag_beam_cost",
        beam_size=beam_size,
        max_candidate_depth=max_candidate_depth,
        max_candidates_evaluated=beam_size,
        timeout_seconds=5,
        allow_positive_real_rules=True,
        rule_mode="positive_real_formal",
    )
