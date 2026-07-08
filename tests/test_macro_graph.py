from __future__ import annotations

import sympy as sp
from geml.compression.macro_expansions import (
    compute_pure_eml_node_count,
    expand_macro_graph,
    validate_expansion_against_official,
)
from geml.compression.macro_graph import build_macro_graph, refs_by_parent
from geml.compression.macro_metrics import compute_macro_graph_analysis
from geml.symbolic.official_eml_compiler import (
    compile_to_official_eml_subtree,
    emit_official_eml_string,
)


def test_exp_macro_graph_expands_to_official_pure_eml() -> None:
    x = sp.Symbol("x")
    assert_macro_expands_to_official(sp.exp(x, evaluate=False))


def test_log_macro_graph_expands_to_official_pure_eml() -> None:
    x = sp.Symbol("x")
    assert_macro_expands_to_official(sp.log(x, evaluate=False))


def test_add_macro_graph_expands_to_official_pure_eml() -> None:
    x, y = sp.symbols("x y")
    assert_macro_expands_to_official(sp.Add(x, y, evaluate=False))


def test_mul_macro_graph_expands_to_official_pure_eml() -> None:
    x, y = sp.symbols("x y")
    assert_macro_expands_to_official(sp.Mul(x, y, evaluate=False))


def test_pow_macro_graph_expands_to_official_pure_eml() -> None:
    x = sp.Symbol("x")
    assert_macro_expands_to_official(sp.Pow(x, 2, evaluate=False))


def test_no_macro_graph_is_labeled_as_pure_eml() -> None:
    x = sp.Symbol("x")
    graph = build_macro_graph(sp.Add(x, 1, evaluate=False))

    assert graph.representation_mode == "macro_graph_v1"
    assert graph.metadata["is_pure_eml"] is False
    assert all(node.metadata["is_pure_eml"] is False for node in graph.nodes)
    assert all(node.macro_name != "eml" for node in graph.nodes)


def test_repeated_child_references_are_preserved() -> None:
    x = sp.Symbol("x")
    graph = build_macro_graph(sp.Add(x, x, evaluate=False))
    root = graph.nodes[graph.root_id]
    child_refs = refs_by_parent(graph)[graph.root_id]

    assert root.macro_name == "eml_add"
    assert [ref.child_slot for ref in child_refs] == ["left", "right"]
    assert child_refs[0].child_id == child_refs[1].child_id
    assert graph.statistics.child_reference_count == 2


def test_macro_graph_metrics_do_not_exceed_expanded_pure_eml_tree_where_expected() -> None:
    x, y = sp.symbols("x y")
    examples = [
        sp.exp(x, evaluate=False),
        sp.log(x, evaluate=False),
        sp.Add(x, y, evaluate=False),
        sp.Mul(x, y, evaluate=False),
        sp.Pow(x, 2, evaluate=False),
    ]

    for expr in examples:
        analysis = compute_macro_graph_analysis(expr)
        expanded = expand_macro_graph(analysis.macro_graph)
        assert analysis.metrics.macro_graph_nodes <= compute_pure_eml_node_count(expanded)
        assert analysis.metrics.expansion_valid is True
        assert analysis.metrics.pure_eml_equivalent is True


def assert_macro_expands_to_official(expr: sp.Expr) -> None:
    graph = build_macro_graph(expr)
    expanded = expand_macro_graph(graph)
    official = compile_to_official_eml_subtree(expr)
    validation = validate_expansion_against_official(graph, expr)

    assert emit_official_eml_string(expanded) == emit_official_eml_string(official)
    assert validation.expansion_valid is True
    assert validation.pure_eml_equivalent is True
    assert graph.representation_mode == "macro_graph_v1"
