"""Expansion helpers from macro graphs back into official pure EML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp
from pydantic import BaseModel

from geml.compression.macro_graph import MacroGraph, MacroNode, refs_by_parent
from geml.symbolic import official_eml_compiler as _official
from geml.symbolic.ast_graph import sympy_to_ast_tree
from geml.symbolic.eml_nodes import EmlEdge, EmlNode, EmlTree, MetadataValue
from geml.symbolic.metrics import compute_tree_statistics
from geml.symbolic.official_eml_compiler import (
    PureEmlNode,
    compile_to_official_eml_subtree,
    emit_official_eml_string,
)

type PureEmlSignature = tuple[Any, ...]


class MacroExpansionValidation(BaseModel):
    """Strict expansion validation against the official pure EML compiler."""

    expansion_valid: bool
    pure_eml_equivalent: bool
    expanded_official_string: str | None = None
    official_compiler_string: str | None = None
    error: str | None = None


def expand_macro_graph(graph: MacroGraph) -> PureEmlNode:
    """Expand a macro graph root into an official pure EML subtree."""
    return expand_macro_graph_node(graph, graph.root_id)


def expand_macro_graph_node(graph: MacroGraph, node_id: int) -> PureEmlNode:
    """Expand a macro graph node into an official pure EML subtree."""
    nodes_by_id = {node.id: node for node in graph.nodes}
    if node_id not in nodes_by_id:
        raise ValueError(f"macro node {node_id} is not present in graph")
    refs = refs_by_parent(graph)

    def expand(current_id: int) -> PureEmlNode:
        node = nodes_by_id[current_id]
        children = [expand(ref.child_id) for ref in refs.get(current_id, [])]
        return _expand_node(node, children)

    return expand(node_id)


def expand_macro_graph_to_eml_tree(
    graph: MacroGraph,
    *,
    source_expr: sp.Expr | None = None,
) -> EmlTree:
    """Materialize a macro graph expansion as a restricted pure EML tree."""
    subtree = expand_macro_graph(graph)
    return materialize_pure_eml_subtree(
        subtree,
        source_expr=source_expr if source_expr is not None else _source_expr_from_graph(graph),
        source_metadata={
            "converter": "macro_graph_v1_expansion",
            "representation_mode": "restricted_eml_pure",
            "source_macro_representation_mode": graph.representation_mode,
            "macro_graph_is_pure_eml": False,
            "macro_graph_node_count": graph.statistics.node_count,
        },
    )


def validate_expansion_against_official(
    graph: MacroGraph,
    expr: sp.Expr | str | int | float,
) -> MacroExpansionValidation:
    """Check that macro graph expansion matches official pure EML structurally."""
    try:
        sympy_expr = sp.sympify(expr)
        expanded = expand_macro_graph(graph)
        official = compile_to_official_eml_subtree(sympy_expr)
        expanded_string = emit_official_eml_string(expanded)
        official_string = emit_official_eml_string(official)
        equivalent = expanded_string == official_string
        return MacroExpansionValidation(
            expansion_valid=equivalent,
            pure_eml_equivalent=equivalent,
            expanded_official_string=expanded_string,
            official_compiler_string=official_string,
            error=None if equivalent else "expanded macro graph differs from official compiler",
        )
    except Exception as exc:
        return MacroExpansionValidation(
            expansion_valid=False,
            pure_eml_equivalent=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def compute_pure_eml_node_count(node: PureEmlNode) -> int:
    """Count tree nodes in an expanded pure EML subtree."""
    return 1 + sum(compute_pure_eml_node_count(child) for child in node.children)


def compute_pure_eml_dag_node_count(node: PureEmlNode) -> int:
    """Count exact structural DAG nodes in an expanded pure EML subtree."""
    signatures: set[PureEmlSignature] = set()

    def visit(current: PureEmlNode) -> PureEmlSignature:
        signature = pure_eml_structural_signature(current)
        signatures.add(signature)
        for child in current.children:
            visit(child)
        return signature

    visit(node)
    return len(signatures)


def pure_eml_structural_signature(node: PureEmlNode) -> PureEmlSignature:
    """Return an exact structural signature for a pure EML subtree."""
    return (
        node.kind,
        node.label,
        tuple(pure_eml_structural_signature(child) for child in node.children),
    )


def materialize_pure_eml_subtree(
    subtree: PureEmlNode,
    *,
    source_expr: sp.Expr,
    source_metadata: dict[str, MetadataValue] | None = None,
) -> EmlTree:
    """Materialize a pure EML subtree using the same tree shape as the compiler."""
    builder = _PureEmlTreeMaterializer()
    root_id = builder.materialize(subtree)
    edge_pairs = [(edge.source, edge.target) for edge in builder.edges]
    operator_node_ids = [node.id for node in builder.nodes if node.kind == "eml"]
    statistics = compute_tree_statistics(
        root_id=root_id,
        node_ids=[node.id for node in builder.nodes],
        edges=edge_pairs,
        operator_node_ids=operator_node_ids,
    )
    ast_tree = sympy_to_ast_tree(source_expr)
    alpha = statistics.node_count / ast_tree.statistics.node_count
    metadata: dict[str, MetadataValue] = {
        "converter": "macro_graph_v1_expansion",
        "representation_mode": "restricted_eml_pure",
        "expression": str(source_expr),
        "srepr": sp.srepr(source_expr),
        "alpha_policy": "alpha is valid only for pure trees with no derived leaves",
        "alpha_valid": True,
    }
    if source_metadata:
        metadata.update(source_metadata)
    return EmlTree(
        representation_mode="restricted_eml_pure",
        nodes=builder.nodes,
        edges=builder.edges,
        root_id=root_id,
        node_labels={node.id: node.label for node in builder.nodes},
        metadata=metadata,
        statistics=statistics,
        normal_leaf_count=statistics.leaf_count,
        derived_leaf_count=0,
        hidden_compound_leaf_count=0,
        ast_statistics=ast_tree.statistics,
        alpha=alpha,
        alpha_valid=True,
    )


def _expand_node(node: MacroNode, children: list[PureEmlNode]) -> PureEmlNode:
    name = node.macro_name
    if len(children) != node.arity:
        raise ValueError(f"macro {node.id} expected {node.arity} children, got {len(children)}")
    if name == "eml_variable":
        symbol_name = node.metadata.get("symbol_name")
        if not isinstance(symbol_name, str):
            raise ValueError(f"eml_variable node {node.id} is missing symbol_name")
        return _official.eml_variable(symbol_name)
    if name == "eml_integer":
        return _official.eml_int(_int_metadata(node, "value"))
    if name == "eml_rational":
        return _official.eml_rational(
            _int_metadata(node, "numerator"),
            _int_metadata(node, "denominator"),
        )
    if name == "eml_const":
        return _expand_const(node)
    if name == "eml_zero":
        return _official.eml_zero()
    if name == "eml_exp":
        return _official.eml_exp(children[0])
    if name == "eml_log":
        return _official.eml_log(children[0])
    if name == "eml_neg":
        return _official.eml_neg(children[0])
    if name == "eml_inv":
        return _official.eml_inv(children[0])
    if name == "eml_add":
        return _official.eml_add(children[0], children[1])
    if name == "eml_mul":
        return _official.eml_mul(children[0], children[1])
    if name == "eml_sub":
        return _official.eml_sub(children[0], children[1])
    if name == "eml_div":
        return _official.eml_div(children[0], children[1])
    if name == "eml_pow":
        return _official.eml_pow(children[0], children[1])
    raise ValueError(f"unknown macro name {name!r}")


def _expand_const(node: MacroNode) -> PureEmlNode:
    value = node.metadata.get("value")
    if isinstance(value, int):
        return _official.eml_int(value)
    if isinstance(value, float):
        rational = sp.Rational(str(value))
        return _official.eml_rational(int(rational.p), int(rational.q))
    numerator = node.metadata.get("numerator")
    denominator = node.metadata.get("denominator")
    if isinstance(numerator, int) and isinstance(denominator, int):
        return _official.eml_rational(numerator, denominator)
    raise ValueError(f"eml_const node {node.id} has no supported constant metadata")


def _int_metadata(node: MacroNode, key: str) -> int:
    value = node.metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"macro node {node.id} metadata {key!r} must be an integer")
    return value


def _source_expr_from_graph(graph: MacroGraph) -> sp.Expr:
    source_srepr = graph.metadata.get("source_srepr")
    if isinstance(source_srepr, str):
        from geml.symbolic.srepr import parse_srepr

        return parse_srepr(source_srepr)
    source_expression = graph.metadata.get("source_expression")
    if isinstance(source_expression, str):
        return sp.sympify(source_expression)
    raise ValueError("macro graph metadata is missing source expression")


@dataclass
class _PureEmlTreeMaterializer:
    nodes: list[EmlNode]
    edges: list[EmlEdge]
    next_node_id: int

    def __init__(self) -> None:
        self.nodes = []
        self.edges = []
        self.next_node_id = 0

    def materialize(self, subtree: PureEmlNode) -> int:
        if subtree.kind == "constant":
            metadata: dict[str, MetadataValue] = {
                "expression": "1",
                "sympy_func": "One",
                "value": 1,
            }
        elif subtree.kind == "variable":
            metadata = {"expression": subtree.label, "sympy_func": "Symbol"}
        else:
            metadata = {"expression": "exp(left) - log(right)", "arity": 2}

        node_id = self.next_node_id
        self.next_node_id += 1
        self.nodes.append(
            EmlNode(
                id=node_id,
                label=subtree.label,
                kind=subtree.kind,
                metadata=metadata,
            )
        )
        for position, child in enumerate(subtree.children):
            child_id = self.materialize(child)
            self.edges.append(EmlEdge(source=node_id, target=child_id, position=position))
        return node_id
