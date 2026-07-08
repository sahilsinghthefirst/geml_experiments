"""Transparent macro graph representation for official EML compiler macros.

Macro graphs are not pure EML. They are compressed graph records whose nodes
name official compiler macros and whose expansion rules point back to the
official pure EML compiler implementation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import sympy as sp
from pydantic import BaseModel, Field

from geml.symbolic import official_eml_compiler as _official

type MacroRepresentationMode = Literal["macro_graph_v1"]
type MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]
type MacroStructuralSignature = tuple[Any, ...]

MACRO_REPRESENTATION_MODE: MacroRepresentationMode = "macro_graph_v1"
MACRO_ARITIES: dict[str, int] = {
    "eml_variable": 0,
    "eml_integer": 0,
    "eml_rational": 0,
    "eml_const": 0,
    "eml_zero": 0,
    "eml_exp": 1,
    "eml_log": 1,
    "eml_neg": 1,
    "eml_inv": 1,
    "eml_add": 2,
    "eml_mul": 2,
    "eml_sub": 2,
    "eml_div": 2,
    "eml_pow": 2,
}
MACRO_INPUT_SLOTS: dict[str, tuple[str, ...]] = {
    "eml_variable": (),
    "eml_integer": (),
    "eml_rational": (),
    "eml_const": (),
    "eml_zero": (),
    "eml_exp": ("value",),
    "eml_log": ("value",),
    "eml_neg": ("value",),
    "eml_inv": ("value",),
    "eml_add": ("left", "right"),
    "eml_mul": ("left", "right"),
    "eml_sub": ("left", "right"),
    "eml_div": ("numerator", "denominator"),
    "eml_pow": ("base", "exponent"),
}
MACRO_EXPANSION_RULES: dict[str, str] = {
    "eml_variable": "official_eml_compiler.eml_variable",
    "eml_integer": "official_eml_compiler.eml_int",
    "eml_rational": "official_eml_compiler.eml_rational",
    "eml_const": "official_eml_compiler.eml_int_or_eml_rational",
    "eml_zero": "official_eml_compiler.eml_zero",
    "eml_exp": "official_eml_compiler.eml_exp",
    "eml_log": "official_eml_compiler.eml_log",
    "eml_neg": "official_eml_compiler.eml_neg",
    "eml_inv": "official_eml_compiler.eml_inv",
    "eml_add": "official_eml_compiler.eml_add",
    "eml_mul": "official_eml_compiler.eml_mul",
    "eml_sub": "official_eml_compiler.eml_sub",
    "eml_div": "official_eml_compiler.eml_div",
    "eml_pow": "official_eml_compiler.eml_pow",
}


class MacroGraphError(ValueError):
    """Raised when a source expression cannot be represented as a macro graph."""


class MacroNode(BaseModel):
    """One transparent official compiler macro node."""

    id: int = Field(ge=0)
    macro_name: str
    arity: int = Field(ge=0)
    input_slots: tuple[str, ...] = Field(default_factory=tuple)
    source_subtree_id: int = Field(ge=0)
    source_expression: str
    source_srepr: str
    expansion_rule_name: str
    expansion_to_pure_eml_available: bool = True
    pure_eml_expansion_node_count: int = Field(ge=0)
    pure_eml_expansion_dag_node_count: int = Field(ge=0)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class MacroChildRef(BaseModel):
    """An ordered parent-to-child reference in a macro graph."""

    parent_id: int = Field(ge=0)
    child_id: int = Field(ge=0)
    child_slot: str
    slot_index: int = Field(ge=0)


class MacroGraphStatistics(BaseModel):
    """Macro graph statistics, separate from pure EML-DAG statistics."""

    node_count: int = Field(ge=0)
    child_reference_count: int = Field(ge=0)
    depth: int = Field(ge=0)
    leaf_count: int = Field(ge=0)
    shared_node_count: int = Field(ge=0)


class MacroGraph(BaseModel):
    """Serializable macro graph for official compiler concepts."""

    representation_mode: MacroRepresentationMode = MACRO_REPRESENTATION_MODE
    nodes: list[MacroNode]
    child_refs: list[MacroChildRef]
    root_id: int
    node_macro_names: dict[int, str]
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    statistics: MacroGraphStatistics


@dataclass(frozen=True)
class _MacroBuildResult:
    node_id: int
    signature: MacroStructuralSignature


class _MacroGraphBuilder:
    def __init__(self, source_expr: sp.Expr) -> None:
        self.source_expr = source_expr
        self.nodes: list[MacroNode] = []
        self.child_refs: list[MacroChildRef] = []
        self._signature_to_id: dict[MacroStructuralSignature, int] = {}
        self._next_source_subtree_id = 0

    def build(self) -> MacroGraph:
        root = self._compile_expr(self.source_expr)
        statistics = compute_macro_graph_statistics(
            root_id=root.node_id,
            node_ids=[node.id for node in self.nodes],
            child_refs=self.child_refs,
            source_subtree_ids_by_node_id={
                node.id: _source_subtree_ids(node) for node in self.nodes
            },
        )
        graph = MacroGraph(
            nodes=self.nodes,
            child_refs=self.child_refs,
            root_id=root.node_id,
            node_macro_names={node.id: node.macro_name for node in self.nodes},
            metadata={
                "representation_mode": MACRO_REPRESENTATION_MODE,
                "is_pure_eml": False,
                "converter": "official_macro_graph_v1",
                "source_expression": str(self.source_expr),
                "source_srepr": sp.srepr(self.source_expr),
                "official_compiler": "official_recursive_eml_compiler_v4_port",
                "expansion_policy": "each macro node expands through official compiler macros",
                "macro_graph_size_policy": (
                    "macro graph node counts are not pure EML alpha counts"
                ),
                "sharing_policy": "exact macro structural subtree signature equality",
            },
            statistics=statistics,
        )
        _populate_pure_eml_expansion_counts(graph)
        validate_macro_graph(graph)
        return graph

    def _compile_expr(self, expr: sp.Expr) -> _MacroBuildResult:
        source_subtree_id = self._next_source_id()
        if expr.is_Atom:
            return self._compile_atom(expr, source_subtree_id)

        if expr.func == sp.exp and len(expr.args) == 1:
            child = self._compile_expr(expr.args[0])
            return self._add_macro("eml_exp", (child,), expr, source_subtree_id)

        if expr.func == sp.log and len(expr.args) == 1:
            child = self._compile_expr(expr.args[0])
            return self._add_macro("eml_log", (child,), expr, source_subtree_id)

        if expr.func == sp.Pow and len(expr.args) == 2:
            base, power = expr.as_base_exp()
            children = (self._compile_expr(base), self._compile_expr(power))
            return self._add_macro("eml_pow", children, expr, source_subtree_id)

        if expr.func == sp.Mul:
            factors = list(expr.args)
            if not factors:
                raise MacroGraphError("Mul requires at least one factor")
            division = _official._binary_division_factors(factors)
            if division is not None:
                numerator, denominator = division
                children = (
                    self._compile_expr(numerator),
                    self._compile_expr(denominator),
                )
                return self._add_macro("eml_div", children, expr, source_subtree_id)

            acc = self._compile_expr(factors[0])
            for factor in factors[1:]:
                acc = self._add_macro(
                    "eml_mul",
                    (acc, self._compile_expr(factor)),
                    expr,
                    source_subtree_id,
                )
            return acc

        if expr.func == sp.Add:
            terms = _official._ordered_add_terms(expr)
            if not terms:
                raise MacroGraphError("Add requires at least one term")
            subtraction = _official._binary_subtraction_terms(terms)
            if subtraction is not None:
                left, right = subtraction
                children = (self._compile_expr(left), self._compile_expr(right))
                return self._add_macro("eml_sub", children, expr, source_subtree_id)

            acc = self._compile_expr(terms[0])
            for term in terms[1:]:
                acc = self._add_macro(
                    "eml_add",
                    (acc, self._compile_expr(term)),
                    expr,
                    source_subtree_id,
                )
            return acc

        raise MacroGraphError(f"unsupported SymPy expression node {expr.func.__name__}: {expr}")

    def _compile_atom(self, expr: sp.Expr, source_subtree_id: int) -> _MacroBuildResult:
        if isinstance(expr, sp.Integer):
            if int(expr) == 0:
                return self._add_macro(
                    "eml_zero",
                    (),
                    expr,
                    source_subtree_id,
                    metadata={"value": 0, "numeric_kind": "integer"},
                )
            return self._add_macro(
                "eml_integer",
                (),
                expr,
                source_subtree_id,
                metadata={"value": int(expr), "numeric_kind": "integer"},
            )
        if isinstance(expr, sp.Rational):
            return self._add_macro(
                "eml_rational",
                (),
                expr,
                source_subtree_id,
                metadata={
                    "numerator": int(expr.p),
                    "denominator": int(expr.q),
                    "numeric_kind": "rational",
                },
            )
        if isinstance(expr, sp.Float):
            rational = sp.Rational(str(expr))
            return self._add_macro(
                "eml_rational",
                (),
                expr,
                source_subtree_id,
                metadata={
                    "numerator": int(rational.p),
                    "denominator": int(rational.q),
                    "numeric_kind": "float_as_rational",
                    "source_float": float(expr),
                },
            )
        if isinstance(expr, sp.Symbol):
            return self._add_macro(
                "eml_variable",
                (),
                expr,
                source_subtree_id,
                metadata={"symbol_name": expr.name},
            )
        raise MacroGraphError(f"unsupported atomic expression: {expr!r}")

    def _add_macro(
        self,
        macro_name: str,
        children: Sequence[_MacroBuildResult],
        expr: sp.Expr,
        source_subtree_id: int,
        *,
        metadata: dict[str, MetadataValue] | None = None,
    ) -> _MacroBuildResult:
        if macro_name not in MACRO_ARITIES:
            raise MacroGraphError(f"unknown macro name {macro_name!r}")
        input_slots = MACRO_INPUT_SLOTS[macro_name]
        if len(children) != len(input_slots):
            raise MacroGraphError(
                f"macro {macro_name} expected {len(input_slots)} children, got {len(children)}"
            )
        child_signatures = tuple(
            (slot_index, slot_name, child.signature, child.node_id)
            for slot_index, (slot_name, child) in enumerate(zip(input_slots, children, strict=True))
        )
        node_metadata = dict(metadata or {})
        signature = _macro_signature(macro_name, child_signatures, node_metadata)
        existing_node_id = self._signature_to_id.get(signature)
        if existing_node_id is not None:
            self._append_source_subtree_id(existing_node_id, source_subtree_id)
            return _MacroBuildResult(node_id=existing_node_id, signature=signature)

        node_id = len(self.nodes)
        self._signature_to_id[signature] = node_id
        self.nodes.append(
            MacroNode(
                id=node_id,
                macro_name=macro_name,
                arity=MACRO_ARITIES[macro_name],
                input_slots=input_slots,
                source_subtree_id=source_subtree_id,
                source_expression=str(expr),
                source_srepr=sp.srepr(expr),
                expansion_rule_name=MACRO_EXPANSION_RULES[macro_name],
                expansion_to_pure_eml_available=True,
                pure_eml_expansion_node_count=0,
                pure_eml_expansion_dag_node_count=0,
                metadata={
                    **node_metadata,
                    "source_subtree_ids": [source_subtree_id],
                    "is_pure_eml": False,
                },
            )
        )
        for slot_index, (slot_name, child) in enumerate(zip(input_slots, children, strict=True)):
            self.child_refs.append(
                MacroChildRef(
                    parent_id=node_id,
                    child_id=child.node_id,
                    child_slot=slot_name,
                    slot_index=slot_index,
                )
            )
        return _MacroBuildResult(node_id=node_id, signature=signature)

    def _append_source_subtree_id(self, node_id: int, source_subtree_id: int) -> None:
        source_ids = self.nodes[node_id].metadata.setdefault("source_subtree_ids", [])
        if not isinstance(source_ids, list):
            raise TypeError("source_subtree_ids metadata must be a list")
        source_ids.append(source_subtree_id)

    def _next_source_id(self) -> int:
        source_id = self._next_source_subtree_id
        self._next_source_subtree_id += 1
        return source_id


def build_macro_graph(expr: sp.Expr | str | int | float) -> MacroGraph:
    """Build a transparent macro graph from a source expression."""
    return _MacroGraphBuilder(sp.sympify(expr)).build()


def validate_macro_graph(graph: MacroGraph) -> None:
    """Validate macro graph integrity and ordered child-slot preservation."""
    if graph.representation_mode != MACRO_REPRESENTATION_MODE:
        raise ValueError(f"macro graph must use representation_mode={MACRO_REPRESENTATION_MODE!r}")
    if graph.metadata.get("is_pure_eml") is True:
        raise ValueError("macro graph must not be labeled as pure EML")
    node_ids = [node.id for node in graph.nodes]
    computed = compute_macro_graph_statistics(
        root_id=graph.root_id,
        node_ids=node_ids,
        child_refs=graph.child_refs,
        source_subtree_ids_by_node_id={node.id: _source_subtree_ids(node) for node in graph.nodes},
    )
    if computed != graph.statistics:
        raise ValueError("macro graph statistics do not match graph structure")
    nodes_by_id = {node.id: node for node in graph.nodes}
    if len(nodes_by_id) != len(graph.nodes):
        raise ValueError("macro node ids must be unique")
    for node in graph.nodes:
        expected_arity = MACRO_ARITIES.get(node.macro_name)
        if expected_arity is None:
            raise ValueError(f"unknown macro node {node.macro_name!r}")
        if node.arity != expected_arity:
            raise ValueError(f"macro {node.id} arity mismatch")
        expected_slots = MACRO_INPUT_SLOTS[node.macro_name]
        if tuple(node.input_slots) != expected_slots:
            raise ValueError(f"macro {node.id} slot names mismatch")
        if node.expansion_rule_name != MACRO_EXPANSION_RULES[node.macro_name]:
            raise ValueError(f"macro {node.id} expansion rule mismatch")
        if not node.expansion_to_pure_eml_available:
            raise ValueError(f"macro {node.id} has no pure EML expansion")
        if node.macro_name == "eml":
            raise ValueError("macro graph nodes must not be pure EML primitive nodes")

    refs_by_parent: dict[int, list[MacroChildRef]] = defaultdict(list)
    for ref in graph.child_refs:
        refs_by_parent[ref.parent_id].append(ref)
    for node in graph.nodes:
        refs = sorted(refs_by_parent.get(node.id, []), key=lambda ref: ref.slot_index)
        if len(refs) != node.arity:
            raise ValueError(f"macro {node.id} expected {node.arity} child refs")
        for slot_index, ref in enumerate(refs):
            if ref.slot_index != slot_index:
                raise ValueError(f"macro {node.id} has non-contiguous child refs")
            if ref.child_slot != node.input_slots[slot_index]:
                raise ValueError(f"macro {node.id} child slot mismatch")


def compute_macro_graph_statistics(
    *,
    root_id: int,
    node_ids: Sequence[int],
    child_refs: Sequence[MacroChildRef],
    source_subtree_ids_by_node_id: dict[int, list[int]] | None = None,
) -> MacroGraphStatistics:
    """Compute macro graph statistics and validate rooted DAG invariants."""
    node_id_set = set(node_ids)
    if len(node_id_set) != len(node_ids):
        raise ValueError("macro node ids must be unique")
    if root_id not in node_id_set:
        raise ValueError(f"root_id {root_id} is not present in macro graph")

    refs_by_parent: dict[int, list[MacroChildRef]] = defaultdict(list)
    incoming_counts: dict[int, int] = {node_id: 0 for node_id in node_id_set}
    for ref in child_refs:
        if ref.parent_id not in node_id_set:
            raise ValueError(f"child ref parent {ref.parent_id} is not present")
        if ref.child_id not in node_id_set:
            raise ValueError(f"child ref child {ref.child_id} is not present")
        refs_by_parent[ref.parent_id].append(ref)
        incoming_counts[ref.child_id] += 1

    for parent_id, refs in refs_by_parent.items():
        _validate_macro_child_refs(parent_id, refs)

    visited: set[int] = set()
    active: set[int] = set()

    def visit(node_id: int) -> None:
        if node_id in active:
            raise ValueError(f"cycle detected at macro graph node {node_id}")
        if node_id in visited:
            return
        active.add(node_id)
        for ref in refs_by_parent.get(node_id, []):
            visit(ref.child_id)
        active.remove(node_id)
        visited.add(node_id)

    visit(root_id)
    if visited != node_id_set:
        unreachable = sorted(node_id_set - visited)
        raise ValueError(f"macro graph contains unreachable nodes: {unreachable}")

    depth_cache: dict[int, int] = {}

    def depth_from(node_id: int) -> int:
        if node_id in depth_cache:
            return depth_cache[node_id]
        refs = refs_by_parent.get(node_id, [])
        if not refs:
            depth_cache[node_id] = 0
            return 0
        depth_cache[node_id] = 1 + max(depth_from(ref.child_id) for ref in refs)
        return depth_cache[node_id]

    source_ids = source_subtree_ids_by_node_id or {}
    shared_node_count = sum(
        1
        for node_id in node_id_set
        if incoming_counts[node_id] > 1 or len(source_ids.get(node_id, [])) > 1
    )
    leaf_count = sum(1 for node_id in node_id_set if not refs_by_parent.get(node_id))
    return MacroGraphStatistics(
        node_count=len(node_id_set),
        child_reference_count=len(child_refs),
        depth=depth_from(root_id),
        leaf_count=leaf_count,
        shared_node_count=shared_node_count,
    )


def refs_by_parent(graph: MacroGraph) -> dict[int, list[MacroChildRef]]:
    """Return ordered child refs keyed by parent id."""
    grouped: dict[int, list[MacroChildRef]] = defaultdict(list)
    for ref in graph.child_refs:
        grouped[ref.parent_id].append(ref)
    return {
        parent_id: sorted(refs, key=lambda ref: ref.slot_index)
        for parent_id, refs in grouped.items()
    }


def _populate_pure_eml_expansion_counts(graph: MacroGraph) -> None:
    from geml.compression.macro_expansions import (
        compute_pure_eml_dag_node_count,
        compute_pure_eml_node_count,
        expand_macro_graph_node,
    )

    for node in graph.nodes:
        expanded = expand_macro_graph_node(graph, node.id)
        node.pure_eml_expansion_node_count = compute_pure_eml_node_count(expanded)
        node.pure_eml_expansion_dag_node_count = compute_pure_eml_dag_node_count(expanded)


def _validate_macro_child_refs(parent_id: int, refs: Sequence[MacroChildRef]) -> None:
    sorted_refs = sorted(refs, key=lambda ref: ref.slot_index)
    slot_indices = [ref.slot_index for ref in sorted_refs]
    if len(set(slot_indices)) != len(slot_indices):
        raise ValueError(f"macro node {parent_id} has duplicate child slot indices")
    expected_indices = list(range(len(sorted_refs)))
    if slot_indices != expected_indices:
        raise ValueError(
            f"macro node {parent_id} has child slot indices {slot_indices}; "
            f"expected {expected_indices}"
        )


def _source_subtree_ids(node: MacroNode) -> list[int]:
    source_ids = node.metadata.get("source_subtree_ids", [])
    if not isinstance(source_ids, list):
        raise TypeError("source_subtree_ids metadata must be a list")
    return [int(source_id) for source_id in source_ids]


def _macro_signature(
    macro_name: str,
    child_signatures: tuple[tuple[int, str, MacroStructuralSignature, int], ...],
    metadata: dict[str, MetadataValue],
) -> MacroStructuralSignature:
    return (
        MACRO_REPRESENTATION_MODE,
        macro_name,
        _leaf_signature_value(macro_name, metadata),
        child_signatures,
    )


def _leaf_signature_value(
    macro_name: str,
    metadata: dict[str, MetadataValue],
) -> tuple[tuple[str, Any], ...]:
    structural_keys_by_macro = {
        "eml_variable": ("symbol_name",),
        "eml_integer": ("value",),
        "eml_rational": ("numerator", "denominator"),
        "eml_const": ("value", "numerator", "denominator"),
        "eml_zero": ("value",),
    }
    keys = structural_keys_by_macro.get(macro_name, ())
    return tuple((key, _freeze_value(metadata.get(key))) for key in keys)


def _freeze_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_value(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value
