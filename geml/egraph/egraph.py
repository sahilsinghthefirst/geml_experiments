"""E-graph primitives for the GEML Goal 4 source IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from geml.egraph.ir import Add, Const, Div, Exp, Expr, Log, Mul, Neg, Pow, Sub, Var


@dataclass(frozen=True, slots=True)
class ENode:
    """An ordered e-node. Child order is part of identity."""

    op: str
    children: tuple[int, ...] = ()
    value: str | Fraction | None = None

    def canonicalize_children(self, egraph: EGraph) -> ENode:
        """Return the same node with child ids replaced by canonical e-class ids."""
        return ENode(
            op=self.op,
            children=tuple(egraph.find(child_id) for child_id in self.children),
            value=self.value,
        )

    def sort_key(self) -> tuple[str, str, tuple[int, ...]]:
        """Return a deterministic sort key."""
        return (self.op, repr(self.value), self.children)


@dataclass(slots=True)
class EClass:
    """An equivalence class of e-nodes."""

    id: int
    nodes: set[ENode] = field(default_factory=set)


class EGraph:
    """A small transparent e-graph with union-find and rebuild closure."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._classes: dict[int, EClass] = {}
        self._memo: dict[ENode, int] = {}
        self._next_class_id = 0

    @property
    def enode_count(self) -> int:
        """Return the number of unique e-nodes in canonical e-classes."""
        return sum(len(eclass.nodes) for eclass in self._canonical_classes())

    @property
    def eclass_count(self) -> int:
        """Return the number of canonical e-classes."""
        return len(self._canonical_classes())

    def add_expr(self, expr: Expr) -> int:
        """Add an IR expression and return its canonical e-class id."""
        if isinstance(expr, Var):
            return self.add_enode(ENode("var", value=expr.name))
        if isinstance(expr, Const):
            return self.add_enode(ENode("const", value=expr.value))
        if isinstance(expr, Add):
            return self.add_enode(
                ENode("add", (self.add_expr(expr.left), self.add_expr(expr.right)))
            )
        if isinstance(expr, Mul):
            return self.add_enode(
                ENode("mul", (self.add_expr(expr.left), self.add_expr(expr.right)))
            )
        if isinstance(expr, Neg):
            return self.add_enode(ENode("neg", (self.add_expr(expr.value),)))
        if isinstance(expr, Sub):
            return self.add_enode(
                ENode("sub", (self.add_expr(expr.left), self.add_expr(expr.right)))
            )
        if isinstance(expr, Div):
            return self.add_enode(
                ENode("div", (self.add_expr(expr.left), self.add_expr(expr.right)))
            )
        if isinstance(expr, Pow):
            return self.add_enode(
                ENode("pow", (self.add_expr(expr.base), self.add_expr(expr.exponent)))
            )
        if isinstance(expr, Exp):
            return self.add_enode(ENode("exp", (self.add_expr(expr.value),)))
        if isinstance(expr, Log):
            return self.add_enode(ENode("log", (self.add_expr(expr.value),)))
        raise TypeError(f"unsupported IR expression type: {type(expr).__name__}")

    def add_enode(self, enode: ENode) -> int:
        """Add an e-node and return its canonical e-class id."""
        canonical_enode = enode.canonicalize_children(self)
        existing_id = self._memo.get(canonical_enode)
        if existing_id is not None:
            canonical_id = self.find(existing_id)
            self._memo[canonical_enode] = canonical_id
            return canonical_id

        class_id = self._new_class()
        self._classes[class_id].nodes.add(canonical_enode)
        self._memo[canonical_enode] = class_id
        return class_id

    def union(self, left_id: int, right_id: int) -> bool:
        """Union two e-classes. Return True when the graph changed."""
        left_root = self.find(left_id)
        right_root = self.find(right_id)
        if left_root == right_root:
            return False

        winner = min(left_root, right_root)
        loser = max(left_root, right_root)
        self._parent[loser] = winner
        self._classes[winner].nodes.update(self._classes[loser].nodes)
        del self._classes[loser]
        return True

    def find(self, class_id: int) -> int:
        """Find the canonical representative for an e-class id."""
        if class_id not in self._parent:
            raise KeyError(f"unknown e-class id: {class_id}")
        parent = self._parent[class_id]
        if parent != class_id:
            parent = self.find(parent)
            self._parent[class_id] = parent
        return parent

    def rebuild(self) -> None:
        """Restore congruence closure after unions."""
        while True:
            self._compact_classes()
            new_memo: dict[ENode, int] = {}
            changed = False
            for eclass in list(self._canonical_classes()):
                canonical_nodes: set[ENode] = set()
                for node in eclass.nodes:
                    canonical_node = node.canonicalize_children(self)
                    owner_id = new_memo.get(canonical_node)
                    if owner_id is not None and self.find(owner_id) != eclass.id:
                        self.union(owner_id, eclass.id)
                        changed = True
                    else:
                        new_memo[canonical_node] = self.find(eclass.id)
                    canonical_nodes.add(canonical_node)
                if eclass.id in self._classes:
                    self._classes[eclass.id].nodes = canonical_nodes
            if not changed:
                self._memo = {node: self.find(class_id) for node, class_id in new_memo.items()}
                return

    def get_eclass_nodes(self, class_id: int) -> tuple[ENode, ...]:
        """Return the ordered e-nodes in an e-class."""
        root_id = self.find(class_id)
        return tuple(sorted(self._classes[root_id].nodes, key=lambda node: node.sort_key()))

    def eclass_ids(self) -> tuple[int, ...]:
        """Return canonical e-class ids."""
        self._compact_classes()
        return tuple(sorted(self._classes))

    def _new_class(self) -> int:
        class_id = self._next_class_id
        self._next_class_id += 1
        self._parent[class_id] = class_id
        self._classes[class_id] = EClass(id=class_id)
        return class_id

    def _canonical_classes(self) -> list[EClass]:
        return [
            eclass for class_id, eclass in self._classes.items() if self.find(class_id) == class_id
        ]

    def _compact_classes(self) -> None:
        for class_id in list(self._parent):
            self.find(class_id)
        for class_id in list(self._classes):
            root_id = self.find(class_id)
            if root_id != class_id:
                self._classes[root_id].nodes.update(self._classes[class_id].nodes)
                del self._classes[class_id]
