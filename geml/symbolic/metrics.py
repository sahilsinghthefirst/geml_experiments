"""Tree statistics for symbolic graph representations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, Field


class TreeStatistics(BaseModel):
    """Basic tree statistics with leaves at depth 0."""

    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    depth: int = Field(ge=0)
    leaf_count: int = Field(ge=0)
    operator_count: int = Field(ge=0)


def compute_tree_statistics(
    *,
    root_id: int,
    node_ids: Iterable[int],
    edges: Sequence[tuple[int, int]],
    operator_node_ids: Iterable[int],
) -> TreeStatistics:
    """Compute structural statistics for a rooted tree."""
    node_id_set = set(node_ids)
    if root_id not in node_id_set:
        raise ValueError(f"root_id {root_id} is not present in node_ids")

    children: dict[int, list[int]] = {node_id: [] for node_id in node_id_set}
    parent_counts: dict[int, int] = {node_id: 0 for node_id in node_id_set}
    for source_id, target_id in edges:
        if source_id not in node_id_set:
            raise ValueError(f"edge source {source_id} is not present in node_ids")
        if target_id not in node_id_set:
            raise ValueError(f"edge target {target_id} is not present in node_ids")
        children[source_id].append(target_id)
        parent_counts[target_id] += 1

    if parent_counts[root_id] != 0:
        raise ValueError(f"root_id {root_id} must not have a parent")

    for node_id, parent_count in parent_counts.items():
        if node_id == root_id:
            continue
        if parent_count != 1:
            raise ValueError(f"node {node_id} must have exactly one parent, got {parent_count}")

    reachable: set[int] = set()

    def visit(node_id: int) -> None:
        if node_id in reachable:
            raise ValueError(f"cycle or repeated path detected at node {node_id}")
        reachable.add(node_id)
        for child_id in children[node_id]:
            visit(child_id)

    visit(root_id)
    if reachable != node_id_set:
        unreachable = sorted(node_id_set - reachable)
        raise ValueError(f"tree contains unreachable nodes: {unreachable}")

    def depth_from(node_id: int) -> int:
        if not children[node_id]:
            return 0
        return 1 + max(depth_from(child_id) for child_id in children[node_id])

    leaf_count = sum(1 for node_id in node_id_set if not children[node_id])

    return TreeStatistics(
        node_count=len(node_id_set),
        edge_count=len(edges),
        depth=depth_from(root_id),
        leaf_count=leaf_count,
        operator_count=len(set(operator_node_ids)),
    )
