"""Build multi-level hierarchical EML graph exports."""

from __future__ import annotations

from collections.abc import Sequence

from geml.compression.graph_schema import (
    GraphExportEdge,
    GraphExportNode,
    GraphExportRecord,
    GraphValidationStatus,
    validate_graph_record,
)


def build_hierarchical_eml_graph(
    *,
    graph_id: str,
    source_expression_id: int,
    subset_label: str,
    split: str,
    ast_graph: GraphExportRecord,
    macro_graph: GraphExportRecord,
    pure_eml_graph: GraphExportRecord,
    motif_graph: GraphExportRecord,
    learned_motif_graph: GraphExportRecord | None = None,
) -> GraphExportRecord:
    """Build a hierarchical graph from AST, macro, EML, and motif levels."""
    levels = [
        ("level0_ast", ast_graph),
        ("level1_macro", macro_graph),
        ("level2_pure_eml_dag", pure_eml_graph),
        ("level3_motif", motif_graph),
    ]
    if learned_motif_graph is not None:
        levels.append(("level4_learned_motif", learned_motif_graph))

    id_map: dict[str, str] = {}
    for level_name, record in levels:
        for node in record.nodes:
            id_map[node.node_id] = _hier_node_id(graph_id, level_name, node.node_id)

    pure_root_target = id_map[pure_eml_graph.root_node_id]
    nodes: list[GraphExportNode] = []
    edges: list[GraphExportEdge] = []
    for level_index, (level_name, record) in enumerate(levels):
        for node in record.nodes:
            expansion_targets = [
                id_map.get(target_id, pure_root_target) for target_id in node.expansion_target_ids
            ]
            if node.node_type in {"macro", "motif", "learned_motif"} and not expansion_targets:
                expansion_targets = [pure_root_target]
            nodes.append(
                node.model_copy(
                    update={
                        "node_id": id_map[node.node_id],
                        "graph_id": graph_id,
                        "representation_mode": "hierarchical_eml_graph",
                        "expansion_target_ids": expansion_targets,
                        "metadata": {
                            **node.metadata,
                            "hierarchy_level": level_index,
                            "hierarchy_level_name": level_name,
                            "source_graph_id": record.graph_id,
                            "source_representation_mode": record.representation_mode,
                        },
                    }
                )
            )
        for edge in record.edges:
            source_id = id_map.get(edge.source_id)
            target_id = id_map.get(edge.target_id, pure_root_target)
            if source_id is None:
                continue
            edges.append(
                edge.model_copy(
                    update={
                        "edge_id": f"{graph_id}:{level_name}:{edge.edge_id}",
                        "graph_id": graph_id,
                        "source_id": source_id,
                        "target_id": target_id,
                        "metadata": {
                            **edge.metadata,
                            "hierarchy_level": level_index,
                            "source_graph_id": record.graph_id,
                        },
                    }
                )
            )

    root_links = _cross_level_edges(graph_id, levels, id_map)
    edges.extend(root_links)
    metadata = {
        "source": "goal5_hierarchical_eml_graph_v1",
        "pure_eml_valid": False,
        "reconstruction_valid": all(
            record.validation.reconstruction_valid for _level, record in levels
        ),
        "contains_hidden_target_labels": False,
        "level_count": len(levels),
        "levels": [
            {
                "level_index": index,
                "level_name": level_name,
                "graph_id": record.graph_id,
                "representation_mode": record.representation_mode,
            }
            for index, (level_name, record) in enumerate(levels)
        ],
    }
    record = GraphExportRecord(
        graph_id=graph_id,
        source_expression_id=source_expression_id,
        representation_mode="hierarchical_eml_graph",
        root_node_id=id_map[ast_graph.root_node_id],
        nodes=nodes,
        edges=edges,
        subset_label=subset_label,
        split=split,
        metadata=metadata,
        validation=GraphValidationStatus(
            schema_valid=True,
            expansion_valid=True,
            reconstruction_valid=True,
            missing_expansion_count=0,
            pure_eml_valid=False,
            errors=[],
        ),
    )
    return record.model_copy(update={"validation": validate_graph_record(record)})


def _cross_level_edges(
    graph_id: str,
    levels: Sequence[tuple[str, GraphExportRecord]],
    id_map: dict[str, str],
) -> list[GraphExportEdge]:
    edges: list[GraphExportEdge] = []
    for index, ((source_level, source_record), (target_level, target_record)) in enumerate(
        zip(levels, levels[1:], strict=False)
    ):
        edge_type = "ast_to_macro" if index == 0 else "hierarchy_parent_child"
        edges.append(
            GraphExportEdge(
                edge_id=f"{graph_id}:hierarchy:{source_level}:{target_level}",
                graph_id=graph_id,
                source_id=id_map[source_record.root_node_id],
                target_id=id_map[target_record.root_node_id],
                edge_type=edge_type,
                child_slot=target_level,
                slot_index=index,
                metadata={
                    "source_level": source_level,
                    "target_level": target_level,
                },
            )
        )
    return edges


def _hier_node_id(graph_id: str, level_name: str, source_node_id: str) -> str:
    safe_source = source_node_id.replace(":", "_")
    return f"{graph_id}:{level_name}:{safe_source}"
