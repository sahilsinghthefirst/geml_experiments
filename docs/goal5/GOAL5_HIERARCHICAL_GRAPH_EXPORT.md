# Goal 5.5 Hierarchical Graph Export

Goal 5.5 packages the existing v1 graph representations into a neutral,
auditable graph dataset for future Goal 6 GNN work. It does not train GNNs and
does not make downstream model-performance claims.

## Purpose

Goals 5.1 through 5.4 produced several validated compression-facing graph
representations:

- macro graphs over official compiler macros
- frequent motif compressed graphs
- learned motif compressed graphs
- e-graph extracted pure EML-DAGs

Goal 5.5 exports these representations with enough metadata to audit how each
compressed node expands back to official pure EML. The export is graph-format
neutral JSONL first. No PyTorch Geometric dependency is added.

## Exported Representations

The v1 export writes one graph record per expression and representation mode:

- `ast_tree_graph`
- `ast_dag_graph`
- `pure_eml_dag_graph`
- `egraph_safe_eml_dag_graph`
- `egraph_positive_real_eml_dag_graph`
- `macro_graph`
- `frequent_motif_graph`
- `learned_motif_graph`
- `hierarchical_eml_graph`

The e-graph modes are only exported for rows with validated Goal 4 extraction
outputs. Missing or invalid e-graph rows are counted in the summary rather than
filled in silently.

## Hierarchical Levels

`hierarchical_eml_graph` combines the validated levels for an expression:

| Level | Source |
| --- | --- |
| 0 | source AST |
| 1 | macro graph |
| 2 | pure EML-DAG expansion |
| 3 | frequent motif graph |
| 4 | learned motif graph |

Expansion edges point from compressed nodes toward the official pure EML-DAG
level. Standalone macro and motif graph records may point to the companion pure
EML-DAG graph for the same expression; the hierarchical graph remaps those
targets inside one combined graph record.

## Schema Contract

Each graph record includes:

- `graph_id`
- `source_expression_id`
- `split`
- `subset_label`
- `representation_mode`
- `nodes`
- `edges`
- `validation`
- `metadata`

Each node includes the required audit metadata:

- `node_id`
- `graph_id`
- `representation_mode`
- `node_type`
- `label`
- `arity`
- `child_slot`
- `source_expression_id`
- `expansion_available`
- `expansion_target_ids`
- `pure_eml_valid`
- `motif_id`
- `macro_name`

Required edge types are represented explicitly:

- `ast_child`
- `ast_to_macro`
- `macro_child`
- `macro_expands_to_eml`
- `eml_child`
- `motif_instance`
- `motif_expands_to_eml`
- `learned_motif_instance`
- `hierarchy_parent_child`

Child slots are preserved on ordered child edges. Duplicate references are
stored as distinct child-reference edges, so repeated child usage is not
collapsed during export.

## Validation

The exporter validates:

- schema shape for every graph record
- unique graph IDs
- compressed nodes have expansion mappings
- macro graphs expand to official pure EML
- motif graphs preserve expansion maps to official pure EML
- learned motif graphs preserve expansion maps to official pure EML
- hierarchical graphs preserve internal expansion links
- train, validation, and test splits are deterministic

Compressed graph nodes are not labeled as pure EML nodes. The graph records do
not include hidden target labels for future ML tasks.

## Outputs

The v1 run writes:

- `outputs/v1/goal5_hierarchical_graphs.jsonl`
- `outputs/v1/goal5_graph_splits.json`
- `outputs/v1/goal5_graph_schema.json`
- `outputs/v1/goal5_hierarchical_export_summary.json`

## V1 Run Summary

The configured v1 run processed all 10,000 expressions and exported 88,257
graph records.

Graph ID and validation results:

| Metric | Value |
| --- | ---: |
| graph records | 88,257 |
| unique graph IDs | 88,257 |
| all graph IDs unique | true |
| expansion validation rate | 100.000% |
| reconstruction validation rate | 100.000% |
| missing expansion count | 0 |

Deterministic split counts:

| Split | Expressions | Graph records |
| --- | ---: | ---: |
| train | 7,021 | 61,979 |
| validation | 1,491 | 13,164 |
| test | 1,488 | 13,114 |

Mode-level graph counts and median sizes:

| Representation mode | Graphs | Median nodes | Median edges |
| --- | ---: | ---: | ---: |
| `ast_tree_graph` | 10,000 | 10 | 9 |
| `ast_dag_graph` | 10,000 | 8 | 9 |
| `pure_eml_dag_graph` | 10,000 | 42 | 79 |
| `egraph_safe_eml_dag_graph` | 9,316 | 38 | 70 |
| `egraph_positive_real_eml_dag_graph` | 8,941 | 35 | 64 |
| `macro_graph` | 10,000 | 8 | 17 |
| `frequent_motif_graph` | 10,000 | 6 | 9 |
| `learned_motif_graph` | 10,000 | 6 | 9 |
| `hierarchical_eml_graph` | 10,000 | 73 | 128 |

Skipped e-graph rows:

| Rule mode | Skipped rows |
| --- | ---: |
| `safe` | 684 |
| `positive_real_formal` | 1,059 |

All exported modes reported 100% expansion validation and 100% reconstruction
validation in the summary. The skipped e-graph counts reflect missing or invalid
validated Goal 4 extraction rows, not dropped base expressions.

## Integrity Contract

- Exporting graph data does not imply model performance.
- Compressed graph nodes are not pure EML nodes.
- Every compressed node has an expansion path back to official pure EML.
- Safe and positive-real e-graph modes are labeled separately.
- Official EML compiler formulas are not modified.
- No final symbolic-reasoning GNN is trained.
- No hidden answer labels are added to the graph records.

## Reproducibility

Run:

```bash
.venv/bin/python -m geml.experiments.goal5_hierarchical_export --config configs/hierarchical_graph_export_v1.yaml
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format . --check
```
