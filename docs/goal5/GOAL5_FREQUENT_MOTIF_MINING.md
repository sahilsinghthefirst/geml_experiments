# Goal 5.2 Frequent Motif Mining

Goal 5.2 mines frequent connected motifs from the v1 official pure EML-DAGs and
Goal 5.1 macro graphs. This is still a non-neural baseline. Learned motif
selection is reserved for Goal 5.3.

## Representation Contract

- Motif vocabulary mode: `frequent_motifs_v1`
- Compressed graph mode: `motif_compressed_graph_v1`
- Motif nodes are not pure EML nodes
- Every motif record stores an expansion map back to the source graph template
- Every selected motif replacement stores a concrete expansion map back to the
  original graph occurrence
- Metrics are reported separately from Goal 3 pure EML-DAG metrics and Goal 5.1
  macro graph metrics

Motif types:

- `pure_eml_dag`: structural motifs mined from official pure EML-DAGs
- `macro_graph`: structural motifs mined from Goal 5.1 macro graphs
- `mixed_macro_expansion`: macro motifs paired with official pure EML expansion
  metadata

## Mining Method

The miner enumerates rooted connected descendant subgraphs with bounded internal
node count. Child refs that leave the motif are stored as boundary refs, so the
motif preserves node labels, edge direction, child slot order, repeated refs,
and input/output boundaries.

The default v1 config mines:

- `min_motif_nodes: 1`
- `max_motif_nodes: 2`
- `min_support: 50`
- `max_vocab_size: 90`

Goal 5R.6 adds a leakage-control variant:

- `configs/frequent_motifs_v1.yaml` keeps the original full-corpus mining
  baseline for comparison.
- `configs/frequent_motifs_train_only_v1.yaml` mines candidate motifs from the
  deterministic train split only, then applies that vocabulary to train,
  validation, and test rows.
- The train-only vocabulary records the exact candidate-discovery expression
  indices and reports whether validation/test rows were used for discovery.

## Greedy Compression Baseline

The baseline sorts motifs by compression score, then selects non-overlapping,
replacement-safe occurrences. Overlapping replacements are rejected. A
replacement is safe only when removing its internal nodes does not break
external references to non-root internal nodes.

For each expression, the runner compresses both the pure EML-DAG and the macro
graph, then reports the smaller motif-compressed graph as the selected baseline.
Expansion validity requires both compressed graph families to expand back to
their original source graph.

## Outputs

The v1 run writes:

- `outputs/v1/goal5_frequent_motif_vocab.json`
- `outputs/v1/goal5_frequent_motif_metrics.csv`
- `outputs/v1/goal5_frequent_motif_metrics.jsonl`
- `outputs/v1/goal5_frequent_motif_summary.json`

The train-only candidate-discovery run writes:

- `outputs/v1/goal5_frequent_motif_train_only_vocab.json`
- `outputs/v1/goal5_frequent_motif_train_only_metrics.csv`
- `outputs/v1/goal5_frequent_motif_train_only_metrics.jsonl`
- `outputs/v1/goal5_frequent_motif_train_only_summary.json`

The summary includes top motifs by support, top motifs by compression saved, top
motifs by `nontrivial_v1` coverage, motifs corresponding to official macros,
and motifs that are not obvious official macros.

## V1 Run Summary

The current v1 artifact run completed with:

- Processed expressions: 10,000
- Successful motif-compressed expansions: 10,000
- Expansion validation failures: 0
- Motif vocabulary size: 70
- Pure EML-DAG motifs: 10
- Macro graph motifs: 30
- Mixed macro-expansion motifs: 30
- Median motif coverage: 57.142857142857146%
- Median compression gain vs Goal 3 EML-DAG: 7.4
- Median compression gain vs Goal 5.1 macro graph: 1.4

Subset medians:

| subset | processed | success | median gain vs Goal 3 EML-DAG | median gain vs macro graph | median coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all_v1` | 10,000 | 10,000 | 7.4 | 1.4 | 57.142857142857146% |
| `nontrivial_v1` | 3,666 | 3,666 | 7.75 | 1.4285714285714286 | 60.0% |
| `identity_heavy_v1` | 6,334 | 6,334 | 7.222222222222222 | 1.4 | 57.142857142857146% |

## Train-Only Candidate Discovery Variant

The Goal 5R.6 train-only run mined candidates from 7,021 train expressions and
used 0 validation and 0 test expressions for motif discovery. The resulting
vocabulary still had 70 motifs and reconstructed all 10,000 rows with 0
expansion validation failures.

Train-only split medians:

| split | processed | median gain vs Goal 3 EML-DAG | median coverage |
| --- | ---: | ---: | ---: |
| `train` | 7,021 | 7.428571428571429 | 57.142857142857146% |
| `validation` | 1,491 | 7.4 | 57.142857142857146% |
| `test` | 1,488 | 7.333333333333333 | 57.142857142857146% |

Train-only `nontrivial_v1` median gain vs Goal 3 EML-DAG was 7.75 with median
coverage 60.0%.

Compared with full-corpus motif mining, median coverage loss was 0.0 percentage
points overall. On test rows, median coverage loss was also 0.0 percentage
points; median gain was 7.333333333333333 for train-only mining versus
7.369318181818182 for full-corpus mining. These numbers are compression
comparisons only, not reasoning-performance claims.

## Reproducibility

Run:

```bash
.venv/bin/python -m geml.experiments.goal5_frequent_motif_mining --config configs/frequent_motifs_v1.yaml
.venv/bin/python -m geml.experiments.goal5_frequent_motif_mining --config configs/frequent_motifs_train_only_v1.yaml
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format . --check
```

This baseline does not modify the official EML compiler, does not train final
GNNs, and does not make model-performance claims.
