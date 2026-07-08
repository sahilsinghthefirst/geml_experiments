# Goal 5.3 Learned Motif Compression

Goal 5.3 adds a deterministic learned motif-selection layer over the Goal 5.2
candidate motif pool. It is ML-facing compression only. It does not train final
symbolic-reasoning GNNs and does not make reasoning-performance claims.

## Method

The selector uses a lightweight linear scoring objective:

```text
score(motif) =
  node_savings
  + coverage_bonus * coverage
  + nontrivial_coverage_bonus * nontrivial_coverage
  - vocab_complexity_penalty * node_count
  - expansion_complexity_penalty * expansion_complexity
```

No PyTorch dependency is added for Goal 5.3. The current layer is deterministic
hyperparameter search over a discrete motif vocabulary, not a neural model.

## Data Split

Expressions are assigned to `train`, `validation`, and `test` by deterministic
hash of expression index and seed:

- train: 70%
- validation: 15%
- test: 15%

Train scores choose motif rankings. Validation objective chooses hyperparameters
and vocabulary size. Test rows are only used for final reporting.

## Baselines

The runner reports:

- Goal 5.2 frequent motif baseline
- learned motif vocabulary
- random motif vocabulary with the same size as the learned vocabulary
- Goal 5.1 macro graph baseline

All metrics keep compressed motif graph sizes separate from pure EML-DAG sizes.

## Integrity Contract

- Learned motif IDs are not pure EML nodes
- Every learned motif stores an expansion map
- Reconstruction validity is required and reported
- Test split is not used for selection
- Official EML compiler formulas are not modified
- No final symbolic-reasoning GNN is trained

## Outputs

The v1 run writes:

- `outputs/v1/goal5_learned_motif_vocab.json`
- `outputs/v1/goal5_learned_motif_metrics.csv`
- `outputs/v1/goal5_learned_motif_metrics.jsonl`
- `outputs/v1/goal5_learned_motif_summary.json`
- `outputs/v1/goal5_learned_motif_train_log.json`

## V1 Run Summary

The configured v1 run processed all 10,000 expressions with 10,000 successful
reconstructions and 0 reconstruction failures.

The deterministic split produced:

- train: 7,021 expressions
- validation: 1,491 expressions
- test: 1,488 expressions

The candidate pool contained 39 graph-applicable motifs from Goal 5.2. The
selector evaluated 32 hyperparameter trials and selected a 30-motif learned
vocabulary. The random baseline also used 30 motifs.

Selected weights:

- `coverage_bonus`: 0.01
- `nontrivial_coverage_bonus`: 0.02
- `vocab_complexity_penalty`: 0.0
- `expansion_complexity_penalty`: 0.0

Aggregate medians:

| Metric | Median |
| --- | ---: |
| learned gain vs Goal 3 EML-DAG | 7.125 |
| learned vs frequent motif compression | 1.000 |
| learned vs random motif compression | 1.000 |
| learned vs macro graph baseline | 1.333 |
| learned motif coverage percent | 50.000 |

Held-out test medians:

| Metric | Median |
| --- | ---: |
| learned gain vs Goal 3 EML-DAG | 7.000 |
| learned vs frequent motif compression | 1.000 |
| learned vs random motif compression | 1.000 |

On `nontrivial_v1`, the median learned gain vs Goal 3 EML-DAG was 7.429, with
median parity against both the frequent motif and random motif baselines.

Interpretation: the learned selector preserves exact reconstruction and keeps
strong compression relative to pure EML-DAG and macro graphs. In this v1 setup,
it mostly matches the frequent motif baseline at the median rather than clearly
beating it. The held-out test metrics are close to the full-corpus metrics, so
the selected vocabulary is not only fitting the training rows under these
compression metrics.

## Reproducibility

Run:

```bash
.venv/bin/python -m geml.experiments.goal5_learned_motif_compression --config configs/learned_motifs_v1.yaml
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format . --check
```
