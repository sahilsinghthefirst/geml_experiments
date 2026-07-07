# GEML Agent Roadmap

This file is the current lightweight roadmap for agents working in this repo.
It supersedes the old scaffold-era `AGENTS.md`.

## Baseline Corpus Policy

- `outputs/v1` is the default corpus for all future compression and ML-facing
  experiments.
- `outputs/v0` is a pilot corpus and is deprecated for result claims.
- Existing v0 reports remain useful as historical diagnostics only.
- Any e-graph result computed on v0 must be labeled diagnostic-only and must
  not be reported as the authoritative Goal 4 result.

## Goal Status

- Goal 1: completed core generation, AST, EML, metrics, and sample pipeline.
- Goal 2: completed raw official pure EML expansion study.
- Goal 3: completed exact structural DAG compression study.
- Goal 3R: completed generator repair, v1 corpus generation, and v1 Goal 2/3
  reruns.
- Goal 4: e-graph compression must use v1 by default.
- Goal 5: ML-facing compression experiments must use v1 by default.
- Goal 6: GNN training and evaluation must use v1 by default.

## Future Subset Labels

Future configs may select a subset label. The reserved labels are:

- `all_v1`: the full repaired v1 corpus.
- `nontrivial_v1`: a future filtered v1 subset with low triviality.
- `identity_heavy_v1`: a future diagnostic v1 subset enriched for identity
  simplification opportunities.

Until subset filtering is implemented, use `all_v1`.

## Current Commands

```bash
.venv/bin/python -m geml.data.generate_exprs --config configs/expression_v1.yaml
.venv/bin/python -m geml.experiments.run_goal2_expansion_pipeline --config configs/expansion_v1.yaml
.venv/bin/python -m geml.experiments.run_goal3_dag_pipeline --config configs/dag_compression_v1.yaml
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format . --check
```
