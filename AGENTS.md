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
- Goal 4: completed v1 e-graph compression baselines. Reports must label
  success-only and all-processed threshold denominators separately.
- Goal 5: completed v1 ML-facing compression infrastructure. Macro graphs are
  validated/useful and frequent motifs are the strongest simple compression
  baseline; learned motif selection does not beat frequent/random baselines at
  the median, and the neural e-graph ranker is a speed/ranking baseline rather
  than a major compression claim.
- Goal 5R: repair pass completed for fresh-clone tests/CI, real motif
  reconstruction, cyclic e-graph extraction, denominator audits, null-result
  framing, train-only motif discovery, and reproducibility metadata.
- Goal 6: GNN training and evaluation must use v1 by default. Do not start Goal
  6 until the Goal 5R repair boundary is complete and the default tests/lint
  checks pass.

## Goal 5 Boundaries

- Macro, motif, learned motif, and hierarchical compressed nodes are not pure
  EML nodes unless expanded back to official pure EML.
- Learned motif selection and the neural e-graph extractor should be treated as
  Goal 6 baselines, not headline reasoning-performance claims.
- The neural extractor's 109x speedup is scoped to candidate cost scoring only.
- Do not train final symbolic-reasoning GNNs or claim downstream reasoning
  improvement from Goal 5 compression metrics.
- Learned motif candidate discovery must use the train-only motif vocabulary for
  leakage-controlled claims. Full-corpus motif mining is a comparison baseline
  only.
- Goal 5 run summaries and train logs should carry reproducibility metadata:
  git commit hash, dirty state, Python version, platform, and core package
  versions.

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
