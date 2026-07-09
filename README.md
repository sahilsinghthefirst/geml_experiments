# GEML Experiments

[![CI](https://github.com/sahilsinghthefirst/geml_experiments/actions/workflows/ci.yml/badge.svg)](https://github.com/sahilsinghthefirst/geml_experiments/actions/workflows/ci.yml)

Experiments for comparing source ASTs, official pure EML graphs, e-graph
compression, and ML-facing compressed graph representations for symbolic math.

## Status

- Goals 1-5R are complete.
- Goal 6 GNN training has not started.
- `outputs/v1` is the default result-bearing corpus.
- `outputs/v0` is pilot/diagnostic only. Do not use it for new claims.
- Goal 5R fixed fresh-clone tests/CI, real motif reconstruction, cyclic
  e-graph extraction, denominator reporting, train-only motif discovery, and
  reproducibility metadata.

Goal 5 result framing:

- Macro graphs are validated and useful.
- Frequent motifs are the strongest simple compression baseline.
- Learned motif selection does not beat frequent/random baselines at the median.
- The neural e-graph extractor is a speed/ranking baseline, not a reasoning or
  major-compression claim.
- No final symbolic-reasoning GNNs are trained here.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format . --check
```

The default test suite uses small fixtures and should pass on a fresh clone. Full
artifact checks are marked `slow`.

```bash
.venv/bin/python -m pytest -m slow
```

## Main Commands

```bash
.venv/bin/python -m geml.data.generate_exprs --config configs/expression_v1.yaml
.venv/bin/python -m geml.experiments.run_goal2_expansion_pipeline --config configs/expansion_v1.yaml
.venv/bin/python -m geml.experiments.run_goal3_dag_pipeline --config configs/dag_compression_v1.yaml
.venv/bin/python -m geml.experiments.run_goal4_egraph_pipeline --config configs/egraph_compression_v1.yaml
.venv/bin/python -m geml.experiments.run_goal5_compression_pipeline --config configs/goal5_compression_v1.yaml
```

## Docs

- Goal docs: `docs/goal1/` through `docs/goal5/`
- Current Goal 5 summary: `docs/goal5/GOAL5_SUMMARY.md`
- License: MIT
