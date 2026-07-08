# Goal 5.4 Neural E-Graph Extractor

Goal 5.4 trains a neural cost model to rank bounded e-graph extraction
candidates by expected official pure EML-DAG cost. This is a compression/ranking
tool only. It does not train final symbolic-reasoning GNNs and does not make
downstream reasoning-performance claims.

## Method

The candidate dataset is regenerated from v1 expressions when no saved candidate
records are available. For each expression and rule mode, the runner:

- builds and saturates the e-graph with the existing Goal 4 rules
- enumerates bounded root candidates with the same beam-style candidate generator
- labels every candidate by compiling through the official pure EML compiler
- computes the exact structural EML-DAG node count as the ground-truth cost
- records candidate features, validation status, and exact-label timing

The model is a lightweight feature-based MLP trained with pairwise ranking loss
inside each `(expression_id, rule_mode)` group. Lower model score means lower
predicted cost. No PyTorch dependency is added; the MLP is deterministic and
implemented directly for this baseline.

## Baselines

The evaluator reports:

- exact best candidate by official EML-DAG cost
- estimated EML-cost heuristic
- AST-node-cost heuristic
- neural cost-model ranking

Exact official EML-DAG cost remains the ground-truth evaluation target. The
neural model does not define mathematical truth and does not claim global
optimality.

## Validation

Neural-selected candidates must still satisfy:

- official pure EML compilation
- pure EML-DAG structural integrity
- same-root e-class validation
- positive-real numeric validation used by Goal 4

Failed validation rows are kept in the metrics and counted in the summary.

## Outputs

The v1 run writes:

- `outputs/v1/goal5_neural_egraph_candidate_dataset.jsonl`
- `outputs/v1/goal5_neural_egraph_metrics.csv`
- `outputs/v1/goal5_neural_egraph_summary.json`
- `outputs/v1/goal5_neural_egraph_train_log.json`

## Reported Metrics

The summary reports neural extraction against exact, estimated, and AST
baselines:

- top-1 selected candidate EML-DAG cost
- regret vs exact best candidate
- percent matching exact best
- median compression gain vs Goal 3 EML-DAG
- cost-scoring speedup vs exact beam scoring
- results by `safe` and `positive_real_formal`
- results by `all_v1`, `nontrivial_v1`, and `identity_heavy_v1`
- train, validation, and test split results

Runtime speedup is scoped to candidate cost scoring. It excludes e-graph
saturation and candidate enumeration because those steps are shared by the exact
and neural rankers in this baseline.

## V1 Run Summary

The configured v1 run produced 171,545 labeled candidates across 20,000
expression/rule-mode groups:

- `safe`: 86,420 candidates
- `positive_real_formal`: 85,125 candidates
- official label failures: 0
- trained candidates: 120,820
- train pair count: 30,577
- model epochs: 12

Evaluation processed all 20,000 groups. The neural-selected candidate validated
successfully for 18,871 groups; 1,129 groups were reported as validation
failures and were not dropped.

Aggregate successful-row metrics:

| Metric | Value |
| --- | ---: |
| neural percent matching exact best | 64.236% |
| neural median regret vs exact best | 0.000 |
| neural p90 regret vs exact best | 3.000 |
| neural median top-1 EML-DAG nodes | 37.000 |
| exact-best median EML-DAG nodes | 37.000 |
| neural median compression gain vs Goal 3 EML-DAG | 1.074 |
| exact-best median compression gain vs Goal 3 EML-DAG | 1.091 |
| median neural speedup vs exact cost scoring | 109.305x |

Baseline regret medians were also 0.000 for estimated EML cost and AST-node
cost, with p90 regret 3.000. Mean regret was lower for the neural ranker
(`0.506`) than for estimated EML cost (`0.597`) and AST-node cost (`0.698`).

Rule-mode breakdown:

| Rule mode | Success / Processed | Exact-match % | Median neural gain | Median speedup |
| --- | ---: | ---: | ---: | ---: |
| `safe` | 9,498 / 10,000 | 64.529% | 1.000 | 114.704x |
| `positive_real_formal` | 9,373 / 10,000 | 63.939% | 1.139 | 103.570x |

Subset breakdown:

| Subset | Success / Processed | Exact-match % | Median neural gain | Median speedup |
| --- | ---: | ---: | ---: | ---: |
| `nontrivial_v1` | 6,688 / 7,332 | 62.590% | 1.000 | 112.934x |
| `identity_heavy_v1` | 12,183 / 12,668 | 65.140% | 1.184 | 106.730x |

Held-out test split:

| Metric | Value |
| --- | ---: |
| success / processed | 2,788 / 2,976 |
| neural percent matching exact best | 64.168% |
| neural median regret vs exact best | 0.000 |
| neural p90 regret vs exact best | 3.000 |
| neural median compression gain vs Goal 3 EML-DAG | 1.073 |
| median speedup vs exact cost scoring | 106.006x |

Interpretation: the learned ranker often selects the exact-best candidate and
has median zero regret on successful rows, but it is not globally optimal and
does not eliminate validation failures. Its main value in this run is replacing
official-cost scoring with a much faster learned ranking step while preserving
official EML-DAG cost as the evaluation target.

## Reproducibility

Run:

```bash
.venv/bin/python -m geml.experiments.goal5_neural_egraph_extractor --config configs/neural_egraph_extractor_v1.yaml
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format . --check
```
