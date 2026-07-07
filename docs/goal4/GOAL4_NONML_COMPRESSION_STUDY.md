# Goal 4 Non-ML Compression Study

## Goal 4 Question

Can e-graph equality saturation and EML-aware extraction reduce official pure EML-DAG size beyond Goal 3 exact structural DAG sharing?

Goal 4 is non-ML compression. It uses algebraic rewrite rules before official pure EML compilation, then scores the extracted expression by official pure EML-DAG size.

## Relation To Goals 2, 3, And 3R

- Goal 2 measured raw official pure EML expansion and showed the pure trees are representation-complete but structurally expensive.
- Goal 3 added exact structural DAG compression after official pure EML compilation.
- Goal 3R repaired the corpus and established v1 as the serious baseline.
- Goal 4 performs non-ML algebraic compression before EML compilation, then compares against the Goal 3 v1 exact EML-DAG baseline.

## Why V1 Is Used

`outputs/v1` is the default result-bearing corpus. `outputs/v0` is pilot and diagnostic only. V1 fixes depth, duplicate, log-argument, and triviality artifacts that would otherwise overstate or distort e-graph compression.

- configured count: `10000`
- v1 input JSONL: `outputs/v1/dag_compression_inputs.jsonl`
- v1 Goal 3 metrics: `outputs/v1/dag_compression_metrics.csv`
- v1 Goal 3 summary processed count: `10000`

## Rule Modes And Assumptions

- `safe`: commutativity, associativity, identities, safe inverse forms, sub lowering, double negation, and exact bounded constant folding. It excludes branch-sensitive log/exp identities.
- `positive_real_formal`: includes safe rules plus positive-real formal log/exp rules. This mode is branch-sensitive, relies on the v1 positive-real domain convention, and makes no universal complex-domain validity claim.

The two modes are reported separately. Goal 4 metrics must not be mixed with Goal 3 exact-DAG metrics without naming the mode.

## Extractor Objective

The headline extractor is `exact_eml_dag_beam_cost`. It enumerates source candidates from the root e-class, converts each candidate to SymPy without simplification, compiles with the official pure EML compiler, converts the result to an exact structural EML DAG, and selects the candidate with minimum official pure EML-DAG node count.

Ordinary AST node count is only a baseline. It is not an EML-optimal objective.

Tie-breaking order:

1. extracted official pure EML-DAG nodes
2. extracted official pure EML-tree nodes
3. extracted source AST-DAG nodes
4. extracted source AST-tree nodes
5. stable expression string

## Resource Limits

- max_iterations: `4`
- max_enodes: `5000`
- max_eclasses: `5000`
- timeout_seconds: `0.5`
- row_timeout_seconds: `2.0`
- beam_size: `12`
- max_candidate_depth: `7`
- max_candidates_evaluated: `12`

## 10k V1 Results: Safe Mode

- processed: `10000`
- success: `9316`
- timeout: `241`
- validation failures: `607`
- median Goal 3 DAG alpha: `4.0`
- median optimized DAG alpha: `3.6363636363636362`
- median compression gain vs Goal 3 DAG: `1.0454545454545454`
- percent improved: `53.027%`
- percent unchanged: `45.298%`
- percent worse: `1.675%`
- below threshold before e-graph: `0.220%`
- below threshold after e-graph: `1.020%`
- median runtime per expression: `0.03714627100271173` seconds

## 10k V1 Results: Positive-Real Formal Mode

- processed: `10000`
- success: `8941`
- timeout: `522`
- validation failures: `878`
- median Goal 3 DAG alpha: `4.0`
- median optimized DAG alpha: `3.3636363636363638`
- median compression gain vs Goal 3 DAG: `1.1692307692307693`
- percent improved: `72.934%`
- percent unchanged: `25.791%`
- percent worse: `1.275%`
- below threshold before e-graph: `0.220%`
- below threshold after e-graph: `5.827%`
- median runtime per expression: `0.04460766648116987` seconds

Positive-real rows are labeled with branch-sensitive assumptions and branch-sensitive rule usage. They are not complex-domain algebra claims.

## Subset Analysis

The corpus is split by measured triviality features, not guesses. `identity_heavy_v1` contains measured identity or trivial simplification opportunities. `nontrivial_v1` excludes those measured features.

| Subset | Mode | Count | Success | Median Goal 3 alpha | Median optimized alpha | Median gain | Improved | Unchanged | Worse | Below before | Below after | Timeout | Validation failure | Branch-sensitive usage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all_v1` | `safe` | 10000 | 9316 | 4.0 | 3.6363636363636362 | 1.0454545454545454 | 53.027% | 45.298% | 1.675% | 0.220% | 1.020% | 2.410% | 6.840% | 0.000% |
| `all_v1` | `positive_real_formal` | 10000 | 8941 | 4.0 | 3.3636363636363638 | 1.1692307692307693 | 72.934% | 25.791% | 1.275% | 0.220% | 5.827% | 5.220% | 10.590% | 100.000% |
| `nontrivial_v1` | `safe` | 3733 | 3428 | 4.111111111111111 | 4.071428571428571 | 1.0 | 25.554% | 71.120% | 3.326% | 0.402% | 0.292% | 0.884% | 8.170% | 0.000% |
| `nontrivial_v1` | `positive_real_formal` | 3733 | 3359 | 4.111111111111111 | 3.933333333333333 | 1.0 | 37.571% | 59.244% | 3.185% | 0.402% | 0.298% | 1.286% | 10.019% | 100.000% |
| `identity_heavy_v1` | `safe` | 6267 | 5888 | 4.0 | 3.4166666666666665 | 1.1538461538461537 | 69.022% | 30.265% | 0.713% | 0.112% | 1.444% | 3.319% | 6.048% | 0.000% |
| `identity_heavy_v1` | `positive_real_formal` | 6267 | 5582 | 4.0 | 3.0444664031620556 | 1.2692307692307692 | 94.214% | 5.661% | 0.125% | 0.112% | 9.154% | 7.563% | 10.930% | 100.000% |

The median nontrivial compression gain remains much closer to `1.0` than the identity-heavy gain. Goal 4 therefore helps, but easy identity simplifications explain a large share of the aggregate improvement.

## Operator-Family Analysis

- top success family: `positive_real_formal:mixed_exp+log[log+exp]`
- top failure family: `positive_real_formal:exp[Mul+exp]`

| Mode | Family | Contains | Count | Median optimized alpha | Median gain | Improved | Timeout | Validation failure |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `positive_real_formal` | `mixed_exp+log` | `log+exp` | 8 | 0.3333333333333333 | 6.0 | 100.000% | 0.000% | 0.000% |
| `positive_real_formal` | `exp` | `log+exp` | 19 | 0.75 | 2.3333333333333335 | 100.000% | 0.000% | 5.263% |
| `positive_real_formal` | `mixed_exp+log` | `Mul+log+exp` | 6 | 2.428571428571429 | 1.8157894736842106 | 100.000% | 0.000% | 0.000% |
| `positive_real_formal` | `log` | `log+exp` | 2 | 1.25 | 1.8 | 100.000% | 0.000% | 0.000% |
| `positive_real_formal` | `exp` | `Mul+log+exp` | 110 | 2.6666666666666665 | 1.5555555555555556 | 87.356% | 1.818% | 20.909% |
| `positive_real_formal` | `log` | `Mul+log+exp` | 32 | 3.0 | 1.5153256704980844 | 93.750% | 0.000% | 0.000% |
| `positive_real_formal` | `mixed_Mul+exp` | `Mul+log+exp` | 57 | 3.0 | 1.5 | 92.000% | 0.000% | 12.281% |
| `positive_real_formal` | `mixed_Mul+log` | `Mul+log+exp` | 40 | 3.25 | 1.4615384615384615 | 94.595% | 7.500% | 7.500% |
| `positive_real_formal` | `Mul` | `Mul+log` | 146 | 3.625 | 1.4523809523809523 | 94.737% | 8.904% | 8.904% |
| `positive_real_formal` | `exp` | `Mul+exp` | 88 | 3.0 | 1.4421906693711968 | 63.889% | 0.000% | 59.091% |
| `positive_real_formal` | `mixed_Mul+exp+log` | `Mul+log+exp` | 70 | 3.4 | 1.4210526315789473 | 82.609% | 0.000% | 1.429% |
| `positive_real_formal` | `Mul` | `Mul+log+exp` | 248 | 3.4226190476190474 | 1.4027027027027028 | 90.566% | 10.887% | 14.516% |

Add/Mul-heavy and mixed-operator expressions still dominate much of the remaining difficulty because algebraic source simplification does not remove the recursive official pure EML expansion cost in the general case. Pure `exp` groups are small but validation-heavy in the mined failure ranking; larger mixed families such as `Mul+exp`, `Add+exp`, and Add/Mul/log/exp mixtures remain important failure regimes.

## Runtime And Timeout Analysis

| Mode | Median runtime seconds | Mean runtime seconds | Timeout count | All-v1 timeout rate | Nontrivial timeout rate | Identity-heavy timeout rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `safe` | 0.03714627100271173 | 0.09217351757819124 | 241 | 2.410% | 0.884% | 3.319% |
| `positive_real_formal` | 0.04460766648116987 | 0.14658737937592958 | 522 | 5.220% | 1.286% | 7.563% |

Timeout rows are retained in the CSV/JSONL artifacts. They are included in processed counts, timeout rates, and failure summaries rather than silently dropped.

## Success And Failure Case Studies

Top safe-mode successes:

| Rank | Index | Mode | Subset | Original EML-DAG | Extracted EML-DAG | Gain | Optimized alpha | Threshold improved | Expression |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1491 | `safe` | `identity_heavy_v1` | 27 | 1 | 27.0 | 0.2 | True | `(1*1)*x` |
| 2 | 5655 | `safe` | `identity_heavy_v1` | 27 | 1 | 27.0 | 0.2 | True | `(1*1)*y` |
| 3 | 308 | `safe` | `identity_heavy_v1` | 24 | 1 | 24.0 | 0.2 | True | `(y*1)*1` |
| 4 | 2360 | `safe` | `identity_heavy_v1` | 24 | 1 | 24.0 | 0.2 | True | `(1*x)*1` |
| 5 | 50 | `safe` | `identity_heavy_v1` | 16 | 1 | 16.0 | 0.3333333333333333 | True | `y*1` |

Top positive-real successes:

| Rank | Index | Mode | Subset | Original EML-DAG | Extracted EML-DAG | Gain | Optimized alpha | Threshold improved | Expression |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 204 | `positive_real_formal` | `identity_heavy_v1` | 64 | 1 | 64.0 | 0.07142857142857142 | True | `1*x + ((y*y)*(x + y))*log(1)` |
| 2 | 4609 | `positive_real_formal` | `identity_heavy_v1` | 61 | 1 | 61.0 | 0.07692307692307693 | True | `exp((x*x + y*y)*log(1*1))` |
| 3 | 4458 | `positive_real_formal` | `identity_heavy_v1` | 59 | 1 | 59.0 | 0.07142857142857142 | True | `y + ((x + 1)*(y + 1))*log(1*1)` |
| 4 | 6286 | `positive_real_formal` | `identity_heavy_v1` | 59 | 1 | 59.0 | 0.07142857142857142 | True | `exp(((x + x)*log(1))*(y*y + exp(1)))` |
| 5 | 4481 | `positive_real_formal` | `identity_heavy_v1` | 56 | 1 | 56.0 | 0.07692307692307693 | True | `y + ((y + 1)*exp(x))*log(1*1)` |

Top nontrivial successes:

| Rank | Index | Mode | Subset | Original EML-DAG | Extracted EML-DAG | Gain | Optimized alpha | Threshold improved | Expression |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 9315 | `positive_real_formal` | `nontrivial_v1` | 26 | 14 | 1.8571428571428572 | 2.3333333333333335 | False | `log(exp(1)*exp(y))` |
| 2 | 2939 | `positive_real_formal` | `nontrivial_v1` | 28 | 16 | 1.75 | 2.6666666666666665 | False | `log(exp(x)*exp(y))` |
| 3 | 2465 | `positive_real_formal` | `nontrivial_v1` | 26 | 15 | 1.7333333333333334 | 3.0 | False | `log(x*exp(x))` |
| 4 | 5874 | `positive_real_formal` | `nontrivial_v1` | 26 | 15 | 1.7333333333333334 | 3.0 | False | `log(y*exp(y))` |
| 5 | 5834 | `positive_real_formal` | `nontrivial_v1` | 36 | 21 | 1.7142857142857142 | 2.3333333333333335 | False | `log((y + 1)*exp(y)) + 1` |

Top identity-heavy successes:

| Rank | Index | Mode | Subset | Original EML-DAG | Extracted EML-DAG | Gain | Optimized alpha | Threshold improved | Expression |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 204 | `positive_real_formal` | `identity_heavy_v1` | 64 | 1 | 64.0 | 0.07142857142857142 | True | `1*x + ((y*y)*(x + y))*log(1)` |
| 2 | 4609 | `positive_real_formal` | `identity_heavy_v1` | 61 | 1 | 61.0 | 0.07692307692307693 | True | `exp((x*x + y*y)*log(1*1))` |
| 3 | 4458 | `positive_real_formal` | `identity_heavy_v1` | 59 | 1 | 59.0 | 0.07142857142857142 | True | `y + ((x + 1)*(y + 1))*log(1*1)` |
| 4 | 6286 | `positive_real_formal` | `identity_heavy_v1` | 59 | 1 | 59.0 | 0.07142857142857142 | True | `exp(((x + x)*log(1))*(y*y + exp(1)))` |
| 5 | 4481 | `positive_real_formal` | `identity_heavy_v1` | 56 | 1 | 56.0 | 0.07692307692307693 | True | `y + ((y + 1)*exp(x))*log(1*1)` |

Top safe-mode failures:

| Rank | Index | Mode | Subset | Original EML-DAG | Extracted EML-DAG | Gain | Optimized alpha | Threshold improved | Expression |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 5487 | `safe` | `identity_heavy_v1` | 50 |  |  |  | False | `y*(x*(1*(x*y)))` |
| 2 | 3428 | `safe` | `identity_heavy_v1` | 71 |  |  |  | False | `y*(((1*x)*(x*y))*log(exp(x)))` |
| 3 | 5397 | `safe` | `identity_heavy_v1` | 74 |  |  |  | False | `((1*x)*exp(1))*((x*y)*(x + y))` |
| 4 | 9523 | `safe` | `identity_heavy_v1` | 63 |  |  |  | False | `y*(((1*x)*(y + y))*log(x))` |
| 5 | 4122 | `safe` | `identity_heavy_v1` | 68 |  |  |  | False | `((x*1)*exp(x))*((x*y)*log(1))` |

Top positive-real failures:

| Rank | Index | Mode | Subset | Original EML-DAG | Extracted EML-DAG | Gain | Optimized alpha | Threshold improved | Expression |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 6065 | `positive_real_formal` | `identity_heavy_v1` | 45 |  |  |  | False | `y*((x*y)*log(1))` |
| 2 | 5487 | `positive_real_formal` | `identity_heavy_v1` | 50 |  |  |  | False | `y*(x*(1*(x*y)))` |
| 3 | 3428 | `positive_real_formal` | `identity_heavy_v1` | 71 |  |  |  | False | `y*(((1*x)*(x*y))*log(exp(x)))` |
| 4 | 2491 | `positive_real_formal` | `identity_heavy_v1` | 48 |  |  |  | False | `((x*y)*exp(1))*log(1)` |
| 5 | 5397 | `positive_real_formal` | `identity_heavy_v1` | 74 |  |  |  | False | `((1*x)*exp(1))*((x*y)*(x + y))` |

Best operator-signature groups:

| Rank | Mode | Signature | Count | Success | Median optimized alpha | Median gain | Improved | Timeout | Validation failure |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `positive_real_formal` | `exp+log` | 29 | 28 | 0.6 | 2.3333333333333335 | 100.000% | 0.000% | 3.448% |
| 2 | `positive_real_formal` | `Mul+log` | 185 | 172 | 3.625 | 1.4482758620689655 | 90.116% | 7.027% | 7.027% |
| 3 | `positive_real_formal` | `Mul+exp+log` | 563 | 493 | 3.25 | 1.4444444444444444 | 89.655% | 5.684% | 12.433% |
| 4 | `positive_real_formal` | `leaf_only` | 3 | 3 | 1.0 | 1.0 | 0.000% | 0.000% | 0.000% |
| 5 | `safe` | `leaf_only` | 3 | 3 | 1.0 | 1.0 | 0.000% | 0.000% | 0.000% |

Worst operator-signature groups:

| Rank | Mode | Signature | Count | Success | Median optimized alpha | Median gain | Improved | Timeout | Validation failure |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `positive_real_formal` | `exp` | 12 | 7 | 1.3333333333333333 | 1.0 | 0.000% | 0.000% | 41.667% |
| 2 | `safe` | `exp` | 12 | 7 | 1.3333333333333333 | 1.0 | 0.000% | 0.000% | 41.667% |
| 3 | `positive_real_formal` | `Mul+exp` | 307 | 225 | 3.8 | 1.3333333333333333 | 80.444% | 0.977% | 26.710% |
| 4 | `safe` | `Mul+exp` | 307 | 248 | 4.0 | 1.2242857142857142 | 70.565% | 0.977% | 19.218% |
| 5 | `positive_real_formal` | `Add+exp` | 290 | 252 | 3.1666666666666665 | 1.0 | 24.206% | 0.345% | 13.103% |

## Semantic And Provenance Audit

- audit rows: `28`
- all semantic validation valid: `True`
- all EML-DAG validation valid: `True`
- all structural purity valid: `True`
- safe branch-sensitive applications: `0`
- positive-real branch-sensitive applications: `5`
- provenance invalid count: `0`
- SymPy simplify rewrite path free: `True`
- audit JSON: `outputs/v1/goal4_egraph_semantic_audit.json`
- audit CSV: `outputs/v1/goal4_egraph_semantic_audit.csv`
- audit docs: `docs/goal4/GOAL4_EGRAPH_SEMANTIC_AUDIT.md`

The audit checks selected expressions in both modes, records rewrite provenance by rule name and tier, confirms safe mode does not apply branch-sensitive rules, and verifies the EML-DAG evaluator agrees with positive-real numeric probes. SymPy `simplify` is allowed only as an optional diagnostic outside the rewrite path.

## Integrity Statement

Final EML outputs remain official pure EML after extraction. The pipeline checks for:

- no derived leaves
- no hidden compound-expression leaves
- no fake macro leaves
- no macro/template EML nodes
- no modified official EML compiler formulas
- internal EML nodes labeled only `eml`
- EML leaves restricted to variables or constant `1`
- no SymPy.simplify rewrite shortcut

Improvements are structural non-ML compression results, not GNN or neural-model evidence.

## Recommendation For Goal 5

Non-ML e-graph compression is a useful baseline and should remain in the evaluation stack, but it is not enough by itself. The threshold pass rate remains low after safe mode (`1.020%`) and positive-real mode (`5.827%`), while nontrivial positive-real median gain is `1.0`. Goal 5 should therefore still investigate ML-facing motif or macro compression, with honest separation between structural compression results and future model-performance claims.

Prepare ML-facing graph representations for later experiments, but do not start Goal 5 here. The next graph surfaces should keep separate views for source AST trees/DAGs, official pure EML DAGs, e-graph-optimized source ASTs, and e-graph-optimized official pure EML DAGs, with rule mode, assumptions, subset labels, validation status, and rewrite provenance carried as metadata.

## Reproducible Command

```bash
.venv/bin/python -m geml.experiments.run_goal4_egraph_pipeline --config configs/egraph_compression_v1.yaml
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format . --check
```
