# Goal 4.5 E-Graph Compression Audit

This report audits selected expressions before any 10k v1 Goal 4 run. It is a
semantic and compression sanity check for algebraic e-graph rewriting, not a
large-scale corpus result.

## Artifacts

- CSV: `outputs/v1/goal4_egraph_audit.csv`
- JSON: `outputs/v1/goal4_egraph_audit.json`
- report: `docs/goal4/GOAL4_EGRAPH_AUDIT.md`

## Method

- run modes: `safe`, `positive_real_formal`
- extractor: `exact_eml_dag_beam_cost`
- final EML metrics: official pure EML compiler, then exact Goal 3 EML-DAG
- positive-real results are branch-sensitive and reported separately
- no 10k pipeline, neural model, or visualization was run

## Key Checks

- `x+y` and `y+x` end in the same e-class in safe mode: `True`
- `x+y` and `y+x` end in the same e-class in positive-real mode: `True`
- `x+1` and `x+2-1` end in the same e-class in safe mode: `True`
- `x+1` and `x+2-1` end in the same e-class in positive-real mode: `True`
- `log(exp(x))` is equivalent to `x` in safe mode: `False`
- `log(exp(x))` is equivalent to `x` in positive-real mode: `True`
- `exp(log(x))` is equivalent to `x` in safe mode: `False`
- `exp(log(x))` is equivalent to `x` in positive-real mode: `True`
- `log(exp(x))` extracted expression in safe mode: `Log(Exp(x))`
- `log(exp(x))` extracted expression in positive-real mode: `x`
- `exp(log(x))` extracted expression in safe mode: `Exp(Log(x))`
- `exp(log(x))` extracted expression in positive-real mode: `x`
- `x**2` safe-mode EML-DAG nodes: `29 -> 29`
- `x**2` positive-real EML-DAG nodes: `29 -> 29`

## Purity And Worsening

- all final EML outputs pure: `True`
- rows with worse extracted EML-DAG size: `0`
- no audited extraction worsened official pure EML-DAG size

## Audit Table

| Expression | Mode | Extracted | Original D_EML | Extracted D_EML | Gain | Status | Pure |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `x + y` | `safe` | `Add(x,y)` | 16 | 16 | 1 | `completed` | `True` |
| `x + y` | `positive_real_formal` | `Add(x,y)` | 16 | 16 | 1 | `completed` | `True` |
| `y + x` | `safe` | `Add(x,y)` | 16 | 16 | 1 | `completed` | `True` |
| `y + x` | `positive_real_formal` | `Add(x,y)` | 16 | 16 | 1 | `completed` | `True` |
| `x + 1` | `safe` | `Add(1,x)` | 14 | 14 | 1 | `completed` | `True` |
| `x + 1` | `positive_real_formal` | `Add(1,x)` | 14 | 14 | 1 | `completed` | `True` |
| `x + 2 - 1` | `safe` | `Add(1,x)` | 22 | 14 | 1.57143 | `completed` | `True` |
| `x + 2 - 1` | `positive_real_formal` | `Add(1,x)` | 22 | 14 | 1.57143 | `completed` | `True` |
| `x * 1` | `safe` | `x` | 19 | 1 | 19 | `completed` | `True` |
| `x * 1` | `positive_real_formal` | `x` | 19 | 1 | 19 | `completed` | `True` |
| `x * x` | `safe` | `Mul(x,x)` | 19 | 19 | 1 | `completed` | `True` |
| `x * x` | `positive_real_formal` | `Mul(x,x)` | 19 | 19 | 1 | `completed` | `True` |
| `x**2` | `safe` | `Pow(x,2)` | 29 | 29 | 1 | `completed` | `True` |
| `x**2` | `positive_real_formal` | `Pow(x,2)` | 29 | 29 | 1 | `completed` | `True` |
| `(x + 1) * (x + 1)` | `safe` | `Mul(Add(1,x),Add(1,x))` | 25 | 25 | 1 | `completed` | `True` |
| `(x + 1) * (x + 1)` | `positive_real_formal` | `Mul(Add(1,x),Add(1,x))` | 25 | 25 | 1 | `completed` | `True` |
| `log(x) + log(x)` | `safe` | `Add(Log(x),Log(x))` | 18 | 18 | 1 | `completed` | `True` |
| `log(x) + log(x)` | `positive_real_formal` | `Add(Log(x),Log(x))` | 18 | 18 | 1 | `completed` | `True` |
| `log(exp(x))` | `safe` | `Log(Exp(x))` | 6 | 6 | 1 | `completed` | `True` |
| `log(exp(x))` | `positive_real_formal` | `x` | 6 | 1 | 6 | `completed` | `True` |
| `exp(log(x))` | `safe` | `Exp(Log(x))` | 6 | 6 | 1 | `completed` | `True` |
| `exp(log(x))` | `positive_real_formal` | `x` | 6 | 1 | 6 | `completed` | `True` |
| `log(x*y)` | `safe` | `Log(Mul(x,y))` | 26 | 26 | 1 | `completed` | `True` |
| `log(x*y)` | `positive_real_formal` | `Add(Log(x),Log(y))` | 26 | 22 | 1.18182 | `completed` | `True` |
| `exp(x+y)` | `safe` | `Exp(Add(x,y))` | 17 | 17 | 1 | `completed` | `True` |
| `exp(x+y)` | `positive_real_formal` | `Exp(Add(x,y))` | 17 | 17 | 1 | `completed` | `True` |
| `((x*x)*(y*y))*((x*x)*(x + 1))` | `safe` | `Mul(Add(1,x),Mul(Mul(Mul(Mul(x,x),Mul(x,x)),y),y))` | 70 | 70 | 1 | `completed` | `True` |
| `((x*x)*(y*y))*((x*x)*(x + 1))` | `positive_real_formal` | `Mul(Add(1,x),Mul(Mul(Mul(Mul(x,x),Mul(x,x)),y),y))` | 70 | 70 | 1 | `completed` | `True` |

## Interpretation Boundary

This is a selected-expression audit only. It confirms that the current Goal 4
infrastructure preserves pure EML outputs on these cases and that
positive-real log/exp simplifications remain separated from safe mode.
It does not prove global minimal EML form and does not replace the future
v1 corpus-scale Goal 4 run.
