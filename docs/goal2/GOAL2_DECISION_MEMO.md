# Goal 2 Decision Memo

## Decision

Raw pure EML is representation-complete but structurally expensive; therefore, the next research step should test whether DAG compression and learned graph models recover enough structure to make EML useful.

Goal 2 should not be interpreted as evidence that GEML fails. It is evidence that the uncompressed pure EML tree representation is the wrong object to compare directly against ASTs for size or computational efficiency.

## What Goal 2 Shows

Raw official pure EML is not smaller than AST. In the fixed-seed 10,000-expression run, every expression compiled successfully to official pure EML, but expansion was large:

- mean alpha: `10.648995087903057`
- median alpha: `11.375`
- p90 alpha: `14.208333333333334`
- p95 alpha: `14.733333333333333`
- max alpha: `17.366666666666667`

Alpha almost never falls below the theoretical threshold:

- current grammar, `K=4`, `L=3`: `0.0%` below threshold
- generous vocabulary, `K=20`, `L=3`: `0.25%` below threshold
- larger vocabulary, `K=50`, `L=3`: `0.25%` below threshold

The conclusion is clear: raw pure EML trees are much larger than ASTs for this expression distribution.

## Worst Families

The worst structural failures are Add/Mul-heavy expressions. The highest-risk operator signatures were:

- `Add+Mul`
- `Add+Mul+log`
- `Add+Mul+exp`
- `Add+Mul+exp+log`

The dominant operator family with the highest median alpha was `Mul`. This matches the official macro structure: multiplication expands through logs, addition, and exp; addition expands through subtraction and negation; constants are also recursively built from `1`. Nested Add/Mul expressions therefore duplicate substantial macro structure.

## Least Bad Families

The least bad signatures were simple `exp` and `exp+log` families. `exp` is closest to the threshold because the official macro is just `EML(z, 1)`. Even so, its median alpha was still above the current threshold, so it is not a robust safe regime for raw pure EML.

## Does This Kill GEML?

No. It narrows the hypothesis.

Goal 2 rules out a naive claim: raw pure EML trees are not smaller or computationally cheaper than ASTs by default. But GEML’s interesting claim can still survive if EML becomes useful after graph sharing, compression, or learning.

The useful property of EML is not raw tree compactness. The useful property may be that all operations reduce to a uniform primitive, which could make learned graph representations simpler after repeated substructures are shared.

## Why DAG Compression Is Necessary

DAG compression is now necessary because the official pure compiler expands common source operations into repeated EML macro patterns. Those repeated patterns are visible throughout Add/Mul/log/exp combinations. A tree counts each repetition separately; a DAG can share repeated subtrees and reveal whether pure EML has a compact graph form even when its tree form is large.

The next size question is therefore:

```text
alpha_dag = |DAG_EML| / |DAG_AST or AST|
```

not raw tree alpha alone.

## Required Baselines

The next research step should include both AST-GNN and EML-DAG-GNN baselines.

AST-GNN is required because it tests whether graph learning over the ordinary symbolic representation already solves the task. Without this baseline, any EML result is uninterpretable.

EML-DAG-GNN is required because raw EML trees are too large. The fair EML test is whether a compressed EML DAG plus a learned graph model can recover enough shared structure to offset expansion and exploit the uniform `eml` primitive.

Only after those baselines exist should GEML move toward stronger ML claims.
