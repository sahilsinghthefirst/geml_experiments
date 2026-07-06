# AGENTS.md

## Project Workflow

- Development happens in this repository: `sahilsinghthefirst/geml_experiments`.
- After every stage or mini stage of development, commit the new changes and push them to the remote repository.
- Keep commits scoped to the completed stage whenever practical.

## Project Name

GEML-v0: EML-Native Graph Models for Symbolic Mathematical Reasoning

## Project Goal

This repo builds a small-scale experimental benchmark for testing whether EML operator-collapse can improve neural symbolic reasoning despite possible EML tree-expansion.

The central research question is:

> When does EML operator-collapse beat tree-expansion for neural symbolic reasoning?

Do not assume EML is automatically more efficient. The whole point of this repo is to test that claim rigorously.

---

## Scientific Framing

GEML uses the EML operator:

[
\operatorname{eml}(x,y)=\exp(x)-\ln(y)
]

The idea is to convert standard mathematical expressions into binary trees where every internal node is the same EML operation.

The hypothesis is that reducing operator diversity may help graph/tree neural models learn symbolic equivalence, simplification, and rewrite behavior.

However, the main risk is that EML may greatly increase tree size. Therefore, all experiments must measure whether the benefit from reduced operator complexity outweighs the cost of larger trees.

---

## Main Experimental Question

For every expression (E), compute:

[
\alpha(E)=\frac{|T_{\text{EML}}(E)|}{|T_{\text{AST}}(E)|}
]

Then compare EML performance against the theoretical threshold:

[
\alpha < 1+\log_{4L}(K)
]

where:

* (K) is the number of standard operator types.
* (L) is the number of leaf symbol/constant choices.
* (\alpha) is the EML expansion factor.

The project should test whether EML helps when (\alpha) is below this threshold and fails when tree expansion dominates.

---

## Core Rule

Do not claim that EML is better because it has fewer operators.

Instead, always frame results as:

> EML reduces operator diversity but may increase structural complexity. We test when the tradeoff is beneficial.

---

## Required Baselines

Every serious experiment must compare against strong baselines.

Required representations:

1. Prefix/LaTeX transformer
2. AST-GNN
3. AST-DAG GNN
4. Pure EML-tree GNN
5. Compressed EML-DAG GNN

Without AST-GNN and AST-DAG baselines, it is impossible to know whether gains come from EML specifically or simply from using graph structure.

---

## High-Level Roadmap

### Goal 1 — Core Representation + Data Generation

Build the foundation for generating and representing mathematical expressions.

Deliverables:

1. SymPy expression generator with bounded depth
2. Normal AST binary-tree converter
3. Restricted EML binary-tree converter
4. Tree statistics:

   * node count
   * edge count
   * depth
   * branching
   * leaf count
   * operator count
5. Dataset generator for expression pairs
6. JSONL/CSV export format
7. Unit tests for all major components

Initial expression subset:

[
x,\ y,\ 1,\ +,\ \times,\ \exp,\ \log
]

Do not start with trigonometry. Add trig only after the core pipeline works.

Success criteria:

* Generate 10,000+ valid expressions.
* Convert each expression into AST and restricted EML form.
* Compute (|T_{\text{AST}}|), (|T_{\text{EML}}|), and (\alpha = |T_{\text{EML}}| / |T_{\text{AST}}|).
* Create equivalence and non-equivalence pairs.

---

### Goal 2 — Expansion-Factor Study

Before training serious models, measure whether EML is structurally reasonable.

For every expression, compute:

[
\alpha=\frac{|T_{\text{EML}}|}{|T_{\text{AST}}|}
]

Compare this to:

[
\alpha < 1+\log_{4L}(K)
]

Deliverables:

1. `expansion_stats.csv`
2. Alpha histogram
3. Alpha by expression depth
4. Alpha by operator family
5. List of expressions where EML explodes badly
6. Summary report describing the “safe EML regime”

Success criteria:

* Identify expression classes where (\alpha) is below threshold.
* Identify expression classes where EML expansion dominates.
* Define the safe first-experiment regime for EML-GNN training.

---

### Goal 3 — Small Equivalence Dataset

Create the first ML task.

Task:

[
(E_1,E_2)\rightarrow \text{equivalent or not equivalent}
]

Datasets to create:

1. Shallow arithmetic/log/exp equivalence
2. Medium-depth generated equivalence
3. Hard negative pairs
4. Out-of-distribution depth split

Example positive pairs:

[
\log(ab)\equiv \log a+\log b
]

[
e^{a+b}\equiv e^ae^b
]

[
(x+1)^2\equiv x^2+2x+1
]

Example negative pairs:

[
\log(a+b)\not\equiv \log a+\log b
]

[
e^{ab}\not\equiv e^ae^b
]

Success criteria:

* 50,000+ train pairs
* 5,000+ validation pairs
* 5,000+ test pairs
* OOD test set by higher depth or unseen identity families
* All pairs verified by SymPy simplification and randomized numeric checks where appropriate

---

### Goal 4 — Train First Small GNN

Train a small model first. Do not begin with large foundation models.

Initial architecture:

Siamese GNN:

[
G_1 \rightarrow \text{encoder} \rightarrow z_1
]

[
G_2 \rightarrow \text{encoder} \rightarrow z_2
]

[
[z_1,z_2,|z_1-z_2|,z_1\odot z_2]\rightarrow \text{classifier}
]

Start with:

* GIN or GraphSAGE
* 3–5 message-passing layers
* hidden size 128 or 256
* batch size based on graph size

Initial representations:

1. AST-GNN
2. EML-tree GNN

Success criteria:

* Model trains without graph/data bugs.
* AST-GNN and EML-GNN both solve shallow equivalence above random baseline.
* Accuracy can be compared as a function of (\alpha).

---

### Goal 5 — Baseline Phase

Run serious comparisons across representations.

Baselines:

1. Prefix transformer
2. AST-GNN
3. AST-DAG GNN
4. Pure EML-tree GNN
5. Compressed EML-DAG GNN

Main question:

[
\text{Does EML help when } \alpha < 1+\log_{4L}(K)?
]

Required metrics:

* accuracy
* F1 score
* OOD accuracy
* training time
* inference time
* GPU memory
* average graph size
* average alpha
* runtime per graph
* memory per graph

Expected result table:

| Representation     | Accuracy | OOD Accuracy | Runtime | Memory | Avg Alpha |
| ------------------ | -------: | -----------: | ------: | -----: | --------: |
| Prefix Transformer |          |              |         |        |       N/A |
| AST-GNN            |          |              |         |        |       N/A |
| AST-DAG GNN        |          |              |         |        |       N/A |
| EML-tree GNN       |          |              |         |        |           |
| EML-DAG GNN        |          |              |         |        |           |

Success criteria:

* Determine exactly where EML helps, matches, or fails.
* Separate “graph representation benefit” from “EML representation benefit.”
* Test whether DAG compression rescues EML tree explosion.

---

### Goal 6 — DAG Compression

Implement shared-subexpression compression.

Example:

[
(x+1)^2+(x+1)^3
]

A normal tree repeats:

[
(x+1)
]

A DAG stores it once and reuses it.

Deliverables:

1. AST-DAG converter
2. EML-DAG converter
3. Compression-ratio metric
4. Experiments comparing tree vs DAG

Metrics:

[
\text{DAG compression ratio}=\frac{|T|}{|D|}
]

[
\text{EML-DAG gain}=\frac{|T_{\text{EML}}|}{|D_{\text{EML}}|}
]

Success criteria:

* EML-DAG significantly reduces node count for repeated-subexpression cases.
* Determine whether EML-DAG beats raw EML-tree.
* Determine whether DAG compression changes the useful (\alpha) regime.

---

### Goal 7 — Add Trigonometry Carefully

Only add trigonometry after the arithmetic/log/exp pipeline works.

Add:

[
\sin,\ \cos,\ \tan
]

Test identities such as:

[
\sin^2x+\cos^2x\equiv 1
]

[
\sin(2x)\equiv 2\sin x\cos x
]

[
\cos(2x)\equiv \cos^2x-\sin^2x
]

Treat trig as a stress test because it may cause large EML expansion.

Success criteria:

* Measure alpha explosion for trig.
* Determine whether DAG compression helps.
* Determine whether macro-rewrite compression helps.
* Do not overclaim if raw EML fails on trig-heavy expressions.

---

### Goal 8 — Rewrite-Path Prediction

After equivalence classification works, move toward proof-like behavior.

Task:

[
E_{\text{current}}\rightarrow E_{\text{next}}
]

The model predicts the next valid rewrite step.

Example:

[
\log(ab)\rightarrow \log a+\log b
]

[
\sin^2x+\cos^2x\rightarrow 1
]

Deliverables:

1. Rewrite-rule library
2. Generated rewrite traces
3. Next-step prediction model
4. SymPy/CAS verifier
5. Short proof-path generation for symbolic identities

Success criteria:

* Model proposes valid rewrites.
* SymPy or another verifier checks each step.
* The system can generate short proof paths for symbolic identities.
* Invalid rewrites are rejected before they propagate.

---

## Implementation Standards

Use:

* Python 3.12
* type hints
* pytest
* ruff
* pydantic
* YAML configs
* JSONL/CSV outputs
* modular files
* reproducible experiment logs

Every new feature must include tests.

Do not add large dependencies without explaining why.

---

## Recommended Repo Structure

```text
GEML/
  README.md
  AGENTS.md
  pyproject.toml
  configs/
    data_v0.yaml
    expansion_v0.yaml
    equiv_ast.yaml
    equiv_eml.yaml
  geml/
    data/
      generate_exprs.py
      identities.py
      dataset.py
    symbolic/
      ast_graph.py
      dag_graph.py
      eml_nodes.py
      eml_transpile.py
      metrics.py
    models/
      graph_encoder.py
      siamese_equiv.py
      transformer_prefix.py
    train/
      train_equiv.py
      eval_equiv.py
    experiments/
      expansion_study.py
      baseline_grid.py
  outputs/
    v0/
      expansion_stats.csv
      results.csv
      plots/
  tests/
    test_expr_generator.py
    test_ast_graph.py
    test_dag_graph.py
    test_eml_transpile.py
    test_metrics.py
```

---

## Experiment Logging Rules

Every experiment must save:

1. Config YAML
2. Metrics CSV/JSON
3. Expansion factor statistics
4. Runtime
5. GPU memory if applicable
6. Git commit hash
7. Random seed
8. Dataset version
9. Model checkpoint metadata
10. Human-readable summary

No result should be reported unless it is reproducible from saved configs and scripts.

---

## First Codex Task

The first task is Goal 1 only.

Prompt:

```text
Build GEML-v0 Goal 1: core expression representation and data generation.

Implement:
1. SymPy expression generator with bounded depth.
2. Normal AST binary-tree converter.
3. Restricted EML binary-tree converter for variables, constants, Add, Mul, Exp, and Log.
4. Tree statistics: node count, edge count, depth, leaf count, operator count.
5. Alpha metric: |T_EML| / |T_AST|.
6. JSONL/CSV dataset export.
7. Unit tests for all converters and metrics.

Do not implement neural models yet.
Save outputs under outputs/v0/.
Use Python 3.12, type hints, pytest, ruff, pydantic, and YAML configs.
```

---

## What Not To Do

Do not start with:

* giant foundation models
* 1B–3B parameter training
* IMO theorem proving
* full geometry reasoning
* unsupported claims of GPT-level math ability
* trig-heavy experiments before the basic pipeline works
* EML-only experiments without AST/DAG baselines

---

## Claim Discipline

Avoid claims like:

> EML makes math reasoning efficient.

Use:

> EML may improve neural symbolic reasoning in regimes where operator-collapse outweighs tree-expansion.

Avoid:

> GEML solves theorem proving.

Use:

> GEML may serve as an algebraic equivalence and rewrite substrate inside a broader verifier-guided theorem-proving system.

Avoid:

> The model only needs to learn graph connectivity.

Use:

> The model learns over a homogeneous graph topology rather than heterogeneous operator labels, but still must learn variable identity, subtree roles, equivalence classes, and rewrite validity.

---

## Current Best Framing

The strongest project framing is:

> We introduce and evaluate EML-native graph representations for neural symbolic reasoning, testing when a single-operator mathematical substrate improves or degrades equivalence and rewrite learning relative to standard AST, DAG, and transformer baselines.

This is the framing all code, experiments, and writing should support.
