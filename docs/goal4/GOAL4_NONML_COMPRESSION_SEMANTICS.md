# Goal 4.0 Non-ML Compression Semantics

Goal 4 introduces non-ML algebraic compression before EML compilation. It is a
specification stage only. It does not implement an e-graph engine, neural
models, visualization, or any change to the official pure EML compiler formulas.

## Central Question

Goal 2 showed that raw official pure EML trees are representation-complete but
structurally expensive. Goal 3 showed that exact structural EML-DAG compression
helps materially, but does not broadly rescue EML under the current structural
threshold.

Goal 4 asks:

```text
Can non-ML equivalence search reduce the final official pure EML-DAG size by
finding smaller equivalent source expressions before EML compilation?
```

## Corpus Baseline

Goal 4 result-bearing runs must use the repaired v1 corpus under `outputs/v1/`.
The v0 corpus is a pilot corpus and is deprecated for results after Goal 3R.
E-graph runs on v0 may be used for engine debugging or diagnostics only, and
reports must label them as diagnostic-only.

The default subset label is `all_v1`. Reserved optional future subset labels
are:

- `all_v1`
- `nontrivial_v1`
- `identity_heavy_v1`

The main object of study is not raw source AST size. The main object is the
official pure EML DAG obtained after extracting a source expression, compiling
that expression through the official pure EML compiler, and applying exact
structural DAG sharing to the resulting pure EML tree.

## Representation Modes

Goal 4 keeps the Goal 3 exact structural modes and adds separate e-graph modes.
Every output row, report, and plot must name the mode used.

- `ast_tree`: the ordinary binary AST tree emitted by the source AST converter.
- `ast_dag`: an exact structural DAG obtained by sharing identical AST subtrees
  from `ast_tree`.
- `restricted_eml_pure_tree`: the official recursive pure restricted EML tree
  emitted by the official compiler.
- `restricted_eml_pure_dag`: an exact structural DAG obtained by sharing
  identical pure EML subtrees from `restricted_eml_pure_tree`.
- `egraph_safe_ast`: a source AST expression extracted from an e-graph using
  only the safe rule policy for the current run.
- `egraph_safe_eml_dag`: the exact structural pure EML DAG obtained by compiling
  `egraph_safe_ast` through the official pure EML compiler and then applying
  exact structural EML-DAG sharing.
- `egraph_positive_real_ast`: a source AST expression extracted from an e-graph
  using a separately labeled positive-real formal rule policy.
- `egraph_positive_real_eml_dag`: the exact structural pure EML DAG obtained by
  compiling `egraph_positive_real_ast` through the official pure EML compiler
  and then applying exact structural EML-DAG sharing.

Goal 4 is no longer exact structural compression only. Goal 4 uses algebraic
rewrite rules before official EML compilation. Therefore, Goal 4 metrics must
not be mixed with Goal 3 exact-DAG metrics unless the mode is named explicitly.
For example, `restricted_eml_pure_dag` and `egraph_safe_eml_dag` are different
measurement modes even when both end as pure EML DAGs.

## GEML Integrity Requirements

Goal 4 may rewrite source expressions, but it must not hide complexity inside
the final EML representation.

The following remain forbidden:

- derived leaves
- hidden compound-expression leaves
- fake macro leaves
- hiding `Add`, `Mul`, `Log`, `Exp`, or any other source operator inside EML
  leaves
- final EML DAG nodes labeled as helper macros such as `eml_add`, `eml_mul`,
  `eml_log`, `eml_exp`, `eml_zero`, or `eml_int`
- treating an e-graph rewrite as a substitute for official EML compilation

Final EML outputs must still compile through the official pure EML compiler.
Final pure EML trees and DAGs must still have only:

- `eml` internal nodes
- source variable leaves
- constant `1` leaves

E-graph source expressions may contain normal operators such as `Add`, `Mul`,
`Pow`, `exp`, and `log`. Final EML metrics must be computed only after the
extracted source expression is compiled through the official pure EML compiler.
If the extracted expression is not accepted by the official compiler, that
candidate is not a valid EML metric row.

## Rule Tiers

Every e-graph run must record the rule tier used and the rule set version.
Rules outside the selected tier are not available to the extractor.

### Tier A: Safe Algebraic Rules

Tier A contains algebraic rules intended to be safe for the current symbolic
setting without relying on log/exp inverse identities or positive-real
assumptions.

- commutativity of `Add` and `Mul`
- associativity of `Add` and `Mul`
- additive identity: `a + 0 = a`
- multiplicative identity: `a * 1 = a`
- multiplication by zero: `a * 0 = 0`
- double negation: `-(-a) = a`
- subtraction as addition of negation where subtraction is represented
- simple constant folding over exact integers and rationals

### Tier B: Guarded Or Cautious Algebraic Rules

Tier B rules may be used only when their guard is satisfied by the run's
assumption policy or by the local expression structure. They must not silently
become default safe rewrites.

- `a - a = 0`
- `a / a = 1` only if nonzero assumptions exist
- `x**1 = x`
- `x**0 = 1` only under the appropriate nonzero/domain policy
- `x**2 = x*x` only if the cost extractor decides it is cheaper

If a Tier B rule fires, the output row must make the guard policy auditable.

### Tier C: Positive-Real Formal Log/Exp Rules

Tier C rules are not universal complex-domain identities. They may be used only
in a separately labeled mode with:

```text
rule_tier = positive_real_formal_rules
```

Examples:

- `log(1) = 0`
- `exp(0) = 1`
- `log(exp(a)) = a`
- `exp(log(a)) = a` under a positive-real assumption
- `log(a*b) = log(a) + log(b)` under a positive-real assumption
- `exp(a+b) = exp(a)*exp(b)`

Tier C outputs must be reported as positive-real formal results, not as
domain-universal algebraic facts.

## Forbidden Default Rewrites

The following rewrites are unsafe as default safe-mode rewrites:

- Do not use `log(exp(x)) = x` in safe mode.
- Do not use `exp(log(x)) = x` in safe mode unless positive-real mode is
  selected.
- Do not use `log(a*b) = log(a)+log(b)` in safe mode.
- Do not use `a/a = 1` without nonzero assumptions.
- Do not use `sqrt` or power branch rewrites unless a rule tier explicitly
  supports them.

These rules may appear only under a named guarded or positive-real rule policy.

## Extractor Objective

The main extractor must be EML-aware.

Primary target:

```text
minimize |D_EML(extracted_expression)|
```

where `D_EML` is computed as:

```text
extracted AST
  -> official pure EML tree
  -> exact structural EML DAG
```

A candidate extraction is valid for the primary objective only if the official
pure EML compiler accepts it and the final EML DAG passes the pure EML integrity
checks. Candidates that fail official compilation or pure-DAG validation must
receive invalid or infinite EML cost.

Do not extract by ordinary AST node count and claim that result is EML-optimal.
AST-cost extraction may exist only as a baseline mode. Reports must name it as
an AST-cost baseline and must compare it against the EML-aware extractor.

## Required Cost Metrics

Goal 4 output rows must preserve the Goal 3 baseline metrics and add optimized
metrics for the extracted expression.

For the original expression, record:

- `source_ast_tree_nodes_original`
- `source_ast_dag_nodes_original`
- `official_pure_eml_tree_nodes_original`
- `official_pure_eml_dag_nodes_original`
- `tree_alpha`
- `dag_alpha_vs_ast_tree`
- `dag_alpha_vs_ast_dag`

For the extracted expression, record:

- `source_ast_tree_nodes_extracted`
- `source_ast_dag_nodes_extracted`
- `official_pure_eml_tree_nodes_extracted`
- `official_pure_eml_dag_nodes_extracted`
- `optimized_dag_alpha_vs_ast_tree`
- `optimized_dag_alpha_vs_ast_dag`
- `compression_gain_vs_goal3_dag`

Definitions:

```text
tree_alpha =
  official_pure_eml_tree_nodes_original / source_ast_tree_nodes_original

dag_alpha_vs_ast_tree =
  official_pure_eml_dag_nodes_original / source_ast_tree_nodes_original

dag_alpha_vs_ast_dag =
  official_pure_eml_dag_nodes_original / source_ast_dag_nodes_original

optimized_dag_alpha_vs_ast_tree =
  official_pure_eml_dag_nodes_extracted / source_ast_tree_nodes_original

optimized_dag_alpha_vs_ast_dag =
  official_pure_eml_dag_nodes_extracted / source_ast_dag_nodes_original

compression_gain_vs_goal3_dag =
  official_pure_eml_dag_nodes_original / official_pure_eml_dag_nodes_extracted
```

The optimized alpha denominators use the original AST baselines so the Goal 4
result measures whether algebraic extraction reduced final EML-DAG size relative
to the same source problem measured in Goal 3. Additional extracted-denominator
ratios may be reported, but they must be named separately.

## E-Graph Resource Limits

Every e-graph run must expose and save resource limits:

- `max_iterations`
- `max_e_nodes`
- `max_e_classes`
- `timeout_seconds`
- `saturation_status`
- `extraction_status`

Saturation and extraction are distinct statuses. A run may time out before
saturation but still produce an extracted expression. Reports must distinguish:

- saturation completed
- saturation stopped by iteration limit
- saturation stopped by e-node limit
- saturation stopped by e-class limit
- saturation stopped by timeout
- extraction completed
- extraction failed
- extracted expression failed official EML compilation
- extracted expression failed semantic validation

## Required Output Fields

All Goal 4 outputs must record:

- `representation_mode`
- `rule_tier`
- `rule_set_version`
- `egraph_limits`
- `saturation_completed`
- `saturation_status`
- `extraction_completed`
- `extraction_status`
- `original_expression`
- `original_srepr`
- `extracted_expression`
- `extracted_srepr`
- `semantic_validation_status`
- `semantic_validation_method`
- `official_eml_compilation_status`
- `pure_eml_dag_validation_status`

Semantic validation must be interpreted under the selected rule policy. Safe
mode should validate without positive-real log/exp inverse assumptions.
Positive-real formal mode must state the positive-real assumption and must not be
reported as a universal complex-domain proof.

## Interpretation Warning

E-graph optimization is rule-set dependent. Goal 4 does not prove a global
minimal EML form unless exhaustive search under a finite grammar is proven.

A smaller extracted EML DAG is evidence that the selected rule set and extractor
found a better representation under the recorded limits. It is not evidence that
no smaller equivalent expression exists.

## Non-Goals

Goal 4.0 does not implement:

- an e-graph engine
- neural models
- visualization
- equivalence-pair generation
- a rewrite proof system
- changes to the official EML compiler formulas
