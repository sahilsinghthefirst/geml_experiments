# Goal 4.4 EML-Aware Extraction

Goal 4.4 adds expression-level extraction objectives for saturated e-graphs.
It does not run the 10k pipeline, train neural models, add visualization, or
modify the official pure EML compiler formulas.

## Central Objective

Goal 4 is not trying to find the smallest source AST. The primary objective is:

```text
minimize |D_EML(extracted_expression)|
```

where `D_EML` is computed by the official path:

```text
e-graph IR candidate
  -> SymPy expression without intentional normalization
  -> official pure EML tree
  -> exact structural EML DAG
```

The final selector for the main mode therefore evaluates candidates by official
pure EML-DAG node count, not by source AST node count.

## Extractor Modes

`ast_node_cost` is a baseline. It chooses the expression with the smallest
source IR tree node count. It must never be called EML-optimal.

`estimated_eml_cost` is an approximate local dynamic-programming estimate. It
uses operator weights only. It does not compile candidates and does not account
for exact structural DAG sharing. It is useful only for diagnostics and beam
ordering, not final Goal 4 headline numbers.

`exact_eml_dag_beam_cost` is the main extractor. It enumerates top-K candidates
from the root e-class, compiles each candidate through the official pure EML
compiler, converts the resulting EML tree to a Goal 3 exact structural DAG, and
selects the candidate with the smallest pure EML-DAG node count.

Tie-breaking order is:

1. `extracted_eml_dag_nodes`
2. `extracted_eml_tree_nodes`
3. `extracted_ast_dag_nodes`
4. `extracted_ast_tree_nodes`
5. stable expression string

## Required Configuration

Extraction records these configuration fields:

- `extractor_mode`
- `beam_size`
- `max_candidate_depth`
- `max_candidates_evaluated`
- `timeout_seconds`
- `allow_positive_real_rules`
- `rule_mode`

The default rule mode remains `safe`. `positive_real_formal` extraction requires
explicit opt-in through `allow_positive_real_rules=True`.

## Output Row Fields

Each extraction result records:

- `original_expression`
- `extracted_expression`
- `rule_mode`
- `extractor_mode`
- `beam_size`
- `max_candidate_depth`
- `max_candidates_evaluated`
- `timeout_seconds`
- `allow_positive_real_rules`
- `candidate_count`
- `selected_candidate_rank`
- `extracted_ast_tree_nodes`
- `extracted_ast_dag_nodes`
- `extracted_eml_tree_nodes`
- `extracted_eml_dag_nodes`
- `extraction_status`
- `extraction_timeout`
- `tie_break_info`
- `validation_status`

Rows for `positive_real_formal` mode also carry:

- `assumptions`
- `branch_sensitive_rules_used`
- `branch_sensitive_rule_count`
- `branch_sensitive_rule_names`

## Validation And Integrity

Extraction validates that the selected expression is still in the same root
e-class as the original expression. Numeric validation samples positive real
values and records the maximum absolute error. SymPy equality checks may remain
diagnostic-only validation helpers; they are not used as a rewrite engine or as
the extractor objective.

After extraction, the candidate is compiled through the official pure EML
compiler and checked for GEML integrity:

- no derived leaves
- no hidden compound leaves
- no macro or template nodes
- only `eml` internal nodes
- only source variables and constant `1` leaves
- exact EML-DAG metrics computed after official compilation

If official compilation fails, the candidate is not a valid exact EML-DAG
candidate. Timeout and candidate failure cases return status rows instead of
crashing the run.

## Reporting Boundary

Goal 4.4 is still expression-level infrastructure. Future large-scale results
should compare against the repaired v1 Goal 3 baselines under `outputs/v1/`.
Goal 4 metrics must name their mode, because algebraic e-graph extraction is no
longer exact structural compression only and must not be mixed with Goal 3
exact-DAG metrics without labeling.
