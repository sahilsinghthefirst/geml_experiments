# Goal 3.0 DAG Semantics

Goal 3.0 locks the semantics for exact DAG compression before any compression
implementation is written. Goal 2 showed that raw official pure EML trees are
representation-complete but structurally expensive. Goal 3 may test whether
exact structural sharing reduces that cost, but it must not reintroduce hidden
complexity through derived leaves, macro nodes, or algebraic shortcuts.

This document is a specification only. It does not implement DAG compression,
neural models, equivalence-pair generation, or any change to the official EML
compiler formulas.

## DAG Modes

Goal 3 uses explicit DAG/tree modes that distinguish source representation from
sharing policy:

- `ast_tree`: the ordinary binary AST tree emitted by the source AST converter.
- `ast_dag`: an exact structural DAG obtained by sharing identical AST
  subtrees from `ast_tree`.
- `restricted_eml_pure_tree`: the official recursive pure restricted EML tree.
- `restricted_eml_pure_dag`: an exact structural DAG obtained by sharing
  identical pure EML subtrees from `restricted_eml_pure_tree`.

The DAG modes are not new mathematical representations. They are compressed
views of already-materialized trees. A DAG implementation must first construct
the corresponding tree representation, then share only subtrees that are exactly
identical under the canonical structural-signature rules below.

The diagnostic `restricted_eml_with_derived` representation is excluded from
Goal 3 DAG compression metrics because it can contain derived leaves and hidden
compound-expression leaves. It may remain useful for debugging old behavior, but
it must not be treated as a valid DAG compression input.

## Exact Structural Sharing

A DAG node represents one unique structural subtree from the source tree. Two
subtrees may be represented by the same DAG node only if their full canonical
structural signatures are identical.

Sharing is exact structural sharing, not semantic sharing. A canonical
structural signature must be derived from the tree node and its ordered children,
not from simplified algebraic meaning. At minimum, the signature must include:

- representation family: AST or restricted pure EML
- node kind, such as `operator`, `symbol`, `constant`, `variable`, or `eml`
- node label, such as `add`, `mul`, `pow`, `exp`, `log`, `x`, `y`, `1`, or
  `eml`
- arity
- child slot numbers
- child signatures in slot order
- leaf value metadata needed to distinguish structurally different constants

For a leaf, the full signature is the node kind, label, and relevant leaf value.
For an internal node, the full signature is the node kind, label, arity, and the
ordered list of child signatures.

Example schematic signatures:

```text
AST leaf x:
  ("ast", "symbol", "x")

AST add(x, y):
  ("ast", "operator", "add", ((0, ("ast", "symbol", "x")),
                              (1, ("ast", "symbol", "y"))))

Pure EML EML(x, 1):
  ("restricted_eml_pure", "eml", "eml",
    ((0, ("restricted_eml_pure", "variable", "x")),
     (1, ("restricted_eml_pure", "constant", "1"))))
```

An implementation may encode signatures differently, but it must preserve the
same information and equality semantics.

## Explicitly Forbidden Compression Mechanisms

Goal 3 DAG compression must not introduce any of the following:

- derived leaves
- hidden compound-expression leaves
- macro or template nodes
- parameterized macro sharing
- algebraic simplification for compression
- sharing "patterns with holes", such as `EML(1, z)`

The DAG must not contain final helper labels such as `eml_log`, `eml_exp`,
`eml_add`, `eml_mul`, or any other macro name. These are compiler construction
concepts only. Final pure EML DAG nodes must still be only `eml`, source
variable leaves, and constant `1` leaves.

Sharing `EML(1, x)` with `EML(1, y)` by treating both as the pattern
`EML(1, z)` is forbidden. Those subtrees have different full structural
signatures because the right child signatures differ.

## Structural, Not Algebraic

The Goal 3 DAG is structural, not algebraic. It compresses repeated syntax
already present in a tree. It does not prove or search for mathematical
equivalence.

Consequences:

- `x + y` and `y + x` may be shared only if the upstream AST converter has
  already normalized or ordered them into an identical structural tree.
- Goal 3 must not add a new commutative reordering pass solely to improve DAG
  compression.
- `x * x` and `x**2` must not be treated as identical unless the source
  converter represents them identically.
- `exp(log(x))` and `x` must not be treated as identical for DAG compression.
- `x + 0` and `x` must not be treated as identical for DAG compression.

Any future algebraic or rewrite-aware graph representation must be a separately
named mode and must not be mixed into these Goal 3 structural DAG metrics.

## Edge Semantics

DAG edges are directed child references from parent nodes to child nodes. They
must preserve child slot and order.

For binary nodes, child references must distinguish left and right children:

- position `0` is the left child
- position `1` is the right child

Unary AST nodes, such as `exp` and `log`, use position `0`. Pure EML internal
nodes are always binary and must have positions `0` and `1`.

Repeated child references are allowed. For example, `EML(a, a)` may have one
unique DAG node for `a`, but the parent `EML` node still has two child
references to that same node:

```text
parent EML:
  position 0 -> a
  position 1 -> a
```

Those are two child references, even if they point to the same target node.

## Metrics

Tree metrics keep the existing Goal 1 and Goal 2 conventions:

- `T_AST_nodes`: node count in `ast_tree`
- `T_AST_edges`: edge count in `ast_tree`
- `T_AST_depth`: longest path length from root to any leaf in `ast_tree`
- `T_EML_nodes`: node count in `restricted_eml_pure_tree`
- `T_EML_edges`: edge count in `restricted_eml_pure_tree`
- `T_EML_depth`: longest path length from root to any leaf in
  `restricted_eml_pure_tree`

DAG metrics use unique node count and child-reference count:

- `D_AST_unique_nodes`: unique structural nodes in `ast_dag`
- `D_AST_child_references`: ordered child references in `ast_dag`
- `D_AST_depth`: longest path length from root to any leaf in `ast_dag`
- `D_EML_unique_nodes`: unique structural nodes in `restricted_eml_pure_dag`
- `D_EML_child_references`: ordered child references in
  `restricted_eml_pure_dag`
- `D_EML_depth`: longest path length from root to any leaf in
  `restricted_eml_pure_dag`

Compression and DAG alpha metrics:

```text
ast_dag_compression = T_AST_nodes / D_AST_unique_nodes
eml_dag_compression = T_EML_nodes / D_EML_unique_nodes

dag_alpha_vs_ast_tree = D_EML_unique_nodes / T_AST_nodes
dag_alpha_vs_ast_dag  = D_EML_unique_nodes / D_AST_unique_nodes
```

Tree edge counts remain ordinary tree edge counts. DAG edge counts must be
reported as child-reference counts, not unique unordered edges. If two child
slots point to the same target, they count as two child references. If the same
ordered parent-child reference appears because the parent has repeated child
slots, each slot still counts separately.

Depth remains the longest path length from the root to any leaf after sharing.
Sharing reduces duplicated nodes, but it does not shorten a path unless the
underlying structural path is shorter. A leaf has depth `0`, matching the
existing tree metric convention.

## Later Implementation Checklist

When Goal 3 compression is implemented, tests or documentation checks should
verify all of the following:

- Mode names include exactly `ast_tree`, `ast_dag`,
  `restricted_eml_pure_tree`, and `restricted_eml_pure_dag` for Goal 3 DAG
  comparisons.
- DAG construction starts from the already-materialized tree for the selected
  mode.
- Identical leaf signatures are shared.
- Identical internal subtree signatures are shared only when labels, kinds,
  arity, child positions, and child signatures all match.
- Non-identical leaves or subtrees are not shared, even if they are
  algebraically equivalent.
- `x + y` and `y + x` share only when the upstream AST conversion already emits
  identical structures.
- `x * x` and `x**2` are not merged unless the source converter emits identical
  structures.
- `EML(1, x)` and `EML(1, y)` are not merged through a pattern such as
  `EML(1, z)`.
- No DAG node has a derived, hidden-compound, macro, template, or parameterized
  macro kind.
- Pure EML DAG leaves are only variables or constant `1`.
- Pure EML DAG internal nodes are only `eml`.
- Edges preserve child position.
- `EML(a, a)` records two child references even when both references target the
  same unique node.
- DAG child-reference count is not deduplicated as unordered graph edges.
- DAG depth is computed as longest root-to-leaf path length after sharing.
- Metrics export reports both tree and DAG metrics so `ast_dag_compression`,
  `eml_dag_compression`, `dag_alpha_vs_ast_tree`, and
  `dag_alpha_vs_ast_dag` are reproducible.
- Tests include repeated-subtree examples for both AST and pure EML.
- Tests include negative examples that would be incorrectly compressed by
  semantic simplification or pattern-with-hole sharing.

Until those checks exist, DAG compression results should be treated as
provisional and should not be used for scientific claims.
