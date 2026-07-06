"""Placeholder import tests for the Goal 1 scaffold."""

from __future__ import annotations

import importlib


MODULES = [
    "geml",
    "geml.data.generate_exprs",
    "geml.data.identities",
    "geml.data.dataset",
    "geml.symbolic.ast_graph",
    "geml.symbolic.dag_graph",
    "geml.symbolic.eml_nodes",
    "geml.symbolic.eml_transpile",
    "geml.symbolic.metrics",
    "geml.models.graph_encoder",
    "geml.models.siamese_equiv",
    "geml.models.transformer_prefix",
    "geml.train.train_equiv",
    "geml.train.eval_equiv",
    "geml.experiments.expansion_study",
    "geml.experiments.baseline_grid",
]


def test_scaffold_modules_import() -> None:
    """All scaffold modules should import before core logic is implemented."""
    for module_name in MODULES:
        assert importlib.import_module(module_name)
