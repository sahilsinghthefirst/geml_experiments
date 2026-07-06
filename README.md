# GEML-v0

GEML-v0 is an experimental benchmark for testing when EML-native graph representations help or hurt symbolic mathematical reasoning.

The project compares standard expression representations against restricted EML representations while measuring the tradeoff between reduced operator diversity and increased tree size.

## Goal 1 Scaffold

This repository is currently scaffolded for Goal 1: core expression representation and data generation.

The initial scaffold includes:

- Python 3.12 package metadata
- pytest configuration
- ruff linting and formatting configuration
- YAML configs
- package directories for data generation, symbolic graph conversion, models, training, and experiments
- JSONL/CSV output directories under `outputs/v0/`
- placeholder tests that verify imports work

Expression generation and dataset export are implemented. Graph conversion, EML transpilation, metrics, and model logic will be implemented in later stages.

## Expression Generation

Generate the initial Goal 1 expression dataset with:

```bash
python -m geml.data.generate_exprs --config configs/data_v0.yaml
```

The generator supports `x`, `y`, `1`, `Add`, `Mul`, `Exp`, and `Log`, with configurable depth, seed, expression count, and operator probabilities. It writes JSONL and CSV outputs under `outputs/v0/` by default.
