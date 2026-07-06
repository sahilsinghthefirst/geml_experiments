# GEML-v0

GEML-v0 is an experimental benchmark for testing when EML-native graph representations help or hurt symbolic mathematical reasoning.

The project compares standard expression representations against restricted EML representations while measuring the tradeoff between reduced operator diversity and increased tree size.

## Goal 1 Scaffold

This repository is currently scaffolded for Goal 1: core expression representation and data generation.

The initial scaffold includes:

- Python 3.12 package metadata
- pytest configuration
- ruff linting and formatting configuration
- YAML config placeholders
- package directories for data generation, symbolic graph conversion, models, training, and experiments
- JSONL/CSV output directories under `outputs/v0/`
- placeholder tests that verify imports work

Core expression generation, graph conversion, EML transpilation, metrics, and dataset export logic will be implemented in later stages.
