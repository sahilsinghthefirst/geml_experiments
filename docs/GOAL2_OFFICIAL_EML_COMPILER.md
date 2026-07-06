# Goal 2 Official EML Compiler

Goal 2.1b ports the official recursive EML compiler definitions into GEML's native tree representation.

## Official Source

Source used:

- Repository: `VA00/SymbolicRegressionPackage`
- File: `EML_toolkit/EmL_compiler/eml_compiler_v4.py`
- URL: <https://github.com/VA00/SymbolicRegressionPackage/blob/master/EML_toolkit/EmL_compiler/eml_compiler_v4.py>

The upstream repository is MIT licensed. GEML attributes the macro definitions in `geml/symbolic/official_eml_compiler.py` and ports them as native tree constructors rather than string builders.

## Core Operator

```text
EML(a, b) = exp(a) - log(b)
```

In the final GEML tree, every internal node is labeled `eml`. Leaves are only variable symbols or the constant `1`.

## Ported Macro Definitions

Helper names are macros only. They never appear as final tree node labels.

```text
eml_exp(z)      = EML(z, 1)
eml_log(z)      = EML(1, eml_exp(EML(1, z)))
                = EML(1, EML(EML(1, z), 1))
eml_zero()      = eml_log(1)
eml_sub(a, b)   = EML(eml_log(a), eml_exp(b))
eml_neg(z)      = eml_sub(eml_zero(), z)
eml_add(a, b)   = eml_sub(a, eml_neg(b))
eml_inv(z)      = eml_exp(eml_neg(eml_log(z)))
eml_mul(a, b)   = eml_exp(eml_add(eml_log(a), eml_log(b)))
eml_div(a, b)   = eml_mul(a, eml_inv(b))
eml_pow(a, b)   = eml_exp(eml_mul(b, eml_log(a)))
eml_one()       = 1
```

Integers are compiled with the official repeated-doubling construction:

- `1 -> 1`
- `0 -> eml_zero()`
- negative integers use `eml_neg(eml_int(abs(n)))`
- positive integers greater than `1` are built by binary repeated doubling with `eml_add`

Rationals are compiled as:

- if denominator is `1`, use `eml_int(p)`
- otherwise compile `abs(p)` and `q`, then use `eml_mul(num, eml_inv(den))`
- apply `eml_neg` when the numerator is negative

Floats are converted through `Rational(str(x))` before rational compilation.

## Supported Nodes

The official pure compiler supports:

- `Symbol`
- `Integer`
- `Rational`
- `Float`
- `Add`
- `Mul`
- `Pow`
- `exp`
- `log`

Generated arithmetic/log/exp expressions now compile into pure EML without derived leaves.

## Unsupported Nodes

These remain unsupported for this stage:

- trigonometric functions
- inverse trigonometric functions
- hyperbolic functions
- `Abs`
- arbitrary unsupported SymPy nodes

## Derived Leaves Are Invalid For Alpha

The old Goal 1 lift:

```text
E -> eml(log(E), 1)
```

is still isolated as `restricted_eml_with_derived` for diagnostics only. It can hide compound expressions inside a derived leaf, so rows using it must have `alpha_valid=false`.

Goal 2 alpha plots must use:

```text
representation_mode == "restricted_eml_pure"
alpha_valid == true
```

## Domain Caveats

The official macros rely on formal log/exp inverse identities. Real and complex branch behavior is not fully handled yet.

Numeric tests use positive real values first:

```text
x = 1.3
y = 2.1
z = 3.2
```

The numeric EML evaluator computes `exp(eval(a)) - log(eval(b))`. To evaluate the official negation macro over real floats, it treats `log(0)` as extended-real `-inf`; this matches the formal `eml_zero()` path used by `eml_neg`.

## Run Commands

Run tests and checks:

```bash
python -m pytest
python -m ruff check .
python -m ruff format . --check
```

Run the Goal 2.1 expansion pipeline:

```bash
python -m geml.experiments.expansion_study --config configs/expansion_v0.yaml
```

Outputs:

- `outputs/v0/expansion_inputs.jsonl`
- `outputs/v0/expansion_raw_metrics.jsonl`
- `outputs/v0/expansion_raw_metrics.csv`
- `outputs/v0/expansion_alpha_summary.csv`
- `outputs/v0/expansion_alpha_summary.json`
- `outputs/v0/official_eml_compiler_summary.json`
- `outputs/v0/official_eml_top20_alpha.json`
- `outputs/v0/official_eml_top20_depth.json`
- `outputs/v0/official_eml_simple_examples.json`

## Goal 2.1c Expansion Size Audit

Pure official EML expansion has high alpha because each familiar source operator is not a primitive operation in the final tree. For example:

- `log(x)` expands to `EML[1,EML[EML[1,x],1]]`, so it adds nested EML structure for a single source `log`.
- `x + y` expands through `eml_add(a, b) = eml_sub(a, eml_neg(b))`.
- `x * y` expands through `eml_mul(a, b) = eml_exp(eml_add(eml_log(a), eml_log(b)))`, so multiplication includes log, add, and exp macro expansions.
- `x**2` expands through `eml_pow(a, b) = eml_exp(eml_mul(b, eml_log(a)))`, and the integer exponent `2` is itself compiled with the official integer construction rather than a constant leaf.

Simple-expression audit counts:

| Expression | AST nodes | EML nodes | Alpha | EML depth | Derived leaves |
| ---------- | --------: | --------: | ----: | --------: | -------------: |
| `x+y`      |         3 |        27 |  9.0  |         9 |              0 |
| `x*y`      |         3 |        41 | 13.67 |        10 |              0 |
| `log(x)`   |         2 |         7 |  3.5  |         3 |              0 |
| `exp(x)`   |         2 |         3 |  1.5  |         1 |              0 |
| `x**2`     |         3 |        75 | 25.0  |        18 |              0 |

These counts are high by design. They reflect fully expanded pure EML trees with no derived leaves, no hidden compound expression leaves, and no final `Add`, `Mul`, `Pow`, `exp`, or `log` operation nodes.

## Goal 2.2 Alpha Threshold

Goal 2.2 compares the official pure EML expansion factor against:

```text
alpha_threshold = 1 + log(K) / log(4L)
```

The default raw metric rows are annotated with the current-grammar threshold using `K=4` and `L=3`:

- `alpha`
- `alpha_threshold`
- `below_threshold`

Aggregate threshold summaries are written to:

- `outputs/v0/expansion_alpha_summary.csv`
- `outputs/v0/expansion_alpha_summary.json`

The configured scenarios are:

| Scenario | K | L | Alpha threshold | Percent below | Percent above |
| -------- | --: | --: | --------------: | ------------: | ------------: |
| `current_grammar` | 4 | 3 | 1.5578858913022597 | 0.0 | 100.0 |
| `generous_operator_vocab` | 20 | 3 | 2.2055713536802566 | 0.25 | 99.75 |
| `larger_operator_vocab` | 50 | 3 | 2.574313870407124 | 0.25 | 99.75 |

The fixed-seed Goal 2.2 run has mean alpha `10.648995087903057`, median alpha `11.375`, p90 alpha `14.208333333333334`, p95 alpha `14.733333333333333`, and max alpha `17.366666666666667`. These values are far above all configured thresholds for almost every generated expression.

This means raw official pure EML trees are unlikely to be computationally smaller than ASTs by tree size alone. Any later size advantage would need to come from a separate mechanism such as compression or sharing; Goal 2.2 does not implement DAG compression or neural models.

## 10,000-Expression Rerun Summary

Using `configs/expansion_v0.yaml`:

- processed count: `10000`
- official pure EML supported count: `10000`
- unsupported count: `0`
- mean alpha: `10.648995087903057`
- median alpha: `11.375`
- p90 alpha: `14.208333333333334`
- p95 alpha: `14.733333333333333`
- max alpha: `17.366666666666667`

The top 20 largest-alpha expressions are recorded in `outputs/v0/official_eml_top20_alpha.json` and repeated in `outputs/v0/official_eml_compiler_summary.json`. The top 20 deepest EML expressions are recorded in `outputs/v0/official_eml_top20_depth.json`.
