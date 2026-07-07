"""Validation helpers for Goal 4 e-graph outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import sympy as sp

from geml.egraph.egraph import EGraph
from geml.egraph.ir import Add, Const, Div, Exp, Expr, Log, Mul, Neg, Pow, Sub, Var, to_sympy

type ValidationStatus = Literal["valid", "invalid", "error"]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """A semantic validation result."""

    status: ValidationStatus
    method: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class PositiveRealNumericValidationResult:
    """Positive-real numeric validation result."""

    validation_status: ValidationStatus
    max_abs_error: float | None
    sample_count: int
    assumptions: str = "positive_real_formal"
    method: str = "positive_real_numeric_sampling"
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class SameEClassValidationResult:
    """Validation that two e-class ids are equivalent in the current e-graph."""

    validation_status: ValidationStatus
    left_eclass_id: int
    right_eclass_id: int
    method: str = "same_root_eclass"
    detail: str | None = None


def validate_with_sympy(left: Expr, right: Expr) -> ValidationResult:
    """Validate equality with SymPy as a final validation step."""
    try:
        difference = to_sympy(left) - to_sympy(right)
        is_zero = sp.simplify(difference) == 0
    except Exception as exc:
        return ValidationResult(
            status="error",
            method="sympy_simplify_final_validation",
            detail=f"{type(exc).__name__}: {exc}",
        )
    return ValidationResult(
        status="valid" if is_zero else "invalid",
        method="sympy_simplify_final_validation",
        detail=None,
    )


def validate_same_eclass(
    egraph: EGraph,
    left_eclass_id: int,
    right_eclass_id: int,
) -> SameEClassValidationResult:
    """Validate that two ids resolve to the same canonical e-class."""
    try:
        left_root = egraph.find(left_eclass_id)
        right_root = egraph.find(right_eclass_id)
    except Exception as exc:
        return SameEClassValidationResult(
            validation_status="error",
            left_eclass_id=left_eclass_id,
            right_eclass_id=right_eclass_id,
            detail=f"{type(exc).__name__}: {exc}",
        )
    return SameEClassValidationResult(
        validation_status="valid" if left_root == right_root else "invalid",
        left_eclass_id=left_root,
        right_eclass_id=right_root,
        detail=None if left_root == right_root else "ids resolve to different e-classes",
    )


def positive_real_numeric_validation(
    original: Expr,
    extracted: Expr,
    *,
    tolerance: float = 1e-9,
) -> PositiveRealNumericValidationResult:
    """Validate two expressions on positive real samples only.

    This is a numeric positive-real check. It is not a complex-domain proof.
    """
    try:
        variable_names = sorted(_collect_variables(original) | _collect_variables(extracted))
        samples = _positive_samples(variable_names)
        errors: list[float] = []
        for sample in samples:
            original_value = _evaluate_positive_real(original, sample)
            extracted_value = _evaluate_positive_real(extracted, sample)
            error = abs(original_value - extracted_value)
            if not math.isfinite(error):
                return PositiveRealNumericValidationResult(
                    validation_status="error",
                    max_abs_error=None,
                    sample_count=len(errors),
                    detail="non-finite numeric error",
                )
            errors.append(error)
    except Exception as exc:
        return PositiveRealNumericValidationResult(
            validation_status="error",
            max_abs_error=None,
            sample_count=0,
            detail=f"{type(exc).__name__}: {exc}",
        )

    max_abs_error = max(errors) if errors else 0.0
    return PositiveRealNumericValidationResult(
        validation_status="valid" if max_abs_error <= tolerance else "invalid",
        max_abs_error=max_abs_error,
        sample_count=len(errors),
        detail=None,
    )


def _collect_variables(expr: Expr) -> set[str]:
    if isinstance(expr, Var):
        return {expr.name}
    if isinstance(expr, Const):
        return set()
    if isinstance(expr, Neg | Exp | Log):
        return _collect_variables(expr.value)
    if isinstance(expr, Add | Mul | Sub | Div):
        return _collect_variables(expr.left) | _collect_variables(expr.right)
    if isinstance(expr, Pow):
        return _collect_variables(expr.base) | _collect_variables(expr.exponent)
    raise TypeError(f"unsupported expression type: {type(expr).__name__}")


def _positive_samples(variable_names: list[str]) -> list[dict[str, float]]:
    base_samples = [
        {"x": 1.3, "y": 2.1},
        {"x": 0.7, "y": 3.4},
        {"x": 4.2, "y": 1.6},
    ]
    samples: list[dict[str, float]] = []
    for sample_index, base_sample in enumerate(base_samples):
        sample = dict(base_sample)
        for variable_index, name in enumerate(variable_names):
            sample.setdefault(name, 1.2 + sample_index + (0.3 * variable_index))
        samples.append(sample)
    return samples


def _evaluate_positive_real(expr: Expr, values: dict[str, float]) -> float:
    if isinstance(expr, Var):
        return values[expr.name]
    if isinstance(expr, Const):
        return float(expr.value)
    if isinstance(expr, Add):
        return _evaluate_positive_real(expr.left, values) + _evaluate_positive_real(
            expr.right,
            values,
        )
    if isinstance(expr, Mul):
        return _evaluate_positive_real(expr.left, values) * _evaluate_positive_real(
            expr.right,
            values,
        )
    if isinstance(expr, Neg):
        return -_evaluate_positive_real(expr.value, values)
    if isinstance(expr, Sub):
        return _evaluate_positive_real(expr.left, values) - _evaluate_positive_real(
            expr.right,
            values,
        )
    if isinstance(expr, Div):
        denominator = _evaluate_positive_real(expr.right, values)
        if denominator == 0:
            raise ZeroDivisionError("division by zero in positive-real validation sample")
        return _evaluate_positive_real(expr.left, values) / denominator
    if isinstance(expr, Pow):
        return _evaluate_positive_real(expr.base, values) ** _evaluate_positive_real(
            expr.exponent,
            values,
        )
    if isinstance(expr, Exp):
        return math.exp(_evaluate_positive_real(expr.value, values))
    if isinstance(expr, Log):
        value = _evaluate_positive_real(expr.value, values)
        if value <= 0:
            raise ValueError("log argument is not positive in validation sample")
        return math.log(value)
    raise TypeError(f"unsupported expression type: {type(expr).__name__}")
