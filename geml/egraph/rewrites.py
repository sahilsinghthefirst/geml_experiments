"""Rewrite rules and saturation loop for the Goal 4 e-graph core."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from geml.egraph.egraph import EGraph
from geml.egraph.ir import Add, Const, Div, Mul, Neg, Sub
from geml.egraph.patterns import Pattern, Substitution, add_pattern, match_eclass, pvar

type RuleTier = Literal["safe", "guarded", "positive_real_formal_rules", "placeholder"]
type RuleDirection = Literal["forward", "backward", "bidirectional"]
type GuardStatus = Literal["guarded", "unguarded"]
type RuleAssumptions = Literal["positive_real_formal"] | None
type SaturationStatus = Literal[
    "saturated",
    "iteration_limit",
    "enode_limit",
    "eclass_limit",
    "timeout",
]
type Guard = Callable[[EGraph, Substitution], bool]
type RhsBuilder = Callable[[EGraph, Substitution], int]


@dataclass(frozen=True, slots=True)
class RewriteRule:
    """A single directed rewrite rule."""

    name: str
    tier: RuleTier
    left: Pattern
    right: Pattern | RhsBuilder
    direction: RuleDirection = "forward"
    guard: Guard | None = None
    guard_status: GuardStatus = "unguarded"
    branch_sensitive: bool = False
    assumptions: RuleAssumptions = None
    max_applications_per_iteration: int | None = None

    def __post_init__(self) -> None:
        if self.guard is not None and self.guard_status == "unguarded":
            object.__setattr__(self, "guard_status", "guarded")

    @property
    def rule_name(self) -> str:
        """Provenance-compatible rule name."""
        return self.name

    @property
    def rule_tier(self) -> RuleTier:
        """Provenance-compatible rule tier."""
        return self.tier


@dataclass(frozen=True, slots=True)
class RulePlaceholder:
    """Document a rule that is intentionally not active yet."""

    name: str
    tier: RuleTier
    reason: str


@dataclass(frozen=True, slots=True)
class RewriteApplicationResult:
    """Result for one rewrite pass."""

    rule_name: str
    applications: int
    changed: bool


@dataclass(frozen=True, slots=True)
class SaturationLimits:
    """Resource limits for equality saturation."""

    max_iterations: int = 10
    max_enodes: int = 10_000
    max_eclasses: int = 10_000
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.max_enodes <= 0:
            raise ValueError("max_enodes must be positive")
        if self.max_eclasses <= 0:
            raise ValueError("max_eclasses must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class SaturationResult:
    """Summary of a saturation run."""

    status: SaturationStatus
    iterations_completed: int
    total_applications: int
    enode_count: int
    eclass_count: int
    elapsed_seconds: float


def apply_rewrite(egraph: EGraph, rule: RewriteRule) -> RewriteApplicationResult:
    """Apply one rewrite rule over a snapshot of the current e-classes."""
    applications = 0
    changed = False
    max_applications = rule.max_applications_per_iteration

    for eclass_id in egraph.eclass_ids():
        matches = match_eclass(egraph, rule.left, eclass_id)
        for substitution in matches:
            if max_applications is not None and applications >= max_applications:
                return RewriteApplicationResult(
                    rule_name=rule.name,
                    applications=applications,
                    changed=changed,
                )
            if rule.guard is not None and not rule.guard(egraph, substitution):
                continue
            before = (egraph.enode_count, egraph.eclass_count)
            rhs_id = (
                rule.right(egraph, substitution)
                if callable(rule.right)
                else add_pattern(egraph, rule.right, substitution)
            )
            union_changed = egraph.union(eclass_id, rhs_id)
            after = (egraph.enode_count, egraph.eclass_count)
            if union_changed or after != before:
                applications += 1
                changed = True

    return RewriteApplicationResult(
        rule_name=rule.name,
        applications=applications,
        changed=changed,
    )


def saturate(
    egraph: EGraph,
    rules: Sequence[RewriteRule],
    *,
    limits: SaturationLimits,
) -> SaturationResult:
    """Run equality saturation until stable or a resource limit is hit."""
    started_at = time.monotonic()
    total_applications = 0

    if limits.max_iterations == 0:
        return _result("iteration_limit", 0, total_applications, egraph, started_at)

    for iteration in range(limits.max_iterations):
        status = _limit_status(egraph, limits, started_at)
        if status is not None:
            return _result(status, iteration, total_applications, egraph, started_at)

        iteration_changed = False
        for rule in rules:
            if (status := _limit_status(egraph, limits, started_at)) is not None:
                return _result(status, iteration, total_applications, egraph, started_at)
            result = apply_rewrite(egraph, rule)
            total_applications += result.applications
            if result.changed:
                iteration_changed = True
            egraph.rebuild()

        if not iteration_changed:
            return _result("saturated", iteration + 1, total_applications, egraph, started_at)

    return _result(
        "iteration_limit",
        limits.max_iterations,
        total_applications,
        egraph,
        started_at,
    )


def add_commutativity_rule() -> RewriteRule:
    """Add commutativity: a + b -> b + a."""
    a = pvar("a")
    b = pvar("b")
    return RewriteRule(name="add_commutativity", tier="safe", left=Add(a, b), right=Add(b, a))


def mul_commutativity_rule() -> RewriteRule:
    """Multiplication commutativity: a * b -> b * a."""
    a = pvar("a")
    b = pvar("b")
    return RewriteRule(name="mul_commutativity", tier="safe", left=Mul(a, b), right=Mul(b, a))


def add_associativity_rule() -> RewriteRule:
    """Addition associativity: (a + b) + c -> a + (b + c)."""
    a = pvar("a")
    b = pvar("b")
    c = pvar("c")
    return RewriteRule(
        name="add_associativity_left_to_right",
        tier="safe",
        left=Add(Add(a, b), c),
        right=Add(a, Add(b, c)),
    )


def mul_associativity_rule() -> RewriteRule:
    """Multiplication associativity: (a * b) * c -> a * (b * c)."""
    a = pvar("a")
    b = pvar("b")
    c = pvar("c")
    return RewriteRule(
        name="mul_associativity_left_to_right",
        tier="safe",
        left=Mul(Mul(a, b), c),
        right=Mul(a, Mul(b, c)),
    )


def additive_identity_rules() -> tuple[RewriteRule, RewriteRule]:
    """Additive identity rewrites."""
    a = pvar("a")
    zero = Const(0)
    return (
        RewriteRule(name="add_zero_right", tier="safe", left=Add(a, zero), right=a),
        RewriteRule(name="add_zero_left", tier="safe", left=Add(zero, a), right=a),
    )


def multiplicative_identity_rules() -> tuple[RewriteRule, RewriteRule]:
    """Multiplicative identity rewrites."""
    a = pvar("a")
    one = Const(1)
    return (
        RewriteRule(name="mul_one_right", tier="safe", left=Mul(a, one), right=a),
        RewriteRule(name="mul_one_left", tier="safe", left=Mul(one, a), right=a),
    )


def multiplication_by_zero_rules() -> tuple[RewriteRule, RewriteRule]:
    """Multiplication-by-zero rewrites."""
    a = pvar("a")
    zero = Const(0)
    return (
        RewriteRule(name="mul_zero_right", tier="safe", left=Mul(a, zero), right=zero),
        RewriteRule(name="mul_zero_left", tier="safe", left=Mul(zero, a), right=zero),
    )


def double_negation_rule() -> RewriteRule:
    """Double negation: -(-a) -> a."""
    a = pvar("a")
    return RewriteRule(name="double_negation", tier="safe", left=Neg(Neg(a)), right=a)


def subtraction_as_add_negation_rule() -> RewriteRule:
    """Represent subtraction as addition of negation."""
    a = pvar("a")
    b = pvar("b")
    return RewriteRule(name="sub_as_add_neg", tier="safe", left=Sub(a, b), right=Add(a, Neg(b)))


def constant_folding_rules() -> tuple[RewriteRule, ...]:
    """Simple exact constant-folding rewrites."""
    a = pvar("a")
    b = pvar("b")
    return (
        RewriteRule(
            name="fold_add_constants",
            tier="safe",
            left=Add(a, b),
            right=_fold_add,
            guard=_both_constants("?a", "?b"),
        ),
        RewriteRule(
            name="fold_mul_constants",
            tier="safe",
            left=Mul(a, b),
            right=_fold_mul,
            guard=_both_constants("?a", "?b"),
        ),
        RewriteRule(
            name="fold_sub_constants",
            tier="safe",
            left=Sub(a, b),
            right=_fold_sub,
            guard=_both_constants("?a", "?b"),
        ),
        RewriteRule(
            name="fold_div_constants",
            tier="safe",
            left=Div(a, b),
            right=_fold_div,
            guard=lambda graph, subst: (
                _constant(graph, subst["?a"]) is not None
                and _constant(graph, subst["?b"]) not in {None, Fraction(0, 1)}
            ),
        ),
    )


def tier_a_rules() -> tuple[RewriteRule, ...]:
    """Return the implemented Tier A safe algebraic rules."""
    return (
        add_commutativity_rule(),
        mul_commutativity_rule(),
        add_associativity_rule(),
        mul_associativity_rule(),
        *additive_identity_rules(),
        *multiplicative_identity_rules(),
        *multiplication_by_zero_rules(),
        double_negation_rule(),
        subtraction_as_add_negation_rule(),
        *constant_folding_rules(),
    )


def positive_real_formal_rule_placeholders() -> tuple[RulePlaceholder, ...]:
    """Document positive-real log/exp rules that are intentionally inactive in Goal 4.1."""
    return (
        RulePlaceholder(
            name="log_exp_inverse",
            tier="positive_real_formal_rules",
            reason="placeholder only; Goal 4.1 does not activate log/exp rewrites",
        ),
        RulePlaceholder(
            name="exp_log_inverse",
            tier="positive_real_formal_rules",
            reason="placeholder only; Goal 4.1 does not activate log/exp rewrites",
        ),
        RulePlaceholder(
            name="log_product",
            tier="positive_real_formal_rules",
            reason="placeholder only; Goal 4.1 does not activate log/exp rewrites",
        ),
        RulePlaceholder(
            name="exp_sum",
            tier="positive_real_formal_rules",
            reason="placeholder only; Goal 4.1 does not activate log/exp rewrites",
        ),
    )


def _both_constants(left_var: str, right_var: str) -> Guard:
    def guard(egraph: EGraph, substitution: Substitution) -> bool:
        return (
            _constant(egraph, substitution[left_var]) is not None
            and _constant(egraph, substitution[right_var]) is not None
        )

    return guard


def _constant(egraph: EGraph, eclass_id: int) -> Fraction | None:
    for node in egraph.get_eclass_nodes(eclass_id):
        if node.op == "const" and isinstance(node.value, Fraction):
            return node.value
    return None


def _add_const(egraph: EGraph, value: Fraction) -> int:
    return egraph.add_expr(Const(value))


def _fold_add(egraph: EGraph, substitution: Substitution) -> int:
    left = _required_constant(egraph, substitution["?a"])
    right = _required_constant(egraph, substitution["?b"])
    return _add_const(egraph, left + right)


def _fold_mul(egraph: EGraph, substitution: Substitution) -> int:
    left = _required_constant(egraph, substitution["?a"])
    right = _required_constant(egraph, substitution["?b"])
    return _add_const(egraph, left * right)


def _fold_sub(egraph: EGraph, substitution: Substitution) -> int:
    left = _required_constant(egraph, substitution["?a"])
    right = _required_constant(egraph, substitution["?b"])
    return _add_const(egraph, left - right)


def _fold_div(egraph: EGraph, substitution: Substitution) -> int:
    left = _required_constant(egraph, substitution["?a"])
    right = _required_constant(egraph, substitution["?b"])
    return _add_const(egraph, left / right)


def _required_constant(egraph: EGraph, eclass_id: int) -> Fraction:
    value = _constant(egraph, eclass_id)
    if value is None:
        raise ValueError(f"e-class {eclass_id} does not contain a constant")
    return value


def _limit_status(
    egraph: EGraph,
    limits: SaturationLimits,
    started_at: float,
) -> SaturationStatus | None:
    if time.monotonic() - started_at >= limits.timeout_seconds:
        return "timeout"
    if egraph.enode_count >= limits.max_enodes:
        return "enode_limit"
    if egraph.eclass_count >= limits.max_eclasses:
        return "eclass_limit"
    return None


def _result(
    status: SaturationStatus,
    iterations_completed: int,
    total_applications: int,
    egraph: EGraph,
    started_at: float,
) -> SaturationResult:
    return SaturationResult(
        status=status,
        iterations_completed=iterations_completed,
        total_applications=total_applications,
        enode_count=egraph.enode_count,
        eclass_count=egraph.eclass_count,
        elapsed_seconds=time.monotonic() - started_at,
    )
