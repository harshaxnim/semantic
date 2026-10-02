"""Minimal generic STRIPS simulator, used as an independent second verifier.

The Blocksworld task has a hand-written simulator; this module checks the
same plans from a declarative operator table instead, so the two can be
cross-checked (a verifier we rely on must itself be verified). It is also
the starting point for further PDDL domains (e.g. Logistics).
"""
from __future__ import annotations

from dataclasses import dataclass

Fact = tuple[str, ...]


@dataclass(frozen=True)
class Operator:
    params: tuple[str, ...]
    pre: tuple[Fact, ...]
    add: tuple[Fact, ...]
    delete: tuple[Fact, ...]


def _ground(facts: tuple[Fact, ...], binding: dict[str, str]) -> set[Fact]:
    return {tuple(binding.get(t, t) for t in f) for f in facts}


BLOCKSWORLD: dict[str, Operator] = {
    "pick-up": Operator(("?x",), (("clear", "?x"), ("ontable", "?x"), ("handempty",)),
                        (("holding", "?x"),), (("ontable", "?x"), ("clear", "?x"), ("handempty",))),
    "put-down": Operator(("?x",), (("holding", "?x"),),
                         (("clear", "?x"), ("handempty",), ("ontable", "?x")), (("holding", "?x"),)),
    "stack": Operator(("?x", "?y"), (("holding", "?x"), ("clear", "?y")),
                      (("clear", "?x"), ("handempty",), ("on", "?x", "?y")), (("holding", "?x"), ("clear", "?y"))),
    "unstack": Operator(("?x", "?y"), (("on", "?x", "?y"), ("clear", "?x"), ("handempty",)),
                        (("holding", "?x"), ("clear", "?y")),
                        (("on", "?x", "?y"), ("clear", "?x"), ("handempty",))),
}


def blocksworld_state(on: dict[str, str]) -> set[Fact]:
    s: set[Fact] = {("handempty",)}
    for b, sup in on.items():
        s.add(("ontable", b) if sup == "table" else ("on", b, sup))
    for b in on:
        if b not in on.values():
            s.add(("clear", b))
    return s


def run_plan(domain: dict[str, Operator], state: set[Fact], plan: list[tuple[str, ...]], objects: set[str]) -> tuple[bool, set[Fact], int]:
    """Apply a grounded plan. Returns (all applicable, final state, steps applied)."""
    state = set(state)
    for i, (name, *args) in enumerate(plan):
        op = domain.get(name)
        if op is None or len(args) != len(op.params) or not set(args) <= objects:
            return False, state, i
        b = dict(zip(op.params, args))
        if len(set(args)) != len(args):  # distinct-argument convention as in PlanBench's domain
            return False, state, i
        if not _ground(op.pre, b) <= state:
            return False, state, i
        state = (state - _ground(op.delete, b)) | _ground(op.add, b)
    return True, state, len(plan)
