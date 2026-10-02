"""Blocksworld plan generation in the style of PlanBench (Valmeekam et al., 2023).

Complexity = number of blocks. Instances are a random initial configuration
and a goal given as a set of on(x, y) facts from a second random
configuration. The verifier simulates the 4-operator STRIPS domain exactly
(the same semantics VAL checks for PlanBench's domain file).

`MysteryBlocksworld` is the same problem with the action and predicate names
obfuscated, as in PlanBench's "Mystery Blocksworld". A system that reasons
rather than recalls should score the same on both.
"""
from __future__ import annotations

import random
import re

from .base import Instance, Task, Verdict, answer_section

ACTIONS = ("pick-up", "put-down", "stack", "unstack")

PLAIN = {
    "pick-up": "pick-up", "put-down": "put-down", "stack": "stack", "unstack": "unstack",
    "clear": "clear", "ontable": "on the table", "on": "on", "handempty": "the hand is empty",
    "holding": "holding",
}
# Names follow PlanBench's Mystery Blocksworld obfuscation.
MYSTERY = {
    "pick-up": "attack", "put-down": "succumb", "stack": "overcome", "unstack": "feast",
    "clear": "province", "ontable": "planet", "on": "craves", "handempty": "harmony",
    "holding": "pain",
}

PROMPT_PLAIN = """I am playing with a set of blocks where I need to arrange the blocks into stacks.
Here are the actions I can do:
  (pick-up x): pick up block x from the table
  (put-down x): put down block x that I am holding onto the table
  (stack x y): stack block x that I am holding on top of block y
  (unstack x y): unstack block x from on top of block y
Restrictions:
  I can only pick up or unstack one block at a time, and only if my hand is empty.
  I can only pick up a block if it is on the table and clear (no block on it).
  I can only unstack x from y if x is really on top of y and x is clear.
  I can only put down or stack a block that I am holding.
  I can only stack x on y if y is clear. Once stacked or put down, my hand is empty.

Initial state: {init}
Goal: {goal}

Write a plan that achieves the goal. After the line "ANSWER:" write one action per
line in the form shown above, e.g. "(unstack b1 b2)". Write nothing else after ANSWER:."""

PROMPT_MYSTERY = """I am playing with a set of objects. Here are the actions I can do:
  (attack x), (succumb x), (overcome x y), (feast x y)
Facts that can hold: "province x", "planet x", "harmony", "pain x", "x craves y".
Rules:
  To attack x: province x, planet x and harmony must hold. Afterwards pain x holds,
    and province x, planet x and harmony no longer hold.
  To succumb x: pain x must hold. Afterwards province x, planet x and harmony hold,
    and pain x no longer holds.
  To overcome x y: province y and pain x must hold. Afterwards harmony, province x
    and x craves y hold, and province y and pain x no longer hold.
  To feast x y: x craves y, province x and harmony must hold. Afterwards pain x and
    province y hold, and x craves y, province x and harmony no longer hold.

Initial state: {init}
Goal: {goal}

Write a plan that achieves the goal. After the line "ANSWER:" write one action per
line, e.g. "(feast b1 b2)". Write nothing else after ANSWER:."""


def _names(n: int) -> list[str]:
    return [f"b{i + 1}" for i in range(n)]


def _random_config(blocks: list[str], rng: random.Random) -> dict[str, str]:
    """Random set of towers: maps each block to what it sits on ('table' or a block)."""
    order = blocks[:]
    rng.shuffle(order)
    on: dict[str, str] = {}
    towers: list[list[str]] = []
    for b in order:
        if towers and rng.random() < 0.6:
            t = rng.choice(towers)
            on[b] = t[-1]
            t.append(b)
        else:
            on[b] = "table"
            towers.append([b])
    return on


def _towers(on: dict[str, str]) -> list[list[str]]:
    above = {v: k for k, v in on.items() if v != "table"}
    out = []
    for b, s in on.items():
        if s == "table":
            t = [b]
            while t[-1] in above:
                t.append(above[t[-1]])
            out.append(t)
    return sorted(out)


class Blocksworld(Task):
    name = "blocksworld"
    complexity_label = "blocks"
    description = "PlanBench-style Blocksworld plan generation; exact STRIPS simulation."
    vocab = PLAIN
    template = PROMPT_PLAIN

    # ---- rendering -------------------------------------------------------
    def _describe(self, on: dict[str, str], full: bool) -> str:
        v = self.vocab
        facts = []
        for b, s in sorted(on.items()):
            if s == "table":
                if full:
                    facts.append(f"{v['ontable']} {b}" if v is MYSTERY else f"{b} is on the table")
            else:
                facts.append(f"{b} {v['on']} {s}" if v is MYSTERY else f"{b} is on {s}")
        if full:
            covered = {s for s in on.values()}
            for b in sorted(on):
                if b not in covered:
                    facts.append(f"{v['clear']} {b}" if v is MYSTERY else f"{b} is clear")
            facts.append(v["handempty"])
        return ", ".join(facts) + "."

    def generate(self, complexity: int, seed: int) -> Instance:
        rng = random.Random(f"{self.name}:{complexity}:{seed}")
        blocks = _names(complexity)
        init = _random_config(blocks, rng)
        goal_cfg = _random_config(blocks, rng)
        for _ in range(20):  # avoid trivial instances where the goal already holds
            if any(init.get(b) != s for b, s in goal_cfg.items() if s != "table"):
                break
            goal_cfg = _random_config(blocks, rng)
        goal = {b: s for b, s in goal_cfg.items() if s != "table"}
        prompt = self.template.format(init=self._describe(init, True), goal=self._describe(goal, False))
        inst = Instance(self.name, complexity, seed, prompt, {"init": init, "goal": goal})
        inst.reference = self.join(self._reference_plan(init, goal_cfg))
        return inst

    def _reference_plan(self, init: dict[str, str], goal_cfg: dict[str, str]) -> list[str]:
        """Valid (not optimal) plan: clear everything to the table, then build the goal towers."""
        plan = []
        for t in _towers(init):
            for i in range(len(t) - 1, 0, -1):
                plan += [f"(unstack {t[i]} {t[i - 1]})", f"(put-down {t[i]})"]
        for t in _towers(goal_cfg):
            for i in range(1, len(t)):
                plan += [f"(pick-up {t[i]})", f"(stack {t[i]} {t[i - 1]})"]
        return [self._to_vocab(a) for a in plan]

    # ---- answer format ---------------------------------------------------
    ACT_RE = re.compile(r"\(\s*([a-z\-]+)\s+([a-z0-9]+)(?:\s+([a-z0-9]+))?\s*\)", re.IGNORECASE)

    def _to_vocab(self, action: str) -> str:
        m = self.ACT_RE.match(action)
        name = self.vocab[m.group(1)]
        args = " ".join(a for a in m.groups()[1:] if a)
        return f"({name} {args})"

    def steps(self, answer: str) -> list[str]:
        return [ln.strip() for ln in answer_section(answer).splitlines() if ln.strip()]

    def join(self, steps: list[str]) -> str:
        return "\n".join(steps)

    def corrupt_step(self, inst: Instance, step: str, rng: random.Random) -> str:
        m = self.ACT_RE.search(step)
        if not m:
            return step
        blocks = list(inst.data["init"])
        name, x, y = m.groups()
        if y and rng.random() < 0.5:
            choices = [b for b in blocks if b not in (x, y)] or [x]
            return f"({name} {x} {rng.choice(choices)})"
        choices = [b for b in blocks if b != x] or [x]
        nx = rng.choice(choices)
        return f"({name} {nx} {y})" if y else f"({name} {nx})"

    def verify(self, inst: Instance, answer: str) -> Verdict:
        inverse = {v: k for k, v in self.vocab.items() if k in ACTIONS}
        on = dict(inst.data["init"])
        holding: str | None = None
        steps = self.steps(answer)
        n = len(steps)

        def clear(b: str) -> bool:
            return holding != b and b not in on.values()

        for i, s in enumerate(steps):
            m = self.ACT_RE.fullmatch(s.strip())
            if not m or m.group(1).lower() not in inverse:
                return Verdict(False, f"step {i + 1}: unparseable {s!r}", i, n)
            act = inverse[m.group(1).lower()]
            x, y = m.group(2).lower(), (m.group(3) or "").lower() or None
            if x not in on or (y is not None and y not in on):
                return Verdict(False, f"step {i + 1}: unknown block", i, n)
            ok = False
            if act == "pick-up" and y is None:
                ok = holding is None and on[x] == "table" and clear(x)
                if ok:
                    holding = x; on[x] = "hand"
            elif act == "put-down" and y is None:
                ok = holding == x
                if ok:
                    holding = None; on[x] = "table"
            elif act == "stack" and y is not None:
                ok = holding == x and x != y and clear(y) and on[y] != "hand"
                if ok:
                    holding = None; on[x] = y
            elif act == "unstack" and y is not None:
                ok = holding is None and on[x] == y and clear(x)
                if ok:
                    holding = x; on[x] = "hand"
            if not ok:
                return Verdict(False, f"step {i + 1}: {s} not applicable", i, n)
        missing = [f"{b} on {s}" for b, s in inst.data["goal"].items() if on.get(b) != s]
        if missing:
            return Verdict(False, f"goal not reached: {', '.join(missing[:3])}", n, n)
        return Verdict(True, f"valid plan, {n} actions", n, n)


class MysteryBlocksworld(Blocksworld):
    name = "blocksworld-mystery"
    description = "Blocksworld with obfuscated action/predicate names (PlanBench 'Mystery')."
    vocab = MYSTERY
    template = PROMPT_MYSTERY
