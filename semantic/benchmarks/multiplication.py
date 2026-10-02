"""Multi-digit multiplication, the compositional task from "Faith and Fate"
(Dziri et al., 2023). Complexity = digits per operand.

Steps are the digits of the product, so per-step error models act per digit.
"""
from __future__ import annotations

import random
import re

from .base import Instance, Task, Verdict, answer_section

PROMPT = """Compute the exact product {a} x {b}.
Work it out however you like, then on the last line write "ANSWER: <the product>"
with digits only (no commas or spaces)."""


class Multiplication(Task):
    name = "multiplication"
    complexity_label = "digits"
    description = "n-digit x n-digit multiplication (Faith and Fate); exact answer."

    def generate(self, complexity: int, seed: int) -> Instance:
        rng = random.Random(f"{self.name}:{complexity}:{seed}")
        lo, hi = 10 ** (complexity - 1), 10 ** complexity - 1
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        return Instance(self.name, complexity, seed, PROMPT.format(a=a, b=b),
                        {"a": a, "b": b}, str(a * b))

    def _number(self, answer: str) -> str:
        nums = re.findall(r"\d[\d,]*", answer_section(answer))
        return nums[-1].replace(",", "") if nums else ""

    def steps(self, answer: str) -> list[str]:
        return list(self._number(answer))

    def join(self, steps: list[str]) -> str:
        return "".join(steps)

    def corrupt_step(self, inst: Instance, step: str, rng: random.Random) -> str:
        return rng.choice([d for d in "0123456789" if d != step])

    def verify(self, inst: Instance, answer: str) -> Verdict:
        want = str(inst.data["a"] * inst.data["b"])
        got = self._number(answer)
        prefix = 0
        for x, y in zip(got, want):
            if x != y:
                break
            prefix += 1
        if got == want:
            return Verdict(True, "exact", len(want), len(want))
        return Verdict(False, f"got {got or 'nothing'}", prefix, len(want))
