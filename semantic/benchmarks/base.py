"""Common interface every benchmark task implements.

A task is a generator of instances with a complexity knob plus a sound,
automatic verifier. Nothing here calls a model.
"""
from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

ANSWER_MARK = re.compile(r"ANSWER\s*:", re.IGNORECASE)


@dataclass
class Instance:
    task: str
    complexity: int
    seed: int
    prompt: str
    data: dict[str, Any] = field(default_factory=dict)
    # A known-correct answer in the task's answer format. Used by the oracle
    # model and by the verifier self-checks; never shown to real models.
    reference: str = ""

    @property
    def id(self) -> str:
        return f"{self.task}/c{self.complexity}/s{self.seed}"


@dataclass
class Verdict:
    correct: bool
    detail: str = ""
    # Number of leading steps that were valid; useful to locate where a
    # long answer went wrong (and for per-step reliability estimates).
    valid_prefix: int = 0
    total_steps: int = 0


def answer_section(text: str) -> str:
    """Return the text after the last 'ANSWER:' marker, or all text if absent."""
    parts = ANSWER_MARK.split(text)
    return parts[-1] if len(parts) > 1 else text


class Task(ABC):
    name: str = ""
    complexity_label: str = "complexity"
    description: str = ""

    @abstractmethod
    def generate(self, complexity: int, seed: int) -> Instance: ...

    @abstractmethod
    def verify(self, inst: Instance, answer: str) -> Verdict: ...

    @abstractmethod
    def steps(self, answer: str) -> list[str]:
        """Split an answer into its atomic steps (moves, actions, digits)."""

    @abstractmethod
    def join(self, steps: list[str]) -> str:
        """Inverse of steps(): render steps back into the answer format."""

    @abstractmethod
    def corrupt_step(self, inst: Instance, step: str, rng: random.Random) -> str:
        """Return a plausible but different step (used to simulate errors)."""

    def corrupt(self, inst: Instance, answer: str, rng: random.Random) -> str:
        """Return a mutated answer that differs from the input.

        Mutations: change one step, drop one step, or swap two adjacent steps.
        A mutant can still be a valid answer (e.g. two independent actions
        swapped); callers must not assume every mutant is wrong.
        """
        for _ in range(10):
            mutant = self._mutate(answer, inst, rng)
            if self.steps(mutant) != self.steps(answer):
                return mutant
        return mutant

    def _mutate(self, answer: str, inst: Instance, rng: random.Random) -> str:
        s = self.steps(answer)
        if not s:
            return answer
        kind = rng.choice(["change", "drop", "swap"] if len(s) > 1 else ["change"])
        i = rng.randrange(len(s))
        if kind == "change":
            s[i] = self.corrupt_step(inst, s[i], rng)
        elif kind == "drop":
            del s[i]
        else:
            j = min(i + 1, len(s) - 1)
            if i == j:
                i -= 1
            s[i], s[j] = s[j], s[i]
        return self.join(s)
