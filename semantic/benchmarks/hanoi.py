"""Tower of Hanoi, as used in "The Illusion of Thinking" (Shojaee et al., 2025).

Complexity = number of disks; the optimal solution has 2^n - 1 moves.
"""
from __future__ import annotations

import random
import re

from .base import Instance, Task, Verdict, answer_section

PEGS = "ABC"
MOVE_RE = re.compile(r"^\s*(?:move\s+)?(?:disk\s+)?(\d+)\s*(?:from\s+)?([ABC])\s*(?:->|to|\s)\s*([ABC])\s*$", re.IGNORECASE)

PROMPT = """You are solving a Tower of Hanoi puzzle.
There are {n} disks numbered 1 (smallest) to {n} (largest) and three pegs A, B and C.
Initially all disks are stacked on peg A, largest at the bottom.
Goal: move all disks to peg C.
Rules: move one disk at a time; only the top disk of a peg can be moved;
never place a larger disk on a smaller one.

Give the complete sequence of moves. After the line "ANSWER:" write one move per
line in the form "<disk> <from> <to>", for example "1 A C". Write nothing else
after ANSWER:."""


def _solve(n: int, src: str, dst: str, via: str, out: list[str]) -> None:
    if n == 0:
        return
    _solve(n - 1, src, via, dst, out)
    out.append(f"{n} {src} {dst}")
    _solve(n - 1, via, dst, src, out)


class Hanoi(Task):
    name = "hanoi"
    complexity_label = "disks"
    description = "Tower of Hanoi; full move sequence required (2^n - 1 moves)."

    def generate(self, complexity: int, seed: int) -> Instance:
        moves: list[str] = []
        _solve(complexity, "A", "C", "B", moves)
        return Instance(self.name, complexity, seed, PROMPT.format(n=complexity),
                        {"disks": complexity}, "\n".join(moves))

    def steps(self, answer: str) -> list[str]:
        return [ln.strip() for ln in answer_section(answer).splitlines() if ln.strip()]

    def join(self, steps: list[str]) -> str:
        return "\n".join(steps)

    def corrupt_step(self, inst: Instance, step: str, rng: random.Random) -> str:
        m = MOVE_RE.match(step)
        if not m:
            return step
        d, a, b = m.groups()
        a2, b2 = rng.sample(PEGS, 2)
        if (a2, b2) == (a.upper(), b.upper()):
            a2, b2 = b2, a2
        return f"{d} {a2} {b2}"

    def verify(self, inst: Instance, answer: str) -> Verdict:
        n = inst.data["disks"]
        pegs = {"A": list(range(n, 0, -1)), "B": [], "C": []}
        steps = self.steps(answer)
        for i, s in enumerate(steps):
            m = MOVE_RE.match(s)
            if not m:
                return Verdict(False, f"step {i + 1}: unparseable {s!r}", i, len(steps))
            d, a, b = int(m.group(1)), m.group(2).upper(), m.group(3).upper()
            if a == b:
                return Verdict(False, f"step {i + 1}: same source and target", i, len(steps))
            if not pegs[a] or pegs[a][-1] != d:
                return Verdict(False, f"step {i + 1}: disk {d} is not on top of {a}", i, len(steps))
            if pegs[b] and pegs[b][-1] < d:
                return Verdict(False, f"step {i + 1}: disk {d} onto smaller disk {pegs[b][-1]}", i, len(steps))
            pegs[b].append(pegs[a].pop())
        if len(pegs["C"]) != n:
            return Verdict(False, "goal not reached", len(steps), len(steps))
        return Verdict(True, f"solved in {len(steps)} moves", len(steps), len(steps))
