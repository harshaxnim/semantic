"""Aggregate results/*.jsonl and verifier self-checks into JSON for the progress site.

  python -m semantic.report            # writes docs/data/results.json and docs/data/verifier.json
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from semantic.benchmarks import TASKS
from semantic.benchmarks.blocksworld import Blocksworld
from semantic.benchmarks.strips import BLOCKSWORLD, blocksworld_state, run_plan


def n50(points: list[dict]) -> float | None:
    """Complexity where accuracy first crosses below 50% (linear interpolation)."""
    prev = None
    for p in points:
        if p["acc"] < 0.5:
            if prev is None:
                return float(p["complexity"])
            a0, a1 = prev["acc"], p["acc"]
            return prev["complexity"] + (a0 - 0.5) / (a0 - a1) * (p["complexity"] - prev["complexity"])
        prev = p
    return None  # never collapsed within the tested range


def aggregate(results_dir: Path) -> dict:
    groups: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(results_dir.glob("*.jsonl")):
        for ln in f.read_text().splitlines():
            r = json.loads(ln)
            groups[(r["task"], r["model"])][r["complexity"]].append(r)
    series = []
    for (task, model), by_c in sorted(groups.items()):
        pts = []
        for c in sorted(by_c):
            rs = by_c[c]
            ok = sum(r["correct"] for r in rs)
            costs = [r["cost_usd"] for r in rs]
            cost = None if any(x is None for x in costs) else sum(costs)
            pts.append({"complexity": c, "n": len(rs), "correct": ok, "acc": ok / len(rs),
                        "cost_usd": cost, "output_tokens": sum(r["output_tokens"] for r in rs)})
        series.append({"task": task, "model": model, "simulated": "oracle" in model,
                       "points": pts, "n50": n50(pts)})
    return {"series": series}


def _strips_verdict(inst, answer: str, task: Blocksworld) -> bool:
    inverse = {v: k for k, v in task.vocab.items()}
    plan = []
    for s in task.steps(answer):
        m = task.ACT_RE.fullmatch(s.strip())
        if not m:
            return False
        plan.append((inverse.get(m.group(1).lower(), "?"), *[g.lower() for g in m.groups()[1:] if g]))
    ok, state, _ = run_plan(BLOCKSWORLD, blocksworld_state(inst.data["init"]), plan, set(inst.data["init"]))
    return ok and all(("on", b, s) in state for b, s in inst.data["goal"].items())


def verifier_checks(seeds: int = 25, mutants: int = 8) -> dict:
    """Soundness checks for each verifier.

    - reference answers must be accepted;
    - mutated answers (one step changed, dropped or swapped) are mostly
      rejected; for planning tasks every verdict is cross-checked against an
      independent STRIPS simulator and any disagreement is a verifier bug.
    """
    out = []
    for task in TASKS.values():
        cs = {"hanoi": range(1, 9), "multiplication": range(1, 13)}.get(task.name, range(2, 13))
        ref_ok = ref_total = mut_rej = mut_total = disagree = cross = 0
        for c in cs:
            for s in range(seeds):
                inst = task.generate(c, s)
                ref_total += 1
                ref_ok += task.verify(inst, inst.reference).correct
                rng = random.Random(f"mut:{inst.id}")
                for _ in range(mutants):
                    ans = task.corrupt(inst, inst.reference, rng)
                    v = task.verify(inst, ans).correct
                    mut_total += 1
                    mut_rej += not v
                    if isinstance(task, Blocksworld):
                        cross += 1
                        disagree += v != _strips_verdict(inst, ans, task)
        out.append({"task": task.name, "description": task.description,
                    "complexity_label": task.complexity_label,
                    "complexities": [min(cs), max(cs)], "references_accepted": ref_ok,
                    "references": ref_total, "mutants_rejected": mut_rej, "mutants": mut_total,
                    "cross_checked": cross, "cross_check_disagreements": disagree})
    return {"tasks": out}


def main() -> None:
    data = Path("docs/data")
    data.mkdir(parents=True, exist_ok=True)
    (data / "results.json").write_text(json.dumps(aggregate(Path("results")), indent=1))
    (data / "verifier.json").write_text(json.dumps(verifier_checks(), indent=1))
    print("wrote docs/data/results.json and docs/data/verifier.json")


if __name__ == "__main__":
    main()
