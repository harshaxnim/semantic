"""Run a sweep: task x model x complexity x seeds -> results/*.jsonl.

Example:
  python -m semantic.run --task hanoi --model noisy-oracle:0.99 --complexity 2-10 --n 20
Re-running the same command skips instances that already have a result.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from semantic.benchmarks import get_task
from semantic.models import get_model


def parse_range(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def result_path(out: Path, task: str, model: str) -> Path:
    return out / f"{task}__{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}.jsonl"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--complexity", required=True, help="e.g. 3-8 or 3,5,8")
    ap.add_argument("--n", type=int, default=20, help="instances per complexity")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    task, model = get_task(a.task), get_model(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = result_path(out, task.name, model.name)
    done = set()
    if path.exists():
        for ln in path.read_text().splitlines():
            r = json.loads(ln)
            done.add((r["complexity"], r["seed"]))

    jobs = [(c, s) for c in parse_range(a.complexity) for s in range(a.n) if (c, s) not in done]

    def run(job):
        c, s = job
        inst = task.generate(c, s)
        comp = model.complete(inst.prompt, task=task, inst=inst, max_tokens=a.max_tokens)
        v = task.verify(inst, comp.text)
        return {
            "task": task.name, "model": model.name, "complexity": c, "seed": s,
            "correct": v.correct, "detail": v.detail, "valid_prefix": v.valid_prefix,
            "total_steps": v.total_steps, "input_tokens": comp.input_tokens,
            "output_tokens": comp.output_tokens, "cost_usd": comp.cost_usd,
            "latency_s": round(comp.latency_s, 3), "ts": int(time.time()),
        }

    with ThreadPoolExecutor(a.workers) as ex, path.open("a") as f:
        for r in ex.map(run, jobs):
            f.write(json.dumps(r) + "\n")
    print(f"{len(jobs)} new results -> {path} ({len(done)} already present)")


if __name__ == "__main__":
    main()
