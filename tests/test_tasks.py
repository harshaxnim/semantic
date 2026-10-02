import random

import pytest

from semantic.benchmarks import TASKS, get_task
from semantic.benchmarks.blocksworld import Blocksworld
from semantic.models import get_model
from semantic.report import _strips_verdict, n50

ALL = list(TASKS.values())


@pytest.mark.parametrize("task", ALL, ids=lambda t: t.name)
@pytest.mark.parametrize("c", [1, 2, 3, 5, 8])
def test_reference_is_accepted(task, c):
    for seed in range(10):
        inst = task.generate(c, seed)
        assert task.verify(inst, inst.reference).correct, inst.id


@pytest.mark.parametrize("task", ALL, ids=lambda t: t.name)
def test_generation_is_deterministic(task):
    a, b = task.generate(5, 3), task.generate(5, 3)
    assert a.prompt == b.prompt and a.reference == b.reference


@pytest.mark.parametrize("task", ALL, ids=lambda t: t.name)
def test_answer_marker_and_noise_before_it(task):
    inst = task.generate(4, 0)
    assert task.verify(inst, "Let me think step by step...\nANSWER:\n" + inst.reference).correct


@pytest.mark.parametrize("task", ALL, ids=lambda t: t.name)
def test_empty_answer_rejected(task):
    assert not task.verify(task.generate(4, 0), "").correct


def test_hanoi_rules():
    t = get_task("hanoi")
    inst = t.generate(2, 0)
    assert t.verify(inst, "1 A B\n2 A C\n1 B C").correct
    assert not t.verify(inst, "2 A C").correct          # disk 2 is not on top
    assert not t.verify(inst, "1 A C\n2 A C").correct   # larger onto smaller
    assert not t.verify(inst, "1 A B\n2 A C").correct   # goal not reached


def test_blocksworld_rules():
    t = get_task("blocksworld")
    inst = t.generate(3, 0)
    inst.data = {"init": {"b1": "table", "b2": "b1", "b3": "table"}, "goal": {"b1": "b3"}}
    assert not t.verify(inst, "(pick-up b1)").correct                  # b1 not clear
    assert not t.verify(inst, "(unstack b2 b1)\n(unstack b3 b1)").correct  # hand not empty
    plan = "(unstack b2 b1)\n(put-down b2)\n(pick-up b1)\n(stack b1 b3)"
    assert t.verify(inst, plan).correct
    assert _strips_verdict(inst, plan, t)


def test_mystery_uses_obfuscated_names_only():
    t = get_task("blocksworld-mystery")
    inst = t.generate(4, 0)
    for word in ("pick-up", "unstack", "table", "block"):
        assert word not in inst.prompt.split("Initial state:")[1]
    assert "(feast" in inst.reference or "(attack" in inst.reference
    assert not t.verify(inst, inst.reference.replace("feast", "unstack").replace("attack", "pick-up")).correct


@pytest.mark.parametrize("name", ["blocksworld", "blocksworld-mystery"])
def test_blocksworld_verifier_agrees_with_strips_on_random_plans(name):
    """Random action sequences: hand-written verifier and STRIPS simulator must agree."""
    t: Blocksworld = get_task(name)
    rng = random.Random(0)
    names = {v: k for k, v in t.vocab.items()}
    agree = valid = 0
    for seed in range(200):
        inst = t.generate(rng.randint(2, 5), seed)
        blocks = list(inst.data["init"])
        acts = []
        for _ in range(rng.randint(1, 12)):
            a = rng.choice(["pick-up", "put-down", "stack", "unstack"])
            args = rng.sample(blocks, 2) if a in ("stack", "unstack") else [rng.choice(blocks)]
            acts.append(f"({t.vocab[a]} {' '.join(args)})")
        plan = "\n".join(acts)
        v = t.verify(inst, plan).correct
        valid += v
        agree += v == _strips_verdict(inst, plan, t)
    assert agree == 200
    assert names  # vocab is invertible


def test_noisy_oracle_matches_p_to_the_n():
    t, m = get_task("multiplication"), get_model("noisy-oracle:0.9")
    inst_steps = []
    ok = 0
    for s in range(400):
        inst = t.generate(4, s)
        inst_steps.append(len(t.steps(inst.reference)))
        ok += t.verify(inst, m.complete(inst.prompt, task=t, inst=inst).text).correct
    expected = sum(0.9 ** n for n in inst_steps) / len(inst_steps)
    assert abs(ok / 400 - expected) < 0.07


def test_n50():
    pts = [{"complexity": 1, "acc": 1.0}, {"complexity": 2, "acc": 0.7}, {"complexity": 3, "acc": 0.3}]
    assert n50(pts) == pytest.approx(2.5)
    assert n50([{"complexity": 1, "acc": 0.9}]) is None
