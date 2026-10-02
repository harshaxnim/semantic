from .base import Instance, Task, Verdict
from .blocksworld import Blocksworld, MysteryBlocksworld
from .hanoi import Hanoi
from .multiplication import Multiplication

TASKS: dict[str, Task] = {t.name: t for t in (Hanoi(), Blocksworld(), MysteryBlocksworld(), Multiplication())}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise SystemExit(f"unknown task {name!r}; choose from {', '.join(TASKS)}")


__all__ = ["Instance", "Task", "Verdict", "TASKS", "get_task"]
