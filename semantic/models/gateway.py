from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic.benchmarks import Instance, Task

# USD per million tokens (input, output). Verify against current provider
# pricing before quoting costs; unknown models are reported with cost None.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}

CACHE_DIR = Path(os.environ.get("SEMANTIC_CACHE", ".cache/completions"))


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    latency_s: float = 0.0
    cached: bool = False


class Model:
    name: str = ""

    def complete(self, prompt: str, *, task: "Task", inst: "Instance", max_tokens: int = 4096,
                 temperature: float = 0.0, sample: int = 0) -> Completion:
        raise NotImplementedError


def _cost(model: str, tin: int, tout: int) -> float | None:
    p = PRICES.get(model)
    return None if p is None else (tin * p[0] + tout * p[1]) / 1e6


def _post(url: str, headers: dict[str, str], body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json", **headers})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{url}: HTTP {e.code}: {e.read()[:500]!r}") from e
    raise AssertionError("unreachable")


class CachedAPIModel(Model):
    """Base for network models: disk cache keyed by (model, prompt, params, sample)."""

    def complete(self, prompt, *, task, inst, max_tokens=4096, temperature=0.0, sample=0):
        key = hashlib.sha256(json.dumps([self.name, prompt, max_tokens, temperature, sample]).encode()).hexdigest()
        path = CACHE_DIR / key[:2] / f"{key}.json"
        if path.exists():
            c = Completion(**json.loads(path.read_text()))
            c.cached = True
            return c
        t0 = time.time()
        c = self._call(prompt, max_tokens, temperature)
        c.latency_s = time.time() - t0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(c)))
        return c

    def _call(self, prompt: str, max_tokens: int, temperature: float) -> Completion:
        raise NotImplementedError


class AnthropicModel(CachedAPIModel):
    def __init__(self, model: str):
        self.model, self.name = model, f"anthropic:{model}"
        self.key = os.environ.get("ANTHROPIC_API_KEY") or ""
        # Separate variable so the harness never picks up a base URL meant for other tools.
        self.base = os.environ.get("SEMANTIC_ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    def _call(self, prompt, max_tokens, temperature):
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        r = _post(f"{self.base}/v1/messages",
                  {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                  {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                   "messages": [{"role": "user", "content": prompt}]})
        text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
        u = r.get("usage", {})
        tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
        return Completion(text, tin, tout, _cost(self.model, tin, tout))


class OpenAICompatModel(CachedAPIModel):
    def __init__(self, model: str, base: str, key_env: str | None, prefix: str):
        self.model, self.name = model, f"{prefix}:{model}"
        self.base = base.rstrip("/")
        self.key = os.environ.get(key_env, "") if key_env else ""

    def _call(self, prompt, max_tokens, temperature):
        headers = {"authorization": f"Bearer {self.key}"} if self.key else {}
        r = _post(f"{self.base}/chat/completions", headers,
                  {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                   "messages": [{"role": "user", "content": prompt}]})
        text = r["choices"][0]["message"].get("content") or ""
        u = r.get("usage", {})
        tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        return Completion(text, tin, tout, _cost(self.model, tin, tout))


class OracleModel(Model):
    """Returns the reference answer, optionally with per-step noise.

    With per-step reliability p, an answer of n steps survives intact with
    probability p**n: the error-compounding curve from the design note,
    produced by the real pipeline (generate -> answer -> verify).
    """

    def __init__(self, p: float = 1.0):
        self.p = p
        self.name = "oracle" if p >= 1.0 else f"noisy-oracle:{p:g}"

    def complete(self, prompt, *, task, inst, max_tokens=4096, temperature=0.0, sample=0):
        rng = random.Random(f"{self.name}:{inst.id}:{sample}")
        steps = task.steps(inst.reference)
        steps = [s if rng.random() < self.p else task.corrupt_step(inst, s, rng) for s in steps]
        return Completion("ANSWER:\n" + task.join(steps), cost_usd=0.0)


def get_model(spec: str) -> Model:
    kind, _, arg = spec.partition(":")
    if kind == "anthropic":
        return AnthropicModel(arg)
    if kind == "openai":
        return OpenAICompatModel(arg, os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), "OPENAI_API_KEY", "openai")
    if kind == "vllm":
        return OpenAICompatModel(arg, os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"), None, "vllm")
    if kind == "oracle":
        return OracleModel(1.0)
    if kind == "noisy-oracle":
        return OracleModel(float(arg))
    raise SystemExit(f"unknown model spec {spec!r}")
