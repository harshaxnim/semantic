"""Model gateway: one interface over API models, local OpenAI-compatible
servers (vLLM), and offline simulated models used to test the pipeline.

Model specs (strings) accepted by get_model():
  anthropic:<model-id>            Anthropic Messages API (ANTHROPIC_API_KEY)
  openai:<model-id>               OpenAI-compatible API (OPENAI_API_KEY, OPENAI_BASE_URL)
  vllm:<model-id>                 local vLLM server (VLLM_BASE_URL, default http://localhost:8000/v1)
  oracle                          returns the task's reference answer (pipeline check)
  noisy-oracle:<p>                reference answer with each step independently
                                  corrupted with probability 1 - p (simulates per-step reliability p)
"""
from .gateway import Completion, Model, get_model

__all__ = ["Completion", "Model", "get_model"]
