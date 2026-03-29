"""
backends/llm.py — Single LLM factory used by all backends.

Reads LLM_PROVIDER, OLLAMA_*, and OPENAI_* from environment.
"""

from __future__ import annotations
import re
import os
from langchain_core.language_models import BaseChatModel


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen3.5:27b"),
            temperature=temperature,
        )


def strip_json_fence(raw: str) -> str:
    """Strip markdown code fences and Qwen3.5 <think> blocks from LLM JSON responses."""
    raw = raw.strip()
    # Strip Qwen3.5 thinking blocks before the JSON
    if "<think>" in raw:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()
