"""
backends/llm.py — Single LLM factory used by all backends.

Reads LLM_PROVIDER, OLLAMA_*, and OPENAI_* from environment.
"""

from __future__ import annotations
import re
import os
import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable


def get_llm(temperature: float = 0.0, json_mode: bool = False) -> Runnable:
    """
    json_mode=True constrains output to syntactically valid JSON via the
    provider's own grammar/schema enforcement, rather than relying on the
    prompt alone ("Return ONLY a JSON array..."). Confirmed necessary, not
    theoretical: with a small model (phi4-mini:3.8b), planner_node's
    json.loads() failed on every single call without this — not just under
    adversarial input, on ordinary benign prompts too.

    Doesn't guarantee the JSON matches the *expected shape* (array of 3
    strings, {"confidence": ...} etc.) — only that it's valid JSON at all.
    planner_node/verifier_node's existing try/except still matters as a
    second layer for shape mismatches, not made redundant by this.

    Also wraps the model with with_retry(), retrying transient network
    failures talking to the backing LLM server (httpx.TransportError and
    its subclasses: ConnectError, ReadTimeout, RemoteProtocolError, etc.).
    Confirmed necessary, not preemptive: a live run under sustained load hit
    httpx.RemoteProtocolError ("peer closed connection without sending
    complete message body") mid-stream from Ollama, unhandled anywhere in
    the call chain — it propagated straight through FastAPI's ASGI stack
    and crashed the request. This likely also explains earlier crashes that
    showed no corresponding server-side trace at all: if the same drop
    happens before response headers are sent, the client sees what looks
    like a hang/timeout rather than a clean error.

    num_predict caps how many tokens a single Ollama call can generate.
    Confirmed necessary, not preemptive, from Ollama's own server log during
    a live run: individual calls (not the whole request — one single
    generation) ran past 12,000 tokens and 5-6 minutes each. Ollama's own
    documented default for this is -1 (infinite generation) — every prior
    version of this function left it completely unset. That's the real
    mechanism behind every timeout crash this session, not request volume
    or model slowness: a handful of runaway single-call generations alone
    can exceed any request-level timeout, no matter how generous. Reads
    OLLAMA_MAX_TOKENS from .env — a variable that already existed there,
    unused, under "Pipeline Tuning" (see the earlier finding on that same
    block). Falls back to 4096 if unset, matching that file's own stated
    intent, not an arbitrary number.

    Return type is Runnable, not BaseChatModel — with_retry() returns a
    RunnableRetry wrapper. Only .invoke() is used anywhere in this codebase,
    which RunnableRetry supports transparently; chat-model-specific methods
    like .bind_tools() would NOT work on the wrapped object, but nothing
    here calls those.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            model_kwargs=kwargs,
        )
    else:
        from langchain_ollama import ChatOllama
        model = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen36-27b-fable:latest"),
            temperature=temperature,
            format="json" if json_mode else None,
            num_predict=int(os.getenv("OLLAMA_MAX_TOKENS", "4096")),
        )
    return model.with_retry(
        retry_if_exception_type=(httpx.TransportError,),
        stop_after_attempt=3,
        wait_exponential_jitter=True,
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
